# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from subdirectories of the `data` directory. It finds all directories starting with `sub-` and all `.nwb` files within each. It uses `h5py` (not `pynwb`) to read the NWB files directly via HDF5 paths.

ii.
```python
def get_all_nwb_files(data_dir, sample=False):
    subjects = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append({
                'subject': subj.replace('sub-', ''),
                'filepath': os.path.join(subj_dir, nwb_file),
                'filename': nwb_file,
            })
    return all_files
```

Loading via h5py:
```python
with h5py.File(filepath, 'r') as f:
    behav = f['processing']['behavior']['BehavioralTimeSeries']
    position = behav['position']['data'][:]
    ...
```

iii. The CONVERSION_NOTES.md documents that there are 11 subjects and 152 sessions. Using h5py directly rather than pynwb is a performance choice for faster loading.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories of `data` starting with `sub-`. The subject ID is extracted by stripping the `sub-` prefix. Subjects are accumulated dynamically as sessions are processed.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
'subject': subj.replace('sub-', ''),
```

iii. The directory structure naturally organizes data by subject, with 11 subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All `.nwb` files in each subject directory are processed as separate sessions.

ii.
```python
nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
```

iii. The NWB file naming convention `sub-{id}_ses-{nn}_behavior+ophys.nwb` makes the session structure clear.

## 1-d. How are the data split into trials?

i. Trial boundaries are found using `trial_start_signal` and `teleport_signal`. Trial starts are all frames where `trial_start_signal > 0`. Trial ends are all frames where `teleport_signal > 0`. These are then paired and filtered to ensure teleport comes after trial start.

ii.
```python
trial_starts = np.where(trial_start_signal > 0)[0]
teleports = np.where(teleport_signal > 0)[0]
n_trials = min(len(trial_starts), len(teleports))
trial_starts = trial_starts[:n_trials]
teleports = teleports[:n_trials]
valid = teleports > trial_starts
trial_starts = trial_starts[valid]
teleports = teleports[valid]
```

iii. The AI notes that trial boundaries are defined by trial_start and teleport signals. However, it uses `np.where(teleport_signal > 0)` which finds ALL frames where teleport > 0 (not just rising edges), so if teleport stays high for multiple frames, this could produce multiple indices per teleport event. The reference code detects rising edges instead.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 5 timepoints are skipped. A lick error correction is also applied: trials where >35% of samples have cumulative lick count > 2 have their lick data set to NaN (but the trial is still included).

ii.
```python
if n_timepoints < 5:
    continue  # Skip very short trials
...
frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
if frac_bad > LICK_ERROR_FRACTION:
    lick_binary[start:end] = np.nan
    lick_error_trials.append(t)
```

iii. The lick error correction follows the reference code's approach of identifying bad lick sensor trials. The minimum trial length threshold of 5 is much lower than the reference's 50.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses the NWB's pre-computed `Deconvolved` data from `processing/ophys/Deconvolved/`. It also loads `Fluorescence` and `Neuropil` data, but only for computing dF/F for interneuron detection — NOT for the final neural signal.

ii.
```python
deconv_data = ophys['Deconvolved']['plane0']['data'][:]
...
neural_all = deconv_data[:, final_neuron_mask]
```

iii. The CONVERSION_NOTES states: "NWB files already contain deconvolved activity ... we do NOT need to compute dF/F from scratch". However, this is suite2p's own deconvolution, NOT the paper's custom processing pipeline. The reference solution explicitly notes: "The NWB 'Deconvolved' array is NOT the signal the paper analyses."

## 2-b. How is the `neural` data processed?

i. No processing is applied to the neural data beyond filtering neurons (iscell + interneuron removal). The NWB Deconvolved data is used directly. NaN values are replaced with 0.

ii.
```python
neural_all = deconv_data[:, final_neuron_mask]
...
trial_neural = neural_all[start:end, :].T.copy()
trial_neural[np.isnan(trial_neural)] = 0
```

iii. The AI decided the NWB already contained the properly processed neural data. This differs from the reference, which recomputes dF/F and deconvolution from raw Fluorescence and Neuropil using the paper's pipeline (neuropil subtraction with mean add-back, maximin baseline, Gaussian smoothing, OASIS deconvolution with tau=0.7).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) `iscell` from the NWB's ImageSegmentation PlaneSegmentation, and (2) putative interneuron exclusion based on correlation of dF/F with speed > 0.5.

ii.
```python
iscell = seg['iscell'][:, 0].astype(bool)
...
is_interneuron = identify_interneurons(dff, speed, iscell)
final_neuron_mask = np.zeros(n_total_rois, dtype=bool)
final_neuron_mask[iscell_indices[non_interneuron]] = True
```

iii. Both filters follow the paper's methodology. The AI computes dF/F for interneuron detection but uses a simplified version without neuropil mean add-back or keep_teleports logic.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The data is aligned to trial start. Neural data is sliced from `trial_starts[t]` to `teleports[t]` for each trial, so the first timepoint is the trial start frame.

ii.
```python
trial_neural = neural_all[start:end, :].T.copy()
```

iii. Since the instructions specify aligning to "start of trial", and the neural data is split at trial boundaries, no additional shifting is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is set to `1000.0 / IMAGING_RATE` = 1000/15.5078125 = ~64.48 ms. No temporal rebinning is applied; data is kept at the native imaging frame rate.

ii.
```python
IMAGING_RATE = 15.5078125  # Hz
...
'time_bin_size': 1000.0 / IMAGING_RATE,  # ms per frame
```

iii. The imaging rate is hardcoded rather than read from the NWB file, though the value matches the single-plane rate reported in the paper. For two-plane animals (m17, m18) where the scanner rate is ~31 Hz, the per-plane rate would be ~15.5 Hz, which matches.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived by computing `np.arange(n_timepoints) * FRAME_PERIOD` where `FRAME_PERIOD = 1.0 / 15.5078125`.

ii.
```python
FRAME_PERIOD = 1.0 / IMAGING_RATE
...
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. Rather than using the actual behavioral timestamps from the NWB file, the AI constructs synthetic timestamps from the constant frame period. This assumes perfectly uniform sampling.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The time from start is computed as frame index * frame period, with the first frame at t=0.

ii.
```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. The reference uses actual behavioral timestamps and subtracts the first timestamp. The AI's approach produces slightly different values if the actual timestamps are not perfectly uniform.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Since both are indexed by the same frame indices (trial_starts[t] to teleports[t]), they are inherently aligned.

ii.
```python
n_timepoints = end - start
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
trial_neural = neural_all[start:end, :].T.copy()
```

iii. The alignment is implicit through shared indexing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral time series in the NWB file.

ii.
```python
environment = behav['environment']['data'][:]
...
trial_env[t] = int(np.median(valid_env))
```

iii. The environment variable is 0 or 1, corresponding to ENV1 and ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the median of the valid (>= 0) environment values within the trial is taken and cast to int.

ii.
```python
env_vals = environment[start:end]
valid_env = env_vals[env_vals >= 0]
if len(valid_env) > 0:
    trial_env[t] = int(np.median(valid_env))
else:
    trial_env[t] = 0
```

iii. Taking the median handles any noise in the signal. The reference simply indexes the environment array directly for each trial's time range without taking a median.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the loop counter `t` over trials within the session.

ii.
```python
trial_num = float(t)
```

iii. The trial number is the 0-indexed sequential trial within the session, not the NWB `trial number` variable.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing — just the loop index cast to float.

ii.
```python
trial_num = float(t)
```

iii. The value is constant across all timepoints within a trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` event timestamps in the NWB file. Reward timestamps are mapped to frame indices using `searchsorted`, then a per-trial rewarded flag is computed.

ii.
```python
reward_timestamps = behav['Reward']['timestamps'][:]
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
reward_frame_indices = np.clip(reward_frame_indices, 0, n_samples - 1)
...
trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
    trial_rewarded[t] = trial_rewards
```

iii. The Reward time series has its own timestamps that need to be mapped to frame indices.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's reward status is used: 1 if the previous trial was rewarded, 0 otherwise. The first trial defaults to 0.

ii.
```python
prev_trial_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_trial_outcome[t] = int(trial_rewarded[t - 1])
```

iii. This follows the instruction specification of binary previous trial outcome.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone boundaries. The reward zone for each trial is identified by finding which predefined zone (A, B, or C) is closest to the mean position where `reward_zone_signal > 0`.

ii.
```python
def identify_reward_zone(position, reward_zone_signal, trial_start, trial_end):
    pos_trial = position[trial_start:trial_end]
    rz_trial = reward_zone_signal[trial_start:trial_end]
    in_rz = rz_trial > 0
    if not np.any(in_rz):
        return None
    rz_pos = pos_trial[in_rz]
    mean_rz_pos = np.mean(rz_pos)
    best_zone = None
    best_dist = np.inf
    for zone_name, (zone_start, zone_end) in REWARD_ZONES.items():
        zone_center = (zone_start + zone_end) / 2
        dist = abs(mean_rz_pos - zone_center)
        if dist < best_dist:
            best_dist = dist
            best_zone = zone_name
    return best_zone
```

iii. The AI uses a simple closest-zone matching by mean position, with forward-fill for trials where no reward zone entry occurred. The reference uses a Viterbi algorithm to handle noisy zone assignments.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative before the zone, 0 inside, positive after. Then discretized into 7 bins.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    distance[before] = position[before] - rz_start
    distance[inside] = 0.0
    distance[after] = position[after] - rz_end
    return distance
```

iii. This matches the standard signed-distance-to-interval computation used in the reference.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized using explicit conditional assignments into 7 bins matching the instruction specification.

ii.
```python
def discretize_distance(distance):
    bins = np.zeros(len(distance), dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```

iii. The bin boundaries match the instructions. The reference uses `np.digitize` with edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]` where `1e-6` separates exactly 0 from slightly positive. The AI uses `distance == 0` for bin 3, which is functionally similar but uses `>=` for bin 4 boundary at `distance > 0` vs the reference's `> 1e-6`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Both use the same frame indexing (start:end) for each trial, so alignment is inherent.

ii.
```python
trial_pos = position[start:end]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
```

iii. Shared indexing ensures temporal alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
position = behav['position']['data'][:]
...
trial_pos = position[start:end]
```

iii. The position variable records the animal's location on the virtual track in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is discretized into 5 equal bins of 90 cm each using `np.floor(position / 90.0)` clipped to [0, 4].

ii.
```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

iii. This divides the 450 cm track into 5 equal 90 cm bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.floor(position / 90.0)` clipped to [0, 4]. This means:
- 0: [0, 90)
- 1: [90, 180)
- 2: [180, 270)
- 3: [270, 360)
- 4: [360, inf)

ii.
```python
bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```

iii. The reference uses `np.digitize` with edges `[-inf, 90, 180, 270, 360, inf]` which produces:
- 0: (-inf, 90)
- 1: [90, 180)
- 2: [180, 270)
- 3: [270, 360)
- 4: [360, inf)

The AI's approach treats positions < 0 the same (mapped to bin 0 via clip), so functionally equivalent for typical track positions. However for exactly 90 cm, `floor(90/90)=1` vs `digitize` also gives 1, so both agree.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indexing as neural data.

ii.
```python
trial_pos = position[start:end]
pos_bins = discretize_position(trial_pos)
```

iii. Shared indexing ensures alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick = behav['lick']['data'][:]
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick data is first clipped to [0, 1] (binarized). Then a lick error correction is applied per trial: if >35% of samples have cumulative lick > 2, the trial's lick data is set to NaN. NaN values are then replaced with 0 in the final output.

ii.
```python
lick_binary = np.clip(lick, 0, 1).astype(np.float64)
...
frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
if frac_bad > LICK_ERROR_FRACTION:
    lick_binary[start:end] = np.nan
    lick_error_trials.append(t)
...
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. The lick error correction follows the reference code's approach. The binarization uses `clip(lick, 0, 1)` rather than `lick > 0`, which will produce different results if lick values are between 0 and 1. The reference uses `licks > 0` for binarization.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indexing as neural data.

ii.
```python
trial_lick = lick_binary[start:end]
```

iii. Shared indexing ensures alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavioral time series combined with `position`. For each trial, the mean position where `reward_zone > 0` is computed and matched to the closest predefined zone (A, B, or C).

ii.
```python
zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
if zone is not None:
    last_known_zone = zone
trial_rz_label.append(zone if zone is not None else last_known_zone)
```

iii. When no reward zone entry is detected, the last known zone is forward-filled. Leading None values are filled backward from the first known zone.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Simple closest-zone matching: compute mean position where reward_zone > 0, find closest of A (105), B (225), C (345) by center distance. Forward-fill for missing zones. Map to categorical: A=0, B=1, C=2.

ii.
```python
mean_rz_pos = np.mean(rz_pos)
for zone_name, (zone_start, zone_end) in REWARD_ZONES.items():
    zone_center = (zone_start + zone_end) / 2
    dist = abs(mean_rz_pos - zone_center)
    if dist < best_dist:
        best_dist = dist
        best_zone = zone_name
...
rz_label_to_idx = {'A': 0, 'B': 1, 'C': 2}
```

iii. The reference uses a more sophisticated Viterbi algorithm that encourages stable zone assignments and handles noisy data better. The AI's simpler approach may produce different results for edge cases.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` event timestamps. Reward timestamps are mapped to frame indices, and a trial is considered rewarded if any reward event falls within its time range.

ii.
```python
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
...
trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
trial_rewarded[t] = trial_rewards
```

iii. Uses the same reward detection as for previous trial outcome.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if any reward event occurred within the trial time range, 0 otherwise. Constant across all timepoints in the trial.

ii.
```python
reward_out = int(trial_rewarded[t])
trial_output[5, :] = reward_out
```

iii. The approach is equivalent to the reference's `np.any(isreward[idx])`.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Neural/behavior length mismatch**: Truncated to common length.
- **Short trials**: Trials with < 5 timepoints skipped.
- **Missing reward zone**: Forward-fill from last known zone; backward-fill for leading unknowns.
- **Lick errors**: Trials with >35% of samples having cumulative lick > 2 have lick set to NaN, then NaN replaced with 0.
- **NaN in neural data**: Replaced with 0.
- **Trial start/teleport mismatch**: Truncated to min count and filtered to ensure teleport > trial_start.

ii.
```python
n_samples = min(n_behav_samples, n_neural_samples)
...
trial_neural[np.isnan(trial_neural)] = 0
...
trial_rz_label.append(zone if zone is not None else last_known_zone)
```

iii. These defensive checks handle known data quality issues.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files via h5py (I/O bound)
2. Computing dF/F for interneuron detection (per session)
3. Processing all 152 sessions sequentially

ii. N/A

iii. The CONVERSION_NOTES reports total full conversion time of ~879 seconds (~14.6 minutes) for 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops for computing trial_rewarded, trial_env, prev_trial_outcome, and lick_error_trials could potentially be vectorized. The interneuron detection is already vectorized.

ii. N/A

iii. The AI did vectorize the interneuron correlation computation for efficiency.

## 13-c. What processing does the code repeat multiple times?

i. The dF/F computation is performed for interneuron detection, but then the Deconvolved data from NWB is used for the actual neural signal. The Fluorescence and Neuropil data are loaded but only used for dF/F computation for interneuron filtering.

ii. N/A

iii. Since the AI uses NWB Deconvolved data directly (rather than computing its own deconvolution), the dF/F computation is only done once for interneuron detection.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The full dF/F computation is performed for all neurons just to identify interneurons. Since only the correlation with speed matters, a simpler approach might suffice. The code also loads the Deconvolved data in addition to Fluorescence/Neuropil, using only Deconvolved for the final output.

ii. N/A

iii. The dF/F computation for interneuron detection is necessary per the paper's methodology, even though it's not used for the final neural signal.
