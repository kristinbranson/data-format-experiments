# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans the `data/` directory for subdirectories starting with `sub-`, then within each finds all `.nwb` files sorted alphabetically. Each NWB file is opened with `h5py` and behavioral timeseries, neural data (Deconvolved, Fluorescence, Neuropil), and metadata are extracted. Multi-plane sessions (m17, m18) have neural data concatenated across planes.

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

iii. The agent documented in CONVERSION_NOTES that NWB files are organized as `data/sub-{id}/sub-{id}_ses-{nn}_behavior+ophys.nwb`, one per session per mouse. The agent correctly identified 11 subjects and 152 total sessions.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the subdirectory names (e.g., `sub-m3` -> subject `m3`). A unique list of subjects is built as sessions are processed, and `subject_idx` maps each session to its subject.

ii.
```python
subj = session_info['subject']
if subj not in subjects_list:
    subjects_list.append(subj)
subject_idx_list.append(subjects_list.index(subj))
```

iii. The agent identified 11 subjects matching the paper (m3, m4, m7, m11-m15, m17-m19). Subject ID is also read from NWB metadata (`general/subject/subject_id`).

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed sequentially and each produces one entry in the `neural`, `input`, and `output` lists.

ii.
```python
for i, file_info in enumerate(nwb_files):
    neural_trials, input_trials, output_trials, session_info = process_session(
        file_info['filepath'], show_processing=show_processing, session_idx=i)
    if len(neural_trials) < 2:
        print(f"  WARNING: Skipping session with <2 valid trials")
        continue
    all_neural.append(neural_trials)
```

iii. CONVERSION_NOTES documents 152 sessions total (14 per subject except m11 with 12), consistent with the paper.

## 1-d. How are the data split into trials?

i. Trial boundaries are defined by the `trial_start` and `teleport` signals in the behavioral timeseries. Frames where `trial_start > 0` mark trial beginnings, and frames where `teleport > 0` mark trial ends. Each trial spans from `trial_start` index to `teleport` index (exclusive).

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

iii. The agent noted the reference code uses `start-1:stop-1` indexing (1-indexed), while NWB indices from `np.where` are 0-indexed. The agent verified this by checking that trial counts (~80.5 per session) match the paper's reported value.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 5 timepoints are skipped. Sessions with fewer than 2 valid trials are excluded entirely. Lick error trials have their lick data set to NaN but the trials are still included.

ii.
```python
if n_timepoints < 5:
    continue  # Skip very short trials

# In build_dataset:
if len(neural_trials) < 2:
    print(f"  WARNING: Skipping session with <2 valid trials")
    continue
```

iii. The agent documented that ~80 trials per session are retained, with only very short trials (< 5 frames) excluded. Lick error correction flags trials but does not remove them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the pre-computed Deconvolved calcium events stored in the NWB files at `processing/ophys/Deconvolved/plane0/data` (and `plane1` for multi-plane sessions).

ii.
```python
deconv_data = ophys['Deconvolved']['plane0']['data'][:]  # (n_samples, n_rois)
# For multi-plane:
deconv_data = np.concatenate(deconv_parts, axis=1)
```

iii. The agent identified that NWB files contain pre-computed deconvolved events corresponding to `sess.timeseries['events']` in the reference code, making separate dF/F computation and OASIS deconvolution unnecessary for the neural data itself.

## 2-b. How is the `neural` data processed?

i. The deconvolved data is filtered by the `iscell` mask (Suite2p manual curation) and an interneuron exclusion mask (Pearson correlation of dF/F with speed > 0.5). NaN values in the neural data are replaced with 0. Data is transposed to (n_neurons, n_timepoints) per trial.

ii.
```python
final_neuron_mask = np.zeros(n_total_rois, dtype=bool)
final_neuron_mask[iscell_indices[non_interneuron]] = True
neural_all = deconv_data[:, final_neuron_mask]

# Per trial:
trial_neural = neural_all[start:end, :].T.copy()
trial_neural[np.isnan(trial_neural)] = 0
```

iii. The agent documented following the reference code's pipeline: iscell filtering + interneuron exclusion. The NaN-to-0 replacement follows the reference code comment `X[np.isnan(X)] = 0`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) `iscell` mask from Suite2p manual curation, and (2) interneuron exclusion based on Pearson correlation of dF/F with running speed exceeding 0.5.

ii.
```python
iscell = seg['iscell'][:, 0].astype(bool)

# Interneuron detection:
def identify_interneurons(dff, speed, iscell_mask):
    r = cov_XY / (std_X * std_Y)
    is_interneuron = r > INTERNEURON_CORR_THRESHOLD  # 0.5
    return is_interneuron
```

iii. The paper states "Additional putative interneurons were detected for exclusion from further analysis by a Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed, excluding 0.42 +/- 0.85% of cells." The agent computed dF/F from Fluorescence and Neuropil data specifically for this interneuron check.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each trial. Each trial's neural data starts at the `trial_start` frame and ends at the `teleport` frame. Time 0 corresponds to the first frame of the trial.

ii.
```python
trial_neural = neural_all[start:end, :].T.copy()  # start = trial_starts[t], end = teleports[t]
```

iii. The instructions specify "Temporally align based on start of the trial." The agent aligns by extracting neural data from the trial_start index to the teleport index for each trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is set to 1 imaging frame, approximately 64.5 ms (1/15.5078125 Hz). No temporal rebinning is applied. However, the frame period is hardcoded as a constant rather than being read per-session.

ii.
```python
IMAGING_RATE = 15.5078125  # Hz
FRAME_PERIOD = 1.0 / IMAGING_RATE  # seconds

# In metadata:
'time_bin_size': 1000.0 / IMAGING_RATE,  # ms per frame
```

iii. The agent noted the imaging rate is ~15.5 Hz. However, multi-plane sessions (m17, m18) have a 31 Hz imaging rate (15.5 Hz per plane), but the agent hardcodes the single-plane rate. This means the `time_bin_size` in metadata and `time_from_trial_start` input are incorrect for those 28 sessions (14 each for m17 and m18).

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Time from start of trial is computed from the frame index within the trial, not from any raw NWB variable directly. It uses the constant `FRAME_PERIOD` and the number of timepoints in the trial.

ii.
```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. The agent decided to compute time as frame_index * frame_period, where frame_period = 1/15.5078125 Hz.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A simple linear computation: multiply the 0-based frame index by the frame period constant.

ii.
```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
# FRAME_PERIOD = 1.0 / 15.5078125  # ~0.0645 seconds
```

iii. No complex processing. However, the hardcoded frame period is wrong for multi-plane sessions (m17, m18) which run at 31 Hz, resulting in time values that are 2x too large for those sessions.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Time from start is computed with the same number of timepoints as the neural data (`n_timepoints = end - start`), so they are inherently aligned frame-by-frame.

ii.
```python
n_timepoints = end - start
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
trial_input[0, :] = time_from_start
```

iii. Since neural data and time are indexed from the same trial start/end indices, they share identical temporal alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type comes from the NWB behavioral timeseries `environment` variable.

ii.
```python
environment = behav['environment']['data'][:]
```

iii. The NWB `environment` variable encodes -1 (pre-TTL), 0 (ENV1), or 1 (ENV2), matching the paper's two-environment design.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median environment value across valid samples in each trial is taken, filtering out negative values (pre-TTL markers).

ii.
```python
trial_env = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    env_vals = environment[start:end]
    valid_env = env_vals[env_vals >= 0]
    if len(valid_env) > 0:
        trial_env[t] = int(np.median(valid_env))
    else:
        trial_env[t] = 0
```

iii. Using the median is a reasonable approach since the environment is constant within a trial, but the agent filters negative values and uses median rather than simply taking the first valid value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index `t` during trial iteration (0-based), NOT from the NWB `trial number` field.

ii.
```python
trial_num = float(t)  # t is the loop variable iterating over n_trials
```

iii. The NWB file contains a `trial number` behavioral timeseries that the agent loads but does not use for this input. The loop index and the NWB trial number would typically match since both are 0-based, but could diverge if trial boundaries are processed differently.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond casting the loop index to float.

ii.
```python
trial_num = float(t)
trial_input[2, :] = trial_num  # broadcast to all timepoints
```

iii. The trial number is broadcast as a constant across all timepoints in the trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from reward event timestamps (`Reward/timestamps`) and behavioral timestamps, mapped to determine which trial received reward.

ii.
```python
reward_data = behav['Reward']['data'][:]
reward_timestamps = behav['Reward']['timestamps'][:]
behav_timestamps = behav['position']['timestamps'][:]
```

iii. Reward events are stored as timestamp-based events in the NWB file, separate from the frame-rate behavioral timeseries.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward timestamps are mapped to frame indices using `np.searchsorted`. For each trial, the code checks if any reward frame falls within the trial boundaries. The previous trial's reward status becomes the current trial's input. The first trial defaults to 0 (no previous outcome).

ii.
```python
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
reward_frame_indices = np.clip(reward_frame_indices, 0, n_samples - 1)

trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
    trial_rewarded[t] = trial_rewards

prev_trial_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_trial_outcome[t] = int(trial_rewarded[t - 1])
```

iii. The instruction specifies "omitted = 0, rewarded = 1". The agent implements this correctly with 0 for the first trial (no previous trial). The reward detection uses `np.searchsorted` to find frame indices for reward timestamps.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from the NWB `position` and `reward_zone` behavioral timeseries. The reward zone location per trial is identified from where `reward_zone > 0`, then mapped to predefined zone boundaries (A: 80-130, B: 200-250, C: 320-370).

ii.
```python
trial_pos = position[start:end]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
```

iii. The agent uses the NWB `reward_zone` signal to infer which of the three reward zones (A, B, C) is active, since the NWB format does not directly store the zone label.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed from position to the nearest edge of the reward zone. If position < zone start, distance is negative (position - zone_start). If position is within the zone, distance is 0. If position > zone end, distance is positive (position - zone_end).

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    distance[before] = position[before] - rz_start  # negative
    distance[inside] = 0.0
    distance[after] = position[after] - rz_end  # positive
    return distance
```

iii. The agent computes distance to the nearest edge of the zone, giving a signed measure where negative = approaching, 0 = inside, positive = past the zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Distance is discretized into 7 bins matching the instruction specification.

ii.
```python
def discretize_distance(distance):
    bins = np.zeros(len(distance), dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3  # in reward zone
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```

iii. The bins match the instruction: 0: < -50 cm, 1: -50 to -10 cm, 2: -10 cm to < 0 cm, 3: 0 cm, 4: >0 cm to +10 cm, 5: +10 to +50 cm, 6: > +50 cm.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Distance is computed from the same trial segment (start:end) as the neural data, so they are frame-aligned.

ii.
```python
trial_pos = position[start:end]  # same start:end as neural
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
dist_bins = discretize_distance(dist_to_rz)
trial_output[0, :] = dist_bins
```

iii. Alignment is inherent from using the same frame indices.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position comes from the NWB `position` behavioral timeseries.

ii.
```python
position = behav['position']['data'][:]
# Per trial:
trial_pos = position[start:end]
```

iii. Position is recorded in cm along the 450 cm virtual track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is discretized into 5 equal-sized bins of 90 cm each (0-90, 90-180, 180-270, 270-360, 360-450).

ii.
```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

iii. The instruction says "discretized into 5 equal-sized bins" on a 450 cm track, giving 90 cm bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is divided by 90 cm, floored, and clipped to the range [0, 4].

ii.
```python
bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```

iii. This produces bins: 0 (0-90 cm), 1 (90-180 cm), 2 (180-270 cm), 3 (270-360 cm), 4 (360-450 cm).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is extracted from the same frame range (start:end) as neural data.

ii.
```python
trial_pos = position[start:end]
pos_bins = discretize_position(trial_pos)
trial_output[1, :] = pos_bins
```

iii. Alignment is inherent from using the same frame indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick data comes from the NWB `lick` behavioral timeseries.

ii.
```python
lick = behav['lick']['data'][:]
```

iii. The NWB lick data contains per-frame lick counts (integer values 0-6), not cumulative counts.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick values are clipped to [0, 1] for binarization. Error correction is applied: if > 35% of samples in a trial have lick count > 2 (checked on original data), lick values for that trial are set to NaN. NaN values are then replaced with 0 in the final output.

ii.
```python
lick_binary = np.clip(lick, 0, 1).astype(np.float64)

# Error correction per trial:
for t in range(n_trials):
    trial_lick = lick[start:end]  # original lick values
    frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
    if frac_bad > LICK_ERROR_FRACTION:  # 0.35
        lick_binary[start:end] = np.nan

# In output construction:
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. The reference code (glmUtils.py) similarly binarizes licks (`licks[licks > 1] = 1`) and applies lick error correction with a 35% threshold for samples with lick count > 2. The agent's CONVERSION_NOTES describe this as matching the reference.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick data is extracted from the same trial frame range as neural data.

ii.
```python
trial_lick = lick_binary[start:end]
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
trial_output[3, :] = lick_vals
```

iii. Same frame-level alignment as all other variables.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is inferred from the NWB `position` and `reward_zone` behavioral timeseries. The position where `reward_zone > 0` is used to determine which predefined zone (A, B, or C) the animal entered.

ii.
```python
reward_zone_signal = behav['reward_zone']['data'][:]

def identify_reward_zone(position, reward_zone_signal, trial_start, trial_end):
    rz_trial = reward_zone_signal[trial_start:trial_end]
    in_rz = rz_trial > 0
    rz_pos = pos_trial[in_rz]
    mean_rz_pos = np.mean(rz_pos)
    # Map to closest zone center
    for zone_name, (zone_start, zone_end) in REWARD_ZONES.items():
        zone_center = (zone_start + zone_end) / 2
        dist = abs(mean_rz_pos - zone_center)
        ...
```

iii. The agent could not use the reference code's `get_reward_zones()` function (which uses session metadata/scene strings not available in NWB), so it inferred the zone from where the animal was when `reward_zone > 0`. For trials where the animal didn't enter the reward zone, the last known zone is carried forward.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, the mean position where `reward_zone > 0` is computed and mapped to the nearest predefined zone (A: 80-130, B: 200-250, C: 320-370). Trials without reward zone entry inherit the last known zone (forward/backward fill).

ii.
```python
# Forward fill for unknown zones:
last_known_zone = None
for t in range(n_trials):
    zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
    if zone is not None:
        last_known_zone = zone
    trial_rz_label.append(zone if zone is not None else last_known_zone)

# Map to indices:
rz_label_to_idx = {'A': 0, 'B': 1, 'C': 2}
trial_rz_idx = np.array([rz_label_to_idx.get(lbl, 0) for lbl in trial_rz_label])
```

iii. The agent maps zones to 0=A, 1=B, 2=C as specified in the instructions. The forward-fill approach handles omission trials where the animal may not lick in the zone.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the NWB `Reward` event timestamps.

ii.
```python
reward_data = behav['Reward']['data'][:]
reward_timestamps = behav['Reward']['timestamps'][:]
```

iii. Reward events are stored separately from the frame-rate behavioral data, with their own timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to frame indices using `np.searchsorted` on behavioral timestamps. A trial is marked as rewarded (1) if any reward frame falls within its boundaries, otherwise not rewarded (0).

ii.
```python
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
    trial_rewarded[t] = trial_rewards

reward_out = int(trial_rewarded[t])
```

iii. The ~15% omission rate reported in the paper (verified at 15.3% in the converted data) confirms this processing is correct.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- Behavioral/neural length mismatch: truncated to the shorter of the two.
- NaN in neural data: replaced with 0.
- Trials with < 5 frames: skipped.
- Sessions with < 2 trials: skipped.
- Lick sensor errors: trials with >35% samples having lick count > 2 get lick set to NaN, then NaN replaced with 0 in output.
- Reward zone not detected: last known zone carried forward; backward fill for leading unknowns.
- Mismatched trial_start/teleport counts: truncated to minimum count, filtered to ensure teleport > trial_start.

ii.
```python
# Length mismatch:
n_samples = min(n_behav_samples, n_neural_samples)

# NaN in neural:
trial_neural[np.isnan(trial_neural)] = 0

# Short trial skip:
if n_timepoints < 5:
    continue

# Lick error -> 0:
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. The agent documented handling these cases in CONVERSION_NOTES under edge cases and verified no NaN or negative values remain in the final neural data.

## 13-a. What are the most time-consuming steps of the code?

i. Based on timing information in the code, the most time-consuming steps are: (1) dF/F computation for interneuron detection, (2) loading NWB files with h5py, and (3) the overall per-session processing loop.

ii.
```python
t1 = time.time()
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
t_dff = time.time() - t1

# Reported: load_time, dff_time, interneuron_time, total_time per session
```

iii. The agent documented that the full conversion takes ~879 seconds (~14.6 min) for 152 sessions. The dF/F computation is done per-trial with Gaussian filtering, baseline computation, and smoothing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
- The `compute_dff` function loops over trials (line 115: `for i, (start, stop) in enumerate(zip(trial_starts, teleports))`)
- The trial-level loop for building per-trial data (line 509: `for t in range(n_trials)`)
- The reward detection loop per trial (line 430: `for t in range(n_trials)`)
- The environment extraction loop (line 487: `for t in range(n_trials)`)
- The lick error correction loop (line 475: `for t in range(n_trials)`)

ii.
```python
# Example: reward detection loop
trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
    trial_rewarded[t] = trial_rewards
```

iii. The agent did vectorize the interneuron detection (noted saving from 4.3s to 0.4s per session) but left other per-trial loops unvectorized.

## 13-c. What processing does the code repeat multiple times?

i. The dF/F computation is performed for interneuron detection even though the deconvolved data is already available pre-computed in the NWB files. This is necessary because interneuron detection requires dF/F (not deconvolved), but it represents significant computation that wouldn't be needed if interneuron labels were pre-computed.

ii.
```python
# dF/F computed solely for interneuron check:
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
is_interneuron = identify_interneurons(dff, speed, iscell)
```

iii. The agent noted this in CONVERSION_NOTES but determined it was necessary since the NWB files don't contain pre-computed interneuron labels.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The dF/F computation is the main unnecessary processing - it is computed for all neurons across all timepoints but only used to compute a single correlation value per neuron for interneuron detection. The actual neural data used downstream is the pre-computed deconvolved data.

ii.
```python
# Full dF/F computation (with neuropil subtraction, baseline, smoothing) just for:
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
# -> used only for:
is_interneuron = identify_interneurons(dff, speed, iscell)
# The dff array itself is discarded after this
```

iii. Additionally, the code loads `fluor_data` and `neuropil_data` arrays from the NWB files (which are large), only to compute dF/F for the interneuron check. These arrays are not used elsewhere.
