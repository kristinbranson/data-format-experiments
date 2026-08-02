# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from subdirectories of the `data` directory. It first runs a "survey" pass over every NWB file to collect metadata (neuron counts, reward zone positions, etc.), then runs a second "conversion" pass to extract and format all data. Files are loaded using `pynwb.NWBHDF5IO`. Subject directories match the pattern `sub-<id>`, and NWB files are named `sub-<subject>_ses-<session>_behavior+ophys.nwb`. Neural data comes from `processing/ophys/Deconvolved`, and behavioral data from `processing/behavior/BehavioralTimeSeries`.

ii.
```python
def get_subjects(datadir=DATADIR):
    subjects = [d.split('-')[1] for d in os.listdir(datadir) if d.startswith('sub-')]
    return subjects

def get_sessions(subject, datadir=DATADIR):
    subject_dir = os.path.join(datadir, f"sub-{subject}")
    nwb_files = [f for f in os.listdir(subject_dir) if f.endswith('.nwb')]
    sessions = [int(f.split('_')[1].split('-')[1]) for f in nwb_files]
    return sessions

# Loading:
with pynwb.NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
```

iii. The AI documents that all `sub-*` directories are found and all `.nwb` files within each are treated as sessions. The count of 11 subject directories matches the paper. The AI uses `pynwb` following standard NWB loading documentation.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories of `data/` matching the pattern `sub-<id>`. Each unique subject ID becomes an entry in `data['subjects']`.

ii.
```python
def get_subjects(datadir=DATADIR):
    subjects = [d.split('-')[1] for d in os.listdir(datadir) if d.startswith('sub-')]
    return subjects
...
data['subjects'] = get_subjects(datadir=args.datadir)
```

iii. The AI verified that 11 subject directories exist, matching the paper's 11 switch-task mice.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session numbers are parsed from the filename pattern `ses-<number>`. Each session becomes one entry in `data['neural']`, `data['input']`, and `data['output']`.

ii.
```python
def get_sessions(subject, datadir=DATADIR):
    subject_dir = os.path.join(datadir, f"sub-{subject}")
    nwb_files = [f for f in os.listdir(subject_dir) if f.endswith('.nwb')]
    sessions = [int(f.split('_')[1].split('-')[1]) for f in nwb_files]
    return sessions
```

iii. The AI notes that filenames include a parsable `ses-<number>` and that the total of 152 sessions matches the expected count (11 mice x 14 days minus 2 missing early m11 sessions).

## 1-d. How are the data split into trials?

i. Trials are identified using the `trial_start` and `teleport` behavior time series. Trial starts are found where `trial_start > 0`. Trial ends are found at the onset of teleport (transition from `teleport <= 0` to `teleport > 0`). Data between these indices defines each trial.

ii.
```python
def get_trial_starts_ends(trial_start, teleport):
    trial_start_idx = np.nonzero(trial_start)[0]
    is_teleport_start = (teleport[1:] > 0) & (teleport[:-1] <= 0)
    trial_end_idx = np.nonzero(is_teleport_start)[0] + 1
    assert len(trial_start_idx) == len(trial_end_idx)
    return trial_start_idx, trial_end_idx
```

iii. The AI investigated using the `trial_number` variable but found it inconsistent with `trial_start`. The AI chose `trial_start`/`teleport` boundaries, matching the reference code's trial segmentation logic. The number and length of trials were consistent with this choice.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 50 timepoints are removed. No lick-based trial filtering is applied.

ii.
```python
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    if idx.sum() < min_ntimepoints:
        print(f"Skipping trial {trial}...")
        istrial[trial] = False
        continue
```

iii. The AI mentions removing short trials based on a minimum timepoint threshold. There is no lick quality control filtering, unlike the reference solution which removes trials where >35% of samples have lick count >2 (matching the paper's lick-sensor error exclusion).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` field in `processing/ophys/Deconvolved`.

ii.
```python
deconvolved_data = nwbfile.processing["ophys"]["Deconvolved"]
for plane_data in deconvolved_data.roi_response_series.values():
    spike_data_curr = plane_data.data[:]
```

iii. The AI notes that the paper trains its decoder based on deconvolved neural activity data, which matches the reference code's use of `events` (deconvolved calcium activity).

## 2-b. How is the `neural` data processed?

i. The only processing is combining cells from multiple imaging planes for multi-plane sessions and filtering by `iscell`. For each plane, the deconvolved data is loaded and cells marked as `iscell[:,0] == 1` are kept. Data from all planes are concatenated along the neuron axis.

ii.
```python
deconvolved_data = nwbfile.processing["ophys"]["Deconvolved"]
spike_data = []
nplanes = len(deconvolved_data.roi_response_series)
for plane_data in deconvolved_data.roi_response_series.values():
    roi_data = plane_data.rois[:]
    iscell = np.array(roi_data['iscell'].to_list())[:,0] == 1
    spike_data_curr = plane_data.data[:]
    spike_data.append(spike_data_curr[:,iscell])
spike_data = np.concatenate(spike_data, axis=1)
```

iii. The AI notes that the deconvolved data are already processed and that the neural data rate is consistent across recordings once multi-plane scanning is accounted for.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered based on the `iscell` variable from Suite2p. Only ROIs where `iscell[:,0] == 1` are kept.

ii.
```python
roi_data = plane_data.rois[:]
iscell = np.array(roi_data['iscell'].to_list())[:,0] == 1
spike_data_curr = plane_data.data[:]
spike_data.append(spike_data_curr[:,iscell])
```

iii. The AI notes that the paper describes quality metrics and that the number of cells after `iscell` filtering approximately matches the ranges in the paper (155-2341 vs paper's 155-2172).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data are aligned to trial start, which is inherent in the trial splitting. Since trials are defined from `trial_start` to teleport onset, the first timepoint of each trial corresponds to trial start. No additional temporal shifting is needed.

ii. N/A (alignment is implicit in the trial slicing)

iii. The AI states that aligning to trial start doesn't require additional processing beyond splitting into trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data is kept at its stored rate (~15.5 Hz, ~64.5 ms per frame). The time bin size is computed as `nplanes / plane_data.rate * 1000` ms. No temporal rebinning is applied.

ii.
```python
neural_time_bin_size = nplanes/plane_data.rate*1000 # ms
assert np.isclose(neural_time_bin_size, time_bin_size, rtol=1e-2)
```

iii. The AI verifies that the rate is consistent across recordings and matches the expected ~15.5 Hz frame rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` array associated with the behavior time series (specifically `behavior_data['trial number'].timestamps[:]`).

ii.
```python
timestamps = behavior_data['trial number'].timestamps[:]
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    timestamps_curr = timestamps[idx]
    input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. The AI notes that behavior timestamps were consistent across all behavior variables, so the choice of which variable's timestamps to use was arbitrary.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The initial timestamp of the trial is subtracted from all timestamps within that trial, giving time relative to trial start in seconds.

ii.
```python
input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. Straightforward subtraction to get relative time.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices (same sampling rate and temporal alignment in the NWB file). An assertion checks that neural and behavioral data have the same number of timepoints. If they differ, both are cropped to the minimum length.

ii.
```python
if T != len(timestamps):
    print(f"Warning: number of neural time bins ({T}) does not match...")
    T = min(T, len(timestamps))
    spike_data = spike_data[:T]
    timestamps = timestamps[:T]
```

iii. The AI verified that data rates are the same and all offsets are 0.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series in the NWB file.

ii.
```python
vr_environment = behavior_data['environment'].data[:][:T]
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    vr_environment_curr = vr_environment[idx].astype(int)
    input_curr[1] = vr_environment_curr
```

iii. The AI inspected that the environment variable is constant within a trial and takes values 0 or 1, matching the ENV1/ENV2 description in the paper.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. No processing beyond casting to int and extracting per-trial slices. The raw values (0 or 1) are used directly.

ii.
```python
input_curr[1] = vr_environment_curr
```

iii. The values are already in the desired binary format.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop counter over reconstructed trials (0-indexed sequential index), not from the NWB `trial number` variable.

ii.
```python
for trial in range(ntrials):
    ...
    input_curr[2] = trial
```

iii. The AI notes that the stored `trial number` variable did not agree with the `trial_start` variable, so the sequential loop index was used instead.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial.

ii.
```python
input_curr[2] = trial
```

iii. Simple sequential indexing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series. Reward event timestamps are mapped to behavior timestamps using `np.searchsorted`, and a binary `isreward` array marks timepoints where rewards occurred.

ii.
```python
reward_amounts = behavior_data['Reward'].data[:][:T]
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. The `Reward` time series has its own timestamps (not aligned to the behavior sampling rate), so `searchsorted` is used for alignment. An assertion checks alignment is within half a time bin.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the code checks whether any reward event occurred in the previous trial's time range. For the first trial, it defaults to 0. The value is constant across all timepoints.

ii.
```python
if trial == 0:
    input_curr[3,:] = 0
else:
    prev_reward_idx = np.arange(trial_start_idx[trial-1], trial_end_idx[trial-1], dtype=int)
    isprevreward = np.any(isreward[prev_reward_idx])
    input_curr[3,:] = int(isprevreward)
```

iii. The instructions specify previous trial outcome as binary (omitted=0, rewarded=1). The code checks if any reward event fell within the previous trial's time boundaries.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone location for each trial. Reward zone identity (A, B, or C) is determined using a Viterbi algorithm applied to the positions where `reward_zone > 0`, combined with known reward zone coordinate ranges from the paper (A: 80-130, B: 200-250, C: 320-370 cm).

ii.
```python
# Reward zone assignment via Viterbi:
reward_zone_dict = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}

def segment_reward_positions(reward_zone_positions):
    zone_names = sorted(reward_zone_dict.keys())
    zone_ranges = np.array([reward_zone_dict[z] for z in zone_names])
    # Gaussian emission based on distance to zone range
    dist = np.maximum(zone_ranges[None, :, 0] - startpoints[valid, None],
                      startpoints[valid, None] - zone_ranges[None, :, 1])
    dist = np.maximum(dist, 0)
    log_emission[valid] = -0.5 * (dist / sigma) ** 2
    # Viterbi forward pass with pstay=0.9 transition matrix
    ...

# Distance computation:
def compute_distance_to_reward_zone(position_curr, reward_zone_curr):
    reward_zone_range = reward_zone_dict[reward_zone_curr]
    distance_to_reward_zone = np.zeros(ntimepoints)
    idx_before_zone = position_curr < reward_zone_range[0]
    distance_to_reward_zone[idx_before_zone] = position_curr[idx_before_zone] - reward_zone_range[0]
    idx_after_zone = position_curr > reward_zone_range[1]
    distance_to_reward_zone[idx_after_zone] = position_curr[idx_after_zone] - reward_zone_range[1]
    return distance_to_reward_zone
```

iii. The AI investigated the `reward_zone` behavior variable and found it was not a direct A/B/C label but rather an occupancy indicator. The Viterbi algorithm assigns zones so that the initial rewarded positions are close to the known reward zone coordinates and transitions between zones are discouraged. The resulting class fractions were approximately 1/3 each.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, signed distance is computed from the animal's position to the nearest edge of the assigned reward zone. Distance is 0 inside the zone, negative before the zone, and positive after the zone.

ii.
```python
def compute_distance_to_reward_zone(position_curr, reward_zone_curr):
    ntimepoints = len(position_curr)
    reward_zone_range = reward_zone_dict[reward_zone_curr]
    distance_to_reward_zone = np.zeros(ntimepoints)
    idx_before_zone = position_curr < reward_zone_range[0]
    distance_to_reward_zone[idx_before_zone] = position_curr[idx_before_zone] - reward_zone_range[0]
    idx_after_zone = position_curr > reward_zone_range[1]
    distance_to_reward_zone[idx_after_zone] = position_curr[idx_after_zone] - reward_zone_range[1]
    return distance_to_reward_zone
```

iii. Matches the paper's concept of distance relative to the reward zone. The reward zone boundaries are taken from the paper's definitions.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using `np.digitize` with bin edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`. The `1e-6` boundary separates exactly 0 (in-zone) from slightly positive values.

ii.
```python
distance_to_reward_zone_bins = [-np.inf, -50, -10, 0, 1e-6, 10, 50, np.inf]
output_curr[0] = np.digitize(distance_to_reward_zone, distance_to_reward_zone_bins) - 1
```

iii. The bin edges match the instruction specifications.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same time indices within each trial (same `idx` array), so no additional alignment is needed.

ii.
```python
idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
spike_data_curr = spike_data[idx]
position_curr = position[idx]
```

iii. Both neural and behavioral data use the same time indexing, verified by timestamp assertions.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = behavior_data['position'].data[:][:T]
assert np.allclose(behavior_data['position'].timestamps[:][:T], timestamps)
position_curr = position[idx]
```

iii. The `position` variable records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
position_curr = position[idx]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `np.digitize` with bin edges `[-inf, 50, 150, 250, 350, inf]`, creating 100 cm wide bins.

ii.
```python
position_bins = [-np.inf, 50, 150, 250, 350, np.inf]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. The AI notes the corridor spans roughly -50 to 450 cm from the data survey, so 100 cm bins are approximately equal-sized. However, this does not divide the 0-450 cm track into 5 equal bins (which would be 90 cm each with edges at 0, 90, 180, 270, 360, 450).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii. Same indexing: `position_curr = position[idx]`

iii. Verified by timestamp assertions.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
licks = behavior_data['lick'].data[:][:T]
assert np.allclose(behavior_data['lick'].timestamps[:][:T], timestamps)
licks_curr = licks[idx]
```

iii. The `lick` variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
output_curr[3] = (licks_curr > 0).astype(int)
```

iii. The raw lick values can be >1, so thresholding at >0 converts to binary as required by the instructions.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii. Same indexing: `licks_curr = licks[idx]`

iii. Verified by timestamp assertions.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` and `position` behavior time series, processed through the Viterbi algorithm described in 7-a. The Viterbi assigns each trial to zone A, B, or C based on the position of the animal when `reward_zone > 0`.

ii. See answer 7-a for the Viterbi algorithm. The zone label is then mapped to an integer:
```python
reward_zone_curr = reward_zone[trial].item()
output_curr[4] = ['A','B','C'].index(reward_zone_curr)
```

iii. See 7-a. The Viterbi approach was used because the `reward_zone` variable in the NWB file is not a direct A/B/C label.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. After Viterbi assignment, the zone letter (A/B/C) is mapped to an integer (0/1/2). The value is constant across all timepoints in a trial.

ii.
```python
output_curr[4] = ['A','B','C'].index(reward_zone_curr)
```

iii. Simple mapping from letter to index.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. The `Reward` time series records reward delivery events with separate timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward event timestamps are mapped to the nearest behavior timepoint using `searchsorted`. For each trial, the output is 1 if any reward event occurred within the trial, 0 otherwise. The value is constant across all timepoints in the trial.

ii.
```python
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
...
output_curr[5] = np.any(isreward[idx]).astype(int)
```

iii. An assertion checks alignment error is within half a time bin.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: If neural timepoints differ from behavior timestamps, data is cropped to the minimum with a warning.
- **Short trials**: Trials with fewer than 50 timepoints are skipped.
- **Missing reward zone data**: Trials where `reward_zone` is never active get NaN positions; the Viterbi algorithm handles these via uniform emission.
- **Reward timestamp alignment**: An assertion verifies reward times align within half a time bin.
- **Trial start/end count mismatch**: An assertion requires equal numbers of trial starts and ends.

ii.
```python
if T != len(timestamps):
    T = min(T, len(timestamps))
    spike_data = spike_data[:T]
    timestamps = timestamps[:T]

if idx.sum() < min_ntimepoints:
    istrial[trial] = False
    continue
```

iii. These are defensive checks found during data exploration. The neural/behavior mismatch occurs in a small number of sessions.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files using `pynwb` (I/O bound, reads large arrays)
2. The survey step (loads every NWB file to gather statistics and reward zone positions)
3. The full conversion step (loads every NWB file again)
4. Saving the pickle file

ii. N/A

iii. The NWB files are large and contain full neural recordings.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session_data` iterates over each trial sequentially. Operations like `np.digitize` and `compute_distance_to_reward_zone` could theoretically be applied to full session arrays before splitting into trials. The Viterbi inner loop over states could be vectorized with matrix operations.

ii. N/A

iii. Variable trial lengths make full vectorization awkward, requiring padding or masking.

## 13-c. What processing does the code repeat multiple times?

i. The survey step and the conversion step both load every NWB file and read the same data arrays (neural, behavior). Neural data is read during survey to count neurons and check rates, then read again during conversion. The `reward_zone_positions` are computed in the survey, but other data streams are read twice.

ii. N/A

iii. The AI notes that combining the two passes would require too much RAM and that the Viterbi output from the survey is needed before conversion can proceed.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI reports N/A for this question, claiming no unnecessary processing. However, the code computes and stores `speed` as an output, which is requested by the instructions. The survey step generates and saves reward zone position plots (`reward_zone_positions.png/pdf/svg`) which are diagnostic but not used downstream. The survey also collects many statistics (unique values, ranges) that are not directly used in the conversion.

ii. N/A

iii. N/A
