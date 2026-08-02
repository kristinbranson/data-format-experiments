# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all subdirectories of the `data` directory starting with `sub-`, then finds all `.nwb` files within each subdirectory. Each NWB file is opened using the `pynwb` library. All subjects, sessions, and trials are included.

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

iii. The AI verified that the number of `sub-` directories (11) matches the number of mice in the paper. All NWB files in each subject directory are treated as sessions. The AI uses the `pynwb` library following standard NWB loading conventions.

## 1-b. How are the data split into subjects?

i. Each subdirectory of `data/` matching the pattern `sub-<id>` corresponds to one subject. The subject ID is extracted from the directory name.

ii.
```python
def get_subjects(datadir=DATADIR):
    subjects = [d.split('-')[1] for d in os.listdir(datadir) if d.startswith('sub-')]
    return subjects
```

iii. The number of subject directories (11) matches the paper. Each directory name follows `sub-<id>` convention.

## 1-c. How are the data split into sessions?

i. Each NWB file within a subject's directory corresponds to one session. Session numbers are parsed from filenames matching `sub-<subject>_ses-<NN>_behavior+ophys.nwb`.

ii.
```python
def get_sessions(subject, datadir=DATADIR):
    subject_dir = os.path.join(datadir, f"sub-{subject}")
    nwb_files = [f for f in os.listdir(subject_dir) if f.endswith('.nwb')]
    sessions = [int(f.split('_')[1].split('-')[1]) for f in nwb_files]
    return sessions
```

iii. File naming convention includes a parsable `ses-<number>` indicating session identity. Total of 152 sessions matches expected count (10 mice x 14 + m11 x 12).

## 1-d. How are the data split into trials?

i. Trial boundaries are defined using the `trial_start` and `teleport` behavior time series. Trial starts are identified where `trial_start` is nonzero. Trial ends are identified at rising edges of the `teleport` signal (where teleport transitions from <=0 to >0).

ii.
```python
def get_trial_starts_ends(trial_start, teleport):
    trial_start_idx = np.nonzero(trial_start)[0]
    is_teleport_start = (teleport[1:] > 0) & (teleport[:-1] <= 0)
    trial_end_idx = np.nonzero(is_teleport_start)[0] + 1
    assert len(trial_start_idx) == len(trial_end_idx)
    return trial_start_idx, trial_end_idx
```

iii. The AI investigated using the `trial_number` variable but found it inconsistent with `trial_start`. The `trial_start`/`teleport` approach matches the reference code's trial epoch definition and the paper's description of trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI attempts to filter trials with fewer than 50 timepoints using a minimum timepoint threshold. No lick-artifact filtering is implemented.

ii.
```python
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    if idx.sum() < min_ntimepoints:
        print(f"Skipping trial {trial}...")
        istrial[trial] = False
        continue
```

iii. The AI noted that some short trials existed. However, the code has a bug: `idx.sum()` computes the sum of index values (e.g., [100, 101, 102, ...] sums to thousands), not the number of timepoints (`len(idx)`). This means the filter effectively never activates. Additionally, the reference code and paper describe lick-artifact trial filtering (>35% of frames with lick count > 2), which is not implemented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` field in the `ophys` processing module of each NWB file.

ii.
```python
deconvolved_data = nwbfile.processing["ophys"]["Deconvolved"]
for plane_data in deconvolved_data.roi_response_series.values():
    spike_data_curr = plane_data.data[:]
```

iii. The paper describes training decoders on deconvolved calcium activity data, which corresponds to the NWB `Deconvolved` data.

## 2-b. How is the `neural` data processed?

i. The only processing is combining cells from multiple imaging planes for multi-plane sessions and filtering by `iscell`. No frame-level validity filtering is applied. No dF/F recomputation or additional processing is done.

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

iii. The AI reasoned that deconvolved data are already processed and no further transformation is needed. The reference code additionally filters out individual frames with NaN or invalid values, which this code does not do.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered using the `iscell` variable. Only ROIs where `iscell[:,0] == 1` (accepted cells from suite2p curation) are kept.

ii.
```python
roi_data = plane_data.rois[:]
iscell = np.array(roi_data['iscell'].to_list())[:,0] == 1
spike_data_curr = plane_data.data[:]
spike_data.append(spike_data_curr[:,iscell])
```

iii. The paper describes manual ROI curation and additional exclusion of putative interneurons by speed correlation. The AI implements only the `iscell` filter (manual curation), not the speed-correlation interneuron exclusion. The AI noted that neuron counts after `iscell` filtering approximately match paper ranges.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by indexing into the session-wide neural array using trial boundary indices. Since the instructions specify alignment to trial start, no additional temporal shifting is needed beyond extracting the trial epoch.

ii.
```python
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    spike_data_curr = spike_data[idx]
    neural.append(spike_data_curr.T)
```

iii. Neural and behavioral data share the same frame-aligned time base in the NWB files, so trial start alignment is achieved by indexing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The data is kept at the original sampling rate of ~15.5 Hz (~64.5 ms per frame). For multi-plane sessions, the effective rate is computed as `nplanes / plane_data.rate * 1000` ms.

ii.
```python
neural_time_bin_size = nplanes/plane_data.rate*1000 # ms
assert np.isclose(neural_time_bin_size, time_bin_size, rtol=1e-2)
```

iii. The stored sampling rate is consistent across sessions once multi-plane correction is applied. No rebinning is needed since all sessions share the same effective rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` array associated with the behavior time series (specifically `trial number` timestamps, though all behavior variables share the same timestamps).

ii.
```python
timestamps = behavior_data['trial number'].timestamps[:]
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    timestamps_curr = timestamps[idx]
    input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. Behavior timestamps are consistent with the neural data rate and all behavior time series share the same timestamp array.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in each trial is subtracted from all timestamps within that trial, yielding time relative to trial start in seconds.

ii.
```python
input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. Standard approach to compute time relative to an event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data are already temporally aligned in the NWB files (same frame-aligned sampling grid). Both use the same index array for trial extraction. A check verifies that neural and behavior lengths match, cropping to the shorter if they differ.

ii.
```python
if T != len(timestamps):
    print(f"Warning: number of neural time bins ({T}) does not match...")
    T = min(T, len(timestamps))
    spike_data = spike_data[:T]
    timestamps = timestamps[:T]
```

iii. Data rates for neural and behavior are the same. The AI does not apply per-frame validity filtering (checking for NaN/invalid values), unlike the reference code.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
vr_environment = behavior_data['environment'].data[:][:T]
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    vr_environment_curr = vr_environment[idx].astype(int)
    input_curr[1] = vr_environment_curr
```

iii. The environment variable is constant within each trial and takes values 0 or 1, matching the ENV1/ENV2 description in the paper.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The raw environment values are cast to int and used directly. No aggregation (e.g., modal value) is computed since the values are constant within trials.

ii.
```python
input_curr[1] = vr_environment_curr
```

iii. Since environment doesn't change within a trial, using the raw values produces the same result as computing a per-trial summary statistic.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop counter over trials (0-indexed sequential index within the session), NOT from the `trial number` variable stored in the NWB file.

ii.
```python
for trial in range(ntrials):
    ...
    input_curr[2] = trial
```

iii. The AI investigated the stored `trial number` variable but found it inconsistent with the `trial_start` variable and chose to use the sequential loop index instead.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial.

ii.
```python
input_curr[2] = trial
```

iii. The trial number is simply the sequential 0-based index of the trial within the session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series. Reward event timestamps are matched to behavior timestamps using `np.searchsorted`, and a binary `isreward` array marks timepoints where rewards occurred.

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. The `Reward` time series has its own timestamps (sparse events), so `searchsorted` maps them to the nearest behavior frame. An assertion checks alignment error is within half a time bin.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check whether any reward event occurred in the previous trial's time range (using `isreward` array). For the first trial in a session, set to 0.

ii.
```python
if trial == 0:
    input_curr[3,:] = 0
else:
    prev_reward_idx = np.arange(trial_start_idx[trial-1], trial_end_idx[trial-1], dtype=int)
    isprevreward = np.any(isreward[prev_reward_idx])
    input_curr[3,:] = int(isprevreward)
```

iii. The code checks for any reward event within the previous trial's time boundaries. The reference code uses `reward_by_trial_number.get(trial_num - 1, 0)` which looks up by actual trial number rather than sequential index.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone location for each trial. The reward zone is determined using a Viterbi algorithm that assigns zone labels (A/B/C) based on the position of the mouse when the `reward_zone` behavior variable is nonzero.

ii. Reward zone assignment uses Viterbi with Gaussian emission on distance to known zone ranges:
```python
reward_zone_dict = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}
sigma = 10.0
zone_ranges = np.array([reward_zone_dict[z] for z in zone_names])
dist = np.maximum(zone_ranges[None, :, 0] - startpoints[valid, None],
                  startpoints[valid, None] - zone_ranges[None, :, 1])
dist = np.maximum(dist, 0)
log_emission[valid] = -0.5 * (dist / sigma) ** 2

# Transition encourages stability:
pstay = .9
log_trans = np.full((n_states, n_states), np.log((1 - pstay) / (n_states - 1)))
np.fill_diagonal(log_trans, np.log(pstay))
```

iii. The AI plotted reward zone positions and found that the start position of reward zone activation was the cleanest signal for inferring zone labels. The Viterbi algorithm encourages stable zone assignments across consecutive trials, producing approximately 1/3 representation for each zone.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, compute the signed distance from the animal's position to the nearest edge of the reward zone. Distance is 0 when inside the zone, negative when before it, positive when past it.

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

iii. Matches the paper's concept of distance relative to the reward zone. The same zone coordinate dictionary as the reference code is used.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using `np.digitize` with bin edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`, then offset by -1 for 0-indexing.

ii.
```python
distance_to_reward_zone_bins = [-np.inf, -50, -10, 0, 1e-6, 10, 50, np.inf]
output_curr[0] = np.digitize(distance_to_reward_zone, distance_to_reward_zone_bins) - 1
```

iii. The bin edges match the instruction specification. The `1e-6` boundary separates exactly 0 (inside zone) from slightly positive values. The `-1` corrects `np.digitize` to 0-indexed output.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same time indices within each trial, so both are extracted using the same index array.

ii.
```python
idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
spike_data_curr = spike_data[idx]
position_curr = position[idx]
```

iii. Both neural and behavioral data use the same time indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = behavior_data['position'].data[:][:T]
position_curr = position[idx]
```

iii. The `position` variable directly records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
position_curr = position[idx]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `np.digitize` with bin edges `[-inf, 50, 150, 250, 350, inf]`, yielding approximately 100 cm wide bins.

ii.
```python
position_bins = [-np.inf, 50, 150, 250, 350, np.inf]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. The AI noted position ranged from approximately -50 to 450 cm and chose 100 cm bins. The reference code uses 90 cm bins (450/5 = 90) based on the 0-450 cm track length, clipping positions to [0, 450).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
position_curr = position[idx]  # same idx as spike_data[idx]
```

iii. Verified by timestamp assertions in the code.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
licks = behavior_data['lick'].data[:][:T]
licks_curr = licks[idx]
```

iii. The `lick` variable records lick events/counts at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
output_curr[3] = (licks_curr > 0).astype(int)
```

iii. The instructions specify binary output (no/yes). Raw lick values can be > 1 (cumulative counts), so thresholding at > 0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
licks_curr = licks[idx]  # same idx as spike_data[idx]
```

iii. Same frame-aligned sampling grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series combined with the `position` behavior time series. The AI uses a Viterbi algorithm (described in 7-a) to assign each trial a zone label (A, B, or C) based on the mouse's position when the `reward_zone` signal is active.

ii. See 7-a for the Viterbi implementation. Conversion to output:
```python
reward_zone = info['reward_zone_labels'][subject][trial_offset:trial_offset+ntrials]
reward_zone_curr = reward_zone[trial].item()
output_curr[4] = ['A','B','C'].index(reward_zone_curr)
```

iii. The reference code instead parses the scene name from the NWB `identifier` field (e.g., `Env1_LocationA_to_C`) and uses a switch rule (trial >= 30 switches to the destination zone). The AI's Viterbi approach infers zone labels from position data rather than metadata.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The Viterbi algorithm assigns zone labels based on the mouse's position when `reward_zone > 0`, encouraging stable assignments across consecutive trials. The zone label is then converted to a numeric code (A=0, B=1, C=2). The value is per-trial, repeated across timepoints.

ii.
```python
# From segment_reward_positions():
zone_labels = np.array([zone_names[s] for s in states])
# In convert_session_data():
output_curr[4] = ['A','B','C'].index(reward_zone_curr)
```

iii. The Viterbi approach produced approximately equal zone fractions (~1/3 each), consistent with the paper's counterbalanced design. The reference code's scene-parsing approach is more direct and uses established metadata.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. The `Reward` time series records reward delivery events with separate timestamps. The reference code additionally checks that `reward_zone > 0` during the trial, requiring both a reward delivery AND reward zone entry.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward event timestamps are mapped to the nearest behavior timepoint using `searchsorted`. For each trial, the output is 1 if any reward event occurred within the trial, 0 otherwise. Per-trial, repeated across timepoints.

ii.
```python
output_curr[5] = np.any(isreward[idx]).astype(int)
```

iii. The AI checks only for the presence of a reward timestamp within the trial. The reference code additionally requires `np.any(reward_zone_segment > 0)` to confirm the mouse entered the reward zone, which could differ for edge cases like auto-rewards.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: Data is cropped to the minimum of the two array lengths.
- **Short trials**: An attempted filter for trials with < 50 timepoints exists but contains a bug (`idx.sum()` instead of `len(idx)`).
- **Missing reward zone data**: The Viterbi algorithm handles trials with no reward zone activation via uniform emission probabilities.
- **Reward timestamp alignment**: Assertions verify alignment within half a time bin.

Missing from the code:
- **No lick-artifact trial filtering**: The reference code and paper describe dropping trials where >35% of frames have cumulative lick count > 2.
- **No per-frame validity filtering**: The reference code masks out frames with NaN/invalid behavior or neural values.

ii.
```python
if T != len(timestamps):
    T = min(T, len(timestamps))
    spike_data = spike_data[:T]
    timestamps = timestamps[:T]

# Bugged short trial filter:
if idx.sum() < min_ntimepoints:  # should be len(idx)
    istrial[trial] = False
```

iii. The neural/behavior mismatch occurs in a small number of sessions. The short-trial filter bug means no trials are actually filtered by length. The absence of lick-artifact filtering leaves potentially corrupted lick data in the dataset.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** via `pynwb.NWBHDF5IO` and reading large arrays (I/O bound)
2. **The survey step** which loads every NWB file to gather statistics and compute Viterbi zone assignments
3. **Full conversion** which loads every NWB file again to extract trial data
4. **Saving the pickle file**

ii. N/A (analysis question)

iii. NWB files are large. The two-pass design (survey + convert) doubles the I/O cost.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session_data` iterates sequentially over trials. Operations like `np.digitize` and `compute_distance_to_reward_zone` could be applied to full session arrays before splitting. The Viterbi inner loop over states could be vectorized with matrix operations.

ii. N/A (analysis question)

iii. Variable trial lengths make full vectorization awkward but not impossible with masking or padding.

## 13-c. What processing does the code repeat multiple times?

i. The survey and conversion steps both load every NWB file and read the same data arrays (neural data, behavior data). The survey reads neural data to count neurons and check rates, then the conversion reads it again. Reward zone positions are computed in the survey and reused, but other data is read twice.

ii. N/A (analysis question)

iii. The two-pass design was chosen because the Viterbi zone assignment (computed during survey) is needed before conversion can proceed. Merging passes would increase memory requirements.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The survey step collects many statistics (unique values, ranges, etc.) that are not directly used in the conversion or downstream analyses. The survey also saves multiple plot formats (PNG, PDF, SVG) of reward zone positions. However, the core conversion processing does not include obviously unnecessary computation.

ii. N/A (analysis question)

iii. The extensive survey statistics were useful during development but are not needed for the final conversion.
