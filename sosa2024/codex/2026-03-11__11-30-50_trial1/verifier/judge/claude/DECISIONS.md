# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from subdirectories of the `data/` directory using `pynwb`. It first runs a "survey" step that loads every NWB file to gather statistics and compute reward zone assignments (Viterbi), then runs a "conversion" step that loads every NWB file again to extract trial-level data. All 152 NWB files across 11 subjects are processed.

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
```
Loading:
```python
with pynwb.NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
```

iii. The AI found 11 subject directories matching `sub-*` and 152 NWB files total, consistent with the paper (11 mice, 14 sessions each except m11 which starts on day 3, yielding 11*14 - 2 = 152).

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories of `data/` matching the pattern `sub-<id>`. The subject ID is parsed from the directory name.

ii.
```python
def get_subjects(datadir=DATADIR):
    subjects = [d.split('-')[1] for d in os.listdir(datadir) if d.startswith('sub-')]
    return subjects
...
data['subjects'] = get_subjects(datadir=args.datadir)
```

iii. The naming convention `sub-<id>` is standard for BIDS-formatted neuroscience data. The count of 11 matches the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session numbers are parsed from filenames (e.g., `sub-m3_ses-01_behavior+ophys.nwb`).

ii.
```python
def get_sessions(subject, datadir=DATADIR):
    subject_dir = os.path.join(datadir, f"sub-{subject}")
    nwb_files = [f for f in os.listdir(subject_dir) if f.endswith('.nwb')]
    sessions = [int(f.split('_')[1].split('-')[1]) for f in nwb_files]
    return sessions
```

iii. Each NWB file stores one imaging session (one day). This is consistent with the paper's one-session-per-day design.

## 1-d. How are the data split into trials?

i. Trial boundaries are reconstructed from the `trial_start` and `teleport` behavior time series. Trial starts are identified by nonzero values of `trial_start`. Trial ends are identified by the rising edge of `teleport` (transition from <=0 to >0). An assertion verifies that the number of starts equals the number of ends.

ii.
```python
def get_trial_starts_ends(trial_start, teleport):
    trial_start_idx = np.nonzero(trial_start)[0]
    is_teleport_start = (teleport[1:] > 0) & (teleport[:-1] <= 0)
    trial_end_idx = np.nonzero(is_teleport_start)[0] + 1
    assert len(trial_start_idx) == len(trial_end_idx)
    return trial_start_idx, trial_end_idx
```

iii. The paper defines trials as laps from trial start to teleport. Using `trial_start` impulses and `teleport` rising edges matches this definition.

## 1-e. How are trials filtered based on quality controls?

i. The code intends to filter trials with fewer than 50 timepoints (`min_ntimepoints=50`). However, the filter condition uses `idx.sum()` (sum of index values) instead of `len(idx)` (number of timepoints), which means the filter almost never triggers. There is no lick-error-based trial filtering in the final code, despite the CONVERSION_NOTES mentioning it.

ii.
```python
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    if idx.sum() < min_ntimepoints:  # BUG: should be len(idx)
        print(f"Skipping trial {trial}...")
        istrial[trial] = False
        continue
```

iii. The AI's CONVERSION_NOTES mention lick-trial QC, but this was not implemented in the final code. The `idx.sum()` bug means the short-trial filter is essentially non-functional. No "Skipping trial" messages appear in the conversion output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p traces: `Fluorescence` (F) and `Neuropil` (Fneu) from the NWB ophys processing module. It is NOT the NWB's pre-computed `Deconvolved` field.

ii.
```python
fluorescence_data = nwbfile.processing["ophys"]["Fluorescence"]
neuropil_data = nwbfile.processing["ophys"]["Neuropil"]
for plane_name, plane_data in fluorescence_data.roi_response_series.items():
    fluorescence_curr = plane_data.data[:]
    neuropil_curr = neuropil_data.roi_response_series[plane_name].data[:]
```

iii. The paper computes its own dF/F and deconvolved events from raw fluorescence, rather than using a pre-stored deconvolved signal. The AI replicates this pipeline.

## 2-b. How is the `neural` data processed?

i. The AI copied the paper's `preprocessing.dff()` function and applies the full processing pipeline: neuropil subtraction (0.7 * Fneu), per-trial maximin baseline (Gaussian sigma=15, 300-sample min/max filters), dF/F computation, 2-sample Gaussian smoothing, and OASIS deconvolution with tau=0.7. The `keep_teleports` flag is set per-session using metadata from the paper's `teleport_metadata.py`. Multi-plane sessions use per-plane frame rates.

ii.
```python
dff_curr, events_curr = dff(fluorescence_curr.T.astype(np.float64),
                            pd.Series(trial_start_idx + 1), pd.Series(trial_end_idx + 1),
                            f_neu=neuropil_curr.T.astype(np.float64),
                            neuropil_method='subtract', baseline_method='maximin',
                            subtract_baseline=True, neu_coef=0.7, tau=0.7,
                            frame_rate=frame_rate, n_planes=nplanes, deconvolve=True,
                            keep_teleports=keep_teleports)
```

iii. This replicates the paper's Methods: "baseline fluorescence was calculated within each trial independently using a maximin procedure with a 20 s sliding window ... dF/F was then calculated ... smoothed with a two-sample (~0.129 s) s.d. Gaussian kernel. The activity rate was extracted by deconvolving dF/F with a canonical calcium kernel using the OASIS algorithm."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) ROIs are restricted to those with `iscell[:,0] == 1` (Suite2p manual curation); (2) putative interneurons are excluded if their dF/F correlates with running speed at r > 0.5.

ii.
```python
iscell = np.array(roi_data['iscell'].to_list())[:,0] == 1
fluorescence.append(fluorescence_curr[:,iscell])
neuropil.append(neuropil_curr[:,iscell])
...
isint = is_putative_interneuron(dff_curr, speed)
spike_data.append(events_curr[~isint].T)
```

iii. Both filters are described in the paper's Methods. The paper reports excluding 0.42 +/- 0.85% of cells as putative interneurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, which requires no additional processing beyond splitting data into trials using the trial start/end indices. The data is naturally aligned since neural and behavioral data share the same time indices.

ii. N/A (alignment is implicit in the trial slicing)

iii. The instructions specify "Temporally align based on start of the trial," which is achieved by extracting data from trial_start to teleport.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate (~15.5 Hz, ~64.5 ms per frame) is preserved. No temporal rebinning is applied. For multi-plane sessions, the per-plane rate is used (scanner_rate / n_planes).

ii.
```python
neural_time_bin_size = nplanes/plane_data.rate*1000  # ms
...
assert np.isclose(neural_time_bin_size, time_bin_size, rtol=1e-2)
```

iii. The paper states data is "sampled at ~15.5 Hz" with "0.0645 s imaging frame samples."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` array associated with the behavior time series (specifically `trial number`'s timestamps).

ii.
```python
timestamps = behavior_data['trial number'].timestamps[:]
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    timestamps_curr = timestamps[idx]
    input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. All behavior time series share the same timestamps. The choice of `trial number` timestamps is arbitrary since they are all identical.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of each trial is subtracted from all timestamps in that trial, yielding time from trial start in seconds.

ii.
```python
input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. Simple subtraction to make time relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices (verified by assertions). Both are indexed by the same `idx` array per trial. If neural and behavior arrays have different lengths, they are cropped to the minimum.

ii.
```python
if T != len(timestamps):
    print(f"Warning: ... cropping to the minimum of the two")
    T = min(T, len(timestamps))
```

iii. The paper states "All behavioral and neural time series were sampled at ~15.5 Hz," confirming they are on the same time grid.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series in the NWB file.

ii.
```python
vr_environment = behavior_data['environment'].data[:][:T]
```

iii. The `environment` variable contains values 0 and 1, corresponding to ENV1 and ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. No processing. The raw environment values (0 or 1) are used directly, cast to int.

ii.
```python
vr_environment_curr = vr_environment[idx].astype(int)
input_curr[1] = vr_environment_curr
```

iii. The environment variable already encodes ENV1=0, ENV2=1 in the NWB data.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the loop counter over reconstructed trials, not from the NWB `trial number` variable.

ii.
```python
for trial in range(ntrials):
    ...
    input_curr[2] = trial
```

iii. The within-session trial index (0, 1, 2, ...) is used. The stored `trial number` variable was not used because it did not agree with the `trial_start` variable.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial.

ii.
```python
input_curr[2] = trial
```

iii. The trial number is simply the sequential index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series. Reward event timestamps are mapped to behavior timepoints using `np.searchsorted`, and a binary `isreward` array marks frames where rewards occurred.

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. The `Reward` time series has its own timestamps (not aligned to the behavior sampling rate). An assertion checks alignment error is within half a time bin.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check whether any reward event occurred in the previous trial's time range. For the first trial, set to 0. The value is constant across all timepoints.

ii.
```python
if trial == 0:
    input_curr[3,:] = 0
else:
    prev_reward_idx = np.arange(trial_start_idx[trial-1], trial_end_idx[trial-1], dtype=int)
    isprevreward = np.any(isreward[prev_reward_idx])
    input_curr[3,:] = int(isprevreward)
```

iii. Binary encoding: omitted=0, rewarded=1, as specified in the instructions.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and a per-trial reward zone label (A, B, or C). The reward zone label is determined by a Viterbi algorithm applied during a preprocessing "survey" step, using the `reward_zone` behavior variable and `position` to identify which known zone (A=[80,130], B=[200,250], C=[320,370]) the animal was rewarded in.

ii.
```python
# Survey step builds reward zone positions:
reward_zone_trial = reward_zone[idx] > 0
info['reward_zone_positions'][trial] = [np.min(pos_trial[reward_zone_trial]),
                                         np.max(pos_trial[reward_zone_trial])]
# Viterbi assigns zone labels:
zone_ranges = np.array([reward_zone_dict[z] for z in zone_names])
dist = np.maximum(zone_ranges[None, :, 0] - startpoints[valid, None],
                  startpoints[valid, None] - zone_ranges[None, :, 1])
dist = np.maximum(dist, 0)
log_emission[valid] = -0.5 * (dist / sigma) ** 2
```

iii. The paper defines reward zones A, B, C at specific position ranges. The AI used a Viterbi approach to handle noise in the `reward_zone` variable and unrewarded trials. The resulting zone distribution is approximately balanced at 1/3 each.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, signed distance from the animal's position to the nearest edge of the reward zone is computed. Distance is 0 inside the zone, negative before, positive after.

ii.
```python
def compute_distance_to_reward_zone(position_curr, reward_zone_curr):
    reward_zone_range = reward_zone_dict[reward_zone_curr]
    distance_to_reward_zone = np.zeros(ntimepoints)
    idx_before_zone = position_curr < reward_zone_range[0]
    distance_to_reward_zone[idx_before_zone] = position_curr[idx_before_zone] - reward_zone_range[0]
    idx_after_zone = position_curr > reward_zone_range[1]
    distance_to_reward_zone[idx_after_zone] = position_curr[idx_after_zone] - reward_zone_range[1]
    return distance_to_reward_zone
```

iii. Matches the paper's concept of reward-relative distance.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using `np.digitize` with bin edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`, producing bins 0-6. The `1e-6` boundary separates exactly-zero (in-zone) from slightly positive.

ii.
```python
distance_to_reward_zone_bins = [-np.inf, -50, -10, 0, 1e-6, 10, 50, np.inf]
output_curr[0] = np.digitize(distance_to_reward_zone, distance_to_reward_zone_bins) - 1
```

iii. The bin edges match the instructions' specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data share the same time indices within each trial. No additional alignment needed.

ii.
```python
idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
spike_data_curr = spike_data[idx]
position_curr = position[idx]
```

iii. Same indexing for both neural and behavioral arrays.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = behavior_data['position'].data[:][:T]
position_curr = position[idx]
```

iii. The `position` variable records the animal's position in cm along the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice. The raw position values are used directly.

ii.
```python
position_curr = position[idx]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. The raw position is in cm on the 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `np.digitize` with edges `[-inf, 90, 180, 270, 360, inf]`, yielding 90 cm bins spanning the 450 cm track.

ii.
```python
position_bins = [-np.inf, 90, 180, 270, 360, np.inf]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. The instructions specify "5 equal-sized bins spanning the 450 cm track," which is 90 cm per bin.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial. No additional alignment needed.

ii. Same indexing: `position_curr = position[idx]`

iii. Verified by timestamp assertions.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
licks = behavior_data['lick'].data[:][:T]
licks_curr = licks[idx]
```

iii. The `lick` variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
output_curr[3] = (licks_curr > 0).astype(int)
```

iii. The instructions specify binary output (no/yes). Raw lick values can be >1, so thresholding at >0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data within each trial. No additional alignment needed.

ii. Same indexing: `licks_curr = licks[idx]`

iii. Verified by timestamp assertions.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series and `position` behavior time series, processed through the Viterbi algorithm during the survey step (see 7-a).

ii. See 7-a.

iii. See 7-a.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The Viterbi algorithm assigns each trial a zone label (A, B, or C) based on the position where reward_zone > 0 and known zone coordinates. The zone label is then mapped to 0/1/2 and repeated across all timepoints in the trial.

ii.
```python
reward_zone_curr = reward_zone[trial].item()
output_curr[4] = ['A','B','C'].index(reward_zone_curr)
```

iii. The resulting distribution is approximately 1/3 for each zone, consistent with the experimental design.

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

i. For each trial, the output is 1 if any reward event occurred within the trial's time range, 0 otherwise. The value is constant across all timepoints.

ii.
```python
output_curr[5] = np.any(isreward[idx]).astype(int)
```

iii. Binary per-trial output matching the ~85% reward / ~15% omission rate described in the paper.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Neural/behavior length mismatch**: If neural time bins and behavior timestamps differ in count, data is cropped to the minimum with a warning.
- **Short trials**: A filter for trials with fewer than 50 timepoints exists but is non-functional due to a bug (`idx.sum()` instead of `len(idx)`).
- **Missing reward zone data**: Trials where `reward_zone` is never active get NaN positions in the survey; the Viterbi algorithm handles these via uniform emission.
- **Reward timestamp alignment**: An assertion verifies reward times align within half a time bin.
- **Multi-plane sessions**: ROIs from multiple planes are correctly pooled.

ii.
```python
if T != len(timestamps):
    print(f"Warning: ... cropping to the minimum of the two")
    T = min(T, len(timestamps))
...
if idx.sum() < min_ntimepoints:  # BUG: should be len(idx)
    istrial[trial] = False
```

iii. The neural/behavior mismatch handling is defensive. The Viterbi approach handles missing reward zone data gracefully. The short-trial filter bug means effectively no short-trial filtering occurs.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **The survey step**: Loads every NWB file to gather statistics and compute reward zone positions via Viterbi.
2. **The dF/F + deconvolution computation**: The `dff()` function runs per-plane per-session, involving Gaussian smoothing, min/max filtering, and OASIS deconvolution.
3. **NWB file I/O**: Loading large fluorescence/neuropil arrays from HDF5 files via pynwb.
4. **Saving the pickle file**: The output file is ~4.5 GB.

ii. N/A

iii. The dF/F computation is particularly expensive since it processes every cell's full session time series.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session_data` (lines 569-634) iterates over each trial sequentially. Operations like `np.digitize`, `compute_distance_to_reward_zone`, and input/output construction could be applied to full session arrays before splitting into trials. The `is_putative_interneuron` function loops over each cell to compute correlations. The Viterbi inner loop iterates over states.

ii. N/A

iii. The per-trial loop is natural given variable-length trials. Vectorizing would require padding or masking.

## 13-c. What processing does the code repeat multiple times?

i. The survey step and conversion step both load every NWB file. The survey reads Deconvolved data (just to count neurons and check rates), while conversion reads Fluorescence and Neuropil. Behavior arrays (position, speed, lick, teleport, trial_start) are read in both steps. The survey could potentially be merged with conversion, but the Viterbi reward zone assignment requires a complete survey first.

ii. N/A

iii. The two-pass approach (survey then convert) was a design choice to separate data exploration from conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The survey step reads and processes the `Deconvolved` neural data from NWB (in `survey_session_data`) to count neurons and check rates, but this data is never used in the final conversion — the conversion step reads `Fluorescence` and `Neuropil` instead and recomputes everything from scratch. The survey also computes and saves reward zone position plots (`reward_zone_positions.png/pdf/svg`), which are diagnostic artifacts not used downstream.

ii.
```python
# In survey_session_data:
deconvolved_data = nwbfile.processing["ophys"]["Deconvolved"]
spike_data = []
# ... reads all deconvolved data just for shape/rate info
```

iii. The Deconvolved data read in the survey is purely for metadata validation; the actual neural signal comes from recomputing dF/F in the conversion step.
