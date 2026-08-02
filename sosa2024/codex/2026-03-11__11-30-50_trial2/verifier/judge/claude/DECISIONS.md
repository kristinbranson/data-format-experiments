# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are discovered by scanning `data/sub-*/` directories. Each NWB file is loaded using `pynwb.NWBHDF5IO`. A preliminary `survey()` step reads every NWB file to gather statistics and compute reward-zone labels, then the main conversion loop reads every file again to extract trial-level data.

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

iii. The code discovers all `sub-*` directories and all `.nwb` files within them, resulting in 11 subjects and 152 sessions matching the paper's switch-task cohort. The CONVERSION_NOTES confirm the count matches expectations.

## 1-b. How are the data split into subjects?

i. Subjects correspond to `sub-*` subdirectories under `data/`. The subject ID is parsed from the directory name by splitting on `-`.

ii.
```python
def get_subjects(datadir=DATADIR):
    subjects = [d.split('-')[1] for d in os.listdir(datadir) if d.startswith('sub-')]
    return subjects
...
data['subjects'] = get_subjects(datadir=args.datadir)
```

iii. The 11 directories match the 11 switch-task mice reported in the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session numbers are parsed from the filename pattern `sub-<subject>_ses-<NN>_behavior+ophys.nwb`.

ii.
```python
def get_sessions(subject, datadir=DATADIR):
    subject_dir = os.path.join(datadir, f"sub-{subject}")
    nwb_files = [f for f in os.listdir(subject_dir) if f.endswith('.nwb')]
    sessions = [int(f.split('_')[1].split('-')[1]) for f in nwb_files]
    return sessions
```

iii. This yields 152 sessions total (11 mice x 14 days minus 2 for m11 starting on day 3), consistent with the paper.

## 1-d. How are the data split into trials?

i. Trial boundaries are reconstructed from the `trial_start` and `teleport` behavior time series. Trial starts are frames where `trial_start` is nonzero. Trial ends are detected as the rising edge of the `teleport` signal (first frame where teleport transitions from <=0 to >0). Each trial spans from `trial_start_idx` to `trial_end_idx` (exclusive), so the teleport onset frame itself is excluded.

ii.
```python
def get_trial_starts_ends(trial_start, teleport):
    trial_start_idx = np.nonzero(trial_start)[0]
    is_teleport_start = (teleport[1:] > 0) & (teleport[:-1] <= 0)
    trial_end_idx = np.nonzero(is_teleport_start)[0] + 1
    assert len(trial_start_idx) == len(trial_end_idx)
    return trial_start_idx, trial_end_idx
...
idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
```

iii. The CONVERSION_NOTES state that `trial_number` was investigated but `trial_start`/`teleport` gave cleaner results. Trial counts per session (~80) are consistent with the paper.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 50 timepoints (`min_ntimepoints=50`) are skipped. No lick-artifact filtering is implemented in the code.

ii.
```python
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    if idx.sum() < min_ntimepoints:
        print(f"Skipping trial {trial}...")
        istrial[trial] = False
        continue
```

iii. The CONVERSION_NOTES mention lick-artifact removal and claim 69 trials were removed, but the actual code only implements a minimum-length filter. The paper describes removing trials where >30% of imaging frames have cumulative lick count >2 (removing 81/12,376 trials ~0.65%), which is not implemented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the `Deconvolved` field in `processing/ophys`.

ii.
```python
deconvolved_data = nwbfile.processing["ophys"]["Deconvolved"]
for plane_data in deconvolved_data.roi_response_series.values():
    spike_data_curr = plane_data.data[:]
```

iii. The paper states it trains decoders on deconvolved calcium activity (events), and the CONVERSION_NOTES confirm this choice.

## 2-b. How is the `neural` data processed?

i. Neurons from all imaging planes are combined by iterating over `roi_response_series`, filtering each plane by `iscell`, and concatenating along the neuron axis. For multi-plane sessions (31 Hz), no temporal rebinning is applied. The effective time bin size is computed as `nplanes / rate * 1000 ms`.

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
neural_time_bin_size = nplanes/plane_data.rate*1000  # ms
```

iii. The AI notes that data rates are the same for all recordings once multi-plane is accounted for. The reference paper notes m17 and m18 were imaged at ~31 Hz with two interleaved planes, yielding ~15.5 Hz per plane.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the `iscell` variable from the ROI table. Only ROIs where `iscell[:,0] == 1` are retained.

ii.
```python
roi_data = plane_data.rois[:]
iscell = np.array(roi_data['iscell'].to_list())[:,0] == 1
spike_data_curr = plane_data.data[:]
spike_data.append(spike_data_curr[:,iscell])
```

iii. The paper describes manual suite2p curation and an additional speed-correlation filter (excluding cells with dF/F-speed r > 0.5). The speed-correlation filter is not implemented, but the CONVERSION_NOTES acknowledge this as an archive-level limitation since the exact dF/F timeseries is not available.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data are aligned to trial start by extracting the slice from `trial_start_idx` to `trial_end_idx`. Since the first timepoint of each trial IS the trial start, no further temporal shifting is needed.

ii.
```python
spike_data_curr = spike_data[idx]  # idx = arange(trial_start_idx[trial], trial_end_idx[trial])
neural.append(spike_data_curr.T)
```

iii. The instructions specify alignment to trial start, which is inherent in the trial slicing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The code computes the time bin size as the mode of `nplanes / rate * 1000` across all sessions and checks that each session's rate matches. For 15.5 Hz sessions this yields ~64.5 ms bins. For 31 Hz sessions with 2 planes, the formula gives the same effective bin size, but the underlying data may still have 2x temporal resolution.

ii.
```python
neural_time_bin_size = nplanes/plane_data.rate*1000  # ms
...
values, counts = np.unique(info['neural_time_bin_size'], return_counts=True)
time_bin_size = values[np.argmax(counts)].item()
...
assert np.isclose(neural_time_bin_size, time_bin_size, rtol=1e-2)
```

iii. The CONVERSION_NOTES say "Timestamps are not available for the neural data, just a stored rate. This rate is the same for all recordings, once one accounts for the multi-plane recordings."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` array of the `trial number` behavior time series.

ii.
```python
timestamps = behavior_data['trial number'].timestamps[:]
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    timestamps_curr = timestamps[idx]
    input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. The behavior timestamps were verified to be consistent with the neural data rate.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of each trial is subtracted from all timestamps within that trial, giving seconds since trial onset.

ii.
```python
input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. Straightforward relative time computation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame indices. The same `idx` array is used for both. An assertion checks that the number of neural and behavior timepoints match (with cropping to minimum if they differ).

ii.
```python
if T != len(timestamps):
    T = min(T, len(timestamps))
    spike_data = spike_data[:T]
    timestamps = timestamps[:T]
```

iii. Both streams have the same sampling rate, so indexing by the same `idx` ensures alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavior time series.

ii.
```python
vr_environment = behavior_data['environment'].data[:][:T]
...
vr_environment_curr = vr_environment[idx].astype(int)
input_curr[1] = vr_environment_curr
```

iii. The environment variable has values 0 or 1 within valid trial periods, matching ENV1/ENV2 from the paper.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The raw environment values are cast to int and stored time-varying (per-timepoint within the trial), rather than taking a single per-trial value.

ii.
```python
input_curr[1] = vr_environment_curr
```

iii. Since environment is constant within a trial, the time-varying representation is functionally equivalent to a per-trial value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The trial number is the within-session loop counter (0-indexed), not from the NWB `trial number` variable.

ii.
```python
for trial in range(ntrials):
    ...
    input_curr[2] = trial
```

iii. The CONVERSION_NOTES state that the stored `trial number` variable did not agree with `trial_start`, so the sequential index was used instead.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is assigned directly as a constant across all timepoints in the trial.

ii.
```python
input_curr[2] = trial
```

iii. Simple 0-indexed sequential numbering within the session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series. Reward events have separate timestamps that are mapped to the nearest behavior frame using `searchsorted`.

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. An assertion checks that the alignment error between reward timestamps and behavior frames is within half a time bin.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the code checks whether any reward event occurred within the previous trial's time range. The first trial defaults to 0. The result is constant across all timepoints.

ii.
```python
if trial == 0:
    input_curr[3,:] = 0
else:
    prev_reward_idx = np.arange(trial_start_idx[trial-1], trial_end_idx[trial-1], dtype=int)
    isprevreward = np.any(isreward[prev_reward_idx])
    input_curr[3,:] = int(isprevreward)
```

iii. The instructions specify binary previous trial outcome (omitted=0, rewarded=1).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone assignment for each trial. The reward zone labels (A, B, C) are determined in a `survey()` preprocessing step that examines positions where `reward_zone > 0` and applies a Viterbi algorithm to assign zone labels to each trial.

ii. Survey step collecting reward zone positions:
```python
reward_zone = behavior_data['reward_zone'].data[:]
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    pos_trial = position[idx]
    reward_zone_trial = reward_zone[idx] > 0
    if not np.any(reward_zone_trial):
        continue
    info['reward_zone_positions'][trial] = [np.min(pos_trial[reward_zone_trial]),
                                             np.max(pos_trial[reward_zone_trial])]
```
Viterbi zone assignment:
```python
reward_zone_dict = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}
...
log_emission[valid] = -0.5 * (dist / sigma) ** 2
```

iii. The Viterbi algorithm encourages stable zone assignments (high self-transition probability `pstay=0.9`) while using position information as emission evidence. The resulting zone fractions (~1/3 each) match expectations.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the signed distance from the animal's position to the nearest edge of the assigned reward zone is computed. Distance is 0 when inside the zone, negative when before it, positive when past it.

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

iii. The reward zone boundaries (A=[80,130], B=[200,250], C=[320,370]) match the paper. The distance computation follows the paper's reward-relative concept.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using `np.digitize` with edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`. The `1e-6` boundary separates exactly 0 (inside zone boundary) from slightly positive values.

ii.
```python
distance_to_reward_zone_bins = [-np.inf, -50, -10, 0, 1e-6, 10, 50, np.inf]
output_curr[0] = np.digitize(distance_to_reward_zone, distance_to_reward_zone_bins) - 1
```

iii. The bin edges follow the instruction specification: <-50, -50 to -10, -10 to <0, 0, >0 to +10, +10 to +50, >+50.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same time indices within each trial, so no additional alignment is needed.

ii.
```python
idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
spike_data_curr = spike_data[idx]
position_curr = position[idx]
```

iii. Verified by timestamp assertions comparing behavior and neural timepoints.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = behavior_data['position'].data[:][:T]
assert np.allclose(behavior_data['position'].timestamps[:][:T], timestamps)
...
position_curr = position[idx]
```

iii. The `position` variable records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice. The raw position values are used directly without clipping.

ii.
```python
position_curr = position[idx]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. The raw position values are used as-is.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `np.digitize` with edges `[-inf, 50, 150, 250, 350, inf]`. This produces bins of widths approximately 50, 100, 100, 100, 100 cm given the observed position range (~0 to ~450 cm).

ii.
```python
position_bins = [-np.inf, 50, 150, 250, 350, np.inf]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. The CONVERSION_NOTES say "The corridor spans roughly -50 to 450 cm (from survey), so 100 cm bins are approximately equal-sized." However, the standard track length is 450 cm (0 to 450), and the bins are not truly equal-sized (the first bin is ~50 cm wide vs ~100 cm for the others).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii. `position_curr = position[idx]` using the same `idx` as neural data.

iii. Verified by timestamp assertions.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
licks = behavior_data['lick'].data[:][:T]
assert np.allclose(behavior_data['lick'].timestamps[:][:T], timestamps)
...
licks_curr = licks[idx]
```

iii. The `lick` variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
output_curr[3] = (licks_curr > 0).astype(int)
```

iii. The instructions specify binary output (no/yes). Raw lick values can be >1.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii. `licks_curr = licks[idx]` using the same `idx` as neural data.

iii. Verified by timestamp assertions.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` and `position` behavior time series, processed through the Viterbi-based zone inference in the `survey()` step (same as described in 7-a).

ii. See 7-a.

iii. See 7-a.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The Viterbi algorithm assigns each trial to zone A (0), B (1), or C (2) based on positions where `reward_zone > 0`. Trials without reward zone activity (e.g., omission trials) are assigned based on neighboring trials' transition probabilities. The assigned zone label is mapped to a categorical index.

ii.
```python
reward_zone_curr = reward_zone[trial].item()
output_curr[4] = ['A','B','C'].index(reward_zone_curr)
```

iii. The paper defines three reward zones A, B, C that switch after 30 trials on switch days. The Viterbi approach handles noisy or missing zone data.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series, which has separate timestamps from the main behavior sampling.

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. The `Reward` time series records reward delivery events.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward event timestamps are mapped to behavior frames using `searchsorted`. For each trial, the output is 1 if any reward event occurred within the trial's time range, 0 otherwise. Constant across all timepoints in the trial.

ii.
```python
output_curr[5] = np.any(isreward[idx]).astype(int)
```

iii. An alignment-error check verifies reward timestamps are within half a time bin of behavior timestamps.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: If the number of neural frames differs from behavior timestamps, data is cropped to the minimum of the two.
- **Short trials**: Trials with fewer than 50 timepoints are skipped.
- **Missing reward zone data**: Trials where `reward_zone` is never active get NaN positions in the survey; the Viterbi algorithm handles these via uniform emission (assignment driven by neighboring trials).
- **Reward timestamp alignment**: An assertion verifies alignment within half a time bin.

ii.
```python
if T != len(timestamps):
    print(f"Warning: ...")
    T = min(T, len(timestamps))
    spike_data = spike_data[:T]
    timestamps = timestamps[:T]
...
if idx.sum() < min_ntimepoints:
    istrial[trial] = False
    continue
```

iii. The CONVERSION_NOTES document 10 sessions with 1-sample length mismatches between neural and behavior data.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** via `pynwb.NWBHDF5IO` - I/O bound
2. **The survey step** - loads every NWB file to gather statistics and compute reward zone labels
3. **Full conversion** - loads every NWB file again for trial extraction
4. **Saving the pickle file**

ii. N/A (timing information in CONVERSION_NOTES: full conversion ~95 s for 152 sessions, ~1.2-1.35 s/session)

iii. NWB files are large; the conversion itself is computationally lightweight once data is loaded.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session_data` iterates over each trial sequentially. Operations like `np.digitize`, `compute_distance_to_reward_zone`, and input/output construction could theoretically be applied to session-wide arrays before splitting into trials. The Viterbi inner loop over states could be vectorized with matrix operations.

ii. N/A

iii. Variable trial lengths make full vectorization awkward, requiring padding or masking.

## 13-c. What processing does the code repeat multiple times?

i. The `survey()` step and `convert_session_data()` both load every NWB file and read the same data arrays (neural data, behavior data). The survey reads neural data to count neurons and check rates, then the conversion reads it again. The reward zone positions computed during survey are reused, but other data (position, speed, lick) is read twice.

ii. N/A

iii. The survey must complete before conversion because the Viterbi-based reward zone labels are needed during conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The survey step computes many diagnostic statistics (max spike rate ranges, unique variable values, position ranges, etc.) and generates reward zone position plots. These are used for human inspection and quality control but are not consumed by the conversion or the downstream decoder. The survey also saves its results to a pickle cache file.

ii. N/A

iii. While not strictly necessary for conversion, these diagnostics serve a useful QC purpose.
