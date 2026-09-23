# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all subdirectories starting with `sub-` in the data directory, and all `.nwb` files within each. Each NWB file is loaded twice: first in a `survey()` pass (to gather metadata and reward zone positions), then in `convert_session_data()` for the actual conversion. The AI uses `pynwb.NWBHDF5IO` to load the NWB files (the reference uses `h5py` directly).

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
...
with pynwb.NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
```

iii. The AI verified that 11 subject directories and 152 NWB files were found, matching the paper's description. Filenames are parsed for subject and session IDs.

## 1-b. How are the data split into subjects?

i. Subjects correspond to `sub-<id>` subdirectories in the data directory. The subject list is extracted from directory names.

ii.
```python
def get_subjects(datadir=DATADIR):
    subjects = [d.split('-')[1] for d in os.listdir(datadir) if d.startswith('sub-')]
    return subjects
...
data['subjects'] = get_subjects(datadir=args.datadir)
```

iii. 11 subjects found, matching the paper's count of 11 switch-task mice.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are parsed from the filename pattern `sub-<subject>_ses-<NN>_behavior+ophys.nwb`.

ii.
```python
def get_sessions(subject, datadir=DATADIR):
    subject_dir = os.path.join(datadir, f"sub-{subject}")
    nwb_files = [f for f in os.listdir(subject_dir) if f.endswith('.nwb')]
    sessions = [int(f.split('_')[1].split('-')[1]) for f in nwb_files]
    return sessions
```

iii. 152 sessions total (12 for m11 starting from day 3, 14 for all other mice), consistent with the paper.

## 1-d. How are the data split into trials?

i. Trial starts are identified from the `trial_start` behavior variable (nonzero entries). Trial ends are identified from the **rising edge** of the `teleport` variable (transition from <= 0 to > 0). Data between these indices constitutes one trial.

ii.
```python
def get_trial_starts_ends(trial_start, teleport):
    trial_start_idx = np.nonzero(trial_start)[0]
    is_teleport_start = (teleport[1:] > 0) & (teleport[:-1] <= 0)
    trial_end_idx = np.nonzero(is_teleport_start)[0] + 1
    assert len(trial_start_idx) == len(trial_end_idx)
    return trial_start_idx, trial_end_idx
```

iii. The AI investigated using `trial_number` but found it inconsistent with `trial_start`. Using `trial_start` and teleport rising edges produces consistent trial counts matching the paper (~80 trials/session).

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 50 timepoints are intended to be removed. However, the filter has a bug: `idx.sum()` computes the sum of index values rather than the number of timepoints (`len(idx)`). As a result, this filter effectively never triggers. **No lick-sensor error filtering is implemented**, despite the paper explicitly removing 81 trials (0.65%) where >30% of samples had cumulative lick count > 2.

ii.
```python
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    if idx.sum() < min_ntimepoints:  # BUG: should be len(idx)
        print(f"Skipping trial {trial}...")
        istrial[trial] = False
        continue
```

iii. The AI mentioned seeing a few short trials but did not implement the paper's lick-sensor error filtering (`behavior.correct_lick_sensor_error` with threshold 0.3).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` (F) and `Neuropil` (Fneu) arrays stored in the NWB file under `processing/ophys/Fluorescence` and `processing/ophys/Neuropil`. The NWB's `Deconvolved` field (suite2p's own deconvolution of raw F) is not used.

ii.
```python
fluorescence_data = nwbfile.processing["ophys"]["Fluorescence"]
neuropil_data = nwbfile.processing["ophys"]["Neuropil"]
for plane_name, plane_data in fluorescence_data.roi_response_series.items():
    fluorescence_curr = plane_data.data[:]
    neuropil_curr = neuropil_data.roi_response_series[plane_name].data[:]
```

iii. The AI correctly identified that the NWB `Deconvolved` array is suite2p's deconvolution of raw F, not the signal the paper analyzes. The paper recomputes its own dF/F and events from F and Fneu.

## 2-b. How is the `neural` data processed?

i. The AI copies the paper's `preprocessing.dff` function and calls it with `deconvolve=True`, producing OASIS-deconvolved "events" (not just dF/F). The processing pipeline is: (1) filter to `iscell` ROIs, (2) per-plane, pass F and Fneu to the copied `dff()` function which subtracts `0.7 * Fneu`, adds back the per-trial mean neuropil, applies a maximin baseline (Gaussian smooth sigma=15, then 300-sample minimum filter, then 300-sample maximum filter), computes `(F - baseline) / |baseline|`, smooths with a 2-sample Gaussian, and deconvolves with OASIS (`tau=0.7`, `frame_rate/n_planes`). The `keep_teleports` parameter is set conditionally based on `teleport_metadata.py` (whether the laser was blanked between trials for that mouse/day). The deconvolved events are used as the neural signal.

Note: The AI's CONVERSION_NOTES.md and metadata claim the neural signal is "dF/F" and state "We also do not re-deconvolve the dF/F", but the code clearly calls `dff(..., deconvolve=True)` and stores the `events_curr` output, not the `dff_curr` output.

ii.
```python
dff_curr, events_curr = dff(fluorescence_curr.T.astype(np.float64),
                            pd.Series(trial_start_idx + 1), pd.Series(trial_end_idx + 1),
                            f_neu=neuropil_curr.T.astype(np.float64),
                            neuropil_method='subtract', baseline_method='maximin',
                            subtract_baseline=True, neu_coef=0.7, tau=0.7,
                            frame_rate=frame_rate, n_planes=nplanes, deconvolve=True,
                            keep_teleports=keep_teleports)
...
spike_data.append(events_curr[~isint].T)  # uses events, not dff
```

iii. The AI's CONVERSION_NOTES describe replicating the paper's `preprocessing.dff` exactly. The deconvolution with OASIS at `tau=0.7` matches the paper's processing. The conditional `keep_teleports` based on `teleport_metadata.py` is more faithful to the paper than always using `keep_teleports=False`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) `iscell` curation (suite2p + manual, `iscell[:,0] == 1`), and (2) putative interneuron exclusion (cells with Pearson r(dF/F, speed) > 0.5).

ii.
```python
iscell = np.array(roi_data['iscell'].to_list())[:,0] == 1
fluorescence.append(fluorescence_curr[:,iscell])
neuropil.append(neuropil_curr[:,iscell])
...
isint = is_putative_interneuron(dff_curr, speed)
spike_data.append(events_curr[~isint].T)
```

iii. Both filters match the paper's Methods: suite2p curation and interneuron exclusion at r > 0.5. The AI's CONVERSION_NOTES report an interneuron exclusion rate consistent with the paper (~0.42 ± 0.85%).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is at trial start. Since the neural and behavioral data share the same time indexing (one sample per imaging frame), splitting at trial boundaries automatically aligns the data. No additional temporal shifting is needed.

ii. N/A (alignment is implicit via the shared indexing of neural and behavioral data at trial boundaries)

iii. The AI noted that neural and behavioral timestamps are consistent, verified by assertions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate (~15.5 Hz, ~64.48 ms per frame) is used. No temporal rebinning is applied. The time bin size is computed as `nplanes / frame_rate * 1000` ms, accounting for multi-plane sessions.

ii.
```python
neural_time_bin_size = nplanes/plane_data.rate*1000 # ms
...
assert np.isclose(neural_time_bin_size, time_bin_size, rtol=1e-2)
```

iii. The AI verified that the time bin size is consistent across all 152 sessions once multi-plane sessions are accounted for.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` associated with the `trial number` behavior time series.

ii.
```python
timestamps = behavior_data['trial number'].timestamps[:]
```

iii. The AI noted that all behavior time series share the same timestamps. Any could have been used.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial's first timestamp is subtracted from all timestamps in that trial to get time relative to trial start.

ii.
```python
timestamps_curr = timestamps[idx]
input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. Straightforward computation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data are already on the same time grid (one sample per imaging frame). The same index array `idx` is used for both, so they are inherently aligned.

ii. N/A (same `idx` array indexes both neural and behavioral data)

iii. The AI verified that behavior timestamps and neural data have the same number of samples (with a fallback to crop to the shorter one if mismatched).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
vr_environment = behavior_data['environment'].data[:][:T]
...
input_curr[1] = vr_environment_curr
```

iii. The environment variable takes values 0 (ENV1) or 1 (ENV2), matching the binary encoding specified.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Cast to int; no other processing. The raw values (0 or 1) are used directly.

ii.
```python
vr_environment_curr = vr_environment[idx].astype(int)
input_curr[1] = vr_environment_curr
```

iii. The environment is constant within a trial, confirmed by inspection.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the loop counter over trials (0-indexed). The stored `trial number` variable in the NWB was not used.

ii.
```python
for trial in range(ntrials):
    ...
    input_curr[2] = trial
```

iii. The AI noted that the stored `trial number` variable did not agree with `trial_start`, so the sequential loop index was used instead.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is broadcast across all timepoints in the trial.

ii.
```python
input_curr[2] = trial
```

iii. Trial number is simply the 0-based index of the trial within the session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series. Reward event timestamps are mapped to the nearest behavior frame using `searchsorted`, creating a binary `isreward` array.

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. The `Reward` time series uses its own timestamps (one per reward delivery), which are mapped to the imaging frame grid. An assertion checks alignment within half a time bin.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial > 0, check whether any reward event occurred in the previous trial's time range. For the first trial, set to 0. The AI does NOT incorporate the `reward_zone` variable (whether the animal was in the zone), unlike the reference which uses `any(reward) & any(rzone)`.

ii.
```python
if trial == 0:
    input_curr[3,:] = 0
else:
    prev_reward_idx = np.arange(trial_start_idx[trial-1], trial_end_idx[trial-1], dtype=int)
    isprevreward = np.any(isreward[prev_reward_idx])
    input_curr[3,:] = int(isprevreward)
```

iii. The AI checks only reward delivery events, not reward zone entry. The reference code implements `behavior.get_trial_types` which checks both `any(reward > 0)` and `any(rzone > 0)`. In practice, rewards are only delivered in the reward zone, so the difference may be negligible, but conceptually it deviates from the paper's definition.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` and `reward_zone` behavior time series. The reward zone label for each trial is determined using a Viterbi algorithm applied to observed reward zone entry positions, rather than parsing the scene name as the reference code does (`behavior.get_reward_zones`).

ii. The Viterbi approach first collects the start/end position of the animal when `reward_zone > 0` for each trial:
```python
# In survey_session_data:
pos_trial = position[idx]
reward_zone_trial = reward_zone[idx] > 0
info['reward_zone_positions'][trial] = [np.min(pos_trial[reward_zone_trial]),
                                         np.max(pos_trial[reward_zone_trial])]
```
Then `segment_reward_positions()` uses Viterbi with Gaussian emission and transition probabilities:
```python
reward_zone_dict = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}
sigma = 10.0
pstay = .9
log_emission[valid] = -0.5 * (dist / sigma) ** 2
log_trans = np.full((n_states, n_states), np.log((1 - pstay) / (n_states - 1)))
np.fill_diagonal(log_trans, np.log(pstay))
```

iii. The AI plotted reward zone positions and used the Viterbi algorithm to handle noise in the reward zone variable. The reference code derives the zone from the scene name + switch after 30 trials, which is simpler and metadata-driven.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, compute the signed distance from the animal's position to the nearest edge of the reward zone: negative if before the zone, 0 if inside, positive if past it.

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

iii. The distance computation is standard: 0 inside the zone, negative before, positive after.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized using `np.digitize` with bin edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`, producing 7 bins (0-6). The `1e-6` edge separates "exactly 0" (bin 3) from small positive values.

ii.
```python
distance_to_reward_zone_bins = [-np.inf, -50, -10, 0, 1e-6, 10, 50, np.inf]
output_curr[0] = np.digitize(distance_to_reward_zone, distance_to_reward_zone_bins) - 1
```

iii. The bin edges match the instruction specification. The 1e-6 boundary ensures exactly 0 maps to bin 3 ("0 cm").

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices (`idx`) are used for both neural and behavioral data within each trial. No additional alignment needed.

ii.
```python
idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
spike_data_curr = spike_data[idx]
position_curr = position[idx]
```

iii. Both share the same imaging frame grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = behavior_data['position'].data[:][:T]
...
position_curr = position[idx]
```

iii. The `position` variable records the animal's position in cm along the VR track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
position_curr = position[idx]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. Raw position values used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `np.digitize` with edges `[-inf, 90, 180, 270, 360, inf]`, producing 90 cm bins spanning the 450 cm track.

ii.
```python
position_bins = [-np.inf, 90, 180, 270, 360, np.inf]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. The bin edges match the instruction specification of 5 equal bins over 450 cm. Open-ended first and last bins handle the few samples slightly outside [0, 450].

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data. No additional alignment needed.

ii. `position_curr = position[idx]` (same `idx` as neural data)

iii. Shared imaging frame grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
licks = behavior_data['lick'].data[:][:T]
...
licks_curr = licks[idx]
```

iii. The `lick` variable records lick events (cumulative count per frame).

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value mapped to 1, otherwise 0.

ii.
```python
output_curr[3] = (licks_curr > 0).astype(int)
```

iii. Matches the instruction specification (binary: no/yes) and the reference code's `licks[licks>0] = 1`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data. No additional alignment needed.

ii. `licks_curr = licks[idx]` (same `idx` as neural data)

iii. Shared imaging frame grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` and `position` behavior time series via a Viterbi algorithm applied during the survey step, rather than from the scene name as in the reference code.

ii. See answer 7-a for the Viterbi approach. The zone label per trial is then mapped:
```python
reward_zone = info['reward_zone_labels'][subject][trial_offset:trial_offset+ntrials]
...
output_curr[4] = ['A','B','C'].index(reward_zone_curr)
```

iii. The AI validated that zone fractions were approximately 1/3 each. The reference code uses `behavior.get_reward_zones` which parses the scene name and switches at trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The Viterbi-assigned zone label ('A'/'B'/'C') is mapped to integers (0/1/2). The value is constant across all timepoints in the trial.

ii.
```python
reward_zone_curr = reward_zone[trial].item()
output_curr[4] = ['A','B','C'].index(reward_zone_curr)
```

iii. The AI's Viterbi approach should give the same labels as the scene-based approach when the data is clean, since the reward zone positions clearly cluster around the three known locations.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series (event timestamps mapped to imaging frames).

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. The Reward time series records reward delivery events.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the output is 1 if any reward event occurred within the trial's time range, 0 otherwise. The AI does NOT check whether the animal was in the reward zone (`reward_zone > 0`), unlike the reference which uses `any(reward) & any(rzone)`.

ii.
```python
output_curr[5] = np.any(isreward[idx]).astype(int)
```

iii. The reference code's `behavior.get_trial_types` defines `isreward = any(reward > 0) and any(rzone > 0)`. The AI checks only reward delivery. In practice, rewards are delivered only when the animal licks in the reward zone, so the results should be nearly identical, but the definitions differ.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive checks:
- **Neural/behavior length mismatch**: If neural and behavioral data have different lengths, data is cropped to the minimum with a warning.
- **Short trials**: A min-timepoints filter is intended but has a bug (`idx.sum()` instead of `len(idx)`), making it non-functional.
- **Missing reward zone data**: Trials where `reward_zone` is never active get NaN positions; the Viterbi algorithm handles these via uniform emission.
- **Reward timestamp alignment**: Assertions verify alignment within half a time bin.
- **Lick-sensor errors**: NOT handled. The paper removes 81 trials with sensor errors, but the AI does not implement this filtering.

ii.
```python
if T != len(timestamps):
    print(f"Warning: number of neural time bins ({T}) does not match...")
    T = min(T, len(timestamps))
...
if idx.sum() < min_ntimepoints:  # BUG: idx.sum() is sum of indices, not count
    istrial[trial] = False
```

iii. The AI's notes mention the neural/behavior mismatch occurring in some sessions. The missing lick-sensor error handling is a notable omission compared to the reference.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** (I/O bound, using pynwb which is slower than h5py)
2. **The survey step** — loads every NWB file to gather statistics and reward zone positions
3. **The dF/F computation** (including deconvolution) in `convert_session_data`
4. **Writing the pickle file**

ii. N/A

iii. The AI noted these in CONVERSION_NOTES but did not provide detailed timing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session_data` (iterating over each trial to extract neural, input, and output data) could potentially be partially vectorized for the output discretization steps, though variable trial lengths complicate this. The Viterbi inner loop in `segment_reward_positions` iterates over states and could be vectorized with matrix operations.

ii. N/A

iii. Variable trial lengths make full vectorization difficult. The per-trial loop is the natural structure.

## 13-c. What processing does the code repeat multiple times?

i. The survey step and the conversion step both load every NWB file and read the same data arrays. The survey reads neural data to count neurons and check rates, then the conversion reads it again. The `reward_zone_positions` computed in the survey are reused, but neural data, position, speed, lick, etc. are read twice.

ii. N/A

iii. The AI noted this was intentional because running both simultaneously would require too much RAM, and the survey results inform the conversion (Viterbi labels).

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The survey step computes various statistics (unique values, ranges) that are only used for diagnostic printing and not for the actual conversion. The `dff_curr` output from the `dff()` function is computed but not used (only `events_curr` is stored). The AI also saves survey results to a pickle file that is not needed by downstream analyses.

ii.
```python
dff_curr, events_curr = dff(...)  # dff_curr is discarded
spike_data.append(events_curr[~isint].T)  # only events used
```

iii. The dF/F is needed as an intermediate step for the interneuron correlation test, but the dF/F array itself is not stored in the output.
