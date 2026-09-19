# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data from NWB files in the `data/` directory. It first runs a "survey" step that scans all NWB files to gather metadata (neuron counts, reward zone positions, time bin sizes), then runs the full conversion. Data is loaded using `pynwb.NWBHDF5IO`. All subdirectories matching `sub-*` are enumerated, and all `.nwb` files within each are processed.

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

iii. The code finds all `sub-*` directories and all `.nwb` files within them. The number of subjects (11) matches the paper. The AI verified this during data exploration (Step 2 of CONVERSION_NOTES.md).

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories of `data/` named `sub-<id>`. The subject ID is parsed from the directory name.

ii.
```python
def get_subjects(datadir=DATADIR):
    subjects = [d.split('-')[1] for d in os.listdir(datadir) if d.startswith('sub-')]
    return subjects

data['subjects'] = get_subjects(datadir=args.datadir)
```

iii. Directory naming convention `sub-<id>` is standard for BIDS-formatted data. The count of 11 subjects matches the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are identified by parsing the session number from the filename pattern `sub-<subject>_ses-<number>_behavior+ophys.nwb`.

ii.
```python
def get_sessions(subject, datadir=DATADIR):
    subject_dir = os.path.join(datadir, f"sub-{subject}")
    nwb_files = [f for f in os.listdir(subject_dir) if f.endswith('.nwb')]
    sessions = [int(f.split('_')[1].split('-')[1]) for f in nwb_files]
    return sessions
```

iii. The NWB file naming convention includes a parsable `ses-<number>` field. The resulting session counts per subject match the dataset structure.

## 1-d. How are the data split into trials?

i. Trials are defined using the `trial_start` and `teleport` behavior time series. Trial starts are identified as timepoints where `trial_start` is nonzero. Trial ends are identified as the rising edge of the `teleport` signal (transition from <=0 to >0). Data between these timepoints constitutes a trial.

ii.
```python
def get_trial_starts_ends(trial_start, teleport):
    trial_start_idx = np.nonzero(trial_start)[0]
    is_teleport_start = (teleport[1:] > 0) & (teleport[:-1] <= 0)
    trial_end_idx = np.nonzero(is_teleport_start)[0] + 1
    assert len(trial_start_idx) == len(trial_end_idx)
    return trial_start_idx, trial_end_idx
```

iii. The AI investigated using the `trial_number` variable but found it did not agree with the `trial_start` variable. The trial_start/teleport approach was chosen for cleaner data. However, the verification output shows positions only in bins 2-4 (>=180 cm), suggesting this trial definition may exclude the early portion of each trial where positions are 0-180 cm.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 50 timepoints (`min_ntimepoints=50`) are removed.

ii.
```python
if idx.sum() < min_ntimepoints:
    print(f"Skipping trial {trial}...")
    istrial[trial] = False
    continue
```

iii. This filters a small number of very short trials that would not provide meaningful data for decoding.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p traces: `Fluorescence` (F) and `Neuropil` (Fneu) from `processing/ophys/Fluorescence` and `processing/ophys/Neuropil`. The NWB's `Deconvolved` field is NOT used, as it represents suite2p's own deconvolution rather than the paper's custom processing.

ii.
```python
fluorescence_data = nwbfile.processing["ophys"]["Fluorescence"]
neuropil_data = nwbfile.processing["ophys"]["Neuropil"]
for plane_name, plane_data in fluorescence_data.roi_response_series.items():
    fluorescence_curr = plane_data.data[:]
    neuropil_curr = neuropil_data.roi_response_series[plane_name].data[:]
```

iii. The paper computes its own neural signal via `preprocessing.dff(F, Fneu, ..., deconvolve=True)`. The AI copied the paper's `dff()` function and uses it to recompute the signal from raw fluorescence, matching the paper's methodology.

## 2-b. How is the `neural` data processed?

i. The AI applies the paper's full processing pipeline per plane:
1. Restrict to lap samples (or lap+teleport based on per-session teleport metadata)
2. Subtract `0.7 * Fneu` (neuropil correction)
3. Add back per-trial mean neuropil for true dF/F ratio
4. Compute maximin baseline (Gaussian smooth sigma=15, 300-sample running min, 300-sample running max)
5. Compute dF/F = (F - baseline) / |baseline|
6. Smooth with 2-sample Gaussian
7. Deconvolve with OASIS (tau=0.7, frame_rate/n_planes)
8. Pool cells from multiple planes

The `keep_teleports` flag is set based on `teleport_metadata.py` (copied from the paper's code), which specifies which sessions had continuous imaging during the teleport.

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

iii. This matches the Methods description: "baseline fluorescence was calculated within each trial independently using a maximin procedure with a 20 s sliding window... dF/F was then calculated... smoothed with a two-sample Gaussian kernel. The activity rate was extracted by deconvolving dF/F with a canonical calcium kernel using the OASIS algorithm."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied, both from the paper:
1. Only `iscell` ROIs are kept (suite2p's manual curation)
2. Putative interneurons are dropped: cells whose dF/F correlates with running speed at r > 0.5

ii.
```python
iscell = np.array(roi_data['iscell'].to_list())[:,0] == 1
fluorescence.append(fluorescence_curr[:,iscell])
neuropil.append(neuropil_curr[:,iscell])
...
isint = is_putative_interneuron(dff_curr, speed)
spike_data.append(events_curr[~isint].T)
```

iii. The `iscell` filter is the paper's manual curation. The interneuron filter matches the Methods: "a Pearson correlation of >0.5". The `is_putative_interneuron()` function was copied from the paper's `spatial.py`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to the trial start. Since the neural data is already split into trials starting at `trial_start_idx`, no additional alignment is needed.

ii. N/A (alignment is implicit in the trial splitting)

iii. The instructions specify "Temporally align based on start of the trial." The trial splitting already produces data starting at trial onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data is kept at the original stored rate with no rebinning. The time bin size is computed from the stored frame rate and number of imaging planes: `nplanes / rate * 1000` ms. For single-plane sessions this is ~64.5 ms; for two-plane sessions the scanner rate is doubled but the per-plane rate is the same.

ii.
```python
neural_time_bin_size = nplanes/plane_data.rate*1000 # ms
assert np.isclose(neural_time_bin_size, time_bin_size, rtol=1e-2)
```

iii. No rebinning is needed as the time bin size is consistent across sessions once multi-plane recording is accounted for.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` associated with behavior time series (specifically the `trial number` time series timestamps, though all behavior streams share the same timestamps).

ii.
```python
timestamps = behavior_data['trial number'].timestamps[:]
...
input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. Behavior timestamps are consistent with the neural data rate. Multiple behavior variables share the same timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The initial timestamp for each trial is subtracted from all timestamps within that trial.

ii.
```python
timestamps_curr = timestamps[idx]
input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. Simple relative timing computation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same sampling rate and time indices. The AI verifies this by asserting that behavior timestamps match neural time bins, and crops to the minimum length if they differ.

ii.
```python
if T != len(timestamps):
    T = min(T, len(timestamps))
    fluorescence = [f[:T] for f in fluorescence]
    timestamps = timestamps[:T]
```

iii. Data rates for neural and behavior are the same, verified by assertions.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series in the NWB file.

ii.
```python
vr_environment = behavior_data['environment'].data[:][:T]
...
vr_environment_curr = vr_environment[idx].astype(int)
input_curr[1] = vr_environment_curr
```

iii. The environment variable is constant within each trial and takes values 0 or 1, matching ENV1/ENV2 in the paper.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. No processing beyond extracting the per-trial values and casting to int. The variable is used as-is.

ii.
```python
input_curr[1] = vr_environment_curr
```

iii. The raw values already represent the binary environment type.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the loop counter over trials (0, 1, 2, ...), not from the NWB `trial number` variable. The trial boundaries come from `trial_start` and `teleport`.

ii.
```python
for trial in range(ntrials):
    ...
    input_curr[2] = trial
```

iii. The stored `trial number` variable was not used because it did not agree with the `trial_start` variable. The loop index provides a sequential trial count within each session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing. The loop index is assigned directly as a constant value across all timepoints within the trial.

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

iii. The `Reward` time series has its own timestamps (different from the behavior sampling rate), so `searchsorted` aligns them.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check whether any reward event occurred in the previous trial's time range. For the first trial, set to 0. The value is constant across all timepoints within the trial.

ii.
```python
if trial == 0:
    input_curr[3,:] = 0
else:
    prev_reward_idx = np.arange(trial_start_idx[trial-1], trial_end_idx[trial-1], dtype=int)
    isprevreward = np.any(isreward[prev_reward_idx])
    input_curr[3,:] = int(isprevreward)
```

iii. Binary encoding as specified: omitted=0, rewarded=1. Uses the previous trial's time boundaries to check for reward events.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone location for each trial. The reward zone boundaries come from the `reward_zone` behavior variable combined with position data. The AI uses a Viterbi algorithm to assign reward zone labels (A, B, or C) to each trial, using known zone positions: A=[80,130], B=[200,250], C=[320,370].

ii.
```python
# Survey step: find reward zone positions per trial
reward_zone_trial = reward_zone[idx] > 0
info['reward_zone_positions'][trial] = [np.min(pos_trial[reward_zone_trial]),
                                         np.max(pos_trial[reward_zone_trial])]

# Viterbi assignment
reward_zone_dict = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}
zone_ranges = np.array([reward_zone_dict[z] for z in zone_names])
dist = np.maximum(zone_ranges[None, :, 0] - startpoints[valid, None],
                  startpoints[valid, None] - zone_ranges[None, :, 1])
dist = np.maximum(dist, 0)
log_emission[valid] = -0.5 * (dist / sigma) ** 2
```

iii. The Viterbi algorithm assigns zones based on the starting position of the rewarded zone, with a transition penalty to discourage switching. This handles noise and missing data robustly.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, compute the signed distance from the animal's position to the nearest edge of the assigned reward zone. Distance is 0 when inside the zone, negative when before the zone, and positive when past it.

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

iii. The reward zone boundaries are taken from `reward_zone_dict`, inspired by the reference code in `Sosa_et_al_2024/src/reward_relative/behavior.py`.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins using `np.digitize` with bin edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`.

ii.
```python
distance_to_reward_zone_bins = [-np.inf, -50, -10, 0, 1e-6, 10, 50, np.inf]
output_curr[0] = np.digitize(distance_to_reward_zone, distance_to_reward_zone_bins) - 1
```

iii. The bin edges match the instructions. The `1e-6` boundary separates exactly 0 (inside zone boundary) from slightly positive. The `-1` corrects `np.digitize` to be 0-indexed.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data share the same time indices within each trial, so no additional alignment is needed.

ii.
```python
idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
spike_data_curr = spike_data[idx]
position_curr = position[idx]
```

iii. Both are indexed by the same `idx` array within each trial.

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

i. Discretized into 5 bins using `np.digitize` with fixed bin edges `[-inf, 90, 180, 270, 360, inf]`, corresponding to 90 cm wide bins spanning the 450 cm track.

ii.
```python
position_bins = [-np.inf, 90, 180, 270, 360, np.inf]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. The instructions specify "5 equal-sized bins spanning the 450 cm track", which gives 90 cm per bin. However, the verification output shows only bins 2, 3, 4 are populated (positions >= 180 cm), suggesting the trial definition excludes the early portion of each trial where positions are 0-180 cm.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial.

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

i. Same time indices as neural data within each trial.

ii. Same indexing: `licks_curr = licks[idx]`

iii. Verified by timestamp assertions.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series and `position` behavior time series, processed through the Viterbi algorithm described in 7-a. Each trial is assigned a zone label (A, B, or C).

ii. See 7-a for the Viterbi zone assignment code.

iii. The Viterbi approach handles noisy reward_zone values and missing data robustly.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The Viterbi-assigned zone label is mapped to an integer: A=0, B=1, C=2. The value is constant across all timepoints in the trial.

ii.
```python
reward_zone_curr = reward_zone[trial].item()
output_curr[4] = ['A','B','C'].index(reward_zone_curr)
```

iii. Direct mapping from zone name to integer index.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series, which has its own timestamps separate from the behavior sampling rate.

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
output_curr[5] = np.any(isreward[idx]).astype(int)
```

iii. Alignment error is verified to be within half a time bin. `np.any` captures whether the trial was rewarded.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: If lengths differ, data is cropped to the minimum with a warning.
- **Short trials**: Trials with <50 timepoints are skipped.
- **Missing reward zone data**: Trials where `reward_zone` is never active get NaN positions; the Viterbi algorithm handles these via uniform emission.
- **Reward timestamp alignment**: An assertion verifies alignment within half a time bin.

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
1. Loading NWB files (I/O bound, large files)
2. The survey step (loads every NWB file to gather statistics)
3. Full conversion (loads every NWB file again for extraction)
4. The dff() computation (per-plane fluorescence processing with OASIS deconvolution)
5. Saving the .pkl file

ii. N/A

iii. NWB files are large and contain full neural recordings. The dff() processing adds computation time compared to using pre-computed Deconvolved data.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session_data` iterates over each trial sequentially. Operations like `np.digitize` and `compute_distance_to_reward_zone` could theoretically be applied to full session arrays before splitting. The Viterbi inner loop over states could use matrix operations.

ii. N/A

iii. Variable trial lengths make vectorization awkward, requiring padding or masking.

## 13-c. What processing does the code repeat multiple times?

i. The survey step and conversion step both load every NWB file and read the same data arrays. The survey reads neural data to count neurons, then conversion reads it again for processing. Reward zone positions are computed in the survey and used in conversion, but other data (position, speed, lick) is read twice.

ii. N/A

iii. The two-pass approach (survey then convert) was necessary because Viterbi reward zone assignment from the survey is needed before conversion can proceed.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `dff()` function computes both dF/F and deconvolved events, but only the deconvolved events are used as neural data. The dF/F traces are only used for the interneuron correlation check and then discarded. Additionally, the survey step computes various statistics (position ranges, lick values, etc.) that are informational but not directly used in conversion.

ii. N/A

iii. The dF/F computation is required as an intermediate step for deconvolution, so it's not entirely unnecessary, but it does consume memory.
