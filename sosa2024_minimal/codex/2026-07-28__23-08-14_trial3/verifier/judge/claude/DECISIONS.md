# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all subdirectories of the data directory matching the pattern `sub-*`, and within each finds all `.nwb` files. Each NWB file is loaded using `pynwb.NWBHDF5IO`. Neural data is read from `processing/ophys/Fluorescence` and `processing/ophys/Neuropil`, and behavioral data from `processing/behavior/BehavioralTimeSeries`. A preliminary "survey" pass loads every NWB file to collect statistics and compute reward zone labels, then a second "conversion" pass re-loads each file to extract trial-level data.

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

iii. The agent explored the data directory structure, confirmed 11 subject directories matching the paper, and verified that all NWB files were parsable. It used pynwb following standard NWB documentation.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories of the data directory, each named `sub-<id>`. The subject ID is extracted by splitting on `-`.

ii.
```python
def get_subjects(datadir=DATADIR):
    subjects = [d.split('-')[1] for d in os.listdir(datadir) if d.startswith('sub-')]
    return subjects
```

iii. The number of subject directories (11) matches the paper's description.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session number is parsed from the filename pattern `sub-<subject>_ses-<number>_behavior+ophys.nwb`.

ii.
```python
def get_sessions(subject, datadir=DATADIR):
    subject_dir = os.path.join(datadir, f"sub-{subject}")
    nwb_files = [f for f in os.listdir(subject_dir) if f.endswith('.nwb')]
    sessions = [int(f.split('_')[1].split('-')[1]) for f in nwb_files]
    return sessions
```

iii. The `ses-<number>` naming convention in the NWB filenames identifies sessions.

## 1-d. How are the data split into trials?

i. Trial start is identified by frames where `trial_start > 0`. Trial end is identified by detecting the rising edge of the `teleport` signal (transition from <= 0 to > 0). Data between these boundaries constitutes a trial.

ii.
```python
def get_trial_starts_ends(trial_start, teleport):
    trial_start_idx = np.nonzero(trial_start)[0]
    is_teleport_start = (teleport[1:] > 0) & (teleport[:-1] <= 0)
    trial_end_idx = np.nonzero(is_teleport_start)[0] + 1
    assert len(trial_start_idx) == len(trial_end_idx)
    return trial_start_idx, trial_end_idx
```

iii. The agent investigated using the `trial_number` variable but found it inconsistent with `trial_start`. The rising-edge detection for teleport ensures exactly one end index per teleport event. Trial counts were consistent with expectations.

## 1-e. How are trials filtered based on quality controls?

i. The code attempts to filter trials with fewer than 50 timepoints. However, the filtering condition uses `idx.sum()` (sum of index values) rather than `len(idx)` (number of timepoints), so this filter is effectively non-functional for trials at any position beyond the first ~50 frames of the session.

ii.
```python
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    if idx.sum() < min_ntimepoints:  # Bug: should be len(idx)
        istrial[trial] = False
        continue
```

iii. The agent mentioned that "looking at trial lengths, there were a couple short trials that this removed," but due to the bug, the filter would rarely if ever trigger. No lick-sensor error filtering is applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` (F) and `Neuropil` (Fneu) traces stored in the NWB file under `processing/ophys/Fluorescence` and `processing/ophys/Neuropil`. The NWB's pre-computed `Deconvolved` field is NOT used.

ii.
```python
fluorescence_data = nwbfile.processing["ophys"]["Fluorescence"]
neuropil_data = nwbfile.processing["ophys"]["Neuropil"]
for plane_name, plane_data in fluorescence_data.roi_response_series.items():
    fluorescence_curr = plane_data.data[:]
    neuropil_curr = neuropil_data.roi_response_series[plane_name].data[:]
```

iii. The agent recognized that the paper computes its own neural signal from raw traces using `preprocessing.dff()`, and that the NWB's `Deconvolved` field is suite2p's own deconvolution which the paper does not use.

## 2-b. How is the `neural` data processed?

i. The paper's full dF/F and deconvolution pipeline is reproduced: neuropil subtraction (F - 0.7*Fneu), add back per-trial neuropil mean, maximin baseline (Gaussian smooth sigma=15, then 300-sample running min then max), compute dF/F = (F - baseline) / |baseline|, smooth with 2-sample Gaussian, and deconvolve with OASIS (tau=0.7, frame_rate/n_planes). The `keep_teleports` flag is set per session using the paper's `teleport_metadata.py` table. All planes are processed separately and then pooled.

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

iii. The agent copied the paper's `dff()` function and its dependencies (`nansmooth`, `nansmooth_1d`), adapting minor details. Parameters match those in the paper's notebooks (neu_coef=0.7, tau=0.7, maximin baseline). The `teleport_sessions` dictionary was copied from the paper's `teleport_metadata.py`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) Only ROIs marked as `iscell` by suite2p's manual curation are retained. (2) Putative interneurons (cells whose dF/F correlates with running speed at r > 0.5) are excluded.

ii.
```python
iscell = np.array(roi_data['iscell'].to_list())[:,0] == 1
fluorescence.append(fluorescence_curr[:,iscell])
neuropil.append(neuropil_curr[:,iscell])
...
isint = is_putative_interneuron(dff_curr, speed)
spike_data.append(events_curr[~isint].T)
```

iii. The `iscell` filter applies the paper's manual curation. The interneuron filter matches the paper's Methods: "a Pearson correlation of >0.5" between dF/F and running speed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instructions specify alignment to trial start. Since the neural data is split at trial boundaries (trial_start to teleport), the first time bin of each trial IS the trial start. No additional temporal shifting is needed.

ii. N/A — alignment is implicit in the trial splitting.

iii. The alignment event is the trial start, which corresponds to the first frame of each trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Data is kept at the native per-plane imaging rate. The time bin size is computed as `nplanes / rate * 1000` ms (~64.5 ms). For multi-plane sessions (2 planes at ~31 Hz scanner rate), the per-plane rate is ~15.5 Hz, giving the same ~64.5 ms bin size.

ii.
```python
neural_time_bin_size = nplanes / plane_data.rate * 1000  # ms
assert np.isclose(neural_time_bin_size, time_bin_size, rtol=1e-2)
```

iii. The stored `rate` on the NWB series is the scanner rate; for two-plane sessions it is twice the per-plane sampling rate. The code accounts for this by dividing by `nplanes`.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` associated with the `trial number` behavioral time series.

ii.
```python
timestamps = behavior_data['trial number'].timestamps[:]
```

iii. All behavioral time series share the same timestamps, so any could be used. The agent chose `trial number`.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of the trial is subtracted from all timestamps in the trial.

ii.
```python
timestamps_curr = timestamps[idx]
input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. Standard approach to compute time relative to trial onset.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame-level time indexing, so no additional alignment is needed. A check verifies that neural and behavioral data have the same number of timepoints, cropping to the minimum if they differ.

ii.
```python
if T != len(timestamps):
    T = min(T, len(timestamps))
    fluorescence = [f[:T] for f in fluorescence]
    timestamps = timestamps[:T]
```

iii. The neural sampling rate matches the behavioral sampling rate (verified by assertion).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral time series.

ii.
```python
vr_environment = behavior_data['environment'].data[:][:T]
vr_environment_curr = vr_environment[idx].astype(int)
input_curr[1] = vr_environment_curr
```

iii. The environment variable is constant within each trial and takes values 0 or 1, matching ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The raw environment values are cast to int. No other processing is applied.

ii.
```python
input_curr[1] = vr_environment_curr  # already cast to int above
```

iii. The raw values are already binary (0 or 1).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session loop counter (sequential index 0, 1, 2, ...), NOT the `trial number` variable from the NWB file.

ii.
```python
for trial in range(ntrials):
    ...
    input_curr[2] = trial
```

iii. The agent noted that the stored `trial number` variable did not agree with `trial_start` boundaries.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing — the loop index is assigned directly. The value is constant across all timepoints within a trial.

ii.
```python
input_curr[2] = trial
```

iii. Simple sequential index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavioral time series. Reward event timestamps are mapped to behavior frame indices using `searchsorted`.

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. The `Reward` time series has its own timestamps separate from the behavior sampling rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check whether any reward event occurred in the previous trial. For the first trial, set to 0. The value is constant across timepoints.

ii.
```python
if trial == 0:
    input_curr[3,:] = 0
else:
    prev_reward_idx = np.arange(trial_start_idx[trial-1], trial_end_idx[trial-1], dtype=int)
    isprevreward = np.any(isreward[prev_reward_idx])
    input_curr[3,:] = int(isprevreward)
```

iii. Binary outcome: 1 if any reward in previous trial, 0 otherwise. First trial defaults to 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone label for each trial. The reward zone label is determined via a Viterbi algorithm applied during the survey step, using the `reward_zone` signal and `position` to infer which zone (A, B, or C) was active on each trial.

ii.
```python
# In survey, identify reward zone positions:
reward_zone_trial = reward_zone[idx] > 0
info['reward_zone_positions'][trial] = [np.min(pos_trial[reward_zone_trial]),
                                         np.max(pos_trial[reward_zone_trial])]

# Viterbi assignment:
reward_zone_dict = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}
# ... Viterbi with Gaussian emission on distance to zone + transition probabilities
```

iii. The agent used the Viterbi algorithm to handle noisy reward zone observations and omission trials where the zone signal is absent.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, compute the signed distance from position to the nearest edge of the assigned reward zone. Distance is 0 when inside the zone, negative when before it, positive when past it.

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

iii. Matches the paper's concept of distance relative to the reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using `np.digitize` with edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`. The `1e-6` separates exactly 0 from slightly positive values.

ii.
```python
distance_to_reward_zone_bins = [-np.inf, -50, -10, 0, 1e-6, 10, 50, np.inf]
output_curr[0] = np.digitize(distance_to_reward_zone, distance_to_reward_zone_bins) - 1
```

iii. Bin edges match the instructions. The `-1` makes digitize 0-indexed.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data share the same time indices within each trial. No additional alignment needed.

ii.
```python
idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
spike_data_curr = spike_data[idx]
position_curr = position[idx]
```

iii. Same frame-level indexing ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
position = behavior_data['position'].data[:][:T]
position_curr = position[idx]
```

iii. The `position` variable records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing. The raw position values are used directly.

ii.
```python
position_curr = position[idx]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. The track is 450 cm long, and the five 90 cm bins span it.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal bins using `np.digitize` with edges `[-inf, 90, 180, 270, 360, inf]`.

ii.
```python
position_bins = [-np.inf, 90, 180, 270, 360, np.inf]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. Five 90 cm bins spanning the 450 cm track, as specified in the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data — no additional alignment needed.

ii. Same indexing: `position_curr = position[idx]`

iii. Verified by timestamp assertions.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

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

iii. The instructions specify binary output (no/yes). Raw lick values can be > 1, so thresholding at > 0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data — no additional alignment needed.

ii. Same indexing: `licks_curr = licks[idx]`

iii. Same frame-level indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavioral time series and `position`, processed through a Viterbi algorithm during the survey step (see 7-a).

ii. See 7-a.

iii. See 7-a.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The Viterbi-assigned zone label (A, B, or C) is converted to an index (0, 1, or 2). The value is constant across all timepoints in the trial.

ii.
```python
reward_zone_curr = reward_zone[trial].item()
output_curr[4] = ['A','B','C'].index(reward_zone_curr)
```

iii. Direct mapping from zone label to index.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavioral time series timestamps.

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. Reward events have their own timestamps separate from the behavior sampling rate.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the output is 1 if any reward event occurred within the trial's time window, 0 otherwise. The value is constant across all timepoints in the trial.

ii.
```python
output_curr[5] = np.any(isreward[idx]).astype(int)
```

iii. Binary per-trial outcome matching the instructions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: Data is cropped to the minimum of the two with a warning.
- **Short trials**: An attempted filter for < 50 timepoints exists but has a bug (`idx.sum()` instead of `len(idx)`).
- **Missing reward zone data**: Trials where `reward_zone` is never active get NaN positions; the Viterbi algorithm handles these via uniform emission.
- **Reward timestamp alignment**: An assertion verifies reward times align within half a time bin.
- **Deconvolved events NaN check**: An assertion ensures no NaN values exist inside trial boundaries.

ii.
```python
if T != len(timestamps):
    T = min(T, len(timestamps))
    ...

# Viterbi handles NaN reward zone positions with uniform emission
log_emission = np.zeros((n_trials, n_states))  # uniform for NaN trials
```

iii. Defensive checks discovered during data exploration.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **dF/F and deconvolution** via the paper's `dff()` function — iterates over trials, applies Gaussian smoothing, min/max filters, and OASIS deconvolution per cell.
2. **Loading NWB files** twice — once during survey, once during conversion.
3. **The survey step** — loads every NWB file to gather statistics and compute reward zone labels.

ii. N/A

iii. The dF/F computation is computationally intensive due to per-cell, per-trial operations including Gaussian smoothing and OASIS deconvolution.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session_data` (iterates over each trial sequentially for slicing and binning). The Viterbi inner loop iterates over states for each trial. The `is_putative_interneuron` function loops over each cell to compute speed correlations.

ii.
```python
for c in range(dff.shape[0]):
    speed_corr[c] = np.corrcoef(dff[c, nanmask], speed[nanmask])[0, 1]
```

iii. Variable trial lengths make vectorization of the per-trial loop awkward. The speed correlation loop could be replaced with matrix correlation.

## 13-c. What processing does the code repeat multiple times?

i. The survey step and conversion step both load every NWB file and read similar data arrays. The survey reads neural data to count neurons and check rates, then the conversion reads the raw fluorescence and neuropil again. The `trial_start` / `teleport` parsing is done in both passes.

ii. N/A

iii. The survey must run first to compute reward zone labels via Viterbi, which the conversion step needs. The agent noted that combining them would require too much RAM.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `dff()` function computes both dF/F and deconvolved events, but only the events are used as the final neural data — the dF/F array is used only for interneuron detection and then discarded. The survey step generates and saves plots (`reward_zone_positions.png/pdf/svg`) that are not needed for the conversion. The survey also computes various summary statistics (speed ranges, unique values) that are printed but not used in conversion.

ii.
```python
dff_curr, events_curr = dff(...)  # dff_curr used only for interneuron check
isint = is_putative_interneuron(dff_curr, speed)
spike_data.append(events_curr[~isint].T)  # only events kept
```

iii. The dF/F computation is necessary for interneuron filtering but is otherwise discarded. The survey plots aid exploration but are not required.
