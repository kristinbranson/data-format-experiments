# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from the `data` directory using `pynwb`. Subject directories are identified by the `sub-*` pattern. Within each subject directory, all `.nwb` files are discovered. Each NWB file is opened with `pynwb.NWBHDF5IO` and its behavioral and optical physiology processing groups are read. A two-pass approach is used: first a `survey()` pass reads metadata and reward zone positions from all files, then a `convert_session_data()` pass loads and converts each session.

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
```python
with pynwb.NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
```

iii. The AI confirms 11 subject directories match the paper's 11 mice. All NWB files are collected per subject. The `pynwb` library is used following its documentation for loading NWB files.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories of `data/` matching the pattern `sub-<id>`. The subject ID is extracted by splitting on `-`.

ii.
```python
def get_subjects(datadir=DATADIR):
    subjects = [d.split('-')[1] for d in os.listdir(datadir) if d.startswith('sub-')]
    return subjects
```

iii. The number of subject directories (11) matches the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session numbers are parsed from filenames matching `sub-<subject>_ses-<number>_behavior+ophys.nwb`.

ii.
```python
def get_sessions(subject, datadir=DATADIR):
    subject_dir = os.path.join(datadir, f"sub-{subject}")
    nwb_files = [f for f in os.listdir(subject_dir) if f.endswith('.nwb')]
    sessions = [int(f.split('_')[1].split('-')[1]) for f in nwb_files]
    return sessions
```

iii. Each NWB filename contains a parsable `ses-<number>` field identifying the session.

## 1-d. How are the data split into trials?

i. Trial starts are identified by nonzero values of the `trial_start` behavior time series. Trial ends are identified by the rising edge of the `teleport` behavior time series (transition from <=0 to >0). An assertion verifies the number of starts equals the number of ends.

ii.
```python
def get_trial_starts_ends(trial_start, teleport):
    trial_start_idx = np.nonzero(trial_start)[0]
    is_teleport_start = (teleport[1:] > 0) & (teleport[:-1] <= 0)
    trial_end_idx = np.nonzero(is_teleport_start)[0] + 1
    assert len(trial_start_idx) == len(trial_end_idx)
    return trial_start_idx, trial_end_idx
```

iii. The AI investigated using the `trial_number` variable but found it less clean. The `trial_start` + `teleport` approach matches the paper's definition of a trial as a lap between trial start and teleport.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 50 timepoints are skipped and flagged.

ii.
```python
if idx.sum() < min_ntimepoints:
    print(f"Skipping trial {trial} in subject {subject} session {session} because it has only {idx.sum()} < {min_ntimepoints} time points")
    istrial[trial] = False
    continue
```

iii. The AI states that looking at trial lengths revealed some very short trials that warranted removal. The threshold of 50 was chosen as a minimum.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p `Fluorescence` (F) and `Neuropil` (Fneu) traces stored in the NWB's `processing/ophys/Fluorescence` and `processing/ophys/Neuropil` groups. The NWB's `Deconvolved` field is explicitly NOT used.

ii.
```python
fluorescence_data = nwbfile.processing["ophys"]["Fluorescence"]
neuropil_data = nwbfile.processing["ophys"]["Neuropil"]
for plane_name, plane_data in fluorescence_data.roi_response_series.items():
    fluorescence_curr = plane_data.data[:]
    neuropil_curr = neuropil_data.roi_response_series[plane_name].data[:]
```

iii. The paper computes its own neural signal from raw fluorescence via `preprocessing.dff(F, Fneu, ..., deconvolve=True)` producing "events", rather than using suite2p's stored deconvolution.

## 2-b. How is the `neural` data processed?

i. The AI copied the paper's `dff()` function and applied it: neuropil subtraction (0.7 * Fneu), trial-mean neuropil correction, maximin baseline (Gaussian smoothing sigma=15, 300-sample min/max filter = 20s window), dF/F computation, 2-sample Gaussian smoothing, and OASIS deconvolution with tau=0.7. The `keep_teleports` flag is set per session using metadata from `teleport_metadata.py`. Multiple planes are processed separately then concatenated.

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

iii. This matches the Methods description: "baseline fluorescence was calculated within each trial independently using a maximin procedure with a 20 s sliding window... dF/F was then calculated... smoothed with a two-sample Gaussian kernel. The activity rate was extracted by deconvolving dF/F with a canonical calcium kernel using the OASIS algorithm." Parameters match the reference code: `neu_coef=0.7`, `tau=0.7`, `baseline_method='maximin'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters applied: (1) suite2p's manual cell curation via `iscell` flag restricts to curated ROIs, and (2) putative interneurons are removed (cells whose dF/F correlates with running speed at r > 0.5).

ii.
```python
iscell = np.array(roi_data['iscell'].to_list())[:,0] == 1
fluorescence.append(fluorescence_curr[:,iscell])
neuropil.append(neuropil_curr[:,iscell])
...
isint = is_putative_interneuron(dff_curr, speed)
spike_data.append(events_curr[~isint].T)
```

iii. Both filters match the paper's Methods: "iscell" for manual curation and the interneuron filter described as "a Pearson correlation of >0.5" with speed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The data is aligned to trial start, which is inherent in the trial splitting. No additional temporal shifting is needed since trial boundaries already define the start.

ii. N/A (alignment is inherent in trial splitting)

iii. The instructions specify "Temporally align based on start of the trial", which is satisfied by the trial splitting approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data is kept at its native sampling rate. The time bin size is computed as `nplanes / frame_rate * 1000` ms. No rebinning is applied. For single-plane sessions (~15.5 Hz), this is ~64.5 ms; for two-plane sessions the scanner rate is ~31 Hz but the per-plane rate is still ~15.5 Hz.

ii.
```python
neural_time_bin_size = nplanes/plane_data.rate*1000 # ms
assert np.isclose(neural_time_bin_size, time_bin_size, rtol=1e-2)
```

iii. The `rate` stored on the NWB series is the scanner rate; dividing by number of planes gives the true per-plane sampling rate. This is consistent across sessions.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` of the `trial number` behavior time series.

ii.
```python
timestamps = behavior_data['trial number'].timestamps[:]
...
input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. All behavior time series share the same timestamps, so the choice of which one to reference is arbitrary.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of each trial is subtracted from all timestamps in that trial.

ii.
```python
input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. Simple subtraction to get time relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same sampling indices within the NWB file. The code verifies lengths match and crops to the minimum if they differ. An assertion checks behavioral time bin size matches neural time bin size.

ii.
```python
if T != len(timestamps):
    T = min(T, len(timestamps))
    ...
assert np.isclose(behavior_time_bin_size, time_bin_size, rtol=1e-2)
```

iii. Data rates for neural and behavior data were verified to be the same, and timestamp offsets were 0.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
vr_environment = behavior_data['environment'].data[:][:T]
...
input_curr[1] = vr_environment_curr
```

iii. The environment variable was confirmed to be constant within each trial and take values 0 or 1, matching ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. No processing — the raw values (0 or 1) are used directly, cast to int.

ii.
```python
vr_environment_curr = vr_environment[idx].astype(int)
input_curr[1] = vr_environment_curr
```

iii. The raw data already contains binary 0/1 values.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session sequential index (loop counter), not the NWB's stored `trial number` variable.

ii.
```python
for trial in range(ntrials):
    ...
    input_curr[2] = trial
```

iii. The AI chose not to use the stored `trial number` because it did not agree with the `trial_start` variable.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing — just the loop index (0, 1, 2, ...) broadcast across all timepoints in the trial.

ii.
```python
input_curr[2] = trial
```

iii. Sequential index within session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series. Reward event timestamps are mapped to behavior timepoints using `searchsorted`.

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. The `Reward` time series has its own timestamps (not aligned to behavior sampling rate), so `searchsorted` maps them to the nearest behavior timepoint.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check whether any reward event occurred in the previous trial's time range. First trial is set to 0.

ii.
```python
if trial == 0:
    input_curr[3,:] = 0
else:
    prev_reward_idx = np.arange(trial_start_idx[trial-1], trial_end_idx[trial-1], dtype=int)
    isprevreward = np.any(isreward[prev_reward_idx])
    input_curr[3,:] = int(isprevreward)
```

iii. Binary encoding per instructions (omitted=0, rewarded=1). Checks if any reward event fell within the previous trial boundaries.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` and `reward_zone` behavior time series. The reward zone for each trial is determined by a Viterbi algorithm that assigns trials to one of three known zones (A: [80,130], B: [200,250], C: [320,370]) based on the position of the mouse when `reward_zone > 0`. The Viterbi uses Gaussian emission probabilities centered on zone ranges and transition probabilities that favor staying in the same zone.

ii.
```python
reward_zone_dict = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}
# In segment_reward_positions():
dist = np.maximum(zone_ranges[None, :, 0] - startpoints[valid, None],
                  startpoints[valid, None] - zone_ranges[None, :, 1])
dist = np.maximum(dist, 0)
log_emission[valid] = -0.5 * (dist / sigma) ** 2
```

iii. The zone boundaries are from the reference code (`behavior.py`). The Viterbi approach handles noise in the `reward_zone` variable and trials with no reward zone signal.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, compute the signed distance from the animal's position to the nearest edge of the assigned reward zone. Distance is 0 inside the zone, negative before it, positive after it.

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

iii. Matches the paper's concept of distance relative to the reward zone. Distance is to the nearest edge of the zone range, not to a center point.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using `np.digitize` with edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`. The `1e-6` boundary separates exactly 0 (inside zone boundary) from slightly positive.

ii.
```python
distance_to_reward_zone_bins = [-np.inf, -50, -10, 0, 1e-6, 10, 50, np.inf]
output_curr[0] = np.digitize(distance_to_reward_zone, distance_to_reward_zone_bins) - 1
```

iii. Bin edges match the instructions. The `-1` corrects `np.digitize` to be 0-indexed.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as neural data — both use the same `idx` array within each trial.

ii.
```python
idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
spike_data_curr = spike_data[idx]
position_curr = position[idx]
```

iii. Verified by timestamp assertions that neural and behavioral data share the same indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = behavior_data['position'].data[:][:T]
position_curr = position[idx]
```

iii. The `position` variable records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice.

ii.
```python
position_curr = position[idx]
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal-sized bins using `np.digitize` with fixed edges `[-inf, 90, 180, 270, 360, inf]`, corresponding to 90 cm bins spanning the 450 cm track.

ii.
```python
position_bins = [-np.inf, 90, 180, 270, 360, np.inf]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. The instructions specify "5 equal-sized bins spanning the 450 cm track", giving 90 cm per bin. The open first and last edges handle edge cases.

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

i. Derived from the `reward_zone` and `position` behavior time series. The `survey()` function computes reward zone positions for all trials (min/max position when `reward_zone > 0`), then the `segment_reward_positions()` function uses a Viterbi algorithm to assign each trial to one of three canonical zones (A, B, C).

ii.
```python
# In survey_session_data():
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    pos_trial = position[idx]
    reward_zone_trial = reward_zone[idx] > 0
    if not np.any(reward_zone_trial):
        continue
    info['reward_zone_positions'][trial] = [np.min(pos_trial[reward_zone_trial]), np.max(pos_trial[reward_zone_trial])]
```
```python
# Viterbi assignment in segment_reward_positions()
zone_labels = np.array([zone_names[s] for s in states])
```

iii. The paper defines three reward zones at specific positions. The Viterbi approach handles noise and missing data in the `reward_zone` signal.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The Viterbi algorithm assigns each trial to zone A (0), B (1), or C (2) based on reward position observations and transition smoothing. The zone labels are then used during conversion to set the per-trial output.

ii.
```python
output_curr[4] = ['A','B','C'].index(reward_zone_curr)
```

iii. The zone boundaries [80,130], [200,250], [320,370] are from the reference code. The Viterbi encourages stable zone assignments across consecutive trials.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. The `Reward` time series records reward delivery events with their own timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the output is 1 if any reward event occurred within the trial's time range, 0 otherwise. The value is constant across all timepoints in the trial.

ii.
```python
output_curr[5] = np.any(isreward[idx]).astype(int)
```

iii. `np.any` determines if any reward event was delivered during the trial.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Cropped to the minimum of the two with a warning.
- **Short trials**: Trials with <50 timepoints skipped with a warning.
- **Missing reward zone data**: Trials where `reward_zone` is never active get NaN positions; the Viterbi handles these via uniform emission.
- **Reward timestamp alignment**: Assertion verifies reward times align within half a time bin.

ii.
```python
if T != len(timestamps):
    T = min(T, len(timestamps))
    spike_data = spike_data[:T]
    timestamps = timestamps[:T]
...
if idx.sum() < min_ntimepoints:
    istrial[trial] = False
    continue
```

iii. Defensive checks found during data exploration. The neural/behavior mismatch occurs in a small number of sessions.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files (I/O bound)
2. The survey step (loads every NWB file for statistics)
3. The dff/deconvolution computation per session
4. Full conversion (loads every NWB file again)
5. Saving the .pkl file

ii. N/A

iii. NWB files are large; the dff computation involves per-cell Gaussian smoothing, min/max filtering, and OASIS deconvolution.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session_data` iterates over each trial sequentially. Operations like `np.digitize` and `compute_distance_to_reward_zone` could theoretically be applied to full session arrays before splitting. The Viterbi inner loop over states could be vectorized with matrix operations. The `is_putative_interneuron` loop over cells could be vectorized.

ii. N/A

iii. The per-trial loop is natural given variable-length trials. Vectorizing would require padding.

## 13-c. What processing does the code repeat multiple times?

i. The survey step and conversion step both load every NWB file and read the same data arrays. The survey reads neural data to count neurons and check rates, then the conversion reads it again. Other behavior data (position, speed, lick) is also read twice.

ii. N/A

iii. The survey must run first to compute reward zone assignments (via Viterbi) before conversion can proceed.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI states N/A — no unnecessary processing identified.

ii. N/A

iii. N/A
