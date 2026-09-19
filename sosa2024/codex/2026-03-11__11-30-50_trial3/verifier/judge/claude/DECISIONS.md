# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from all NWB files found under `data/sub-*/`. It discovers subject directories matching the `sub-<id>` pattern, finds all `.nwb` files within each, and opens them using `pynwb.NWBHDF5IO`. Neural data is read from `processing/ophys/Fluorescence` and `processing/ophys/Neuropil` (raw suite2p traces), and behavioral data from `processing/behavior/BehavioralTimeSeries`. A survey step runs first to gather statistics and reward zone assignments across all sessions, then the conversion step re-reads each file.

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

Loading an NWB file:
```python
with pynwb.NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
```

iii. The code finds all `sub-` directories and all `.nwb` files within them. The number of subjects (11) and sessions (152) matches the paper's description. The agent documented this in CONVERSION_NOTES.md.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories of `data/` matching the `sub-<id>` pattern. The subject ID is extracted from the directory name.

ii.
```python
def get_subjects(datadir=DATADIR):
    subjects = [d.split('-')[1] for d in os.listdir(datadir) if d.startswith('sub-')]
    return subjects
...
data['subjects'] = get_subjects(datadir=args.datadir)
```

iii. The directory structure `sub-<id>` directly encodes the subject identity. The 11 subjects found match the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session numbers are parsed from filenames matching `sub-<subject>_ses-<NN>_behavior+ophys.nwb`.

ii.
```python
def get_sessions(subject, datadir=DATADIR):
    subject_dir = os.path.join(datadir, f"sub-{subject}")
    nwb_files = [f for f in os.listdir(subject_dir) if f.endswith('.nwb')]
    sessions = [int(f.split('_')[1].split('-')[1]) for f in nwb_files]
    return sessions
```

iii. The file naming convention makes sessions explicit. 152 total sessions matches expectations (10 mice x 14 + m11 x 12).

## 1-d. How are the data split into trials?

i. Trial boundaries are identified using the `trial_start` and `teleport` behavior time series. Trial starts are found where `trial_start > 0`. Trial ends are found at the onset of `teleport` (where `teleport` transitions from <=0 to >0). Data between these indices constitutes a trial.

ii.
```python
def get_trial_starts_ends(trial_start, teleport):
    trial_start_idx = np.nonzero(trial_start)[0]
    is_teleport_start = (teleport[1:] > 0) & (teleport[:-1] <= 0)
    trial_end_idx = np.nonzero(is_teleport_start)[0] + 1
    assert len(trial_start_idx) == len(trial_end_idx)
    return trial_start_idx, trial_end_idx
```

iii. The agent investigated using the `trial_number` variable but found it inconsistent with `trial_start`. Using `trial_start` and `teleport` onset edges matches the reference code's trial segmentation logic from `trial_start_inds` and `teleport_inds`.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 50 timepoints are removed. There is NO lick-error filtering applied.

ii.
```python
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    if idx.sum() < min_ntimepoints:
        print(f"Skipping trial {trial}...")
        istrial[trial] = False
        continue
```

iii. The agent mentioned short trials being removed. The reference code also filters lick-artifact trials (>35% of frames with lick count >2), but the AI does not implement this filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p `Fluorescence` (F) and `Neuropil` (Fneu) traces stored in the NWB file. The agent explicitly notes that the NWB `Deconvolved` field is NOT used, as it is suite2p's own deconvolution rather than the paper's processing.

ii.
```python
fluorescence_data = nwbfile.processing["ophys"]["Fluorescence"]
neuropil_data = nwbfile.processing["ophys"]["Neuropil"]
for plane_name, plane_data in fluorescence_data.roi_response_series.items():
    fluorescence_curr = plane_data.data[:]
    neuropil_curr = neuropil_data.roi_response_series[plane_name].data[:]
```

iii. The agent identified that the paper computes its own dF/F and deconvolution rather than using a stored deconvolved signal, and copied the paper's `preprocessing.dff()` function.

## 2-b. How is the `neural` data processed?

i. The agent recomputes dF/F and deconvolved events using a copy of the paper's `dff()` function. Per plane: (1) restrict to lap samples (or lap+teleport based on `teleport_metadata.py`), (2) subtract 0.7*Fneu, (3) add back trial mean neuropil, (4) compute maximin baseline (Gaussian smoothing sigma=15, 300-sample running min then max), (5) compute dF/F = (F - baseline)/|baseline|, (6) smooth with 2-sample Gaussian, (7) deconvolve with OASIS (tau=0.7, frame_rate/n_planes). Cells from multiple planes are pooled.

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

iii. The agent copied the paper's `dff()` function and the `teleport_sessions` metadata. This matches the Methods description of baseline computation, neuropil subtraction, and OASIS deconvolution.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) cells are restricted to `iscell` ROIs (suite2p manual curation), (2) putative interneurons are dropped (cells whose dF/F correlates with running speed at r > 0.5).

ii.
```python
iscell = np.array(roi_data['iscell'].to_list())[:,0] == 1
fluorescence.append(fluorescence_curr[:,iscell])
neuropil.append(neuropil_curr[:,iscell])
...
isint = is_putative_interneuron(dff_curr, speed)
spike_data.append(events_curr[~isint].T)
```

iii. Both filters match the paper's Methods description: manual curation and putative interneuron exclusion at r > 0.5.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. Since neural and behavioral data share the same time indices, splitting into trials using `trial_start_idx` and `trial_end_idx` inherently aligns to trial start. No additional temporal shifting is needed.

ii. N/A (alignment is implicit in the trial segmentation)

iii. The instructions specify alignment to trial start, which is the natural alignment from the trial boundary detection.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The data is kept at the native sampling rate (~15.5 Hz, ~64.5 ms per frame). The time bin size is computed as `nplanes / rate * 1000` ms and verified to be consistent across sessions.

ii.
```python
neural_time_bin_size = nplanes/plane_data.rate*1000 # ms
...
assert np.isclose(neural_time_bin_size, time_bin_size, rtol=1e-2)
```

iii. The native sampling rate matches the paper's description of ~15.5 Hz frame-aligned sampling.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the behavior `timestamps` array (from the `trial number` time series, though all behavior series share the same timestamps).

ii.
```python
timestamps = behavior_data['trial number'].timestamps[:]
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    timestamps_curr = timestamps[idx]
    input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. Behavior timestamps are consistent with neural data rates; any behavior series' timestamps would give the same result.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in the trial is subtracted from all timestamps in that trial to get time relative to trial start.

ii.
```python
input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. Straightforward computation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data are already on the same time grid (verified by length matching and assertions). If the number of neural and behavioral timepoints differ, data is cropped to the minimum.

ii.
```python
if T != len(timestamps):
    print(f"Warning: number of neural time bins ({T}) does not match...")
    T = min(T, len(timestamps))
```

iii. The shared frame-aligned sampling means no additional alignment is needed.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
vr_environment = behavior_data['environment'].data[:][:T]
...
vr_environment_curr = vr_environment[idx].astype(int)
input_curr[1] = vr_environment_curr
```

iii. The environment variable is constant within a trial and binary (0 or 1), matching ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The raw environment values are cast to int and used directly. No modal/median computation is applied; the values are used as time-varying (though they happen to be constant within a trial).

ii.
```python
input_curr[1] = vr_environment_curr
```

iii. Since environment is constant within a trial, this produces the same result as taking a per-trial modal value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session loop counter (0-indexed), NOT the stored `trial number` variable from the NWB file.

ii.
```python
for trial in range(ntrials):
    ...
    input_curr[2] = trial
```

iii. The agent noted that the stored `trial number` variable did not agree with `trial_start`, so the loop index was used instead.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is assigned as a constant across all timepoints in the trial.

ii.
```python
input_curr[2] = trial
```

iii. No additional processing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series. Reward event timestamps are mapped to behavior timepoints using `np.searchsorted`.

ii.
```python
reward_amounts = behavior_data['Reward'].data[:][:T]
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. The `Reward` series has its own timestamps (sparse events), so alignment to the behavior grid is needed.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check whether any reward event occurred in the *previous* trial's time range. For the first trial, set to 0.

ii.
```python
if trial == 0:
    input_curr[3,:] = 0
else:
    prev_reward_idx = np.arange(trial_start_idx[trial-1], trial_end_idx[trial-1], dtype=int)
    isprevreward = np.any(isreward[prev_reward_idx])
    input_curr[3,:] = int(isprevreward)
```

iii. Binary encoding: omitted=0, rewarded=1, as specified in instructions.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` behavior time series and the reward zone location. The reward zone for each trial is determined using a Viterbi algorithm applied to the `reward_zone` behavior variable and position data. The Viterbi approach assigns trials to zones A/B/C based on where the mouse is when `reward_zone > 0`, with a transition penalty to discourage frequent zone switches.

ii.
```python
reward_zone_dict = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}
# In survey/segment_reward_positions:
zone_ranges = np.array([reward_zone_dict[z] for z in zone_names])
dist = np.maximum(zone_ranges[None, :, 0] - startpoints[valid, None],
                  startpoints[valid, None] - zone_ranges[None, :, 1])
dist = np.maximum(dist, 0)
log_emission[valid] = -0.5 * (dist / sigma) ** 2
```

iii. The agent used position during `reward_zone > 0` events to infer zone identity, combined with a Viterbi HMM to ensure stable assignments. This differs from the reference approach of parsing the scene name from the NWB identifier.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed from the animal's position to the nearest edge of the assigned reward zone. Negative when before the zone, zero when inside, positive when past.

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

iii. The distance computation logic is the same as the reference: signed distance to nearest edge of the zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized using `np.digitize` with bin edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`, then subtract 1 for 0-indexing. The `1e-6` boundary separates exactly 0 (inside zone boundary) from slightly positive.

ii.
```python
distance_to_reward_zone_bins = [-np.inf, -50, -10, 0, 1e-6, 10, 50, np.inf]
output_curr[0] = np.digitize(distance_to_reward_zone, distance_to_reward_zone_bins) - 1
```

iii. The bin structure matches the instructions' 7 categories. The use of `1e-6` instead of `0` for the boundary between "0 cm" and ">0 cm to +10 cm" is a reasonable way to handle the exact-zero case with `np.digitize`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Both position and neural data use the same time indices within each trial, so no additional alignment is needed.

ii.
```python
idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
spike_data_curr = spike_data[idx]
position_curr = position[idx]
```

iii. Verified by timestamp assertions.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = behavior_data['position'].data[:][:T]
position_curr = position[idx]
```

iii. The position variable directly records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting per-trial slices and discretizing.

ii.
```python
position_curr = position[idx]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `np.digitize` with edges `[-inf, 90, 180, 270, 360, inf]`, then subtract 1 for 0-indexing. Each bin spans 90 cm of the 450 cm track.

ii.
```python
position_bins = [-np.inf,90,180,270,360,np.inf]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. Five equal bins spanning 450 cm as specified in the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii. Same `idx` array used for both neural and position data.

iii. Verified by timestamp assertions.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
licks = behavior_data['lick'].data[:][:T]
licks_curr = licks[idx]
```

iii. The lick variable records lick counts at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
output_curr[3] = (licks_curr > 0).astype(int)
```

iii. The instructions specify binary output (no/yes). Raw lick values can exceed 1 (cumulative counts), so thresholding at >0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii. Same `idx` array.

iii. Verified by timestamp assertions.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series and `position` data, via the Viterbi-based zone assignment described in 7-a. The `reward_zone` variable indicates when the mouse is in a reward zone, and the position at those times is used to determine which zone (A, B, or C).

ii. See 7-a for the Viterbi assignment code.

```python
reward_zone = info['reward_zone_labels'][subject][trial_offset:trial_offset+ntrials]
...
output_curr[4] = ['A','B','C'].index(reward_zone_curr)
```

iii. The Viterbi approach assigns zone labels based on observed reward-zone positions, rather than parsing scene names from metadata.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The Viterbi algorithm assigns each trial to zone A, B, or C based on the position of the mouse when `reward_zone > 0`. Zone labels are encoded as integers (A=0, B=1, C=2) and repeated across all timepoints in the trial.

ii.
```python
output_curr[4] = ['A','B','C'].index(reward_zone_curr)
```

iii. See 7-a for the full Viterbi-based assignment logic.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
...
output_curr[5] = np.any(isreward[idx]).astype(int)
```

iii. The Reward time series records sparse reward delivery events.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to behavior timepoints using `searchsorted`. For each trial, the output is 1 if any reward event occurred within the trial, 0 otherwise. The value is constant across all timepoints.

ii.
```python
output_curr[5] = np.any(isreward[idx]).astype(int)
```

iii. An assertion checks alignment error is within half a time bin. Binary per-trial output.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: Cropped to the minimum of the two with a warning.
- **Short trials**: Trials with <50 timepoints skipped.
- **Missing reward zone data**: Trials with no `reward_zone` activity get NaN positions; the Viterbi algorithm handles these via uniform emission.
- **Reward timestamp alignment**: Assertion verifies alignment within half a time bin.
- No lick-error filtering or invalid-frame masking is applied.

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
1. Loading NWB files (I/O bound, using `pynwb`)
2. The survey step (loads every NWB file to gather statistics)
3. The dF/F computation and OASIS deconvolution (compute-intensive per session)
4. Full conversion (re-reads all NWB files)
5. Saving the pickle file

ii. N/A

iii. NWB files are large. The dF/F recomputation adds significant processing time compared to using pre-computed deconvolved data.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session_data` iterates over each trial sequentially. Operations like `np.digitize` and `compute_distance_to_reward_zone` could be applied to full session arrays before splitting into trials. The Viterbi inner loop iterates over states and could be vectorized with matrix operations. The `is_putative_interneuron` function loops over cells for correlation computation.

ii. N/A

iii. Variable trial lengths make vectorization awkward for the per-trial loop.

## 13-c. What processing does the code repeat multiple times?

i. The survey step and conversion step both load every NWB file and read the same data arrays (neural, behavior). The survey reads data to count neurons and check rates, then the conversion reads it all again. The `reward_zone_positions` are computed in the survey and used in conversion, but position, speed, lick etc. are read twice.

ii. N/A

iii. The survey is needed before conversion to compute Viterbi zone assignments. However, the double loading of NWB files is redundant.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The dF/F computation produces both `dff` and `events` (deconvolved), but only `events` is kept for the neural data. The `dff` output is only used transiently for interneuron detection and then discarded. The survey step also computes many statistics (speed ranges, unique values, etc.) that are only used for debugging/printing and not for conversion. The `show_processing` plotting code recomputes some values (distance to reward zone) that were already computed in the main conversion loop.

ii. N/A

iii. The dF/F itself is not stored in the output, only the deconvolved events derived from it.
