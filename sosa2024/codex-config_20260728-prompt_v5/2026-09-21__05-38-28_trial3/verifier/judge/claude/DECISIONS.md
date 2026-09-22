# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from all NWB files in the `/app/data` directory. Subject directories matching `sub-*` are discovered, and all `.nwb` files within each are treated as sessions. Data are loaded using `pynwb` (for the conversion step) and the `survey()` function pre-scans all sessions first to gather statistics and reward zone positions. The actual conversion in `convert_session_data()` reads neural data from `Fluorescence` and `Neuropil` (not `Deconvolved`), and behavioral data from `BehavioralTimeSeries`.

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

iii. The AI verified that the directory structure contains 11 subjects (matching the paper) with `sub-<id>` pattern, and that each contains parseable NWB files. The survey step confirms data coverage before conversion.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories of `data` matching the `sub-*` pattern. The subject ID is extracted by splitting on `-`.

ii.
```python
def get_subjects(datadir=DATADIR):
    subjects = [d.split('-')[1] for d in os.listdir(datadir) if d.startswith('sub-')]
    return subjects
...
data['subjects'] = get_subjects(datadir=args.datadir)
```

iii. The number of subject directories (11) matches the paper's description of 11 switch-task mice.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are identified by parsing the session number from the NWB filename pattern `sub-<subject>_ses-<NN>_behavior+ophys.nwb`.

ii.
```python
def get_sessions(subject, datadir=DATADIR):
    subject_dir = os.path.join(datadir, f"sub-{subject}")
    nwb_files = [f for f in os.listdir(subject_dir) if f.endswith('.nwb')]
    sessions = [int(f.split('_')[1].split('-')[1]) for f in nwb_files]
    return sessions
```

iii. The naming convention includes a parseable `ses-<number>` component. 152 total sessions were found, consistent with 14 days x 11 mice minus 2 missing m11 days.

## 1-d. How are the data split into trials?

i. Trial boundaries are defined by the `trial_start` and `teleport` behavior time series. Trial starts are identified as nonzero values in `trial_start`, and trial ends are identified as rising edges of `teleport` (transitions from <=0 to >0). The code asserts equal numbers of starts and ends.

ii.
```python
def get_trial_starts_ends(trial_start, teleport):
    trial_start_idx = np.nonzero(trial_start)[0]
    is_teleport_start = (teleport[1:] > 0) & (teleport[:-1] <= 0)
    trial_end_idx = np.nonzero(is_teleport_start)[0] + 1
    assert len(trial_start_idx) == len(trial_end_idx)
    return trial_start_idx, trial_end_idx
```

iii. The AI investigated using the `trial_number` variable but found it did not agree with `trial_start`. The `trial_start`/`teleport` approach matches the paper's definition of trials as on-track laps. The AI noted that `sub-m11_ses-03` has an extra `trial_number` label that is a trailing post-teleport tunnel fragment.

## 1-e. How are trials filtered based on quality controls?

i. The code includes a minimum-timepoints filter (`min_ntimepoints=50`), but the implementation contains a bug: it checks `idx.sum()` (sum of index values) instead of `len(idx)` (number of timepoints). Since `idx` contains absolute frame indices that sum to large values, this filter is effectively a no-op for most trials.

ii.
```python
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    if idx.sum() < min_ntimepoints:  # BUG: should be len(idx)
        print(f"Skipping trial {trial}...")
        istrial[trial] = False
        continue
```

iii. The AI's CONVERSION_NOTES mention plans for lick QC filtering (matching the reference paper's approach of removing trials with >35% of samples having lick count >2), and claim 69 trials were excluded by lick QC. However, the final code does not implement lick QC. The discrepancy between notes and code suggests the notes may reflect an earlier version.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` (F) and `Neuropil` (Fneu) time series from the NWB ophys processing module — NOT the pre-computed `Deconvolved` field. The AI recomputes dF/F and deconvolution using the paper's own `preprocessing.dff` function.

ii.
```python
fluorescence_data = nwbfile.processing["ophys"]["Fluorescence"]
neuropil_data = nwbfile.processing["ophys"]["Neuropil"]
for plane_name, plane_data in fluorescence_data.roi_response_series.items():
    fluorescence_curr = plane_data.data[:]
    neuropil_curr = neuropil_data.roi_response_series[plane_name].data[:]
```

iii. The AI determined that the NWB `Deconvolved` field is suite2p's own deconvolution of raw fluorescence, not the paper's custom processing. The paper computes its own deconvolved events via `preprocessing.dff(F, Fneu, ..., deconvolve=True)`.

## 2-b. How is the `neural` data processed?

i. The AI copied the paper's `dff()` function and applies the full preprocessing pipeline per imaging plane: subtract 0.7 * neuropil, add back trial-mean neuropil for true dF/F, compute maximin baseline (Gaussian smooth sigma=15, then 300-sample running min/max), form dF/F = (F - baseline)/|baseline|, smooth with 2-sample Gaussian, deconvolve with OASIS (tau=0.7, frame_rate/n_planes). Cells from multiple planes are pooled. The code also accounts for whether the laser was blanked between trials using `teleport_sessions` metadata.

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

iii. The AI matched the paper's Methods description of baseline fluorescence calculation, dF/F computation, smoothing, and deconvolution. Parameters (neu_coef=0.7, tau=0.7, maximin baseline) were traced to the reference code and notebooks.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) Only cells with `iscell[:,0] == 1` (Suite2p manual curation) are kept. (2) Putative interneurons are dropped — any cell whose dF/F correlates with running speed at r > 0.5.

ii.
```python
iscell = np.array(roi_data['iscell'].to_list())[:,0] == 1
fluorescence.append(fluorescence_curr[:,iscell])
neuropil.append(neuropil_curr[:,iscell])
...
isint = is_putative_interneuron(dff_curr, speed)
spike_data.append(events_curr[~isint].T)
```

iii. `iscell` is the manual curation described in the Methods. The interneuron filter matches the paper's description: "a Pearson correlation of >0.5" with running speed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start is implicit — the data are split at trial boundaries, so the first timepoint of each trial's neural data corresponds to the trial start. No additional temporal shifting is needed.

ii. N/A (alignment is inherent in the trial splitting)

iii. Since neural and behavioral data share the same frame-level sampling and the trial boundaries are defined from behavioral markers, the neural data is automatically aligned to trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data is kept at the native imaging rate (~15.5 Hz, or ~64.5 ms per bin). No rebinning is applied. For multi-plane sessions (m17, m18), the per-plane rate is used (scanner rate / n_planes).

ii.
```python
neural_time_bin_size = nplanes/plane_data.rate*1000 # ms
...
assert np.isclose(neural_time_bin_size, time_bin_size, rtol=1e-2)
```

iii. The time bin size is validated to be consistent across sessions. Behavior timestamps confirm the same effective rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the behavior `timestamps` array (specifically from `trial number` time series timestamps, which are the same across all behavior variables).

ii.
```python
timestamps = behavior_data['trial number'].timestamps[:]
for trial in range(ntrials):
    idx = np.arange(trial_start_idx[trial], trial_end_idx[trial], dtype=int)
    timestamps_curr = timestamps[idx]
    input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. Behavior timestamps are consistent with the constant neural data rate; any behavior time series could have been used since they all share the same timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of the trial is subtracted from all timestamps within the trial, yielding elapsed seconds from trial start.

ii.
```python
input_curr[0] = timestamps_curr - timestamps_curr[0]
```

iii. Direct subtraction of the trial-start timestamp.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame-level time indices, so they are inherently aligned. The code checks that the number of neural and behavioral timepoints match, and crops to the minimum if there is a mismatch.

ii.
```python
if T != len(timestamps):
    print(f"Warning: number of neural time bins ({T}) does not match...")
    T = min(T, len(timestamps))
```

iii. Data rates for neural and behavior are the same, and all offsets are 0.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
vr_environment = behavior_data['environment'].data[:][:T]
...
vr_environment_curr = vr_environment[idx].astype(int)
input_curr[1] = vr_environment_curr
```

iii. The environment variable is constant within each trial and takes values 0 or 1, matching the ENV1/ENV2 description.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. No processing beyond extracting the per-trial slice and casting to int. The variable is stored as time-varying (repeating the per-trial value across timepoints).

ii.
```python
input_curr[1] = vr_environment_curr
```

iii. N/A

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop counter over trials within a session, NOT from the NWB `trial number` field.

ii.
```python
for trial in range(ntrials):
    ...
    input_curr[2] = trial
```

iii. The AI chose not to use the stored `trial number` variable because it did not agree with the `trial_start` variable. Instead, the 0-based loop index is used.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing — the loop index is directly assigned. The value is constant across all timepoints within a trial.

ii.
```python
input_curr[2] = trial
```

iii. Simple sequential indexing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series, which has its own sparse timestamps. Reward event timestamps are mapped to behavior timepoints using `searchsorted`, creating a framewise `isreward` binary array.

ii.
```python
reward_amounts = behavior_data['Reward'].data[:][:T]
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
```

iii. The `Reward` time series has its own timestamps (not aligned to the behavior sampling rate), so `searchsorted` is used. An assertion checks alignment error is within half a time bin.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the code checks whether any reward event occurred within the previous trial's time range. For the first trial, the value is 0. The result is constant across all timepoints in the trial.

ii.
```python
if trial == 0:
    input_curr[3,:] = 0
else:
    prev_reward_idx = np.arange(trial_start_idx[trial-1], trial_end_idx[trial-1], dtype=int)
    isprevreward = np.any(isreward[prev_reward_idx])
    input_curr[3,:] = int(isprevreward)
```

iii. The instructions specify binary (omitted=0, rewarded=1). The code checks if any reward event fell within the previous trial's boundaries.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` behavior time series and the reward zone location for the current trial. Reward zone locations are determined by a Viterbi algorithm applied to the `reward_zone` behavior time series (which indicates reward zone occupancy) and `position`, using the reward zone coordinate dictionary `{'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}`.

ii.
```python
# In survey step:
reward_zone_trial = reward_zone[idx] > 0
info['reward_zone_positions'][trial] = [np.min(pos_trial[reward_zone_trial]),
                                         np.max(pos_trial[reward_zone_trial])]
# Viterbi assignment in segment_reward_positions():
zone_ranges = np.array([reward_zone_dict[z] for z in zone_names])
dist = np.maximum(zone_ranges[None, :, 0] - startpoints[valid, None],
                  startpoints[valid, None] - zone_ranges[None, :, 1])
dist = np.maximum(dist, 0)
log_emission[valid] = -0.5 * (dist / sigma) ** 2
```

iii. The AI used the Viterbi algorithm because the `reward_zone` NWB variable indicates occupancy, not zone identity. The Viterbi approach assigns zone labels using Gaussian emission on distance to known zone coordinates and a transition model that discourages frequent switching. The reference approach would be to derive zone labels from the session scene metadata.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the signed distance from the animal's position to the nearest edge of the assigned reward zone is computed. Distance is 0 inside the zone, negative before the zone, positive after. The continuous distance is then discretized into 7 bins.

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

iii. This matches the paper's concept of distance relative to the reward zone using the same zone coordinate definitions.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using `np.digitize` with edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`. The `1e-6` boundary separates exactly 0 (inside the zone boundary) from slightly positive.

ii.
```python
distance_to_reward_zone_bins = [-np.inf, -50, -10, 0, 1e-6, 10, 50, np.inf]
output_curr[0] = np.digitize(distance_to_reward_zone, distance_to_reward_zone_bins) - 1
```

iii. The bin edges match the instructions. The `1e-6` handles the 0 cm category (inside zone boundaries). The `-1` corrects np.digitize to 0-indexed.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same time indices within each trial, so no additional alignment is needed.

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
...
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

i. Discretized into 5 bins using `np.digitize` with edges `[-inf, 90, 180, 270, 360, inf]`, corresponding to 90 cm bins spanning the 450 cm track.

ii.
```python
position_bins = [-np.inf, 90, 180, 270, 360, np.inf]
output_curr[1] = np.digitize(position_curr, position_bins) - 1
```

iii. Five 90 cm bins over the 0-450 cm track, as specified in the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data — no additional alignment needed.

ii. Same indexing: `position_curr = position[idx]`

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

iii. The instructions specify binary output (no/yes). Raw lick values can be >1, so thresholding at >0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data — no additional alignment needed.

ii. Same indexing: `licks_curr = licks[idx]`

iii. Verified by timestamp assertions.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series and `position` behavior time series, processed through the Viterbi algorithm described in 7-a to assign A/B/C labels per trial.

ii. See 7-a for Viterbi assignment code.

iii. The AI used a data-driven approach rather than parsing the session scene metadata (which is what the reference code does).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The Viterbi algorithm assigns each trial to one of three reward zones (A, B, C) based on the position of the mouse when `reward_zone > 0`. The zone label is then mapped to 0/1/2.

ii.
```python
output_curr[4] = ['A','B','C'].index(reward_zone_curr)
```

iii. See 7-a for full Viterbi details.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series, which records reward delivery events with their own timestamps.

ii.
```python
reward_times = behavior_data['Reward'].timestamps[:]
reward_time_indices = np.searchsorted(timestamps, reward_times)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
...
output_curr[5] = np.any(isreward[idx]).astype(int)
```

iii. The `Reward` time series records reward delivery events with separate timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward event timestamps are mapped to the nearest behavior timepoint using `searchsorted`. For each trial, the output is 1 if any reward event occurred within the trial, 0 otherwise. The value is constant across all timepoints.

ii.
```python
reward_time_indices = np.searchsorted(timestamps, reward_times)
err = timestamps[reward_time_indices] - reward_times
assert np.all(np.abs(err) <= time_bin_size/2)
isreward = np.zeros(T)
isreward[reward_time_indices] = 1
...
output_curr[5] = np.any(isreward[idx]).astype(int)
```

iii. Alignment error is checked to be within half a time bin. `np.any` converts the per-frame reward indicator to a per-trial binary.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: If the number of neural time bins differs from behavior timestamps, data is cropped to the minimum with a warning.
- **Short trials**: A filter for trials with few timepoints exists but has a bug (`idx.sum()` instead of `len(idx)`), making it largely ineffective.
- **Missing reward zone data**: Trials where `reward_zone` is never active get NaN positions; the Viterbi algorithm handles these via uniform emission.
- **Reward timestamp alignment**: An assertion verifies reward times align within half a time bin.
- **Extra trial labels**: The `sub-m11_ses-03` extra trial label is automatically handled by using `trial_start`/`teleport` pairs rather than `trial number`.

ii.
```python
if T != len(timestamps):
    print(f"Warning: ...")
    T = min(T, len(timestamps))
    spike_data = spike_data[:T]
    timestamps = timestamps[:T]
```

iii. These are defensive checks found during data exploration.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **The survey step**: Loads every NWB file to gather statistics and compute reward zone positions via Viterbi.
2. **The dF/F computation and deconvolution**: For each session, the full `dff()` pipeline (Gaussian smoothing, min/max filtering, OASIS deconvolution) runs on all curated cells.
3. **Loading NWB files with pynwb**: I/O bound, loading large fluorescence/neuropil arrays.

ii. N/A

iii. The survey and conversion steps each load every NWB file, effectively doubling I/O.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session_data` (iterating over trials sequentially) could partially be vectorized — operations like `np.digitize` and `compute_distance_to_reward_zone` could be applied to the full session arrays before splitting. The Viterbi inner loop over states could be vectorized with matrix operations. The `is_putative_interneuron` function loops over each cell for correlation computation.

ii. N/A

iii. Variable trial lengths make full vectorization awkward. The Viterbi loop is small (3 states) so optimization benefit is minimal.

## 13-c. What processing does the code repeat multiple times?

i. The survey step and the conversion step both load every NWB file and read overlapping data arrays. The survey reads neural data (via Deconvolved) to count neurons and check rates, then the conversion reads Fluorescence/Neuropil from the same files. Reward zone positions are computed in the survey and reused in conversion, but other data (position, speed, lick) is read twice.

ii. N/A

iii. The two-pass approach is needed because the Viterbi reward-zone assignment must be computed before conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `survey()` function generates detailed statistics, plots reward zone positions to PNG/PDF/SVG files, and saves a survey pickle — all of which are not part of the final converted data and are not needed for the decoder. The dF/F intermediate is computed but only the deconvolved events are kept.
