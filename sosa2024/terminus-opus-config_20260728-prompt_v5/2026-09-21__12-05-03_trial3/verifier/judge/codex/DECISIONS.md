# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every `sub-*` directory under `/app/data`, globs every `.nwb` file inside each subject directory, and processes each file as one session. Within each session it opens the NWB file with `h5py` and reads ophys arrays (`Fluorescence`, `Neuropil`, `iscell`) plus behavioral time series arrays into memory.

ii. 
```python
data_dir = '/app/data'
subjects = sorted([d.replace('sub-', '') for d in os.listdir(data_dir) if d.startswith('sub-')])

all_sessions = []
for subj in subjects:
    subj_dir = os.path.join(data_dir, f'sub-{subj}')
    nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
    for nwb_file in nwb_files:
        ses_str = os.path.basename(nwb_file).split('ses-')[1].split('_')[0]
        ses_num = int(ses_str)
        all_sessions.append((subj, ses_num, nwb_file))
```
```python
with h5py.File(nwb_path, 'r') as f:
    fluor_group = f['processing']['ophys']['Fluorescence']
    ...
    beh = f['processing']['behavior']['BehavioralTimeSeries']
```

iii. In `CONVERSION_NOTES.md`, Step 2 says the dataset consists of 11 subject folders and 152 NWB sessions, and Step 10 says the AI intentionally used “h5py from NWB” rather than the paper code’s higher-level loaders.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directory name. The code strips the `sub-` prefix from each directory name and uses the resulting mouse IDs as subjects.

ii.
```python
subjects = sorted([d.replace('sub-', '') for d in os.listdir(data_dir) if d.startswith('sub-')])
...
if subj not in subjects_seen:
    subjects_seen.append(subj)
subject_idx_list.append(subjects_seen.index(subj))
```

iii. `CONVERSION_NOTES.md` Step 2 explicitly documents “11 subjects: m3, m4, m7, m11-m15, m17-m19,” which is the basis for this split.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session number is parsed from the `ses-XX` part of the filename and carried through the converted metadata.

ii.
```python
for nwb_file in nwb_files:
    ses_str = os.path.basename(nwb_file).split('ses-')[1].split('_')[0]
    ses_num = int(ses_str)
    all_sessions.append((subj, ses_num, nwb_file))
```

iii. `CONVERSION_NOTES.md` Step 2 states “NWB files: `sub-{id}_ses-{num}_behavior+ophys.nwb`” and lists session counts per subject, showing the AI inferred sessions directly from filenames.

## 1-d. How are the data split into trials?

i. Trials are split by taking every frame where `trial_start == 1` as a start index and every frame where `teleport == 1` as an end index. The code truncates to the smaller of the number of starts and ends, then slices every modality with `[start:stop]`.

ii.
```python
tstart_idx = np.where(trial_start_arr == 1)[0]
teleport_idx = np.where(teleport_arr == 1)[0]
n_trials = min(len(tstart_idx), len(teleport_idx))

tstart_idx = tstart_idx[:n_trials]
teleport_idx = teleport_idx[:n_trials]
...
start = tstart_idx[t]
stop = teleport_idx[t]
neural = dff_valid[:, start:stop].astype(np.float32)
```

iii. In `CONVERSION_NOTES.md`, Step 5 lists “Trial boundaries: trial_start to teleport markers,” and trajectory step 45 says the AI believed this matched the reference trial structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI does almost no trial-quality filtering. It skips a session if it has fewer than 2 detected trials, and within a session it only drops trials with fewer than 2 timepoints.

ii.
```python
if n_trials < 2:
    print(f"    Skipping: only {n_trials} trials")
    return None
...
n_tp = stop - start
if n_tp < 2:
    continue
...
if len(neural_trials) < 2:
    print(f"    Skipping: only {len(neural_trials)} valid trials after filtering")
    return None
```

iii. The notes emphasize retaining “all data within trials” for decoder use and explicitly mention not applying the paper’s speed-based filtering. There is no documented additional trial QC beyond these minimum-size checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from raw fluorescence `Fluorescence`, raw neuropil `Neuropil`, and the ROI quality mask `iscell`. Running speed is additionally used later for interneuron exclusion.

ii.
```python
F_planes.append(f['processing']['ophys']['Fluorescence'][pk]['data'][()].T)
Fneu_planes.append(f['processing']['ophys']['Neuropil'][pk]['data'][()].T)
...
iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][()]
...
speed = beh['speed']['data'][()]
```

iii. `CONVERSION_NOTES.md` Step 5 states “F, Fneu, iscell -> neural,” and trajectory steps 40 and 43 explain that the AI rejected the NWB `Deconvolved` array because it viewed it as suite2p output, not the paper’s custom signal.

## 2-b. How is the `neural` data processed?

i. The AI concatenates planes, computes dF/F from raw `F` and `Fneu`, filters cells, and stores the resulting dF/F directly. The dF/F pipeline is: subtract `0.7 * Fneu`, add back each trial’s mean neuropil, smooth with Gaussian sigma 15, apply 300-sample min and max filters for a maximin baseline, compute `(F - baseline)/|baseline|`, then smooth each trial with Gaussian sigma 2. It does **not** deconvolve with OASIS before saving.

ii.
```python
def compute_dff(F, Fneu, trial_start_idx, teleport_idx, neu_coef=0.7):
    f_ = F.astype(np.float64).copy()
    f_neu_ = Fneu.astype(np.float64).copy()
    f_ -= neu_coef * f_neu_
    ...
    f_[:, start:stop] = f_[:, start:stop] + neu_coef * np.nanmean(
        f_neu_[:, start:stop], axis=1, keepdims=True)
    flow[:, start:stop] = nansmooth(trial_data, 15, axis=1)
    flow[:, start:stop] = minimum_filter1d(flow[:, start:stop], 300, axis=-1)
    flow[:, start:stop] = maximum_filter1d(flow[:, start:stop], 300, axis=-1)
    ...
    dff[valid] = (f_[valid] - flow[valid]) / np.abs(flow[valid])
    ...
    dff[:, start:stop] = nansmooth(dff[:, start:stop], 2, axis=1)
    return dff
```
```python
dff = compute_dff(F_raw, Fneu_raw, tstart_idx, teleport_idx, neu_coef=0.7)
...
neural = dff_valid[:, start:stop].astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 says the key decision was “Neural signal: dF/F (not deconvolved),” and trajectory steps 43 and 47 show the AI consciously chose dF/F instead of reproducing the paper’s final deconvolved events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only ROIs with `iscell[:, 0] == 1`, then removes putative interneurons defined by correlation between dF/F and speed greater than 0.5. Sessions with fewer than 2 surviving cells are dropped.

ii.
```python
cell_mask = iscell[:, 0] == 1
...
dff_cells = dff[cell_mask]
is_interneuron = detect_interneurons(dff_cells, speed, threshold=0.5)
...
valid_cell_indices = cell_indices[~is_interneuron]
dff_valid = dff[valid_cell_indices]
...
if n_cells < 2:
    print(f"    Skipping: only {n_cells} valid cells")
    return None
```

iii. `CONVERSION_NOTES.md` Step 3 lists “iscell (manual) + exclude interneurons (corr > 0.5)” as the neuron curation rule, and Step 10 claims this matches the paper’s curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to trial start simply by slicing each trial from `trial_start` to `teleport`; timepoint 0 of each stored trial is the first frame in that slice.

ii.
```python
start = tstart_idx[t]
stop = teleport_idx[t]
neural = dff_valid[:, start:stop].astype(np.float32)
time_from_start = timestamps[start:stop] - timestamps[start]
```

iii. `CONVERSION_NOTES.md` Step 5 identifies “Temporally align based on start of the trial” as the mapping rule, and the code implements no additional offsetting.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native frame resolution and applies no temporal rebinning. It treats the bin size as approximately 64.5 ms and stores that fixed approximate value in metadata.

ii.
```python
dt = np.mean(np.diff(timestamps))  # ~0.0645s
...
'metadata': {
    ...
    'time_bin_size': 64.5,  # ms, approximate
```

iii. `CONVERSION_NOTES.md` Step 2 reports a frame rate of about 15.51 Hz, and Step 5 maps this directly to the decoder without resampling.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavioral timestamps attached to the `position` time series.

ii.
```python
timestamps = beh['position']['timestamps'][()]
...
time_from_start = timestamps[start:stop] - timestamps[start]
```

iii. The notes repeatedly describe the behavior and neural streams as frame-aligned, so the AI treated any behavior timestamp vector as interchangeable and chose the `position` timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first timestamp in the sliced interval is subtracted from every timestamp in that trial.

ii.
```python
time_from_start = timestamps[start:stop] - timestamps[start]
...
inp[0, :] = time_from_start
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly maps this input as `time_from_trial_start = ts - ts[trial_start]`.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by using the exact same `[start:stop]` indices used for neural slicing. Before trial extraction, the code globally truncates all arrays to the minimum neural/behavior length if they differ.

ii.
```python
n_timepoints = min(n_timepoints_neural, n_timepoints_beh)
if n_timepoints_neural != n_timepoints_beh:
    ...
    timestamps = timestamps[:n_timepoints]
```
```python
start = tstart_idx[t]
stop = teleport_idx[t]
neural = dff_valid[:, start:stop].astype(np.float32)
time_from_start = timestamps[start:stop] - timestamps[start]
```

iii. `CONVERSION_NOTES.md` Step 4 says off-by-one mismatches are handled by truncation, and Step 10 lists a “Time input” sanity check based on the raw timestamps.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the raw behavioral `environment` time series.

ii.
```python
environment = beh['environment']['data'][()]
...
env_vals = environment[start:stop]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `environment -> input[1]` and states that `0=ENV1, 1=ENV2`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the AI drops negative environment values, takes the median of the remaining values, converts that to `int`, and broadcasts it across the trial.

ii.
```python
env_vals = environment[start:stop]
env_vals = env_vals[env_vals >= 0]  # exclude -1
if len(env_vals) > 0:
    env_per_trial[t] = int(np.median(env_vals))
...
inp[1, :] = env_per_trial[t]
```

iii. The trajectory shows the AI explored `environment` values early and concluded that the sessions were effectively binary ENV1 vs ENV2, with occasional negative values outside valid data periods.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The converted trial number is not taken from the raw `trial number` field. It is derived from the trial loop index after trial segmentation.

ii.
```python
trial_number = beh['trial number']['data'][()]
...
for t in range(n_trials):
    ...
    inp[2, :] = t  # trial number (0-indexed)
```

iii. `CONVERSION_NOTES.md` Step 5 says the trial number is “0-indexed trial number,” and trajectory step 15 notes the AI distrusted the raw trial-number stream for defining trials.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index `t` is written as a constant value across all timepoints in the trial.

ii.
```python
inp = np.zeros((4, n_tp), dtype=np.float32)
...
inp[2, :] = t  # trial number (0-indexed)
```

iii. This follows directly from the AI’s decision to represent trial number as a per-trial variable rather than a sampled behavioral trace.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the raw `Reward` event timestamps, not from `autoreward` or a raw trial-level outcome array.

ii.
```python
reward_ts = beh['Reward']['timestamps'][()]
...
rewarded_trials = np.zeros(n_trials, dtype=bool)
```

iii. `CONVERSION_NOTES.md` Step 5 maps “reward (prev trial) -> Previous trial rewarded? 0/1,” and trajectory step 46 says the AI verified reward rates using reward timestamps.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first determines whether each trial was rewarded by asking whether any reward timestamp falls between that trial’s start and end times. It then sets each trial’s previous-trial-outcome input to the previous entry of that boolean array, or 0 for the first trial.

ii.
```python
for t in range(n_trials):
    t_start_time = timestamps[tstart_idx[t]]
    t_end_time = timestamps[teleport_idx[t]]
    rewarded_trials[t] = np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time))
...
prev_outcome = float(rewarded_trials[t-1]) if t > 0 else 0.0
...
inp[3, :] = prev_outcome
```

iii. In the trajectory, the AI explicitly checked that the resulting reward rate was near the paper’s ~15% omission rate and used that as justification for this implementation.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The distance output is derived from raw `position` plus a per-trial reward-zone identity inferred from the session `scene`, using the hardcoded `SESSIONS_SCENES` table and `get_reward_zone_for_trial`. The raw behavioral `reward_zone` stream is loaded but not used in this computation.

ii.
```python
scene = SESSIONS_SCENES[subject_id][ses_num]
...
trial_pos = position[start:stop]
...
coords, label = get_reward_zone_for_trial(scene, t, n_trials, change_trial=30)
...
rz_start_cm, rz_end_cm = rz_coords[t]
dist_to_rz = np.where(trial_pos < rz_start_cm, trial_pos - rz_start_cm,
                      np.where(trial_pos > rz_end_cm, trial_pos - rz_end_cm, 0.0))
```

iii. `CONVERSION_NOTES.md` Step 5 states “Reward zone from scene names,” and trajectory steps 52-54 show the AI deliberately switched to a `sessions_dict.py`-style scene mapping after noticing manual scene mismatches.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Once the reward-zone interval is known for a trial, the AI computes signed distance to the nearest edge: negative before the zone, zero inside the zone, positive after the zone. It then discretizes that continuous distance.

ii.
```python
dist_to_rz = np.where(trial_pos < rz_start_cm, trial_pos - rz_start_cm,
                      np.where(trial_pos > rz_end_cm, trial_pos - rz_end_cm, 0.0))
...
dist_bins = discretize_distance_to_rz(dist_to_rz)
```

iii. The trajectory ends with an explicit spot-check of this signed-distance logic and concludes it matches the task specification.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is discretized into 7 categories using manual threshold comparisons matching the requested bins.

ii.
```python
def discretize_distance_to_rz(distance):
    bins = np.full_like(distance, -1, dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```

iii. `CONVERSION_NOTES.md` Step 5 says this output uses the task’s 7 requested bins, and trajectory step 102 documents a final manual verification of the bin boundaries.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by taking `position[start:stop]` from the same trial boundaries used for neural slicing, then computing the distance per timepoint in that trial.

ii.
```python
neural = dff_valid[:, start:stop].astype(np.float32)
trial_pos = position[start:stop]
...
out_tv[0, :] = dist_bins
```

iii. The AI’s overall alignment strategy is to use the same trial index windows for every modality.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the raw behavioral `position` time series.

ii.
```python
position = beh['position']['data'][()]
...
trial_pos = position[start:stop]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `position` directly to both distance-to-zone and absolute-position outputs.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices `position` over each trial and discretizes the raw centimeter values into five 90 cm bins spanning the track.

ii.
```python
trial_pos = position[start:stop]
pos_bins = discretize_position(trial_pos)
...
out_tv[1, :] = pos_bins
```

iii. The notes state that the track is 450 cm long and that the output should be represented as five equal bins over that corridor.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The AI uses manual thresholds: `<90`, `90-180`, `180-270`, `270-360`, and `>=360`.

ii.
```python
def discretize_position(position):
    bins = np.full_like(position, -1, dtype=np.int64)
    bins[position < 90] = 0
    bins[(position >= 90) & (position < 180)] = 1
    bins[(position >= 180) & (position < 270)] = 2
    bins[(position >= 270) & (position < 360)] = 3
    bins[position >= 360] = 4
    return bins
```

iii. `CONVERSION_NOTES.md` Step 5 says “5 bins of 90cm,” which is exactly what this helper implements.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by taking the per-trial `position[start:stop]` slice using the same indices as the neural data.

ii.
```python
start = tstart_idx[t]
stop = teleport_idx[t]
neural = dff_valid[:, start:stop].astype(np.float32)
trial_pos = position[start:stop]
```

iii. The AI’s alignment approach is the same shared trial-window slicing used throughout the conversion.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw behavioral `lick` time series.

ii.
```python
lick = beh['lick']['data'][()]
...
trial_lick = lick[start:stop]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `lick -> output[3]`, and trajectory step 47 says the AI inspected the raw lick signal and concluded it should be binarized.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick signal is binarized per frame: any positive value becomes 1 and zero stays 0.

ii.
```python
trial_lick = lick[start:stop]
lick_binary = (trial_lick > 0).astype(np.int64)
...
out_tv[3, :] = lick_binary
```

iii. `CONVERSION_NOTES.md` Step 5 states “Lick binarization: lick > 0 = lick present,” and the trajectory justifies this as the correct interpretation of the cumulative frame-level lick counter.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by slicing `lick[start:stop]` with the same trial boundaries as the neural matrix.

ii.
```python
neural = dff_valid[:, start:stop].astype(np.float32)
trial_lick = lick[start:stop]
```

iii. The AI never performs an additional lick/neural interpolation step; shared frame indexing is its alignment method.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The converted reward-zone-location output is derived from session metadata (`scene` from the hardcoded `SESSIONS_SCENES` table plus session number and trial index), not from the raw `reward_zone` time series in the NWB file.

ii.
```python
scene = SESSIONS_SCENES[subject_id][ses_num]
...
coords, label = get_reward_zone_for_trial(scene, t, n_trials, change_trial=30)
...
rz_label_int = {'A': 0, 'B': 1, 'C': 2}[rz_labels[t]]
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says “Reward zone from scene names,” and trajectory steps 52-54 show the AI preferred the reference repository’s session metadata over interpreting the raw `reward_zone` trace.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the scene string to determine the pre-switch and post-switch reward zones. For switch sessions it changes from the first zone to the second at trial 30; then it maps `A/B/C` to integers `0/1/2` and broadcasts that value across the trial.

ii.
```python
def get_reward_zone_for_trial(scene, trial_idx, n_trials, change_trial=30):
    ...
    if trial_idx < change_trial:
        return REWARD_ZONE_DICT[pre_zone], pre_zone
    else:
        return REWARD_ZONE_DICT[post_zone], post_zone
```
```python
rz_label_int = {'A': 0, 'B': 1, 'C': 2}[rz_labels[t]]
...
out[4, :] = rz_label_int
```

iii. The notes cite “Switch trial = 30” from the paper, and the trajectory shows the AI matched scenes to session days using the reference `sessions_dict.py` information.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward` event timestamps.

ii.
```python
reward_ts = beh['Reward']['timestamps'][()]
...
rewarded_trials[t] = np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time))
```

iii. `CONVERSION_NOTES.md` Step 5 maps this output from reward events and uses the paper’s expected omission rate as a sanity check.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The AI marks a trial as rewarded if any reward timestamp falls within that trial’s start/end times, converts the boolean to `0/1`, and stores it as a constant across the whole trial.

ii.
```python
rewarded_trials[t] = np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time))
...
reward_outcome = int(rewarded_trials[t])
...
out[5, :] = reward_outcome
```

iii. The AI justified this by checking that the resulting omission rate was close to the paper’s reported ~15%.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles a few edge cases defensively: it truncates neural and behavioral streams to a common minimum length if they differ; it drops negative environment samples before summarizing environment per trial; it skips sessions with too few trials or valid cells; it skips trials with fewer than 2 timepoints; and it replaces NaNs in trial neural matrices with zeros before saving.

ii.
```python
n_timepoints = min(n_timepoints_neural, n_timepoints_beh)
if n_timepoints_neural != n_timepoints_beh:
    ...
    timestamps = timestamps[:n_timepoints]
```
```python
env_vals = environment[start:stop]
env_vals = env_vals[env_vals >= 0]  # exclude -1
```
```python
if n_tp < 2:
    continue
...
neural = np.nan_to_num(neural, nan=0.0)
```

iii. `CONVERSION_NOTES.md` Step 4 documents the neural/behavior off-by-one issue and says it was resolved by truncation. The notes and trajectory do not show any more elaborate missing-data policy than these simple guards.

## 13-a. What are the most time-consuming steps of the code?

i. The dominant costs are per-session NWB I/O and the dF/F computation in `compute_dff`, especially the per-trial Gaussian smoothing and min/max filtering. Interneuron detection is also a nontrivial per-cell loop.

ii.
```python
with h5py.File(nwb_path, 'r') as f:
    ...
```
```python
dff = compute_dff(F_raw, Fneu_raw, tstart_idx, teleport_idx, neu_coef=0.7)
...
is_interneuron = detect_interneurons(dff_cells, speed, threshold=0.5)
```

iii. `CONVERSION_NOTES.md` Step 7 reports “dF/F computation | 0.8-8.4s per session” and an overall runtime of about 800 s for the full conversion, indicating the AI itself identified dF/F as the bottleneck.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain scalar or repeated per trial: the per-cell loop in `detect_interneurons`, the per-trial loops for reward outcome, environment extraction, and trial assembly, and the per-trial loops inside `compute_dff`.

ii.
```python
for i in range(n_cells):
    valid = ~np.isnan(dff[i]) & speed_valid
    ...
```
```python
for t in range(n_trials):
    ...
    rewarded_trials[t] = np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time))
...
for t in range(n_trials):
    ...
    env_vals = environment[start:stop]
...
for t in range(n_trials):
    ...
    neural_trials.append(neural)
```

iii. The notes mention speed optimization work and the trajectory specifically calls out interneuron detection as an optimization target before the full run.

## 13-c. What processing does the code repeat multiple times?

i. The code makes multiple passes over the same trial boundaries: once to compute reward outcomes, once to compute per-trial environments, and once more to assemble neural/input/output trial arrays. `compute_dff` also loops over all trials twice, once for baseline estimation and once for smoothing.

ii.
```python
for t in range(n_trials):
    ...
    rewarded_trials[t] = ...
...
for t in range(n_trials):
    ...
    env_per_trial[t] = ...
...
for t in range(n_trials):
    ...
    neural_trials.append(neural)
```
```python
for i in range(len(trial_start_idx)):
    ...
    flow[:, start:stop] = ...
...
for i in range(len(trial_start_idx)):
    ...
    dff[:, start:stop] = nansmooth(dff[:, start:stop], 2, axis=1)
```

iii. The AI’s notes focus on runtime and dF/F cost rather than this repetition explicitly, but the structure of the final script clearly shows these repeated passes.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads several arrays that are never used downstream (`trial_number`, `scanning`, `autoreward`, `reward_zone`), computes `dt` without using it, imports `pearsonr` and `gaussian_filter` without using them, and creates an `out_trial` array that is immediately discarded.

ii.
```python
trial_number = beh['trial number']['data'][()]
scanning = beh['scanning']['data'][()]
autoreward = beh['autoreward']['data'][()]
reward_zone = beh['reward_zone']['data'][()]
...
dt = np.mean(np.diff(timestamps))  # ~0.0645s
```
```python
out_trial = np.array([rz_label_int, reward_outcome], dtype=np.int64)
...
out = np.zeros((6, n_tp), dtype=np.int64)
out[:4, :] = out_tv
out[4, :] = rz_label_int
out[5, :] = reward_outcome
```

iii. The trajectory shows the AI explored many variables during development, and some of that exploration remained in the final script as unused loads or intermediates.
