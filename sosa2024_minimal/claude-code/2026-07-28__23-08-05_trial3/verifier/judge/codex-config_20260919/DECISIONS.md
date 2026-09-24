# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers sorted `sub-*` directories, then every sorted `*.nwb` file in each directory, and reads each file directly with `h5py`. Every successfully converted file becomes one session.

ii.
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
with h5py.File(nwb_path, 'r') as f:
```

iii. The trajectory says inspection found 11 subjects and 152 NWB sessions and treated NWB as containing all behavioral and imaging streams. It chose direct HDF5 access and later reported all 152 sessions converted.

## 1-b. How are the data split into subjects?

i. A subject is each `sub-*` directory; its ID is the directory name with `sub-` removed. Session-to-subject membership is stored as the directory's enumerated index.

ii.
```python
subjects = [d.replace('sub-', '') for d in subjects_dirs]
for subj_i, (subj_dir, subj_id) in enumerate(zip(subjects_dirs, subjects)):
    ...
    subject_idx_list.append(subj_i)
```

iii. The agent observed that subject IDs map directly to mouse/GCAMP numbers and that the 11 directories match the paper's switch mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session, appended independently to the top-level session lists.

ii.
```python
for nwb_file in nwb_files:
    result = convert_session(nwb_file, subj_id, session_num)
    all_neural.append(result['neural'])
```

iii. The trajectory notes that most mice have 14 files and m11 has 12, matching the observed session numbering and producing 152 sessions.

## 1-d. How are the data split into trials?

i. Starts are every sample where `trial_start > 0`; ends are every sample where `teleport > 0`. The two lists are truncated to equal count, pairs with end at/before start are removed, and slices use `[start:end)`.

ii.
```python
tstart_inds = np.where(trial_start > 0)[0]
teleport_inds = np.where(teleport > 0)[0]
n_trials = min(len(tstart_inds), len(teleport_inds))
valid = teleport_inds > tstart_inds
...
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. The agent reasoned that `trial_start` and teleport define trials. After observing trials as long as 216 seconds, it acknowledged likely boundary errors but retained the method because the median duration and aggregate trial count looked plausible.

## 1-e. How are trials filtered based on quality controls?

i. Invalid start/end pairs are removed. Sessions need at least two valid trials and two retained cells; during assembly, only trials shorter than two samples are skipped. There is no 50-sample trial filter.

ii.
```python
valid = teleport_inds > tstart_inds
if n_trials < 2: return None
...
if n_timepoints < 2:
    continue
```

iii. The trajectory relied on format validation and aggregate trial statistics; it did not justify the two-sample cutoff or implement the reference's short-trial criterion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Final neural data comes directly from NWB `processing/ophys/Deconvolved/plane*/data`. `Fluorescence` and `Neuropil` are loaded only to compute a separate dF/F estimate for interneuron filtering.

ii.
```python
deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
...
deconv_filtered = deconv_all[:, cell_mask]
```

iii. The agent believed the NWB deconvolved traces were already processed OASIS activity. The trajectory explicitly says it chose this pragmatic shortcut while uncertain whether it should recompute the paper's signal.

## 2-b. How is the `neural` data processed?

i. Plane arrays are concatenated, truncated to the behavioral length, filtered by cell mask, sliced into trials, transposed to neuron-by-time, and cast to `float32`. The stored signal is not baseline-corrected, smoothed, or deconvolved by this script.

ii.
```python
deconv_all = np.concatenate([deconv_p0, deconv_p1], axis=1)
deconv_all = deconv_all[:min_len]
deconv_filtered = deconv_all[:, cell_mask]
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. The script header and trajectory characterize the stored array as OASIS calcium activity and therefore treat it as ready for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells must have `iscell[:,0] == 1` and must not have estimated dF/F–speed Pearson correlation above 0.5. The dF/F estimate subtracts 0.7 neuropil, adds trial mean neuropil, applies a uniform filter and maximin filters, then correlates each cell over trial samples.

ii.
```python
curated_mask = iscell[:, 0] == 1
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, ..., threshold=0.5)
cell_mask = curated_mask & ~is_interneuron
```

iii. The agent cites the Methods' manual curation and `r > 0.5` putative-interneuron rule. It called its dF/F implementation simplified and accepted 119 removals because the paper reports few interneurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial-start alignment is implicit: each neural matrix begins at the detected `trial_start` index, with no interpolation or shifting.

ii.
```python
s, e = tstart_inds[i], teleport_inds[i]
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. The agent stated that neural and behavior samples were already aligned and used common indices.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The code uses the stored imaging samples and reports `1000 / 15.5078125 = 64.48 ms` in metadata.

ii.
```python
frame_time = 1.0 / imaging_rate
...
'time_bin_size': 1000.0 / 15.5078125,
```

iii. The agent found an imaging rate near 15.5 Hz and chose to preserve it.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from sample number within the sliced trial and the NWB acquisition imaging rate, not directly from behavioral timestamps (although position timestamps are loaded).

ii.
```python
imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. The agent reasoned that neural and behavior sampling are aligned and approximately 15.5 Hz.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based arithmetic sequence is multiplied by seconds per frame.

ii.
```python
frame_time = 1.0 / imaging_rate
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. This was chosen as a simple representation of elapsed time at a constant acquisition rate.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It has exactly the sliced neural trial's number of columns and starts at zero in its first column.

ii.
```python
n_timepoints = e - s
neural = deconv_filtered[s:e, :].T
input_data[0, :] = np.arange(n_timepoints) * frame_time
```

iii. The agent assumed shared indices and verified dimensions through the decoder validator.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
env = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
env_vals = env[s:e]
```

iii. Exploration showed values 0 and 1 correspond directly to ENV1 and ENV2, with negative values marking intertrial periods.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative values are discarded; the median remaining value is cast to integer and broadcast over the trial. If none remain, it defaults to 0.

ii.
```python
env_valid = env_vals[env_vals >= 0]
env_per_trial[i] = int(np.median(env_valid)) if len(env_valid) > 0 else 0
input_data[1, :] = env_type
```

iii. The agent viewed environment as constant within each trial and used a robust per-trial summary.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the conversion loop index, not the NWB `trial_number` stream.

ii.
```python
for i in range(n_trials):
    trial_number = float(i + 1)
```

iii. The trajectory focused on detected boundaries and sequential trial order; no separate justification for ignoring the stored field was recorded.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The zero-based loop index is incremented to make it one-based, cast to float, and broadcast over every timepoint.

ii.
```python
trial_number = float(i + 1)
input_data[2, :] = trial_number
```

iii. It was treated as a continuous per-trial covariate; the one-based convention was not justified.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It derives from sparse `Reward/timestamps`, position timestamps, and the detected previous trial boundaries.

ii.
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

iii. The agent inspected reward events and decided that any reward timestamp within a trial means rewarded.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Current trial outcomes are computed by inclusive timestamp-window tests. Trial 0 gets 0; later trials receive the preceding outcome, broadcast over time.

ii.
```python
n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
...
prev_outcome = 0.0 if i == 0 else float(reward_outcomes[i - 1])
input_data[3, :] = prev_outcome
```

iii. This directly implements omitted=0/rewarded=1 and assigns no previous reward to the first trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw position plus a per-trial zone inferred from samples where `reward_zone > 0`; the median such position is matched to fixed A/B/C ranges with ±20 cm tolerance. Missing labels are forward-filled then backward-filled.

ii.
```python
rz_mask = rzone[s:e] > 0
rz_positions = pos[s:e][rz_mask]
median_pos = np.median(position_when_in_rzone)
...
zone_labels = determine_reward_zone_per_trial(...)
```

iii. Exploration showed reward-zone activation at the expected three spatial ranges. The agent used neighboring trials for omission trials because zones persist in blocks.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position before the zone gets `position - zone_start`, position inside gets zero, and position after gets `position - zone_end`; the result is then discretized.

ii.
```python
distance[before] = position[before] - rz_start
distance[inside] = 0
distance[after] = position[after] - rz_end
```

iii. The agent followed the requested signed distance to the nearest zone edge.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks produce seven requested classes at -50, -10, 0, 10, and 50 cm.

ii.
```python
bins[distance < -50] = 0
bins[(distance >= -50) & (distance < -10)] = 1
bins[(distance >= -10) & (distance < 0)] = 2
bins[distance == 0] = 3
bins[(distance > 0) & (distance <= 10)] = 4
bins[(distance > 10) & (distance <= 50)] = 5
bins[distance > 50] = 6
```

iii. The agent states these masks reproduce the task's category definitions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with the same `[s:e)` indices and the resulting category vector fills the same number of columns as neural data.

ii.
```python
trial_pos = pos[s:e]
output_data[0, :] = dist_disc
```

iii. The agent assumed common neural/behavior sampling and relied on equal array dimensions.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes from the behavioral `position/data` array.

ii.
```python
pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
trial_pos = pos[s:e]
```

iii. The agent identified this as corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Trial position is clipped to `[0, 450]`, divided by 90, floored, and clipped to class 0–4.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
```

iii. The agent used the stated 450 cm track and five equal bins, clipping out-of-range noise.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90 cm classes are used: floor(position/90), constrained to 0 through 4.

ii.
```python
bin_size = TRACK_LENGTH / n_bins
bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
```

iii. This follows the instruction to divide a 450 cm track into five equal bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is sliced using the same start/end indices and stored column-for-column with neural data.

ii.
```python
trial_pos = pos[s:e]
output_data[1, :] = pos_disc
```

iii. The agent assumed the NWB behavioral and imaging arrays are sample-aligned.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It derives from behavioral `lick/data` and the trial boundaries.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
trial_lick = lick_binary[s:e]
```

iii. The agent identified the raw stream as lick counts/events.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Before binarization, any trial where more than 35% of samples have lick value above 2 is overwritten with zero. Remaining positive values become 1.

ii.
```python
if frac_high > correction_thr:
    licks_corrected[s:e] = 0
lick_binary = (lick_corrected > 0).astype(float)
```

iii. The agent cites a reference `correction_thr=0.35` sensor-error rule, but chose zero rather than NaN to keep a binary decoder target.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Corrected lick data is sliced with `[s:e)` and assigned to the neural trial's columns.

ii.
```python
trial_lick = lick_binary[s:e]
output_data[3, :] = lick_disc
```

iii. Common NWB indices were assumed to provide alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from `reward_zone/data` jointly with `position/data`.

ii.
```python
zone_labels = determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds)
```

iii. The agent observed reward-zone-positive samples clustered around the paper's A/B/C ranges.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Median active-zone position is matched to A/B/C with tolerance; missing labels are neighbor-filled, remaining missing labels default to A, and labels map to 0/1/2 and are broadcast.

ii.
```python
zone_labels[i] = get_reward_zone_label(rz_positions)
...
if rz_label is None: rz_label = 'A'
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
output_data[4, :] = rz_loc
```

iii. The agent reasoned that omission trials can inherit the block's zone from nearby rewarded trials.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses sparse NWB Reward timestamps, position timestamps, and trial start/end indices; loaded Reward values are unused.

ii.
```python
reward_data = f['.../Reward/data'][:]
reward_ts = f['.../Reward/timestamps'][:]
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, ...)
```

iii. The trajectory confirmed the timestamped Reward series records delivery events.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if at least one reward timestamp lies inclusively between its boundary timestamps, otherwise 0; it is broadcast over trial columns.

ii.
```python
n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
outcomes[i] = 1 if n_rewards > 0 else 0
output_data[5, :] = rew_out
```

iii. The agent compared its approximately 15.3% omission rate with the paper's approximately 15% as validation.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and all behavior streams are truncated using the minimum of position length and neural length; start/end lists are truncated and invalid pairs removed. Missing environment defaults to ENV1, missing zone labels are filled from neighbors then default to A, constant/insufficient dF/F correlations do not remove cells, and sessions/trials below minimal counts are skipped.

ii.
```python
min_len = min(len(pos), deconv_all.shape[0])
...
n_trials = min(len(tstart_inds), len(teleport_inds))
...
if rz_label is None: rz_label = 'A'
```

iii. The agent encountered a multi-plane off-by-one mismatch and added truncation. Other fallbacks were defensive choices intended to keep a complete, validator-compatible dataset.

## 13-a. What are the most time-consuming steps of the code?

i. Reading every large NWB array, computing trial-wise dF/F filters, per-cell Pearson correlations, serializing the 9.2 GB pickle, and optional decoder training are the expensive operations.

ii.
```python
with h5py.File(nwb_path, 'r') as f:
    deconv_p0 = ...[:]
...
for cell in range(n_cells):
    r, _ = stats.pearsonr(dff_valid, speed_valid)
pickle.dump(data, f, protocol=4)
```

iii. The trajectory reports full conversion and especially full decoder preprocessing/training as long-running; it does not provide formal profiling.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Interneuron correlations could be computed in matrix form; reward-outcome timestamp tests, environment summaries, reward-zone detection/filling, trial dF/F processing, sanity-check counts, and per-trial assembly contain Python loops that could partly be vectorized.

ii.
```python
for cell in range(n_cells): ...
for i in range(n_trials): ...
for i, (s, e) in enumerate(zip(tstart_inds, teleport_inds)): ...
```

iii. The agent did not discuss vectorization in the recorded rationale; its emphasis was correctness checks and completing all sessions.

## 13-c. What processing does the code repeat multiple times?

i. Trial boundaries are traversed separately for dF/F, valid-mask creation, lick correction, zone inference/filling, reward outcomes, environment, and final assembly. Sanity checks later traverse every session/trial again.

ii.
```python
for s, e in zip(tstart_inds, teleport_inds):  # dF/F
for s, e in zip(tstart_inds, teleport_inds):  # mask
for i, (s, e) in enumerate(...):              # lick
for i in range(n_trials):                     # zone/outcome/env/build
```

iii. No explicit justification was recorded; the modular helper design makes the repetition straightforward but adds passes over the data.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `planeIdx` and Reward values without using them; computes/returns diagnostic counts, labels, outcomes, environments, and lick-error lists mostly for logging; deep-copies a sample dataset; and runs extensive sanity summaries not needed by the saved full dataset.

ii.
```python
planeIdx = f['.../planeIdx'][:]
reward_data = f['.../Reward/data'][:]
...
sample = copy.deepcopy(data)
```

iii. These operations supported debugging, reporting, and creation of an extra sample file, but the trajectory gives no downstream-analysis need for them.
