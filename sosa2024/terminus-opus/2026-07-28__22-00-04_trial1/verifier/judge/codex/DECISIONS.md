# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by globbing every NWB file matching `data/sub-*/sub-*_ses-*_behavior+ophys.nwb`, then iterating over those files and opening each one with `h5py`. Each file is treated as one session, and all trial-level data are extracted inside `process_session`.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*_ses-*_behavior+ophys.nwb'))
...
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, show_processing=args.show_processing, session_idx=i)
```
```python
f = h5py.File(nwb_path, 'r')
```

iii. In `CONVERSION_NOTES.md` Step 6, the AI says the script "Loads NWB files with h5py." In trajectory step 45, it concludes that all 11 subjects in the NWB release should be included, so the file-glob is intended to capture the entire dataset.

## 1-b. How are the data split into subjects (mice)?

i. The AI does not pre-split by directory. Instead, it reads each file's `general/subject/subject_id`, stores that per session, then builds `subjects` as the sorted unique set of those IDs and `subject_idx` as a per-session mapping.

ii.
```python
subj_id = f['general']['subject']['subject_id'][()].decode()
...
all_subj_ids.append(result['subj_id'])
...
unique_subjects = sorted(set(all_subj_ids), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
subject_idx = np.array([unique_subjects.index(s) for s in all_subj_ids])
```

iii. In Step 9 of `CONVERSION_NOTES.md`, the AI reports 11 subjects and treats the file metadata as authoritative. In trajectory steps 46-47, it explicitly maps the released mouse IDs to the paper's animal IDs and decides all 11 should be included.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. The session identifier is read from NWB metadata, but the main split is simply one file per session in the top-level processing loop.

ii.
```python
sess_id = f['general']['session_id'][()].decode()
...
for i, nwb_file in enumerate(nwb_files):
    result = process_session(nwb_file, show_processing=args.show_processing, session_idx=i)
    all_neural.append(result['neural_trials'])
```

iii. `CONVERSION_NOTES.md` Step 5 lists "Each session = one NWB file" as a key decision, and trajectory step 49 repeats that choice explicitly.

## 1-d. How are the data split into trials?

i. Trials are defined by indices where `trial_start > 0` and `teleport > 0`. For each trial, the AI slices arrays over `[trial_start_idx[t] : teleport_idx[t]]`. If the counts differ, both arrays are truncated to the shorter count.

ii.
```python
trial_start_inds = np.where(trial_start_markers > 0)[0]
teleport_inds = np.where(teleport_markers > 0)[0]
...
if n_trials != len(teleport_inds):
    n_trials = min(n_trials, len(teleport_inds))
    trial_start_inds = trial_start_inds[:n_trials]
    teleport_inds = teleport_inds[:n_trials]
...
t_start = trial_start_inds[t]
t_end = teleport_inds[t]
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 3 says temporal alignment is "per-trial, from trial_start to teleport," and trajectory step 49 records the decision "Each trial = trial_start to teleport."

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is minimal. Trials with fewer than 2 timepoints are skipped, and sessions with fewer than 2 remaining trials are dropped. No 50-sample minimum or other trial-quality curation is applied.

ii.
```python
n_tp = t_end - t_start
if n_tp < 2:
    continue
...
if len(result['neural_trials']) < 2:
    print(f"  SKIPPING: fewer than 2 trials")
    continue
```

iii. The AI gives no detailed trial-QC justification beyond making the decoder input valid. In trajectory step 62, it explicitly decides to keep unusually long trials because it considers them "real data."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` directly from the NWB `processing/ophys/Deconvolved/plane*/data` arrays, not from fluorescence and neuropil traces.

ii.
```python
deconv_group = f['processing']['ophys']['Deconvolved']
plane_keys = sorted([k for k in deconv_group.keys() if k.startswith('plane')])
if len(plane_keys) == 1:
    deconv_data = deconv_group[plane_keys[0]]['data'][:]
else:
    plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
    deconv_data = np.concatenate(plane_data, axis=1)
```

iii. In `CONVERSION_NOTES.md` Step 1, the AI states "The NWB files contain pre-computed deconvolved events, so dF/F computation is not needed." Trajectory step 45 makes the same justification more directly: the NWB already has the deconvolved data.

## 2-b. How is the `neural` data processed?

i. Neural processing is limited to concatenating multiple imaging planes, filtering ROIs by `iscell`, transposing to `(n_neurons, n_timepoints)`, slicing by trial, and converting to `float32`. The AI does not recompute dF/F, baseline, or deconvolution.

ii.
```python
plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
deconv_data = np.concatenate(plane_data, axis=1)
...
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T
...
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 lists "Use iscell as-is" and "Multi-plane pooling" as key decisions. Trajectory steps 56-57 justify concatenating planes because the paper pools multi-plane ROIs for analysis.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural QC applied is keeping ROIs with `iscell[:, 0] == 1`. The AI does not apply the paper's putative-interneuron exclusion.

ii.
```python
iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][:]
...
cell_mask = iscell[:, 0] == 1
neural_all = deconv_data[:, cell_mask].T
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly writes: "Use iscell as-is: NWB iscell includes manual curation; interneuron filtering skipped (~0.42% effect)." Trajectory step 48 shows it considered interneuron filtering but decided not to add it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start implicitly by splitting each session into trial slices beginning at `trial_start_inds[t]`.

ii.
```python
t_start = trial_start_inds[t]
t_end = teleport_inds[t]
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
```

iii. The output metadata says the temporal alignment event is `"trial_start (entry to linear track)"`, and the notes repeatedly describe the conversion as aligned from trial start to teleport.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native sampling resolution and does not rebin. It estimates the time bin size as the median difference in `position.timestamps`, then uses that `dt` for metadata and time input.

ii.
```python
pos_timestamps = beh['position']['timestamps'][:]
...
dt = np.median(np.diff(pos_timestamps))
...
'time_bin_size': median_dt * 1000,
```

iii. `CONVERSION_NOTES.md` Step 5 states "Native frame rate: Use ~64.5 ms time bins (no resampling)," and Steps 7 and 9 report a final bin size of 64.48 ms.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps on the `position` time series, specifically via the session-wide step size computed from consecutive `position.timestamps`.

ii.
```python
pos_timestamps = beh['position']['timestamps'][:]
dt = np.median(np.diff(pos_timestamps))
```

iii. The AI does not justify choosing `position.timestamps` over another behavior stream in the code, but its notes emphasize that all behavior streams share the same frame rate and that native timing should be preserved.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI creates a regular time vector as `np.arange(n_tp) * dt`, using the session-wide median timestep rather than subtracting the raw timestamp of the first sample in that trial.

ii.
```python
n_tp = t_end - t_start
time_from_start = np.arange(n_tp) * dt
...
input_data[0, :] = time_from_start
```

iii. The justification is implicit in the notes' emphasis on a fixed native frame rate and no resampling. The AI treats the behavior/neural streams as uniformly sampled once `dt` has been estimated.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector is aligned by construction: each trial's `n_tp` is taken from the same `[t_start:t_end]` slice used for the neural data, so the time axis has exactly one value per neural sample.

ii.
```python
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
...
n_tp = t_end - t_start
input_data = np.zeros((4, n_tp), dtype=np.float32)
input_data[0, :] = np.arange(n_tp) * dt
```

iii. `CONVERSION_NOTES.md` Step 10 says neural and behavioral spot-checks matched on sample trials, reflecting the AI's view that the stored streams were already aligned.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavior `environment` time series.

ii.
```python
env_data = beh['environment']['data'][:]
...
env_vals = env_data[t_start:t_end]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `environment data` directly to `input[1]`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the AI drops negative values, takes the median of the remaining `environment` samples, and broadcasts that single trial-level value across all timepoints in the trial.

ii.
```python
env_vals = env_data[t_start:t_end]
env_vals = env_vals[env_vals >= 0]
if len(env_vals) > 0:
    trial_env[t] = int(np.median(env_vals))
...
input_data[1, :] = env_type
```

iii. The notes treat environment as a per-trial binary decoder input, not as a time-varying behavior. That is the AI's rationale for collapsing the trial to one value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the within-session trial index `t` in the loop over trial boundaries; indirectly this depends on `trial_start` and `teleport` because they define the trial list.

ii.
```python
for t in range(n_trials):
    ...
    trial_num = float(t)
```

iii. `CONVERSION_NOTES.md` Step 5 lists "Trial index" as the mapping for `input[2]`, and trajectory step 49 lists this as a key design decision.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No additional processing is applied beyond converting the loop index to float and repeating it across the trial.

ii.
```python
trial_num = float(t)
...
input_data[2, :] = trial_num
```

iii. The AI treats trial number as a simple per-trial context variable, so it is constant across timepoints.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward` timestamps, aligned to the behavior frame indices using `position.timestamps`.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
pos_timestamps = beh['position']['timestamps'][:]
...
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
```

iii. `CONVERSION_NOTES.md` Step 5 maps previous trial reward directly from reward delivery. The AI's implementation assumes behavior timestamps are the common frame of reference.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a per-trial binary `trial_rewarded` label by checking whether any reward timestamp falls in the current trial. Then, for trial `t`, it sets `previous_trial_outcome` to `trial_rewarded[t-1]`, with the first trial forced to 0.

ii.
```python
reward_in_trial = np.any(
    (reward_frame_inds >= t_start) & (reward_frame_inds < t_end)
)
trial_rewarded[t] = 1 if reward_in_trial else 0
...
if t == 0:
    prev_outcome = 0.0
else:
    prev_outcome = float(trial_rewarded[t - 1])
```

iii. The notes describe this variable as "Previous trial reward" and the trajectory treats it as a straightforward per-trial binary context input.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the behavior `position` time series plus a per-trial inferred reward-zone start. That reward-zone start is inferred from `reward_zone` activity within each trial; when `reward_zone` is absent, the AI fills the label from the nearest neighboring labeled trial.

ii.
```python
position = beh['position']['data'][:]
rzone = beh['reward_zone']['data'][:]
...
trial_rzone = rzone[t_start:t_end]
trial_pos = pos[t_start:t_end]
rz_active = trial_rzone > 0
if np.any(rz_active):
    rz_start_pos = trial_pos[rz_active].min()
    rz_labels[t] = determine_reward_zone_label(rz_start_pos)
```
```python
label = get_reward_zone_for_trial(rz_labels, t)
trial_rz_start[t] = REWARD_ZONES[label][0]
```

iii. `CONVERSION_NOTES.md` Step 10 says the reward zone was "inferred from position data" and verified on sampled trials. Trajectory step 45 says the AI decided the position where `reward_zone > 0` reveals the reward-zone start.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the AI computes signed distance to the nearest edge of a 50 cm reward zone: negative before the zone, zero inside it, and positive after it. It assumes the end of the zone is `rz_start + 50`.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start):
    rz_end = rz_start + 50.0
    distance = np.zeros_like(position)
    distance[position < rz_start] = position[position < rz_start] - rz_start
    distance[position > rz_end] = position[position > rz_end] - rz_end
    return distance
```

iii. `CONVERSION_NOTES.md` Step 5 lists this as "Position - reward zone" with 7-bin discretization, and the code comment states the intended signed-distance convention.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is discretized into 7 categories using explicit threshold comparisons that match the task specification.

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

iii. The notes explicitly say `distance_to_reward_zone` was "7-bin discretization" and the code mirrors the written task bins.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by slicing `position` over the same trial indices used for the neural data, then building one distance label per sample.

ii.
```python
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
trial_pos = position[t_start:t_end]
distance = compute_distance_to_reward_zone(trial_pos, rz_start)
```

iii. The AI uses the same `[t_start:t_end]` interval for neural and behavioral data throughout the session conversion.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior `position` time series.

ii.
```python
position = beh['position']['data'][:]
...
trial_pos = position[t_start:t_end]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `Position` directly to `output[1]`.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI takes the per-trial position samples and discretizes them into 5 equal-width bins spanning the 450 cm track with `np.digitize`, clipping any out-of-range values back into the end bins.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. `CONVERSION_NOTES.md` Step 5 calls this "5-bin equal discretization," directly matching the decoder spec.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The thresholds are the equally spaced bin edges `[0, 90, 180, 270, 360, 450]`, implemented via `np.digitize` on the upper five edges and clipping to bin indices 0-4.

ii.
```python
bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
bins = np.digitize(position, bin_edges[1:])
bins = np.clip(bins, 0, n_bins - 1)
```

iii. The AI's notes explicitly reference 5 equal bins over a 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by slicing `position` with the same trial boundaries used for `trial_neural`.

ii.
```python
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
trial_pos = position[t_start:t_end]
pos_bins = discretize_position(trial_pos, n_bins=5)
```

iii. The code treats position and neural activity as already synchronized sample-by-sample within a session.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior `lick` time series.

ii.
```python
lick_data = beh['lick']['data'][:]
...
trial_licks = licks_processed[t_start:t_end]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `lick` directly to `output[3]`.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick processing happens in two stages. First, per-trial lick traces are flagged as sensor errors if more than 35% of samples exceed 2, and those samples are set to `NaN`; all remaining values above 1 are clipped to 1. Second, within each kept trial, the lick trace is Gaussian-smoothed with `sigma=2`, thresholded at `> 0.5`, and any `NaN` samples are set to 0 in the binary output.

ii.
```python
if len(trial_licks) > 0:
    frac_high = np.sum(trial_licks > 2) / len(trial_licks)
    if frac_high > LICK_ERROR_THRESHOLD:
        licks[t_start:t_end] = np.nan
licks[licks > 1] = 1
```
```python
smoothed_licks = nansmooth(trial_licks, LICK_SMOOTH_SIGMA)
lick_binary = (smoothed_licks > 0.5).astype(np.int64)
lick_binary[np.isnan(smoothed_licks)] = 0
```

iii. `CONVERSION_NOTES.md` Steps 1, 3, and 10 cite the paper code's lick error correction threshold and Gaussian smoothing, and the AI says it followed that logic.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick trace is aligned by slicing the processed lick array over the same `[t_start:t_end]` interval used for the neural data, then emitting one binary label per sample.

ii.
```python
trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
trial_licks = licks_processed[t_start:t_end]
```

iii. The AI treats the behavior samples as already synchronized to the neural recording grid and preserves that shared indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `reward_zone` and `position`: the AI infers a per-trial reward-zone label from where `reward_zone > 0` occurs in position space.

ii.
```python
rzone = beh['reward_zone']['data'][:]
position = beh['position']['data'][:]
...
rz_start_pos = trial_pos[rz_active].min()
rz_labels[t] = determine_reward_zone_label(rz_start_pos)
```

iii. The notes and trajectory both say reward-zone identity was inferred from the position of samples where the reward-zone signal was active.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI maps the inferred zone start to one of fixed labels `A/B/C` using tolerant range checks. If a trial has no active reward-zone signal, it copies the nearest non-missing label from a neighboring trial, with a final fallback to `A`. It then maps `A/B/C` to `0/1/2`.

ii.
```python
def determine_reward_zone_label(rz_start_pos):
    for label, (start, end) in REWARD_ZONES.items():
        if start - 20 < rz_start_pos < end + 20:
            return label
    return None
```
```python
label = get_reward_zone_for_trial(rz_labels, t)
trial_rz_label[t] = rz_label_map.get(label, 0)
```

iii. `CONVERSION_NOTES.md` Step 10 says these labels were spot-checked and appeared to match expected switch-session patterns. The code comments justify neighbor-based filling specifically for omission trials.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward` timestamps in the behavior group, aligned onto the frame index using `position.timestamps`.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
pos_timestamps = beh['position']['timestamps'][:]
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
```

iii. `CONVERSION_NOTES.md` Step 5 maps reward delivery to `output[5]`, and the code treats those timestamps as the source of truth.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are converted to frame indices with `searchsorted`. A trial is marked rewarded if any converted reward frame falls in that trial's `[t_start, t_end)` interval. The output is constant across timepoints within the trial.

ii.
```python
reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
reward_frame_inds = np.clip(reward_frame_inds, 0, n_timepoints - 1)
...
reward_in_trial = np.any(
    (reward_frame_inds >= t_start) & (reward_frame_inds < t_end)
)
trial_rewarded[t] = 1 if reward_in_trial else 0
...
output_data[5, :] = reward_out
```

iii. The notes describe this variable as a simple binary per-trial output and report that sampled spot-checks matched expectation.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases, but only with lightweight heuristics:
- If `trial_start` and `teleport` counts disagree, both are truncated to the shorter count.
- If a trial has no active `reward_zone`, reward-zone identity is borrowed from the nearest neighboring labeled trial, falling back to `A`.
- Negative environment values are discarded before taking the per-trial median.
- Lick sensor artifacts are turned into `NaN` per trial, then become zeros after smoothing/thresholding.
- Very short trials (`n_tp < 2`) are dropped; unusually long trials are kept.

ii.
```python
if n_trials != len(teleport_inds):
    n_trials = min(n_trials, len(teleport_inds))
    trial_start_inds = trial_start_inds[:n_trials]
    teleport_inds = teleport_inds[:n_trials]
```
```python
if rz_labels[trial_idx] is not None:
    return rz_labels[trial_idx]
...
return 'A'
```

iii. The justifications are scattered across `CONVERSION_NOTES.md`: Step 4 resolves some discrepancies pragmatically, Step 10 says edge cases looked reasonable on spot-checks, and trajectory step 62 explicitly keeps very long trials because the AI judged them legitimate.

## 13-a. What are the most time-consuming steps of the code?

i. In the AI code, the expensive steps are loading large NWB arrays for each session, iterating over all trials to build neural/input/output arrays, and saving the final multi-gigabyte pickle file. Optional plotting also adds overhead.

ii.
```python
f = h5py.File(nwb_path, 'r')
...
for t in range(n_trials):
    ...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. `CONVERSION_NOTES.md` Step 7 gives per-session runtime estimates, and Step 9 reports a 9.4 GB output file and about 70 seconds total conversion time, indicating I/O dominates.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several per-trial loops could have been consolidated or partially vectorized: reward-zone start detection, lick QC, per-trial reward assignment, environment summarization, reward-zone label filling, reward-zone start lookup, and the final trial-building loop. Some session-wide discretizations could also be done before splitting into trials.

ii.
```python
for t in range(n_trials):
    ...
```
This pattern appears in `get_reward_zone_start_per_trial`, `process_licks`, the `trial_rewarded` loop, the `trial_env` loop, the `trial_rz_label` and `trial_rz_start` loops, and the main loop that builds `neural_trials`, `input_trials`, and `output_trials`.

iii. The AI does not discuss vectorization explicitly; this is inferred from the structure of `convert_data.py`.

## 13-c. What processing does the code repeat multiple times?

i. The code makes several repeated passes over the same trial boundaries before the final output is built:
- one pass to infer reward-zone starts and labels,
- one pass to do lick QC,
- one pass to compute `trial_rewarded`,
- one pass to compute `trial_env`,
- one pass to compute `trial_rz_label`,
- one pass to compute `trial_rz_start`,
- one final pass to slice data and construct trial arrays.
It also calls `get_reward_zone_for_trial` twice per trial and traverses lick data once for QC and again for smoothing/binarization.

ii.
```python
rz_starts, rz_labels = get_reward_zone_start_per_trial(...)
licks_processed = process_licks(...)
...
for t in range(n_trials):
    ...
for t in range(n_trials):
    ...
for t in range(n_trials):
    ...
for t in range(n_trials):
    ...
for t in range(n_trials):
    ...
```

iii. There is no explicit justification in the notes; the repetition appears to be a straightforward implementation choice.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some work that is not used in the final saved dataset:
- it reads `scanning` but never uses it,
- it stores `all_sess_ids` during conversion but does not write them to the output dictionary,
- it computes `rz_starts` even though the saved outputs ultimately use canonical starts from `REWARD_ZONES[label][0]`,
- it contains plotting-only summary computations when `--show-processing` is used.

ii.
```python
scanning = beh['scanning']['data'][:]
...
all_sess_ids.append(result['sess_id'])
...
rz_starts, rz_labels = get_reward_zone_start_per_trial(...)
...
all_dist = np.concatenate([o[0, :] for o in output_trials])
all_pos = np.concatenate([o[1, :] for o in output_trials])
all_spd = np.concatenate([o[2, :] for o in output_trials])
all_lck = np.concatenate([o[3, :] for o in output_trials])
```

iii. The AI does not justify these extra computations explicitly. They appear to come from exploratory diagnostics and optional plotting support rather than the final decoder dataset itself.
