# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by globbing every NWB file under `/app/data/sub-*`, then processes each file one at a time with `h5py`. Within each file it reads fluorescence, neuropil, and the needed behavior arrays, then later splits them into trials.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/sub-*_ses-*_behavior+ophys.nwb'))
...
for i, nwb_path in enumerate(nwb_files):
    sess = process_session(nwb_path, show_processing=args.show_processing and i < 2)
```

```python
with h5py.File(nwb_path, 'r') as f:
    F, Fneu, n_planes = load_neural_data(f)
    behav = f['processing/behavior/BehavioralTimeSeries']
    position = behav['position/data'][:]
    speed = behav['speed/data'][:]
```

iii. In `CONVERSION_NOTES.md`, the AI says the dataset is organized as one NWB file per session under subject folders and reports finding 11 subjects and 152 sessions. The notes justify using all matching files so the converted dataset covers the full release.

## 1-b. How are the data split into subjects?

i. Subjects are not taken from directory names during final conversion. Instead, each processed session reads `general/subject/subject_id`, and the final `subjects` list is the sorted set of those IDs. `subject_idx` is built by mapping each session to that subject ID.

ii.
```python
with h5py.File(nwb_path, 'r') as f:
    subject_id = f['general/subject/subject_id'][()].decode()
```

```python
subjects = sorted(set(s['subject_id'] for s in all_sessions))
sub2idx = {s: i for i, s in enumerate(subjects)}
...
subject_idx.append(sub2idx[s['subject_id']])
```

iii. The notes say there are 11 mice and list the subject IDs found in the data. Using the NWB subject metadata is the AI's justification for not depending on path parsing after discovery.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The AI reads the NWB `session_id` for documentation, but the session split itself is the per-file loop over `nwb_files`.

ii.
```python
for i, nwb_path in enumerate(nwb_files):
    sess = process_session(nwb_path, show_processing=args.show_processing and i < 2)
```

```python
with h5py.File(nwb_path, 'r') as f:
    session_id = f['general/session_id'][()].decode()
```

iii. The notes describe the native organization as `sub-{mouse}/sub-{mouse}_ses-{session}_behavior+ophys.nwb`, so one file naturally corresponds to one session.

## 1-d. How are the data split into trials?

i. Trials are split by taking every sample where `trial_start==1` as a trial onset and every sample where `teleport==1` as a trial end, then pairing those arrays in order. The AI slices each trial with `[trial_start_idx:teleport_idx]`.

ii.
```python
trial_start_inds = np.where(trial_start_sig == 1)[0]
teleport_inds = np.where(teleport_sig == 1)[0]
n_trials = min(len(trial_start_inds), len(teleport_inds))
trial_start_inds = trial_start_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]
```

```python
for ti in range(n_trials):
    ts, te = trial_start_inds[ti], teleport_inds[ti]
    nf = te - ts
```

iii. The notes say "Trial = track only: trial_start to teleport (excludes teleport zone)." They also state that `trial_start` and `teleport` matched the intended lap boundaries during exploration.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies very little trial filtering. It drops whole sessions with fewer than 2 detected trials, and within a kept session it skips trials with fewer than 2 frames. It does not apply the human reference's 50-frame minimum or any explicit scanning-based exclusion.

ii.
```python
if n_trials < 2:
    print(f"    Skipping: {n_trials} trials"); return None
```

```python
for ti in range(n_trials):
    ts, te = trial_start_inds[ti], teleport_inds[ti]
    nf = te - ts
    if nf < 2: continue
```

iii. The notes justify this by saying "All trials included: No speed filtering" and later report that no very short trials were found in practice. The code therefore keeps nearly everything unless the trial is degenerate.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from the raw optical physiology fluorescence and neuropil traces, not from the NWB `Deconvolved` array. It loads `Fluorescence` and `Neuropil` plane by plane and keeps only ROIs marked `iscell==1`.

ii.
```python
fluor = f['processing/ophys/Fluorescence']
neu = f['processing/ophys/Neuropil']
seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = seg['iscell'][:]
plane_idx = seg['planeIdx'][:]
```

```python
F_list.append(fluor[pname]['data'][:, pcell])
Fneu_list.append(neu[pname]['data'][:, pcell])
```

iii. The notes explicitly say "NWB Deconvolved is suite2p raw, NOT the post-dF/F events from reference pipeline" and justify rebuilding the neural signal from `F` and `Fneu`.

## 2-b. How is the `neural` data processed?

i. The AI computes a dF/F-like signal and uses that directly as `neural`. It subtracts `0.7 * Fneu`, estimates a within-trial maximin baseline with 300-sample min and max filters, computes `(F - baseline)/|baseline|`, smooths each trial with a Gaussian sigma of 2, and then stops. It does not deconvolve into events, does not add back the trial mean neuropil before division, does not do the reference code's pre-baseline Gaussian smoothing with sigma 15, and does not use teleport-session metadata.

ii.
```python
f_ = (F.T - NEU_COEF * Fneu.T).astype(np.float64)
...
flow[:, s:e] = scipy.ndimage.minimum_filter1d(f_[:, s:e], w, axis=-1)
flow[:, s:e] = scipy.ndimage.maximum_filter1d(flow[:, s:e], w, axis=-1)
...
dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
...
dff[:, s:e] = nansmooth(dff[:, s:e], SMOOTH_SIGMA, axis=1)
```

```python
trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. The notes justify this two ways: they say this is closer to the paper than using NWB `Deconvolved`, and they explicitly prefer dF/F over deconvolved events because it "preserves more info for decoding." They also claim the code follows the reference pipeline, although the implementation only matches part of that pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons only by suite2p's `iscell` flag. It does not remove putative interneurons or apply any later neural-quality curation beyond replacing NaN/Inf values with 0 at the per-trial export step.

ii.
```python
pcell = iscell[pmask, 0] == 1
F_list.append(fluor[pname]['data'][:, pcell])
Fneu_list.append(neu[pname]['data'][:, pcell])
```

```python
trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. The notes say "All iscell=1 neurons: No place cell filtering (analysis-specific)" and, in the curation summary, "Use iscell flag from suite2p (manual curation). No additional filtering."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The per-trial neural matrices are aligned to trial start simply by slicing the dF/F array from `trial_start` to `teleport`. Since trial start is the left edge of every trial slice, no extra shift is applied.

ii.
```python
for ti in range(n_trials):
    ts, te = trial_start_inds[ti], teleport_inds[ti]
    trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. The notes say "Trial = track only: trial_start to teleport (excludes teleport zone)" and describe the temporal alignment event as `trial_start (entry to linear track)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native frame rate and does not rebin. It estimates an `effective_rate` from the position timestamps for each session, but the exported metadata uses a fixed target rate of 15.5078125 Hz, corresponding to about 64.48 ms bins.

ii.
```python
TARGET_RATE_HZ = 15.5078125
...
dt = np.median(np.diff(frame_timestamps))
effective_rate = 1.0 / dt
...
time_bin_ms = 1000.0 / TARGET_RATE_HZ
```

iii. The notes justify this with "No downsampling: Multi-plane data already at ~15.5 Hz" and "Multi-plane sessions ... have data already at per-plane rate (~15.5 Hz)."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the position-series timestamps, which the AI treats as the behavior frame clock for the session.

ii.
```python
frame_timestamps = behav['position/timestamps'][:]
...
time_from_start = (frame_timestamps[ts:te] - frame_timestamps[ts]).astype(np.float32)
```

iii. The notes say "frame timestamps" map to `input[0]` and emphasize using actual NWB timestamps for alignment.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI subtracts the timestamp of the trial's first frame so time starts at 0 for that trial. The result is stored as a time-varying float input.

ii.
```python
time_from_start = (frame_timestamps[ts:te] - frame_timestamps[ts]).astype(np.float32)
...
inp[0] = time_from_start
```

iii. The notes describe this variable as "Time from trial start (s)" and the code implements exactly that subtraction.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The AI aligns time to neural activity by using the same trial start and end indices for both the neural slice and the timestamp slice. There is no interpolation or resampling.

ii.
```python
trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
time_from_start = (frame_timestamps[ts:te] - frame_timestamps[ts]).astype(np.float32)
```

iii. The notes repeatedly state that the behavior timestamps already match the imaging rate and that no downsampling is needed.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavioral time series.

ii.
```python
environment = behav['environment/data'][:]
...
env_vals = environment[ts:te]
```

iii. The notes map `environment` directly to `input[1]` and interpret it as `ENV1=0`, `ENV2=1`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI converts the per-frame environment signal to one per-trial label by taking the median of the nonnegative values within each trial, rounding to int, and then repeating that constant value across all time points in the trial.

ii.
```python
trial_env = np.zeros(n_trials, dtype=np.int64)
for ti in range(n_trials):
    ts, te = trial_start_inds[ti], teleport_inds[ti]
    env_vals = environment[ts:te]
    valid = env_vals[env_vals >= 0]
    if len(valid) > 0:
        trial_env[ti] = int(np.round(np.median(valid)))
```

```python
inp[1] = trial_env[ti]
```

iii. The notes say environment is binary and constant within a trial, so collapsing to a trial-level label is the AI's way of making that assumption explicit while ignoring any invalid negative values.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The AI does not use a raw NWB `trial number` stream. It derives trial number from the loop index over the detected trial boundaries.

ii.
```python
for ti in range(n_trials):
    ts, te = trial_start_inds[ti], teleport_inds[ti]
```

```python
inp[2] = ti  # trial number
```

iii. The notes map `trial index` to `input[2]` and explicitly describe it as a 0-indexed trial number.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial index `ti` is written as a constant value across all time bins within a trial.

ii.
```python
inp = np.zeros((4, nf), dtype=np.float32)
...
inp[2] = ti  # trial number
```

iii. The notes justify this by treating trial number as a per-trial contextual variable rather than a behavior stream.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived indirectly from the per-trial `isreward` label, which the AI computes from `Reward/timestamps` together with whether `reward_zone_raw > 0` occurs during that trial.

ii.
```python
reward_zone_raw = behav['reward_zone/data'][:]
reward_timestamps = behav['Reward/timestamps'][:]
...
has_rzone = np.any(reward_zone_raw[ts:te] > 0)
has_reward = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
if has_reward and has_rzone:
    isreward[ti] = 1
```

```python
prev_outcome = int(isreward[ti-1]) if ti > 0 else 0
```

iii. The notes justify this as matching the reference code's notion of reward detection and say "Previous trial outcome: First trial defaults to 0."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. After computing a binary reward/no-reward label for each trial, the AI shifts that vector by one trial. Trial 0 gets `0`, and every later trial gets the previous trial's label, repeated across time bins.

ii.
```python
prev_outcome = int(isreward[ti-1]) if ti > 0 else 0
...
inp[3] = prev_outcome
```

iii. The notes explicitly state "First trial defaults to 0" and frame the variable as a binary per-trial context value.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The AI derives it from the per-trial position samples plus reward-zone coordinates inferred from the session `identifier`/scene name and the fixed switch trial. It does not derive the reward-zone identity from the noisy `reward_zone` behavioral stream.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene_name(identifier)
...
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
```

```python
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
rz_s, rz_e = rz_coords[ti]
dist = np.where(trial_pos < rz_s, trial_pos - rz_s,
               np.where(trial_pos > rz_e, trial_pos - rz_e, 0.0))
```

iii. The notes justify this by citing the reference code's reward-zone mapping from scene names, and by stating that the NWB `reward_zone` field looked like a cumulative count rather than a clean zone label.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI clips position to `[0, 450]` cm, then computes signed distance to the current trial's reward-zone interval: negative before the zone, zero inside, positive after.

ii.
```python
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
...
dist = np.where(trial_pos < rz_s, trial_pos - rz_s,
               np.where(trial_pos > rz_e, trial_pos - rz_e, 0.0))
```

iii. The notes describe this target as "Signed distance" from position to the active reward zone, matching the decoder specification.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is discretized into the 7 bins specified in the instructions using explicit boolean comparisons.

ii.
```python
def discretize_distance(d):
    r = np.zeros_like(d, dtype=np.int64)
    r[d < -50] = 0
    r[(d >= -50) & (d < -10)] = 1
    r[(d >= -10) & (d < 0)] = 2
    r[d == 0] = 3
    r[(d > 0) & (d <= 10)] = 4
    r[(d > 10) & (d <= 50)] = 5
    r[d > 50] = 6
```

iii. The notes say the distance output uses the instructed 7-bin discretization.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing the position-based distance on the exact same `[ts:te]` trial slice used for the neural data.

ii.
```python
trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
...
out[0] = dist_disc
```

iii. The notes say the trial is the common alignment unit for neural and behavioral signals.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavioral time series.

ii.
```python
position = behav['position/data'][:]
...
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
```

iii. The notes map `position` directly to the absolute-position output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI clips position into the legal 0 to 450 cm track range and then discretizes that clipped signal.

ii.
```python
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
pos_disc = discretize_position(trial_pos)
```

iii. The notes describe the output as 5 bins across the 450 cm track. Clipping is the AI's way of handling slight out-of-range values before binning.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into 5 bins with boundaries at 90, 180, 270, and 360 cm.

ii.
```python
def discretize_position(p):
    r = np.zeros_like(p, dtype=np.int64)
    r[p < 90] = 0
    r[(p >= 90) & (p < 180)] = 1
    r[(p >= 180) & (p < 270)] = 2
    r[(p >= 270) & (p < 360)] = 3
    r[p >= 360] = 4
```

iii. The notes say this is "5 bins of 90 cm" across the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same per-trial slice as the neural data, so every neural time bin gets the corresponding position bin from the same indices.

ii.
```python
trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
out[1] = pos_disc
```

iii. The notes rely on the shared behavior/imaging frame clock and trial boundaries for alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavioral time series.

ii.
```python
lick = behav['lick/data'][:]
...
trial_lick = lick[ts:te]
```

iii. The notes map `lick` directly to the lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes the lick signal at `> 0`.

ii.
```python
lick_bin = (trial_lick > 0).astype(np.int64)
...
out[3] = lick_bin
```

iii. The notes describe this output as binary `yes/no`, so thresholding positive values to 1 is the stated choice.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by slicing `lick` with the same trial indices used for neural data.

ii.
```python
trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
trial_lick = lick[ts:te]
out[3] = lick_bin
```

iii. The notes assume all behavior streams are already sampled on the same frame clock used for the trial split.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The AI derives reward-zone location from the NWB `identifier` string via the parsed scene name, plus the fixed switch-at-trial-30 rule. It does not use the raw `reward_zone` stream to identify whether the active zone is A, B, or C.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene_name(identifier)
...
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
```

```python
rz_label_int = {'A': 0, 'B': 1, 'C': 2}.get(rz_labels[ti], 0)
```

iii. The notes justify this by citing the reference `get_reward_zones()` logic and by saying the NWB `reward_zone` values were not a clean zone label.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the scene string into pre-switch and post-switch reward locations, applies the switch at trial 30 when needed, then maps zone letters to integers `A=0`, `B=1`, `C=2` and repeats the label across the whole trial.

ii.
```python
if '_to_Env' in scene:
    parts = scene.split('_to_')
    pre_zone = parts[0][-1]
    post_zone = parts[1][-1]
    ct = min(change_trial, n_trials)
    rz_coords[:ct] = zone_to_coords[pre_zone]; rz_labels[:ct] = pre_zone
    rz_coords[ct:] = zone_to_coords[post_zone]; rz_labels[ct:] = post_zone
```

```python
rz_label_int = {'A': 0, 'B': 1, 'C': 2}.get(rz_labels[ti], 0)
...
out[4] = rz_label_int
```

iii. The notes say reward-zone labels should switch at trial 30 and report that this matched their sanity checks.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The AI derives reward outcome from `Reward/timestamps` and the raw `reward_zone` activity inside the trial. The final per-trial label is taken from the intermediate `isreward` array.

ii.
```python
reward_zone_raw = behav['reward_zone/data'][:]
reward_timestamps = behav['Reward/timestamps'][:]
...
has_rzone = np.any(reward_zone_raw[ts:te] > 0)
has_reward = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
if has_reward and has_rzone:
    isreward[ti] = 1
```

iii. The notes justify this as matching the paper code's reward-detection logic and explicitly say "reward events + rzone" map to `output[5]`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI checks whether any reward timestamp falls between the trial start and end timestamps and whether reward-zone occupancy was present. That per-trial binary value is then written as a constant across all time bins in the trial.

ii.
```python
for ti in range(n_trials):
    ts, te = trial_start_inds[ti], teleport_inds[ti]
    has_rzone = np.any(reward_zone_raw[ts:te] > 0)
    t_start = frame_timestamps[ts]
    t_end = frame_timestamps[min(te, n_timepoints-1)]
    has_reward = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
    if has_reward and has_rzone:
        isreward[ti] = 1
```

```python
out[5] = isreward[ti]
```

iii. The notes say this uses actual NWB timestamps for reward detection and was chosen to match the reference behavior logic.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles a few edge cases but only lightly: it truncates mismatched numbers of trial starts and teleports with `min(...)`, ignores negative environment values when estimating trial environment, skips trials shorter than 2 frames, replaces NaN/Inf neural values with 0, and drops sessions with fewer than 2 trials. It does not implement the reference solution's neural/behavior cropping or explicit reward-time alignment assertions.

ii.
```python
n_trials = min(len(trial_start_inds), len(teleport_inds))
trial_start_inds = trial_start_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]
```

```python
valid = env_vals[env_vals >= 0]
if len(valid) > 0:
    trial_env[ti] = int(np.round(np.median(valid)))
...
if nf < 2: continue
...
trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. The notes call out two Inf-producing trials that were zeroed after baseline division and say no extremely short trials were encountered in practice.

## 13-a. What are the most time-consuming steps of the code?

i. The AI's code is structured so that the heaviest steps are likely per-session NWB I/O, the dF/F computation in `compute_dff`, and final pickle writing. The script prints timing per session and separately times the dF/F block.

ii.
```python
t_dff = time.time()
dff = compute_dff(F, Fneu, trial_start_inds, teleport_inds)
print(f"    dF/F: {time.time()-t_dff:.1f}s, shape={dff.shape}")
```

```python
t0 = time.time()
sess = process_session(nwb_path, show_processing=args.show_processing and i < 2)
...
print(f"    Time: {dt:.1f}s, Est remaining: {rem/60:.1f}min")
...
pickle.dump(data, pf, protocol=4)
```

iii. The notes summarize runtime as about 2 s/session and about 4.5 minutes for the full run, implying the AI considered performance primarily in terms of session I/O plus signal processing time.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain scalar/per-trial: the baseline loops in `compute_dff`, the reward-outcome loop, the environment-summary loop, and the main trial-construction loop. These are natural candidates for vectorization or consolidation.

ii.
```python
for s, e in zip(trial_start_inds, teleport_inds):
    if s < e: nanmask[s:e] = True
...
for s, e in zip(trial_start_inds, teleport_inds):
    if s >= e: continue
    flow[:, s:e] = scipy.ndimage.minimum_filter1d(f_[:, s:e], w, axis=-1)
```

```python
for ti in range(n_trials):
    ...
for ti in range(n_trials):
    ...
for ti in range(n_trials):
    ...
```

iii. The notes say the code runs quickly enough already, but the structure shows repeated per-trial passes rather than a more fused vectorized implementation.

## 13-c. What processing does the code repeat multiple times?

i. Within each session, the AI repeats multiple passes over the same trial list: once to build the dF/F mask and baseline, once to compute reward outcomes, once to compute environment labels, and once to build neural/input/output trial arrays. It does not repeat a full survey pass over all files the way the human reference solution does.

ii.
```python
for s, e in zip(trial_start_inds, teleport_inds):
    ...
for s, e in zip(trial_start_inds, teleport_inds):
    ...
for s, e in zip(trial_start_inds, teleport_inds):
    ...
```

```python
for ti in range(n_trials):
    ...
for ti in range(n_trials):
    ...
for ti in range(n_trials):
    ...
```

iii. The notes emphasize that the full conversion is already fairly fast, so the AI did not try to fuse these repeated passes further.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and keeps several session-local values that are not saved into the final dataset: `session_id`, `scene`, `n_trials`, and `effective_rate` are returned from `process_session` but dropped when constructing `data`. Optional plotting also recreates derived quantities purely for visualization. Session-level summaries of output distributions are printed and not saved.

ii.
```python
return {
    'neural_trials': neural_trials, 'input_trials': input_trials,
    'output_trials': output_trials, 'subject_id': subject_id,
    'session_id': session_id, 'scene': scene, 'n_cells': n_cells,
    'n_trials': len(neural_trials), 'effective_rate': effective_rate,
}
```

```python
for s in all_sessions:
    neural.append(s['neural_trials'])
    inputs.append(s['input_trials'])
    outputs.append(s['output_trials'])
    subject_idx.append(sub2idx[s['subject_id']])
    br_idx.append(np.zeros(s['n_cells'], dtype=np.int64))
```

iii. The notes present these computations as validation and debugging aids rather than part of the final exported representation.
