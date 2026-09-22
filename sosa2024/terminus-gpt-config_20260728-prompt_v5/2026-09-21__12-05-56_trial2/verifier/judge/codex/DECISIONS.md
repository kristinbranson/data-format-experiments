# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every NWB file matching `/app/data/sub-*/*.nwb`, treats each file as one session, and reads both the `behavior` and `ophys` processing groups with `pynwb.NWBHDF5IO`.

ii.
```python
def load_nwb_session(path):
    with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
        nwb = io.read()
        beh = nwb.processing['behavior']['BehavioralTimeSeries'].time_series
        ophys = nwb.processing['ophys']
```
```python
files = sorted(Path('/app/data').glob('sub-*/*.nwb'))
data = convert_dataset(files)
```

iii. In `CONVERSION_NOTES.md` Step 2 and Step 5, the AI justified this by noting that the data are organized as one NWB file per imaging/behavior session under subject directories, and that the NWB files contain the needed behavior and ophys streams.

## 1-b. How are the data split into subjects?

i. Subjects are split by NWB parent directory. The stored subject identifier is the directory name such as `sub-m12`, and a unique subject list is built as sessions are processed.

ii.
```python
out = {
    'subject': path.parent.name,
    'session': path.stem,
    ...
}
```
```python
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
subject_idx.append(subject_to_idx[subj])
```

iii. The AI's notes say subjects correspond to the NWB subdirectories under `/app/data`, which matched the observed inventory of 11 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session identifier is the file stem.

ii.
```python
out = {
    'subject': path.parent.name,
    'session': path.stem,
    ...
}
```
```python
for path in files:
    sess = load_nwb_session(path)
```

iii. `CONVERSION_NOTES.md` Step 2 states that each file appears to contain one imaging/behavior session, so the AI used file-level session boundaries.

## 1-d. How are the data split into trials?

i. Trials are primarily grouped by the raw `trial number` stream after filtering to valid frames. Within each grouped trial, the AI further aligns the start to the first `trial_start` pulse found inside that trial. It does not use `teleport` to define trial ends.

ii.
```python
trnum = np.asarray(sess['trial number']).astype(int)
tstart = np.asarray(sess['trial_start']) > 0
valid = (trnum >= 0) & scanning & np.isfinite(pos) & np.isfinite(speed)
trial_ids = np.unique(trnum[valid])
start_trial_ids = set(trnum[tstart & (trnum >= 0)].tolist())
```
```python
for trial in trial_ids:
    m = valid & (trnum == trial)
    idx = np.flatnonzero(m)
    ...
    start_candidates = idx[tstart[idx]]
    start_idx = int(start_candidates[0]) if len(start_candidates) else int(idx[0])
    idx = idx[idx >= start_idx]
```

iii. The AI's notes say canonical NWB trial tables were empty, so trial structure had to be reconstructed from behavioral time series. In trajectory steps 34-37 it also noted that it wanted trial starts to come from `trial_start`, but ultimately used `trial number` plus a `trial_start` check after sample-validation debugging.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials with at least two valid samples, excludes trials without a matching `trial_start` pulse when computing reward labels, restricts frames to `trial number >= 0`, `scanning > 0`, and finite `position` and `speed`, and drops whole sessions with fewer than two kept trials.

ii.
```python
valid = (trnum >= 0) & scanning & np.isfinite(pos) & np.isfinite(speed)
trial_ids = np.unique(trnum[valid])
start_trial_ids = set(trnum[tstart & (trnum >= 0)].tolist())
```
```python
if len(idx) < 2:
    continue
...
if len(neural_trials) < 2:
    continue
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI justified valid-frame filtering as a way to avoid invalid pre/post-trial periods. In trajectory step 37 it also tightened trial handling after finding an extra trial in the sample output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` directly from the NWB `ophys/Deconvolved` ROI response series, after subsetting to the ROI table-region referenced by that series.

ii.
```python
deconv = next(iter(ophys['Deconvolved'].roi_response_series.values()))
neural = np.asarray(deconv.data[:], dtype=np.float32)  # time x roi
roi_idx = np.asarray(deconv.rois.data[:], dtype=np.int64) if deconv.rois is not None else np.arange(neural.shape[1], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` Step 4 and Step 5 explicitly state that the methods mention deconvolved activity matrices, so the AI chose the stored deconvolved signal instead of recomputing from `Fluorescence` and `Neuropil`.

## 2-b. How is the `neural` data processed?

i. Processing is limited to selecting the referenced ROIs, thresholding `iscell`, transposing trial slices to neuron-by-time, and otherwise keeping the NWB deconvolved values at native resolution. No recomputation of dF/F or deconvolution is done.

ii.
```python
iscell = sess['iscell'] > 0.5
neural = neural[:, iscell]
...
neu = neural[idx].T.astype(np.float32)
```

iii. The AI justified this in `CONVERSION_NOTES.md` by reading the methods phrase "deconvolved activity matrices" and concluding the stored `Deconvolved` series matched the analysis target.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are filtered only by `iscell > 0.5`. The AI also fixes ROI-table mismatches by restricting the segmentation arrays to the ROIs referenced by the deconvolved series. It does not remove putative interneurons.

ii.
```python
iscell = np.asarray(seg['iscell'].data[:])
...
iscell = np.asarray(iscell, dtype=np.float32).reshape(-1)
...
roi_idx = np.asarray(deconv.rois.data[:], dtype=np.int64) if deconv.rois is not None else np.arange(neural.shape[1], dtype=np.int64)
iscell = iscell[roi_idx]
```
```python
iscell = sess['iscell'] > 0.5
neural = neural[:, iscell]
```

iii. `CONVERSION_NOTES.md` Step 3 and Step 5 say the NWB `iscell` field looked like suite2p-style confidence data, so thresholding at `0.5` was chosen as neuron curation. Trajectory steps 36 and 40 show additional debugging around the shape of `iscell` and ROI table-region indexing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by resetting time to the first `trial_start` pulse inside each grouped trial and keeping only samples at or after that pulse.

ii.
```python
start_candidates = idx[tstart[idx]]
start_idx = int(start_candidates[0]) if len(start_candidates) else int(idx[0])
idx = idx[idx >= start_idx]
trial_t = timestamps[idx] - timestamps[start_idx]
...
neu = neural[idx].T.astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI said the decoder should be aligned to trial start and that `trial_start > 0` marks that event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native sample rate from the deconvolved timestamps and does not rebin or resample.

ii.
```python
timestamps = np.asarray(
    deconv.timestamps[:] if deconv.timestamps is not None
    else np.arange(neural.shape[0]) / float(deconv.rate),
    dtype=np.float64
)
rate = float(np.median(1.0 / np.diff(timestamps))) if len(timestamps) > 1 else float(getattr(deconv, 'rate', np.nan))
```
```python
'time_bin_size': float(1000.0 / np.median([s['native_rate_hz'] for s in session_info])) if session_info else None,
```

iii. `CONVERSION_NOTES.md` Step 5 says neural and behavioral streams are already frame-aligned at about 15.5 Hz, so native frame bins were retained.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the deconvolved neural timestamps, using the trial start index as the zero point.

ii.
```python
timestamps = sess['timestamps']
...
trial_t = timestamps[idx] - timestamps[start_idx]
```

iii. The AI justified this in `CONVERSION_NOTES.md` Step 5 by saying the neural and behavior streams were already aligned sample-by-sample, so frame timestamps could define elapsed trial time directly.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each kept trial, the timestamp at the aligned start sample is subtracted from all timestamps in that trial.

ii.
```python
trial_t = timestamps[idx] - timestamps[start_idx]
...
inp = np.vstack([
    trial_t.astype(np.float32),
    ...
])
```

iii. No separate justification beyond alignment to trial start appears in the notes; this is the direct implementation of that alignment.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The AI uses the exact same `idx` array to slice timestamps and neural activity, so the time input is frame-aligned with the neural matrix within each trial.

ii.
```python
trial_t = timestamps[idx] - timestamps[start_idx]
...
neu = neural[idx].T.astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says all streams are already aligned at native frame resolution, so shared indexing is sufficient.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment` time series.

ii.
```python
env = np.asarray(sess['environment']).astype(int)
```

iii. `CONVERSION_NOTES.md` Step 4 says the valid environment values are `0` and `1`, matching the required binary environment input.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the AI takes the median of the nonnegative environment values, rounds it to an integer, and repeats that scalar across all time bins in the trial.

ii.
```python
np.full(
    len(idx),
    int(np.round(np.median(trial_env[trial_env >= 0]))) if np.any(trial_env >= 0) else 0,
    dtype=np.float32
)
```

iii. `CONVERSION_NOTES.md` Step 4 and trajectory step 34 say environment is stable within a trial, so summarizing it to one binary per-trial value was considered safe.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived directly from the raw behavioral `trial number` values for each grouped trial.

ii.
```python
trnum = np.asarray(sess['trial number']).astype(int)
...
np.full(len(idx), float(trial), dtype=np.float32)
```

iii. The AI's notes planned to use `trial number` as the per-trial identifier after reconstructing trials from behavior streams, and the code carries that raw trial label through unchanged.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation is applied beyond repeating the grouped trial label across the time bins of that trial.

ii.
```python
inp = np.vstack([
    ...,
    np.full(len(idx), float(trial), dtype=np.float32),
    ...
])
```

iii. No separate justification was given beyond treating the raw trial label as the trial number input.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward event timestamps from the behavioral `Reward` time series, converted into a per-trial reward dictionary.

ii.
```python
reward_times = np.asarray(sess['Reward_t'], dtype=np.float64)
...
trial_reward[trial] = int(np.any((reward_times >= t0) & (reward_times <= t1)))
```

iii. In trajectory step 34, the AI notes that reward events map cleanly to rewarded trials, so previous trial outcome can be obtained by shifting per-trial reward outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the AI marks reward outcome as `1` if any `Reward` timestamp falls between the first and last timestamps of that trial. The input value for the current trial is then `trial_reward[trial - 1]`, defaulting to `0`.

ii.
```python
for trial in trial_ids:
    ...
    t0 = timestamps[idx[0]]
    t1 = timestamps[idx[-1]]
    trial_reward[trial] = int(np.any((reward_times >= t0) & (reward_times <= t1)))
```
```python
prev_outcome = trial_reward.get(trial - 1, 0)
...
np.full(len(idx), float(prev_outcome), dtype=np.float32),
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly describes previous reward outcome as a one-trial shift of the derived per-trial reward outcome, with the first trial defaulting to `0`.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the behavioral `position` and `reward_zone` time series. The AI infers one reward-zone center for each trial from the positions where `reward_zone > 0`.

ii.
```python
pos = np.asarray(sess['position'], dtype=np.float32)
reward_zone = np.asarray(sess['reward_zone']).astype(int)
```
```python
def infer_zone_center(pos_trial, reward_zone_trial):
    m = reward_zone_trial > 0
    if np.any(m):
        center = float(np.nanmean(pos_trial[m]))
```

iii. `CONVERSION_NOTES.md` Step 4 and trajectory step 35 say `reward_zone > 0` clustered near three canonical corridor locations, so the AI treated it as a sparse indicator from which trial-level zone centers could be inferred.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI computes a signed distance from position to the inferred reward-zone center, not to the nearest zone edge. If `reward_zone > 0` never occurs in a trial, it falls back to the nearest canonical center based on median position.

ii.
```python
center = infer_zone_center(trial_pos, trial_rz)
dist = trial_pos - center
```
```python
valid = pos_trial[(pos_trial >= 0) & (pos_trial <= 450)]
center = float(np.nanmedian(valid)) if len(valid) else 205.0
center = float(CANONICAL_ZONE_CENTERS[np.argmin(np.abs(CANONICAL_ZONE_CENTERS - center))])
```

iii. The AI justified this in `CONVERSION_NOTES.md` Step 5 by saying that position relative to inferred zone center better matched the requested reward-relative decoder output than the raw `reward_zone` codes.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI discretizes the center-relative distance into seven bins using hard threshold comparisons.

ii.
```python
def discretize_distance(dist):
    out = np.full(dist.shape, -1, dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist <= -10)] = 1
    out[(dist > -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out
```

iii. `CONVERSION_NOTES.md` Step 5 says the bin edges were chosen to match the decoder task specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Distance-to-zone is computed from the same per-trial frame indices used to extract the neural data, so it is aligned sample-by-sample with the neural matrix.

ii.
```python
trial_pos = pos[idx]
...
dist = trial_pos - center
...
neu = neural[idx].T.astype(np.float32)
```

iii. The AI's general justification, stated in `CONVERSION_NOTES.md` Step 5, is that all streams are already frame-aligned in the NWB files.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position` time series.

ii.
```python
pos = np.asarray(sess['position'], dtype=np.float32)
...
trial_pos = pos[idx]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `position` directly to the absolute-position output on the 450 cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices position by trial, clips values into `[0, 450]`, then discretizes the clipped values.

ii.
```python
out = np.vstack([
    ...,
    discretize_position(np.clip(trial_pos, 0, 450)),
    ...
])
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI justified this by the 450 cm corridor length used throughout the paper and task.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is discretized into five bins: `<90`, `90-180`, `180-270`, `270-360`, and `>=360`.

ii.
```python
def discretize_position(pos):
    out = np.full(pos.shape, -1, dtype=np.int64)
    out[pos < 90] = 0
    out[(pos >= 90) & (pos < 180)] = 1
    out[(pos >= 180) & (pos < 270)] = 2
    out[(pos >= 270) & (pos < 360)] = 3
    out[pos >= 360] = 4
    return out
```

iii. `CONVERSION_NOTES.md` Step 5 says these bins came directly from the decoder task's equal-width partition of the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The AI slices both position and neural activity with the same trial index vector, so they stay frame-aligned.

ii.
```python
trial_pos = pos[idx]
...
neu = neural[idx].T.astype(np.float32)
```

iii. The AI's justification is the same shared-index alignment used throughout `build_trials`.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii.
```python
lick = (np.asarray(sess['lick']) > 0).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `lick` directly to the decoder lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick values are binarized as `lick > 0`, then trial slices are emitted directly.

ii.
```python
lick = (np.asarray(sess['lick']) > 0).astype(np.int64)
...
trial_lick = lick[idx]
...
trial_lick.astype(np.int64),
```

iii. The AI justified this in `CONVERSION_NOTES.md` Step 5 because the decoder output is binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural activity use the same per-trial frame indices.

ii.
```python
trial_lick = lick[idx]
...
neu = neural[idx].T.astype(np.float32)
```

iii. The justification is the same framewise alignment assumption used for the other behavioral streams.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the behavioral `reward_zone` and `position` time series through trial-wise center inference.

ii.
```python
trial_pos = pos[idx]
trial_rz = reward_zone[idx]
center = infer_zone_center(trial_pos, trial_rz)
```

iii. In `CONVERSION_NOTES.md` Step 4, the AI says the raw `reward_zone` stream is sparse and must be converted into a trial-level A/B/C location by using the associated positions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI computes the mean position of frames where `reward_zone > 0` in a trial, then assigns the nearest canonical center `85`, `205`, or `325` cm and converts that to labels `A/B/C` as `0/1/2`. If no such frames exist, it falls back to the nearest canonical center from the trial's median position.

ii.
```python
def infer_zone_center(pos_trial, reward_zone_trial):
    m = reward_zone_trial > 0
    if np.any(m):
        center = float(np.nanmean(pos_trial[m]))
    else:
        valid = pos_trial[(pos_trial >= 0) & (pos_trial <= 450)]
        center = float(np.nanmedian(valid)) if len(valid) else 205.0
        center = float(CANONICAL_ZONE_CENTERS[np.argmin(np.abs(CANONICAL_ZONE_CENTERS - center))])
    return center
```
```python
def zone_label_from_center(center):
    return int(np.argmin(np.abs(CANONICAL_ZONE_CENTERS - center)))
```

iii. The AI justified this in trajectory step 35 by saying reward-zone-positive positions cluster near three canonical corridor locations and that the raw `reward_zone` codes are not themselves the final A/B/C labels.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from reward event timestamps from the behavioral `Reward` time series.

ii.
```python
reward_times = np.asarray(sess['Reward_t'], dtype=np.float64)
...
trial_reward[trial] = int(np.any((reward_times >= t0) & (reward_times <= t1)))
```

iii. The AI justified this by noting in trajectory step 34 that reward events map cleanly onto rewarded trials.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, reward outcome is `1` if any reward timestamp lies between that trial's first and last timestamps; otherwise `0`. The per-trial value is repeated across the trial's time bins.

ii.
```python
this_outcome = trial_reward.get(trial, 0)
...
np.full(len(idx), this_outcome, dtype=np.int64),
```

iii. `CONVERSION_NOTES.md` Step 5 defines reward outcome exactly this way for the decoder output.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several irregularities: structured or 2D `iscell` arrays are normalized; response-series ROI subsets are respected via `deconv.rois`; invalid frames are removed with `trial number >= 0`, `scanning > 0`, and finite `position`/`speed`; missing `reward_zone > 0` frames trigger a fallback canonical reward-zone center; and missing/invalid environment values fall back to `0`.

ii.
```python
iscell = np.asarray(seg['iscell'].data[:])
if iscell.ndim > 1:
    iscell = iscell[:, 0]
if iscell.dtype.fields is not None:
    first_field = list(iscell.dtype.fields)[0]
    iscell = iscell[first_field]
```
```python
roi_idx = np.asarray(deconv.rois.data[:], dtype=np.int64) if deconv.rois is not None else np.arange(neural.shape[1], dtype=np.int64)
iscell = iscell[roi_idx]
plane_idx = plane_idx[roi_idx]
```
```python
valid = (trnum >= 0) & scanning & np.isfinite(pos) & np.isfinite(speed)
...
if np.any(m):
    center = float(np.nanmean(pos_trial[m]))
else:
    ...
```

iii. `CONVERSION_NOTES.md` Step 4 and trajectory steps 36, 37, and 40 document these as defensive fixes discovered during sample and full conversion, especially the `iscell` shape issue and ROI-table mismatch.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading each NWB session, materializing the large deconvolved matrices and behavioral arrays in memory, looping over every session in `convert_dataset`, and writing the final pickle.

ii.
```python
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read()
```
```python
for path in files:
    t0 = time.time()
    sess = load_nwb_session(path)
    ...
with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. The AI notes in `CONVERSION_NOTES.md` Step 6 that full NWB sessions are loaded eagerly, and the trajectory shows full-conversion runtime was dominated by iterating over 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The two trial loops in `build_trials` are the clearest vectorization targets: one loop builds `trial_reward`, and a second loop rebuilds trial indices to construct outputs. Some per-trial constant features and discretizations could be precomputed session-wide before slicing.

ii.
```python
trial_reward = {}
for trial in trial_ids:
    ...
    trial_reward[trial] = int(np.any((reward_times >= t0) & (reward_times <= t1)))
```
```python
for trial in trial_ids:
    m = valid & (trnum == trial)
    idx = np.flatnonzero(m)
    ...
    out = np.vstack([
        discretize_distance(dist),
        discretize_position(np.clip(trial_pos, 0, 450)),
        discretize_speed(trial_speed),
```

iii. The AI did not explicitly write this analysis in its notes, but `CONVERSION_NOTES.md` Step 6 says vectorized NumPy operations were used "where possible," implying the remaining per-trial loops were accepted for convenience.

## 13-c. What processing does the code repeat multiple times?

i. The code loops over trials twice per session: once to compute `trial_reward` and again to emit per-trial neural/input/output arrays. It also repeatedly computes masks and slices separately for each trial.

ii.
```python
for trial in trial_ids:
    ...
    trial_reward[trial] = int(np.any((reward_times >= t0) & (reward_times <= t1)))
```
```python
for trial in trial_ids:
    m = valid & (trnum == trial)
    idx = np.flatnonzero(m)
    ...
    neural_trials.append(neu)
```

iii. There is no explicit written justification beyond implementation simplicity; the trajectory suggests the AI prioritized getting a working conversion over optimizing away repeated per-trial work.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and stores several items that are not used in downstream decoder construction: `autoreward`, `teleport`, behavior timestamps for every stream, `plane_idx` returned from `build_trials`, and session-level `zone_centers`/`zone_labels` whose only use is metadata logging. It also computes `brain_region_idx` as all zeros without using the plane information it loaded.

ii.
```python
for key in ['Reward', 'autoreward', 'environment', 'lick', 'position', 'reward_zone', 'scanning', 'speed', 'teleport', 'trial number', 'trial_start']:
    ts = beh[key]
    out[key] = np.asarray(ts.data[:])
    out[key + '_t'] = np.asarray(ts.timestamps[:]) if ts.timestamps is not None else timestamps.copy()
```
```python
return neural_trials, input_trials, output_trials, plane_idx, zone_centers, zone_labels
```
```python
brain_region_idx.append(np.zeros(np.sum(sess['iscell'] > 0.5), dtype=np.int64))
```

iii. The AI did not explicitly call these out in `CONVERSION_NOTES.md`; they are inferred from the final script structure.
