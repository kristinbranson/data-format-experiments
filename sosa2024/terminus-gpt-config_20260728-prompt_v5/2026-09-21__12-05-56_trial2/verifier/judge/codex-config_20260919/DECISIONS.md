# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script recursively enumerates every NWB file one directory below `/app/data`, sorts the paths, and opens each file with `pynwb.NWBHDF5IO`. It eagerly reads the relevant ophys and behavioral arrays into NumPy arrays. `--sample` limits this to the first two files; otherwise `--full` is effectively a no-op and all files are used.

ii.
```python
files = sorted(Path('/app/data').glob('sub-*/*.nwb'))
if args.sample:
    files = files[:2]
...
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read()
```

iii. The notes say the data consist of 11 subject directories and 152 NWB sessions, with one NWB file per imaging/behavior session. They justify NWB loading from the observed `behavior` and `ophys` modules and report that the full conversion retained all 152 files.

## 1-b. How are the data split into subjects?

i. A subject is the name of an NWB file's parent directory (for example, `sub-m3`). Subjects are registered on first encounter and each retained session receives the corresponding integer `subject_idx`.

ii.
```python
'subject': path.parent.name,
...
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
subject_idx.append(subject_to_idx[subj])
```

iii. The notes identify directories matching `/app/data/sub-m*/` as subjects and verify that this produces 11 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. A session is appended only if it yields at least two trials.

ii.
```python
for path in files:
    sess = load_nwb_session(path)
    neural_trials, input_trials, output_trials, plane_idx, zone_centers, zone_labels = build_trials(sess)
    if len(neural_trials) < 2:
        continue
    all_neural.append(neural_trials)
```

iii. The notes state that each file “appears to contain one imaging/behavior session” and report 152 converted sessions.

## 1-d. How are the data split into trials?

i. Trials are grouped by nonnegative values of the raw `trial number` series, restricted to scanning frames with finite position and speed. A trial must have an explicit `trial_start` pulse to enter the reward lookup, but the construction loop itself accepts every valid trial ID; it starts at the first pulse if present, otherwise at its first valid frame. Trial ends are therefore the last valid frame with that trial number, not teleport onsets.

ii.
```python
valid = (trnum >= 0) & scanning & np.isfinite(pos) & np.isfinite(speed)
trial_ids = np.unique(trnum[valid])
...
for trial in trial_ids:
    m = valid & (trnum == trial)
    idx = np.flatnonzero(m)
    start_candidates = idx[tstart[idx]]
    start_idx = int(start_candidates[0]) if len(start_candidates) else int(idx[0])
    idx = idx[idx >= start_idx]
```

iii. The notes explain that the NWB trials table is empty, so trials were reconstructed from `trial_start` and `trial number`; they also say requiring a start pulse mitigated an off-by-one/extra-trial concern, although the final construction code still has a fallback for trials lacking a pulse.

## 1-e. How are trials filtered based on quality controls?

i. Frames are filtered by nonnegative trial number, active scanning, and finite position and speed. Trials with fewer than two remaining frames are discarded; sessions with fewer than two retained trials are discarded. There is no 50-frame minimum.

ii.
```python
valid = (trnum >= 0) & scanning & np.isfinite(pos) & np.isfinite(speed)
...
if len(idx) < 2:
    continue
...
if len(neural_trials) < 2:
    continue
```

iii. The notes justify excluding invalid pre/post periods via `scanning == 1` and `trial number >= 0`. They do not justify the two-frame cutoff beyond satisfying the decoder’s minimum-session/trial shape constraints.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from the first ROI response series in the NWB `ophys/Deconvolved` container, plus `ImageSegmentation.iscell` for ROI selection. The available `Fluorescence` and `Neuropil` streams are not used.

ii.
```python
deconv = next(iter(ophys['Deconvolved'].roi_response_series.values()))
neural = np.asarray(deconv.data[:], dtype=np.float32)
...
iscell = np.asarray(seg['iscell'].data[:])
```

iii. The notes interpret the Methods phrase “deconvolved activity matrices” as support for using the stored NWB `Deconvolved` stream.

## 2-b. How is the `neural` data processed?

i. No fluorescence correction, dF/F calculation, smoothing, or deconvolution is performed. The stored deconvolved matrix is cast to `float32`, subset by ROI curation and trial indices, and transposed from time-by-ROI to neuron-by-time.

ii.
```python
neural = np.asarray(deconv.data[:], dtype=np.float32)
...
neural = neural[:, iscell]
...
neu = neural[idx].T.astype(np.float32)
```

iii. The notes say the NWB stream is already deconvolved and matches the paper’s stated neural representation, so native framewise values were retained.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are retained when the selected `iscell` value exceeds 0.5. The DynamicTableRegion indices are applied before this filter. No speed-correlation/putative-interneuron exclusion is performed.

ii.
```python
roi_idx = np.asarray(deconv.rois.data[:], dtype=np.int64) if deconv.rois is not None else np.arange(neural.shape[1], dtype=np.int64)
iscell = iscell[roi_idx]
...
iscell = sess['iscell'] > 0.5
neural = neural[:, iscell]
```

iii. The notes describe `iscell` as suite2p-style manual/confidence curation and say the ROI table-region indexing fixed segmentation/response-series mismatches. They do not discuss the paper’s putative-interneuron filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural trial begins at the first indexed frame carrying `trial_start`; when no such pulse exists, it begins at the first valid frame for that trial number. The same integer indices slice neural and behavioral arrays.

ii.
```python
start_candidates = idx[tstart[idx]]
start_idx = int(start_candidates[0]) if len(start_candidates) else int(idx[0])
idx = idx[idx >= start_idx]
...
neu = neural[idx].T.astype(np.float32)
```

iii. The notes say neural and behavior streams have matching frame counts and are already framewise aligned, and identify `trial_start` as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Data remain at native deconvolved timestamps, approximately 15.5 Hz or 64.5 ms per bin. No temporal rebinning is applied. Dataset metadata stores 1000 divided by the median session rate.

ii.
```python
rate = float(np.median(1.0 / np.diff(timestamps)))
...
'time_bin_size': float(1000.0 / np.median([s['native_rate_hz'] for s in session_info]))
```

iii. The notes report matching neural/behavior frame counts at about 15.5 Hz and choose native resolution because the streams appear sample-aligned.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the timestamps of the NWB `Deconvolved` ROI response series, not from behavioral timestamps.

ii.
```python
timestamps = np.asarray(deconv.timestamps[:] if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate), dtype=np.float64)
```

iii. The notes state that behavior and neural data share frame counts/rate and treat the deconvolved timestamps as the common frame clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at `start_idx` is subtracted from every selected frame timestamp and the result is cast to `float32`.

ii.
```python
trial_t = timestamps[idx] - timestamps[start_idx]
...
trial_t.astype(np.float32)
```

iii. This directly implements seconds elapsed from the trial-start alignment event; no additional rationale is given.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same `idx` array selects timestamps and neural rows, so columns correspond by index. The script checks only that deconvolved data length equals deconvolved timestamp length; it does not compare behavioral timestamps with neural timestamps.

ii.
```python
assert neural.shape[0] == len(timestamps)
trial_t = timestamps[idx] - timestamps[start_idx]
neu = neural[idx].T.astype(np.float32)
```

iii. The notes cite representative sessions with identical behavior/neural frame counts as evidence of framewise alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the behavioral `environment` time series.

ii.
```python
env = np.asarray(sess['environment']).astype(int)
trial_env = env[idx]
```

iii. The notes observe valid environment values 0 and 1, matching ENV1 and ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Nonnegative environment samples in a trial are median-aggregated, rounded, converted to an integer, and repeated over all trial frames; if none are valid, 0 is used.

ii.
```python
np.full(len(idx), int(np.round(np.median(trial_env[trial_env >= 0])))
        if np.any(trial_env >= 0) else 0, dtype=np.float32)
```

iii. The notes describe environment as a per-trial binary variable and say valid values can be used directly; aggregation makes that value constant as required.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the raw behavioral `trial number` value used as the trial ID.

ii.
```python
trnum = np.asarray(sess['trial number']).astype(int)
trial_ids = np.unique(trnum[valid])
...
np.full(len(idx), float(trial), dtype=np.float32)
```

iii. The notes choose the raw series because the canonical NWB trial table is empty and use it with `trial_start` to reconstruct trials.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Unique valid raw IDs are sorted by `np.unique`; each ID is converted to float and repeated across its trial. It is not renumbered after filtering.

ii.
```python
for trial in trial_ids:
    ...
    np.full(len(idx), float(trial), dtype=np.float32)
```

iii. The notes call this a framewise-constant continuous per-trial input and do not describe further transformation.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the timestamps of the behavioral `Reward` series and the deconvolved frame timestamps/derived trial intervals.

ii.
```python
reward_times = np.asarray(sess['Reward_t'], dtype=np.float64)
...
trial_reward[trial] = int(np.any((reward_times >= t0) & (reward_times <= t1)))
```

iii. The notes define reward outcome as whether any reward event occurs during a trial and then shift that binary result by one trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward events are tested against each valid trial’s inclusive first-to-last timestamp. For trial ID `n`, the lookup uses ID `n-1`, defaulting to 0 when absent, and repeats the result across time.

ii.
```python
prev_outcome = trial_reward.get(trial - 1, 0)
...
np.full(len(idx), float(prev_outcome), dtype=np.float32)
```

iii. The notes say the first trial defaults to omitted (0) and the per-trial reward outcome is shifted to form previous outcome.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from behavioral `position` and `reward_zone`. The trial’s inferred zone center is the mean position at frames where `reward_zone > 0`; if none exist, the median valid position is snapped to one of `[85, 205, 325]` cm.

ii.
```python
m = reward_zone_trial > 0
if np.any(m):
    center = float(np.nanmean(pos_trial[m]))
else:
    center = float(np.nanmedian(valid)) if len(valid) else 205.0
    center = float(CANONICAL_ZONE_CENTERS[np.argmin(np.abs(CANONICAL_ZONE_CENTERS - center))])
```

iii. The notes say reward-zone-positive positions empirically cluster near three canonical locations and justify inferring zone location from those samples.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The inferred scalar center is subtracted from position, producing center-relative signed distance. Thus distance is zero only exactly at the inferred center, rather than throughout the reward-zone interval.

ii.
```python
center = infer_zone_center(trial_pos, trial_rz)
dist = trial_pos - center
```

iii. The notes claim center-relative position better matches the requested reward-relative output; they do not discuss distance to the nearest location/edge within the full zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven Boolean masks implement the requested thresholds. Exactly -10 is assigned category 1, and exactly +10 category 4.

ii.
```python
out[dist < -50] = 0
out[(dist >= -50) & (dist <= -10)] = 1
out[(dist > -10) & (dist < 0)] = 2
out[dist == 0] = 3
out[(dist > 0) & (dist <= 10)] = 4
out[(dist > 10) & (dist <= 50)] = 5
out[dist > 50] = 6
```

iii. The notes say outputs are discretized into the seven requested bins. They do not acknowledge that the stated `-50 to -10` and `-10 to <0` labels overlap at -10; the code resolves it into the former.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position, reward-zone samples, timestamps, and neural rows are all indexed with the same trial `idx`; the discretized distance therefore has one value per neural column.

ii.
```python
trial_pos = pos[idx]
trial_rz = reward_zone[idx]
...
neu = neural[idx].T.astype(np.float32)
```

iii. The notes rely on the observed samplewise alignment of behavioral and neural arrays.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position` series.

ii.
```python
pos = np.asarray(sess['position'], dtype=np.float32)
trial_pos = pos[idx]
```

iii. The notes identify position as the animal’s coordinate on the 450 cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Trial position is clipped to `[0, 450]` before categorization. With open-ended first/last categories this clipping does not change category assignments for finite out-of-range values.

ii.
```python
discretize_position(np.clip(trial_pos, 0, 450))
```

iii. The notes cite the 450 cm corridor and the requested five equal-width position bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five masks implement cut points at 90, 180, 270, and 360 cm.

ii.
```python
out[pos < 90] = 0
out[(pos >= 90) & (pos < 180)] = 1
out[(pos >= 180) & (pos < 270)] = 2
out[(pos >= 270) & (pos < 360)] = 3
out[pos >= 360] = 4
```

iii. The notes explain that five equal bins across 450 cm are 90 cm wide.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural activity are sliced by the same `idx` array.

ii.
```python
trial_pos = pos[idx]
neu = neural[idx].T.astype(np.float32)
```

iii. The notes state the streams are framewise aligned at native resolution.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii.
```python
lick = (np.asarray(sess['lick']) > 0).astype(np.int64)
```

iii. The notes map the paper’s licking behavior stream to the requested lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Values greater than zero become 1 and all others become 0; the per-trial slice is retained as an integer time series.

ii.
```python
lick = (np.asarray(sess['lick']) > 0).astype(np.int64)
trial_lick = lick[idx]
```

iii. The notes specify `lick > 0` because the requested output is binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural arrays use the same trial indices.

ii.
```python
trial_lick = lick[idx]
neu = neural[idx].T.astype(np.float32)
```

iii. The notes rely on native framewise behavioral/neural alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It uses behavioral `reward_zone` and `position`, via the same inferred center described in 7-a.

ii.
```python
center = infer_zone_center(trial_pos, trial_rz)
zlabel = zone_label_from_center(center)
```

iii. The notes say positions with `reward_zone > 0` cluster around three reward locations.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The inferred center is assigned to whichever of 85, 205, or 325 cm is closest; labels 0, 1, and 2 represent A, B, and C and are repeated over time. Missing zone data are imputed from the trial’s median position before this nearest-center assignment.

ii.
```python
def zone_label_from_center(center):
    return int(np.argmin(np.abs(CANONICAL_ZONE_CENTERS - center)))
...
np.full(len(idx), zlabel, dtype=np.int64)
```

iii. The notes justify nearest-center mapping from exploratory clustering and report the resulting label distributions as a sanity check.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from timestamps of the behavioral `Reward` time series and deconvolved timestamps delimiting each reconstructed trial.

ii.
```python
reward_times = np.asarray(sess['Reward_t'], dtype=np.float64)
trial_reward[trial] = int(np.any((reward_times >= t0) & (reward_times <= t1)))
```

iii. The notes interpret a Reward event within the trial as a rewarded outcome and report one-to-one spot checks.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The code checks whether any reward timestamp lies inclusively between the first and last valid timestamp of each start-marked trial. Missing entries default to 0; the result is repeated across time.

ii.
```python
this_outcome = trial_reward.get(trial, 0)
...
np.full(len(idx), this_outcome, dtype=np.int64)
```

iii. The notes describe reward outcome as a binary per-trial variable and say rewarded-trial counts matched Reward events in spot checks.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. ROI table-region mismatches are handled by applying `deconv.rois` indices. Invalid frames are dropped using trial/scanning/finite-value masks. Trials under two samples and sessions under two trials are skipped. Missing environment defaults to 0; missing previous/current rewards default to 0; missing reward-zone samples are imputed from median trial position and snapped to a canonical center. Neural/timestamp length mismatch raises an assertion rather than being repaired, and behavioral timestamps are not checked.

ii.
```python
roi_idx = np.asarray(deconv.rois.data[:], dtype=np.int64) if deconv.rois is not None else np.arange(neural.shape[1], dtype=np.int64)
...
valid = (trnum >= 0) & scanning & np.isfinite(pos) & np.isfinite(speed)
...
center = float(np.nanmedian(valid)) if len(valid) else 205.0
...
assert neural.shape[0] == len(timestamps)
```

iii. The notes document the ROI mismatch fix and describe scanning/nonnegative-trial filtering as exclusion of invalid periods. They call the inferred-zone approach empirically supported, but provide no missing-zone validation and do not document several silent defaults.

## 13-a. What are the most time-consuming steps of the code?

i. The likely dominant steps are eager NWB reads of large deconvolved and behavioral arrays, copying/slicing neural matrices into thousands of trial arrays, and serializing the roughly 8.4 GB pickle. The script records per-session total conversion time but no stage-level profile.

ii.
```python
neural = np.asarray(deconv.data[:], dtype=np.float32)
...
neu = neural[idx].T.astype(np.float32)
...
pickle.dump(data, f)
```

iii. The notes explicitly identify eager full-session loading as the main inefficiency and report the output size, but do not provide timing measurements by operation.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Both loops over `trial_ids` repeatedly build masks and scan reward timestamps. Per-trial reward assignment could be vectorized or obtained by mapping rewards to trial IDs once; some per-frame categorization could be done session-wide before slicing. Trial assembly itself naturally remains a loop because trials have variable lengths.

ii.
```python
for trial in trial_ids:
    m = valid & (trnum == trial)
    ...
for trial in trial_ids:
    m = valid & (trnum == trial)
```

iii. The notes say vectorized NumPy was used “where possible” and do not identify these remaining loops specifically.

## 13-c. What processing does the code repeat multiple times?

i. For every trial, the script constructs `valid & (trnum == trial)` and extracts indices once to compute reward outcome and again to construct outputs. It also reads and stores timestamps for every behavioral stream even though most stream-specific timestamps are never used.

ii.
```python
for trial in trial_ids:
    m = valid & (trnum == trial)
    idx = np.flatnonzero(m)
...
for trial in trial_ids:
    m = valid & (trnum == trial)
    idx = np.flatnonzero(m)
```

iii. The agent’s notes do not discuss repeated processing; unlike the reference pipeline, this script does not run a separate survey pass during conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `load_nwb_session` reads data and timestamps for `autoreward`, `teleport`, and several other series whose stream-specific timestamps are never used. `plane_idx` is filtered and returned but not used to form brain-region indices. `zone_centers` are retained only to compute one metadata median, and the CLI’s `--full` and `--show-processing` flags have no effect.

ii.
```python
for key in ['Reward', 'autoreward', ..., 'teleport', ...]:
    out[key] = np.asarray(ts.data[:])
    out[key + '_t'] = np.asarray(ts.timestamps[:]) ...
...
neural_trials, input_trials, output_trials, plane_idx, zone_centers, zone_labels = build_trials(sess)
```

iii. The notes mention eager loading generally but do not identify these discarded values or inert options.
