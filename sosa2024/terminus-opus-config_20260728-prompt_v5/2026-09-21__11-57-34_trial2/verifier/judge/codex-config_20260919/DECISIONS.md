# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script globs every matching NWB file under every `sub-*` directory, sorts the paths, and processes each file as one session. It uses `h5py` to read NWB datasets directly. Sample mode deliberately reduces this to two named sessions.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/sub-*_ses-*_behavior+ophys.nwb'))
for i, nwb_path in enumerate(nwb_files):
    sess = process_session(nwb_path, show_processing=args.show_processing and i < 2)
    if sess: all_sessions.append(sess)
```

iii. The notes report 152 files, 11 subjects, and 12,216 converted trials, matching the available data. The agent chose direct HDF5 access for speed (about two seconds per session), rather than `pynwb`.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from each NWB file, unique IDs are sorted, and each session receives the corresponding integer `subject_idx`.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode()
subjects = sorted(set(s['subject_id'] for s in all_sessions))
sub2idx = {s: i for i, s in enumerate(subjects)}
subject_idx.append(sub2idx[s['subject_id']])
```

iii. The agent verified that the result contains the expected 11 switch-task mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. A session is retained only if processing returns at least two valid trials.

ii.
```python
with h5py.File(nwb_path, 'r') as f:
    session_id = f['general/session_id'][()].decode()
...
if len(neural_trials) < 2:
    return None
```

iii. This follows the file organization and meets the decoder's two-trial minimum. The notes explain that m11 has only 12 available sessions while the other mice have 14.

## 1-d. How are the data split into trials?

i. Trial starts are all samples where `trial_start == 1`; trial ends are all samples where `teleport == 1`. The two lists are truncated to their common count, and each trial is the half-open slice `[start, teleport)`.

ii.
```python
trial_start_inds = np.where(trial_start_sig == 1)[0]
teleport_inds = np.where(teleport_sig == 1)[0]
n_trials = min(len(trial_start_inds), len(teleport_inds))
...
ts, te = trial_start_inds[ti], teleport_inds[ti]
trial_neural = dff[:, ts:te]
```

iii. The notes call a trial the track period from `trial_start` to `teleport`, excluding the teleport interval, and state that this matches the paper's boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Trials shorter than two frames are skipped. Sessions with fewer than two retained trials are skipped. No speed filter or scanning-validity filter is applied.

ii.
```python
nf = te - ts
if nf < 2: continue
...
if len(neural_trials) < 2:
    return None
```

iii. The notes say all trials should be included, no speed filtering is appropriate for this decoder, and no trials shorter than two frames were found. Although the notes say frames with `scanning == -1` should be excluded, the code does not load or use `scanning`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB `Fluorescence` (F) and `Neuropil` (Fneu) series for all imaging planes, restricted to ROIs whose `iscell` flag is one.

ii.
```python
fluor = f['processing/ophys/Fluorescence']
neu = f['processing/ophys/Neuropil']
iscell = seg['iscell'][:]
F_list.append(fluor[pname]['data'][:, pcell])
Fneu_list.append(neu[pname]['data'][:, pcell])
```

iii. The agent reasoned that the stored `Deconvolved` field is Suite2p output rather than the signal produced by the paper's later preprocessing, so it recomputed a signal from F and Fneu.

## 2-b. How is the `neural` data processed?

i. The code concatenates curated cells across planes, subtracts `0.7 * Fneu`, computes a separate per-trial maximin baseline using minimum and maximum filters of up to 300 samples, forms dF/F, Gaussian-smooths it with sigma two samples, replaces non-finite values with zero, and stores float32 dF/F. It does not perform the reference OASIS deconvolution.

ii.
```python
f_ = (F.T - NEU_COEF * Fneu.T).astype(np.float64)
flow[:, s:e] = scipy.ndimage.minimum_filter1d(f_[:, s:e], w, axis=-1)
flow[:, s:e] = scipy.ndimage.maximum_filter1d(flow[:, s:e], w, axis=-1)
dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
dff[:, s:e] = nansmooth(dff[:, s:e], SMOOTH_SIGMA, axis=1)
trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The notes acknowledge that the reference analyses use deconvolved events, but deliberately choose dF/F because it allegedly preserves more temporal information for decoding and because the NWB deconvolution is not the paper's recomputed OASIS signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `iscell[:, 0] == 1` are retained. Cells are not removed based on speed correlation or any other quality criterion.

ii.
```python
pcell = iscell[pmask, 0] == 1
F_list.append(fluor[pname]['data'][:, pcell])
```

iii. The notes identify `iscell` as Suite2p/manual curation and explicitly decide on “all iscell=1 neurons” with no additional filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural arrays are sliced beginning at the `trial_start` sample, so column zero is aligned to trial start; the slice ends immediately before teleport.

ii.
```python
ts, te = trial_start_inds[ti], teleport_inds[ti]
trial_neural = dff[:, ts:te]
```

iii. The agent describes this as temporal alignment to trial start and reports manual alignment checks against behavioral timestamps.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is performed. Samples remain at approximately 15.5078125 Hz (64.48 ms), including multi-plane data already represented at the per-plane rate. Metadata uses the fixed target rate.

ii.
```python
TARGET_RATE_HZ = 15.5078125
time_bin_ms = 1000.0 / TARGET_RATE_HZ
...
'time_bin_size': time_bin_ms
```

iii. The agent observed roughly 15.5 Hz from behavioral timestamps and concluded that the multi-plane files already have the proper per-plane temporal sampling.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the NWB `position/timestamps` array and the trial-start indices derived from `trial_start`.

ii.
```python
frame_timestamps = behav['position/timestamps'][:]
time_from_start = frame_timestamps[ts:te] - frame_timestamps[ts]
```

iii. The notes say actual NWB frame timestamps are used and report a manual equality check against the converted input.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in each trial is subtracted from every timestamp in that trial, then the values are cast to float32.

ii.
```python
time_from_start = (frame_timestamps[ts:te] - frame_timestamps[ts]).astype(np.float32)
inp[0] = time_from_start
```

iii. This directly implements elapsed seconds from the alignment event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The timestamps and neural activity use the identical `[ts:te]` indices and therefore have the same number and ordering of samples.

ii.
```python
trial_neural = dff[:, ts:te]
time_from_start = frame_timestamps[ts:te] - frame_timestamps[ts]
```

iii. The agent states that the behavior and imaging streams are already frame-aligned and validated this on a sample.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
environment = behav['environment/data'][:]
env_vals = environment[ts:te]
```

iii. The notes identify raw values zero and one as ENV1 and ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative samples are discarded, the median valid within-trial value is rounded to an integer, and that scalar is repeated across the trial.

ii.
```python
valid = env_vals[env_vals >= 0]
if len(valid) > 0:
    trial_env[ti] = int(np.round(np.median(valid)))
...
inp[1] = trial_env[ti]
```

iii. The target calls environment a binary per-trial input; the median makes the nominally constant trial value robust to invalid samples.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the zero-based loop index over detected trial boundaries, not the NWB `trial number` series.

ii.
```python
for ti in range(n_trials):
    ...
    inp[2] = ti
```

iii. The agent treats this as the within-session sequential trial number and reports checking it against expectations.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transform is applied beyond broadcasting the zero-based trial index across all frames of that trial as float32.

ii.
```python
inp = np.zeros((4, nf), dtype=np.float32)
inp[2] = ti
```

iii. This makes the requested per-trial continuous variable compatible with the common `(variables, time)` input shape.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps`, position timestamps, raw `reward_zone`, and the detected trial bounds. A trial is rewarded only if a reward timestamp lies in its time interval and `reward_zone` is positive somewhere in the trial.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][:]
has_rzone = np.any(reward_zone_raw[ts:te] > 0)
has_reward = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
if has_reward and has_rzone: isreward[ti] = 1
```

iii. The notes say reward timestamps use wall-clock time and claim that requiring both reward and reward-zone evidence matches the reference trial-type logic.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The current trial receives the preceding trial's binary `isreward`; the first trial defaults to zero. The scalar is repeated across time.

ii.
```python
prev_outcome = int(isreward[ti-1]) if ti > 0 else 0
inp[3] = prev_outcome
```

iii. This directly implements omitted=0/rewarded=1 and the notes explicitly document the first-trial default.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` plus reward-zone coordinates inferred from the NWB identifier/scene name and a fixed switch after trial 29. The raw `reward_zone` signal is not used to select the zone.

ii.
```python
scene = parse_scene_name(identifier)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
rz_s, rz_e = rz_coords[ti]
```

iii. The notes cite the paper's A/B/C coordinates and scene-to-zone reference logic, and state that reward-zone labels correctly switch at trial 30.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is clipped to 0–450 cm. Distance is negative before the zone, zero within its inclusive bounds, and positive after the zone, measured to the nearest boundary; it is then discretized.

ii.
```python
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
dist = np.where(trial_pos < rz_s, trial_pos - rz_s,
       np.where(trial_pos > rz_e, trial_pos - rz_e, 0.0))
dist_disc = discretize_distance(dist)
```

iii. The agent says this implements reward-relative position and reports manually validating the calculation and class balance.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks implement the seven requested intervals, including a distinct exact-zero class.

ii.
```python
r[d < -50] = 0
r[(d >= -50) & (d < -10)] = 1
r[(d >= -10) & (d < 0)] = 2
r[d == 0] = 3
r[(d > 0) & (d <= 10)] = 4
r[(d > 10) & (d <= 50)] = 5
r[d > 50] = 6
```

iii. The thresholds are taken directly from the decoder specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity are sliced with the same trial start and end indices, producing one distance category per neural column.

ii.
```python
trial_neural = dff[:, ts:te]
trial_pos = position[ts:te]
out[0] = dist_disc
```

iii. The agent relies on the NWB's common frame indexing and reports sample alignment checks.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the NWB `position/data` behavioral series.

ii.
```python
position = behav['position/data'][:]
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
```

iii. The notes identify position as centimeters along the 450 cm corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is sliced per trial, clipped into `[0, 450]`, and passed to the five-bin discretizer.

ii.
```python
trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
pos_disc = discretize_position(trial_pos)
```

iii. Clipping was intended to enforce the known physical track range before categorization, although the notes do not specifically justify changing small out-of-range raw values.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The code uses classes 0–4 at 90, 180, 270, and 360 cm.

ii.
```python
r[p < 90] = 0
r[(p >= 90) & (p < 180)] = 1
r[(p >= 180) & (p < 270)] = 2
r[(p >= 270) & (p < 360)] = 3
r[p >= 360] = 4
```

iii. Five equal 90 cm bins follow directly from the 450 cm track and the requested thresholds.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural activity are extracted with the same `[ts:te]` slice.

ii.
```python
trial_neural = dff[:, ts:te]
trial_pos = position[ts:te]
out[1] = pos_disc
```

iii. Common frame indexing is assumed and was checked on sample trials.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the NWB `lick/data` behavioral series.

ii.
```python
lick = behav['lick/data'][:]
trial_lick = lick[ts:te]
```

iii. The notes map this source directly to the lick decoder output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any raw value greater than zero becomes one; all other values become zero.

ii.
```python
lick_bin = (trial_lick > 0).astype(np.int64)
out[3] = lick_bin
```

iii. This implements the required binary no/yes representation.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural arrays use the same per-trial sample slice.

ii.
```python
trial_neural = dff[:, ts:te]
trial_lick = lick[ts:te]
```

iii. The agent relies on the shared frame grid and reports no validation warnings.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the file identifier's scene string and the trial index. Scene names specify fixed or pre/post-switch A/B/C zones; coordinates come from hard-coded paper values.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene_name(identifier)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
```

iii. The agent judged scene metadata plus the known switch at trial 30 to be the reference mapping and validated approximately balanced A/B/C frequencies.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene-name patterns are parsed into pre/post labels; switch sessions change at index 30. Labels A/B/C are mapped to 0/1/2 and broadcast across the trial. Training labels fall back to class zero when encoded.

ii.
```python
ct = min(change_trial, n_trials)
rz_labels[:ct] = pre_zone
rz_labels[ct:] = post_zone
...
rz_label_int = {'A': 0, 'B': 1, 'C': 2}.get(rz_labels[ti], 0)
out[4] = rz_label_int
```

iii. The notes cite the paper's coordinates and switch design and report that the full data are nearly one-third in each class.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses `Reward/timestamps`, `position/timestamps`, raw `reward_zone`, and trial boundaries.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][:]
has_rzone = np.any(reward_zone_raw[ts:te] > 0)
has_reward = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
```

iii. The notes say the reward events use wall-clock timestamps and that requiring reward-zone evidence matches the reference trial-type calculation.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code tests whether any reward timestamp falls between the timestamps of its start and end and whether any raw reward-zone sample is positive. Their conjunction becomes a binary value repeated across the trial.

ii.
```python
if has_reward and has_rzone:
    isreward[ti] = 1
...
out[5] = isreward[ti]
```

iii. The agent validated an 84.3% reward rate against the expected approximately 85% and used that as evidence for the rule.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Start/end lists are silently truncated to their common length; trials shorter than two frames and sessions with fewer than two trials are skipped; invalid environment samples are ignored with an all-invalid trial defaulting to zero; neural NaN/Inf values are replaced by zero. No general stream-length reconciliation or timestamp assertions are implemented.

ii.
```python
n_trials = min(len(trial_start_inds), len(teleport_inds))
if nf < 2: continue
valid = env_vals[env_vals >= 0]
trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The notes document two trials with infinite dF/F caused by zero baselines and say replacing those values with zero resolved verifier failures. They also record missing m11 sessions and unusually long but valid stopped-running trials.

## 13-a. What are the most time-consuming steps of the code?

i. The dominant conversion computation is dF/F baseline filtering and smoothing over large neuron-by-time arrays. Reading every NWB file and writing the roughly 9.4 GB pickle are also costly. The script times each session and specifically times dF/F.

ii.
```python
t_dff = time.time()
dff = compute_dff(F, Fneu, trial_start_inds, teleport_inds)
print(f"    dF/F: {time.time()-t_dff:.1f}s")
...
pickle.dump(data, pf, protocol=4)
```

iii. The notes report about two seconds per session, 4.5 minutes total conversion, and a 9,386.6 MB result; this makes the full-array filtering and serialization the evident bottlenecks.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Reward detection, environment aggregation, and final trial construction make three separate passes over trials; reward timestamp membership could be found with vectorized search/sorted indexing, and environment summaries could be combined with trial construction. Baseline filtering must still respect trial boundaries, but its loop could be consolidated with trial extraction.

ii.
```python
for ti in range(n_trials):  # reward outcomes
    ...
for ti in range(n_trials):  # environments
    ...
for ti in range(n_trials):  # build arrays
    ...
```

iii. The agent emphasized that runtime was already below the 15-minute target and therefore did not further optimize these small per-session trial loops.

## 13-c. What processing does the code repeat multiple times?

i. It repeatedly traverses the trial boundaries: twice in `compute_dff` (baseline, then smoothing) and three times in `process_session` (reward, environment, output construction). It also recomputes clipped position and reward-relative distance in optional plotting.

ii.
```python
for s, e in zip(trial_start_inds, teleport_inds):  # baseline
    ...
for s, e in zip(trial_start_inds, teleport_inds):  # smoothing
    ...
tp = np.clip(position[ts:te], 0, TRACK_LENGTH)
d = np.where(tp < rz_s, tp - rz_s, np.where(tp > rz_e, tp - rz_e, 0.0))
```

iii. The agent accepted this repetition because conversion completed in about 4.5 minutes; optional plot recomputation is used only for visual validation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion computes and retains diagnostic values such as effective frame rate and scene in each temporary session dictionary, but they are not put into the final dataset. It loads raw `reward_zone` only to gate reward outcomes, and optional plots recompute distance and clipping solely for inspection. More importantly, full-session dF/F arrays include NaN-filled inter-trial columns that are never saved.

ii.
```python
dff = np.full_like(f_, np.nan)
...
return {'scene': scene, 'effective_rate': effective_rate, ...}
...
for s in all_sessions:
    neural.append(s['neural_trials'])
```

iii. The agent did not identify discarded processing in its notes. These costs are modest except for allocating full-session baseline/dF/F arrays, which simplifies faithful indexing and trial extraction.
