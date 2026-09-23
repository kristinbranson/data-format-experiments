# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script finds all NWB files with a glob over `/app/data/sub-*/*.nwb`, treats each NWB file as one session, and reads behavior and ophys arrays directly from the HDF5 structure with `h5py`.

ii.
```python
paths = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
...
def process_session(path, signal='dff'):
    f = h5py.File(path, 'r')
    ...
    b = f['processing/behavior/BehavioralTimeSeries']
    ...
    F, Fneu, iscell, plane_idx = load_fluorescence(f)
```

iii. In the trajectory, the agent says it inspected the NWB tree, confirmed 11 subject directories and 152 NWB sessions, and decided the needed behavior and ophys arrays could be loaded directly from the file structure (steps 12, 30-35).

## 1-b. How are the data split into subjects?

i. Subjects are identified from each NWB file's `general/subject/subject_id`, and the final `subjects` list is built from the unique subject ids encountered while assembling sessions.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
...
sub = res['info']['subject']
if sub not in subjects:
    subjects.append(sub)
data['subject_idx'].append(subjects.index(sub))
```

iii. The trajectory says the agent validated the dataset as 11 mice and used subject/session metadata from each NWB file rather than relying only on directory names (steps 31-35, 85-86).

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session.

ii.
```python
paths = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
...
jobs = [(p, args.signal, args.cache_dir) for p in paths]
...
res = process_session(path, signal=signal)
```

iii. In the trajectory, the agent repeatedly describes the dataset as one NWB per session and reports 152 sessions after scanning the file inventory (steps 31-35, 68, 85-86).

## 1-d. How are the data split into trials?

i. Trials are taken as laps from the first positive `trial_start` sample to the first positive `teleport` sample for that lap, and all per-trial arrays are sliced with the same `[start:stop]` frame indices.

ii.
```python
starts = np.where(get('trial_start') > 0)[0]
stops = np.where(get('teleport') > 0)[0]
...
for i, (s, e) in enumerate(zip(starts, stops)):
    act = activity[:, s:e]
    ...
    t_rel = tstamps[s:e] - tstamps[s]
```

iii. The trajectory says the agent matched the paper/repo convention of "trial_start to teleport" laps and validated this directly against the behavior streams before implementing the conversion (steps 41, 45-49, 55, 85-86).

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they have lick-sensor errors, if neural activity contains non-finite values, or if position/speed/lick slices contain non-finite values. Entire sessions are also skipped if fewer than two usable trials remain.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    got_reward = np.any((reward_frames >= s) & (reward_frames < e))
    in_zone = np.any(rzone_entry[s:e] > 0)
    rewarded[i] = int(bool(got_reward) and bool(in_zone))
    seg = lick[s:e]
    lick_error[i] = (np.sum(seg > LICK_ERR_COUNT) / len(seg)) > LICK_ERR_FRAC
...
if lick_error[i]:
    continue
act = activity[:, s:e]
if not np.all(np.isfinite(act)):
    continue
...
if len(res['neural']) < 2:
    print('skipping %s: fewer than 2 usable trials' % fn)
    continue
```

iii. The trajectory says the agent chose the paper's lick-error rule because it reproduced the paper's count of 81 bad trials exactly, and it also added defensive skipping of non-finite trial slices and sessions with too few remaining trials (steps 44-46, 51-55, 75, 85-86).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from raw fluorescence `Fluorescence`, raw neuropil `Neuropil`, and ROI annotations in `ImageSegmentation/PlaneSegmentation` (`iscell`, `planeIdx`). The stored NWB `Deconvolved` signal is not used.

ii.
```python
ophys = f['processing/ophys']
seg = ophys['ImageSegmentation/PlaneSegmentation']
...
F[rois] = ophys['Fluorescence'][plane]['data'][:].T
Fneu[rois] = ophys['Neuropil'][plane]['data'][:].T
iscell = seg['iscell'][:, 0] > 0
plane_idx = seg['planeIdx'][:]
```

iii. The trajectory says the agent inspected the NWB ophys groups and decided to follow the paper pipeline based on raw `Fluorescence` and `Neuropil`, not the stored suite2p `Deconvolved` array (steps 27, 29-31, 48-49).

## 2-b. How is the `neural` data processed?

i. The script computes trial-restricted dF/F by masking outside-trial frames to `NaN`, subtracting `0.7 * Fneu`, smoothing with a Gaussian, taking a per-trial maximin baseline, adding back the trial mean neuropil offset, and computing `(F - baseline) / abs(baseline)`. It also computes OASIS events for every trial, but by default it exports dF/F rather than the events.

ii.
```python
for s, e in zip(starts, stops):
    f_[:, s:e] = F[:, s:e]
    fneu_[:, s:e] = Fneu[:, s:e]
...
f_ -= NEU_COEF * fneu_
...
seg = nansmooth(f_[:, s:e], [0, BASELINE_SMOOTH])
win = min(BASELINE_WIN, max(1, e - s))
seg = minimum_filter1d(seg, win, axis=-1)
seg = maximum_filter1d(seg, win, axis=-1)
offset = NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
...
dff[:, valid] = (f_[:, valid] - flow[:, valid]) / np.abs(flow[:, valid])
...
events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]),
                            2000, TAU, frame_rate)
...
activity = events if signal == 'events' else dff
```

iii. The trajectory says the agent intentionally matched the paper's dF/F pipeline, benchmarked `events` versus `dff`, and then chose `dff` as the exported neural signal because it decoded better on a 4-session test and retained graded amplitude information (steps 46-49, 55, 61-62, 76, 85-86).

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are first restricted to suite2p-curated cells (`iscell`). Then putative interneurons are removed if the cell's dF/F is correlated with running speed above `r > 0.5`.

ii.
```python
F = F[iscell]
Fneu = Fneu[iscell]
plane_idx = plane_idx[iscell]
...
with np.errstate(invalid='ignore', divide='ignore'):
    speed_corr = (dv @ sv) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
keep_cells = ~is_int
...
activity = activity[keep_cells]
plane_idx = plane_idx[keep_cells]
```

iii. The trajectory says the agent read the paper's interneuron exclusion rule, validated that the excluded fraction was consistent with the paper, and kept the `iscell` manual curation first (steps 41-43, 49-50, 55, 75, 85-86).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned by slicing each trial from `trial_start` to `teleport`, so timepoint 0 of each emitted trial is the start of the trial.

ii.
```python
starts = np.where(get('trial_start') > 0)[0]
stops = np.where(get('teleport') > 0)[0]
...
for i, (s, e) in enumerate(zip(starts, stops)):
    act = activity[:, s:e]
```

iii. The trajectory says the agent chose the paper's lap definition and treated that as the alignment event requested by the decoder instructions (steps 41, 45-49, 55, 85-86).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output keeps the native imaging/behavior frame rate, computed from behavior timestamps. No temporal rebinning or resampling is applied.

ii.
```python
tstamps = b['position/timestamps'][:]
...
frame_rate = 1.0 / np.median(np.diff(tstamps))
...
bin_ms = float(np.mean([1000.0 / s['frame_rate_hz'] for s in session_info]))
...
'time_bin_size': bin_ms,
```

iii. The trajectory says the agent determined that all VR streams were already synchronized to imaging frames at about 15.5 Hz and therefore used the native frame as the time bin (steps 30, 46, 53-55, 85-86).

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps attached to the `position` time series.

ii.
```python
tstamps = b['position/timestamps'][:]
...
t_rel = tstamps[s:e] - tstamps[s]
inp[0] = t_rel
```

iii. The trajectory says the agent concluded the VR streams shared a common time base aligned to imaging frames, so one behavior timestamp stream was sufficient for alignment (steps 30, 46, 53-55, 85-86).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the script subtracts the timestamp of the first frame in the trial from all timestamps in that trial.

ii.
```python
t_rel = tstamps[s:e] - tstamps[s]
inp[0] = t_rel
```

iii. The trajectory treats start-of-trial alignment as a simple within-trial offset from the first frame after `trial_start` (steps 45-49, 55).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the same `[start:stop]` frame indices as the neural data, so alignment is by construction with no interpolation.

ii.
```python
act = activity[:, s:e]
t_rel = tstamps[s:e] - tstamps[s]
inp[0] = t_rel
```

iii. The trajectory explicitly says neural, input, and output streams are already synchronized on imaging frames, so the same frame slice aligns them (steps 30, 46, 53-55, 85-86).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavior time series.

ii.
```python
morph = get('environment')
...
env = np.array([1.0 if np.nanmax(morph[s:e]) > 0.5 else 0.0
                for s, e in zip(starts, stops)])
...
inp[1] = env[i]
```

iii. The trajectory says the agent checked per-trial morph/environment behavior while validating scenes and trial semantics before finalizing the conversion (steps 44-45, 50-52).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The script reduces the time-varying `environment` samples within each trial to a single binary value using `max > 0.5`, then broadcasts that scalar across the whole trial.

ii.
```python
env = np.array([1.0 if np.nanmax(morph[s:e]) > 0.5 else 0.0
                for s, e in zip(starts, stops)])
...
inp[1] = env[i]
```

iii. The trajectory does not give a separate formal justification, but it says the agent checked that environment/morph was consistent at the trial level while validating switch scenes (steps 50-52).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the raw `trial number` behavior time series, using the first sample of that series within each extracted trial.

ii.
```python
trialnum = get('trial number')
...
inp[2] = float(trialnum[s])
```

iii. The trajectory does not record a detailed argument for this choice; the final code simply uses the stored `trial number` stream rather than recomputing a within-session trial index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. There is no further processing beyond taking `trialnum[s]` for a trial and broadcasting that scalar across all timepoints in the trial.

ii.
```python
inp = np.empty((4, T), dtype=np.float32)
...
inp[2] = float(trialnum[s])
```

iii. The trajectory does not explicitly justify this processing step.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward event timestamps (`Reward/timestamps`) plus the `reward_zone` behavior series, because the script defines a rewarded trial as one containing both a reward event and reward-zone occupancy.

ii.
```python
reward_frames = np.searchsorted(tstamps, b['Reward/timestamps'][:])
rzone_entry = get('reward_zone')
...
got_reward = np.any((reward_frames >= s) & (reward_frames < e))
in_zone = np.any(rzone_entry[s:e] > 0)
rewarded[i] = int(bool(got_reward) and bool(in_zone))
...
prev_rewarded = np.concatenate([[1], rewarded[:-1]]).astype(np.float64)
```

iii. The trajectory says the agent followed the paper's `get_trial_types` logic for reward outcome and explicitly reasoned that reward-zone location must be paired with reward events to define reward status (steps 22, 47-49, 55, 85-86).

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The per-trial reward labels are shifted by one trial. Unlike the human reference, the first trial is hard-coded to `1` rather than `0`, based on the assumption that pre-imaging warm-up trials in the same condition were usually rewarded.

ii.
```python
# previous trial outcome; for the first imaged trial of a session the previous
# trial is one of the ~30 un-imaged warm-up trials run in the same condition
# immediately before imaging, which were rewarded on ~85% of trials, so it is
# coded as rewarded.
prev_rewarded = np.concatenate([[1], rewarded[:-1]]).astype(np.float64)
...
inp[3] = prev_rewarded[i]
```

iii. The trajectory says the agent revisited this exact design choice late in the run and kept the first-trial value as rewarded because warm-up trials were assumed to have similar reward statistics (step 77 and the in-code comment).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the `position` behavior series and from a per-trial reward-zone label inferred from the session `identifier`/scene name, not from the raw `reward_zone` series.

ii.
```python
ident = f['identifier'][()].decode()
scene = ident.split('/')[-1]
...
zone_labels = scene_reward_zones(scene, ntrials)
...
p = np.clip(pos[s:e], 0.0, None)
zstart, zend = REWARD_ZONES[zone_labels[i]]
dist = np.where(p < zstart, p - zstart, np.where(p > zend, p - zend, 0.0))
```

iii. The trajectory says the agent determined that the raw `reward_zone` signal only marked rewarded zone entries, so the stable trial-level zone had to come from the scene name plus the 30-trial switch rule; it then validated this against reward-zone-entry positions with zero mismatches (steps 47-52, 55, 85-86).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each frame, the script computes signed distance to the nearest point of the active reward zone: negative before the zone, `0` inside it, positive after it. That continuous distance is then discretized into seven categories.

ii.
```python
zstart, zend = REWARD_ZONES[zone_labels[i]]
dist = np.where(p < zstart, p - zstart, np.where(p > zend, p - zend, 0.0))
...
out[0] = discretize_reward_distance(dist)
```

iii. The trajectory says the agent followed the decoder specification after establishing the reward-zone identity per trial from the scene and switch structure (steps 47-55, 85-86).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The script uses manual threshold comparisons implementing the seven requested bins: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii.
```python
out = np.full(d.shape, 6, dtype=np.int64)
out[d > 50] = 6
out[(d > 10) & (d <= 50)] = 5
out[(d > 0) & (d <= 10)] = 4
out[d == 0] = 3
out[(d >= -10) & (d < 0)] = 2
out[(d >= -50) & (d < -10)] = 1
out[d < -50] = 0
```

iii. The trajectory does not separately justify the thresholds beyond saying the converter follows the decoder task bins.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same trial frame slice `[s:e]` that is used to slice neural activity.

ii.
```python
act = activity[:, s:e]
p = np.clip(pos[s:e], 0.0, None)
...
out[0] = discretize_reward_distance(dist)
```

iii. The trajectory explicitly says the behavior streams are already synchronized to imaging frames, so the same frame indices align the outputs with neural data (steps 30, 46, 53-55, 85-86).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the `position` behavior time series.

ii.
```python
pos = get('position')
...
p = np.clip(pos[s:e], 0.0, None)
...
out[1] = np.clip(np.floor(p / 90.0), 0, 4).astype(np.int64)
```

iii. The trajectory shows the agent inspecting position ranges and validating that laps map onto the 450 cm track used by the decoder task (steps 45, 50-55).

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped at zero and then converted to a five-bin index by dividing by 90 cm, taking the floor, and clipping the result to `[0, 4]`.

ii.
```python
p = np.clip(pos[s:e], 0.0, None)
...
out[1] = np.clip(np.floor(p / 90.0), 0, 4).astype(np.int64)
```

iii. The trajectory indicates the agent checked position ranges and wanted bins matching the 450 cm track while tolerating small out-of-range samples (steps 50-55, 77).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The script uses five 90 cm bins spanning the 450 cm track by computing `floor(position / 90)` and clipping to `[0, 4]`.

ii.
```python
out[1] = np.clip(np.floor(p / 90.0), 0, 4).astype(np.int64)
```

iii. In the trajectory, the agent says it double-checked that the position discretization matched the specification `(<90, 90-180, ..., >360)` before finishing (step 77).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same trial slice `[s:e]` as the neural data.

ii.
```python
act = activity[:, s:e]
p = np.clip(pos[s:e], 0.0, None)
```

iii. The trajectory says framewise alignment is inherited directly from the synchronized NWB behavior and imaging streams (steps 30, 46, 53-55, 85-86).

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavior time series.

ii.
```python
lick = get('lick')
...
lk = lick[s:e]
```

iii. The trajectory says the agent checked lick statistics when validating lick-sensor errors and trial quality (steps 44-46, 51-55).

## 9-b. What processing is involved in computing `output` *Lick*?

i. The raw lick values are binarized so that any value greater than zero becomes `1`, otherwise `0`.

ii.
```python
out[3] = (lk > 0).astype(np.int64)
```

iii. The trajectory does not give a separate argument for this beyond using the binary decoder target requested in the task.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It uses the same trial frame slice `[s:e]` as the neural data.

ii.
```python
act = activity[:, s:e]
lk = lick[s:e]
...
out[3] = (lk > 0).astype(np.int64)
```

iii. The trajectory repeatedly says all behavior streams were already aligned to imaging frames in the NWB files (steps 30, 46, 53-55, 85-86).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB `identifier`/scene string, which is parsed into a session-level reward-zone schedule, plus the number of extracted trials.

ii.
```python
ident = f['identifier'][()].decode()
scene = ident.split('/')[-1]
...
zone_labels = scene_reward_zones(scene, ntrials)
...
out[4] = ZONE_NAMES.index(zone_labels[i])
```

iii. The trajectory says the agent concluded the reward-zone-entry signal was insufficient to recover the active zone on omission trials, so it used the scene name and validated that against observed zone entries with zero mismatches (steps 47-52, 55, 85-86).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed with regexes. Fixed sessions get the same zone label for every trial; switch sessions use the first zone for the first 30 trials and the second zone thereafter. The label is finally converted to integer `0/1/2`.

ii.
```python
def scene_reward_zones(scene, ntrials, change_trial=SWITCH_TRIAL):
    m = re.match(r'^Env\\d_Location([ABC])$', scene)
    if m:
        return [m.group(1)] * ntrials
    m = re.match(r'^Env\\d_Location([ABC])_to_([ABC])$', scene)
    if m is None:
        m = re.match(r'^Env\\d_([ABC])_to_Env\\d_([ABC])$', scene)
    if m:
        return [m.group(1)] * change_trial + [m.group(2)] * (ntrials - change_trial)
...
out[4] = ZONE_NAMES.index(zone_labels[i])
```

iii. The trajectory says the agent enumerated scene names across all sessions, confirmed they matched a small set of patterns, and verified the 30-trial switch rule empirically (steps 50-52, 55).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from reward event timestamps (`Reward/timestamps`) together with the `reward_zone` behavior signal, since the script defines reward outcome as reward delivered while the animal is in the zone.

ii.
```python
reward_frames = np.searchsorted(tstamps, b['Reward/timestamps'][:])
rzone_entry = get('reward_zone')
...
got_reward = np.any((reward_frames >= s) & (reward_frames < e))
in_zone = np.any(rzone_entry[s:e] > 0)
rewarded[i] = int(bool(got_reward) and bool(in_zone))
...
out[5] = rewarded[i]
```

iii. The trajectory says the agent followed the paper's `get_trial_types` definition, namely reward delivered while in the reward zone, rather than using reward timestamps alone (steps 22, 47-49, 55, 85-86).

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are first converted into frame indices with `np.searchsorted`. Then each trial is marked rewarded only if it contains a reward event and a positive `reward_zone` entry.

ii.
```python
reward_frames = np.searchsorted(tstamps, b['Reward/timestamps'][:])
...
for i, (s, e) in enumerate(zip(starts, stops)):
    got_reward = np.any((reward_frames >= s) & (reward_frames < e))
    in_zone = np.any(rzone_entry[s:e] > 0)
    rewarded[i] = int(bool(got_reward) and bool(in_zone))
...
out[5] = rewarded[i]
```

iii. The trajectory says the agent validated that this reproduced the expected omission rate and matched the paper/repo concept of a rewarded trial (steps 47-52, 55, 75, 85-86).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script truncates neural and behavior streams to the common length when they differ, drops trials with lick-sensor errors, drops any trial containing non-finite neural/behavior slices, and skips sessions with fewer than two remaining trials. It does not attempt to infer missing reward-zone labels from behavior because it uses scene metadata instead.

ii.
```python
nsamp = min(F.shape[1], len(pos))
if F.shape[1] != len(pos):
    F = F[:, :nsamp]
    Fneu = Fneu[:, :nsamp]
    pos = pos[:nsamp]; speed = speed[:nsamp]; lick = lick[:nsamp]
    morph = morph[:nsamp]; trialnum = trialnum[:nsamp]
    rzone_entry = rzone_entry[:nsamp]; tstamps = tstamps[:nsamp]
    keep = stops <= nsamp
    starts = starts[keep]; stops = stops[keep]
...
if lick_error[i]:
    continue
...
if not np.all(np.isfinite(act)):
    continue
...
if not (np.all(np.isfinite(p)) and np.all(np.isfinite(spd))
        and np.all(np.isfinite(lk))):
    continue
```

iii. The trajectory explicitly mentions discovering one-frame mismatches in some two-plane sessions and fixing them by truncating to the shared length; it also treats the 81 lick-error trials as known bad data to exclude (steps 52, 65-66, 75, 85-86).

## 13-a. What are the most time-consuming steps of the code?

i. The heaviest steps are per-session NWB I/O and fluorescence loading, trialwise dF/F plus OASIS computation in `compute_activity`, and writing/assembling the large cached and final pickle files.

ii.
```python
F, Fneu, iscell, plane_idx = load_fluorescence(f)
...
dff, events, valid = compute_activity(F, Fneu, starts, stops, frame_rate)
...
with open(outfn, 'wb') as fh:
    pickle.dump(res, fh, protocol=4)
...
with open(args.out, 'wb') as fh:
    pickle.dump(data, fh, protocol=4)
```

iii. The trajectory focuses on estimating whole-dataset size, benchmarking per-session processing, and monitoring the full conversion runtime, which implies these steps dominate cost (steps 33-39, 48-49, 53-55, 63-69).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the loops over trials in `compute_activity`, the loop that computes `rewarded` and `lick_error`, and the final loop that slices every trial and builds `inp`/`out` arrays.

ii.
```python
for s, e in zip(starts, stops):
    f_[:, s:e] = F[:, s:e]
    fneu_[:, s:e] = Fneu[:, s:e]
...
for s, e in zip(starts, stops):
    seg = nansmooth(f_[:, s:e], [0, BASELINE_SMOOTH])
    ...
for i, (s, e) in enumerate(zip(starts, stops)):
    got_reward = np.any((reward_frames >= s) & (reward_frames < e))
    ...
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    neural_trials.append(np.ascontiguousarray(act, dtype=np.float32))
```

iii. The trajectory does not explicitly discuss vectorization; instead, it emphasizes correctness and whole-session multiprocessing/caching as the performance strategy (steps 55-58, 62-69).

## 13-c. What processing does the code repeat multiple times?

i. The code always computes both `dff` and `events` even when only one signal is exported. It also reads each per-session cache file back in during final assembly after already writing it once.

ii.
```python
dff, events, valid = compute_activity(F, Fneu, starts, stops, frame_rate)
...
activity = events if signal == 'events' else dff
...
with open(outfn, 'wb') as fh:
    pickle.dump(res, fh, protocol=4)
...
with open(fn, 'rb') as fh:
    res = pickle.load(fh)
```

iii. The trajectory says the agent deliberately kept support for both neural signal types and used per-session cache files to make reruns and final reassembly practical, even though that means repeating some work and I/O (steps 55-58, 61-62, 76, 79-80).

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is that OASIS `events` are always computed even when the default exported neural signal is `dff`. The script also carries `plane_idx` through each cached session result, but the final exported dataset collapses all neurons into one brain region and never exports plane identity.

ii.
```python
events = np.full(F.shape, np.nan, dtype=np.float32)
for s, e in zip(starts, stops):
    dff[:, s:e] = nansmooth(dff[:, s:e], [0, DFF_SMOOTH])
    events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]),
                                2000, TAU, frame_rate)
...
return {
    'neural': neural_trials,
    'input': input_trials,
    'output': output_trials,
    'plane_idx': plane_idx,
    'info': info,
}
...
data['brain_region_idx'].append(np.zeros(res['neural'][0].shape[0], dtype=np.int64))
```

iii. The trajectory says the agent kept both signal paths so it could benchmark `events` against `dff`, then left `dff` as the default once it decoded better; that makes the extra `events` computation unnecessary in the final default workflow (steps 61-62, 76).
