# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files under `/app/data/sub-*/` are discovered with `glob.glob`. Each NWB file is opened with **h5py** (not pynwb) for direct low-level HDF5 access. Neural data (Fluorescence, Neuropil, iscell) and behavioral time series (position, speed, lick, environment, trial_start, teleport, Reward, reward_zone, trial number) are read from the standard NWB processing groups. Sessions are processed in parallel using multiprocessing with spawn context.

ii.
```python
paths = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
...
f = h5py.File(path, 'r')
b = f['processing/behavior/BehavioralTimeSeries']
get = lambda k: b[k + '/data'][:]
tstamps = b['position/timestamps'][:]
pos = get('position')
speed = get('speed')
lick = get('lick')
morph = get('environment')
trialnum = get('trial number')
starts = np.where(get('trial_start') > 0)[0]
stops = np.where(get('teleport') > 0)[0]
```

iii. The agent chose h5py for fast, direct access to HDF5 datasets. It verified 11 subjects, 152 sessions, ~12,216 trials across the dataset, matching the paper's description of 11 switch-condition mice.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `general/subject/subject_id` field within each NWB file. Unique subjects are collected as sessions are processed and assigned sequential indices.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
...
if sub not in subjects:
    subjects.append(sub)
data['subject_idx'].append(subjects.index(sub))
```

iii. The subject ID is embedded in each NWB file's metadata, providing a reliable source. The directory structure `sub-<id>` is consistent with this.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session (one imaging day per mouse). Session metadata (experiment day, date, scene name) is extracted from the NWB identifier and session_id fields.

ii.
```python
ident = f['identifier'][()].decode()
scene = ident.split('/')[-1]
exp_day = int(f['general/session_id'][()])
date = ident.split('/')[-2]
```

iii. Each NWB file is a self-contained recording session. The agent identified 12-14 sessions per mouse.

## 1-d. How are the data split into trials?

i. Trials are defined from `trial_start` to `teleport` behavioral time series. Trial start indices are frames where `trial_start > 0`, and trial end indices are frames where `teleport > 0`. Neural and behavioral data between these boundaries form each trial.

ii.
```python
starts = np.where(get('trial_start') > 0)[0]
stops = np.where(get('teleport') > 0)[0]
ntrials = len(starts)
assert len(stops) == ntrials
...
for i, (s, e) in enumerate(zip(starts, stops)):
    act = activity[:, s:e]
    p = np.clip(pos[s:e], 0.0, None)
```

iii. The agent followed the paper's trial definition: "laps of the 450 cm virtual linear track, from trial_start (entry to the track) to teleport (entry to the inter-trial interval), as in the sess class used throughout the repo."

## 1-e. How are trials filtered based on quality controls?

i. Three quality filters are applied: (1) Trials with lick sensor errors (>30% of frames with cumulative lick count > 2) are dropped, matching the paper's `behavior.correct_lick_sensor_error` (81 trials dropped). (2) Trials with non-finite neural activity are skipped. (3) Trials with non-finite behavioral data (position, speed, lick) are skipped. Sessions with fewer than 2 usable trials are excluded entirely.

ii.
```python
LICK_ERR_FRAC = 0.3
LICK_ERR_COUNT = 2
...
lick_error[i] = (np.sum(seg > LICK_ERR_COUNT) / len(seg)) > LICK_ERR_FRAC
...
for i, (s, e) in enumerate(zip(starts, stops)):
    if lick_error[i]:
        continue
    act = activity[:, s:e]
    if not np.all(np.isfinite(act)):
        continue
    ...
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(spd))
            and np.all(np.isfinite(lk))):
        continue
...
if len(res['neural']) < 2:
    print('skipping %s: fewer than 2 usable trials' % fn)
    continue
```

iii. The agent validated that the lick error filter matched the paper's count of 81 affected trials out of 12,376. The non-finite checks are defensive measures for data integrity.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p `Fluorescence` (F) and `Neuropil` (Fneu) traces stored in the NWB file, NOT from the pre-computed `Deconvolved` field.

ii.
```python
def load_fluorescence(f):
    ophys = f['processing/ophys']
    ...
    for plane in sorted(ophys['Fluorescence'].keys()):
        rois = ophys['Fluorescence'][plane]['rois'][:]
        F[rois] = ophys['Fluorescence'][plane]['data'][:].T
        Fneu[rois] = ophys['Neuropil'][plane]['data'][:].T
    iscell = seg['iscell'][:, 0] > 0
    ...
    return F, Fneu, iscell, plane_idx
```

iii. The agent correctly identified that the NWB's Deconvolved field is suite2p's own deconvolution, not the signal the paper analyzes. The paper computes its own dF/F and events from F and Fneu.

## 2-b. How is the `neural` data processed?

i. The paper's dF/F pipeline is reimplemented: (1) neuropil subtraction (F - 0.7*Fneu), (2) per-trial maximin baseline (15-sample Gaussian smooth, 300-sample min/max filter), (3) add back mean neuropil offset, (4) dF/F = (F_corrected - baseline) / |baseline|, (5) 2-sample Gaussian smooth of dF/F, (6) OASIS deconvolution (tau=0.7). Both dF/F and events are computed, but **dF/F is used as the default neural signal**.

Note: The AI does NOT implement the `keep_teleports` logic from the paper's `teleport_metadata.py`. The baseline window always restricts to within-trial data only.

ii.
```python
def compute_activity(F, Fneu, starts, stops, frame_rate):
    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        f_[:, s:e] = F[:, s:e]
        fneu_[:, s:e] = Fneu[:, s:e]
    valid = ~np.isnan(f_[0])
    f_ -= NEU_COEF * fneu_
    flow = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        seg = nansmooth(f_[:, s:e], [0, BASELINE_SMOOTH])
        win = min(BASELINE_WIN, max(1, e - s))
        seg = minimum_filter1d(seg, win, axis=-1)
        seg = maximum_filter1d(seg, win, axis=-1)
        offset = NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
        f_[:, s:e] = f_[:, s:e] + offset
        flow[:, s:e] = seg + offset
    dff = np.full(F.shape, np.nan, dtype=np.float32)
    dff[:, valid] = (f_[:, valid] - flow[:, valid]) / np.abs(flow[:, valid])
    events = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        dff[:, s:e] = nansmooth(dff[:, s:e], [0, DFF_SMOOTH])
        events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]),
                                    2000, TAU, frame_rate)
    return dff, events, valid
...
activity = events if signal == 'events' else dff
```

iii. The agent argued dF/F retains graded amplitude information useful for a linear decoder, while deconvolution in the paper served analyses needing asymmetric calcium kinetics removed. They compared both: "dF/F gives markedly better decoding than deconvolved events (position 0.72 vs 0.51)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) suite2p's `iscell` manual curation flag to restrict to curated ROIs, (2) putative interneuron exclusion based on Pearson correlation between dF/F and running speed > 0.5.

ii.
```python
iscell = seg['iscell'][:, 0] > 0
F = F[iscell]
Fneu = Fneu[iscell]
...
dv = dff_valid - dff_valid.mean(axis=1, keepdims=True)
sv = spd_valid - spd_valid.mean()
denom = np.sqrt((dv ** 2).sum(axis=1) * (sv ** 2).sum())
speed_corr = (dv @ sv) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
keep_cells = ~is_int
```

iii. Both filters match the paper's methods. The agent verified the interneuron fraction (~0.2%) matched the paper's reported range.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by slicing neural data between `trial_start` and `teleport` indices. Since all behavioral streams in the NWB files are already synchronized to imaging frames, no additional alignment is needed.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    act = activity[:, s:e]
    neural_trials.append(np.ascontiguousarray(act, dtype=np.float32))
```

iii. "All VR streams in the NWB files are already synchronized to the ~15.5 Hz imaging frames, so a frame is the natural time bin (64.5 ms) and no resampling is needed to align neural, input and output streams."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The native two-photon frame rate (~15.5 Hz, ~64.5 ms time bin) is preserved. The time_bin_size in metadata is the mean across sessions.

ii.
```python
bin_ms = float(np.mean([1000.0 / s['frame_rate_hz'] for s in session_info]))
data['metadata'] = {
    'time_bin_size': bin_ms,
    ...
}
```

iii. "native two-photon frame rate (~15.5 Hz); all VR behavior streams in the NWB files are already synchronized to imaging frames."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` behavioral time series.

ii.
```python
tstamps = b['position/timestamps'][:]
...
t_rel = tstamps[s:e] - tstamps[s]
inp[0] = t_rel
```

iii. The behavior timestamps are consistent with the imaging frame rate. Any behavior time series timestamps would give the same result.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp of the first frame in the trial is subtracted from all timestamps within the trial.

ii.
```python
t_rel = tstamps[s:e] - tstamps[s]
inp[0] = t_rel
```

iii. Standard approach for computing relative time within a trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices (both synchronized to imaging frames), so no additional alignment is needed. Both are sliced using the same `s:e` indices.

ii.
```python
act = activity[:, s:e]
...
t_rel = tstamps[s:e] - tstamps[s]
```

iii. Verified that behavior and neural data are sample-aligned in the NWB files.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` (morph) behavior time series.

ii.
```python
morph = get('environment')
...
env = np.array([1.0 if np.nanmax(morph[s:e]) > 0.5 else 0.0
                for s, e in zip(starts, stops)])
inp[1] = env[i]
```

iii. The environment variable encodes the VR environment (ENV1 vs ENV2) as a value between 0 and 1.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the maximum value of the morph/environment variable is compared to 0.5. If > 0.5, the trial is coded as ENV2 (1.0); otherwise ENV1 (0.0). The value is constant across all timepoints within a trial.

ii.
```python
env = np.array([1.0 if np.nanmax(morph[s:e]) > 0.5 else 0.0
                for s, e in zip(starts, stops)])
inp[1] = env[i]
```

iii. This thresholding approach handles any intermediate values in the morph variable.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series stored in the NWB file. The value at the first frame of each trial is used.

ii.
```python
trialnum = get('trial number')
...
inp[2] = float(trialnum[s])
```

iii. The agent used the stored trial number from the NWB data directly.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The value of `trial number` at the trial start frame is cast to float. The value is constant across all timepoints within a trial.

ii.
```python
inp[2] = float(trialnum[s])
```

iii. Minimal processing, just reading the stored value.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` time series timestamps and the `reward_zone` behavioral time series. Reward event frames are determined by `np.searchsorted` of reward timestamps into behavior timestamps.

ii.
```python
reward_frames = np.searchsorted(tstamps, b['Reward/timestamps'][:])
rzone_entry = get('reward_zone')
...
for i, (s, e) in enumerate(zip(starts, stops)):
    got_reward = np.any((reward_frames >= s) & (reward_frames < e))
    in_zone = np.any(rzone_entry[s:e] > 0)
    rewarded[i] = int(bool(got_reward) and bool(in_zone))
```

iii. Reward events have their own timestamps. `searchsorted` aligns them to behavior frames.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A trial is considered rewarded if both (1) a reward event occurred during the trial AND (2) the reward_zone entry flag was active. Previous trial outcome is the reward status of the preceding trial. For the first trial in each session, previous outcome is set to 1 (rewarded), based on the reasoning that pre-imaging warm-up trials were rewarded ~85% of the time.

ii.
```python
rewarded[i] = int(bool(got_reward) and bool(in_zone))
prev_rewarded = np.concatenate([[1], rewarded[:-1]]).astype(np.float64)
...
inp[3] = prev_rewarded[i]
```

iii. The agent followed `behavior.get_trial_types` for the reward definition. Setting first trial to 1 was a deliberate choice: "for the first imaged trial of a session the previous trial is one of the ~30 un-imaged warm-up trials run in the same condition immediately before imaging, which were rewarded on ~85% of trials."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone identity determined from the VR scene name in the NWB identifier. Reward zone boundaries: A (80-130 cm), B (200-250 cm), C (320-370 cm).

ii.
```python
REWARD_ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
...
zone_labels = scene_reward_zones(scene, ntrials)
...
zstart, zend = REWARD_ZONES[zone_labels[i]]
dist = np.where(p < zstart, p - zstart, np.where(p > zend, p - zend, 0.0))
```

iii. The paper defines these reward zone ranges. The agent verified: "zone inference from scene name + 30-trial switch matches the rzone entry positions exactly (0 mismatches)."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to the nearest edge of the reward zone: negative before the zone, zero inside, positive after. Position is first clipped to >= 0.

ii.
```python
p = np.clip(pos[s:e], 0.0, None)
...
dist = np.where(p < zstart, p - zstart, np.where(p > zend, p - zend, 0.0))
```

iii. Standard signed distance computation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using a manual conditional approach rather than `np.digitize`. Bins: <-50 (0), -50 to -10 (1), -10 to 0 (2), 0 (3), >0 to 10 (4), 10 to 50 (5), >50 (6).

ii.
```python
def discretize_reward_distance(d):
    out = np.full(d.shape, 6, dtype=np.int64)
    out[d > 50] = 6
    out[(d > 10) & (d <= 50)] = 5
    out[(d > 0) & (d <= 10)] = 4
    out[d == 0] = 3
    out[(d >= -10) & (d < 0)] = 2
    out[(d >= -50) & (d < -10)] = 1
    out[d < -50] = 0
    return out
```

iii. Matches the bin definitions in the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as neural data within each trial — both use `s:e` slice.

ii.
```python
p = np.clip(pos[s:e], 0.0, None)
dist = np.where(p < zstart, p - zstart, np.where(p > zend, p - zend, 0.0))
out[0] = discretize_reward_distance(dist)
```

iii. Neural and behavioral data are sample-aligned.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = get('position')
...
p = np.clip(pos[s:e], 0.0, None)
```

iii. The position variable directly records the animal's VR corridor position in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to >= 0, then discretized into 5 bins using `np.floor(p / 90)` clipped to [0, 4].

ii.
```python
out[1] = np.clip(np.floor(p / 90.0), 0, 4).astype(np.int64)
```

iii. Five 90 cm bins span the 450 cm track: [0,90), [90,180), [180,270), [270,360), [360,450].

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.floor(p / 90)` maps positions to bins, clipped to range [0, 4]. This is equivalent to bins: <90 cm (0), 90-180 cm (1), 180-270 cm (2), 270-360 cm (3), >360 cm (4).

ii.
```python
out[1] = np.clip(np.floor(p / 90.0), 0, 4).astype(np.int64)
```

iii. Matches the 5 equal-sized bins specified in the instructions (450 cm / 5 = 90 cm per bin).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data — both use `s:e` slice.

ii. Same indexing: `p = np.clip(pos[s:e], 0.0, None)`

iii. Sample-aligned by construction.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = get('lick')
...
lk = lick[s:e]
out[3] = (lk > 0).astype(np.int64)
```

iii. The lick variable records lick events per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
out[3] = (lk > 0).astype(np.int64)
```

iii. Instructions specify binary output (no/yes). Raw lick values can be >1.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data — both use `s:e` slice.

ii. Same indexing: `lk = lick[s:e]`

iii. Sample-aligned by construction.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the VR scene name in the NWB `identifier` field. The scene name encodes the reward zone(s), e.g., `Env1_LocationB` or `Env1_LocationB_to_A`. On switch days, the zone changes after trial 30.

ii.
```python
def scene_reward_zones(scene, ntrials, change_trial=SWITCH_TRIAL):
    m = re.match(r'^Env\d_Location([ABC])$', scene)
    if m:
        return [m.group(1)] * ntrials
    m = re.match(r'^Env\d_Location([ABC])_to_([ABC])$', scene)
    if m is None:
        m = re.match(r'^Env\d_([ABC])_to_Env\d_([ABC])$', scene)
    if m:
        return [m.group(1)] * change_trial + [m.group(2)] * (ntrials - change_trial)
```

iii. This follows `behavior.get_reward_zones` from the paper's code. The agent verified that zone labels from scene names exactly matched the positions where reward_zone was active.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The zone label (A, B, or C) is converted to an integer index (0, 1, 2). The value is constant across all timepoints within a trial.

ii.
```python
out[4] = ZONE_NAMES.index(zone_labels[i])
```

iii. Direct mapping from zone name to index.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` time series timestamps and the `reward_zone` behavioral time series.

ii.
```python
reward_frames = np.searchsorted(tstamps, b['Reward/timestamps'][:])
rzone_entry = get('reward_zone')
...
got_reward = np.any((reward_frames >= s) & (reward_frames < e))
in_zone = np.any(rzone_entry[s:e] > 0)
rewarded[i] = int(bool(got_reward) and bool(in_zone))
```

iii. Follows `behavior.get_trial_types`: a trial is rewarded only if a reward was delivered while the mouse was in the reward zone.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if both a reward event occurred AND the reward_zone entry flag was active during the trial, 0 otherwise. The value is constant across all timepoints within a trial.

ii.
```python
rewarded[i] = int(bool(got_reward) and bool(in_zone))
...
out[5] = rewarded[i]
```

iii. The dual condition (reward AND zone entry) matches the paper's definition of a rewarded trial.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior frame count mismatch**: 10 two-plane sessions (m17, m18) had one extra imaging frame. All streams are truncated to the common length.
- **Non-finite values**: Trials with NaN/Inf in neural activity, position, speed, or lick are skipped.
- **Lick sensor errors**: Trials with cumulative lick count anomalies are dropped (81 trials).
- **Sessions with < 2 trials**: Skipped entirely.
- **Position clipping**: Position values are clipped to >= 0.

ii.
```python
nsamp = min(F.shape[1], len(pos))
if F.shape[1] != len(pos):
    F = F[:, :nsamp]
    Fneu = Fneu[:, :nsamp]
    pos = pos[:nsamp]; speed = speed[:nsamp]; ...
...
if not np.all(np.isfinite(act)):
    continue
...
if lick_error[i]:
    continue
...
p = np.clip(pos[s:e], 0.0, None)
```

iii. The frame count mismatch was identified in 10 sessions. The lick error count (81 trials) matched the paper exactly.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files and reading large arrays** (F, Fneu, behavior) — I/O bound
2. **dF/F computation**: nansmooth, min/max filtering across all cells and frames
3. **OASIS deconvolution**: per-trial deconvolution of dF/F
4. **Saving the final pickle** (~9.6 GB)

ii. N/A

iii. Each session takes ~6-10 seconds. Multiprocessing (8 workers) parallelizes session processing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The interneuron detection loop is already vectorized (matrix operations on dff_valid). The per-trial loop in `process_session` iterates sequentially but is hard to vectorize due to variable trial lengths. The `compute_activity` per-trial baseline loop could potentially be vectorized if trials had uniform length.

ii. N/A

iii. Variable trial lengths make full vectorization impractical.

## 13-c. What processing does the code repeat multiple times?

i. The code computes BOTH dF/F and deconvolved events for every session, even though only one signal (dF/F by default) is used as the neural output. The deconvolution step is unnecessary when using dF/F mode.

ii.
```python
dff, events, valid = compute_activity(F, Fneu, starts, stops, frame_rate)
...
activity = events if signal == 'events' else dff  # default: dff
```

iii. Computing both signals is wasteful when only one is needed, but ensures both are available if the signal choice changes.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) OASIS deconvolution is always computed but discarded when using dF/F mode (the default). (2) The `plane_idx` array is carried through processing but only used for brain_region_idx (all CA1), where it could be replaced by a simple zero array.

ii.
```python
events = np.full(F.shape, np.nan, dtype=np.float32)
for s, e in zip(starts, stops):
    ...
    events[:, s:e] = dcnv.oasis(...)  # computed but discarded in dff mode
```

iii. The deconvolution adds ~1-2 seconds per session of unnecessary computation in dF/F mode.
