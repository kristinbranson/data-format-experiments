# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI recursively finds every `.nwb` file under `/app/data`, sorts them, and processes each file as one session with `pynwb`. Within each NWB it loads a fixed set of behavioral time series and the ophys `Deconvolved` ROI response series, then truncates all frame-aligned streams to a common length before trial extraction.

ii.
```python
files=sorted(DATA_ROOT.rglob('*.nwb'))
...
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read()
    bts = nwb.processing['behavior']['BehavioralTimeSeries']
    needed = ['environment', 'lick', 'position', 'scanning', 'speed',
              'teleport', 'trial number', 'trial_start']
    arrays = {k: np.asarray(bts.time_series[k].data[:]).squeeze() for k in needed}
    deconv_container = nwb.processing['ophys']['Deconvolved']
    series = list(deconv_container.roi_response_series.values())
    common_n = min([r.data.shape[0] for r in series] + [len(v) for v in arrays.values()])
```

iii. In `CONVERSION_NOTES.md`, the AI says the release contains 152 NWB sessions and that all NWB access must use `pynwb`. It also says sequential per-session loading was chosen to keep memory use manageable on the large source dataset.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from `nwb.subject.subject_id` for each file. After all sessions are processed, unique subject IDs are sorted into `subjects`, and `subject_idx` maps each session back to that list.

ii.
```python
subject = str(nwb.subject.subject_id)
...
subjects.append(info['subject'])
...
subject_names=sorted(set(subjects))
data={
  'subjects':subject_names,
  'subject_idx':np.asarray([subject_names.index(x) for x in subjects],dtype=np.int32),
```

iii. The notes say there are 11 subjects in the release and that subject IDs are consistently present in the NWB metadata, so the AI preferred reading subject identity from the file contents rather than reconstructing it from directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The top-level conversion loop iterates over the sorted file list and appends one session entry each to `neural`, `input`, `output`, `subject_idx`, and `brain_region_idx`.

ii.
```python
files=sorted(DATA_ROOT.rglob('*.nwb'))
...
for j,path in enumerate(files):
    sn,si,so,info=convert_session(path,args.show_processing and j<2)
    neural.append(sn); inputs.append(si); outputs.append(so); infos.append(info); subjects.append(info['subject'])
```

iii. In the notes, the AI explicitly describes the release as “152 NWB files arranged as `sub-<id>/sub-<id>_ses-<nn>_behavior+ophys.nwb`” and treats “one file per imaging session” as the session definition.

## 1-d. How are the data split into trials?

i. Trials are defined by grouping contiguous samples with the same nonnegative native `trial number` value. The code iterates over unique nonnegative trial IDs, gathers all frame indices for each ID, and uses those frames as the trial window. It does not derive start and stop indices directly from `trial_start` and `teleport`.

ii.
```python
trial_ids = np.unique(arrays['trial number'][np.isfinite(arrays['trial number']) &
                                             (arrays['trial number'] >= 0)]).astype(int)
...
for tid in trial_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
```

iii. The notes say the NWB files have no NWB trials table and that the AI considered native `trial number` grouping more robust than depending on `teleport` pulse counts, because resampling made some otherwise complete trials show zero or two positive teleport samples.

## 1-e. How are trials filtered based on quality controls?

i. A trial is kept only if it has at least 20 frames, contains at least one positive `trial_start` sample, reaches at least 440 cm in `position`, and has `scanning == 1` throughout. Sessions with fewer than two surviving trials are rejected.

ii.
```python
valid_ids = []
for tid in trial_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
    complete = (len(ix) >= 20 and np.sum(arrays['trial_start'][ix] > 0) >= 1
                and np.nanmax(arrays['position'][ix]) >= 440
                and np.all(arrays['scanning'][ix] == 1))
    if complete:
        valid_ids.append(tid)
...
if len(valid_ids) < 2:
    raise ValueError(f'Fewer than two valid trials in {path}')
```

iii. The notes say this rule was chosen to exclude only the single malformed terminal fragment while keeping all complete numbered trials, instead of applying a larger ad hoc minimum-length cutoff.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` directly from the NWB ophys `Deconvolved` `RoiResponseSeries`, after applying the Suite2p `iscell` mask through each series’ linked ROI rows.

ii.
```python
deconv_container = nwb.processing['ophys']['Deconvolved']
series = list(deconv_container.roi_response_series.values())
...
linked_rows = np.asarray(r.rois.data[:], dtype=np.int64)
iscell = np.asarray(r.rois.table['iscell'][:])
binary = iscell[:, 0] if iscell.ndim == 2 else iscell
local_cell_mask = (binary[linked_rows] == 1)
neural_parts.append(np.asarray(r.data[:common_n, local_cell_mask], dtype=np.float32))
```

iii. The notes repeatedly justify this by saying the released NWB already contains the reference analysis signal as deconvolved calcium activity and that recomputing dF/F is unnecessary for the decoder task.

## 2-b. How is the `neural` data processed?

i. Processing is minimal: for each response series, the code truncates to the common session length, filters to `iscell` ROIs, casts to `float32`, concatenates planes across columns, and later slices trial windows and transposes them to neuron-by-time matrices. It does not recompute dF/F, smooth, or deconvolve.

ii.
```python
common_n = min([r.data.shape[0] for r in series] + [len(v) for v in arrays.values()])
...
neural_parts.append(np.asarray(r.data[:common_n, local_cell_mask], dtype=np.float32))
...
neural_all = np.ascontiguousarray(np.concatenate(neural_parts, axis=1), dtype=np.float32)
...
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
```

iii. The notes say the AI intentionally preserved “native event amplitudes” and treated the released NWB deconvolved signal as already aligned, curated, and suitable for decoder input.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural quality filter is Suite2p’s binary `iscell` classification. No additional interneuron or activity-based filtering is applied.

ii.
```python
iscell = np.asarray(r.rois.table['iscell'][:])
binary = iscell[:, 0] if iscell.ndim == 2 else iscell
local_cell_mask = (binary[linked_rows] == 1)
neural_parts.append(np.asarray(r.data[:common_n, local_cell_mask], dtype=np.float32))
```

iii. In the notes, the AI argues that place-cell, reward-relative-cell, and other scientific subset labels are downstream analysis choices, not recording-quality filters, so only `iscell` should be enforced for a general decoder.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the start of each numbered trial by extracting exactly the native frame indices belonging to that trial. No interpolation or offsetting is applied beyond the trial split itself.

ii.
```python
for tid in valid_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
    ...
    neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
```

iii. The notes say the release is already frame-aligned and that the requested event is trial start, so slicing the shared trial window is enough.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses a fixed native frame interval of `DT = 0.06448362720402656` s, stored as `time_bin_size = DT*1000.0` ms in metadata. No temporal rebinning or resampling is applied.

ii.
```python
DT = 0.06448362720402656
...
inp[0] = np.arange(nt, dtype=np.float32) * np.float32(DT)
...
'metadata':{
    'time_bin_size':DT*1000.0,
    'sampling_rate_hz':1.0/DT,
```

iii. The notes say all frame-aligned streams in the release use the same ~64.48 ms interval and that native temporal bins should be preserved because the decoder task requires time-varying outputs.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is not derived from stored timestamps directly. Instead, it is derived from the number of frames in the extracted trial window together with the fixed constant `DT`.

ii.
```python
DT = 0.06448362720402656
...
ix = np.flatnonzero(arrays['trial number'] == tid)
nt = len(ix)
...
inp[0] = np.arange(nt, dtype=np.float32) * np.float32(DT)
```

iii. The notes justify this by saying the release has a uniform fixed frame interval, so explicit per-trial timestamp subtraction is unnecessary.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For a trial with `nt` frames, the code constructs `0, DT, 2*DT, ...` up to `(nt-1)*DT`. The resulting vector is time-varying and starts at exactly zero for each trial.

ii.
```python
inp = np.empty((4, nt), dtype=np.float32)
inp[0] = np.arange(nt, dtype=np.float32) * np.float32(DT)
```

iii. The notes say this choice preserves the native frame grid while making the alignment event correspond to time zero.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: the same trial index set `ix` determines the trial length for `neu`, `inp`, and `out`, and the time vector has exactly one element per neural frame.

ii.
```python
ix = np.flatnonzero(arrays['trial number'] == tid)
nt = len(ix)
...
inp[0] = np.arange(nt, dtype=np.float32) * np.float32(DT)
...
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
if neu.shape[1] != nt or inp.shape[1] != nt or out.shape[1] != nt:
    raise AssertionError('Within-trial temporal mismatch')
```

iii. The notes say no interpolation is needed because behavior and neural data already share the same imaging-frame timeline after common-length truncation.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the frame-aligned behavioral `environment` time series.

ii.
```python
needed = ['environment', 'lick', 'position', 'scanning', 'speed',
          'teleport', 'trial number', 'trial_start']
...
env_values = arrays['environment'][ix]
```

iii. The notes say the environment stream is 0 for ENV1 and 1 for ENV2, consistent with the scene names found in the session metadata.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code takes the modal finite environment value within each trial, parses the scene string against that value, and then repeats the resulting binary environment label across all timepoints in the trial.

ii.
```python
env_values = arrays['environment'][ix]
u, c = np.unique(env_values[np.isfinite(env_values)], return_counts=True)
...
env_mode = u[np.argmax(c)]
env, zone = parse_scene(scene, tid, env_mode)
...
inp[1] = env
```

iii. The notes say this was done so the per-trial environment variable would be constant and robust even in switch sessions, while still being cross-checked against the scene metadata.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived directly from the native frame-aligned `trial number` stream. The unique nonnegative value `tid` for each extracted trial is used as that trial’s number.

ii.
```python
trial_ids = np.unique(arrays['trial number'][np.isfinite(arrays['trial number']) &
                                             (arrays['trial number'] >= 0)]).astype(int)
...
for tid in valid_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
    ...
    inp[2] = tid
```

iii. The notes say the AI chose to preserve the native zero-based trial IDs rather than renumber trials after filtering.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation is applied beyond repeating the scalar native trial ID across all timepoints in the trial.

ii.
```python
inp = np.empty((4, nt), dtype=np.float32)
...
inp[2] = tid
```

iii. The notes treat trial number as per-trial context that should stay constant over the full trial window.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the NWB behavioral `Reward` event timestamps after mapping those events onto trial IDs. The current trial’s `previous trial outcome` is then the previous retained trial’s reward outcome.

ii.
```python
reward = bts.time_series['Reward']
reward_times = np.asarray(reward.timestamps[:], dtype=np.float64)
event_frames = nearest_frame_indices(frame_times, reward_times) if len(reward_times) else np.array([], dtype=int)
event_trial_ids = arrays['trial number'][event_frames] if len(event_frames) else np.array([])
rewarded_ids = set(int(x) for x in event_trial_ids if np.isfinite(x) and x >= 0)
```

iii. The notes say `Reward` timestamps are the authoritative source of rewarded versus omitted trials and that `autoreward` is always zero, so it should not be used.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code initializes `prev_outcome = 0` before the first valid trial. For each valid trial, it writes the previous valid trial’s binary outcome across the whole trial and then updates `prev_outcome` to the current trial’s outcome.

ii.
```python
prev_outcome = 0
for tid in valid_ids:
    ...
    outcome = int(tid in rewarded_ids)
    ...
    inp[3] = prev_outcome
    ...
    prev_outcome = outcome
```

iii. The notes justify this as the natural implementation of the task’s binary “omitted = 0, rewarded = 1” previous-outcome variable, with the first valid trial defaulting to 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the behavioral `position` time series and a per-trial reward-zone identity inferred from the session `scene` string plus the trial’s environment mode. The code does not use the frame-level `reward_zone` behavioral stream.

ii.
```python
scene = str(nwb.identifier).rstrip('/').split('/')[-1]
...
pos = arrays['position'][ix].astype(np.float32)
env_values = arrays['environment'][ix]
...
env, zone = parse_scene(scene, tid, env_mode)
low, high = ZONE_BOUNDS[zone]
dist = signed_distance_to_interval(pos, low, high)
```

iii. The notes say the native `reward_zone` values behaved like transient state codes rather than stable A/B/C labels, so the AI derived the active zone from scene/task structure and cross-validated that against the positions of nonzero reward-state samples.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. After assigning a trial-level reward-zone label, the code computes signed distance from each position sample to the closed zone interval: negative before the interval, zero inside it, positive after it. The zone intervals are hard-coded as A = 80–100 cm, B = 200–220 cm, and C = 320–340 cm.

ii.
```python
ZONE_BOUNDS = {'A': (80.0, 100.0), 'B': (200.0, 220.0), 'C': (320.0, 340.0)}
...
def signed_distance_to_interval(position, low, high):
    return np.where(position < low, position-low,
                    np.where(position > high, position-high, 0.0))
...
low, high = ZONE_BOUNDS[zone]
dist = signed_distance_to_interval(pos, low, high)
```

iii. The notes say this was meant to implement “distance to any location in the reward zone” literally, by making the distance zero anywhere inside the active interval.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is discretized into seven classes with explicit boundary handling: `< -50`, `[-50, -10]`, `(-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii.
```python
def discretize_distance(d):
    y = np.empty(d.shape, dtype=np.int8)
    y[d < -50] = 0
    y[(d >= -50) & (d <= -10)] = 1
    y[(d > -10) & (d < 0)] = 2
    y[d == 0] = 3
    y[(d > 0) & (d <= 10)] = 4
    y[(d > 10) & (d <= 50)] = 5
    y[d > 50] = 6
    return y
```

iii. The notes say these boundary rules were unit-tested and chosen to match the task’s exact class definitions, especially the special zero-distance class.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing distance from the same per-trial frame-indexed `position` samples used to slice the neural trial. No resampling or temporal offsetting is applied.

ii.
```python
ix = np.flatnonzero(arrays['trial number'] == tid)
...
pos = arrays['position'][ix].astype(np.float32)
dist = signed_distance_to_interval(pos, low, high)
...
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
```

iii. The notes say all frame-aligned behavior variables already live on the same imaging-frame axis as the deconvolved activity.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the frame-aligned behavioral `position` time series.

ii.
```python
needed = ['environment', 'lick', 'position', 'scanning', 'speed',
          'teleport', 'trial number', 'trial_start']
...
pos = arrays['position'][ix].astype(np.float32)
```

iii. The notes describe `position` as the animal’s absolute location along the 450 cm corridor and treat it as a primary frame-aligned behavioral output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code discretizes the raw per-frame position values directly, without clipping or removing negative/teleport-adjacent values that remain inside the chosen trial window.

ii.
```python
def discretize_position(x):
    y = np.zeros(x.shape, dtype=np.int8)
    y[(x >= 90) & (x < 180)] = 1
    y[(x >= 180) & (x < 270)] = 2
    y[(x >= 270) & (x <= 360)] = 3
    y[x > 360] = 4
    return y
...
out[1] = discretize_position(pos)
```

iii. The notes say native numbered-trial samples were preserved for temporal fidelity and that any values outside the main corridor would naturally fall into the edge bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into five 90 cm bins: `< 90`, `90–180`, `180–270`, `270–360`, and `> 360`, with exact 360 cm assigned to the fourth bin.

ii.
```python
def discretize_position(x):
    y = np.zeros(x.shape, dtype=np.int8)
    y[(x >= 90) & (x < 180)] = 1
    y[(x >= 180) & (x < 270)] = 2
    y[(x >= 270) & (x <= 360)] = 3
    y[x > 360] = 4
    return y
```

iii. The notes say these equality conventions were checked explicitly because the task wording makes class-4 strictly `> 360`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by taking `position[ix]` from the same trial frame indices `ix` used for the neural slice.

ii.
```python
ix = np.flatnonzero(arrays['trial number'] == tid)
pos = arrays['position'][ix].astype(np.float32)
...
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
```

iii. The notes say all frame-aligned streams share the same fixed timebase after the one-frame common-length truncation.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the frame-aligned behavioral `lick` time series.

ii.
```python
needed = ['environment', 'lick', 'position', 'scanning', 'speed',
          'teleport', 'trial number', 'trial_start']
...
lick = arrays['lick'][ix]
```

iii. The notes say native lick values can exceed 1, but the requested decoder output is binary lick.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick is binarized at each frame with the rule `lick > 0`.

ii.
```python
out[3] = (lick > 0).astype(np.int8)
```

iii. The notes say no smoothing was applied because the task explicitly asks for binary lick output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by slicing the same per-trial frame indices used for the neural data.

ii.
```python
ix = np.flatnonzero(arrays['trial number'] == tid)
lick = arrays['lick'][ix]
...
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
```

iii. The notes say all frame-aligned behavior variables and deconvolved calcium traces are already temporally synchronized on the imaging-frame axis.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from session-level scene metadata (`nwb.identifier`), the trial’s environment mode, and the native trial ID. It is not derived from the frame-level `reward_zone` variable itself.

ii.
```python
scene = str(nwb.identifier).rstrip('/').split('/')[-1]
...
env_mode = u[np.argmax(c)]
env, zone = parse_scene(scene, tid, env_mode)
...
out[4] = ZONE_INDEX[zone]
```

iii. The notes say the AI considered scene/task schedule plus aligned environment to be the only stable way to recover A/B/C identity, because the native `reward_zone` stream encoded transient states rather than direct zone labels.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code parses scene names into stable, reward-switch, or combined environment-plus-reward switch cases. For combined scenes like `Env1_A_to_Env2_B`, the environment mode determines which side of the scene applies. For reward-only switches (`LocationX_to_Y`), the code switches to the second zone at trial ID 40. The resulting zone label is then repeated across all timepoints in the trial.

ii.
```python
def parse_scene(scene, trial_id, env_mode):
    env = int(round(float(env_mode)))
    m = re.fullmatch(r'Env([12])_([ABC])_to_Env([12])_([ABC])', scene)
    if m:
        first_env, first_zone = int(m.group(1))-1, m.group(2)
        second_env, second_zone = int(m.group(3))-1, m.group(4)
        if env == first_env:
            return env, first_zone
        if env == second_env:
            return env, second_zone
    ...
    zone = second if second is not None and trial_id >= 40 else first
    return env, zone
...
out[4] = ZONE_INDEX[zone]
```

iii. The notes say this logic was added after the AI inspected switch sessions and found that combined environment/reward scenes switch at trial 30 while reward-only switch scenes follow the trial-40 rule.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward` event time series, using reward timestamps mapped onto frame indices and then onto native trial IDs.

ii.
```python
reward = bts.time_series['Reward']
reward_times = np.asarray(reward.timestamps[:], dtype=np.float64)
event_frames = nearest_frame_indices(frame_times, reward_times) if len(reward_times) else np.array([], dtype=int)
event_trial_ids = arrays['trial number'][event_frames] if len(event_frames) else np.array([])
rewarded_ids = set(int(x) for x in event_trial_ids if np.isfinite(x) and x >= 0)
```

iii. The notes say reward events are the authoritative source of outcome and that `autoreward` is all zeros, so it was intentionally ignored.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are assigned to the nearest frame on the common imaging/behavior timeline. A trial is labeled rewarded if its native trial ID appears among those event-mapped trial IDs. The binary outcome is then repeated across all timepoints in the trial.

ii.
```python
event_frames = nearest_frame_indices(frame_times, reward_times) if len(reward_times) else np.array([], dtype=int)
event_trial_ids = arrays['trial number'][event_frames] if len(event_frames) else np.array([])
rewarded_ids = set(int(x) for x in event_trial_ids if np.isfinite(x) and x >= 0)
...
out[5] = outcome
```

iii. The notes justify nearest-frame mapping as a way to bring event timestamps onto the shared frame grid before converting outcome into a per-trial categorical variable.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases defensively. It truncates neural and behavioral streams to `common_n` when one stream is one frame longer; excludes malformed/incomplete trials using start/scanning/position/length checks; raises errors if a trial has no finite environment values, if trial matrices become temporally inconsistent, or if neural/input values are nonfinite; and rejects sessions with fewer than two valid trials.

ii.
```python
common_n = min([r.data.shape[0] for r in series] + [len(v) for v in arrays.values()])
arrays = {k: v[:common_n] for k, v in arrays.items()}
...
if not len(u):
    raise ValueError(f'No finite environment in {path}, trial {tid}')
...
if neu.shape[1] != nt or inp.shape[1] != nt or out.shape[1] != nt:
    raise AssertionError('Within-trial temporal mismatch')
if not (np.all(np.isfinite(neu)) and np.all(np.isfinite(inp))):
    raise ValueError(f'Nonfinite neural/input values in {path}, trial {tid}')
...
if len(valid_ids) < 2:
    raise ValueError(f'Fewer than two valid trials in {path}')
```

iii. The notes say these checks came from dataset exploration, especially the ten one-frame stream mismatches and the single malformed terminal trial fragment.

## 13-a. What are the most time-consuming steps of the code?

i. The code’s slowest steps are reading each NWB file, materializing the deconvolved response matrices and required behavior arrays, concatenating across planes, and serializing the final 13.33 GB pickle. Optional sample plotting also adds overhead when enabled.

ii.
```python
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read()
...
for j,path in enumerate(files):
    sn,si,so,info=convert_session(path,args.show_processing and j<2)
...
with out.open('wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In Step 9, the notes report 152 sessions processed in 112.0 s including 13.33 GB pickle serialization and explicitly call NWB I/O plus pickle writing the dominant costs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main non-vectorized work is the per-session loop over files, the per-series loop over ROI response series, and especially the per-trial loop over `valid_ids`. Within each trial, discretization functions are vectorized, but trial discovery and assembly still use Python loops.

ii.
```python
for j,path in enumerate(files):
    sn,si,so,info=convert_session(path,args.show_processing and j<2)
...
for r in series:
    linked_rows = np.asarray(r.rois.data[:], dtype=np.int64)
    ...
for tid in valid_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
    ...
    out[0] = discretize_distance(dist)
    out[1] = discretize_position(pos)
    out[2] = discretize_speed(speed)
```

iii. The notes say sequential per-session processing was intentional for memory reasons. They do not claim to have vectorized the trial loop, and variable trial lengths make fully vectorized trial assembly awkward.

## 13-c. What processing does the code repeat multiple times?

i. Within a session, the code scans trials twice: once to decide which native trial IDs are valid and again to actually build `neural`, `input`, and `output`. It also recomputes `np.flatnonzero(arrays['trial number'] == tid)` separately in both passes. Across the whole conversion, optional sample plotting revisits selected trial arrays after they are assembled.

ii.
```python
for tid in trial_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
    ...
    if complete:
        valid_ids.append(tid)
...
for tid in valid_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
    ...
if len(plot_info) < 4:
    plot_info.append((tid, pos.copy(), speed.copy(), (lick > 0).copy(), dist.copy(), out.copy()))
```

iii. The notes emphasize that they avoided the much larger repeated full-dataset “survey then convert” pattern by doing a single conversion pass, but the code still repeats some per-trial indexing work within each session.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main extra work is diagnostic rather than conversion-critical: collecting `plot_info` for up to four trials, generating optional processing plots, and recording rich per-session metadata such as `n_rois`, native series lengths, and reward-event counts that are not used by the supplied decoder itself.

ii.
```python
plot_info = []
...
if len(plot_info) < 4:
    plot_info.append((tid, pos.copy(), speed.copy(), (lick > 0).copy(), dist.copy(), out.copy()))
...
info = {
    'file': str(path), 'subject': subject, 'scene': scene,
    'n_cells': n_cells, 'n_rois': int(n_rois), 'n_planes': len(series),
    'n_trials': len(valid_ids), 'native_trial_ids': valid_ids,
    'common_n': common_n, 'neural_native_n': neural_native_lengths,
    'reward_events': len(reward_times), 'rewarded_trials': len(set(valid_ids) & rewarded_ids),
    'elapsed_s': time.time()-t0,
}
if make_plot:
    plot_processing(info, plot_info)
```

iii. The notes justify these as sanity-check and validation aids, especially during sample conversion and debugging, rather than as data strictly required for downstream training.
