# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every `.nwb` file under `/app/data/sub-*` with `h5py`. Each NWB file is treated as one session, and all sessions found by the glob are iterated over.

ii.
```python
def convert():
    files = sorted(glob.glob(DATA_ROOT + '/sub-*/*.nwb'))
    if not files:
        raise FileNotFoundError('No NWB files found')
    subjects = sorted({Path(f).parent.name.removeprefix('sub-') for f in files},
                      key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    ...
    for si,f in enumerate(files):
        with h5py.File(f,'r') as h:
```

iii. In trajectory steps 5, 7, 21, and 32, the agent states that the DANDI release contains 152 NWB session files across 11 subjects and explicitly decides to keep all released sessions rather than restricting to the 77-session remapping subset from the paper.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `sub-*` directory names, and each session also reads `general/subject/subject_id` from the NWB file.

ii.
```python
subjects = sorted({Path(f).parent.name.removeprefix('sub-') for f in files},
                  key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
subj_map = {s:i for i,s in enumerate(subjects)}
...
subject = text(h['general/subject/subject_id'][()])
...
subject_idx.append(subj_map[subject])
```

iii. In trajectory steps 5 and 7, the agent reports that the dataset is organized as a DANDI/BIDS-style hierarchy with 11 subject directories and treats those directories as the subject split.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session.

ii.
```python
for si,f in enumerate(files):
    with h5py.File(f,'r') as h:
        subject = text(h['general/subject/subject_id'][()])
        session_id = text(h['general/session_id'][()])
```

iii. In trajectory steps 5, 7, and 21, the agent repeatedly describes the 152 NWB files as 152 sessions and uses that as the session unit throughout conversion.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from framewise behavior streams. Trial starts are every nonzero `trial_start` frame, and trial ends are every nonzero `teleport` frame. Trials are then taken as frame slices `[trial_start, teleport)`.

ii.
```python
b = h['processing/behavior/BehavioralTimeSeries']
starts = np.flatnonzero(b['trial_start/data'][:])
ends = np.flatnonzero(b['teleport/data'][:])
if len(starts) != len(ends):
    raise ValueError(f'Unmatched boundaries in {f}')
valid = (ends > starts)
starts, ends = starts[valid], ends[valid]
...
for ti,(a,e) in enumerate(zip(starts,ends)):
    sl = slice(int(a),int(e))
```

iii. In the module docstring and trajectory steps 10, 20, 21, and 33, the agent says the repository uses complete trials bounded by `trial_start` and `teleport`, and that the correct interval is `[trial_start, teleport)`.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not apply a trial-length quality filter. It only drops malformed trial boundaries where `end <= start`, and later skips whole sessions if they would end up with fewer than 2 retained trials.

ii.
```python
if len(starts) != len(ends):
    raise ValueError(f'Unmatched boundaries in {f}')
valid = (ends > starts)
starts, ends = starts[valid], ends[valid]
...
if len(ns) < 2 or n_cells==0:
    continue
```

iii. The trajectory does not mention a minimum-length trial filter. The agent instead focused on keeping all complete start/teleport pairs (steps 11, 21, and 33) and only ensuring that each saved session still has at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derives `neural` directly from NWB `processing/ophys/Deconvolved/*/data`, plus `ImageSegmentation/PlaneSegmentation/iscell` for ROI selection.

ii.
```python
deconv = h['processing/ophys/Deconvolved']
plane_names = sorted(deconv.keys(), key=lambda q: int(re.search(r'([0-9]+)$',q).group(1)))
event_sets = [deconv[q]['data'] for q in plane_names]
...
iscell = h['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:,0] > 0
```

iii. In the docstring and trajectory steps 6, 8, 10, 21, and 33, the agent says the paper’s analyses use deconvolved calcium events and explicitly chooses the stored Suite2p deconvolved events rather than recomputing from fluorescence.

## 2-b. How is the `neural` data processed?

i. Processing is minimal: the agent orders planes, handles dual-plane sessions, selects `iscell` ROIs per plane, slices each trial by frame index, concatenates neurons across planes, and transposes from time-by-ROI to neuron-by-time. It does not recompute dF/F or deconvolution.

ii.
```python
plane_names = sorted(deconv.keys(), key=lambda q: int(re.search(r'([0-9]+)$',q).group(1)))
event_sets = [deconv[q]['data'] for q in plane_names]
...
plane_cell_ids=[]; off=0
for width in widths:
    plane_cell_ids.append(np.flatnonzero(iscell[off:off+width]))
    off += width
...
n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T
                    for d,ids in zip(event_sets,plane_cell_ids)],axis=0)
```

iii. In trajectory steps 8, 21, 24, 25, 29, and 33, the agent justifies this as using already frame-aligned deconvolved activity from the NWBs, with special handling for the 28 dual-plane sessions and the 10 sessions with one extra trailing neural frame.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC consists only of keeping ROIs where `iscell[:, 0] > 0`. Sessions with zero retained cells are discarded.

ii.
```python
iscell = h['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:,0] > 0
...
plane_cell_ids.append(np.flatnonzero(iscell[off:off+width]))
...
if len(ns) < 2 or n_cells==0:
    continue
```

iii. The docstring and trajectory steps 8, 21, and 33 say the agent wanted to match Suite2p/manual curation via `iscell` and did not describe any further cell filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start simply by cutting each trial at the trial-start frame and making trial-relative slices `[start, end)`.

ii.
```python
for ti,(a,e) in enumerate(zip(starts,ends)):
    sl = slice(int(a),int(e))
    ...
    n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T
                        for d,ids in zip(event_sets,plane_cell_ids)],axis=0)
```

iii. In the docstring and trajectory steps 10, 20, 21, and 33, the agent says behavior and neural events are already sampled on imaging frames, so trial-start alignment only requires splitting trials at `trial_start`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent keeps the native imaging-frame resolution and applies no rebinning or resampling. It records the bin size as `1000.0/15.5078125` ms.

ii.
```python
'metadata': {
    ...
    'time_bin_size':1000.0/15.5078125,
    ...
}
```

iii. In trajectory steps 6, 20, 21, 29, and 33, the agent says the data streams already share the native imaging-frame clock at about 15.5 Hz and should be aligned by frame without further temporal resampling.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `position/timestamps`.

ii.
```python
ts = b['position/timestamps'][:]
...
t = np.asarray(ts[sl]-ts[a], dtype=np.float32)
```

iii. In trajectory steps 8, 20, 21, and 29, the agent states that the behavior streams are already on a common framewise clock and uses the per-frame timestamps directly.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the timestamp at the first frame is subtracted from all timestamps in that trial.

ii.
```python
t = np.asarray(ts[sl]-ts[a], dtype=np.float32)
```

iii. The trajectory does not add extra justification beyond using frame-aligned timestamps and measuring time relative to the trial start (steps 20 and 21).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The agent aligns it by using the same per-trial slice `sl` that is used for the neural data.

ii.
```python
for ti,(a,e) in enumerate(zip(starts,ends)):
    sl = slice(int(a),int(e))
    ...
    t = np.asarray(ts[sl]-ts[a], dtype=np.float32)
    ...
    n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T
                        for d,ids in zip(event_sets,plane_cell_ids)],axis=0)
```

iii. In trajectory steps 8, 20, 21, and 29, the agent explicitly says the behavior timestamps and neural frames are already aligned, so sharing the same slice is sufficient.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The agent derives environment type from the NWB root `identifier` string, not from the framewise `environment` behavior time series.

ii.
```python
def parse_scene(identifier):
    scene = text(identifier).rstrip('/').split('/')[-1]
    ...
    pairs = re.findall(r'Env([12])(?:_Location)?([ABC])', scene)
    ...
    return scene, int(e0)-1, int(e1)-1, z0, z1
...
scene,e0,e1,z0,z1 = parse_scene(h['identifier'][()])
```

iii. In trajectory steps 13, 15, 20, and 21, the agent says the raw `environment` stream looked sparse or event-like, while the root `identifier` encoded the scene directly, so scene metadata was treated as the authoritative source.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent parses `Env1`/`Env2` from the scene identifier, converts them to `0/1`, and for sessions with `_to_` changes switches from the first environment to the second at trial index 30.

ii.
```python
switched = '_to_' in scene
for ti,(a,e) in enumerate(zip(starts,ends)):
    ...
    after = switched and ti >= 30
    env = e1 if after else e0
    ...
    x = np.vstack((t,
        np.full(ntime,env,np.float32),
        ...
```

iii. In the docstring and trajectory steps 15, 20, and 21, the agent justifies this by saying the repository defines scene-dependent reward zones and uses trial 30 as the default switch point.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is taken from `trial number/data` at the trial’s first frame, with a fallback to the trial loop index if the stored value is missing or negative.

ii.
```python
trialnum_all = b['trial number/data'][:]
...
trnum = float(trialnum_all[a])
if not np.isfinite(trnum) or trnum < 0: trnum = float(ti)
```

iii. The trajectory does not discuss this choice in detail. The code suggests the agent trusted the stored trial number when available but added a fallback for invalid entries.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation beyond reading the per-trial value and broadcasting it across all timepoints in that trial.

ii.
```python
x = np.vstack((t,
    np.full(ntime,env,np.float32),
    np.full(ntime,trnum,np.float32),
    np.full(ntime,prev,np.float32)))
```

iii. The trajectory contains no separate justification beyond the fallback logic in code.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps` together with per-trial frame timestamps from `position/timestamps`.

ii.
```python
ts = b['position/timestamps'][:]
...
reward_times = b['Reward/timestamps'][:]
...
for a,e in zip(starts,ends):
    outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
```

iii. In the docstring and trajectory steps 8, 14, 20, and 21, the agent treats actual reward delivery times as the authoritative reward-outcome signal.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent first computes a per-trial binary reward outcome for every trial, then assigns each trial the previous trial’s outcome, with the first trial set to 0. The value is broadcast across all frames in the trial.

ii.
```python
outcomes = []
for a,e in zip(starts,ends):
    outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
...
prev = float(outcomes[ti-1] if ti else 0)
...
x = np.vstack((t,
    np.full(ntime,env,np.float32),
    np.full(ntime,trnum,np.float32),
    np.full(ntime,prev,np.float32)))
```

iii. In trajectory steps 14, 20, and 21, the agent says the requested variable is binary and should reflect actual reward delivery on the previous trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position/data` plus the active reward-zone label inferred from the NWB `identifier` scene string and the trial index. The raw `reward_zone` behavior stream is not used.

ii.
```python
scene,e0,e1,z0,z1 = parse_scene(h['identifier'][()])
...
pos_all = b['position/data'][:]
...
after = switched and ti >= 30
zone = z1 if after else z0
lo,hi = ZONE_COORDS[zone]
pos = np.asarray(pos_all[sl], dtype=np.float32)
...
distance_class(pos,lo,hi)
```

iii. In trajectory steps 13, 15, 20, and 21, the agent argues that the framewise `reward_zone` channel is only event-like and that the scene identifier plus paper-defined reward-zone coordinates are the right way to label the active zone.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The agent computes signed distance to the active reward zone: negative before the zone, zero inside it, positive after it. It then converts that signed distance directly into the requested 7 categories.

ii.
```python
def distance_class(pos, lo, hi):
    d = np.where(pos < lo, pos-lo, np.where(pos > hi, pos-hi, 0.0))
    y = np.empty(d.shape, np.int8)
    y[d < -50] = 0
    y[(d >= -50) & (d < -10)] = 1
    y[(d >= -10) & (d < 0)] = 2
    y[d == 0] = 3
    y[(d > 0) & (d <= 10)] = 4
    y[(d > 10) & (d <= 50)] = 5
    y[d > 50] = 6
    return y
```

iii. The docstring and trajectory step 20 explicitly describe this signed-distance rule and the use of the paper’s fixed zone boundaries.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Thresholding is hard-coded in `distance_class` using the bin edges `< -50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, and `> 50`.

ii.
```python
y[d < -50] = 0
y[(d >= -50) & (d < -10)] = 1
y[(d >= -10) & (d < 0)] = 2
y[d == 0] = 3
y[(d > 0) & (d <= 10)] = 4
y[(d > 10) & (d <= 50)] = 5
y[d > 50] = 6
```

iii. The trajectory does not separately justify the thresholds beyond saying they were taken from the task specification (step 21).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by applying `distance_class` to the same trial slice `sl` used for neural data.

ii.
```python
sl = slice(int(a),int(e))
pos = np.asarray(pos_all[sl], dtype=np.float32)
...
y = np.vstack((
    distance_class(pos,lo,hi),
    ...
))
...
n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T
                    for d,ids in zip(event_sets,plane_cell_ids)],axis=0)
```

iii. In trajectory steps 20, 21, and 29, the agent says position and neural data are already frame-aligned and should share the same frame indices.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from `position/data`.

ii.
```python
pos_all = b['position/data'][:]
...
pos = np.asarray(pos_all[sl], dtype=np.float32)
```

iii. The trajectory treats the stored position stream as the framewise VR position signal needed for decoder outputs (steps 8, 10, 20, and 21).

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial position slice is discretized into 5 bins with edges `[90, 180, 270, 360]`.

ii.
```python
y = np.vstack((
    distance_class(pos,lo,hi),
    np.digitize(pos,[90,180,270,360],right=False).astype(np.int8),
    ...
))
```

iii. The trajectory does not add separate justification beyond following the decoder task specification (step 21).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Thresholding uses `np.digitize(pos, [90, 180, 270, 360], right=False)`, giving 5 track-position categories.

ii.
```python
np.digitize(pos,[90,180,270,360],right=False).astype(np.int8)
```

iii. The trajectory does not separately discuss this beyond implementing the requested bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by using the same per-trial frame slice as the neural data.

ii.
```python
sl = slice(int(a),int(e))
pos = np.asarray(pos_all[sl], dtype=np.float32)
...
n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T
                    for d,ids in zip(event_sets,plane_cell_ids)],axis=0)
```

iii. The trajectory says the behavior and neural streams share the imaging-frame clock, so shared indexing is sufficient (steps 20, 21, and 29).

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `lick/data`.

ii.
```python
lick_all = b['lick/data'][:]
...
(np.asarray(lick_all[sl])>0).astype(np.int8)
```

iii. In the docstring and trajectory steps 13, 20, and 21, the agent says the raw lick variable is a framewise count and should be converted to presence/absence.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick counts are binarized as `> 0`.

ii.
```python
(np.asarray(lick_all[sl])>0).astype(np.int8)
```

iii. The docstring states that “Licks are frame counts and are converted to presence/absence,” and trajectory steps 13 and 21 make the same point.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by taking the lick values from the same trial slice as the neural data.

ii.
```python
sl = slice(int(a),int(e))
...
(np.asarray(lick_all[sl])>0).astype(np.int8)
...
n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T
                    for d,ids in zip(event_sets,plane_cell_ids)],axis=0)
```

iii. The trajectory says all relevant behavior streams are already resampled to imaging frames in the NWBs (docstring; steps 8, 20, and 21).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the NWB `identifier` scene string and trial index, using `parse_scene` and the default switch at trial 30.

ii.
```python
scene,e0,e1,z0,z1 = parse_scene(h['identifier'][()])
...
after = switched and ti >= 30
zone = z1 if after else z0
...
np.full(ntime,ZONE_INDEX[zone],np.int8)
```

iii. In trajectory steps 15, 20, and 21, the agent argues that the identifier encodes the session scene and therefore the active reward-zone label more reliably than the raw `reward_zone` event stream.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent parses the first and second zone labels from the identifier, optionally switches at trial 30 if the scene contains `_to_`, maps `A/B/C` to `0/1/2`, and broadcasts the resulting category across all frames in the trial.

ii.
```python
ZONE_INDEX = {'A': 0, 'B': 1, 'C': 2}
...
switched = '_to_' in scene
...
after = switched and ti >= 30
zone = z1 if after else z0
...
np.full(ntime,ZONE_INDEX[zone],np.int8)
```

iii. The docstring and trajectory steps 15, 20, and 21 say the paper code maps A/B/C to fixed zones and uses trial 30 as the default switch point.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward/timestamps` and per-trial timestamp intervals from `position/timestamps`.

ii.
```python
ts = b['position/timestamps'][:]
reward_times = b['Reward/timestamps'][:]
...
for a,e in zip(starts,ends):
    outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
```

iii. In the docstring and trajectory steps 14, 20, and 21, the agent says reward outcome should reflect actual reward delivery within the trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the agent checks whether any reward timestamp falls within that trial’s `[start, end)` interval. The resulting binary outcome is then broadcast across all frames in the trial.

ii.
```python
outcomes = []
for a,e in zip(starts,ends):
    outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
...
y = np.vstack((
    ...
    np.full(ntime,outcomes[ti],np.int8)))
```

iii. The trajectory justifies this as using actual reward delivery, with a binary per-trial outcome requested by the task (steps 14, 20, and 21).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles a few consistency issues but not many missing-data cases. It raises an error if the numbers of start and end boundaries differ, drops boundaries with `end <= start`, allows neural streams to exceed behavior by exactly one trailing frame, falls back to the trial loop index if the stored trial number is invalid, and skips sessions with no retained cells or fewer than 2 trials.

ii.
```python
if len(starts) != len(ends):
    raise ValueError(f'Unmatched boundaries in {f}')
valid = (ends > starts)
starts, ends = starts[valid], ends[valid]
...
if any(d.shape[0] < len(ts) or d.shape[0]-len(ts) > 1 for d in event_sets):
    raise ValueError(f'Behavior/neural frame mismatch in {f}')
...
trnum = float(trialnum_all[a])
if not np.isfinite(trnum) or trnum < 0: trnum = float(ti)
...
if len(ns) < 2 or n_cells==0:
    continue
```

iii. In trajectory steps 24 through 30, the agent explicitly investigates two-plane indexing and one-extra-frame mismatches, concluding that exactly one trailing neural frame should be tolerated while true misalignment should still error out.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading each NWB with `h5py`, reading the large deconvolved arrays, iterating through every session/trial to slice and concatenate neural matrices, and serializing the final large pickle.

ii.
```python
for si,f in enumerate(files):
    with h5py.File(f,'r') as h:
        ...
        for ti,(a,e) in enumerate(zip(starts,ends)):
            ...
            n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T
                                for d,ids in zip(event_sets,plane_cell_ids)],axis=0)
...
with open(OUT+'.tmp','wb') as fp: pickle.dump(data,fp,protocol=4)
os.replace(OUT+'.tmp',OUT)
```

iii. In trajectory steps 5, 11, 21, 31, 32, and 33, the agent notes that the NWB files are large, estimates pickle size in the 10-13 GB range, and polls conversion for a long time while the file is written.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the loop that precomputes `outcomes` over trials and the main per-trial loop that repeatedly slices arrays, builds `np.full` blocks, digitizes position/speed, and concatenates neural data.

ii.
```python
outcomes = []
for a,e in zip(starts,ends):
    outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
...
for ti,(a,e) in enumerate(zip(starts,ends)):
    ...
    x = np.vstack((t,
        np.full(ntime,env,np.float32),
        np.full(ntime,trnum,np.float32),
        np.full(ntime,prev,np.float32)))
    y = np.vstack((
        distance_class(pos,lo,hi),
        np.digitize(pos,[90,180,270,360],right=False).astype(np.int8),
        np.digitize(speed,[2,10,20,40],right=False).astype(np.int8),
        (np.asarray(lick_all[sl])>0).astype(np.int8),
        np.full(ntime,ZONE_INDEX[zone],np.int8),
        np.full(ntime,outcomes[ti],np.int8)))
```

iii. The trajectory does not discuss vectorization directly. This assessment comes from the code structure itself.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats per-trial slicing of the same session arrays for every variable, builds a separate `outcomes` pass before the main trial loop, and repeatedly allocates `np.full` arrays for per-trial constants.

ii.
```python
outcomes = []
for a,e in zip(starts,ends):
    outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
...
for ti,(a,e) in enumerate(zip(starts,ends)):
    sl = slice(int(a),int(e))
    ...
    x = np.vstack((t,
        np.full(ntime,env,np.float32),
        np.full(ntime,trnum,np.float32),
        np.full(ntime,prev,np.float32)))
    y = np.vstack((
        distance_class(pos,lo,hi),
        ...
        np.full(ntime,ZONE_INDEX[zone],np.int8),
        np.full(ntime,outcomes[ti],np.int8)))
```

iii. The trajectory focuses more on correctness than efficiency, but steps 21 and 33 emphasize the large dataset size, making these repeated per-trial allocations notable.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There is little obviously discarded processing. The most likely candidate is the extra first pass over all trials to compute `outcomes`, which is then only used to create two per-trial broadcast variables (`previous trial outcome` and `reward outcome`). Otherwise the code mainly computes values that are saved.

ii.
```python
outcomes = []
for a,e in zip(starts,ends):
    outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
```

iii. The trajectory does not identify any intentionally throwaway processing. Most work directly feeds the saved dataset.
