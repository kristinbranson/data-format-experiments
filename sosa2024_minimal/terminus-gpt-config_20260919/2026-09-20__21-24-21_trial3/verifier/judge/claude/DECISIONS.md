# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are found by globbing `DATA_ROOT/sub-*/*.nwb`. Each file is opened with `h5py` (not `pynwb`). All 152 NWB files across 11 subjects are loaded. Data is read directly from HDF5 paths within each file.

ii.
```python
files = sorted(glob.glob(DATA_ROOT + '/sub-*/*.nwb'))
...
for si,f in enumerate(files):
    with h5py.File(f,'r') as h:
        subject = text(h['general/subject/subject_id'][()])
        ...
```

iii. The agent identified 152 NWB files across 11 subjects. It noted that the paper's "77 sessions" refers to the reward-switch subset, not a quality filter, and decided all 152 sessions should be retained. The agent used h5py rather than pynwb for direct HDF5 access.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `general/subject/subject_id` field within each NWB file. Unique subjects are collected from directory names (`sub-*` prefix removed) and sorted numerically.

ii.
```python
subjects = sorted({Path(f).parent.name.removeprefix('sub-') for f in files},
                  key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
subj_map = {s:i for i,s in enumerate(subjects)}
...
subject = text(h['general/subject/subject_id'][()])
```

iii. The agent used the directory structure to enumerate subjects and the NWB metadata to confirm subject identity per session.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All 152 NWB files are processed as individual sessions.

ii.
```python
files = sorted(glob.glob(DATA_ROOT + '/sub-*/*.nwb'))
...
for si,f in enumerate(files):
    with h5py.File(f,'r') as h:
        ...
```

iii. The agent noted the BIDS-style naming convention (one file per session) and confirmed that the 152-file count matches the full dataset release.

## 1-d. How are the data split into trials?

i. Trial boundaries are determined using `trial_start` (nonzero frames mark trial starts) and `teleport` (nonzero frames mark trial ends). Trials run from trial_start to teleport, exclusive.

ii.
```python
starts = np.flatnonzero(b['trial_start/data'][:])
ends = np.flatnonzero(b['teleport/data'][:])
if len(starts) != len(ends):
    raise ValueError(f'Unmatched boundaries in {f}')
valid = (ends > starts)
starts, ends = starts[valid], ends[valid]
```

iii. The agent characterized `trial_start` and `teleport` as "impulses" from its data inspection. It uses `np.flatnonzero` to find all nonzero frames, then filters to ensure end > start. The assertion that starts and ends have equal length validates that teleport is indeed single-frame.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 trials or 0 cells are skipped. Trials where `ends <= starts` are removed. No minimum trial length filter is applied.

ii.
```python
valid = (ends > starts)
starts, ends = starts[valid], ends[valid]
...
if len(ns) < 2 or n_cells==0:
    continue
```

iii. The agent did not mention a minimum trial length filter. The only filtering is to ensure valid trial boundaries and sufficient data per session for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the NWB's `Deconvolved` response series (suite2p's deconvolved calcium events), NOT from the raw `Fluorescence` and `Neuropil` traces.

ii.
```python
deconv = h['processing/ophys/Deconvolved']
plane_names = sorted(deconv.keys(), key=lambda q: int(re.search(r'([0-9]+)$',q).group(1)))
event_sets = [deconv[q]['data'] for q in plane_names]
...
n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T
                    for d,ids in zip(event_sets,plane_cell_ids)],axis=0)
```

iii. The agent stated that the paper uses "deconvolved calcium events" and chose to use the pre-computed `Deconvolved` data from the NWB. The agent's trajectory notes confirm it identified deconvolved activity as the paper's neural measure, but did not distinguish between suite2p's own deconvolution (stored in NWB) and the paper's custom dF/F + OASIS pipeline.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied. The pre-stored `Deconvolved` data is read directly and sliced per trial.

ii.
```python
n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T
                    for d,ids in zip(event_sets,plane_cell_ids)],axis=0)
```

iii. The agent treated the NWB's deconvolved data as ready-to-use. No neuropil subtraction, baseline estimation, dF/F computation, smoothing, or OASIS deconvolution is performed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only suite2p's `iscell` classification is used to filter ROIs. No putative interneuron filtering is applied.

ii.
```python
iscell = h['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:,0] > 0
...
plane_cell_ids=[]; off=0
for width in widths:
    plane_cell_ids.append(np.flatnonzero(iscell[off:off+width]))
    off += width
```

iii. The agent applied suite2p's manual curation (`iscell`) but did not implement the paper's additional filter of removing putative interneurons (cells with dF/F-speed correlation > 0.5).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is sliced using the same trial_start and teleport indices as the behavioral data. Since both are frame-aligned in the NWB, no additional temporal alignment is needed.

ii.
```python
sl = slice(int(a),int(e))
...
n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T
                    for d,ids in zip(event_sets,plane_cell_ids)],axis=0)
```

iii. The agent verified that neural and behavioral streams share the same frame indexing within the NWB files.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging rate is used (~15.5 Hz, yielding ~64.5 ms bins). No temporal rebinning is applied. The time bin size is hardcoded as `1000.0/15.5078125` ms.

ii.
```python
'time_bin_size':1000.0/15.5078125,
```

iii. The agent determined the imaging rate from NWB metadata. The hardcoded value matches the scanner rate for single-plane sessions.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` behavior time series.

ii.
```python
ts = b['position/timestamps'][:]
...
t = np.asarray(ts[sl]-ts[a], dtype=np.float32)
```

iii. The agent used position timestamps as the common time base for all behavioral variables.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial-start timestamp is subtracted from each frame's timestamp within the trial.

ii.
```python
t = np.asarray(ts[sl]-ts[a], dtype=np.float32)
```

iii. Simple offset subtraction to get time relative to trial onset.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame indices in the NWB, so slicing by the same trial boundaries ensures alignment. No interpolation or resampling is needed.

ii.
```python
sl = slice(int(a),int(e))
...
t = np.asarray(ts[sl]-ts[a], dtype=np.float32)
n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T ...])
```

iii. Frame-aligned by construction of the NWB files.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the NWB root `identifier` field, which contains the scene description string (e.g., `Env1_LocationB_to_A`). Environment type is parsed using regex.

ii.
```python
scene,e0,e1,z0,z1 = parse_scene(h['identifier'][()])
...
def parse_scene(identifier):
    scene = text(identifier).rstrip('/').split('/')[-1]
    pairs = re.findall(r'Env([12])(?:_Location)?([ABC])', scene)
    if len(pairs) >= 2:
        (e0,z0),(e1,z1) = pairs[0],pairs[1]
    else:
        em = re.search(r'Env([12])', scene)
        ...
    return scene, int(e0)-1, int(e1)-1, z0, z1
```

iii. The agent discovered that the `identifier` field contains scene metadata including environment type. It parses `Env1` as 0 and `Env2` as 1.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For switch sessions (`_to_` in scene), the environment changes at trial 30. Before trial 30, `e0` is used; from trial 30 onward, `e1` is used. For non-switch sessions, the environment is constant.

ii.
```python
switched = '_to_' in scene
for ti,(a,e) in enumerate(zip(starts,ends)):
    after = switched and ti >= 30
    env = e1 if after else e0
    ...
    x = np.vstack((t,
        np.full(ntime,env,np.float32),
        ...))
```

iii. The agent identified that cross-environment switches occur at trial 30 from the paper code's `change_reward_trial` constant.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series in the NWB, with a fallback to the sequential trial index if the value is non-finite or negative.

ii.
```python
trialnum_all = b['trial number/data'][:]
...
trnum = float(trialnum_all[a])
if not np.isfinite(trnum) or trnum < 0: trnum = float(ti)
```

iii. The agent chose to use the stored trial number variable from the NWB data, reading the value at the trial start frame.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number is read from the first frame of each trial. If invalid (non-finite or negative), it falls back to the sequential trial index. The value is constant across all timepoints in a trial.

ii.
```python
trnum = float(trialnum_all[a])
if not np.isfinite(trnum) or trnum < 0: trnum = float(ti)
...
np.full(ntime,trnum,np.float32)
```

iii. The agent used the NWB's stored trial number rather than computing it independently.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series, which has its own timestamps separate from the frame-aligned behavioral streams.

ii.
```python
reward_times = b['Reward/timestamps'][:]
...
outcomes = []
for a,e in zip(starts,ends):
    outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
```

iii. The agent identified reward delivery as a sparse event series with independent timestamps.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, reward outcome is determined by checking if any reward timestamp falls within the trial's time window. Previous trial outcome is then the outcome of the preceding trial; for the first trial, it is set to 0.

ii.
```python
outcomes = []
for a,e in zip(starts,ends):
    outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
...
prev = float(outcomes[ti-1] if ti else 0)
```

iii. The agent computed reward outcome per trial first, then used the previous trial's outcome as the input.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone location parsed from the NWB `identifier` field (scene string). Reward zone boundaries come from hardcoded coordinates: A=[80,130], B=[200,250], C=[320,370] cm.

ii.
```python
ZONE_COORDS = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
...
scene,e0,e1,z0,z1 = parse_scene(h['identifier'][()])
...
zone = z1 if after else z0
lo,hi = ZONE_COORDS[zone]
pos = np.asarray(pos_all[sl], dtype=np.float32)
```

iii. The agent traced the reward zone coordinates from the paper code's mapping of task labels A/B/C to physical positions X/Y/Z.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative before the zone, zero within the zone, positive after it. The `distance_class` function computes the continuous distance and discretizes it in one step.

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

iii. The signed distance computation matches the paper's concept of reward-relative position.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using direct comparison operators in the `distance_class` function. Bin 3 (value "0 cm") is assigned only when `d == 0` exactly.

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

iii. The bin edges match the instructions. The "0 cm" category uses exact floating-point equality (`d == 0`), which captures timepoints inside the reward zone boundaries.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced using the same trial boundaries as neural data, so alignment is by shared frame index.

ii.
```python
sl = slice(int(a),int(e))
pos = np.asarray(pos_all[sl], dtype=np.float32)
```

iii. Frame-aligned by construction.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos_all = b['position/data'][:]
...
pos = np.asarray(pos_all[sl], dtype=np.float32)
```

iii. Direct use of position data from the NWB.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extraction and discretization. Position values are used as-is from the NWB.

ii.
```python
np.digitize(pos,[90,180,270,360],right=False).astype(np.int8)
```

iii. Raw position values map directly to the 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `np.digitize` with edges `[90, 180, 270, 360]` (no `-inf`/`inf` wrapping). This produces bins 0-4 directly.

ii.
```python
np.digitize(pos,[90,180,270,360],right=False).astype(np.int8)
```

iii. The 5 bins correspond to 90 cm intervals spanning the 450 cm track, matching the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame-aligned slicing as all other variables.

ii.
```python
sl = slice(int(a),int(e))
pos = np.asarray(pos_all[sl], dtype=np.float32)
```

iii. Frame-aligned by construction.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick_all = b['lick/data'][:]
...
(np.asarray(lick_all[sl])>0).astype(np.int8)
```

iii. The lick data records lick counts per imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
(np.asarray(lick_all[sl])>0).astype(np.int8)
```

iii. The instructions specify binary output (no/yes). Raw lick values can be >1 (frame counts), so thresholding at >0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame-aligned slicing.

ii.
```python
sl = slice(int(a),int(e))
(np.asarray(lick_all[sl])>0).astype(np.int8)
```

iii. Frame-aligned by construction.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` field (scene string), which encodes the reward zone labels (A, B, or C). For switch sessions, the zone changes at trial 30.

ii.
```python
scene,e0,e1,z0,z1 = parse_scene(h['identifier'][()])
...
switched = '_to_' in scene
...
after = switched and ti >= 30
zone = z1 if after else z0
...
np.full(ntime,ZONE_INDEX[zone],np.int8)
```

iii. The agent discovered that the scene identifier contains reward zone labels and that switches occur at trial 30, consistent with the paper code's `change_reward_trial` constant.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The zone label (A, B, or C) is mapped to an integer (0, 1, or 2) using a lookup dictionary. For switch sessions, the zone changes from `z0` to `z1` at trial 30. The value is constant across all timepoints in a trial.

ii.
```python
ZONE_INDEX = {'A': 0, 'B': 1, 'C': 2}
...
zone = z1 if after else z0
...
np.full(ntime,ZONE_INDEX[zone],np.int8)
```

iii. The mapping is straightforward: A=0, B=1, C=2 as specified in the instructions.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_times = b['Reward/timestamps'][:]
...
outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
```

iii. Reward delivery is recorded as sparse timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within the trial's time window `[ts[start], ts[end])`. Binary output: 1 if rewarded, 0 otherwise. The value is constant across all timepoints in the trial.

ii.
```python
outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
...
np.full(ntime,outcomes[ti],np.int8)
```

iii. Uses actual reward delivery events rather than reward zone activation.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior frame mismatch**: Up to 1 extra trailing neural frame is tolerated (10 dual-plane sessions have this). A ValueError is raised for larger mismatches.
- **Invalid trial boundaries**: Trials where `end <= start` are filtered out.
- **Session quality**: Sessions with fewer than 2 trials or 0 cells are skipped entirely.
- **Invalid trial numbers**: Non-finite or negative trial numbers fall back to sequential index.

ii.
```python
if any(d.shape[0] < len(ts) or d.shape[0]-len(ts) > 1 for d in event_sets):
    raise ValueError(f'Behavior/neural frame mismatch in {f}')
...
valid = (ends > starts)
starts, ends = starts[valid], ends[valid]
...
if len(ns) < 2 or n_cells==0:
    continue
...
if not np.isfinite(trnum) or trnum < 0: trnum = float(ti)
```

iii. The agent discovered the 1-frame neural surplus empirically during conversion and patched the code to handle it.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Opening and reading NWB files with h5py** — I/O bound, 87 GB of data across 152 files.
2. **Reading deconvolved neural data arrays** — large per-session arrays.
3. **Serializing the final pickle** — the output is large.

ii. N/A

iii. The agent used h5py for faster I/O than pynwb.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop iterates sequentially over trials within each session. The outcome computation (determining reward per trial) is a separate loop. Both could potentially be vectorized, though variable trial lengths make full vectorization difficult.

ii.
```python
for a,e in zip(starts,ends):
    outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
...
for ti,(a,e) in enumerate(zip(starts,ends)):
    ...
```

iii. The per-trial loop structure is natural given variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. The code reads reward timestamps once per session and iterates over trials twice: once to compute outcomes (for previous trial outcome), and once to build the final trial data. This is a minor duplication as the outcome pre-computation is needed for the previous-trial-outcome input.

ii.
```python
# First loop: compute outcomes
for a,e in zip(starts,ends):
    outcomes.append(...)
# Second loop: build trial data
for ti,(a,e) in enumerate(zip(starts,ends)):
    ...
```

iii. The two-pass approach is intentional: outcomes must be known before assigning previous trial outcome.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `session_info` metadata including scene strings, switch trial indices, and file paths. This is stored in the metadata but not used by the decoder. Additionally, `speed` is included as an output (matching instructions) — this is not unnecessary.

ii.
```python
session_info.append({'file':os.path.relpath(f,DATA_ROOT),'subject':subject,
    'session_id':session_id,'scene':scene,'n_trials':len(ns),
    'n_cells':n_cells,'switch_trial':30 if switched else None})
```

iii. The session_info is useful for debugging and provenance but not directly consumed by the decoder.
