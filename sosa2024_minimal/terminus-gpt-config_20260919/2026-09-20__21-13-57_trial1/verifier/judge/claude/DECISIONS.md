# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `glob.glob` to find all NWB files matching the pattern `sub-*/*_behavior+ophys.nwb` under `/app/data`. Each NWB file is opened with `h5py` (not `pynwb`) and data is read directly from HDF5 groups. All subjects and sessions found this way are included.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*_behavior+ophys.nwb')))
if not files:
    raise FileNotFoundError('No NWB files found')
subjects = sorted({os.path.basename(os.path.dirname(p)).replace('sub-','') for p in files})
```

```python
with h5py.File(path, 'r') as f:
    b = f['processing/behavior/BehavioralTimeSeries']
    ...
    dg = f['processing/ophys/Deconvolved']
```

iii. The agent explored the data directory structure and confirmed there were 152 NWB files across 11 subjects. It chose `h5py` over `pynwb` for direct HDF5 access, which is a valid approach.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the directory names of NWB files by stripping the `sub-` prefix. They are sorted alphabetically.

ii.
```python
subjects = sorted({os.path.basename(os.path.dirname(p)).replace('sub-','') for p in files})
subj_to_idx = {s:i for i,s in enumerate(subjects)}
```

iii. The agent observed that all NWB files reside in `sub-<id>` directories and used this to identify subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All sessions are processed.

ii.
```python
for i,p in enumerate(files):
    nt, it, ot, nc, scene, rate, block = convert_session(p)
    if len(nt) < 2:
        print('SKIP (<2 trials)', p, flush=True)
        continue
```

iii. The agent identified that each NWB file represents a single recording session and processes all of them, skipping sessions with fewer than 2 valid trials.

## 1-d. How are the data split into trials?

i. Trials are identified by the `trial_start` pulse (nonzero values in the `trial_start` behavioral time series). Each trial starts at a `trial_start` pulse and ends at the first subsequent nonzero `teleport` sample.

ii.
```python
starts = np.flatnonzero(trial_start > 0)
for s in starts:
    tq = np.flatnonzero(teleport[s:] > 0)
    e = s + int(tq[0]) if len(tq) else len(pos)
    if e <= s:
        continue
    if not len(tq):
        continue
```

iii. The agent investigated the behavioral streams and determined that `trial_start` pulses mark the beginning of each lap (at 0 cm position crossing) and teleport marks the end. This matches the paper's trial definition.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two ways: (1) if fewer than 2 time bins remain after temporal rebinning (`len(offsets) < 2`), the trial is skipped; (2) trials that never reach teleport (clipped final trials) are skipped; (3) sessions with fewer than 2 valid trials are skipped entirely.

ii.
```python
if len(offsets) < 2:
    continue
# Ignore a clipped final trial if it never reaches teleport.
if not len(tq):
    continue
```

```python
if len(nt) < 2:
    print('SKIP (<2 trials)', p, flush=True)
    continue
```

iii. The agent reasoned that very short trials and incomplete final trials should be excluded. The 2-trial minimum per session ensures decoder evaluation is possible.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the NWB's pre-stored `Deconvolved` ophys data (suite2p's OASIS deconvolution stored in the NWB file), NOT from re-running the paper's own dFF + deconvolution pipeline on raw Fluorescence and Neuropil traces.

ii.
```python
dg = f['processing/ophys/Deconvolved']
plane_names = sorted(dg.keys(), key=lambda x: int(x.replace('plane','')))
...
raw_parts = [dset[s:e, :][:, idx] for dset,idx in plane_series]
raw = np.concatenate(raw_parts, axis=1)
```

iii. The agent identified the `Deconvolved` group in the NWB and used it directly, reasoning that it contains the OASIS-deconvolved activity. The agent did not investigate whether this differed from the paper's own processing pipeline.

## 2-b. How is the `neural` data processed?

i. The deconvolved activity is read from the NWB, subsetted to `iscell`-curated neurons, and then block-averaged into ~258 ms time bins (4 frames at 15.5 Hz or equivalent). Multi-plane sessions have neurons concatenated across planes. The data is stored as float16.

ii.
```python
nbin = binned_mean(raw, np.arange(0, e-s, block), block).T.astype(np.float16)
```

```python
def binned_mean(x, starts, block):
    return np.stack([np.asarray(x[s:min(s+block, len(x))]).mean(axis=0)
                     for s in starts], axis=0)
```

iii. The agent calculated that using native resolution would produce ~7.86 GiB of data, so it chose to temporally bin to a common ~258 ms resolution to make the dataset tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only cells marked as accepted by suite2p's `iscell` flag are retained. No additional filtering (e.g., interneuron exclusion) is applied.

ii.
```python
seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = seg['iscell'][:,0] == 1
plane_idx = seg['planeIdx'][:].astype(int) if 'planeIdx' in seg else np.zeros(len(iscell), int)
plane_series = []
for name in plane_names:
    pi = int(name.replace('plane',''))
    dset = dg[name]['data']
    mask = iscell[plane_idx == pi]
    ...
    plane_series.append((dset, np.flatnonzero(mask)))
```

iii. The agent used the `iscell` curation from the NWB's PlaneSegmentation table. It did not implement the paper's additional interneuron filtering step (excluding cells with speed correlation > 0.5).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to the trial start by slicing from the `trial_start` index to teleport onset. No additional temporal shifting is needed since `trial_start` is the alignment event.

ii.
```python
for s in starts:
    tq = np.flatnonzero(teleport[s:] > 0)
    e = s + int(tq[0]) if len(tq) else len(pos)
    ...
    raw_parts = [dset[s:e, :][:, idx] for dset,idx in plane_series]
```

iii. The agent correctly identified that the trial_start pulse is at the 0 cm corridor crossing and aligned data there.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native frame rate is ~15.5 Hz (~64.5 ms per frame). The AI applies temporal rebinning, averaging blocks of 4 frames to produce ~257.934 ms bins. This is a common resolution across all sessions.

ii.
```python
TARGET_DT = 4 / 15.5078125
...
block = int(round(TARGET_DT / dt))
```

iii. The agent initially used the ophys rate attribute which was inconsistent (31 Hz for multi-plane sessions), then corrected to use behavioral timestamps as the authoritative rate. The block size of 4 was chosen to reduce the dataset from ~8 GiB to ~1 GiB.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the block offsets and TARGET_DT constant, not from NWB timestamps directly.

ii.
```python
inp = np.vstack([
    np.arange(T, dtype=np.float32) * np.float32(TARGET_DT),
    ...
])
```

iii. Since the data is rebinned into fixed-size blocks, time is computed as `bin_index * bin_duration` rather than from raw timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Time is computed as `bin_index * TARGET_DT` where TARGET_DT = 4/15.5078125 seconds (~0.258 s). This creates evenly-spaced time values starting at 0.

ii.
```python
np.arange(T, dtype=np.float32) * np.float32(TARGET_DT)
```

iii. This is a straightforward computation given the fixed bin size.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both use the same binned time axis, so they are inherently aligned. Each bin index corresponds to the same time point for neural and input data.

ii. Same `T = len(offsets)` is used for both neural and input arrays.

iii. By construction, since both derive from the same block offsets.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral time series in the NWB.

ii.
```python
env = b['environment/data'][:]
...
ev = int(round(float(np.median(env[s:e]))))
ev = 1 if ev > 0 else 0
```

iii. The agent observed the environment variable has values -1 and 1, corresponding to ENV1 and ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median environment value within the trial is taken, then converted: values > 0 map to 1 (ENV2), otherwise 0 (ENV1). The value is constant across all time bins in the trial.

ii.
```python
ev = int(round(float(np.median(env[s:e]))))
ev = 1 if ev > 0 else 0
...
np.full(T, ev, np.float32)
```

iii. Since environment is constant within a trial, taking the median is a robust way to get the value. The -1/+1 encoding is converted to 0/1.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavioral time series in the NWB.

ii.
```python
trial_num = b['trial number/data'][:]
...
tr = int(round(float(np.median(trial_num[s:e]))))
```

iii. The agent used the NWB's stored trial number rather than a sequential loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The median value of the `trial number` stream within the trial is rounded to get an integer trial number. This is constant across all time bins in the trial.

ii.
```python
tr = int(round(float(np.median(trial_num[s:e]))))
...
np.full(T, tr, np.float32)
```

iii. The median is used to handle any transient values at trial boundaries.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavioral time series, specifically the reward event timestamps.

ii.
```python
rg = b['Reward']
reward_times = rg['timestamps'][:] if 'timestamps' in rg else np.empty(0)
...
rewarded = int(np.any((reward_times >= t0) & (reward_times < t1)))
```

iii. The agent used reward event timestamps to determine whether each trial was rewarded.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, reward timestamps falling within the trial's time window (from trial start timestamp to trial end timestamp + 1/rate) determine if it was rewarded. Trials are sorted by trial number, and previous trial outcome is the reward status of the preceding trial in chronological order. The first trial gets 0.

ii.
```python
t0, t1 = timestamps[s], timestamps[e-1] + 1/rate
rewarded = int(np.any((reward_times >= t0) & (reward_times < t1)))
...
trial_records.sort(key=lambda x: x[0])
prev = 0
for tr, rewarded, nbin, inp, out in trial_records:
    inp[3,:] = prev
    prev = rewarded
```

iii. The agent correctly sorts trials chronologically and uses the previous trial's reward outcome.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and reward zone coordinates. The reward zone identity (A, B, or C) is determined from the NWB session identifier using scene parsing and the switch trial convention (trial 30).

ii.
```python
scene, z0, z1, switched = scene_info(f['identifier'][()])
...
zone = z1 if (switched and tr >= 30) else z0
lo, hi = ZONE_COORDS[zone]
```

```python
ZONE_COORDS = {'A': (80.,130.), 'B': (200.,250.), 'C': (320.,370.)}
```

iii. The agent parsed NWB identifiers to extract scene/location information and applied the repository's convention that reward zone switches occur at trial 30.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed as: negative if before the zone (position - zone_lo), positive if past (position - zone_hi), zero if inside. The position is first block-averaged into bins, then distance is computed.

ii.
```python
def discretize_distance(pos, lo, hi):
    d = np.where(pos < lo, pos-lo, np.where(pos > hi, pos-hi, 0.0))
```

iii. This follows the standard signed-distance-to-interval calculation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins using explicit comparisons matching the instruction bins:
- 0: < -50 cm
- 1: -50 to -10 cm
- 2: -10 to < 0 cm
- 3: exactly 0 (in zone)
- 4: > 0 to +10 cm
- 5: +10 to +50 cm
- 6: > +50 cm

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

iii. These bin edges match the instructions exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is block-averaged into the same time bins as the neural data, so distance-to-reward-zone is inherently aligned.

ii.
```python
pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
...
dist_cls = discretize_distance(pbin, lo, hi)
```

iii. Both neural and behavioral data use the same block offsets for binning.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
pos = b['position/data'][:]
...
pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
```

iii. The position variable directly records the animal's location in the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is block-averaged into the same time bins as neural data, then discretized.

ii.
```python
pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
pos_cls = np.digitize(pbin, [90,180,270,360], right=False).astype(np.int8)
```

iii. Block averaging smooths position within each bin.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `np.digitize` with edges [90, 180, 270, 360]:
- 0: < 90 cm
- 1: 90 to 180 cm
- 2: 180 to 270 cm
- 3: 270 to 360 cm
- 4: > 360 cm

ii.
```python
pos_cls = np.digitize(pbin, [90,180,270,360], right=False).astype(np.int8)
```

iii. The five 90 cm bins span the 450 cm track as specified in the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same block-averaging scheme as neural data ensures alignment.

ii. Both use the same `np.arange(0, e-s, block)` offsets.

iii. By construction.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick = b['lick/data'][:]
...
lbin = np.asarray([np.any(lick[q:min(q+block,e)] > 0) for q in offsets], np.int8)
```

iii. The lick variable records lick events at each frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Within each time bin, if any frame has a lick value > 0, the bin is marked as 1 (lick), otherwise 0 (no lick). This is a binary any-lick-in-bin operation.

ii.
```python
lbin = np.asarray([np.any(lick[q:min(q+block,e)] > 0) for q in offsets], np.int8)
```

iii. The instructions specify binary lick output. Using `any` within a bin captures lick events that may be brief.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same block offsets as neural data ensure alignment.

ii. Both use the same `offsets = np.arange(s, e, block)` structure.

iii. By construction.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB session `identifier` string, which encodes the scene name (e.g., `Env1_LocationB_to_A`). A regex parser extracts location labels A/B/C.

ii.
```python
def scene_info(identifier):
    scene = identifier.decode() if isinstance(identifier, bytes) else str(identifier)
    scene = scene.rstrip('/').split('/')[-1]
    import re
    locs = re.findall(r'(?:Location)?([ABC])', scene)
    ...
    z0, z1 = locs[0], (locs[-1] if len(locs) > 1 else locs[0])
    switched = z1 != z0
    return scene, z0, z1, switched
```

iii. The agent identified that the NWB identifier contains scene/location information matching the repository's naming conventions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For switch sessions (where z0 != z1), trials before trial 30 use z0 and trials >= 30 use z1. The zone label is mapped to integer: A=0, B=1, C=2.

ii.
```python
zone = z1 if (switched and tr >= 30) else z0
...
np.full(T, ZONE_ID[zone], np.int8)
```

```python
ZONE_ID = {'A':0, 'B':1, 'C':2}
```

iii. The agent followed the repository convention that switch sessions change reward zone at trial 30.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` time series timestamps, checking if any reward event falls within the trial's time window.

ii.
```python
rg = b['Reward']
reward_times = rg['timestamps'][:] if 'timestamps' in rg else np.empty(0)
...
t0, t1 = timestamps[s], timestamps[e-1] + 1/rate
rewarded = int(np.any((reward_times >= t0) & (reward_times < t1)))
```

iii. The agent uses the Reward event timestamps rather than data values, checking if any event falls in the trial window.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if any reward timestamp falls within the trial time window [t0, t1), 0 otherwise. The value is constant across all time bins in the trial.

ii.
```python
rewarded = int(np.any((reward_times >= t0) & (reward_times < t1)))
...
np.full(T, rewarded, np.int8)
```

iii. This is a straightforward binary check for reward delivery during the trial.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Clipped final trials** (never reaching teleport) are skipped.
- **Very short trials** (< 2 bins after rebinning) are skipped.
- **Sessions with < 2 trials** are skipped.
- **Multi-plane rate mismatches** raise an error.
- **Missing reward timestamps** default to empty array (no rewards).

ii.
```python
if not len(tq):
    continue
if len(offsets) < 2:
    continue
if len(nt) < 2:
    print('SKIP (<2 trials)', p, flush=True)
    continue
reward_times = rg['timestamps'][:] if 'timestamps' in rg else np.empty(0)
```

iii. These are defensive checks the agent added during development as edge cases were encountered.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB/HDF5 files and reading large neural arrays
2. Block-averaging neural data within each trial
3. Pickle serialization of the ~1.15 GiB output file

ii. N/A

iii. The agent noted that the full conversion of 152 sessions took substantial time and required multiple polling checks.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over all trial starts sequentially. The lick binning uses a Python list comprehension with `np.any` per bin rather than a vectorized operation. The `binned_mean` function uses a list comprehension over block starts.

ii.
```python
lbin = np.asarray([np.any(lick[q:min(q+block,e)] > 0) for q in offsets], np.int8)
```

```python
def binned_mean(x, starts, block):
    return np.stack([np.asarray(x[s:min(s+block, len(x))]).mean(axis=0)
                     for s in starts], axis=0)
```

iii. These list comprehensions could be replaced with reshape-based vectorized operations for fixed block sizes.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each session only once. There is no repeated processing or survey step.

ii. N/A

iii. The code is efficient in this regard -- each NWB file is opened once and all data is extracted in a single pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `scene_info` function parses scene metadata that is partially unused (the `scene` string itself is stored in metadata but not used for decoding). The `rate` and `block` values are returned from `convert_session` but only used for metadata. The `binned_mean` of speed is computed (for the speed output) which may not be strictly necessary if speed binning could be done differently.

ii. N/A

iii. These are minor -- the code is relatively lean.
