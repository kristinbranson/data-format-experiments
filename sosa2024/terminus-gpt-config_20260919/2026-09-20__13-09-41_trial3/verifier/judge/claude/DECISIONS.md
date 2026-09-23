# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files found recursively under `/app/data` using `glob.glob(DATA_ROOT+'/**/*.nwb', recursive=True)`, sorted in natural order. Each NWB file corresponds to one session. Files are read with `h5py` (not `pynwb`). All subjects and sessions present in the data directory are included.

ii.
```python
files=sorted(glob.glob(DATA_ROOT+'/**/*.nwb',recursive=True),key=natural_key)
if args.sample: files=files[:2]
...
for i,f in enumerate(files):
    ns,xs,ys,info=process_session(f, make_plot=args.show_processing and i<2)
```

```python
with h5py.File(path, 'r') as h:
    b = h['processing/behavior/BehavioralTimeSeries']
    ...
```

iii. The AI documented that the data directory contains 152 NWB files across 11 mice. It uses h5py for direct HDF5 access rather than pynwb, which is faster and avoids pynwb overhead. All files are processed.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `general/subject/subject_id` field within each NWB file. Unique subjects are accumulated in a list as sessions are processed.

ii.
```python
subject = h['general/subject/subject_id'][()].decode()
...
if info['subject'] not in subjects: subjects.append(info['subject'])
subject_idx.append(subjects.index(info['subject']))
```

iii. Subject IDs are read directly from NWB metadata rather than parsed from directory/file names.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. Files are sorted in natural order and processed sequentially.

ii.
```python
files=sorted(glob.glob(DATA_ROOT+'/**/*.nwb',recursive=True),key=natural_key)
...
for i,f in enumerate(files):
    ns,xs,ys,info=process_session(f, ...)
```

iii. The one-file-per-session structure matches the DANDI/NWB organization.

## 1-d. How are the data split into trials?

i. Trials are defined by pairing each `trial_start` pulse (where `trial_start > 0`) with the first subsequent `teleport` pulse (where `teleport > 0`) before the next start. The trial includes the start index and excludes the teleport index.

ii.
```python
def find_complete_trials(b):
    start_flags = np.asarray(b['trial_start/data'][:])
    teleport_flags = np.asarray(b['teleport/data'][:])
    starts = np.flatnonzero(start_flags > 0)
    teleports = np.flatnonzero(teleport_flags > 0)
    trials = []
    for j, start in enumerate(starts):
        next_start = starts[j+1] if j+1 < len(starts) else len(start_flags)
        candidates = teleports[(teleports > start) & (teleports < next_start)]
        if len(candidates) != 1:
            raise ValueError(...)
        trials.append((int(start), int(candidates[0])))
    return trials
```

iii. The AI documented in CONVERSION_NOTES that trial-ID boundaries were unreliable and the explicit start/teleport pairing was more robust. This matches the reference code's approach of using `trial_start` and `teleport` signals.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on lick sensor corruption. If more than 30% of frames in a trial have a cumulative lick count > 2, the trial is excluded. No minimum-timepoint filter is applied.

ii.
```python
lick_raw = streams['lick'][q]
if np.mean(lick_raw > 2) > 0.30:
    bad_lick += 1
    continue
```

iii. The AI identified this criterion from the reference code's `correct_lick_sensor_error` function and the paper's methods. 81 trials were excluded this way, matching the paper's count of corrupt lick trials. The AI argued that since categorical lick output cannot represent NaN, corrupt trials must be fully excluded rather than having only lick set to NaN.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` data stored in the NWB file under `processing/ophys/Deconvolved`. This is Suite2p's own deconvolution of raw fluorescence, NOT the paper's custom dF/F and deconvolution pipeline.

ii.
```python
def load_neural_and_regions(h):
    dec = h['processing/ophys/Deconvolved']
    plane_names = sorted((k for k, v in dec.items() if isinstance(v, h5py.Group)), key=natural_key)
    arrays = [np.asarray(dec[p]['data'][:], dtype=np.float32) for p in plane_names]
    ...
    full = np.concatenate(arrays, axis=1)
```

iii. The AI stated in CONVERSION_NOTES: "Use released `Deconvolved`; no recomputation or spatial binning." and in the reference code comparison: "converter reads released NWB Deconvolved streams; reference `multi_anim_sess` creates that stream through `dff(..., deconvolve=True)`. Same signal, avoiding redundant recomputation." However, this is incorrect -- the NWB's `Deconvolved` is Suite2p's deconvolution, while the paper computes its own dF/F baseline and deconvolution.

## 2-b. How is the `neural` data processed?

i. No processing is applied beyond loading, concatenating planes, filtering by `iscell`, and slicing into trials. The deconvolved data from the NWB is used as-is.

ii.
```python
full = np.concatenate(arrays, axis=1)
seg = h['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = np.asarray(seg['iscell'][:, 0]) > 0
...
return full[:, iscell], plane_idx[iscell], plane_names
...
neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
```

iii. The AI believed the released Deconvolved data was the same signal the paper analyses, avoiding redundant recomputation. This is a misunderstanding -- the paper applies its own neuropil subtraction, baseline estimation, and OASIS deconvolution with specific parameters.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only Suite2p `iscell` filtering is applied. Putative interneurons are NOT filtered out.

ii.
```python
iscell = np.asarray(seg['iscell'][:, 0]) > 0
...
return full[:, iscell], plane_idx[iscell], plane_names
```

iii. The AI explicitly decided not to apply interneuron filtering: "It intentionally does not apply place-cell/RR/interneuron selections because those are analysis-specific rather than recording-quality filters." This is documented in CONVERSION_NOTES Step 4 and Step 10.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by slicing neural data from the `trial_start` index to the `teleport` index. Since behavior and neural data share the same time indices, no additional alignment is needed.

ii.
```python
q = slice(start, stop)
...
neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
```

iii. The AI verified that neural and behavioral data share the same time grid (~15.5 Hz). Alignment is implicit through shared indexing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native imaging frame rate (~15.5 Hz, ~64.48 ms per bin). No temporal rebinning is applied. The time bin size is computed as the median of timestamp differences.

ii.
```python
dt = float(np.median(np.diff(timestamps)))
...
data=dict(..., metadata=dict(..., time_bin_size=float(np.median(dts)*1000), ...))
```

iii. The AI documented that the ~15.5 Hz rate is the per-plane sampling rate, consistent with the paper's methods. Two-plane sessions have a scanner rate of ~62 Hz but per-plane rate remains ~15.5 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` array in the behavior data.

ii.
```python
timestamps = np.asarray(b['position/timestamps'][:], dtype=np.float64)
...
inp = np.vstack([
    (timestamps[q]-timestamps[start]).astype(np.float32),
    ...
])
```

iii. All behavior streams share the same timestamps; the AI used position timestamps as the reference.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The start timestamp is subtracted from all timestamps within the trial.

ii.
```python
(timestamps[q]-timestamps[start]).astype(np.float32)
```

iii. Simple offset subtraction to make time relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices (one row per imaging frame), so they are inherently aligned.

ii. Same `q = slice(start, stop)` indexing is used for both neural and behavioral data.

iii. The AI verified timestamp spacing consistency across sessions.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment/data` behavior time series.

ii.
```python
streams = {n: np.asarray(b[n+'/data'][:]) for n in names}
...
env_values = np.unique(streams['environment'][q])
env_values = env_values[env_values >= 0]
...
env = int(env_values[0])
```

iii. The environment variable is 0 or 1, corresponding to ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The unique non-negative environment value within the trial is extracted and verified to be constant and binary (0 or 1). It is then broadcast as a constant for all timepoints.

ii.
```python
env_values = np.unique(streams['environment'][q])
env_values = env_values[env_values >= 0]
if len(env_values) != 1 or env_values[0] not in (0, 1):
    raise ValueError(...)
env = int(env_values[0])
...
np.full(n, env, dtype=np.float32),
```

iii. Validation ensures the environment is constant within each trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series in the NWB file. The value at the start of each trial is used.

ii.
```python
trial_id = int(round(float(streams['trial number'][start])))
...
np.full(n, trial_id, dtype=np.float32),
```

iii. The AI uses the stored NWB trial number rather than a sequential loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number value at the start index is read, rounded to integer, and broadcast as a constant for all timepoints.

ii.
```python
trial_id = int(round(float(streams['trial number'][start])))
np.full(n, trial_id, dtype=np.float32),
```

iii. The stored trial number is used directly. This means if trials are excluded (e.g., lick-corrupt), the trial numbers are not contiguous.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the sparse `Reward/timestamps` array and behavior timestamps.

ii.
```python
def reward_outcomes(b, timestamps, trials):
    reward_t = np.asarray(b['Reward/timestamps'][:], dtype=float)
    return np.asarray([np.any((reward_t >= timestamps[a]) & (reward_t < timestamps[z]))
                       for a, z in trials], dtype=np.int64)
```

iii. Reward delivery events have separate timestamps from the behavior sampling rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the outcome of the chronologically previous complete trial is used. First trial gets 0. The previous trial outcome is based on the source trial index, not the kept trial index, so excluded (lick-corrupt) trials' outcomes still inform the next trial's input.

ii.
```python
outcomes = reward_outcomes(b, timestamps, trials)
...
prev_outcome = int(outcomes[j-1]) if j > 0 else 0
```

iii. Previous outcome is from the immediately preceding source trial, ensuring lick exclusions don't affect the outcome chain.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position/data` and the reward zone identity for each trial. Reward zone identity is determined by parsing the NWB `identifier` field to extract A/B/C schedule labels, with a switch at trial 30 for two-zone sessions.

ii.
```python
def parse_scene(identifier):
    scene = identifier.rstrip('/').split('/')[-1]
    labels = re.findall(r'(?:Location)?([ABC])', scene)
    ...
    return scene, labels

def zone_for_trial(labels, chronological_index):
    return labels[0] if len(labels) == 1 or chronological_index < 30 else labels[1]

identifier = h['identifier'][()].decode()
scene, labels = parse_scene(identifier)
...
zone = zone_for_trial(labels, j)
zstart, zstop = ZONE_COORDS[zone]
```

iii. The AI's method matches the reference code's `get_reward_zones` which derives zone identity from the session scene/schedule with a change at trial 30.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative when before the zone, 0 when inside, positive when past. Then discretized into 7 classes.

ii.
```python
def distance_classes(position, start, stop):
    distance = np.where(position < start, position-start,
                        np.where(position > stop, position-stop, 0.0))
    out = np.empty(distance.shape, dtype=np.int64)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out, distance
```

iii. The zone coordinates (A: 80-130 cm, B: 200-250 cm, C: 320-370 cm) are taken from the reference code.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit conditional assignments with 7 bins matching the instructions exactly.

ii. See 7-b above. The boundary handling uses strict inequalities matching the instruction specification.

iii. The bin edges match: `<-50`, `[-50,-10)`, `[-10,0)`, `==0`, `(0,10]`, `(10,50]`, `>50`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as neural data -- both use the same `slice(start, stop)`.

ii.
```python
q = slice(start, stop)
pos = streams['position'][q].astype(np.float32)
...
neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
```

iii. Shared indexing ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from `position/data` behavior time series.

ii.
```python
pos = streams['position'][q].astype(np.float32)
```

iii. The position variable records the animal's location in the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
posclass = np.digitize(pos, [90, 180, 270, 360], right=False).astype(np.int64)
```

iii. 5 equal-sized bins spanning the 450 cm track (90 cm each).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with edges `[90, 180, 270, 360]` produces classes 0-4.

ii.
```python
posclass = np.digitize(pos, [90, 180, 270, 360], right=False).astype(np.int64)
```

iii. With `right=False`, values exactly on a boundary go to the higher bin (e.g., 90 -> bin 1). The bins are: `<90`, `[90,180)`, `[180,270)`, `[270,360)`, `>=360`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data via shared `slice(start, stop)`.

ii. Same indexing pattern as other outputs.

iii. Shared time grid ensures alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick/data` behavior time series.

ii.
```python
lick_raw = streams['lick'][q]
...
lickclass = (lick_raw > 0).astype(np.int64)
```

iii. The lick variable records cumulative lick counts per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0. Trials with corrupt lick sensors (>30% frames with count >2) are excluded entirely.

ii.
```python
if np.mean(lick_raw > 2) > 0.30:
    bad_lick += 1
    continue
...
lickclass = (lick_raw > 0).astype(np.int64)
```

iii. The paper's methods describe binarizing lick counts and excluding corrupt sensor trials.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data.

ii. Same `slice(start, stop)` indexing.

iii. Shared time grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB file's `identifier` field, which encodes the session's reward schedule (e.g., "LocationA", "LocationBLocationC").

ii.
```python
identifier = h['identifier'][()].decode()
scene, labels = parse_scene(identifier)
...
zone = zone_for_trial(labels, j)
```

iii. The AI matched the reference code's `get_reward_zones` logic for deriving zone identity from session metadata.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. One or two A/B/C labels are parsed from the identifier. For single-zone sessions, the same zone applies to all trials. For switch sessions, trials 0-29 use the first label and trials 30+ use the second. Zone A=0, B=1, C=2.

ii.
```python
def zone_for_trial(labels, chronological_index):
    return labels[0] if len(labels) == 1 or chronological_index < 30 else labels[1]
...
np.full(n, ZONE_INDEX[zone], dtype=np.int64),
```

iii. The switch at trial 30 matches the reference code's `get_reward_zones(..., change_trial=30)`.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the sparse `Reward/timestamps` array.

ii.
```python
def reward_outcomes(b, timestamps, trials):
    reward_t = np.asarray(b['Reward/timestamps'][:], dtype=float)
    return np.asarray([np.any((reward_t >= timestamps[a]) & (reward_t < timestamps[z]))
                       for a, z in trials], dtype=np.int64)
```

iii. Reward events are matched to trial time windows.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if any reward event timestamp falls within the trial's time window `[timestamps[start], timestamps[stop])`, 0 otherwise. The value is constant for all timepoints in the trial.

ii.
```python
outcomes = reward_outcomes(b, timestamps, trials)
...
np.full(n, outcomes[j], dtype=np.int64),
```

iii. Uses sparse event timestamps rather than the reward data values, which are all identical.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Neural/behavior length mismatch**: Ten two-plane sessions have exactly one extra neural row, which is trimmed. Other mismatches raise an error.
- **Lick sensor corruption**: Trials with >30% frames having lick count >2 are excluded entirely (81 trials).
- **Incomplete trial**: One source trial ID without a matching start is excluded by the start/teleport pairing logic.
- **Non-finite values**: Assertions check for finite neural and input values.

ii.
```python
neural_excess_frames = int(neural_all.shape[0] - ntime)
if neural_excess_frames not in (0, 1):
    raise ValueError(...)
if neural_excess_frames:
    neural_all = neural_all[:ntime]
...
if np.mean(lick_raw > 2) > 0.30:
    bad_lick += 1
    continue
...
if not (np.isfinite(neu).all() and np.isfinite(inp).all()):
    raise ValueError('Nonfinite neural/input values')
```

iii. The AI documented each edge case in CONVERSION_NOTES Step 9 and Step 10.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the NWB files and reading neural arrays. The full conversion takes about 55 seconds for 152 sessions (9 GB output).

ii. N/A

iii. The AI optimized by reading only the Deconvolved arrays (avoiding dF/F recomputation), using float32, and processing sessions serially to bound memory.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop in `process_session` iterates over trials sequentially for construction of inputs/outputs. Some operations could be vectorized across all timepoints before splitting into trials. The `reward_outcomes` function uses a list comprehension over trials that could use vectorized interval checking.

ii.
```python
for j, (start, stop) in enumerate(trials):
    ...
```

iii. Variable trial lengths make full vectorization awkward.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each NWB file only once. There is no survey/pre-scan step that rereads files. This is efficient.

ii. N/A

iii. The AI's design avoids the repeated file loading that occurs in the reference solution's survey + convert two-pass approach.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores per-session `info` dictionaries in the metadata, which include detailed trial-level information (source_trial_index, start/stop indices, distance ranges, etc.) that is not used by the decoder. The `distance` array is computed in `distance_classes` but only the discretized classes are stored.

ii.
```python
kept_info.append(dict(source_trial_index=j, trial_id=trial_id, start=start, stop=stop,
                      zone=zone, outcome=int(outcomes[j]), environment=env,
                      distance_min=float(distance.min()), distance_max=float(distance.max())))
```

iii. The extra metadata is for documentation/debugging and has minimal computational cost.
