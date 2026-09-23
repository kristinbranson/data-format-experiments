# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are found by recursively globbing `*.nwb` under `/app/data`. Each file is opened with `pynwb.NWBHDF5IO` and read. Behavioral time series are loaded from `nwb.processing['behavior']['BehavioralTimeSeries']` and neural data from `nwb.processing['ophys']['Deconvolved']`. Files are sorted by path and processed sequentially.

ii.
```python
files=sorted(DATA_ROOT.rglob('*.nwb'))
# ...
for j,path in enumerate(files):
    sn,si,so,info=convert_session(path,args.show_processing and j<2)
```
```python
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read()
    bts = nwb.processing['behavior']['BehavioralTimeSeries']
    # ...
    deconv_container = nwb.processing['ophys']['Deconvolved']
```

iii. The AI states in CONVERSION_NOTES.md Step 2: "152 NWB files arranged as sub-<id>/sub-<id>_ses-<nn>_behavior+ophys.nwb." The recursive glob finds all NWB files. pynwb is used as required by the instructions.

## 1-b. How are the data split into subjects?

i. Subjects are identified from `nwb.subject.subject_id` within each NWB file. Unique subject names are collected across all sessions and sorted.

ii.
```python
subject = str(nwb.subject.subject_id)
# ...
subjects.append(info['subject'])
# ...
subject_names=sorted(set(subjects))
```

iii. The AI extracts subject IDs directly from the NWB metadata rather than parsing directory names. Both approaches yield the same 11 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed one file at a time.

ii.
```python
files=sorted(DATA_ROOT.rglob('*.nwb'))
for j,path in enumerate(files):
    sn,si,so,info=convert_session(path, ...)
```

iii. One NWB file per session is the natural structure of the data. 152 files = 152 sessions.

## 1-d. How are the data split into trials?

i. Trials are identified by the `trial number` behavioral time series. Nonnegative, finite values of `trial number` define trial membership. Contiguous frames with the same trial number form one trial. The `trial_start` and `teleport` signals are not used for trial splitting.

ii.
```python
trial_ids = np.unique(arrays['trial number'][np.isfinite(arrays['trial number']) &
                                             (arrays['trial number'] >= 0)]).astype(int)
# ...
for tid in valid_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
```

iii. The AI states: "Group contiguous samples by nonnegative trial number, require a start pulse and complete traversal." The reference instead uses `trial_start` and `teleport` signals to define trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have: (a) at least 20 frames, (b) at least one `trial_start` pulse, (c) maximum position >= 440 cm, and (d) all `scanning` values equal to 1. One malformed terminal trial (3 frames, max position 154 cm) is excluded, leaving 12,216 of 12,217 trial IDs.

ii.
```python
complete = (len(ix) >= 20 and np.sum(arrays['trial_start'][ix] > 0) >= 1
            and np.nanmax(arrays['position'][ix]) >= 440
            and np.all(arrays['scanning'][ix] == 1))
if complete:
    valid_ids.append(tid)
```

iii. The AI's CONVERSION_NOTES explains: "Exclude pretrial -1 periods and the sole malformed 3-frame terminal trial; preserve all 12,216 complete trials, including omissions."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the NWB's `Deconvolved` field under `ophys`, which is Suite2p's own deconvolution of raw fluorescence. It is NOT derived from the raw `Fluorescence` and `Neuropil` traces.

ii.
```python
deconv_container = nwb.processing['ophys']['Deconvolved']
series = list(deconv_container.roi_response_series.values())
# ...
neural_parts.append(np.asarray(r.data[:common_n, local_cell_mask], dtype=np.float32))
neural_all = np.ascontiguousarray(np.concatenate(neural_parts, axis=1), dtype=np.float32)
```

iii. The AI's CONVERSION_NOTES Step 3 states: "the reference analyses use deconvolved calcium activity" and "Recomputing delta-F/F is unnecessary for the requested neural decoder." The AI concluded that the stored Deconvolved field was sufficient.

## 2-b. How is the `neural` data processed?

i. No processing is applied beyond loading and filtering by `iscell`. The raw Deconvolved values are used directly, cast to float32.

ii.
```python
neural_parts.append(np.asarray(r.data[:common_n, local_cell_mask], dtype=np.float32))
neural_all = np.ascontiguousarray(np.concatenate(neural_parts, axis=1), dtype=np.float32)
```

iii. The AI decided that the stored Suite2p deconvolved data was the appropriate signal and no further processing (neuropil subtraction, dF/F computation, custom OASIS deconvolution) was needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only Suite2p's `iscell` classification is applied. Cells with `iscell[:,0] == 1` are retained; all others are excluded. Putative interneurons are NOT filtered out.

ii.
```python
iscell = np.asarray(r.rois.table['iscell'][:])
binary = iscell[:, 0] if iscell.ndim == 2 else iscell
local_cell_mask = (binary[linked_rows] == 1)
neural_parts.append(np.asarray(r.data[:common_n, local_cell_mask], dtype=np.float32))
```

iii. The AI states: "Cell curation: Use Suite2p iscell[:,0] == 1 only. Place/RR/TR labels are downstream scientific filters and inappropriate for a general decoder." It also says: "Do not restrict the general decoder to place cells, RR cells, TR cells, or cells passing shuffled spatial-information tests." The putative interneuron filter (speed-correlation > 0.5) was not applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by slicing frames where `trial number == tid`. Since the trial is defined by these frames, alignment to trial start is automatic.

ii.
```python
ix = np.flatnonzero(arrays['trial number'] == tid)
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
```

iii. The first frame of each trial is the trial start. No additional alignment processing is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native sampling rate is preserved at ~64.48 ms per frame (~15.5 Hz). No temporal rebinning is applied. A constant `DT = 0.06448362720402656` is hardcoded.

ii.
```python
DT = 0.06448362720402656
# ...
'time_bin_size':DT*1000.0,
```

iii. The AI keeps the native imaging frame rate. This is consistent with the reference approach.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Time is computed from the frame index multiplied by the constant time interval `DT`. It does not directly use behavior timestamps.

ii.
```python
inp[0] = np.arange(nt, dtype=np.float32) * np.float32(DT)
```

iii. Since the frame interval is constant, multiplying frame index by DT is equivalent to using timestamps minus the first timestamp.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Frame index (0, 1, 2, ...) is multiplied by the constant DT to get time in seconds from trial start.

ii.
```python
inp[0] = np.arange(nt, dtype=np.float32) * np.float32(DT)
```

iii. Simple arithmetic from frame index and constant sampling interval.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time array has exactly `nt` elements, matching the neural trial length. Both are derived from the same frame indices.

ii.
```python
ix = np.flatnonzero(arrays['trial number'] == tid)
nt = len(ix)
inp[0] = np.arange(nt, dtype=np.float32) * np.float32(DT)
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
```

iii. Alignment is inherent since both use the same frame indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral time series.

ii.
```python
env_values = arrays['environment'][ix]
u, c = np.unique(env_values[np.isfinite(env_values)], return_counts=True)
env_mode = u[np.argmax(c)]
env, zone = parse_scene(scene, tid, env_mode)
inp[1] = env
```

iii. The environment values are 0 (ENV1) or 1 (ENV2), matching the binary coding required.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The modal (most frequent) finite environment value across the trial's frames is computed, then passed through `parse_scene()` which extracts the environment integer (0 or 1) from the scene name. The result is repeated across all timepoints.

ii.
```python
u, c = np.unique(env_values[np.isfinite(env_values)], return_counts=True)
env_mode = u[np.argmax(c)]
env, zone = parse_scene(scene, tid, env_mode)
inp[1] = env
```
```python
def parse_scene(scene, trial_id, env_mode):
    env = int(round(float(env_mode)))
    # ...
```

iii. Modal value handles potential edge cases at trial boundaries. For stable sessions environment is constant within trials, so this is equivalent to direct reading.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the native `trial number` behavioral time series. The trial ID from the NWB is used directly.

ii.
```python
inp[2] = tid
```

iii. `tid` comes from `trial_ids = np.unique(arrays['trial number'][...]).astype(int)`. These are the native 0-based trial IDs.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The native trial ID is used as-is, repeated across all timepoints in the trial.

ii.
```python
inp[2] = tid
```

iii. No transformation beyond using the native NWB trial number. This gives values like 0-79 for 80-trial sessions or 0-99 for 100-trial sessions.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` event time series. Reward event timestamps are mapped to frames and then to trial IDs.

ii.
```python
reward = bts.time_series['Reward']
reward_times = np.asarray(reward.timestamps[:], dtype=np.float64)
event_frames = nearest_frame_indices(frame_times, reward_times)
event_trial_ids = arrays['trial number'][event_frames]
rewarded_ids = set(int(x) for x in event_trial_ids if np.isfinite(x) and x >= 0)
```

iii. Reward events are mapped to their nearest frame, then the trial number at that frame determines which trial was rewarded.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A `prev_outcome` variable is tracked across the loop of valid trials. It starts at 0 for the first trial and is updated to the current trial's outcome after processing each trial. The value is repeated across all timepoints.

ii.
```python
prev_outcome = 0
for tid in valid_ids:
    # ...
    outcome = int(tid in rewarded_ids)
    inp[3] = prev_outcome
    # ...
    prev_outcome = outcome
```

iii. This correctly gives 0 for the first trial and the previous trial's reward outcome for subsequent trials. Only valid (retained) trials contribute to the previous outcome chain.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` behavioral time series and the reward zone location. Zone location is determined by parsing the NWB scene name (from `nwb.identifier`) using the `parse_scene()` function, which handles stable, reward-switch, and combined environment+reward switch sessions.

ii.
```python
scene = str(nwb.identifier).rstrip('/').split('/')[-1]
# ...
env, zone = parse_scene(scene, tid, env_mode)
low, high = ZONE_BOUNDS[zone]
dist = signed_distance_to_interval(pos, low, high)
```

iii. The AI uses scene name parsing rather than the `reward_zone` behavioral time series. CONVERSION_NOTES Step 4 explains: "Derive per-trial A/B/C from scene/task schedule."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed to the reward zone interval. Before the zone: negative distance to zone start. Inside: zero. After: positive distance past zone end. Zone bounds used are A=(80,100), B=(200,220), C=(320,340) -- 20 cm zones.

ii.
```python
ZONE_BOUNDS = {'A': (80.0, 100.0), 'B': (200.0, 220.0), 'C': (320.0, 340.0)}
# ...
def signed_distance_to_interval(position, low, high):
    return np.where(position < low, position-low,
                    np.where(position > high, position-high, 0.0))
```

iii. The AI's CONVERSION_NOTES Step 5 states: "Zone intervals from task: A=80-100 cm, B=200-220 cm, C=320-340 cm." The AI determined these from data inspection of the `reward_zone` signal positions.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is discretized into 7 bins using explicit conditional logic matching the instruction specification.

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

iii. The bin boundaries match the instructions: <-50, -50 to -10, -10 to <0, 0, >0 to +10, +10 to +50, >+50.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data uses the same frame indices as neural data within each trial, so distance is automatically aligned.

ii.
```python
ix = np.flatnonzero(arrays['trial number'] == tid)
pos = arrays['position'][ix].astype(np.float32)
dist = signed_distance_to_interval(pos, low, high)
out[0] = discretize_distance(dist)
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
```

iii. Same frame indices ensure alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
pos = arrays['position'][ix].astype(np.float32)
out[1] = discretize_position(pos)
```

iii. Direct use of the position variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond discretization. The raw position values are used directly.

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

iii. The position is already in cm from the NWB data.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins with boundaries at 90, 180, 270, 360 cm. Exact boundary handling: 90 goes to bin 1, 180 to bin 2, 270 to bin 3, 360 to bin 3, >360 to bin 4.

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

iii. Five equal-sized 90 cm bins spanning the 450 cm track as specified.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data. No additional alignment needed.

ii. Same `ix` indexing as neural.

iii. Verified by shared frame indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick = arrays['lick'][ix]
out[3] = (lick > 0).astype(np.int8)
```

iii. Direct use of the lick variable.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
out[3] = (lick > 0).astype(np.int8)
```

iii. The instructions specify binary output (no/yes). Raw lick values can exceed 1, so thresholding at >0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices. No additional alignment needed.

ii. Same `ix` indexing.

iii. Inherent alignment from shared frame indices.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB session identifier (scene name), which encodes the environment and reward zone configuration. The `parse_scene()` function extracts the active zone letter (A, B, or C) based on the scene name pattern and the trial ID (for switch sessions).

ii.
```python
scene = str(nwb.identifier).rstrip('/').split('/')[-1]
env, zone = parse_scene(scene, tid, env_mode)
out[4] = ZONE_INDEX[zone]
```
```python
ZONE_INDEX = {'A': 0, 'B': 1, 'C': 2}
```

iii. The AI's CONVERSION_NOTES Step 4 details the scene parsing logic: "Stable LocationX: X all trials. Reward-only LocationX_to_Y: X for IDs <40, Y for IDs >=40. Combined EnvN_X_to_EnvM_Y sessions switch both variables at native trial 30."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name parsing with regex. For stable sessions, the zone is constant. For reward-switch sessions (`LocationX_to_Y`), zone X is used for trial IDs <40 and zone Y for IDs >=40. For combined environment+reward switches (`EnvN_X_to_EnvM_Y`), the aligned environment stream selects the matching side, with the switch occurring at trial 30.

ii.
```python
def parse_scene(scene, trial_id, env_mode):
    env = int(round(float(env_mode)))
    m = re.fullmatch(r'Env([12])_([ABC])_to_Env([12])_([ABC])', scene)
    if m:
        first_env, first_zone = int(m.group(1))-1, m.group(2)
        second_env, second_zone = int(m.group(3))-1, m.group(4)
        if env == first_env: return env, first_zone
        if env == second_env: return env, second_zone
    m = re.search(r'Env([12])_Location([ABC])(?:_to_([ABC]))?', scene)
    zone = second if second is not None and trial_id >= 40 else first
    return env, zone
```

iii. The AI validates this against native reward-state positions: "Cross-check: native nonzero-state positions center near A~82, B~202, C~322 cm."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` event time series timestamps.

ii.
```python
reward = bts.time_series['Reward']
reward_times = np.asarray(reward.timestamps[:], dtype=np.float64)
event_frames = nearest_frame_indices(frame_times, reward_times)
event_trial_ids = arrays['trial number'][event_frames]
rewarded_ids = set(int(x) for x in event_trial_ids if np.isfinite(x) and x >= 0)
```

iii. Reward timestamps are mapped to nearest frames, then to trial IDs.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, outcome is 1 if the trial ID is in the set of rewarded trial IDs, 0 otherwise. The value is constant across all timepoints in the trial.

ii.
```python
outcome = int(tid in rewarded_ids)
out[5] = outcome
```

iii. Binary per-trial output as specified.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Neural/behavior length mismatch**: Data is cropped to the minimum common length across all series (10 sessions have one extra neural frame).
- **Malformed terminal trial**: One trial with only 3 frames and max position 154 cm is excluded by the filtering criteria (>=20 frames and max position >=440).
- **Multi-plane sessions**: DynamicTableRegion linked rows are used to correctly map iscell across planes.
- **Non-finite values**: Assertions check that all neural and input values are finite.

ii.
```python
common_n = min([r.data.shape[0] for r in series] + [len(v) for v in arrays.values()])
arrays = {k: v[:common_n] for k, v in arrays.items()}
# ...
complete = (len(ix) >= 20 and np.sum(arrays['trial_start'][ix] > 0) >= 1
            and np.nanmax(arrays['position'][ix]) >= 440
            and np.all(arrays['scanning'][ix] == 1))
```

iii. The AI documents handling of 10 one-frame mismatches and one malformed trial in CONVERSION_NOTES Steps 4 and 10.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files with pynwb and reading the large deconvolved neural arrays. The full conversion of 152 sessions completed in 112 seconds including pickle serialization of 13.3 GB.

ii. N/A

iii. CONVERSION_NOTES Step 9: "152 sessions processed in 112.0 s including 13.33 GB pickle serialization."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop iterates over valid trial IDs, computing frame indices, slicing arrays, and building per-trial outputs. Since trials have variable lengths, full vectorization is impractical, but operations like `discretize_distance`, `discretize_position`, and `discretize_speed` could be applied to full-session arrays before splitting into trials.

ii. N/A

iii. The code already uses vectorized numpy operations within each trial. The per-trial loop is necessary for variable-length trial slicing.

## 13-c. What processing does the code repeat multiple times?

i. No significant repeated processing. Each NWB file is opened exactly once. All behavioral and neural data are loaded in a single pass per session.

ii. N/A

iii. The AI designed the code to open each NWB once and extract all needed data in one pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `scene` name is parsed from `nwb.identifier` and used for reward zone determination. The `reward_zone` behavioral time series is loaded but only indirectly used (not in the final conversion). The `scanning` array is loaded for quality filtering but not used in outputs. Additionally, `n_rois` and various diagnostic counters are computed but only used for logging.

ii. N/A

iii. These are minor -- the code is relatively lean with little wasted computation.
