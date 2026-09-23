# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files by recursively globbing `DATA_ROOT.rglob('*.nwb')` (where `DATA_ROOT = Path('/app/data')`). Each NWB file is opened with `pynwb.NWBHDF5IO` and processed individually. All subjects, sessions, and trials present in the data directory are included.

ii.
```python
def choose_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.rglob('*.nwb'))
    if not sample:
        return files
    single = files[0]
    multi = next(p for p in files if p.parent.name in ('sub-m17', 'sub-m18'))
    return [single, multi]
```

```python
with NWBHDF5IO(str(path), 'r') as io:
    nwb = io.read()
```

iii. The AI uses `pynwb` as required. All NWB files are discovered via glob and sorted. The AI confirmed 152 sessions across 11 subjects, matching the data directory contents.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the NWB file's `subject.subject_id` field. A sorted list of unique subject IDs is constructed after processing all sessions.

ii.
```python
subject = str(nwb.subject.subject_id)
...
subjects = sorted(set(x['subject'] for x in infos), key=lambda z: int(re.sub(r'\D','',z)))
subject_idx = np.array([subjects.index(x['subject']) for x in infos], dtype=np.int64)
```

iii. The AI reads subject identity from the NWB metadata rather than parsing directory names. This is a valid approach that yields the same 11 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The AI processes each file independently and appends results as a separate session entry.

ii.
```python
for i, p in enumerate(files, 1):
    ns, xs, ys, info = process_session(p)
    neural.append(ns); inputs.append(xs); outputs.append(ys); infos.append(info)
```

iii. The AI confirmed 152 sessions total (77 switch + 75 stay), consistent with the available data.

## 1-d. How are the data split into trials?

i. Trial starts are identified from `trial_start > 0` frames. Trial ends are identified from `teleport > 0` frames. The `pair_bounds` function pairs each trial start with the next available teleport endpoint, yielding half-open `[start, end)` intervals.

ii.
```python
starts = np.flatnonzero(beh['trial_start'][:common_n] > 0)
ends = np.flatnonzero(beh['teleport'][:common_n] > 0)
pairs = pair_bounds(starts, ends)
```

```python
def pair_bounds(starts, ends):
    pairs, j = [], 0
    for s in starts:
        while j < len(ends) and ends[j] < s:
            j += 1
        if j >= len(ends):
            break
        pairs.append((int(s), int(ends[j])))
        j += 1
    return pairs
```

iii. The AI uses all frames where `teleport > 0` rather than detecting the rising edge of teleport. The `pair_bounds` function greedily matches each start to the next end. The AI verified 12,216 valid trial pairs total.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not explicitly filter trials based on quality controls (e.g., minimum trial length). All paired trials are included. However, it detects and reports lick-artifact trials (>30% of frames with lick count >2) but retains them.

ii.
```python
lick_artifact = bool(np.mean(raw_lick > 2) > 0.30)
lick_artifacts += int(lick_artifact)
```

iii. The AI chose not to remove short trials or lick-artifact trials, reasoning that removing entire trials would discard valid neural and other output data. The CONVERSION_NOTES document 81 lick-artifact trials matching the paper's count.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses the pre-computed `Deconvolved` RoiResponseSeries from `processing/ophys/Deconvolved` in the NWB files. It does NOT recompute dF/F or deconvolved events from the raw Fluorescence and Neuropil traces.

ii.
```python
deconv = nwb.processing['ophys']['Deconvolved'].roi_response_series
...
for key in sorted(deconv.keys()):
    rs = deconv[key]
    ...
    series_info.append((rs, accepted_local_cols))
```

```python
def load_neural_trial(series_info, start, end, factor):
    planes = []
    for rs, accepted_cols in series_info:
        block = np.asarray(rs.data[start:start+n, :], dtype=np.float32)
        block = block[:, accepted_cols]
        ...
        planes.append(block.T)
    return np.concatenate(planes, axis=0).astype(np.float32, copy=False)
```

iii. The AI's CONVERSION_NOTES state: "prefer the corresponding processed neural series already exported in NWB when available rather than recomputing from raw fluorescence." The AI interpreted the NWB Deconvolved field as containing the paper's deconvolved events, which is not what the reference code does.

## 2-b. How is the `neural` data processed?

i. The AI applies minimal processing: it reads the pre-computed Deconvolved data, applies `iscell` filtering via DynamicTableRegion mapping, concatenates planes neuron-wise, and converts to float32. No dF/F computation, neuropil subtraction, baseline estimation, smoothing, or deconvolution is performed.

ii.
```python
block = np.asarray(rs.data[start:start+n, :], dtype=np.float32)
block = block[:, accepted_cols]
...
planes.append(block.T)
return np.concatenate(planes, axis=0).astype(np.float32, copy=False)
```

iii. The AI chose to use the NWB's stored Deconvolved signal directly, avoiding recomputation. The reference solution instead recomputes the paper's full dF/F pipeline (neuropil subtraction, maximin baseline, smoothing, OASIS deconvolution) from the raw Fluorescence and Neuropil traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered using Suite2p's `iscell` labels from the PlaneSegmentation table, accessed through each RoiResponseSeries's DynamicTableRegion. Only cells with `iscell[:,0] > 0.5` are retained. Putative interneurons are NOT filtered.

ii.
```python
region = np.asarray(rs.rois.data[:], dtype=np.int64)
labels = np.asarray(rs.rois.table['iscell'][:])
keep = labels[region, 0] > 0.5
accepted_local_cols = np.flatnonzero(keep)
```

iii. The AI confirmed that 73,512 cells were retained in the 77 switch sessions, matching the paper's count. However, the reference solution additionally removes putative interneurons (cells with speed-dF/F correlation > 0.5), which the AI does not do.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. Since trials are extracted from trial_start to teleport, and neural data is already synchronized with behavior timestamps at the same frame rate, no additional temporal alignment is needed.

ii.
```python
neural = load_neural_trial(series_info, s, e, factor)
```

iii. The AI notes that "NWB streams are already synchronized" and behavior timestamps serve as the authoritative clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 1000/15.5078125 = 64.4836 ms. No temporal rebinning is applied; the data is kept at the native behavior timestamp rate.

ii.
```python
TARGET_RATE = 15.5078125
BIN_MS = 1000.0 / TARGET_RATE
```

iii. The AI determined that all sessions (including two-plane sessions reporting 31 Hz in metadata) have neural data rows aligned one-to-one with behavior timestamps at 15.5078125 Hz. This was a key finding that the AI documented carefully.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the sample index within the trial and the target rate constant. NOT derived from stored behavior timestamps.

ii.
```python
elapsed = np.arange(T, dtype=np.float32) / np.float32(TARGET_RATE)
```

iii. The AI uses a computed elapsed time based on the constant rate rather than reading actual timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The sample index (0, 1, 2, ..., T-1) is divided by the target rate (15.5078125 Hz) to produce elapsed seconds from trial start.

ii.
```python
elapsed = np.arange(T, dtype=np.float32) / np.float32(TARGET_RATE)
```

iii. This produces nearly identical results to subtracting the first timestamp, since the sampling is at a constant rate.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time array has the same length T as the neural data for each trial, so they are inherently aligned.

ii.
```python
T = neural.shape[1]
elapsed = np.arange(T, dtype=np.float32) / np.float32(TARGET_RATE)
```

iii. Since both are derived from the same trial sample indices, alignment is guaranteed.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env_native = aggregate_behavior(beh['environment'][s:e], factor, 'first')
env = int(round(float(env_native[0])))
```

iii. Environment is constant within a trial and takes values 0 or 1, matching ENV1 vs ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The first value of the environment array within the trial is taken, rounded to int, and broadcast across all timepoints.

ii.
```python
env = int(round(float(env_native[0])))
if env not in (0, 1) or not np.all(env_native == env_native[0]):
    raise ValueError(...)
inp = np.vstack([..., np.full(T, env, dtype=np.float32), ...])
```

iii. A validation check ensures the value is 0 or 1 and constant within the trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series, reading the value at the first frame of each trial.

ii.
```python
trial_num = int(round(float(beh['trial number'][s])))
```

iii. The AI uses the stored trial number from the NWB file, not a sequential loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number value at the trial start frame is read, rounded to integer, and broadcast across all timepoints.

ii.
```python
trial_num = int(round(float(beh['trial number'][s])))
...
np.full(T, trial_num, dtype=np.float32)
```

iii. This preserves the original trial numbering from the experiment, which may differ from sequential indexing if trials are filtered or non-contiguous.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the sparse `Reward` timestamps and the behavior position timestamps. Reward delivery is detected by checking if any reward timestamp falls within the previous trial's time window.

ii.
```python
reward_ts = np.asarray(bts['Reward'].timestamps[:], dtype=np.float64)
...
rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
outcomes.append(rewarded)
...
prev = outcomes[-2] if len(outcomes) > 1 else 0
```

iii. The AI uses timestamp-based comparison rather than index-based lookup. For the first trial, previous outcome defaults to 0.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each trial's reward outcome is stored in an `outcomes` list. For trial i, the previous trial outcome is `outcomes[i-1]` (or 0 for the first trial). The value is broadcast across all timepoints.

ii.
```python
prev = outcomes[-2] if len(outcomes) > 1 else 0
...
np.full(T, prev, dtype=np.float32)
```

iii. Note: `outcomes[-2]` is used because `outcomes` already has the current trial's outcome appended before `prev` is computed. This correctly retrieves the previous trial's outcome.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone identity parsed from the NWB file's `identifier` string. The identifier encodes source/destination zone letters (e.g., `A_to_B`).

ii.
```python
def parse_zone_sequence(identifier):
    name = identifier.rstrip('/').split('/')[-1]
    letters = re.findall(r'(?:Location)?([ABC])', name)
    if '_to_' in name and len(letters) >= 2:
        return ord(letters[-2]) - 65, ord(letters[-1]) - 65
    if letters:
        z = ord(letters[-1]) - 65
        return z, z
    raise ValueError(...)
```

```python
src_zone, dst_zone = parse_zone_sequence(nwb.identifier)
zone = zone_for_trial(src_zone, dst_zone, trial_num)
```

iii. The AI determines reward zone from session metadata rather than from the `reward_zone` behavior time series. Zone switches at trial 30 for switch sessions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to the nearest edge of the reward zone interval. Inside the zone = 0, before = negative, after = positive. Zone boundaries are A=[80,130], B=[200,250], C=[320,370].

ii.
```python
ZONE_START = np.array([80.0, 200.0, 320.0], dtype=np.float32)
ZONE_END = ZONE_START + 50.0

def distance_classes(position, zone):
    lo, hi = float(ZONE_START[zone]), float(ZONE_END[zone])
    d = np.where(position < lo, position - lo,
                 np.where(position > hi, position - hi, 0.0)).astype(np.float32)
    ...
```

iii. The signed distance computation matches the reference approach.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses explicit conditional assignments for 7 bins matching the instructions.

ii.
```python
c = np.empty(d.shape, dtype=np.int64)
c[d < -50] = 0
c[(d >= -50) & (d < -10)] = 1
c[(d >= -10) & (d < 0)] = 2
c[d == 0] = 3
c[(d > 0) & (d <= 10)] = 4
c[(d > 10) & (d <= 50)] = 5
c[d > 50] = 6
```

iii. The boundaries match the instructions. The AI uses explicit conditionals rather than `np.digitize`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data is sliced using the same trial indices as neural data, so they are inherently aligned.

ii.
```python
pos = aggregate_behavior(beh['position'][s:e], factor, 'mean').astype(np.float32)
```

iii. Same frame-level indexing ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = aggregate_behavior(beh['position'][s:e], factor, 'mean').astype(np.float32)
```

iii. Direct use of the position stream.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] before discretization.

ii.
```python
def position_classes(position):
    p = np.clip(position, 0.0, 450.0)
    c = np.zeros(p.shape, dtype=np.int64)
    c[p >= 90] = 1
    c[p >= 180] = 2
    c[p >= 270] = 3
    c[p > 360] = 4
    return c
```

iii. The clipping ensures values outside the track are handled. The reference does not clip but uses `-inf/inf` bin edges to absorb out-of-range values.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 bins of 90 cm each spanning 0-450 cm. Uses explicit conditional assignments.

ii.
```python
c[p >= 90] = 1
c[p >= 180] = 2
c[p >= 270] = 3
c[p > 360] = 4
```

iii. The boundary at 360 uses `>` (strict) while 90, 180, 270 use `>=`. This means exactly 360 stays in bin 3.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame-level indexing as neural data within each trial.

ii.
```python
pos = aggregate_behavior(beh['position'][s:e], factor, 'mean').astype(np.float32)
```

iii. Inherently aligned through shared indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
raw_lick = beh['lick'][s:e]
lick = (aggregate_behavior((raw_lick > 0).astype(np.int8), factor, 'max') > 0).astype(np.int64)
```

iii. Direct use of the lick stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick = (aggregate_behavior((raw_lick > 0).astype(np.int8), factor, 'max') > 0).astype(np.int64)
```

iii. The `> 0` threshold converts cumulative lick counts to binary presence.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame-level indexing as neural data within each trial.

ii. Same `[s:e]` slicing.

iii. Inherently aligned through shared indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB file's `identifier` string, which encodes the source and destination reward zones (e.g., `LocationA_to_LocationB`). Zone switches at trial number 30.

ii.
```python
src_zone, dst_zone = parse_zone_sequence(nwb.identifier)
zone = zone_for_trial(src_zone, dst_zone, trial_num)
```

```python
def zone_for_trial(src, dst, trial_number):
    return src if trial_number < 30 else dst
```

iii. The AI parses zone identity from session metadata rather than from the per-timepoint `reward_zone` behavior variable. This is a deterministic approach that avoids noise in the reward_zone signal.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For switch sessions (`_to_` in identifier), trials with number < 30 use the source zone, trials >= 30 use the destination zone. For stay sessions, the single zone is used throughout. Zone is encoded as 0=A, 1=B, 2=C.

ii.
```python
def zone_for_trial(src, dst, trial_number):
    return src if trial_number < 30 else dst
```

iii. The trial-30 switch point comes from the paper's description that reward location changed after the first 30 trials on switch days.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the sparse `Reward` timestamps in the behavior processing module.

ii.
```python
reward_ts = np.asarray(bts['Reward'].timestamps[:], dtype=np.float64)
rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
```

iii. Reward events have their own timestamps separate from the regular behavior sampling grid.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within the trial's time window `[pos_ts[start], pos_ts[end])`. Binary output: 1 if rewarded, 0 if not. The value is constant across all timepoints in the trial.

ii.
```python
rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
...
np.full(T, rewarded, dtype=np.int64)
```

iii. The timestamp-based comparison is robust and avoids index alignment issues.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Data is cropped to the common minimum length across all streams.
- **Lick artifacts**: Detected (>30% frames with lick>2) and reported but trials are retained.
- **Non-finite values**: An assertion checks all neural and input data is finite.
- **Invalid environment values**: Raises an error if environment is not 0 or 1.
- **Negative trial numbers**: Raises an error.

ii.
```python
common_n = min([len(v) for v in beh.values()] +
               [int(rs.data.shape[0]) for rs, _ in series_info])
...
if not np.isfinite(neural).all() or not np.isfinite(inp).all():
    raise ValueError(f'{path}: non-finite converted data')
```

iii. The AI takes a defensive approach, truncating to common length and validating values.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with pynwb (I/O bound)
2. Reading neural data arrays from each trial
3. Writing the final pickle file (9.8 GB)

ii. N/A

iii. The AI reports total conversion time of 85 seconds for all 152 sessions, well within the 15-minute limit.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop within `process_session` iterates over each trial sequentially. Neural data loading (`load_neural_trial`) could potentially be vectorized by reading the full session's neural data at once and then slicing. The `pair_bounds` function uses a sequential loop that could be vectorized.

ii. N/A

iii. Given the 85-second total runtime, vectorization was not necessary.

## 13-c. What processing does the code repeat multiple times?

i. Each NWB file is opened only once. No significant repeated processing is evident in the AI's code, unlike the reference solution which has a separate survey step that re-reads all NWB files.

ii. N/A

iii. The AI's approach is efficient in this regard.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores `lick_artifact` detection results in metadata but does not use them to modify the data. The `trial_info` list stored in metadata contains per-trial details that are not used by the decoder.

ii.
```python
lick_artifact = bool(np.mean(raw_lick > 2) > 0.30)
...
trial_info.append({'trial_number': trial_num, 'start': s, 'end': e, ...})
```

iii. These are informational/diagnostic and have minimal computational cost.
