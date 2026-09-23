# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every `.nwb` file under `/app/data` by recursively globbing the data directory. Each file is opened with `pynwb.NWBHDF5IO`, then its behavior and ophys tables are read session-by-session inside `process_session()`.

ii.
```python
DATA_ROOT = Path('/app/data')

def choose_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.rglob('*.nwb'))
    if not sample:
        return files
    ...

def process_session(path: Path):
    with NWBHDF5IO(str(path), 'r') as io:
        nwb = io.read()
        bts = nwb.processing['behavior']['BehavioralTimeSeries'].time_series
        deconv = nwb.processing['ophys']['Deconvolved'].roi_response_series
```

iii. In `CONVERSION_NOTES.md` Step 2 and Step 5, the AI says the dataset consists of 152 NWB files under subject subdirectories and that all available sessions should be retained. It also explicitly notes that all inspection/loading used `pynwb`, not `h5py`.

## 1-b. How are the data split into subjects (mice)?

i. The AI uses the NWB file metadata field `nwb.subject.subject_id` to assign each session to a mouse, then builds the final `subjects` list and `subject_idx` from the per-session metadata.

ii.
```python
def process_session(path: Path):
    with NWBHDF5IO(str(path), 'r') as io:
        nwb = io.read()
        subject = str(nwb.subject.subject_id)
        ...
        info = {
            'path': str(path), 'subject': subject, 'session_id': str(nwb.session_id),
            ...
        }

subjects = sorted(set(x['subject'] for x in infos), key=lambda z: int(re.sub(r'\D','',z)))
subject_idx = np.array([subjects.index(x['subject']) for x in infos], dtype=np.int64)
```

iii. The notes document that the dataset contains 11 subjects and that session metadata already exposes the subject identity. The AI therefore uses the NWB metadata rather than reparsing directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The main loop iterates over the chosen NWB paths, calls `process_session()` once per file, and appends one session-level entry to `neural`, `input`, and `output`.

ii.
```python
def choose_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.rglob('*.nwb'))
    if not sample:
        return files
    ...

for i, p in enumerate(files, 1):
    ns, xs, ys, info = process_session(p)
    neural.append(ns); inputs.append(xs); outputs.append(ys); infos.append(info)
```

iii. In Step 2 of the notes, the AI describes the dataset organization as `data/sub-mX/sub-mX_ses-YY_behavior+ophys.nwb`, and Step 5 explicitly states that all 152 available sessions are included.

## 1-d. How are the data split into trials?

i. Trials are defined from framewise behavior markers. The AI finds all `trial_start > 0` indices and all `teleport > 0` indices, then pairs each start with the next unused teleport. The resulting bounds are used as half-open intervals `[start, end)`.

ii.
```python
def pair_bounds(starts: np.ndarray, ends: np.ndarray) -> list[tuple[int, int]]:
    pairs, j = [], 0
    for s in starts:
        while j < len(ends) and ends[j] < s:
            j += 1
        if j >= len(ends):
            break
        pairs.append((int(s), int(ends[j])))
        j += 1
    return pairs

starts = np.flatnonzero(beh['trial_start'][:common_n] > 0)
ends = np.flatnonzero(beh['teleport'][:common_n] > 0)
pairs = pair_bounds(starts, ends)
```

iii. Step 5 of `CONVERSION_NOTES.md` says the AI deliberately chose `[trial_start, teleport)` because teleport marks entry into the inter-trial interval and may contain reset-position artifacts. Trajectory step 54 gives the same justification.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a dedicated per-trial quality filter such as a minimum trial length. It keeps every successfully paired trial, only erroring out on malformed session-level conditions such as unmatched markers or impossible lengths. Lick-artifact trials are detected and counted but not removed.

ii.
```python
if len(pairs) != len(starts) or len(starts) != len(ends):
    raise ValueError(f'{path}: unmatched trial markers {len(starts)}/{len(ends)}/{len(pairs)}')

n = ((end - start) // factor) * factor
if n <= 0:
    raise ValueError('Trial has no complete output bins')

lick_artifact = bool(np.mean(raw_lick > 2) > 0.30)
lick_artifacts += int(lick_artifact)
```

iii. Step 10 of the notes says the AI retained lick-artifact trials because the target format has no mask for just the lick output and dropping whole trials would discard otherwise valid neural and behavioral data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` directly from `nwb.processing['ophys']['Deconvolved'].roi_response_series`, i.e. the NWB-exported deconvolved calcium activity. It does not use `Fluorescence` or `Neuropil`.

ii.
```python
with NWBHDF5IO(str(path), 'r') as io:
    nwb = io.read()
    ...
    deconv = nwb.processing['ophys']['Deconvolved'].roi_response_series
```

iii. In Step 4 and Step 5 of the notes, the AI states that the paper mostly uses deconvolved calcium activity and that the NWB already exports a synchronized `Deconvolved` series, so it chose that representation rather than recomputing dF/F and events.

## 2-b. How is the `neural` data processed?

i. Processing is minimal: for each Deconvolved plane, the AI slices the trial interval, keeps only accepted ROI columns, optionally rebins by summing across `factor` samples, and concatenates planes neuron-wise. In the final code `factor = 1`, so there is no temporal rebinning.

ii.
```python
def load_neural_trial(series_info, start: int, end: int, factor: int) -> np.ndarray:
    planes = []
    n = ((end - start) // factor) * factor
    ...
    for rs, accepted_cols in series_info:
        block = np.asarray(rs.data[start:start+n, :], dtype=np.float32)
        block = block[:, accepted_cols]
        if factor > 1:
            block = block.reshape(n // factor, factor, block.shape[1]).sum(axis=1)
        planes.append(block.T)
    return np.concatenate(planes, axis=0).astype(np.float32, copy=False)
```

iii. Step 6 of the notes says the script was designed to avoid loading full uncurated fluorescence arrays and instead slice Deconvolved data trial-by-trial. Step 10 later notes that the earlier two-plane downsampling idea was removed after finding that the exported arrays were already aligned one-to-one to behavior timestamps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters ROIs using the Suite2p `iscell` label, but only after mapping each RoiResponseSeries through its `DynamicTableRegion` so that the `iscell` labels correspond to the columns actually present in that Deconvolved series. It does not perform additional interneuron filtering.

ii.
```python
for key in sorted(deconv.keys()):
    rs = deconv[key]
    region = np.asarray(rs.rois.data[:], dtype=np.int64)
    labels = np.asarray(rs.rois.table['iscell'][:])
    keep = labels[region, 0] > 0.5
    accepted_local_cols = np.flatnonzero(keep)
    ...
    series_info.append((rs, accepted_local_cols))
```

iii. Trajectory step 26 says the AI discovered that counting `iscell` over the full segmentation table was wrong for multi-plane sessions and that correct curation must apply `iscell` through each series’ `DynamicTableRegion`. The notes repeat this as a key curation rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The per-trial neural matrices are aligned to trial start simply by cutting each trial on the `[trial_start, teleport)` bounds. No additional event-shifting is applied.

ii.
```python
for qi, (s, e) in enumerate(pairs):
    ...
    neural = load_neural_trial(series_info, s, e, factor)
```

iii. The notes state that the decoder alignment event is trial start, and trajectory step 54 says this mapping is defensible because teleport marks entry into the inter-trial interval.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses a fixed 15.5078125 Hz common clock, i.e. `1000 / 15.5078125 = 64.4836 ms` bins. After discovering that two-plane sessions were already exported on the behavior clock, it uses `factor = 1` and applies no temporal rebinning in the final script.

ii.
```python
TARGET_RATE = 15.5078125
BIN_MS = 1000.0 / TARGET_RATE

pos_clock = np.asarray(bts['position'].timestamps[:], dtype=np.float64)
effective_rate = 1.0 / float(np.median(np.diff(pos_clock)))
if not np.isclose(effective_rate, TARGET_RATE, rtol=2e-3):
    raise ValueError(f'{path}: unsupported behavior-clock rate {effective_rate}')
factor = 1
```

iii. Step 7 and Step 10 of the notes explain the key justification: the NWB metadata rate is misleading for two-plane sessions, but the row count and behavior timestamps show the processed arrays are already aligned to 15.5078125 Hz. Trajectory steps 58-60 document this correction.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The final code derives elapsed time from the trial length `T` and the fixed common rate `TARGET_RATE`, rather than directly from a raw timestamp array.

ii.
```python
T = neural.shape[1]
elapsed = np.arange(T, dtype=np.float32) / np.float32(TARGET_RATE)
inp = np.vstack([
    elapsed,
    np.full(T, env, dtype=np.float32),
    np.full(T, trial_num, dtype=np.float32),
    np.full(T, prev, dtype=np.float32),
]).astype(np.float32, copy=False)
```

iii. The AI’s notes justify this indirectly by saying the processed neural rows are one-to-one with the 15.5078125 Hz behavior timestamps, so a regular sample clock is sufficient once the synchronized rate has been established.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI creates a regular sequence `0, 1/rate, 2/rate, ...` up to `T-1`, so time is measured relative to the first sample in the trial.

ii.
```python
elapsed = np.arange(T, dtype=np.float32) / np.float32(TARGET_RATE)
```

iii. The notes and trajectory emphasize that all trial streams share the same common sample clock, so a per-trial elapsed-time vector is the intended representation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is created with exactly the same trial length `T` as the neural matrix produced for that trial, and the code asserts that all neural, input, and output streams have equal length.

ii.
```python
T = neural.shape[1]
...
if not (T == len(pos) == len(speed) == len(lick) == n_complete // factor):
    raise AssertionError('Resampling length mismatch')
...
if not (neural.shape[1] == inp.shape[1] == out.shape[1]):
    raise AssertionError('Trial stream lengths differ')
```

iii. The justification in the notes is that the behavior timestamps are the authoritative synchronized clock, so once the trial slice is defined, all derived signals should share the same `T`.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavior time series `environment`.

ii.
```python
names = ['trial_start', 'teleport', 'trial number', 'environment',
         'position', 'speed', 'lick', 'reward_zone']
beh = {k: np.asarray(bts[k].data[:]) for k in names}
...
env_native = aggregate_behavior(beh['environment'][s:e], factor, 'first')
```

iii. In Step 5 of the notes, the AI maps source code `0` to one environment and `1` to the other and describes the variable as binary and constant within trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI takes the first environment value in the trial, checks that the environment is constant and binary for the full trial, converts it to an integer, and repeats that value across all timepoints of the trial.

ii.
```python
env_native = aggregate_behavior(beh['environment'][s:e], factor, 'first')
env = int(round(float(env_native[0])))
if env not in (0, 1) or not np.all(env_native == env_native[0]):
    raise ValueError(f'{path}: invalid/nonconstant environment in trial {qi}')
...
np.full(T, env, dtype=np.float32)
```

iii. The notes say environment is a per-trial contextual variable and that per-trial variables are repeated over time only because the decoder expects `(d, T)` matrices.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the behavior time series `trial number`, specifically the value at the first sample of each trial.

ii.
```python
for qi, (s, e) in enumerate(pairs):
    trial_num = int(round(float(beh['trial number'][s])))
    if trial_num < 0:
        raise ValueError(f'{path}: negative trial number at start {s}')
```

iii. In Step 5 of the notes, the AI explicitly says to preserve the source trial number and repeat it across time. The trajectory shows it previously checked that trial-number irregularities occur around teleport frames and therefore uses the trial-start value.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The AI rounds/casts the `trial number` value at trial start to an integer, rejects negative values, and then repeats that scalar across the whole trial.

ii.
```python
trial_num = int(round(float(beh['trial number'][s])))
...
np.full(T, trial_num, dtype=np.float32)
```

iii. The notes justify this as preserving the source within-session trial identity, which is also needed for the reward-zone switch rule at trial 30.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the sparse `Reward.timestamps` series, combined with the trial boundaries from `trial_start`/`teleport`.

ii.
```python
pos_ts = np.asarray(bts['position'].timestamps[:common_n], dtype=np.float64)
reward_ts = np.asarray(bts['Reward'].timestamps[:], dtype=np.float64)
...
rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
outcomes.append(rewarded)
...
prev = outcomes[-2] if len(outcomes) > 1 else 0
```

iii. Step 5 of the notes says previous outcome is `1` iff reward was delivered in the immediately preceding valid trial, otherwise `0`, with the first trial forced to `0`.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each current trial, the AI first computes whether the current trial was rewarded by checking if any reward timestamp falls inside that trial’s `[start, end)` interval. It stores the sequence of outcomes, then for trial `q` uses the previous entry in that list, or `0` for the first trial. The result is repeated across timepoints.

ii.
```python
rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
outcomes.append(rewarded)
...
prev = outcomes[-2] if len(outcomes) > 1 else 0
...
np.full(T, prev, dtype=np.float32)
```

iii. The notes describe this as the requested binary “omitted vs rewarded” previous-trial variable and say all per-trial variables are repeated over time for decoder compatibility.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the trialwise position trace and a reward-zone identity inferred from the NWB session `identifier` plus the trial number. The behavior `reward_zone` time series is loaded but not used for the final zone assignment.

ii.
```python
src_zone, dst_zone = parse_zone_sequence(nwb.identifier)
...
zone = zone_for_trial(src_zone, dst_zone, trial_num)
_, dist_cls = distance_classes(pos, zone)
```

iii. Step 5 of the notes says zone identity comes from the task schedule encoded in the identifier because omission trials may have no positive `reward_zone` samples. The same section says switch sessions change from the source zone to the destination zone after the first 30 trials.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI converts zone identity into nominal zone boundaries `[80,130]`, `[200,250]`, or `[320,370]` cm, computes signed distance to the nearest point in that 50 cm interval, sets the distance to `0` inside the zone, then discretizes the result.

ii.
```python
ZONE_START = np.array([80.0, 200.0, 320.0], dtype=np.float32)
ZONE_END = ZONE_START + 50.0

def distance_classes(position: np.ndarray, zone: int) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = float(ZONE_START[zone]), float(ZONE_END[zone])
    d = np.where(position < lo, position - lo,
                 np.where(position > hi, position - hi, 0.0)).astype(np.float32)
    ...
    return d, c
```

iii. Step 5 of the notes says it uses the nominal A/B/C reward-zone intervals from the paper and defines distance as signed linear distance along the 450 cm corridor, with zero inside the zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is thresholded manually into seven categories corresponding to the task’s requested bins.

ii.
```python
c[d < -50] = 0
c[(d >= -50) & (d < -10)] = 1
c[(d >= -10) & (d < 0)] = 2
c[d == 0] = 3
c[(d > 0) & (d <= 10)] = 4
c[(d > 10) & (d <= 50)] = 5
c[d > 50] = 6
```

iii. Step 5 of the notes explicitly says boundary semantics were implemented to match the task’s strict inequalities exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance classes are computed from the same per-trial behavior slice `[s:e]` and must have the same length `T` as the neural matrix.

ii.
```python
neural = load_neural_trial(series_info, s, e, factor)
pos = aggregate_behavior(beh['position'][s:e], factor, 'mean').astype(np.float32)
...
_, dist_cls = distance_classes(pos, zone)
...
if not (neural.shape[1] == inp.shape[1] == out.shape[1]):
    raise AssertionError('Trial stream lengths differ')
```

iii. The notes repeatedly justify this by saying the NWB streams are already synchronized on the behavior/imaging frame clock.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the behavior time series `position`.

ii.
```python
pos = aggregate_behavior(beh['position'][s:e], factor, 'mean').astype(np.float32)
```

iii. The variable mapping table in Step 5 of the notes maps behavior `position` directly to absolute-position output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI averages within-bin only if temporal rebinning were needed, then clips position to `[0, 450]` cm before assigning categories.

ii.
```python
def position_classes(position: np.ndarray) -> np.ndarray:
    p = np.clip(position, 0.0, 450.0)
    c = np.zeros(p.shape, dtype=np.int64)
    c[p >= 90] = 1
    c[p >= 180] = 2
    c[p >= 270] = 3
    c[p > 360] = 4
    return c
```

iii. Step 5 of the notes says this clipping is meant to absorb small numerical edge noise while still producing the requested five equal 90 cm bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into five bins using comparisons at 90, 180, 270, and 360 cm, with exactly 360 cm remaining in bin 3 and values above 360 cm in bin 4.

ii.
```python
c = np.zeros(p.shape, dtype=np.int64)
c[p >= 90] = 1
c[p >= 180] = 2
c[p >= 270] = 3
c[p > 360] = 4
```

iii. The notes say this implements the task’s boundary semantics exactly for the five equal bins spanning the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is computed from the same per-trial slice and checked to have the same time dimension as neural, inputs, and the other outputs.

ii.
```python
pos = aggregate_behavior(beh['position'][s:e], factor, 'mean').astype(np.float32)
...
T = neural.shape[1]
if not (T == len(pos) == len(speed) == len(lick) == n_complete // factor):
    raise AssertionError('Resampling length mismatch')
```

iii. The AI’s justification is the same synchronized-clock argument used throughout the notes.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior time series `lick`.

ii.
```python
raw_lick = beh['lick'][s:e]
```

iii. The notes identify the raw lick stream as cumulative counts per frame and say event presence is represented as `> 0`.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI thresholds lick values to binary “lick/no lick” by checking whether the cumulative lick count is positive at a frame. If temporal rebinning were used, it would take the within-bin maximum; with `factor = 1`, this is just framewise thresholding.

ii.
```python
lick = (aggregate_behavior((raw_lick > 0).astype(np.int8), factor, 'max') > 0).astype(np.int64)
```

iii. Step 3 and Step 5 of the notes say the raw lick stream is cumulative and that the paper criterion for severe lick artifacts is tracked separately; the actual decoder output is binary lick presence.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is built from the same `[s:e]` trial slice and must match the neural trial length `T`.

ii.
```python
raw_lick = beh['lick'][s:e]
lick = (aggregate_behavior((raw_lick > 0).astype(np.int8), factor, 'max') > 0).astype(np.int64)
...
if not (T == len(pos) == len(speed) == len(lick) == n_complete // factor):
    raise AssertionError('Resampling length mismatch')
```

iii. The notes justify this by the one-to-one frame alignment of behavior and neural rows.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The AI derives reward-zone location from the NWB `identifier` string and the current trial number. It does not derive the final per-trial zone labels from the behavior `reward_zone` series.

ii.
```python
def parse_zone_sequence(identifier: str) -> tuple[int, int]:
    name = identifier.rstrip('/').split('/')[-1]
    letters = re.findall(r'(?:Location)?([ABC])', name)
    if '_to_' in name and len(letters) >= 2:
        return ord(letters[-2]) - 65, ord(letters[-1]) - 65
    if letters:
        z = ord(letters[-1]) - 65
        return z, z

zone = zone_for_trial(src_zone, dst_zone, trial_num)
```

iii. Step 5 of the notes explicitly says “Zone identity comes from identifier because omission trials may have no positive `reward_zone` samples.”

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses A/B/C letters from the session identifier, determines whether the session is a switch session, uses the source zone for trials `< 30` and the destination zone for trials `>= 30`, and repeats the resulting categorical value across all timepoints in the trial.

ii.
```python
def zone_for_trial(src: int, dst: int, trial_number: int) -> int:
    return src if trial_number < 30 else dst

zone = zone_for_trial(src_zone, dst_zone, trial_num)
...
np.full(T, zone, dtype=np.int64)
```

iii. The notes justify this with the task schedule: switch sessions change reward location after the first 30 trials, while stay sessions keep the same location throughout.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the sparse `Reward.timestamps` series and the per-trial start/end timestamps.

ii.
```python
pos_ts = np.asarray(bts['position'].timestamps[:common_n], dtype=np.float64)
reward_ts = np.asarray(bts['Reward'].timestamps[:], dtype=np.float64)
...
rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
```

iii. The Step 5 mapping table in the notes says reward outcome is `1` iff a reward timestamp lies in `[start, teleport)`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI checks whether any reward timestamp falls within that trial’s `[start, end)` interval, converts the result to `0/1`, and repeats it across all timepoints in the trial.

ii.
```python
rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
...
out = np.vstack([
    dist_cls, pos_cls, speed_cls, lick,
    np.full(T, zone, dtype=np.int64),
    np.full(T, rewarded, dtype=np.int64),
]).astype(np.int64, copy=False)
```

iii. The notes justify this as the requested per-trial reward/omission label; Step 12 further says low decoder performance on this target is a causal limitation, not a labeling bug.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates stream lengths to a common minimum across behavior arrays and Deconvolved arrays, uses behavior timestamps as the authoritative clock, and validates many conditions with hard errors: matched trial markers, constant/binary environment, nonnegative trial number at start, finite converted arrays, and supported effective sample rate. Lick-artifact trials are detected and logged but retained.

ii.
```python
common_n = min([len(v) for v in beh.values()] +
               [int(rs.data.shape[0]) for rs, _ in series_info])
...
if len(pairs) != len(starts) or len(starts) != len(ends):
    raise ValueError(...)
...
if env not in (0, 1) or not np.all(env_native == env_native[0]):
    raise ValueError(...)
...
if not np.isfinite(neural).all() or not np.isfinite(inp).all():
    raise ValueError(...)
```

iii. Step 10 of the notes explains the main discovered issue: two-plane metadata rates were inconsistent with the synchronized behavior clock, so the code was corrected to trust behavior timestamps and truncate only to the shared length. The notes also justify retaining lick-artifact trials because the output format lacks per-output masking.

## 13-a. What are the most time-consuming steps of the code?

i. The code’s most time-consuming work is opening 152 large NWB files with `pynwb`, reading trial slices from the Deconvolved arrays for every plane and trial, and serializing the final multi-gigabyte pickle.

ii.
```python
for i, p in enumerate(files, 1):
    ns, xs, ys, info = process_session(p)
    ...

with NWBHDF5IO(str(path), 'r') as io:
    nwb = io.read()
    ...

with out.open('wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 6 says repeated NWB opens would dominate I/O, so the script was designed to open each file once and process it session-by-session. Step 9 reports that the full conversion took 85 s and produced a 9.843 GB pickle, indicating I/O and serialization dominate runtime.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main candidate is the per-trial loop in `process_session()`, which repeatedly slices behavior and neural data and constructs trial matrices one trial at a time. The start/end pairing loop in `pair_bounds()` is also scalar. Most within-trial discretization is already vectorized.

ii.
```python
def pair_bounds(starts: np.ndarray, ends: np.ndarray) -> list[tuple[int, int]]:
    pairs, j = [], 0
    for s in starts:
        ...

for qi, (s, e) in enumerate(pairs):
    ...
    neural = load_neural_trial(series_info, s, e, factor)
    pos = aggregate_behavior(beh['position'][s:e], factor, 'mean').astype(np.float32)
    ...
```

iii. Step 6 of the notes says the script already uses vectorized discretization/aggregation where practical and avoids materializing full-session matrices, implying the remaining trial loop is mostly a variable-length-interval convenience rather than an oversight.

## 13-c. What processing does the code repeat multiple times?

i. Inside each trial, the code repeatedly slices several behavior arrays from the same `[s:e]` window and separately aggregates position, speed, lick, and environment. At the session level, it also separately computes current reward outcome, previous reward outcome, zone identity, and per-trial metadata from the same trial boundaries.

ii.
```python
for qi, (s, e) in enumerate(pairs):
    ...
    raw_lick = beh['lick'][s:e]
    neural = load_neural_trial(series_info, s, e, factor)
    pos = aggregate_behavior(beh['position'][s:e], factor, 'mean').astype(np.float32)
    speed = aggregate_behavior(beh['speed'][s:e], factor, 'mean').astype(np.float32)
    lick = (aggregate_behavior((raw_lick > 0).astype(np.int8), factor, 'max') > 0).astype(np.int64)
    env_native = aggregate_behavior(beh['environment'][s:e], factor, 'first')
```

iii. Step 6 explicitly says the code was designed to avoid bigger repeated work such as reopening NWB files or loading full-session uncurated fluorescence arrays, so the repetition that remains is mostly within-trial extraction from already opened sessions.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes some values that are not used by the downstream decoder itself: `distance_classes()` returns both continuous distance and categories but only the categories are used; it loads the `reward_zone` behavior stream but does not use it in the final outputs; it computes `is_switch`, `lick_artifact`, and detailed `trial_info`/`session_info` for documentation rather than decoding.

ii.
```python
names = ['trial_start', 'teleport', 'trial number', 'environment',
         'position', 'speed', 'lick', 'reward_zone']
...
_, dist_cls = distance_classes(pos, zone)
...
is_switch = src_zone != dst_zone
...
trial_info.append({'trial_number': trial_num, 'start': s, 'end': e,
                   'environment': env, 'zone': zone, 'rewarded': rewarded,
                   'lick_artifact': lick_artifact, 'T': T})
...
'session_info': infos,
```

iii. The notes emphasize heavy documentation and sanity checking throughout the workflow, so these extra computations are mainly there to support validation, metadata, and debugging rather than the decoder inputs/outputs themselves.
