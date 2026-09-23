# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI recursively discovers every `.nwb` file under `/app/data`, sorts them, and processes every file by default (152 sessions). Each file is opened with `pynwb.NWBHDF5IO`; neural arrays are read per trial and behavioral arrays per session.

ii.
```python
files = sorted(DATA_ROOT.rglob('*.nwb'))
with NWBHDF5IO(str(path), 'r') as io:
    nwb = io.read()
```

iii. It justified this as including all available switch and stay sessions, because “full” means all provided data, and emphasized compliance with the required `pynwb` API.

## 1-b. How are the data split into subjects?

i. Subject IDs come from `nwb.subject.subject_id`. After all sessions are processed, unique IDs are naturally sorted by their numeric component and each session receives an index into that list.

ii.
```python
subject = str(nwb.subject.subject_id)
subjects = sorted(set(x['subject'] for x in infos), key=lambda z: int(re.sub(r'\D','',z)))
subject_idx = np.array([subjects.index(x['subject']) for x in infos], dtype=np.int64)
```

iii. The notes report 11 subjects and validate the IDs and switch-session cell totals against the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; session order is the sorted file-path order. The NWB `session_id` is retained in metadata.

ii.
```python
for i, p in enumerate(files, 1):
    ns, xs, ys, info = process_session(p)
    neural.append(ns); inputs.append(xs); outputs.append(ys); infos.append(info)
```

iii. The AI states that all 152 available files are used: 77 switch and 75 stay sessions, with two expected m11 stay files absent from the supplied data.

## 1-d. How are the data split into trials?

i. Trial starts are positive `trial_start` samples and ends are positive `teleport` samples. `pair_bounds` pairs each start with the next unused teleport, and trials are half-open `[start, teleport)`, excluding the teleport frame.

ii.
```python
starts = np.flatnonzero(beh['trial_start'][:common_n] > 0)
ends = np.flatnonzero(beh['teleport'][:common_n] > 0)
pairs = pair_bounds(starts, ends)
for qi, (s, e) in enumerate(pairs):
```

iii. The notes say this matches the reference trial-start/teleport convention and prevents reset-position artifacts at ITI entry; all 12,216 starts and ends paired uniquely.

## 1-e. How are trials filtered based on quality controls?

i. No paired trial is removed. Trials with severe lick artifacts are counted and annotated, but retained. Invalid boundaries, negative trial numbers, nonconstant/invalid environments, zero complete bins, nonfinite data, or stream-length mismatch cause an exception rather than selective filtering.

ii.
```python
lick_artifact = bool(np.mean(raw_lick > 2) > 0.30)
lick_artifacts += int(lick_artifact)
neural_trials.append(neural)
```

iii. The AI argued that deleting an entire lick-artifact trial would discard otherwise valid outputs and the target format supplies no missing-output mask. It found all trials were at least 97 frames, so no short-trial rule was implemented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB `processing/ophys/Deconvolved` RoiResponseSeries for every plane, plus each series’ ROI region and `iscell` labels.

ii.
```python
deconv = nwb.processing['ophys']['Deconvolved'].roi_response_series
region = np.asarray(rs.rois.data[:], dtype=np.int64)
labels = np.asarray(rs.rois.table['iscell'][:])
```

iii. The AI reasoned that this exported Suite2p deconvolution corresponds to the paper’s processed “events” and avoids wasteful recomputation from Fluorescence and Neuropil.

## 2-b. How is the `neural` data processed?

i. For each plane and trial, the code slices the stored deconvolved matrix, retains accepted columns, optionally sums groups of samples if `factor > 1` (in actual conversion `factor=1`), transposes to neuron-by-time, concatenates planes, and casts to float32. It does not recompute neuropil-corrected dF/F or OASIS events.

ii.
```python
block = np.asarray(rs.data[start:start+n, :], dtype=np.float32)
block = block[:, accepted_cols]
planes.append(block.T)
return np.concatenate(planes, axis=0).astype(np.float32, copy=False)
```

iii. The AI treated the NWB `Deconvolved` stream as already processed and synchronized, citing exact accepted-cell counts as validation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs whose mapped Suite2p `iscell` first column is greater than 0.5 are retained. All planes are included. There is no removal of cells whose dF/F is highly correlated with running speed.

ii.
```python
keep = labels[region, 0] > 0.5
accepted_local_cols = np.flatnonzero(keep)
```

iii. The AI viewed `iscell` as the appropriate generic curation and said place-cell/running filters were analysis-specific and inappropriate for a decoder that includes stationary behavior.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural rows are assumed one-to-one with behavioral timestamp rows. Slicing begins at each `trial_start` index, so column zero is aligned to trial start; no interpolation or offset correction is applied.

ii.
```python
neural = load_neural_trial(series_info, s, e, factor)
elapsed = np.arange(T, dtype=np.float32) / np.float32(TARGET_RATE)
```

iii. The notes report raw spot checks showing aligned neural and behavioral lengths and use the behavior clock as authoritative when NWB series-rate metadata conflict.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is `1000 / 15.5078125 = 64.4836 ms`. No temporal rebinning is applied in the executed path (`factor=1`), including for two-plane files.

ii.
```python
TARGET_RATE = 15.5078125
BIN_MS = 1000.0 / TARGET_RATE
factor = 1
```

iii. The AI found every neural row aligns with the 15.5078125-Hz behavior clock; it concluded the 31.015625-Hz two-plane metadata are inconsistent and should not trigger downsampling.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the trial’s sample count and the fixed target rate; the position timestamps are used only to estimate and validate that rate.

ii.
```python
pos_clock = np.asarray(bts['position'].timestamps[:], dtype=np.float64)
effective_rate = 1.0 / float(np.median(np.diff(pos_clock)))
elapsed = np.arange(T, dtype=np.float32) / np.float32(TARGET_RATE)
```

iii. The AI justified this because behavior and neural samples are synchronized on a uniform clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based sample index is divided by 15.5078125 Hz and cast to float32.

ii.
```python
elapsed = np.arange(T, dtype=np.float32) / np.float32(TARGET_RATE)
```

iii. This makes the first time point exactly zero and advances by one common time bin.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector is created with exactly `neural.shape[1]` samples, so it directly indexes neural columns from trial start.

ii.
```python
T = neural.shape[1]
elapsed = np.arange(T, dtype=np.float32) / np.float32(TARGET_RATE)
```

iii. Shape assertions and raw-versus-converted spot checks were cited as alignment validation.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the behavior `environment` stream within each trial.

ii.
```python
env_native = aggregate_behavior(beh['environment'][s:e], factor, 'first')
env = int(round(float(env_native[0])))
```

iii. The notes state the source codes are already binary ENV1=0 and ENV2=1.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The first sample is rounded to an integer, validated as 0 or 1 and constant over the trial, then repeated across all time points.

ii.
```python
if env not in (0, 1) or not np.all(env_native == env_native[0]):
    raise ValueError(...)
np.full(T, env, dtype=np.float32)
```

iii. This implements the required per-trial binary input while explicitly checking the assumption of constancy.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It uses the `trial number` behavior stream at the trial-start index.

ii.
```python
trial_num = int(round(float(beh['trial number'][s])))
```

iii. The AI chose to preserve the source’s zero-based task trial number, which is also used to locate the trial relative to the reward switch.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The start value is rounded, converted to integer, checked to be nonnegative, and repeated across the trial.

ii.
```python
if trial_num < 0:
    raise ValueError(...)
np.full(T, trial_num, dtype=np.float32)
```

iii. The AI describes per-trial variables as repeated because the supplied decoder expects `(d, T)` matrices.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from sparse `Reward.timestamps`, position timestamps defining trial intervals, and the preceding trial in file order.

ii.
```python
reward_ts = np.asarray(bts['Reward'].timestamps[:], dtype=np.float64)
rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
outcomes.append(rewarded)
```

iii. The AI used timestamp containment as a robust way to determine whether reward was delivered during a trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each current outcome is computed first; `prev` is the prior entry in `outcomes`, or 0 for the first trial, and is repeated over time.

ii.
```python
prev = outcomes[-2] if len(outcomes) > 1 else 0
np.full(T, prev, dtype=np.float32)
```

iii. This directly implements omitted=0/rewarded=1 and the specified first-trial default.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from behavior `position` and a nominal zone identity parsed from the NWB identifier. Zone identity follows the encoded source/destination and switches at trial number 30.

ii.
```python
src_zone, dst_zone = parse_zone_sequence(nwb.identifier)
zone = zone_for_trial(src_zone, dst_zone, trial_num)
_, dist_cls = distance_classes(pos, zone)
```

iii. The AI preferred the known task schedule because omission trials may lack positive `reward_zone` samples; it validated the mapping against observed entries and class balance.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For nominal intervals A=[80,130], B=[200,250], and C=[320,370] cm, signed distance is position minus the near edge before a zone, zero inside, and position minus the far edge after it. Only the categorical result is saved.

ii.
```python
d = np.where(position < lo, position - lo,
             np.where(position > hi, position - hi, 0.0)).astype(np.float32)
```

iii. The AI says this is linear signed distance to the nearest point in the current 50-cm zone, matching the requested before/inside/after semantics.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks produce seven categories: `<-50`, `[-50,-10)`, `[-10,0)`, exactly 0, `(0,10]`, `(10,50]`, and `>50`.

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

iii. The AI explicitly tested boundary semantics against the task wording.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the same `[s:e)` indices and factor; the code asserts their final lengths match.

ii.
```python
neural = load_neural_trial(series_info, s, e, factor)
pos = aggregate_behavior(beh['position'][s:e], factor, 'mean')
if not (T == len(pos) == len(speed) == len(lick)):
    raise AssertionError(...)
```

iii. The AI relies on the verified one-to-one NWB synchronization and structural assertions.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior `position` stream.

ii.
```python
pos = aggregate_behavior(beh['position'][s:e], factor, 'mean').astype(np.float32)
```

iii. The source is documented as corridor position in centimeters on the synchronized behavior clock.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. With the actual factor of one, the per-trial values are unchanged; the helper clips them to [0,450] before categorization. No continuous position is saved.

ii.
```python
p = np.clip(position, 0.0, 450.0)
pos_cls = position_classes(pos)
```

iii. The AI described clipping as handling small numerical excursions outside the physical track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five classes are assigned at 90-cm boundaries. Exactly 90, 180, and 270 enter the higher class; exactly 360 remains class 3 because the last class is strictly `>360`.

ii.
```python
c[p >= 90] = 1
c[p >= 180] = 2
c[p >= 270] = 3
c[p > 360] = 4
```

iii. The AI states these comparisons implement the instruction’s inequalities exactly.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same trial slice and synchronized row clock as neural data; equal length is asserted.

ii.
```python
pos = aggregate_behavior(beh['position'][s:e], factor, 'mean')
T = neural.shape[1]
```

iii. Raw spot checks and stream-length assertions support the alignment decision.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the behavior `lick` stream.

ii.
```python
raw_lick = beh['lick'][s:e]
```

iii. The AI recognized the stream as cumulative counts per frame rather than an already binary variable.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Raw values greater than zero are converted to one. The (inactive) aggregation path takes a maximum within a bin. Severe artifact trials are flagged but their binary output is retained.

ii.
```python
lick = (aggregate_behavior((raw_lick > 0).astype(np.int8), factor, 'max') > 0).astype(np.int64)
lick_artifact = bool(np.mean(raw_lick > 2) > 0.30)
```

iii. The target requires a binary series, and the AI says retaining artifact trials avoids losing otherwise valid data without a masking mechanism.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is sliced with the same `[s:e)` bounds and asserted to have the same final length as neural data.

ii.
```python
raw_lick = beh['lick'][s:e]
if not (T == len(pos) == len(speed) == len(lick)):
    raise AssertionError(...)
```

iii. The behavior clock is treated as synchronized one-to-one with neural rows.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `nwb.identifier` plus the raw `trial number`, not from the framewise `reward_zone` values.

ii.
```python
src_zone, dst_zone = parse_zone_sequence(nwb.identifier)
zone = zone_for_trial(src_zone, dst_zone, trial_num)
```

iii. The AI says identifiers encode fixed or source-to-destination sessions and are reliable even on omitted trials.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regex extracts A/B/C letters and maps them to 0/1/2. Fixed sessions retain one zone; switch sessions use source for trial numbers below 30 and destination from 30 onward. The result is repeated over time.

ii.
```python
def zone_for_trial(src: int, dst: int, trial_number: int) -> int:
    return src if trial_number < 30 else dst
np.full(T, zone, dtype=np.int64)
```

iii. This is justified by the paper’s task schedule and the known switch after the first 30 trials.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It comes from sparse behavior `Reward.timestamps` and position timestamps at trial boundaries.

ii.
```python
pos_ts = np.asarray(bts['position'].timestamps[:common_n], dtype=np.float64)
reward_ts = np.asarray(bts['Reward'].timestamps[:], dtype=np.float64)
```

iii. The AI says sparse timestamps directly represent delivered rewards.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is rewarded if any reward timestamp is in `[pos_ts[s], pos_ts[e])`; the resulting 0/1 value is repeated over the trial.

ii.
```python
rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
np.full(T, rewarded, dtype=np.int64)
```

iii. Half-open timestamp containment matches the trial interval and the requested per-trial binary output.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. All streams are truncated implicitly to `common_n`, the minimum behavior/neural length, before trial markers are found. Inconsistent rates, plane rates, markers, environments, lengths, or nonfinite values raise errors. Small negative speed and out-of-track position are clamped for categorization. Missing supplied sessions are documented rather than fabricated. Missing reward-zone samples do not matter because the task identifier supplies zone identity.

ii.
```python
common_n = min([len(v) for v in beh.values()] +
               [int(rs.data.shape[0]) for rs, _ in series_info])
if not np.isfinite(neural).all() or not np.isfinite(inp).all():
    raise ValueError(...)
```

iii. The notes identify ten one-sample neural/behavior mismatches and choose safe common-length truncation; extensive assertions and exact raw-data spot checks are cited.

## 13-a. What are the most time-consuming steps of the code?

i. Reading and materializing large deconvolved trial slices, retaining them all in memory, serializing the roughly 9.8-GB pickle, optional plotting, and downstream decoder training are the expensive operations. The converter records per-session and total elapsed time; its full run took about 85 seconds.

ii.
```python
block = np.asarray(rs.data[start:start+n, :], dtype=np.float32)
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The AI notes that loading full fluorescence would be wasteful and therefore reads only trial slices of the selected deconvolved stream.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The sequential trial loop could partially vectorize reward-outcome assignment, categorical transforms, and artifact statistics at session level. The plane loop and variable-length trial construction are naturally retained; `pair_bounds` could be replaced by a vectorized search if needed.

ii.
```python
for qi, (s, e) in enumerate(pairs):
    rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
```

iii. The AI did not explicitly enumerate vectorization candidates in its notes, but designed helpers with NumPy operations inside each variable-length trial and prioritized slice-wise memory use.

## 13-c. What processing does the code repeat multiple times?

i. For every trial it rescans all reward timestamps, allocates repeated constant rows, and separately invokes aggregation/slicing for position, speed, lick, and environment. Each plane is also sliced separately for each trial.

ii.
```python
rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
np.full(T, env, dtype=np.float32)
np.full(T, trial_num, dtype=np.float32)
```

iii. The AI’s documentation does not call these out as repeated work; it emphasizes that per-trial repetition is required by the decoder’s matrix format.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes but discards continuous distance, repeatedly parses/retains extensive `session_info`, detects lick artifacts without changing outputs, and computes plotting-only data when requested. `aggregate_behavior` and neural downsampling support factors greater than one even though `factor` is always one.

ii.
```python
_, dist_cls = distance_classes(pos, zone)
lick_artifact = bool(np.mean(raw_lick > 2) > 0.30)
trial_info.append({...})
```

iii. The AI justified artifact detection and metadata as validation/audit information. It explicitly avoided the much larger unnecessary step of loading raw fluorescence.
