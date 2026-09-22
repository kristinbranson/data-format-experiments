# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively finds every `.nwb` file under `/app/data`, sorts the paths, and treats each file as a session. `convert_session` opens each file with `pynwb.NWBHDF5IO`; full mode processes all 152 files (sample mode only the first two).

ii.
```python
files=sorted(DATA_ROOT.rglob('*.nwb'))
if args.sample: files=files[:2]
for j,path in enumerate(files):
    sn,si,so,info=convert_session(path,args.show_processing and j<2)
```
```python
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read()
```

iii. The notes report 152 NWB files in `sub-<id>` directories and emphasize that all access used `pynwb`, as required. Recursive discovery avoids a hard-coded session list.

## 1-b. How are the data split into subjects?

i. Subject identity is read from `nwb.subject.subject_id` for each session. Unique names are sorted, and each session receives an index into that list.

ii.
```python
subject = str(nwb.subject.subject_id)
subject_names=sorted(set(subjects))
'subject_idx':np.asarray([subject_names.index(x) for x in subjects],dtype=np.int32)
```

iii. The notes verified 11 subjects and preferred NWB metadata over parsing directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file is one output session. Its trial lists are appended once to the top-level `neural`, `input`, and `output` lists.

ii.
```python
for j,path in enumerate(files):
    sn,si,so,info=convert_session(path,...)
    neural.append(sn); inputs.append(si); outputs.append(so)
```

iii. The notes identify approximately one NWB file per imaging session and report 152 output sessions.

## 1-d. How are the data split into trials?

i. The agent takes every finite, nonnegative native `trial number`, then obtains all frames equal to each ID. It does not directly slice from `trial_start` through a teleport edge; start pulses and track completion are instead validity checks.

ii.
```python
trial_ids = np.unique(arrays['trial number'][np.isfinite(arrays['trial number']) &
                                             (arrays['trial number'] >= 0)]).astype(int)
for tid in trial_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
```

iii. The agent found no NWB trials table and judged native trial-number grouping more robust than teleport pulses, which could occupy zero, one, or two samples after resampling.

## 1-e. How are trials filtered based on quality controls?

i. A trial is retained if it has at least 20 frames, at least one positive `trial_start`, reaches position 440 cm, and has `scanning == 1` throughout. Sessions must retain at least two trials. This excluded one malformed three-frame terminal fragment.

ii.
```python
complete = (len(ix) >= 20 and np.sum(arrays['trial_start'][ix] > 0) >= 1
            and np.nanmax(arrays['position'][ix]) >= 440
            and np.all(arrays['scanning'][ix] == 1))
if complete:
    valid_ids.append(tid)
```

iii. The notes justify these as completeness and imaging-validity checks while retaining both rewarded and omitted trials; they report 12,216 valid trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB `ophys/Deconvolved` RoiResponseSeries, restricted using the segmentation table's `iscell` field. Raw `Fluorescence` and `Neuropil` are not used.

ii.
```python
deconv_container = nwb.processing['ophys']['Deconvolved']
series = list(deconv_container.roi_response_series.values())
iscell = np.asarray(r.rois.table['iscell'][:])
```

iii. The agent reasoned that the released deconvolved activity was already the paper's analysis signal and explicitly said recomputing dF/F was unnecessary. The human reference reaches the opposite conclusion.

## 2-b. How is the `neural` data processed?

i. For every imaging plane, the code maps the series' linked segmentation rows, selects `iscell == 1`, casts to float32, concatenates planes by neuron, indexes trial frames, and transposes time-by-cell into neuron-by-time. It performs no neuropil subtraction, baseline calculation, smoothing, or fresh OASIS deconvolution.

ii.
```python
linked_rows = np.asarray(r.rois.data[:], dtype=np.int64)
local_cell_mask = (binary[linked_rows] == 1)
neural_parts.append(np.asarray(r.data[:common_n, local_cell_mask], dtype=np.float32))
neural_all = np.ascontiguousarray(np.concatenate(neural_parts, axis=1), dtype=np.float32)
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
```

iii. The notes say this preserves released Suite2p event amplitudes and avoids imposing analysis-specific normalization. They also document a later audit of DynamicTableRegion mapping and multi-plane concatenation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only Suite2p `iscell[:,0] == 1` ROIs are retained. The agent does not remove putative interneurons based on correlation with running speed.

ii.
```python
binary = iscell[:, 0] if iscell.ndim == 2 else iscell
local_cell_mask = (binary[linked_rows] == 1)
```

iii. The agent regarded place-cell/RR/TR filters as downstream scientific subsets and `iscell` as the appropriate general quality filter. It did not recognize the paper/reference's speed-correlation interneuron exclusion as required.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All samples bearing a trial ID are indexed together, and the first such frame is treated as time zero. A start pulse is required somewhere in that group, but the code does not explicitly start `ix` at the pulse.

ii.
```python
ix = np.flatnonzero(arrays['trial number'] == tid)
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
inp[0] = np.arange(nt, dtype=np.float32) * np.float32(DT)
```

iii. The agent states that native numbered-trial samples are already frame-aligned and describes the event as the first aligned imaging frame/trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data remain at a fixed 0.0644836272 s interval (64.483627 ms, about 15.508 Hz); no temporal binning, interpolation, or resampling is applied.

ii.
```python
DT = 0.06448362720402656
'time_bin_size':DT*1000.0,
'sampling_rate_hz':1.0/DT,
```

iii. The notes report that every frame-aligned stream uses this interval and argue native resolution is needed for time-varying decoding.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the number of selected trial frames and the hard-coded native interval `DT`, not directly from stored timestamps.

ii.
```python
nt = len(ix)
inp[0] = np.arange(nt, dtype=np.float32) * np.float32(DT)
```

iii. The agent verified a common fixed sampling interval and therefore treated frame count times `DT` as equivalent to timestamp subtraction.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based frame index is multiplied by `DT`, producing 0, DT, 2DT, and so on.

ii.
```python
inp[0] = np.arange(nt, dtype=np.float32) * np.float32(DT)
```

iii. The stated goal is a trial-relative clock beginning at the first aligned frame without resampling.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Its length is constructed from the same `ix` used to slice neural data, and an assertion enforces equal lengths.

ii.
```python
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
if neu.shape[1] != nt or inp.shape[1] != nt or out.shape[1] != nt:
    raise AssertionError('Within-trial temporal mismatch')
```

iii. The notes describe all behavior and calcium data as sharing the imaging-frame timeline.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the frame-aligned `environment` series; the modal finite value within the trial is used. The scene identifier is also parsed to validate the expected environment.

ii.
```python
env_values = arrays['environment'][ix]
u, c = np.unique(env_values[np.isfinite(env_values)], return_counts=True)
env_mode = u[np.argmax(c)]
env, zone = parse_scene(scene, tid, env_mode)
```

iii. The agent verified that the stream codes ENV1 as 0 and ENV2 as 1, including switch sessions, and regarded it as authoritative.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The per-trial mode is rounded/cast to an integer, checked against the environment encoded in the scene name, and repeated across every trial frame.

ii.
```python
env = int(round(float(env_mode)))
inp[1] = env
```

iii. Modal reduction guards against isolated irregular values while producing the required binary per-trial feature.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the native frame-aligned NWB `trial number` ID (`tid`).

ii.
```python
trial_ids = np.unique(arrays['trial number'][...]).astype(int)
inp[2] = tid
```

iii. The notes call these native zero-based trial numbers and use negative values only to identify outside-trial samples.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Finite nonnegative IDs are selected, cast to integers, sorted by `np.unique`, and the current ID is broadcast over its trial.

ii.
```python
trial_ids = np.unique(...).astype(int)
inp[2] = tid
```

iii. The agent preserves native IDs rather than renumbering retained trials.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It ultimately derives from the timestamped NWB `Reward` events, which are mapped to nearest position-frame timestamps and then to native trial IDs.

ii.
```python
reward_times = np.asarray(reward.timestamps[:], dtype=np.float64)
event_frames = nearest_frame_indices(frame_times, reward_times)
event_trial_ids = arrays['trial number'][event_frames]
rewarded_ids = set(int(x) for x in event_trial_ids if np.isfinite(x) and x >= 0)
```

iii. The notes reject invariant `autoreward` and say timestamped Reward delivery directly defines rewarded versus omitted trials.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The first valid trial gets 0. Thereafter, `prev_outcome` is the outcome of the preceding retained trial, repeated across time, and updated after each valid trial.

ii.
```python
prev_outcome = 0
for tid in valid_ids:
    ...
    inp[3] = prev_outcome
    ...
    prev_outcome = outcome
```

iii. The agent interprets 0 for the first trial as “no prior rewarded trial available.” Its behavior differs from checking the immediately preceding native trial only if an intervening native trial were filtered.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses frame-aligned `position` plus an A/B/C zone inferred from the NWB identifier/scene string, the native trial ID, and (for combined switches) the environment stream. It does not derive the zone from the raw `reward_zone` series.

ii.
```python
scene = str(nwb.identifier).rstrip('/').split('/')[-1]
env, zone = parse_scene(scene, tid, env_mode)
low, high = ZONE_BOUNDS[zone]
dist = signed_distance_to_interval(pos, low, high)
```

iii. The agent concluded that raw `reward_zone` values 0–7 were transient states rather than labels, and used task schedules encoded in scene names, cross-checked against nonzero-state positions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The agent defines zones A/B/C as 80–100, 200–220, and 320–340 cm. Distance is negative before the lower bound, zero inside the closed interval, and positive after the upper bound.

ii.
```python
ZONE_BOUNDS = {'A': (80.0, 100.0), 'B': (200.0, 220.0), 'C': (320.0, 340.0)}
return np.where(position < low, position-low,
                np.where(position > high, position-high, 0.0))
```

iii. The agent says the native reward-state positions center near 82/202/322 cm and treats those scene-defined 20 cm intervals as zone geometry. The human reference instead uses 50 cm zones ending at 130/250/370.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks implement the seven required categories, including exact boundaries: `-50` is class 1, `-10` is class 1, zero is class 3, `10` class 4, and `50` class 5.

ii.
```python
y[d < -50] = 0
y[(d >= -50) & (d <= -10)] = 1
y[(d > -10) & (d < 0)] = 2
y[d == 0] = 3
y[(d > 0) & (d <= 10)] = 4
y[(d > 10) & (d <= 50)] = 5
y[d > 50] = 6
```

iii. The function documents that it follows the task-prescribed seven classes and boundary conventions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity are sliced with the identical native frame indices `ix`; output length is asserted equal to neural length.

ii.
```python
pos = arrays['position'][ix].astype(np.float32)
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
if neu.shape[1] != nt or ...: raise AssertionError(...)
```

iii. The agent's audits found exact neural/input/output length matches on the shared frame timeline.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from the frame-aligned NWB `position` series.

ii.
```python
pos = arrays['position'][ix].astype(np.float32)
out[1] = discretize_position(pos)
```

iii. The notes identify position as the animal's absolute forward position on the 450 cm corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is cast to float32 and categorized directly; it is not smoothed, clipped, interpolated, or spatially rebinned first.

ii.
```python
def discretize_position(x):
    y = np.zeros(x.shape, dtype=np.int8)
    ...
    return y
```

iii. The agent preserves native samples, allowing negative teleport samples to fall naturally into the first class.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Classes are `<90`, `90–<180`, `180–<270`, `270–360` inclusive, and `>360` cm.

ii.
```python
y[(x >= 90) & (x < 180)] = 1
y[(x >= 180) & (x < 270)] = 2
y[(x >= 270) & (x <= 360)] = 3
y[x > 360] = 4
```

iii. The agent explicitly follows the instruction that exact 360 remains in class 3. This differs at exact internal edges from the reference's `np.digitize` defaults.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Both are indexed by the same `ix` and checked for identical time dimension.

ii.
```python
pos = arrays['position'][ix].astype(np.float32)
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
```

iii. The justification is the native shared imaging-frame timebase.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the frame-aligned NWB `lick` stream.

ii.
```python
lick = arrays['lick'][ix]
out[3] = (lick > 0).astype(np.int8)
```

iii. The notes report native lick values of 0–8 and interpret them as event/count-like sensor values.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Every positive value becomes 1 and all other values become 0; no smoothing or sensor-failure trial removal is applied.

ii.
```python
out[3] = (lick > 0).astype(np.int8)
```

iii. The agent says binary thresholding is required by the decoder and GLM smoothing is analysis-specific.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick stream is sliced with the exact same trial-frame indices as neural data.

ii.
```python
lick = arrays['lick'][ix]
neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
```

iii. The notes and diagnostics report exact frame-level raw-to-converted alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from the NWB identifier's scene name, trial ID, and modal environment; the raw `reward_zone` stream is only an exploratory cross-check and is absent from `needed`.

ii.
```python
scene = str(nwb.identifier).rstrip('/').split('/')[-1]
env, zone = parse_scene(scene, tid, env_mode)
out[4] = ZONE_INDEX[zone]
```

iii. The agent considered native reward-zone values ambiguous state/event combinations and trusted the explicit task schedule encoded by scene names.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Stable scenes keep their named location; `LocationX_to_Y` changes at native trial 40; combined environment/location scenes choose the location matching the current environment (described as switching at trial 30). A/B/C are mapped to 0/1/2 and repeated across time.

ii.
```python
zone = second if second is not None and trial_id >= 40 else first
ZONE_INDEX = {'A': 0, 'B': 1, 'C': 2}
out[4] = ZONE_INDEX[zone]
```

iii. The notes say this schedule was checked against native reward-state positions and fixed combined-switch handling during validation.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Outcome derives from timestamped `Reward` events and position timestamps used as the common frame clock, then from the trial-number stream at each mapped event frame.

ii.
```python
frame_times = np.asarray(pts.timestamps[:common_n], dtype=np.float64)
reward_times = np.asarray(reward.timestamps[:], dtype=np.float64)
event_frames = nearest_frame_indices(frame_times, reward_times)
event_trial_ids = arrays['trial number'][event_frames]
```

iii. The agent independently checked rewarded and omitted examples and found exact agreement with raw events.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Each reward timestamp is mapped to the nearest frame (with endpoint clipping). Trials whose native ID occurs among event frames get 1; all others get 0, broadcast over the trial.

ii.
```python
idx = np.clip(np.searchsorted(frame_times, event_times), 0, len(frame_times)-1)
left = np.maximum(idx-1, 0)
return np.where(np.abs(frame_times[left]-event_times) < np.abs(frame_times[idx]-event_times), left, idx)
...
outcome = int(tid in rewarded_ids)
out[5] = outcome
```

iii. Nearest-frame matching was chosen over `searchsorted` alone, and the notes report raw-event audits; unlike the reference, the production code has no explicit half-bin error assertion.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. All neural and required behavior streams are truncated to their common minimum length, handling ten files with one extra terminal neural frame. Trials with missing/invalid environment, nonfinite neural/input data, incomplete traversal, absent start, or invalid scanning are rejected or raise errors. Reward indices are endpoint-clipped; internal shape, finite-value, class-range, and minimum-trial assertions validate output. There is no imputation.

ii.
```python
common_n = min([r.data.shape[0] for r in series] + [len(v) for v in arrays.values()])
arrays = {k: v[:common_n] for k, v in arrays.items()}
...
if not len(u): raise ValueError(...)
if not (np.all(np.isfinite(neu)) and np.all(np.isfinite(inp))): raise ValueError(...)
```

iii. The agent established that all observed length mismatches were a single terminal frame and considered common-length truncation safe. It excluded one malformed terminal trial and used fail-fast validation elsewhere.

## 13-a. What are the most time-consuming steps of the code?

i. The code times each session and the full run. The costly work is opening 152 large NWB files, reading every Deconvolved series and behavioral stream into NumPy arrays, concatenating/copying neural data, constructing trial copies, serializing the approximately 13.3 GB pickle, and later decoder training. The notes do not provide a dedicated complexity analysis.

ii.
```python
t0 = time.time()
neural_parts.append(np.asarray(r.data[:common_n, local_cell_mask], dtype=np.float32))
...
with out.open('wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Documentation warns that the pickle is large and requires substantial RAM; progress logs report per-session and total elapsed time.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The file/session loop is inherently I/O-oriented, but trial validity and per-trial conversion loops could partly be vectorized by computing framewise categories/outcomes once and splitting afterward. `subject_names.index(x)` could use a dictionary. Plane iteration is needed for distinct ROI links. The agent did not document these opportunities.

ii.
```python
for tid in trial_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
...
for tid in valid_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
```

iii. The trajectory emphasizes correctness checks rather than optimization; variable trial lengths make some per-trial work natural, but repeated full-array equality scans are avoidable.

## 13-c. What processing does the code repeat multiple times?

i. For each trial ID, `np.flatnonzero(arrays['trial number'] == tid)` is performed once during validity screening and again during conversion, repeatedly scanning the full session array. Scene parsing, categorical allocation, and per-trial slicing also repeat. Validation later traverses all trials again.

ii.
```python
for tid in trial_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
...
for tid in valid_ids:
    ix = np.flatnonzero(arrays['trial number'] == tid)
```

iii. The agent gives no explicit justification; caching each valid trial's indices would remove the clearest duplicate work.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It always builds `plot_info` copies for up to four trials per session even when `make_plot` is false, although they are then unused. It also collects provenance fields such as all native trial IDs and native neural lengths that the decoder does not use. Outside the converter, extensive audits/plots are validation artifacts, not downstream features.

ii.
```python
plot_info = []
...
if len(plot_info) < 4:
    plot_info.append((tid, pos.copy(), speed.copy(), (lick > 0).copy(), dist.copy(), out.copy()))
...
if make_plot:
    plot_processing(info, plot_info)
```

iii. These copies support optional diagnostic plotting and provenance, but in the full run without plotting the copied arrays are discarded after each session.
