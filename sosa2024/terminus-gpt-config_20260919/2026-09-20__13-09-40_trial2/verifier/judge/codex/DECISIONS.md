# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed every `/app/data/sub-*/*.nwb` file, sorted the paths, and treated each file as one session. It used `h5py` to read behavior and ophys arrays directly; full mode is the default and sample mode limits processing to two files.

ii.
```python
files=sorted(DATA_ROOT.glob('sub-*/*.nwb'))
if args.sample: files=files[:2]
for si,path in enumerate(files):
    n,i,o,subject,summary=convert_session(path, args.show_processing and si<2)
```

iii. The notes say this covers 11 subjects and 152 sessions and that the NWBs are supplied synchronized exports, so raw Scanbox/VR reconstruction is unnecessary. Full verification reported all 152 sessions.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB's `general/subject/subject_id`; unique IDs are naturally sorted, and every session receives an index into that list.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
subjects=sorted(set(session_subjects), key=lambda x:(int(x[1:]) if x[1:].isdigit() else x))
subject_idx=np.array([subjects.index(x) for x in session_subjects],dtype=np.int64)
```

iii. The agent validated that the resulting 11 subjects match the complete export cohort.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; session-level neural, input, and output trial lists are appended in sorted file order.

ii.
```python
for si,path in enumerate(files):
    n,i,o,subject,summary=convert_session(path, ...)
    neural.append(n); inputs.append(i); outputs.append(o)
```

iii. The notes identify one synchronized session per recording and report 152 converted sessions.

## 1-d. How are the data split into trials?

i. Trials are contiguous samples sharing a nonnegative native `trial number`. IDs must be contiguous from zero. Intertrial samples (`-1`) are excluded, and a numbered fragment without a `trial_start` marker is removed.

ii.
```python
trial_ids = np.unique(trial[trial >= 0]).astype(int)
idx = np.flatnonzero(trial == tid)
lo, hi = int(idx[0]), int(idx[-1])
bounds.append((lo, hi + 1))
has_start[tid] = bool(np.any(b['trial_start/data'][idx] > 0))
keep = has_start
```

iii. The agent found native trial numbers more robust than teleport markers: all but one of 12,217 numbered IDs had a start marker. The excluded three-frame edge fragment was incomplete and moved backward over only part of the track.

## 1-e. How are trials filtered based on quality controls?

i. A trial is kept if it is a nonnegative, nonempty, contiguous native trial and contains a `trial_start`. No filtering by speed, outcome, or paper-specific scientific subsets is applied.

ii.
```python
if not len(idx) or np.any(np.diff(idx) != 1):
    raise ValueError(...)
keep = has_start
trial_ids = trial_ids[keep]
```

iii. The notes argue that speed and rewarded/omission selections are analysis-specific, while the decoder explicitly needs low-speed and both outcome classes. Only the unique incomplete recording-edge fragment is discarded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from every plane under `processing/ophys/Deconvolved`, with ROI metadata from `ImageSegmentation/PlaneSegmentation` (`iscell` and `planeIdx`). It is not derived from `Fluorescence` and `Neuropil` in this converter.

ii.
```python
plane_names = sorted(f['processing/ophys/Deconvolved'].keys(), ...)
neural_series = [f['processing/ophys/Deconvolved/'+name+'/data']
                 for name in info['plane_names']]
iscell = ps['iscell'][:]
```

iii. The agent interpreted the NWB as a finalized processed export and asserted that the stored deconvolved events were the signal used by the paper's decoder, making recomputation redundant.

## 2-b. How is the `neural` data processed?

i. For each plane and trial, the code reads one contiguous slice, selects accepted cells, transposes to neuron-by-time, casts to `float32`, concatenates planes, and replaces nonfinite values with zero. It performs no dF/F calculation, smoothing, or OASIS deconvolution itself.

ii.
```python
plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, dtype=np.float32)
              for ds, idx in zip(neural_series, cell_indices)]
neural = np.concatenate(plane_data, axis=0)
if not np.all(np.isfinite(neural)):
    neural = np.nan_to_num(neural, copy=False)
```

iii. The notes justify preserving the synchronized exported deconvolution and avoiding redundant processing; they also cite exact raw-versus-converted spot checks for one- and two-plane sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only Suite2p-accepted ROIs (`iscell > 0`) are retained, separately by plane. No putative-interneuron, place-cell, reward-relative-cell, or track-relative-cell filter is applied.

ii.
```python
accepted = iscell[:, 0] > 0 if iscell.ndim == 2 else iscell > 0
mask = accepted[plane_idx == plane_num]
cell_indices = [np.flatnonzero(mask) for mask in info['plane_masks']]
```

iii. The agent regarded `iscell` as the recording-quality criterion and other masks as scientific subpopulation definitions that could bias a general decoder. It reported exactly 138,678 accepted cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial slices begin at the first sample with the native numbered-trial ID, which the agent describes as corridor entry/trial start. Neural and behavior are indexed with the same `[lo:hi]` row range.

ii.
```python
lo, hi = int(idx[0]), int(idx[-1])
plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, ...)]
t_rel = (timestamps[lo:hi] - timestamps[lo]).astype(np.float32)
```

iii. The notes state that NWB behavior and imaging rows are synchronized one-to-one; direct spot checks confirmed alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code preserves every row without temporal rebinning and records a 64.4836 ms bin (`15.5078125 Hz`).

ii.
```python
BIN_MS = 1000.0 / 15.5078125
metadata=dict(time_bin_size=float(BIN_MS), ...)
```

iii. The agent found that even nominal 31 Hz two-plane series have rows aligned to behavior timestamps at 15.5078125 Hz; downsampling would discard aligned samples.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/trial number/timestamps` and the trial's native index bounds.

ii.
```python
ts = b['trial number/timestamps'][:]
t_rel = (timestamps[lo:hi] - timestamps[lo]).astype(np.float32)
```

iii. The agent notes that these timestamps explicitly encode the synchronized imaging-frame times.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in each retained trial is subtracted from every timestamp in that trial, then values are cast to `float32`.

ii.
```python
t_rel = (timestamps[lo:hi] - timestamps[lo]).astype(np.float32)
```

iii. This makes every trial begin at zero; the global invariant check confirmed monotonic trial-relative times beginning at zero.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the identical `[lo:hi]` slice as neural activity; `hi` is clipped to the common behavior/neural length.

ii.
```python
n_common = min([len(position)] + [ds.shape[0] for ds in neural_series])
hi = min(hi, n_common)
t_rel = timestamps[lo:hi] - timestamps[lo]
```

iii. The agent relied on rowwise NWB synchronization and validates equal temporal dimensions for neural, input, and output.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the behavior `environment/data` stream within each trial.

ii.
```python
environment = b['environment/data'][:]
ev = environment[idx]
```

iii. The notes identify codes 0 and 1 as ENV1 and ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Invalid values are removed, the modal 0/1 value is chosen, and it is broadcast over all trial timepoints.

ii.
```python
ev = ev[(ev == 0) | (ev == 1)]
envs[tid] = int(np.bincount(ev.astype(int), minlength=2).argmax())
np.full(n, info['environments'][trial_idx], np.float32)
```

iii. Environment should be constant per trial; taking its mode defensively yields the requested per-trial binary variable.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It uses the native behavior `trial number/data` ID rather than the position in the retained-trial list.

ii.
```python
trial = b['trial number/data'][:]
tid = int(info['trial_ids'][trial_idx])
```

iii. The notes emphasize preserving source numbering after filtering the incomplete fragment.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The native ID is cast to a float and broadcast across every sample of the trial.

ii.
```python
np.full(n, tid, np.float32)
```

iii. No renumbering is performed, so the variable remains the zero-based within-session trial identity.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from sparse `Reward/timestamps`, the behavior timestamps, and the native previous trial's bounds.

ii.
```python
reward_ts = b['Reward/timestamps'][:]
outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
previous = np.r_[0, outcomes[:-1]].astype(np.int8)
```

iii. The agent treats any reward event within a trial's frame-time interval as rewarded and preserves native adjacency.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Binary current outcomes are shifted by one native trial; trial 0 gets 0. The shifted value is then broadcast over the current trial.

ii.
```python
previous = np.r_[0, outcomes[:-1]].astype(np.int8)
np.full(n, previous, np.float32)
```

iii. This directly implements omitted=0/rewarded=1, including for a retained trial following a filtered native fragment.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses behavior `position/data` plus a per-trial reward-zone identity inferred from the first positive `reward_zone/data` sample's position and canonical starts 80, 200, and 320 cm.

ii.
```python
event_idx = idx[zone_event[idx] > 0]
entry_pos = position[event_idx[0]]
zones[tid] = int(np.argmin(np.abs(ZONE_STARTS-entry_pos)))
zone_start = float(ZONE_STARTS[info['zones'][trial_idx]])
```

iii. The agent found observed entries clustered at the three canonical starts and validated that filled labels form one stable block or one switch per session.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed Euclidean distance is computed to the closed 50 cm zone interval: negative before it, zero inside it, and positive after it; the result is then categorized.

ii.
```python
distance = np.where(p < zone_start, p-zone_start,
                    np.where(p > zone_start+ZONE_WIDTH,
                             p-(zone_start+ZONE_WIDTH), 0.0))
```

iii. The notes argue that “distance to any location in the reward zone” requires zero throughout the zone rather than circular distance to its start.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks implement the seven required categories and their boundary inequalities.

ii.
```python
out[distance < -50] = 0
out[(distance >= -50) & (distance < -10)] = 1
out[(distance >= -10) & (distance < 0)] = 2
out[distance == 0] = 3
out[(distance > 0) & (distance <= 10)] = 4
out[(distance > 10) & (distance <= 50)] = 5
out[distance > 50] = 6
```

iii. Boundary tests were added, and the notes say every class appears in the full result.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with the exact trial bounds used for neural data, after clipping to the common length.

ii.
```python
dist_cls, signed_dist = discretize_distance(position[lo:hi], zone_start)
```

iii. Rowwise synchronization and shape assertions provide the alignment check.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from behavior `position/data`.

ii.
```python
position = b['position/data'][:]
discretize_position(position[lo:hi])
```

iii. Numbered in-trial samples naturally exclude intertrial sentinel positions.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. There is no smoothing or resampling; each raw in-trial position sample is categorized into a 90 cm track bin.

ii.
```python
return np.select([p < 90, p < 180, p < 270, p <= 360],
                 [0, 1, 2, 3], default=4).astype(np.int8)
```

iii. Five equal 90 cm divisions cover the specified 450 cm corridor.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Values `<90`, `[90,180)`, `[180,270)`, `[270,360]`, and `>360` map to classes 0–4.

ii.
```python
np.select([p < 90, p < 180, p < 270, p <= 360],
          [0, 1, 2, 3], default=4)
```

iii. The masks reproduce the inequalities in the task, including 360 in class 3.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural activity use the same `[lo:hi]` rows.

ii.
```python
discretize_position(position[lo:hi])
```

iii. Independent raw-versus-converted checks and temporal-shape validation passed.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from behavior `lick/data`.

ii.
```python
lick = b['lick/data'][:]
```

iii. The agent interpreted this stream as within-frame lick counts, not a globally cumulative counter.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive raw count becomes 1 and all other values become 0.

ii.
```python
(lick[lo:hi] > 0).astype(np.int8)
```

iii. This matches the requested binary no/yes output; raw checks confirmed the transformation.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Licks use the identical trial slice as neural activity.

ii.
```python
(lick[lo:hi] > 0).astype(np.int8)
```

iii. The converter asserts equal neural/input/output time dimensions.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from `reward_zone/data` and `position/data`: the first positive zone event is assigned to the nearest canonical start.

ii.
```python
event_idx = idx[zone_event[idx] > 0]
zones[tid] = int(np.argmin(np.abs(ZONE_STARTS-position[event_idx[0]])))
```

iii. Observations clustered near 80/200/320 cm, corresponding to A/B/C.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Missing trial labels are filled from the nearest trial with an observed zone; the inferred class is broadcast across the trial. Sessions with no observations or more than one inferred switch are rejected.

ii.
```python
nearest = np.argmin(np.abs(missing[:, None] - known[None, :]), axis=1)
labels[missing] = labels[known[nearest]]
np.full(n, info['zones'][trial_idx], np.int8)
```

iii. The notes say nearest-known propagation handles omission trials and yielded the expected one- or two-block session structure and near-balanced A/B/C counts.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It comes from sparse behavior `Reward/timestamps` compared with the `trial number/timestamps` interval for each native trial.

ii.
```python
reward_ts = b['Reward/timestamps'][:]
outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
```

iii. The sparse stream represents reward deliveries; multiple deliveries intentionally remain one binary rewarded label.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if any reward timestamp lies inclusively between its first and last behavior-frame timestamps, otherwise 0; the result is broadcast over the trial.

ii.
```python
outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
np.full(n, info['outcomes'][trial_idx], np.int8)
```

iii. Direct checks of rewarded, omitted, and transition trials matched the raw sparse events.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code rejects noncontiguous trials, invalid environments, missing session-wide zone observations, mask/data mismatches, and multiple inferred zone switches. It fills per-trial missing zone labels by nearest known trial, excludes the unique numbered fragment without a start, clips trials to the common behavior/neural length, and converts nonfinite neural samples to zero.

ii.
```python
zones = nearest_fill(zones)
keep = has_start
n_common = min([len(position)] + [ds.shape[0] for ds in neural_series])
hi = min(hi, n_common)
neural = np.nan_to_num(neural, copy=False)
```

iii. The notes document one incomplete three-frame fragment, ten one-sample neural/behavior discrepancies, and an initially missed two-plane layout. The latter was fixed using `planeIdx`; all final validation and raw spot checks passed.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large deconvolved HDF5 trial slices and serializing the 13.33 GB pickle dominate. Full conversion processed all sessions in 52.57 seconds; downstream decoder training was much longer but is outside this converter.

ii.
```python
plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, dtype=np.float32) ...]
with out.open('wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes identify excess HDF5 I/O and loading rejected ROIs as likely costs and report per-session timing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-metadata loop could partly be vectorized for outcomes, environment modes, start flags, and reward-zone entries. The conversion loop is harder to eliminate because trials have variable lengths and separate arrays are required, though full-session transforms could be computed before slicing.

ii.
```python
for tid in trial_ids:
    idx = np.flatnonzero(trial == tid)
...
for trial_idx, (lo, hi) in enumerate(info['bounds']):
```

iii. The agent instead vectorized within-trial categorization and used contiguous reads, which kept the complete run under a minute.

## 13-c. What processing does the code repeat multiple times?

i. Each NWB is opened twice: `inspect_session` reads metadata/behavior, then `convert_session` reopens it and rereads position plus neural/behavior streams. Position is read in both passes, and trial slicing repeatedly invokes categorization.

ii.
```python
info = inspect_session(path)
with h5py.File(path, 'r') as f:
    ...
```

iii. The notes emphasize memory-efficient small-array inspection and contiguous trial reads; the repeat I/O is a tradeoff rather than a claimed scientific requirement.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes continuous `signed_dist` solely for optional plotting, collects detailed session summaries mainly for metadata, and (when enabled) generates diagnostic plots not consumed by decoder training. `inspect_session` also reads some arrays again during conversion.

ii.
```python
dist_cls, signed_dist = discretize_distance(position[lo:hi], zone_start)
summary=dict(file=path.name, ...)
if make_plot ...: plot_data=(..., signed_dist, out)
```

iii. These diagnostics supported visual validation and provenance. In normal full conversion plotting is off, but continuous distance is still returned and then discarded for every trial.
