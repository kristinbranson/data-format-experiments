# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script scans `/app/data/sub-*/*.nwb`, treats every matching NWB file as one session, and opens files directly with `h5py` instead of `pynwb`. It does not preload the whole dataset into memory. For each session it first reads small behavior arrays and metadata in `inspect_session()`, then reopens the file in `convert_session()` to read per-trial behavior arrays and contiguous trial slices from `processing/ophys/Deconvolved`.

ii.
```python
DATA_ROOT = Path('/app/data')
...
files=sorted(DATA_ROOT.glob('sub-*/*.nwb'))
...
def inspect_session(path):
    with h5py.File(path, 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        trial = b['trial number/data'][:]
        ts = b['trial number/timestamps'][:]
        position = b['position/data'][:]
        zone_event = b['reward_zone/data'][:]
        environment = b['environment/data'][:]
        reward_ts = b['Reward/timestamps'][:]
...
def convert_session(path, make_plot=False):
    info = inspect_session(path)
    with h5py.File(path, 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        neural_series = [f['processing/ophys/Deconvolved/'+name+'/data'] for name in info['plane_names']]
```

iii. In `CONVERSION_NOTES.md` Step 2 and Step 6, the AI justified this as an efficient way to process 152 large NWB files while keeping memory use low: read small metadata first, then use contiguous per-trial HDF5 reads and accepted-cell selection instead of loading all ROI matrices at once.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the `general/subject/subject_id` field inside each NWB file. After converting all sessions, the AI takes the sorted unique subject IDs to form `data['subjects']`, and `subject_idx` maps each session to that subject list.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
...
n,i,o,subject,summary=convert_session(path, args.show_processing and si<2)
neural.append(n); inputs.append(i); outputs.append(o); session_subjects.append(subject); summaries.append(summary)
subjects=sorted(set(session_subjects), key=lambda x:(int(x[1:]) if x[1:].isdigit() else x))
subject_idx=np.array([subjects.index(x) for x in session_subjects],dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` Step 2 records that the dataset contains 11 subjects and that all imaging is CA1. The AI used the embedded subject ID rather than directory parsing so that the session-to-subject mapping came from the file metadata itself.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as a separate session. The outer list entries of `neural`, `input`, `output`, and `brain_region_idx` correspond one-to-one with the sorted file list.

ii.
```python
files=sorted(DATA_ROOT.glob('sub-*/*.nwb'))
...
for si,path in enumerate(files):
    n,i,o,subject,summary=convert_session(path, args.show_processing and si<2)
    neural.append(n); inputs.append(i); outputs.append(o); session_subjects.append(subject); summaries.append(summary)
```

iii. The AI’s notes in Step 2 describe the dataset as 152 NWB files organized under subject folders, with session-level statistics computed per file. That matches the script’s one-file-per-session organization.

## 1-d. How are the data split into trials?

i. Trials are defined from contiguous runs of nonnegative values in the behavior stream `trial number/data`. For each trial ID, the AI finds all matching indices, requires them to be contiguous, and records `(lo, hi + 1)` as the trial bounds. It later excludes numbered fragments that do not contain any positive `trial_start` marker.

ii.
```python
trial = b['trial number/data'][:]
trial_ids = np.unique(trial[trial >= 0]).astype(int)
...
for tid in trial_ids:
    idx = np.flatnonzero(trial == tid)
    if not len(idx) or np.any(np.diff(idx) != 1):
        raise ValueError(f'{path.name}: trial {tid} is empty/noncontiguous')
    lo, hi = int(idx[0]), int(idx[-1])
    bounds.append((lo, hi + 1))
    has_start[tid] = bool(np.any(b['trial_start/data'][idx] > 0))
...
keep = has_start
trial_ids = trial_ids[keep]
bounds = [x for x, k in zip(bounds, keep) if k]
```

iii. The justification appears in `CONVERSION_NOTES.md` Step 4 and trajectory step 44. The AI concluded that `trial number` was the most robust segmentation source, because there was one incomplete recording-edge numbered fragment and many valid trials lacked teleport markers, so it used trial-number segments but required a `trial_start` marker to keep only complete trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI performs only one explicit trial-quality filter: it drops numbered fragments that lack a `trial_start` marker. It does not impose a minimum trial length filter or any rewarded/unrewarded, environment, or speed-based trial filtering.

ii.
```python
has_start[tid] = bool(np.any(b['trial_start/data'][idx] > 0))
...
keep = has_start
if np.sum(~keep):
    print(f'  {path.name}: excluding {int(np.sum(~keep))} numbered fragment(s) without trial_start', flush=True)
trial_ids = trial_ids[keep]
bounds = [x for x, k in zip(bounds, keep) if k]
zones = zones[keep]; outcomes = outcomes[keep]; envs = envs[keep]; previous = previous[keep]
```

iii. In `CONVERSION_NOTES.md` Step 5 and trajectory step 44, the AI justified this as removing the unique incomplete trial fragment while “preserv[ing] every complete numbered trial.” The notes explicitly say paper-specific rewarded/omission and speed-based selections were not used as general trial curation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` matrices are derived from the NWB `processing/ophys/Deconvolved/<plane>/data` arrays, after subsetting to Suite2p-accepted cells using `PlaneSegmentation/iscell`. The AI does not derive neural activity from `Fluorescence` or `Neuropil`.

ii.
```python
ps = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = ps['iscell'][:]
accepted = iscell[:, 0] > 0 if iscell.ndim == 2 else iscell > 0
...
plane_names = sorted(f['processing/ophys/Deconvolved'].keys(), key=lambda x: int(x.replace('plane','')))
...
neural_series = [f['processing/ophys/Deconvolved/'+name+'/data'] for name in info['plane_names']]
cell_indices = [np.flatnonzero(mask) for mask in info['plane_masks']]
```

iii. `CONVERSION_NOTES.md` Step 4 states that the AI resolved the “neural signal” discrepancy by using the provided `Deconvolved` arrays because the decoder methods in the paper use deconvolved calcium events. Step 5 maps `processing/ophys/Deconvolved/plane0/data[:, iscell]` directly to `neural`.

## 2-b. How is the `neural` data processed?

i. Neural processing is minimal. For each trial and each imaging plane, the AI reads the deconvolved rows for that trial, subsets to accepted cells, transposes to neuron-by-time format, concatenates across planes, casts to `float32`, and replaces any non-finite values with zero. It does not recompute dF/F or deconvolution.

ii.
```python
plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, dtype=np.float32)
              for ds, idx in zip(neural_series, cell_indices)]
neural = np.concatenate(plane_data, axis=0)
if not np.all(np.isfinite(neural)):
    neural = np.nan_to_num(neural, copy=False)
```

iii. The justification comes from `CONVERSION_NOTES.md` Step 4: the AI believed the NWB files were “already synchronized, processed exports” and chose to “use provided Deconvolved activity, avoiding redundant dF/F recomputation.” Step 6 also emphasizes speed and memory savings from reading only selected deconvolved columns.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural quality-control filter is Suite2p’s accepted-cell mask `iscell[:,0] == 1`. No additional filtering for putative interneurons, place cells, or reward-relative/track-relative subpopulations is applied.

ii.
```python
iscell = ps['iscell'][:]
accepted = iscell[:, 0] > 0 if iscell.ndim == 2 else iscell > 0
...
mask = accepted[plane_idx == plane_num]
...
cell_indices = [np.flatnonzero(mask) for mask in info['plane_masks']]
```

iii. In `CONVERSION_NOTES.md` Step 3 through Step 5, the AI repeatedly argues that place-cell and other paper-analysis masks are scientific subpopulation definitions rather than general recording-quality filters, so it retained all accepted Suite2p cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned implicitly to the beginning of each retained numbered trial. The trial slice `(lo:hi)` is used both for the neural rows and for the time input, so trial time zero is the first sample of that trial.

ii.
```python
for trial_idx, (lo, hi) in enumerate(info['bounds']):
    ...
    plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, dtype=np.float32)
                  for ds, idx in zip(neural_series, cell_indices)]
    ...
    t_rel = (timestamps[lo:hi] - timestamps[lo]).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 identifies the temporal alignment event as “start of numbered corridor trial (entry to linear track).” The trajectory shows the AI considered trial starts the required alignment event and implemented that by slicing all streams on the same trial bounds.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use one sample per behavior-aligned imaging row at a fixed bin size of `1000 / 15.5078125` ms (about 64.48 ms). No temporal rebinning or resampling is applied.

ii.
```python
BIN_MS = 1000.0 / 15.5078125
...
data=dict(
  ...
  metadata=dict(
   ...
   time_bin_size=float(BIN_MS),
```

iii. `CONVERSION_NOTES.md` Step 4 says the AI investigated the 31.015625 Hz metadata and concluded it conflicted with the explicit behavior timestamps. Trajectory steps 25 and 26 show that it initially considered factor-2 downsampling, then reversed course after finding one-to-one rowwise alignment at ~15.5 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `processing/behavior/BehavioralTimeSeries/trial number/timestamps` array, sliced to each trial’s `(lo:hi)` bounds.

ii.
```python
ts = b['trial number/timestamps'][:]
...
timestamps = info['timestamps']
...
t_rel = (timestamps[lo:hi] - timestamps[lo]).astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI maps “behavior timestamps + trial sample indices” to `input[0]`, describing it as “seconds since first frame of numbered trial.”

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first timestamp of the trial is subtracted from every timestamp in that trial. The resulting value is a continuous time-varying input in seconds.

ii.
```python
t_rel = (timestamps[lo:hi] - timestamps[lo]).astype(np.float32)
...
inp = np.vstack((t_rel,
                 np.full(n, info['environments'][trial_idx], np.float32),
                 np.full(n, tid, np.float32),
                 np.full(n, previous, np.float32))).astype(np.float32)
```

iii. The AI’s Step 5 mapping explicitly states “seconds since first frame of numbered trial.” No additional smoothing or resampling is described in the notes or trajectory.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same trial slice `(lo:hi)` is used for timestamps and neural rows, so the time input and neural matrix have identical numbers of timepoints per trial. The AI also clips each trial’s upper bound to the common minimum length across behavior and neural arrays within that session.

ii.
```python
n_common = min([len(position)] + [ds.shape[0] for ds in neural_series])
for trial_idx, (lo, hi) in enumerate(info['bounds']):
    hi = min(hi, n_common)
    ...
    neural = np.concatenate(plane_data, axis=0)
    t_rel = (timestamps[lo:hi] - timestamps[lo]).astype(np.float32)
    ...
    if neural.shape[1] != inp.shape[1] or inp.shape[1] != out.shape[1]:
        raise ValueError(f'{path.name}: trial {tid} temporal shape mismatch')
```

iii. `CONVERSION_NOTES.md` Step 2 notes that ten sessions have one extra neural sample relative to behavior, and Step 4 says conversion should “use the common stream length.” The AI’s implementation follows that by truncating trial bounds with `n_common`.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavior stream `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
environment = b['environment/data'][:]
...
ev = environment[idx]
ev = ev[(ev == 0) | (ev == 1)]
...
inp = np.vstack((t_rel,
                 np.full(n, info['environments'][trial_idx], np.float32),
```

iii. `CONVERSION_NOTES.md` Step 2 says environment is encoded as 0/1 on trial samples and `-1` outside trials. Step 5 maps that raw variable to `input[1]`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the AI removes invalid values (`-1` outside trials), takes the majority value among valid 0/1 samples using `np.bincount(...).argmax()`, and then broadcasts that per-trial environment label across all timepoints of the trial.

ii.
```python
ev = environment[idx]
ev = ev[(ev == 0) | (ev == 1)]
if not len(ev):
    raise ValueError(f'{path.name}: no valid environment in trial {tid}')
envs[tid] = int(np.bincount(ev.astype(int), minlength=2).argmax())
...
np.full(n, info['environments'][trial_idx], np.float32),
```

iii. The justification is in `CONVERSION_NOTES.md` Step 2 and Step 5: the environment code is effectively constant within trials, and the AI wanted a stable per-trial label even in the presence of any stray invalid values.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the raw behavior stream `processing/behavior/BehavioralTimeSeries/trial number/data`. After filtering to retained trials, the AI uses the native trial ID `tid` for each trial.

ii.
```python
trial = b['trial number/data'][:]
trial_ids = np.unique(trial[trial >= 0]).astype(int)
...
for trial_idx, (lo, hi) in enumerate(info['bounds']):
    tid = int(info['trial_ids'][trial_idx])
```

iii. `CONVERSION_NOTES.md` Step 5 says the conversion preserves “native zero-based continuous trial index.” Trajectory step 44 further notes that after excluding the unique incomplete fragment, the AI still chose to “preserv[e] native trial numbers.”

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation beyond broadcasting the native trial ID across all timepoints in the trial.

ii.
```python
inp = np.vstack((t_rel,
                 np.full(n, info['environments'][trial_idx], np.float32),
                 np.full(n, tid, np.float32),
                 np.full(n, previous, np.float32))).astype(np.float32)
```

iii. The AI’s Step 5 mapping says the converted value should preserve the source numbering rather than reindexing converted trials, so the per-trial ID is copied directly into the time-varying input array.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from sparse reward delivery timestamps in `processing/behavior/BehavioralTimeSeries/Reward/timestamps`, together with the per-trial timestamp bounds from `trial number/timestamps`.

ii.
```python
ts = b['trial number/timestamps'][:]
reward_ts = b['Reward/timestamps'][:]
...
outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
...
previous = np.r_[0, outcomes[:-1]].astype(np.int8)
```

iii. `CONVERSION_NOTES.md` Step 4 says reward is event-based and “must be aligned from its sparse timestamps rather than treated as framewise data.” Step 5 maps “previous trial sparse reward outcome” to `input[3]`.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a binary reward outcome for every native trial: `1` if any reward timestamp falls within that trial’s timestamp interval, else `0`. It then shifts this trial-level vector by one with `np.r_[0, outcomes[:-1]]`, so the first trial gets `0` and each later trial gets the previous native trial’s outcome. That value is then broadcast across all timepoints of the retained trial.

ii.
```python
outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
...
previous = np.r_[0, outcomes[:-1]].astype(np.int8)
...
np.full(n, previous, np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly describes this as “Trial 0=0; otherwise prior trial binary outcome, broadcast.” Trajectory step 44 says the AI intentionally preserved “previous native-trial outcome” even after excluding the incomplete fragment.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from framewise `position/data` and from a per-trial reward-zone identity inferred from the `reward_zone/data` event stream. The AI takes the first positive `reward_zone` frame in a trial, reads the animal’s position there, assigns it to the nearest canonical start in `ZONE_STARTS = [80, 200, 320]`, and fills missing trials from the nearest observed trial.

ii.
```python
ZONE_STARTS = np.array([80.0, 200.0, 320.0], dtype=np.float32)
...
position = b['position/data'][:]
zone_event = b['reward_zone/data'][:]
...
event_idx = idx[zone_event[idx] > 0]
if len(event_idx):
    entry_pos = position[event_idx[0]]
    zones[tid] = int(np.argmin(np.abs(ZONE_STARTS-entry_pos)))
...
zones = nearest_fill(zones)
...
zone_start = float(ZONE_STARTS[info['zones'][trial_idx]])
dist_cls, signed_dist = discretize_distance(position[lo:hi], zone_start)
```

iii. The AI’s Step 4 notes say the `reward_zone` stream is not zone identity but a per-frame entry/count signal. Trajectory step 27 justifies the inference by saying first positive reward-zone frames localize the zone well, missing trials are mostly omissions, and nearest-label filling produces stable one-block or one-switch session structures.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. After inferring the reward-zone start for the trial, the AI computes signed distance to the closed interval `[zone_start, zone_start + 50]`: negative before the zone, zero inside it, and positive after it. It returns both the discrete class and the continuous signed distance, though only the class is kept in the final dataset.

ii.
```python
def discretize_distance(position, zone_start):
    p = np.asarray(position)
    distance = np.where(p < zone_start, p-zone_start,
                        np.where(p > zone_start+ZONE_WIDTH,
                                 p-(zone_start+ZONE_WIDTH), 0.0))
    out = np.empty(p.shape, dtype=np.int8)
    ...
    return out, distance.astype(np.float32)
...
dist_cls, signed_dist = discretize_distance(position[lo:hi], zone_start)
```

iii. `CONVERSION_NOTES.md` Step 4 and Step 5 justify this as matching the decoder task wording “distance to any location in the reward zone,” so class 3 should represent being anywhere inside the 50 cm reward interval, not just at one anchor point.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI thresholds the continuous signed distance manually into seven categories: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

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

iii. Trajectory step 30 says the AI explicitly checked the discretization boundaries and believed these comparisons matched the task specification. The same class names appear in `output_values` in the final dictionary.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The AI computes the distance classes on the same per-trial frame indices used to slice neural activity, so the distance output is framewise aligned to neural data with no extra interpolation.

ii.
```python
plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, dtype=np.float32)
              for ds, idx in zip(neural_series, cell_indices)]
...
dist_cls, signed_dist = discretize_distance(position[lo:hi], zone_start)
...
if neural.shape[1] != inp.shape[1] or inp.shape[1] != out.shape[1]:
    raise ValueError(f'{path.name}: trial {tid} temporal shape mismatch')
```

iii. `CONVERSION_NOTES.md` Step 5 says the conversion preserves one sample per behavior-aligned imaging row. Because both position and deconvolved activity are indexed with the same `lo:hi`, the alignment is by construction.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
position = b['position/data'][:]
...
discretize_position(position[lo:hi])
```

iii. `CONVERSION_NOTES.md` Step 2 and Step 5 identify `position` as the framewise animal location in cm and map it directly to the absolute-position output after discretization.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI extracts each trial’s position trace and discretizes it into five bins spanning the track. There is no smoothing, interpolation, or wrapping.

ii.
```python
def discretize_position(position):
    # <90, [90,180), [180,270), [270,360], >360
    p = np.asarray(position)
    return np.select([p < 90, p < 180, p < 270, p <= 360],
                     [0, 1, 2, 3], default=4).astype(np.int8)
...
discretize_position(position[lo:hi])
```

iii. The justification in Step 5 is that the task asked for five equal-sized bins over a 450 cm track, which the AI interpreted as simple 90 cm thresholding on the raw in-trial positions.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The AI uses five categories with boundaries at 90, 180, 270, and 360 cm. Its implementation assigns positions `<= 360` to class 3 and `> 360` to class 4.

ii.
```python
return np.select([p < 90, p < 180, p < 270, p <= 360],
                 [0, 1, 2, 3], default=4).astype(np.int8)
```

iii. The code comment and the final `output_values` list both show the AI intended the fourth bin to cover `270 to 360 cm` and the fifth to cover `> 360 cm`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position classes are computed from the same `(lo:hi)` trial slices that define the neural frames, so they are directly frame-aligned to the neural matrix.

ii.
```python
for trial_idx, (lo, hi) in enumerate(info['bounds']):
    ...
    neural = np.concatenate(plane_data, axis=0)
    ...
    out = np.vstack((dist_cls,
                     discretize_position(position[lo:hi]),
                     discretize_speed(speed[lo:hi]),
```

iii. The AI’s notes emphasize that the exported NWB streams are already synchronized at the imaging frame rate, so additional alignment logic was unnecessary.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick = b['lick/data'][:]
...
(lick[lo:hi] > 0).astype(np.int8)
```

iii. `CONVERSION_NOTES.md` Step 4 and trajectory step 27 say the AI inspected the lick stream and concluded it already represented within-frame counts rather than a cumulative counter across the session, so the framewise lick array could be thresholded directly.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes the lick counts per frame: values `> 0` become `1`, otherwise `0`.

ii.
```python
out = np.vstack((dist_cls,
                 discretize_position(position[lo:hi]),
                 discretize_speed(speed[lo:hi]),
                 (lick[lo:hi] > 0).astype(np.int8),
                 np.full(n, info['zones'][trial_idx], np.int8),
                 np.full(n, info['outcomes'][trial_idx], np.int8))).astype(np.int8)
```

iii. Trajectory step 27 states the explicit justification: binary lick should be `lick > 0`, not a temporal difference, because the stored values behave like per-frame counts.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The AI applies the `> 0` threshold to the exact same `lo:hi` trial slice used for neural data, so the binary lick output is framewise aligned to neural activity.

ii.
```python
plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, dtype=np.float32)
              for ds, idx in zip(neural_series, cell_indices)]
...
(lick[lo:hi] > 0).astype(np.int8)
```

iii. As in the other framewise outputs, alignment is justified in the notes by the one-row-per-frame synchronization between behavior and neural streams.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `reward_zone/data` and `position/data`. The AI uses the first positive reward-zone frame in a trial to locate the corresponding position, maps that position to the nearest canonical reward-zone start, and then fills missing trials from nearby observed trials.

ii.
```python
position = b['position/data'][:]
zone_event = b['reward_zone/data'][:]
...
event_idx = idx[zone_event[idx] > 0]
if len(event_idx):
    entry_pos = position[event_idx[0]]
    zones[tid] = int(np.argmin(np.abs(ZONE_STARTS-entry_pos)))
...
zones = nearest_fill(zones)
```

iii. `CONVERSION_NOTES.md` Step 4 says the raw `reward_zone` stream does not encode A/B/C directly, so location must be inferred. Trajectory step 27 says this first-entry-plus-nearest-fill approach yielded clean one-block or one-switch session structures.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Processing has three stages: infer observed trial labels from first reward-zone-entry position; fill missing labels with `nearest_fill`; and reject sessions with more than one inferred switch (`changes > 1`). The retained label is then broadcast across the whole trial.

ii.
```python
def nearest_fill(labels):
    labels = np.asarray(labels, dtype=np.float32).copy()
    known = np.flatnonzero(np.isfinite(labels))
    missing = np.flatnonzero(~np.isfinite(labels))
    ...
    labels[missing] = labels[known[nearest]]
    return labels.astype(np.int8)
...
zones = nearest_fill(zones)
changes = int(np.sum(np.diff(zones) != 0))
if changes > 1:
    raise ValueError(f'{path.name}: inferred {changes} reward-zone switches')
...
np.full(n, info['zones'][trial_idx], np.int8)
```

iii. The AI justifies this in Step 4 and trajectory step 27: omission trials often lack an explicit reward-zone entry, but the inferred labels form stable blocks, so nearest-trial propagation is enough without a more complex state model.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from sparse reward delivery timestamps in `Reward/timestamps`, together with per-trial behavior timestamps from `trial number/timestamps`.

ii.
```python
ts = b['trial number/timestamps'][:]
reward_ts = b['Reward/timestamps'][:]
...
outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
```

iii. `CONVERSION_NOTES.md` Step 4 records that reward is a sparse event series and must be aligned by timestamps. Step 5 maps that trial-level binary reward signal to `output[5]`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each native trial, the AI sets reward outcome to `1` if any reward timestamp falls between the first and last timestamps of that trial, else `0`. The per-trial label is then broadcast across all timepoints in the output array.

ii.
```python
outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
...
out = np.vstack((dist_cls,
                 discretize_position(position[lo:hi]),
                 discretize_speed(speed[lo:hi]),
                 (lick[lo:hi] > 0).astype(np.int8),
                 np.full(n, info['zones'][trial_idx], np.int8),
                 np.full(n, info['outcomes'][trial_idx], np.int8))).astype(np.int8)
```

iii. The AI’s Step 5 notes say: “Any event in trial frame-time interval -> 1, else 0. Multiple deliveries remain 1.” That matches the code’s `np.any(...)` logic.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses a mixture of exclusion, truncation, imputation, and assertions:
- It excludes numbered trial fragments that lack a `trial_start` marker.
- It truncates each trial’s upper bound to `n_common`, the common minimum length across behavior and neural arrays.
- It fills missing reward-zone labels from the nearest observed trial.
- It replaces non-finite neural values with zero.
- It raises hard errors for non-contiguous trial indices, missing environment labels, sessions with no observed reward-zone entries, ROI-mask/data mismatches, or more than one inferred reward-zone switch.

ii.
```python
if not len(idx) or np.any(np.diff(idx) != 1):
    raise ValueError(f'{path.name}: trial {tid} is empty/noncontiguous')
...
if not len(ev):
    raise ValueError(f'{path.name}: no valid environment in trial {tid}')
...
zones = nearest_fill(zones)
...
if changes > 1:
    raise ValueError(f'{path.name}: inferred {changes} reward-zone switches')
...
n_common = min([len(position)] + [ds.shape[0] for ds in neural_series])
...
hi = min(hi, n_common)
...
if not np.all(np.isfinite(neural)):
    neural = np.nan_to_num(neural, copy=False)
```

iii. The justifications come from `CONVERSION_NOTES.md` Step 2, Step 4, and trajectory step 44: the AI discovered one incomplete recording-edge trial fragment, occasional one-sample neural/behavior length mismatches, and many omission trials without explicit reward-zone entries, and it implemented these rules to preserve “every complete numbered trial” while keeping synchronization intact.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are HDF5/NWB I/O and per-trial neural extraction: opening each session twice, reading session-level behavior arrays in `inspect_session()`, then reading contiguous trial slices from each deconvolved plane in `convert_session()`. Writing the large pickle is also an expensive end step.

ii.
```python
def inspect_session(path):
    with h5py.File(path, 'r') as f:
        ...
def convert_session(path, make_plot=False):
    info = inspect_session(path)
    with h5py.File(path, 'r') as f:
        ...
        for trial_idx, (lo, hi) in enumerate(info['bounds']):
            ...
            plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, dtype=np.float32)
                          for ds, idx in zip(neural_series, cell_indices)]
...
with out.open('wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. `CONVERSION_NOTES.md` Step 6 and Step 7 emphasize that the code was written to make large-file I/O the main cost, and they report per-session timing and total runtime estimates for the full dataset.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loops are over trials in `inspect_session()` and `convert_session()`, plus the per-plane loop inside each trial read. Reward-zone inference, reward-outcome detection, and per-trial environment extraction all happen one trial at a time and could be further vectorized with precomputed boundaries or grouped indexing, though variable trial lengths make that awkward.

ii.
```python
for tid in trial_ids:
    idx = np.flatnonzero(trial == tid)
    ...
    outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
    ...
for trial_idx, (lo, hi) in enumerate(info['bounds']):
    ...
    plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, dtype=np.float32)
                  for ds, idx in zip(neural_series, cell_indices)]
```

iii. The notes in Step 6 say the AI already vectorized the discretization logic and prioritized contiguous HDF5 reads. It left the trial loops in place because the outputs are stored as variable-length per-trial matrices and labels.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats session opening and some metadata reads. Every session is opened once in `inspect_session()` and then again in `convert_session()`. Reward-zone inference, subject lookup, accepted-cell masking metadata, and behavior-array reads such as trial numbers, timestamps, position, and reward timestamps are therefore done twice per session.

ii.
```python
def convert_session(path, make_plot=False):
    info = inspect_session(path)
    ...
    with h5py.File(path, 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        neural_series = [f['processing/ophys/Deconvolved/'+name+'/data'] for name in info['plane_names']]
        position = b['position/data'][:]
        speed = b['speed/data'][:]
        lick = b['lick/data'][:]
```

iii. In Step 6, the AI explicitly notes memory as the main design constraint and prefers duplicated small metadata reads over holding large arrays in RAM. This is why it split inspection and conversion into two passes per session rather than building everything in one open-file pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes some values that are not saved for downstream decoding:
- `signed_dist` is always computed, but only used for optional plotting.
- `trial` and `plane_shapes` are returned from `inspect_session()` but not used in conversion.
- Plotting support and per-session summary metadata add work that is not needed by the decoder itself.

ii.
```python
dist_cls, signed_dist = discretize_distance(position[lo:hi], zone_start)
...
if make_plot and plot_data is None and trial_idx == min(4, len(info['bounds'])-1):
    plot_data=(tid, neural, t_rel, position[lo:hi], speed[lo:hi],
               lick[lo:hi], signed_dist, out)
...
return dict(trial=trial, timestamps=ts, trial_ids=trial_ids, bounds=bounds,
            zones=zones, outcomes=outcomes, previous_outcomes=previous, environments=envs,
            plane_names=plane_names, plane_masks=plane_masks, plane_shapes=plane_shapes,
            n_rois=len(accepted), n_cells=int(accepted.sum()), subject=subject)
```

iii. The AI’s notes frame these as pragmatic extras for debugging, sanity plots, and metadata reporting rather than part of the core decoder payload. They were kept because the task instructions emphasized validation and sanity checking.
