# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `h5py` to directly read HDF5/NWB files. It globs all `.nwb` files matching `sub-*/*.nwb` in the data directory, sorted by path. Each file is opened with `h5py.File()` and data is read from the HDF5 groups directly (e.g., `f['processing/behavior/BehavioralTimeSeries']`).

ii.
```python
files=sorted(DATA_ROOT.glob('sub-*/*.nwb'))
...
with h5py.File(path, 'r') as f:
    b = f['processing/behavior/BehavioralTimeSeries']
    trial = b['trial number/data'][:]
    ts = b['trial number/timestamps'][:]
    ...
```

iii. The AI chose `h5py` over `pynwb` for speed and directness, reading HDF5 paths directly. This approach loads all 152 NWB files across 11 subjects.

## 1-b. How are the data split into subjects?

i. Subjects are determined from the `general/subject/subject_id` field within each NWB file. Unique subjects are collected after processing all sessions and sorted numerically.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
...
subjects=sorted(set(session_subjects), key=lambda x:(int(x[1:]) if x[1:].isdigit() else x))
subject_idx=np.array([subjects.index(x) for x in session_subjects],dtype=np.int64)
```

iii. Rather than parsing directory names, the AI reads the subject ID from the NWB metadata itself, which is more robust.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are sorted by file path.

ii.
```python
files=sorted(DATA_ROOT.glob('sub-*/*.nwb'))
...
for si,path in enumerate(files):
    n,i,o,subject,summary=convert_session(path, ...)
```

iii. The one-file-per-session structure is inherent to the NWB data organization.

## 1-d. How are the data split into trials?

i. Trials are identified using the `trial number` variable from the behavior time series. Unique non-negative trial numbers are extracted, and each trial consists of contiguous frames with that trial number. The AI verifies trial numbers are contiguous (0..N-1) and that frames within each trial are contiguous.

ii.
```python
trial = b['trial number/data'][:]
trial_ids = np.unique(trial[trial >= 0]).astype(int)
if not np.array_equal(trial_ids, np.arange(len(trial_ids))):
    raise ValueError(f'{path.name}: non-contiguous trial numbers')
for tid in trial_ids:
    idx = np.flatnonzero(trial == tid)
    if not len(idx) or np.any(np.diff(idx) != 1):
        raise ValueError(f'{path.name}: trial {tid} is empty/noncontiguous')
    lo, hi = int(idx[0]), int(idx[-1])
    bounds.append((lo, hi + 1))
```

iii. The AI uses `trial number` as the primary segmentation method, noting it is more robust than `trial_start`/`teleport` event markers (which have one missing globally). Frames with `trial number == -1` (intertrial periods) are excluded.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if they lack a `trial_start` marker (incomplete recording-edge fragments). The AI also validates that each trial has contiguous frames and that trial numbers form a complete 0..N-1 sequence.

ii.
```python
has_start[tid] = bool(np.any(b['trial_start/data'][idx] > 0))
...
keep = has_start
if np.sum(~keep):
    print(f'  {path.name}: excluding {int(np.sum(~keep))} numbered fragment(s) without trial_start', flush=True)
trial_ids = trial_ids[keep]
bounds = [x for x, k in zip(bounds, keep) if k]
```

iii. Only 1 trial globally (in m11 ses-03) lacks a `trial_start` marker and is excluded. No minimum trial length filter is applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` data in the NWB ophys processing module. This is the pre-computed Suite2p deconvolved calcium events stored in the NWB file.

ii.
```python
neural_series = [f['processing/ophys/Deconvolved/'+name+'/data'] for name in info['plane_names']]
cell_indices = [np.flatnonzero(mask) for mask in info['plane_masks']]
...
plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, dtype=np.float32)
              for ds, idx in zip(neural_series, cell_indices)]
neural = np.concatenate(plane_data, axis=0)
```

iii. The AI's CONVERSION_NOTES.md states: "Paper population decoding uses deconvolved calcium events" and "Use provided Deconvolved activity, avoiding redundant dF/F recomputation." The AI decided the NWB Deconvolved field is adequate for decoding.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied to the neural data beyond: (1) selecting accepted cells via `iscell`, (2) transposing to (neurons, timepoints) format, (3) converting to float32, and (4) replacing any NaN/inf values with zero.

ii.
```python
plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, dtype=np.float32)
              for ds, idx in zip(neural_series, cell_indices)]
neural = np.concatenate(plane_data, axis=0)
if not np.all(np.isfinite(neural)):
    neural = np.nan_to_num(neural, copy=False)
```

iii. The AI uses the pre-computed Deconvolved signal rather than recomputing dF/F and deconvolution from raw Fluorescence and Neuropil traces as the reference paper's code does.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only Suite2p accepted cells (`iscell[:,0] > 0`) are retained. No putative interneuron filtering is applied.

ii.
```python
ps = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = ps['iscell'][:]
accepted = iscell[:, 0] > 0 if iscell.ndim == 2 else iscell > 0
...
cell_indices = [np.flatnonzero(mask) for mask in info['plane_masks']]
```

iii. The AI's CONVERSION_NOTES state: "Retain `iscell[:,0] == 1`; do not impose place-cell/RR/TR filters." The AI treats the interneuron filter as an analysis-specific choice rather than a general quality filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by indexing into the deconvolved activity using the trial boundaries derived from the `trial number` variable. Since behavior and neural data share the same frame indexing, no additional temporal alignment is needed.

ii.
```python
plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, dtype=np.float32)
              for ds, idx in zip(neural_series, cell_indices)]
```
where `lo, hi` are the frame boundaries of each trial.

iii. The behavior timestamps and neural data are already synchronized at the imaging frame rate.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is preserved at the native imaging frame rate: `1000.0 / 15.5078125 = 64.4836 ms`. No rebinning is applied.

ii.
```python
BIN_MS = 1000.0 / 15.5078125
...
metadata=dict(
    time_bin_size=float(BIN_MS), ...
)
```

iii. The AI notes that some sessions report 31 Hz metadata but behavior timestamps consistently show ~15.5 Hz spacing, so no resampling is done.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the behavior `timestamps` array (from `trial number/timestamps`).

ii.
```python
ts = b['trial number/timestamps'][:]
...
t_rel = (timestamps[lo:hi] - timestamps[lo]).astype(np.float32)
```

iii. Behavior timestamps are synchronized to imaging frames.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of each trial is subtracted from all timestamps within that trial.

ii.
```python
t_rel = (timestamps[lo:hi] - timestamps[lo]).astype(np.float32)
```

iii. Straightforward time-from-trial-start computation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame indexing, so the same `lo:hi` slice is used for both. No additional alignment is needed.

ii. Same `lo:hi` indices used for both neural and behavioral data.

iii. Verified by the synchronized frame structure of the NWB files.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
environment = b['environment/data'][:]
...
ev = environment[idx]
ev = ev[(ev == 0) | (ev == 1)]
envs[tid] = int(np.bincount(ev.astype(int), minlength=2).argmax())
```

iii. The environment variable is 0 or 1 within trials and -1 outside.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The modal (most common) valid environment value within each trial is used. Values that are not 0 or 1 are excluded before taking the mode. The result is broadcast as a constant across all timepoints of the trial.

ii.
```python
ev = environment[idx]
ev = ev[(ev == 0) | (ev == 1)]
if not len(ev):
    raise ValueError(f'{path.name}: no valid environment in trial {tid}')
envs[tid] = int(np.bincount(ev.astype(int), minlength=2).argmax())
...
np.full(n, info['environments'][trial_idx], np.float32)
```

iii. Taking the modal value is a robust way to handle any edge-case frames with invalid values.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the native `trial number` variable in the NWB behavior time series.

ii.
```python
trial_ids = np.unique(trial[trial >= 0]).astype(int)
...
np.full(n, tid, np.float32)
```

iii. Uses the original zero-based trial numbering from the data.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The native trial number (`tid`) is used directly, broadcast as a constant across all timepoints of the trial. After filtering out trials without `trial_start`, the original trial IDs are preserved (not re-indexed).

ii.
```python
np.full(n, tid, np.float32)
```

iii. The native trial numbering preserves the original sequential index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the sparse `Reward` timestamps and behavior timestamps. For each trial, whether any reward event occurred within the trial's timestamp interval determines the outcome.

ii.
```python
reward_ts = b['Reward/timestamps'][:]
...
outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
```

iii. The Reward time series has its own timestamps separate from the behavior frame rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The previous trial's outcome (rewarded=1, omitted=0) is used. For the first trial, it's set to 0. The "previous" outcome is computed from the native trial sequence before filtering, so even when a trial is filtered out, the previous-trial outcome for the next trial still references the correct native-sequence predecessor.

ii.
```python
previous = np.r_[0, outcomes[:-1]].astype(np.int8)
keep = has_start
...
previous = previous[keep]
...
np.full(n, previous, np.float32)
```

iii. The previous outcome is computed before trial filtering to maintain correct sequencing.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the inferred reward zone location. The reward zone is identified by finding the first frame where `reward_zone > 0` within each trial and matching the position at that frame to the nearest canonical zone start (80, 200, or 320 cm).

ii.
```python
ZONE_STARTS = np.array([80.0, 200.0, 320.0], dtype=np.float32)
ZONE_WIDTH = 50.0
...
event_idx = idx[zone_event[idx] > 0]
if len(event_idx):
    entry_pos = position[event_idx[0]]
    zones[tid] = int(np.argmin(np.abs(ZONE_STARTS-entry_pos)))
```

iii. The three canonical zone starts come from the paper's definitions. Trials without a reward zone event get NaN, which is filled by nearest-known-trial propagation.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to the reward zone interval [zone_start, zone_start+50]. Negative before the zone, zero inside, positive after.

ii.
```python
def discretize_distance(position, zone_start):
    p = np.asarray(position)
    distance = np.where(p < zone_start, p-zone_start,
                        np.where(p > zone_start+ZONE_WIDTH,
                                 p-(zone_start+ZONE_WIDTH), 0.0))
```

iii. This matches the standard signed-distance-to-interval computation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using conditional assignment:
- 0: < -50 cm
- 1: -50 to < -10 cm
- 2: -10 to < 0 cm
- 3: exactly 0 (inside zone)
- 4: > 0 to 10 cm
- 5: > 10 to 50 cm
- 6: > 50 cm

ii.
```python
out = np.empty(p.shape, dtype=np.int8)
out[distance < -50] = 0
out[(distance >= -50) & (distance < -10)] = 1
out[(distance >= -10) & (distance < 0)] = 2
out[distance == 0] = 3
out[(distance > 0) & (distance <= 10)] = 4
out[(distance > 10) & (distance <= 50)] = 5
out[distance > 50] = 6
```

iii. Follows the bin specifications from the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indexing as neural data — `position[lo:hi]` uses the same trial boundaries.

ii.
```python
zone_start = float(ZONE_STARTS[info['zones'][trial_idx]])
dist_cls, signed_dist = discretize_distance(position[lo:hi], zone_start)
```

iii. Neural and behavior data share the same frame structure.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = b['position/data'][:]
...
discretize_position(position[lo:hi])
```

iii. The position variable records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
def discretize_position(position):
    p = np.asarray(position)
    return np.select([p < 90, p < 180, p < 270, p <= 360],
                     [0, 1, 2, 3], default=4).astype(np.int8)
```

iii. Raw position values used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins: < 90, [90,180), [180,270), [270,360], > 360 cm.

ii.
```python
return np.select([p < 90, p < 180, p < 270, p <= 360],
                 [0, 1, 2, 3], default=4).astype(np.int8)
```

iii. Five equal 90 cm bins spanning the 450 cm track per the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices — `position[lo:hi]`.

ii. Same indexing as neural data.

iii. Verified by the synchronized frame structure.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = b['lick/data'][:]
...
(lick[lo:hi] > 0).astype(np.int8)
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value mapped to 1, otherwise 0.

ii.
```python
(lick[lo:hi] > 0).astype(np.int8)
```

iii. The instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices — `lick[lo:hi]`.

ii. Same indexing as neural data.

iii. Verified by the synchronized frame structure.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` and `position` behavior time series. The first frame where `reward_zone > 0` in each trial gives the position at which the animal entered the reward zone, and this is matched to the nearest canonical zone start.

ii.
```python
event_idx = idx[zone_event[idx] > 0]
if len(event_idx):
    entry_pos = position[event_idx[0]]
    zones[tid] = int(np.argmin(np.abs(ZONE_STARTS-entry_pos)))
```

iii. For trials without a reward zone event, nearest-known-trial propagation (`nearest_fill`) is used. The AI validates that at most one zone switch occurs per session.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The reward zone is encoded as 0=A, 1=B, 2=C based on the index of the nearest canonical zone start (80, 200, 320 cm). Missing values are filled using nearest-trial propagation. The value is constant across all timepoints in a trial.

ii.
```python
zones = nearest_fill(zones)
...
np.full(n, info['zones'][trial_idx], np.int8)
```

iii. The AI validates that no session has more than one zone switch.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the sparse `Reward` timestamps and behavior timestamps.

ii.
```python
reward_ts = b['Reward/timestamps'][:]
outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
```

iii. The Reward time series has its own timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, if any reward timestamp falls within the trial's timestamp interval, the outcome is 1 (rewarded), otherwise 0. The value is constant across all timepoints in the trial.

ii.
```python
outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
...
np.full(n, info['outcomes'][trial_idx], np.int8)
```

iii. Binary reward outcome per trial.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: The common minimum length across all neural planes and behavior arrays is used (`n_common`).
- **Missing trial_start**: Trials without a `trial_start` marker are excluded (1 trial globally).
- **Missing reward zone data**: Trials without a reward zone event get NaN, filled by nearest-known-trial propagation.
- **Non-finite neural values**: NaN/inf in neural data is replaced with zero via `np.nan_to_num`.

ii.
```python
n_common = min([len(position)] + [ds.shape[0] for ds in neural_series])
...
if not np.all(np.isfinite(neural)):
    neural = np.nan_to_num(neural, copy=False)
```

iii. These are defensive checks found during data exploration.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is reading the neural data from HDF5 files. The AI optimizes this by reading one contiguous slice per plane per trial and selecting accepted cells via index arrays. Total conversion time for all 152 sessions is ~30-40 seconds based on the output logs.

ii. N/A

iii. The per-session timing shows most sessions process in 0.1-0.5 seconds.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial to build neural/input/output arrays. Within each trial, operations are already vectorized (e.g., `discretize_distance`, `discretize_position`). The trial-level metadata computation in `inspect_session` also loops over trials.

ii. N/A

iii. Variable trial lengths make full vectorization difficult without padding.

## 13-c. What processing does the code repeat multiple times?

i. Each session is opened twice: once in `inspect_session` (to gather metadata, trial bounds, reward zones) and once in `convert_session` (to read neural data and construct arrays). Behavior arrays like position, speed, and lick are read in both passes.

ii.
```python
def convert_session(path, make_plot=False):
    info = inspect_session(path)  # first open
    ...
    with h5py.File(path, 'r') as f:  # second open
```

iii. The two-pass approach separates metadata extraction from data conversion for clarity.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `signed_dist` (continuous float distance) alongside the discretized output, but this is only used for plotting when `--show-processing` is enabled. In normal mode, it is discarded. The `summary` dict is stored in metadata but not used by the decoder.

ii.
```python
dist_cls, signed_dist = discretize_distance(position[lo:hi], zone_start)
# signed_dist only used in plot_data
```

iii. Minor overhead; the continuous distance is cheap to compute.
