# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files matching the pattern `sub-*/sub-*_behavior+ophys.nwb` under `/app/data` are found using `glob.glob`. Each file is opened with `h5py` (not `pynwb`). Behavioral time series are read from the HDF5 path `processing/behavior/BehavioralTimeSeries/`, and neural data from `processing/ophys/Deconvolved/plane0/data`.

ii.
```python
files = sorted(glob.glob(os.path.join(ROOT, 'sub-*', '*_behavior+ophys.nwb')))
...
with h5py.File(path, 'r') as f:
    def beh(name):
        return f[B + name + '/data'][:]
```

iii. From the trajectory, the agent inspected the NWB structure extensively, noted the directory layout with `sub-*` directories, and chose `h5py` for direct HDF5 access rather than `pynwb` for simplicity and speed.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from directory names by stripping the `sub-` prefix, then sorted numerically. A mapping from subject name to index is built.

ii.
```python
subjects = sorted({os.path.basename(os.path.dirname(p)).replace('sub-', '') for p in files},
                  key=lambda x: int(x[1:]) if x.startswith('m') and x[1:].isdigit() else x)
submap = {s: i for i, s in enumerate(subjects)}
```

iii. The agent identified 11 subjects matching the paper's count.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All files are processed, one session per file.

ii.
```python
files = sorted(glob.glob(os.path.join(ROOT, 'sub-*', '*_behavior+ophys.nwb')))
...
for k, path in enumerate(files):
    neu, inp, out, nc, nt = convert_session(path)
```

iii. The agent noted that session numbers are encoded in filenames as `ses-XX`.

## 1-d. How are the data split into trials?

i. Trials are segmented using the `trial_start` and `teleport` behavior time series. The `trial_pairs` function pairs each trial start with the first subsequent teleport onset, ensuring the teleport occurs before the next trial start. The teleport index is used as the exclusive end of the trial.

ii.
```python
starts = np.flatnonzero(beh('trial_start') > 0)
ends = np.flatnonzero(beh('teleport') > 0)
pairs = trial_pairs(starts, ends)
...
def trial_pairs(start, end):
    pairs = []
    j = 0
    for k, a in enumerate(start):
        while j < len(end) and end[j] <= a:
            j += 1
        if j == len(end):
            break
        b = int(end[j])
        next_a = int(start[k + 1]) if k + 1 < len(start) else np.iinfo(np.int64).max
        if b < next_a and b > a:
            pairs.append((int(a), b))
            j += 1
    return pairs
```

iii. The agent investigated trial segmentation carefully, noting that `trial_number` labels were not always cleanly synchronized with teleport/position resets, and chose to use `trial_start` and `teleport` signals directly. The `teleport > 0` condition finds teleport onset frames.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than 2 complete trials raise an error and are skipped. No per-trial minimum timepoint filter is applied.

ii.
```python
if len(pairs) < 2:
    raise ValueError(f'fewer than two complete trials: {path}')
```

iii. The agent reasoned that at least 2 trials are needed for decoder evaluation but did not implement a minimum trial length filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the NWB's `Deconvolved` field (Suite2p's deconvolved calcium activity), accessed at `processing/ophys/Deconvolved/plane0/data`. This is NOT the paper's own dF/F + deconvolution pipeline.

ii.
```python
dset = f['processing/ophys/Deconvolved/plane0/data']
...
deconv = dset[:, cell_idx]
```

iii. The agent's docstring says "Suite2p deconvolved activity is used at its native imaging rate" and the trajectory shows the agent inspected the NWB ophys processing modules and chose to use the pre-computed deconvolved data directly.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied. The pre-stored Suite2p deconvolved traces are used directly after subsetting to `iscell` ROIs.

ii.
```python
deconv = dset[:, cell_idx]
...
neural_trials.append(np.asarray(deconv[a:b, :], dtype=np.float32).T.copy())
```

iii. The agent chose to use the deposited deconvolved data as-is, rather than reimplementing the paper's dF/F + OASIS pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only Suite2p's `iscell` filter is applied. Putative interneurons are NOT filtered out.

ii.
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0] == 1
...
rois_path = 'processing/ophys/Deconvolved/plane0/rois'
if rois_path in f:
    roi_rows = np.asarray(f[rois_path][:], dtype=int)
else:
    roi_rows = np.asarray(f['processing/ophys/Fluorescence/plane0/rois'][:], dtype=int)
cell_idx = np.flatnonzero(iscell[roi_rows])
deconv = dset[:, cell_idx]
```

iii. The agent noted that Suite2p marks about half the ROIs as cells and applied this filter. The putative interneuron filter (speed correlation > 0.5) described in the paper's Methods is not implemented.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by slicing neural data between `trial_start` and `teleport` indices. No additional temporal shifting is needed since trial start IS the alignment event.

ii.
```python
neural_trials.append(np.asarray(deconv[a:b, :], dtype=np.float32).T.copy())
```

iii. The agent noted that behavior and neural data share the same frame indexing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native imaging rate of 15.5078125 Hz (DT_MS = 64.484 ms). No rebinning is applied. The rate is hardcoded.

ii.
```python
RATE = 15.5078125
DT_MS = 1000.0 / RATE
```

iii. The agent determined the imaging rate from inspecting NWB metadata and hardcoded it. For multi-plane sessions (m17, m18), only `plane0` is read, so the effective per-plane rate would be ~7.75 Hz, but this is not accounted for.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the frame count within each trial and the hardcoded imaging rate `RATE`. Does not use stored timestamps.

ii.
```python
inp[0] = np.arange(n, dtype=np.float32) / RATE
```

iii. The agent chose to compute time from the frame index divided by the known rate, rather than using stored timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Frame indices (0 to n-1) are divided by RATE to convert to seconds. This inherently starts at 0 for each trial.

ii.
```python
inp[0] = np.arange(n, dtype=np.float32) / RATE
```

iii. Simple index-to-time conversion.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both use the same frame indices within each trial (a:b slice), so they are inherently aligned.

ii.
```python
n = b - a
...
inp[0] = np.arange(n, dtype=np.float32) / RATE
```

iii. Same indexing ensures alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
environment = beh('environment')
...
ev = environment[a:b]
ev = ev[ev >= 0]
env = int(np.rint(np.median(ev))) if len(ev) else 0
```

iii. The agent identified the `environment` variable as encoding ENV1 vs ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, negative environment values are filtered out, and the median of remaining values is rounded to an integer. If no valid values exist, defaults to 0. The result is constant per trial.

ii.
```python
ev = environment[a:b]
ev = ev[ev >= 0]
env = int(np.rint(np.median(ev))) if len(ev) else 0
inp[1] = env
```

iii. The agent noted that environment can have -1 values during ITI and used median filtering to get a clean per-trial value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the loop index `i` over trial pairs within each session.

ii.
```python
for i, (a, b) in enumerate(pairs):
    ...
    inp[2] = i
```

iii. Sequential trial index within the session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. Constant across all timepoints within a trial.

ii.
```python
inp[2] = i
```

iii. Simple sequential index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and behavior timestamps. Reward events are checked for temporal overlap with each trial.

ii.
```python
reward_times = f[B + 'Reward/timestamps'][:]
rewarded = np.array([
    np.any((reward_times >= ts[a]) & (reward_times < ts[b]))
    for a, b in pairs
], dtype=np.int8)
```

iii. The agent checks whether any reward timestamp falls within each trial's time window.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A `rewarded` array is computed for all trials. Then for each trial, the previous trial's reward outcome is used. The first trial gets `prev=0`.

ii.
```python
prev = 0
for i, (a, b) in enumerate(pairs):
    ...
    inp[3] = prev
    ...
    prev = int(rewarded[i])
```

iii. The agent tracks previous reward outcome with a running variable `prev`, initialized to 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` and the inferred reward zone identity. Zone identity is determined from the `reward_zone` event stream: the median position during reward-zone-active frames is compared to the three nominal zone starts (80, 200, 320 cm), and the nearest zone is assigned. Missing zone trials are filled from the nearest observed trial.

ii.
```python
ZONE_STARTS = np.array([80.0, 200.0, 320.0], dtype=np.float32)
ZONE_WIDTH = 50.0
...
zones = np.full(len(pairs), -1, dtype=np.int8)
for i, (a, b) in enumerate(pairs):
    p = position[a:b][zone_event[a:b] > 0]
    p = p[np.isfinite(p) & (p >= 0) & (p <= 450)]
    if len(p):
        zones[i] = int(np.argmin(np.abs(ZONE_STARTS - np.median(p))))
zones = nearest_fill(zones)
```

iii. The agent used the median of the reward-zone-active positions rather than a Viterbi algorithm approach, with nearest-neighbor filling for omission trials.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative before the zone, zero inside the zone, positive after. Zone boundaries are `[z0, z0+50]` where `z0` is the zone start. Then discretized with `np.digitize`.

ii.
```python
z0 = float(ZONE_STARTS[zones[i]])
z1 = z0 + ZONE_WIDTH
dist = np.where(pos < z0, pos - z0, np.where(pos > z1, pos - z1, 0.0))
```

iii. Standard signed distance computation to zone boundaries.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using `np.digitize` with edges `[-50, -10, 0, nextafter(0,1), 10, 50]`. Without leading `-inf` and trailing `inf`, the default behavior of `np.digitize` creates 7 categories (0-6).

ii.
```python
out[0] = np.digitize(dist, [-50, -10, 0, np.nextafter(0.0, 1.0), 10, 50])
```

iii. The bin edges match the instruction specification. `np.nextafter(0.0, 1.0)` separates exactly 0 from positive values.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices (a:b) used for neural and behavioral data. No additional alignment needed.

ii.
```python
pos = position[a:b]
```

iii. Same indexing as neural data.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = beh('position')
...
pos = position[a:b]
```

iii. Direct access to position data.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting per-trial slice and discretizing.

ii.
```python
out[1] = np.digitize(pos, [90, 180, 270, 360])
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `np.digitize` with edges `[90, 180, 270, 360]`, yielding bins 0-4 corresponding to 90 cm wide bins spanning the 450 cm track.

ii.
```python
out[1] = np.digitize(pos, [90, 180, 270, 360])
```

iii. Matches the instruction specification for 5 equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data. No additional alignment needed.

ii.
```python
pos = position[a:b]
```

iii. Same indexing ensures alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = beh('lick')
```

iii. Direct access to lick data.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
out[3] = (lick[a:b] > 0).astype(np.int8)
```

iii. Instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data. No additional alignment needed.

ii.
```python
lick[a:b]
```

iii. Same indexing ensures alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` event stream and `position` behavior time series, as described in 7-a. The zone identity (0=A, 1=B, 2=C) is determined by nearest zone start using median position during reward-zone-active frames.

ii.
```python
zones[i] = int(np.argmin(np.abs(ZONE_STARTS - np.median(p))))
zones = nearest_fill(zones)
...
out[4] = zones[i]
```

iii. See 7-a.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. See 7-a. The zone index (0, 1, 2) is used directly as the categorical output value.

ii.
```python
out[4] = zones[i]
```

iii. 0=A, 1=B, 2=C mapping.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward` timestamps and behavior timestamps.

ii.
```python
reward_times = f[B + 'Reward/timestamps'][:]
rewarded = np.array([
    np.any((reward_times >= ts[a]) & (reward_times < ts[b]))
    for a, b in pairs
], dtype=np.int8)
```

iii. Checks if any reward event timestamp falls within each trial's time window.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, binary 1 if any reward timestamp is >= trial start time and < trial end time, else 0. Constant per trial.

ii.
```python
rewarded = np.array([
    np.any((reward_times >= ts[a]) & (reward_times < ts[b]))
    for a, b in pairs
], dtype=np.int8)
...
out[5] = rewarded[i]
```

iii. Uses timestamp comparison rather than index-based matching.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing reward zone data**: Trials where `reward_zone` is never active get zone -1, which is filled from nearest observed trial via `nearest_fill`.
- **Invalid positions**: Position values outside [0, 450] or non-finite are excluded when computing zone identity median.
- **Sessions with < 2 trials**: Raise ValueError (session skipped).
- **ROI mapping mismatches**: The code handles cases where `rois` DynamicTableRegion maps a subset of PlaneSegmentation rows.

ii.
```python
zones = nearest_fill(zones)
...
p = p[np.isfinite(p) & (p >= 0) & (p <= 450)]
...
if len(pairs) < 2:
    raise ValueError(...)
```

iii. Defensive handling found during data exploration, especially the ROI mapping issue for m17/m18 sessions.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with `h5py` and reading large neural data arrays
2. Saving the large pickle file

ii. N/A

iii. Using `h5py` instead of `pynwb` is faster for reading.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The reward detection loop uses a list comprehension with per-trial timestamp comparison that could be vectorized with `np.searchsorted`. The `nearest_fill` function loops over missing indices.

ii.
```python
rewarded = np.array([
    np.any((reward_times >= ts[a]) & (reward_times < ts[b]))
    for a, b in pairs
], dtype=np.int8)
```

iii. The per-trial reward check iterates over all reward times for each trial.

## 13-c. What processing does the code repeat multiple times?

i. The code does not repeat processing -- each NWB file is read once and all data extracted in a single pass.

ii. N/A

iii. The single-pass design avoids the overhead of a separate survey step.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads the `speed` variable and includes it as an output, which matches the instructions. No obviously unnecessary processing is performed.

ii. N/A

iii. N/A
