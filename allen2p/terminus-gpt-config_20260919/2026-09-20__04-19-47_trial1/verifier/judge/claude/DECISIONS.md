# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly using `h5py` rather than the Allen SDK's high-level API. It reads a project metadata CSV (`ophys_experiment_table.csv`) to discover all experiment IDs, then maps those IDs to `.nwb` files on disk. Each NWB file is opened with `h5py.File` and relevant datasets (event traces, timestamps, trials, running speed, eye tracking, stimulus presentations) are read directly from HDF5 paths.

ii.
```python
META = ROOT / 'project_metadata' / 'ophys_experiment_table.csv'
NWBDIR = ROOT / 'behavior_ophys_experiments'
...
meta = pd.read_csv(META)
fmap = {int(f.stem.rsplit('_',1)[1]): f for f in NWBDIR.glob('*.nwb')}
meta = meta[meta.ophys_experiment_id.isin(fmap)].sort_values('ophys_experiment_id')
...
for j, row in enumerate(meta.itertuples(index=False), 1):
    eid = int(row.ophys_experiment_id)
    result, reason = process_experiment(fmap[eid], row, ...)
```

iii. The AI justified this approach as more efficient than the SDK's high-level API, avoiding overhead from loading unnecessary data. Direct HDF5 access reads only the required datasets.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `mouse_id` column in the experiment metadata CSV. Each unique mouse_id becomes a subject entry.

ii.
```python
mouse = str(row.mouse_id)
if mouse not in subjects: subjects.append(mouse)
subject_idx.append(subjects.index(mouse))
```

iii. The `mouse_id` field uniquely identifies each animal in the Allen metadata.

## 1-c. How are the data split into sessions?

i. The AI treats each **individual experiment (imaging plane)** as a separate "session" in the output. This means multi-plane sessions are split into separate output sessions rather than being grouped together.

ii.
```python
for j, row in enumerate(meta.itertuples(index=False), 1):
    eid = int(row.ophys_experiment_id)
    result, reason = process_experiment(fmap[eid], row, ...)
    ...
    neural.append(nn); inputs.append(ii); outputs.append(oo)
```

iii. The AI justified this by noting that the paper's decoding analysis was performed per imaging plane, and that multi-plane sessions have staggered timestamps (up to ~70ms offset) that would make direct concatenation invalid.

## 1-d. How are the data split into trials?

i. Trials are defined from the `intervals/trials` table in the NWB file. Only trials where `go` or `catch` is True are included. For each eligible trial, a 30 Hz time grid is constructed from `start_time` to `stop_time`, giving variable-length trials.

ii.
```python
tr = h['intervals/trials']
elig = np.asarray(tr['go'][:], bool) | np.asarray(tr['catch'][:], bool)
trial_idx = np.flatnonzero(elig)
...
for ti in trial_idx:
    a = max(starts[ti], ots[0]); b = min(stops[ti], ots[-1] + DT)
    k0 = int(np.ceil((a - origin) * HZ - 1e-9))
    k1 = int(np.ceil((b - origin) * HZ - 1e-9))
    grid = origin + np.arange(k0, k1, dtype=np.float64) * DT
    if len(grid) < 2: continue
```

iii. The SDK trials table provides pre-computed trial metadata. Go and catch trials are the only categories required by the task instructions (aborted and auto-rewarded are excluded). The trial window spans start_time to stop_time (variable length).

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only `go | catch` (which excludes aborted and auto-rewarded). Trials where the time grid has fewer than 2 points are skipped. Trials whose boundaries fall entirely outside the ophys timestamp range are also excluded. Sessions missing pupil tracking data are excluded entirely (3 experiments).

ii.
```python
elig = np.asarray(tr['go'][:], bool) | np.asarray(tr['catch'][:], bool)
trial_idx = np.flatnonzero(elig)
trial_idx = trial_idx[(stops[trial_idx] >= ots[0]) & (starts[trial_idx] <= ots[-1])]
...
if len(grid) < 2: continue
...
if 'acquisition/EyeTracking/pupil_tracking/area' not in h:
    return None, 'missing pupil tracking'
```

iii. The filtering matches the task instructions to include go and catch but exclude aborted and auto-rewarded. The pupil exclusion was justified because the required decoder output includes pupil diameter, which cannot be constructed without data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the **detected calcium events** (`processing/ophys/event_detection/data`) rather than dF/F traces. Only neurons with `valid_roi == True` are included.

ii.
```python
evg = h['processing/ophys/event_detection']
ots = np.asarray(evg['timestamps'][:], dtype=np.float64)
events = np.asarray(evg['data'][:], dtype=np.float32)
cells = h['processing/ophys/image_segmentation/cell_specimen_table']
valid = np.asarray(cells['valid_roi'][:], bool) if 'valid_roi' in cells else np.ones(events.shape[1], bool)
events = events[:, valid]
```

iii. The AI chose detected calcium events because the paper states it "used detected calcium events for all neural analyses." The valid_roi filter was applied based on the SDK's cell segmentation quality control.

## 2-b. How is the `neural` data processed?

i. The detected event magnitudes are linearly interpolated from the native ophys timestamps onto a 30 Hz grid anchored to the experiment's first ophys timestamp. The interpolation is vectorized across all neurons simultaneously.

ii.
```python
def interp_rows(times, values, grid):
    j = np.searchsorted(times, grid, side='left')
    j = np.clip(j, 1, len(times)-1)
    lo, hi = j-1, j
    den = times[hi]-times[lo]
    a = np.divide(grid-times[lo], den, out=np.zeros_like(grid), where=den != 0)
    return values[lo] + (values[hi]-values[lo]) * a[:, None]
...
n = interp_rows(ots, events, grid).T.astype(np.float32, copy=False)
```

iii. The 30 Hz resampling was justified by the paper's statement about "linearly interpolating onto a common 30hz timeseries." The vectorized interpolation was chosen for efficiency.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by `valid_roi` from the cell specimen table. Only cells marked as valid ROIs are included. No additional amplitude or signal-quality filtering is applied.

ii.
```python
valid = np.asarray(cells['valid_roi'][:], bool) if 'valid_roi' in cells else np.ones(events.shape[1], bool)
events = events[:, valid]
```

iii. The AI relied on the Allen SDK's release QC, which already applies z-drift exclusion, residual-motion review, data-stream integrity checks, and temporal synchronization checks.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to the trial start (`start_time`). A 30 Hz grid is constructed from `start_time` to `stop_time`, anchored to the experiment's first ophys timestamp. The event data is linearly interpolated onto this grid.

ii.
```python
origin = ots[0]
for ti in trial_idx:
    a = max(starts[ti], ots[0])
    b = min(stops[ti], ots[-1] + DT)
    k0 = int(np.ceil((a - origin) * HZ - 1e-9))
    k1 = int(np.ceil((b - origin) * HZ - 1e-9))
    grid = origin + np.arange(k0, k1, dtype=np.float64) * DT
    n = interp_rows(ots, events, grid).T.astype(np.float32, copy=False)
```

iii. The grid is anchored to the first ophys timestamp so that the 30 Hz sampling points are consistent across all trials within an experiment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is resampled to a fixed 30 Hz grid (33.333 ms bins). This is a rebinning from the native ophys frame rate (~11 Hz for single-plane, variable for multi-scope) via linear interpolation.

ii.
```python
HZ = 30.0
DT = 1.0 / HZ
...
grid = origin + np.arange(k0, k1, dtype=np.float64) * DT
```
The metadata reports:
```python
'time_bin_size': 1000.0/HZ  # = 33.333 ms
```

iii. The AI cited the paper's statement about interpolating to a "common 30hz timeseries" as justification. This ensures a consistent bin size across all trials and sessions regardless of the native sampling rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the **stimulus presentations table** in the NWB file (discovered dynamically by looking for a table with `image_name`, `start_time`, `stop_time`, `is_change`, and `omitted` columns). The `image_name` field identifies which of 16 natural images is being shown.

ii.
```python
def stimulus_table(h):
    for name, g in h['intervals'].items():
        if all(c in g for c in ('image_name','start_time','stop_time','is_change','omitted')):
            return g
    raise KeyError('active natural-image stimulus presentation table not found')
...
sg = stimulus_table(h)
ss = np.asarray(sg['start_time'][:], float)
se = np.asarray(sg['stop_time'][:], float)
simg = np.array([dec(x) for x in sg['image_name'][:]], object)
```

iii. The stimulus presentations table provides the precise timing of each image flash, allowing image identity to be assigned only during the actual 250ms non-gray presentation periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image identity is assigned per time bin by overlapping stimulus presentations with the trial's 30 Hz grid. During the ~250ms image presentation, the image class (1-16) is assigned. During gray inter-stimulus intervals and omitted presentations, the value is 0 (gray). A global mapping of 16 sorted image names to codes 1-16 is used, with gray=0.

ii.
```python
IMAGE_NAMES = ['im000','im031','im035','im045','im054','im061','im062','im063',
               'im065','im066','im069','im073','im075','im077','im085','im106']
IMAGE_TO_CODE = {x: i+1 for i, x in enumerate(IMAGE_NAMES)}
...
image = np.zeros(len(grid), dtype=np.int16)  # 0 = gray
for pi in range(max(0,p0), min(len(ss),p1+1)):
    if not somit[pi] and simg[pi] in IMAGE_TO_CODE:
        mask = (grid >= ss[pi]) & (grid < se[pi])
        image[mask] = IMAGE_TO_CODE[simg[pi]]
```

iii. The AI interpreted "Image identity (of the image presented during the non-grey screen)" as requiring that image identity only be labeled during actual image presentations, with gray periods labeled as a separate category (0). This results in ~67% of time bins being gray.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 30 Hz grid as the neural data. Stimulus presentation start/stop times are compared against the grid timestamps to determine which time bins fall within each presentation.

ii.
```python
p0 = np.searchsorted(se, grid[0], side='right')
p1 = np.searchsorted(ss, grid[-1], side='right')
for pi in range(max(0,p0), min(len(ss),p1+1)):
    mask = (grid >= ss[pi]) & (grid < se[pi])
    image[mask] = IMAGE_TO_CODE[simg[pi]]
```

iii. By using the same `grid` for both neural interpolation and output assignment, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column of the stimulus presentations table, combined with the `start_time` of each presentation and the `omitted` flag.

ii.
```python
schange = np.asarray(sg['is_change'][:], float) == 1
somit = np.asarray(sg['omitted'][:], float) == 1
...
if schange[pi] and not somit[pi] and starts[ti] <= ss[pi] < stops[ti]:
    q = np.searchsorted(grid, ss[pi], side='left')
    if q < len(grid): change[q] = 1
```

iii. The `is_change` flag from the stimulus table identifies which presentations are image changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Image change is represented as a **single-bin pulse** (value 1 at one time point) at the first 30 Hz grid sample at or after the change presentation onset. All other time bins are 0.

ii.
```python
change = np.zeros(len(grid), dtype=np.int16)
if schange[pi] and not somit[pi] and starts[ti] <= ss[pi] < stops[ti]:
    q = np.searchsorted(grid, ss[pi], side='left')
    if q < len(grid): change[q] = 1
```

iii. The AI interpreted "Have value of 1 right after a change in image identity, otherwise 0" as a single-bin impulse rather than a sustained window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is a binary variable (0 or 1), so no thresholding is applied. It is directly determined by the presence of a change event.

ii. See 4-b above.

iii. The binary nature follows directly from the instruction specification.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change pulse is placed on the same 30 Hz grid as the neural data using `np.searchsorted` to find the appropriate grid index.

ii. See 4-b above.

iii. Same grid alignment as all other outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB file, which provides speed values and timestamps.

ii.
```python
rg = h['processing/running/speed']
rts, rsp = np.asarray(rg['timestamps'][:], float), np.asarray(rg['data'][:], float)
```

iii. This is the SDK's standard running speed data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 30 Hz trial grid. Interpolation only uses finite values. The continuous speed is then discretized into 5 quintile bins computed **per experiment** (not globally).

ii.
```python
run_cont.append(interp_1d_finite(rts, rsp, grid))
...
run_bins, redges = quintile(run_cont)

def quintile(values):
    allv = np.concatenate(values)
    edges = np.percentile(allv, [20,40,60,80])
    return [np.searchsorted(edges, x, side='right').astype(np.int16) for x in values], edges
```

iii. Per-experiment quintiles were chosen to prevent between-rig calibration and mouse-size differences from dominating labels.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using percentile-based edges at [20, 40, 60, 80] percentiles, computed from all eligible trial data **within each experiment**. `np.searchsorted` assigns each value to the appropriate bin (0-4).

ii.
```python
def quintile(values):
    allv = np.concatenate(values)
    edges = np.percentile(allv, [20,40,60,80])
    return [np.searchsorted(edges, x, side='right').astype(np.int16) for x in values], edges
```

iii. Percentile-based bins ensure roughly equal sample counts per bin within each experiment.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz grid as the neural data for each trial.

ii.
```python
run_cont.append(interp_1d_finite(rts, rsp, grid))
```

iii. Same grid ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` (pupil area in pixels^2) and eye tracking timestamps. The area is converted to equivalent-circle diameter: `2*sqrt(area/pi)`.

ii.
```python
eg = h['acquisition/EyeTracking']
ets = np.asarray(eg['eye_tracking/timestamps'][:], float)
area = np.asarray(eg['pupil_tracking/area'][:], float)
diameter = 2.0 * np.sqrt(np.maximum(area, 0.0) / np.pi)
```

iii. The pupil area was converted to diameter to give physical "diameter" semantics as specified in the task.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is converted to equivalent-circle diameter, then linearly interpolated from eye tracking timestamps to the 30 Hz trial grid using only finite values. The continuous diameter is then discretized into 5 quintile bins computed **per experiment**.

ii.
```python
diameter = 2.0 * np.sqrt(np.maximum(area, 0.0) / np.pi)
...
pupil_cont.append(interp_1d_finite(ets, diameter, grid))
...
pupil_bins, pedges = quintile(pupil_cont)
```

iii. Using only finite values for interpolation avoids propagating NaN/invalid samples. Per-experiment quintiles account for variability across mice and rigs.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same quintile approach as running speed: percentile edges at [20, 40, 60, 80] computed per experiment, producing 5 bins (0-4).

ii.
```python
pupil_bins, pedges = quintile(pupil_cont)
```

iii. Percentile-based binning ensures balanced classes within each experiment.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same 30 Hz trial grid as the neural data.

ii.
```python
pupil_cont.append(interp_1d_finite(ets, diameter, grid))
```

iii. Same grid ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials interval table.

ii.
```python
OUTCOMES = ['hit','miss','false_alarm','correct_reject']
...
out = next((oi for oi, name in enumerate(OUTCOMES) if bool(tr[name][ti])), None)
if out is None: raise ValueError(f'no outcome for eligible trial {ti} in {path.name}')
```

iii. These four columns are mutually exclusive for non-aborted, non-auto-rewarded trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The code is constant across all time bins within a trial (replicated as a full row in the output matrix).

ii.
```python
outcome = np.full(len(g), outcomes[i], dtype=np.int16)
outputs.append(np.vstack((image_rows[i], change_rows[i], run_bins[i], pupil_bins[i], outcome)))
```

iii. Trial outcome is semantically static per trial but replicated across time bins to fit the 2D output format required when other outputs are time-varying.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing pupil tracking**: 3 experiments lacking `acquisition/EyeTracking/pupil_tracking/area` are excluded entirely (returned as `None` with reason documented).
- **Trials outside ophys range**: Trials whose start/stop fall entirely outside the ophys timestamp range are filtered out.
- **Short trials**: Trials with fewer than 2 grid points are skipped.
- **NaN in running/pupil**: `interp_1d_finite` uses only finite values for interpolation, using `np.interp` which extrapolates edge values.
- **Negative pupil area**: `np.maximum(area, 0.0)` clamps negative areas to zero before taking square root.

ii.
```python
if 'acquisition/EyeTracking/pupil_tracking/area' not in h:
    return None, 'missing pupil tracking'
...
trial_idx = trial_idx[(stops[trial_idx] >= ots[0]) & (starts[trial_idx] <= ots[-1])]
...
if len(grid) < 2: continue
...
def interp_1d_finite(times, values, grid):
    ok = np.isfinite(times) & np.isfinite(values)
    if ok.sum() < 2:
        raise ValueError('fewer than two finite samples for interpolation')
    return np.interp(grid, times[ok], values[ok]).astype(np.float32)
```

iii. Excluding sessions without pupil data was justified because the required output cannot be constructed without it. The finite-only interpolation prevents NaN propagation.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file with h5py and reading the large event detection, running speed, and eye tracking arrays. This is I/O bound. The full conversion of 281 experiments took ~282 seconds (~4.7 minutes).

ii. N/A (timing is printed during execution)

iii. Direct h5py access is faster than the SDK's high-level API, which loads additional unnecessary data.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_experiment` iterates over each eligible trial sequentially, performing grid construction, neural interpolation, stimulus lookup, and behavior interpolation. The neural interpolation (`interp_rows`) is already vectorized across neurons but called separately per trial. The stimulus lookup loop over presentations could potentially be vectorized.

ii.
```python
for ti in trial_idx:
    ...
    n = interp_rows(ots, events, grid).T.astype(np.float32, copy=False)
    ...
    for pi in range(max(0,p0), min(len(ss),p1+1)):
        ...
```

iii. The per-trial loop is not a major bottleneck compared to I/O.

## 9-c. What processing does the code repeat multiple times?

i. The running speed and pupil diameter are read once per experiment. However, because the AI treats each experiment as a separate session, behavioral data (trials, running, pupil) that is shared across imaging planes of the same ophys session is loaded redundantly for multi-plane sessions. Of the 247 sessions, 8 have multiple planes (3-7 planes each), so some behavioral data is loaded multiple times.

ii. Each call to `process_experiment` opens the NWB and reads all data fresh:
```python
with h5py.File(path, 'r') as h:
    ...
    rg = h['processing/running/speed']
    ...
    eg = h['acquisition/EyeTracking']
```

iii. Since most sessions have only one plane, this duplication is minor.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes per-experiment quintile edges for running speed and pupil diameter, which are stored in the session info metadata but not used by the downstream decoder (the decoder uses the discretized bin indices). The continuous running/pupil values (`run_cont`, `pupil_cont`) are computed for discretization but then discarded. The code also resamples neural data to 30 Hz via linear interpolation, which upsamples from the native ~11 Hz rate, creating interpolated values that add no new information.
