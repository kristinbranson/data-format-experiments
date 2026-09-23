# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the Allen SDK's `VisualBehaviorOphysProjectCache` loaded via `from_local_cache()`. It reads the experiment table, filters to only locally-available NWB files (by scanning the NWB directory), removes passive sessions (`~et.passive`), and processes each ophys session. For each session, all imaging planes (experiments) are loaded via `cache.get_behavior_ophys_experiment(exp_id)`. A multiprocessing pool (16 workers) processes sessions in parallel.

ii.
```python
cache = get_cache()  # VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)
exp_table = cache.get_ophys_experiment_table()
# filter to locally available NWB files
local_ids = sorted(int(f.split('_')[-1].split('.')[0])
                   for f in os.listdir(nwb_dir) if f.endswith('.nwb'))
et = exp_table.loc[exp_table.index.intersection(local_ids)]
et = et[~et.passive]  # keep only active sessions
# ...
datasets = [cache.get_behavior_ophys_experiment(int(e)) for e in exp_ids]
```

iii. The AI justified using `from_local_cache` because there is no internet access. It scans for local NWB files to avoid trying to download missing ones. Passive sessions are excluded because they have 0 licks and 0 rewards.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the experiment table. Unique mouse IDs are collected from all successfully processed sessions and sorted.

ii.
```python
subjects = sorted({r['mouse_id'] for r in good})
subject_idx = np.array([subjects.index(r['mouse_id']) for r in good], dtype=np.int64)
```

iii. The `mouse_id` field is the SDK's unique identifier for each animal.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a unique `ophys_session_id`. Multiple imaging planes (experiments) within one session are merged into a single session, since they share behavior, stimulus, and clock. Sessions are sorted by session ID.

ii.
```python
session_ids = sorted(et.ophys_session_id.unique())
# ...
for sid in session_ids:
    rows = et[et.ophys_session_id == sid].sort_index()
    jobs.append((int(sid), list(rows.index), rows, ...))
```

iii. The AI groups all experiments with the same `ophys_session_id` together, merging their neural data. This is documented as appropriate because multiscope sessions share behavior and clock.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `trials` table. Only `go` and `catch` trials are kept (filtered by `trials.go | trials.catch`), with `change_time` required to be non-NaN. Each trial uses a fixed window of [-2, +4] seconds around `change_time`, producing 60 bins of 100ms each. Sessions with fewer than 2 valid trials are excluded.

ii.
```python
trials = d0.trials
keep = (trials.go | trials.catch) & trials.change_time.notna()
tr = trials[keep]
if len(tr) < 2:
    return {'session_id': ophys_session_id, 'skipped': 'fewer than 2 go/catch trials'}
# ...
for ct in change_times:
    edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
```

iii. The AI uses `go | catch` rather than `~aborted & ~auto_rewarded` to select trials, which achieves the same result since in the SDK these are mutually exclusive categories (go, catch, aborted, auto_rewarded). The fixed window around change_time was chosen so that every trial has the same number of time bins.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to keep only `go` and `catch` trials with valid `change_time`. Sessions without eye tracking are dropped entirely (3 sessions). Sessions with fewer than 2 valid trials are skipped. Error-causing sessions are caught by try/except and skipped.

ii.
```python
keep = (trials.go | trials.catch) & trials.change_time.notna()
tr = trials[keep]
if len(tr) < 2:
    return {'session_id': ophys_session_id, 'skipped': 'fewer than 2 go/catch trials'}
# ...
eye = d0.eye_tracking
if eye is None or len(eye) == 0:
    return {'session_id': ophys_session_id, 'skipped': 'no eye tracking'}
```

iii. Filtering matches the task specification. Sessions without eye tracking must be dropped because pupil diameter is a required output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `dff_traces.dff` (dF/F calcium fluorescence traces) by default. The script also supports `events` and `filtered_events` via the `--neural-signal` flag, but dF/F is the default.

ii.
```python
if neural_signal == 'dff':
    ev = np.vstack(d.dff_traces.dff.values).astype(np.float32)
elif neural_signal == 'filtered_events':
    ev = np.vstack(d.events.filtered_events.values).astype(np.float32)
else:
    ev = np.vstack(d.events.events.values).astype(np.float32)
```

iii. The AI empirically compared all three neural signals and found dF/F decoded substantially better than events at 100ms bins (e.g., image identity: 0.488 vs 0.236 validation balanced accuracy). This is a documented deviation from the paper which uses events.

## 2-b. How is the `neural` data processed?

i. Neural data from multiple imaging planes within a session are stacked vertically. Then, the data is bin-averaged into 100ms bins using a vectorized cumulative-sum approach. Each bin contains the mean of all ophys samples falling within that bin's time edges.

ii.
```python
plane_events = []
for d, eid in zip(datasets, exp_ids):
    ev = np.vstack(d.dff_traces.dff.values).astype(np.float32)
    plane_events.append(ev)
# ...
def bin_average(values, timestamps, edges):
    idx = np.searchsorted(sub_ts, edges, side='left')
    counts = np.diff(idx)
    csum = np.concatenate([np.zeros(...), np.cumsum(sub..., axis=1)], axis=1)
    sums = csum[:, idx[1:]] - csum[:, idx[:-1]]
    out[:, nz] = sums[:, nz] / counts[nz]
    return out, counts
```

iii. The bin averaging approach is efficient (vectorized via cumsum) and ensures every bin has at least one sample for the 11 Hz mesoscope data.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. The released data already contains only `valid_roi == True` ROIs with upstream filtering (crosstalk removal, demixing, neuropil subtraction).

ii. N/A (no filtering code)

iii. The AI noted that the SDK pipeline already applies quality control, so no further filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to `change_time` (the image change event). For catch trials, this is the sham change time computed by the SDK. A fixed window of [-2, +4] seconds around the change is used, producing 60 bins of 100ms each. Each plane's data is binned on its own ophys timestamps.

ii.
```python
for ct in change_times:
    edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    # neural: bin each plane on its own timestamps, then stack
    parts = []
    for ev, ts in zip(plane_events, plane_ts):
        b, counts = bin_average(ev, ts, edges)
        parts.append(b)
    neural_trials.append(np.vstack(parts).astype(np.float32))
```

iii. The AI chose change_time alignment because it is the defining event of the change detection task and what the paper aligns to. The [-2, +4]s window was verified to always fall within the recording.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned to 100ms (10 Hz) bins, producing 60 bins per trial. This is a deviation from the native ophys frame rate (~11-31 Hz depending on the rig).

ii.
```python
BIN_SIZE = 0.1          # seconds (100 ms)
OFF_START = -2.0
OFF_END = 4.0
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 60
```

iii. The AI justified 100ms bins as the smallest round bin size >= the slowest sampling rate (11 Hz mesoscope), avoiding upsampling. The target format requires consistent time bins across trials and sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, using the `image_name` column, filtered to the `change_detection` stimulus block. For each time bin, the image identity is determined by which 750ms flash interval contains the bin center.

ii.
```python
sp = d0.stimulus_presentations
sp = sp[sp.stimulus_block_name.str.contains('change_detection')]
stim_start = sp.start_time.values.astype(np.float64)
stim_img = sp.image_name.values.astype(str)
# ...
j = np.searchsorted(stim_start, centers, side='right') - 1
j = np.clip(j, 0, len(stim_start) - 1)
within = centers < stim_end[j]
img = np.where(within, stim_code[j], img_to_code['omitted'])
```

iii. The AI uses stimulus_presentations rather than the trials table to determine image identity, which provides flash-by-flash precision. This allows detecting omitted flashes and correctly assigning image identity at every point in the trial window.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes using a fixed global list of 17 classes (16 images from sets A and B + 'omitted'). For each 100ms bin, the image is determined by finding which stimulus flash interval contains the bin center. Bins that fall outside any flash interval are labeled 'omitted'.

ii.
```python
image_names = sorted({
    'im000', 'im031', 'im035', 'im045', 'im054', 'im073', 'im075', 'im106',
    'im061', 'im062', 'im063', 'im065', 'im066', 'im069', 'im077', 'im085'})
image_names = image_names + ['omitted']
img_to_code = {name: i for i, name in enumerate(image_names)}
# ...
j = np.searchsorted(stim_start, centers, side='right') - 1
within = centers < stim_end[j]
img = np.where(within, stim_code[j], img_to_code['omitted'])
```

iii. The global image list is hardcoded to cover both image sets A and B (16 images + omitted = 17 classes). The flash interval ends at the onset of the next flash (or 750ms if last flash), capped at 1s to handle variable inter-flash intervals.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is assigned per 100ms bin using the bin center time. The same bin edges derived from `change_time + OFF_START + BIN_SIZE * np.arange(NBINS + 1)` are used for both neural data and image identity.

ii.
```python
centers = 0.5 * (edges[:-1] + edges[1:])
j = np.searchsorted(stim_start, centers, side='right') - 1
```

iii. Both neural and output data use the same time bins, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change`, which flags the flash where the image identity actually changes. This is combined with the stimulus timing to determine which bins fall within the change flash interval.

ii.
```python
stim_change = sp.is_change.values.astype(bool)
# ...
chg = np.where(within, stim_change[j], False).astype(np.int64)
```

iii. Using `is_change` from stimulus_presentations correctly identifies changes, as the SDK sets this flag only for actual image changes (not sham/catch changes).

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each 100ms bin, if the bin center falls within a flash interval where `is_change` is True, the image_change value is 1; otherwise 0. This means image_change is 1 only during the ~750ms change flash on go trials.

ii.
```python
chg = np.where(within, stim_change[j], False).astype(np.int64)
change_trials.append(chg)
```

iii. The approach naturally handles catch trials (where `is_change` is False for all flashes).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1) - no thresholding is needed.

ii. See 4-b above.

iii. N/A.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same bin structure as neural data and image identity - uses the same 100ms bins derived from change_time alignment.

ii. See 4-a above.

iii. Same time bin alignment as all other variables.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides 60 Hz speed data (10 Hz low-pass filtered by the SDK).

ii.
```python
rs = d0.running_speed
run_ts = rs.timestamps.values.astype(np.float64)
run_sp = interpolate_nans(rs.speed.values)
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is first NaN-interpolated, then bin-averaged into 100ms bins. Empty bins (from dropped frames) are filled by interpolating the raw trace at the bin center. The continuous values are then discretized into 5 equal-percentile bins computed per-session.

ii.
```python
run_sp = interpolate_nans(rs.speed.values)
rb, rc = bin_average(run_sp[None, :], run_ts, edges)
if (rc == 0).any():
    rb[rc == 0] = np.interp(centers[rc == 0], run_ts, run_sp)
# ...
run_lab, run_q = quantile_bins(run_mat.ravel())
```

iii. NaN interpolation, bin averaging, and empty-bin filling ensure no missing data. Per-session quintiles were chosen because absolute speed values vary across mice/rigs.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using per-session percentile thresholds (quintiles). The interior quantiles are computed from all bins within each session.

ii.
```python
def quantile_bins(values, nq=NQUANT):
    qs = np.quantile(values, np.arange(1, nq) / nq)
    labels = np.searchsorted(qs, values, side='right')
    return labels.astype(np.int64), qs
# ...
run_lab, run_q = quantile_bins(run_mat.ravel())
```

iii. Per-session quintiles ensure equal bin counts within each session, giving a consistent meaning across sessions with different baseline speeds.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is bin-averaged into the same 100ms bins used for neural data, using the same time edges derived from `change_time + OFF_START`.

ii.
```python
rb, rc = bin_average(run_sp[None, :], run_ts, edges)
```

iii. Same bin structure as neural data ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `eye_tracking.pupil_area`, converted to diameter via `2 * sqrt(area / pi)`.

ii.
```python
eye = d0.eye_tracking
pupil_area = eye.pupil_area.values.astype(np.float64)
pupil_diam_raw = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. The AI chose `pupil_area` and computed diameter from it, rather than using `pupil_width` directly.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is converted to diameter, NaN values (from blinks, where `pupil_area` is NaN) are linearly interpolated, then bin-averaged into 100ms bins. Empty bins are filled by interpolation. The result is discretized into 5 per-session quintiles.

ii.
```python
pupil_diam_raw = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diam = interpolate_nans(pupil_diam_raw)
# ...
pb, pc = bin_average(pupil_diam[None, :], eye_ts, edges)
if (pc == 0).any():
    pb[pc == 0] = np.interp(centers[pc == 0], eye_ts, pupil_diam)
pupil_trials.append(pb)
# ...
pupil_lab, pupil_q = quantile_bins(pupil_mat.ravel())
```

iii. NaN interpolation across blinks, bin averaging, and per-session quintile discretization.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed - 5 per-session quintile bins using `quantile_bins()`.

ii.
```python
pupil_lab, pupil_q = quantile_bins(pupil_mat.ravel())
```

iii. Per-session quintiles account for between-session/between-mouse variability in pupil size (measured in camera pixels, which depend on rig geometry).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same bin structure as neural data - bin-averaged into 100ms bins with the same time edges.

ii.
```python
pb, pc = bin_average(pupil_diam[None, :], eye_ts, edges)
```

iii. Same alignment approach as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
# ...
outcome = np.full(len(tr), -1, dtype=np.int64)
for k, name in enumerate(OUTCOMES):
    outcome[tr[name].values.astype(bool)] = k
assert (outcome >= 0).all(), 'trial with no outcome label'
```

iii. These are the SDK's canonical trial outcome labels. An assertion verifies every trial has exactly one outcome.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) and broadcast across all 60 time bins of the trial (static per-trial label made time-varying).

ii.
```python
out = np.stack([img_trials[i],
                change_trials[i],
                run_lab[i],
                pupil_lab[i],
                np.full(NBINS, outcome[i], dtype=np.int64)], axis=0)
```

iii. Broadcasting the static label across time bins follows the instruction "If at all possible, make it time-varying."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Sessions without eye tracking**: 3 sessions dropped entirely.
- **NaN values in running/pupil**: Linearly interpolated before binning.
- **Empty bins** (from dropped camera frames): Filled by interpolating the raw trace at the bin center.
- **Failed sessions**: Caught by try/except, logged and skipped.
- **Multi-plane consistency**: Assert that trial tables match across planes.
- **Blink frames**: Pupil area is NaN during blinks; handled by NaN interpolation.

ii.
```python
if eye is None or len(eye) == 0:
    return {'session_id': ophys_session_id, 'skipped': 'no eye tracking'}
# ...
run_sp = interpolate_nans(rs.speed.values)
pupil_diam = interpolate_nans(pupil_diam_raw)
# ...
if (rc == 0).any():
    rb[rc == 0] = np.interp(centers[rc == 0], run_ts, run_sp)
# ...
for d in datasets[1:]:
    assert len(t_other) == len(trials), 'trial tables differ across planes'
```

iii. The approach is robust: NaN interpolation prevents missing data from propagating, empty bins are filled rather than left at zero, and error handling prevents single bad sessions from crashing the pipeline.

## 9-a. What are the most time-consuming steps of the code?

i. NWB file loading via `cache.get_behavior_ophys_experiment()` is the bottleneck (~3s per experiment). The AI mitigated this with multiprocessing (16 workers).

ii.
```python
if args.workers > 1 and len(jobs) > 1:
    with Pool(min(args.workers, len(jobs))) as pool:
        for i, r in enumerate(pool.imap_unordered(process_session, jobs)):
```

iii. Full conversion takes ~69s wall clock with 16 workers.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop over `change_times` in `process_session` iterates sequentially. However, the bin averaging itself is vectorized (cumsum + searchsorted). The image identity assignment uses `np.searchsorted` and `np.where` (vectorized over bins).

ii. N/A (loops are not a bottleneck compared to I/O)

iii. The I/O dominates runtime, making loop vectorization a minor concern.

## 9-c. What processing does the code repeat multiple times?

i. Each session creates a new `get_cache()` instance in the worker process. The `cell_ids` are read from `d.events.index.values` even when using dF/F (the events table is accessed for cell IDs regardless of neural signal choice).

ii.
```python
cache = get_cache()  # called in every worker process
# ...
cell_ids.extend(list(d.events.index.values))  # accessed even in dF/F mode
```

iii. Creating the cache object per worker is necessary for multiprocessing (each process needs its own instance). The events access for cell IDs is a minor inefficiency.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores extensive per-session metadata (cell_specimen_ids, run_quantiles, pupil_quantiles, timing info, etc.) in the intermediate results that are passed to the assembly phase, but only a subset is retained in the final pickle. The processing plots (`--show-processing`) are optional. The code also accesses `d.events.index.values` for cell IDs even when using dF/F.

ii.
```python
result = {
    'cell_specimen_ids': [int(c) for c in cell_ids],
    'run_quantiles': run_q.tolist(),
    'pupil_quantiles': pupil_q.tolist(),
    'timing': timing,
    'total_time': time.time() - t_start,
    # ... many fields not in final output
}
```

iii. The extra metadata is useful for debugging and validation but is not used by the decoder.
