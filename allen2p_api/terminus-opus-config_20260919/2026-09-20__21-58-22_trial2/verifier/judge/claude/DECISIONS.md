# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data are loaded through the AllenSDK `VisualBehaviorOphysProjectCache` using `from_local_cache()`. The experiment table is obtained via `get_ophys_experiment_table()`. Sessions are selected by filtering out passive sessions and intersecting with locally available NWB files. For each session, each imaging plane (experiment) is loaded via `bc.get_behavior_ophys_experiment(eid)`. Multiprocessing (16 workers) is used to process sessions in parallel.

ii.
```python
def get_cache():
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    return VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)

def select_sessions(sample=False):
    bc = get_cache()
    et = bc.get_ophys_experiment_table()
    local_ids = set(int(f.split('_')[-1].split('.')[0]) for f in os.listdir(
        os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0',
                     'behavior_ophys_experiments')))
    et = et.loc[sorted(set(et.index).intersection(local_ids))]
    et = et[~et.passive]
    groups = []
    for sid, sub in et.groupby('ophys_session_id'):
        groups.append((int(sid), sorted(int(i) for i in sub.index)))
    groups.sort()
    return groups, et
```

iii. The AI uses `from_local_cache()` (rather than `from_s3_cache`) and filters by `~et.passive` to select active behavior sessions. It scans the local data directory to determine which experiments are available, then groups experiments by `ophys_session_id`. This approach was justified by the fact that passive sessions have no licking/rewards and would produce degenerate trial outcomes.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `mouse_id` field from each session's metadata. Unique mouse IDs are collected across all processed sessions and sorted.

ii.
```python
subjects = sorted(set(r['mouse_id'] for r in results))
# ...
data['subject_idx'].append(subjects.index(r['mouse_id']))
```

iii. The `mouse_id` from the SDK metadata uniquely identifies each animal. 38 mice are present in the local cache.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a unique `ophys_session_id`. All active imaging planes (experiments) within a session are grouped together and concatenated along the neuron axis.

ii.
```python
def select_sessions(sample=False):
    # ...
    et = et[~et.passive]
    groups = []
    for sid, sub in et.groupby('ophys_session_id'):
        groups.append((int(sid), sorted(int(i) for i in sub.index)))
    groups.sort()
    return groups, et
```

iii. Grouping by `ophys_session_id` ensures all imaging planes from the same behavioral session are treated as one session. Passive sessions are excluded because the animal is not performing the task. Sessions without eye tracking data (3 sessions) are also dropped because the pupil output cannot be defined.

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `dataset.trials` table. Only `go` and `catch` trials are kept (aborted and auto-rewarded are dropped). Each trial is aligned to `change_time` and a fixed window of [-2.0, +3.0] seconds around the change is extracted, binned into 250 ms bins (20 bins per trial).

ii.
```python
trials = ds.trials
keep = trials[(trials.go | trials.catch)]
if len(keep) < 2:
    return None
change_times = keep.change_time.values.astype(float)
edges = bin_edges_for_trials(change_times)

# bin_edges_for_trials:
def bin_edges_for_trials(change_times):
    offsets = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    return change_times[:, None] + offsets[None, :]
```

iii. The AI chose a fixed-length window around `change_time` rather than the variable-length `start_time` to `stop_time` window. The justification was that `change_time` is the natural alignment event for the change-detection task, and the [-2, +3] s window is safe for every trial (minimum change-to-start is 3.02 s, stop_time is always change+4.23 s). The 250 ms bin size was chosen to match the image presentation duration and be 1/3 of the 750 ms flash cycle, phase-locking bins to stimuli.

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded trials are excluded by keeping only `go` and `catch` trials. Sessions with fewer than 2 valid trials are dropped. Sessions without eye tracking data are also excluded.

ii.
```python
keep = trials[(trials.go | trials.catch)]
if len(keep) < 2:
    return None
# ...
if pupil_binned is None:
    return dict(skip=True, ophys_session_id=int(ophys_session_id),
                reason='no eye tracking')
```

iii. Per the task instructions, aborted and auto-rewarded trials are excluded. The `go | catch` filter is equivalent to `~aborted & ~auto_rewarded` since these four categories are mutually exclusive. Sessions without eye tracking are dropped because the pupil diameter output cannot be computed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dataset.dff_traces` (dF/F calcium fluorescence traces). The AI initially planned to use `dataset.events` (L0 calcium events) to match the reference paper, but switched to dF/F after a controlled comparison showed it decoded better on 4 of 5 outputs.

ii.
```python
if signal == 'dff':
    ev = ds.dff_traces
    E = np.vstack(ev['dff'].values).astype(np.float32)
else:
    ev = ds.events
    E = np.vstack(ev[signal].values).astype(np.float32)
```

iii. The reference paper explicitly states "For all analysis of neural data we used the detected calcium events." However, the AI's controlled comparison on 20 sessions showed dF/F decoded better and removed the empty-trial problem in sparse Vip/Sst planes. The `--neural-signal` flag keeps both options available.

## 2-b. How is the `neural` data processed?

i. The dF/F traces are binned into 250 ms bins using a cumulative-sum approach. For dF/F, the bin statistic is the **mean** (not sum), because dF/F is a rate-like quantity and the number of ophys frames per bin varies between rigs (7-8 at 31 Hz, 2-3 at 11 Hz). Multiple imaging planes within a session are concatenated along the neuron axis.

ii.
```python
binned = bin_sum(E, ots, edges)  # (ncells, ntrials, NBINS)
if signal == 'dff':
    counts = bin_sum(np.ones((1, E.shape[1]), dtype=np.float32), ots, edges)
    binned = binned / np.maximum(counts, 1.0)
neural_planes.append(binned)
# ...
neural = np.concatenate(neural_planes, axis=0)  # (nneurons, ntrials, NBINS)
```

iii. The bin mean is justified because dF/F is a rate-like quantity. Summing would create a spurious scale factor between the 31 Hz and 11 Hz rigs. The cumulative-sum approach (`bin_sum`) is vectorized for efficiency.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering beyond what the AllenSDK already applies. The SDK's `exclude_invalid_rois=True` default filters out invalid ROIs (union/duplicate/motion-border/dendrite/too small/dim).

ii. N/A (no explicit filtering code)

iii. The AllenSDK pipeline already applies ROI quality control, and the reference paper adds no additional per-cell filtering beyond the SDK defaults.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to `change_time` (the onset of the changed image on go trials, or the sham change on catch trials). A fixed window of [-2.0, +3.0] s is extracted around this event, binned into 250 ms bins.

ii.
```python
change_times = keep.change_time.values.astype(float)
edges = bin_edges_for_trials(change_times)
# ...
binned = bin_sum(E, ots, edges)
```

iii. `change_time` coincides exactly with a stimulus flash onset (verified with atol 1e-9). The [-2, +3] s window provides 2 s of pre-change baseline and 3 s of post-change response, and is verified to stay within every trial's boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned into 250 ms bins (20 bins per trial). This is a temporal rebinning from the native ophys frame rate (~31 Hz or ~11 Hz) to ~4 Hz.

ii.
```python
BIN_SIZE = 0.25  # s, = image duration, 1/3 of the 750 ms flash cycle
OFF_START = -2.0
OFF_END = 3.0
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))  # 20
```

iii. The 250 ms bin size matches the image presentation duration and is exactly 1/3 of the 750 ms flash cycle (image + gray). This phase-locks the bins to the stimulus, and guarantees >= 2 ophys frames per bin even at the 11 Hz Multiscope rate. The `time_bin_size` metadata is set to 250.0 ms.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations` via the `flash_table()` function. The image name at each bin center is determined by finding which flash's 750 ms presentation interval contains that time point.

ii.
```python
def flash_table(dataset):
    sp = dataset.stimulus_presentations
    sp = sp[sp.stimulus_block_name.str.contains('change_detection')]
    sp = sp.sort_values('start_time')
    names = sp.image_name.astype(str).replace('omitted', np.nan).ffill().bfill().values
    return (sp.start_time.values.astype(float), names,
            sp.is_change.values.astype(bool), sp)

# In process_session:
starts, names, is_change, sp = flash_table(ds)
centres = edges[:, :-1] + BIN_SIZE / 2.0
fidx = np.searchsorted(starts, centres.ravel(), side='right') - 1
fidx = np.clip(fidx, 0, len(starts) - 1)
image_name = names[fidx].reshape(ntrials, NBINS)
```

iii. Using `stimulus_presentations` directly provides precise timing of each image flash. Omitted flashes have their `image_name` replaced with NaN and forward-filled, so the identity of the image that should have been shown is preserved. This matches the paper's treatment of omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global sorted vocabulary of all unique image names across sessions. Each bin is assigned the image whose 750 ms presentation interval contains the bin center.

ii.
```python
images = sorted(set(np.concatenate([r['image_name'].ravel() for r in results])))
img_index = {n: i for i, n in enumerate(images)}
# ...
img_idx = np.vectorize(img_index.get)(r['image_name']).astype(np.int64)
```

iii. A global sorted vocabulary ensures consistent integer codes across sessions. 16 unique images are identified (8 from image set A, 8 from set B).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is determined at each bin center time, using the same bin edges as the neural data. `np.searchsorted` finds which flash's presentation interval contains each bin center.

ii.
```python
centres = edges[:, :-1] + BIN_SIZE / 2.0
fidx = np.searchsorted(starts, centres.ravel(), side='right') - 1
image_name = names[fidx].reshape(ntrials, NBINS)
```

iii. Both neural data and image identity use the same bin edges (derived from `change_times`), ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change` and the flash timing (`start_time`). It is 1 only in the single 250 ms bin that starts with a change, and 0 otherwise.

ii.
```python
starts, names, is_change, sp = flash_table(ds)
# ...
change_flag = np.zeros((ntrials, NBINS), dtype=np.int64)
flash_start = starts[fidx].reshape(ntrials, NBINS)
first_bin = (centres - flash_start) < BIN_SIZE
change_flag[is_change[fidx].reshape(ntrials, NBINS) & first_bin] = 1
```

iii. The AI tested both a single 250 ms bin and a full 750 ms interval encoding. The single bin decoded better (+0.034 accuracy) and is the literal reading of "value of 1 right after a change."

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each bin, the flash whose presentation interval contains the bin center is identified. If that flash is a change (`is_change == True`) and the bin center falls within the first 250 ms of that flash's interval, the change flag is set to 1.

ii. See 4-a code snippet.

iii. This ensures that only the single bin at the onset of the change is flagged, rather than the entire 750 ms presentation interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is needed. On catch trials, `is_change` is False by the SDK's definition, so catch trials have all-zero change flags.

ii. See 4-a.

iii. The binary encoding naturally separates change from no-change bins.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same bin edges as neural data and image identity. The change flag is computed per bin using the same timing framework.

ii. See 4-a.

iii. Alignment is guaranteed by shared bin edges derived from `change_times`.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides speed (cm/s) and timestamps from the running wheel encoder.

ii.
```python
rs = ds.running_speed
speed = np.asarray(rs['speed'].values, dtype=float)
speed_t = np.asarray(rs['timestamps'].values, dtype=float)
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data. The speed is already low-pass filtered at 10 Hz by the pipeline.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each 250 ms bin using `bin_mean_1d()`, then discretized into 5 equal-percentile bins. NaN values are interpolated. Discretization is done **per session** (default) using equal-percentile bins.

ii.
```python
speed = interpolate_nans(speed)
running_binned = bin_mean_1d(speed, speed_t, edges)
# ...
run_q, run_edges = discretize([r['running'] for r in results], scope=args.quantile_scope)
```

iii. Per-session quintiles are justified because each mouse has its own running baseline, so global cuts would partly encode session identity. The `--quantile-scope global` option is available for comparison.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile (quintile) bins. The default scope is per-session, meaning each session contributes exactly 20% of its bins to each quintile.

ii.
```python
def discretize(values_list, nq=NQUANTILES, scope='session'):
    if scope == 'global':
        out = [np.digitize(v, global_qs[1:-1], right=False).astype(np.int64)
               for v in values_list]
    else:
        out = []
        for v in values_list:
            qs = np.percentile(v.ravel(), np.linspace(0, 100, nq + 1))
            out.append(np.digitize(v, qs[1:-1], right=False).astype(np.int64))
    return out, global_qs
```

iii. Per-session percentile bins remove session-specific offsets in running behavior.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is binned using the same bin edges as neural data (`bin_mean_1d` uses the same `edges` array).

ii.
```python
running_binned = bin_mean_1d(speed, speed_t, edges)
```

iii. The shared bin edges guarantee temporal alignment between running speed and neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using both `pupil_width` and `pupil_height`. Diameter is computed as `2 * max(pupil_width, pupil_height)`.

ii.
```python
et = ds.eye_tracking
diam = 2.0 * np.nanmax(
    np.vstack([et['pupil_width'].values, et['pupil_height'].values]), axis=0)
```

iii. The AI used `2 * max(pupil_width, pupil_height)` as the diameter estimate, following the whitepaper's note that "major axis reflects the pupil diameter." This differs from using `pupil_width` alone.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames (NaN values) are linearly interpolated using `interpolate_nans()`. The interpolated diameter is then averaged within each 250 ms bin via `bin_mean_1d()`, and discretized into 5 per-session equal-percentile bins.

ii.
```python
diam = interpolate_nans(diam)
if diam is not None:
    pupil_binned = bin_mean_1d(diam, np.asarray(et['timestamps'].values, float), edges)
# ...
pup_q, pup_edges = discretize([r['pupil'] for r in results], scope=args.quantile_scope)
```

iii. Blink interpolation prevents NaN propagation. Per-session quintiles are used because pupil diameter is measured in camera pixels, which vary between rigs and sessions.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal-percentile bins, per-session by default.

ii. See 5-c code snippet (same `discretize()` function).

iii. Per-session percentile binning accounts for the fact that pupil diameter in camera pixels is not comparable across rigs.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same bin edges as neural data. Pupil diameter is binned using `bin_mean_1d()` with the same `edges` array.

ii.
```python
pupil_binned = bin_mean_1d(diam, np.asarray(et['timestamps'].values, float), edges)
```

iii. Shared bin edges guarantee temporal alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome = np.full(ntrials, -1, dtype=np.int64)
for i, name in enumerate(OUTCOMES):
    outcome[keep[name].values.astype(bool)] = i
assert np.all(outcome >= 0), 'trial with no hit/miss/FA/CR outcome'
```

iii. These four columns are the SDK's canonical trial outcome labels for the change detection task. They are mutually exclusive for go and catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) via the `OUTCOMES` list ordering. The outcome is static per trial and broadcast over all 20 time bins.

ii.
```python
out = np.stack([img_idx[t], r['image_change'][t], run_q[i][t], pup_q[i][t],
                np.full(T, r['outcome'][t], dtype=np.int64)], axis=0)
```

iii. The outcome code is constant across all bins within a trial. An assertion verifies that every kept trial has exactly one of the four outcomes.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **No eye tracking**: Sessions without usable eye tracking are skipped entirely (3 sessions).
- **Blink NaNs**: Linearly interpolated in the pupil trace using `interpolate_nans()`.
- **Running speed NaNs**: Also linearly interpolated.
- **Empty bins**: `bin_mean_1d()` falls back to linear interpolation at the bin center for bins without samples.
- **Failed sessions**: If `process_session` raises an exception, it returns `None` and the session is skipped.
- **Sessions with < 2 trials**: Returned as `None` and skipped.
- **Omitted flashes**: Image name forward-filled from the previous flash.

ii.
```python
def interpolate_nans(values):
    v = np.asarray(values, dtype=float).copy()
    good = np.isfinite(v)
    if not np.any(good):
        return None
    if not np.all(good):
        x = np.arange(len(v))
        v[~good] = np.interp(x[~good], x[good], v[good])
    return v

# Empty eye tracking -> skip session
if pupil_binned is None:
    return dict(skip=True, ophys_session_id=int(ophys_session_id),
                reason='no eye tracking')
```

iii. These approaches prevent crashes from data quality issues while preserving as much data as possible. Sessions without eye tracking are dropped rather than filled because the pupil output would be meaningless.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `bc.get_behavior_ophys_experiment()`, which reads large NWB files from the local cache (3-4 seconds per imaging plane). This dominates runtime.

ii. N/A

iii. Binning and behavior processing take < 0.2 s per session. The NWB file I/O is the bottleneck. Multiprocessing (16 workers) provides ~16x speedup, bringing the full conversion to ~83 seconds.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in the output assembly phase iterates over trials sequentially. However, the main computational work (binning neural data, computing stimulus outputs, binning running/pupil) is already vectorized across all trials simultaneously using `bin_sum()` and `bin_mean_1d()`.

ii.
```python
# Already vectorized:
binned = bin_sum(E, ots, edges)  # all trials at once
running_binned = bin_mean_1d(speed, speed_t, edges)  # all trials at once

# Per-trial loop (output assembly only):
for t in range(ntrials):
    neural_trials.append(np.ascontiguousarray(r['neural'][:, t, :], dtype=np.float32))
```

iii. The per-trial output assembly loop is lightweight (just array slicing and stacking). The expensive operations are already vectorized.

## 9-c. What processing does the code repeat multiple times?

i. Each session reloads the experiment data through the AllenSDK, but within a session, data is loaded once per imaging plane and processed in a single pass. The `flash_table()` function is called once per session in `process_session` and again in `make_plots()` if `--show-processing` is enabled.

ii. N/A

iii. The double `flash_table()` call in plotting mode is minor since it only applies to the first 2 sessions. No significant repeated processing occurs.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores continuous running and pupil values (`r['running']`, `r['pupil']`) in the per-session results alongside the discretized quintile values. These continuous values are only used for plotting (`make_plots()`) and are not included in the final pickle. The `go`/`catch` boolean arrays, `change_times`, and other metadata per session are also kept for sanity checks but not saved.

ii.
```python
result = dict(
    # ...
    running=running_binned.astype(np.float64),  # kept for plotting/sanity
    pupil=pupil_binned.astype(np.float64),       # kept for plotting/sanity
    change_times=keep.change_time.values.astype(float),  # sanity checks
    go=keep.go.values.astype(bool),              # sanity checks
    catch=keep.catch.values.astype(bool),         # sanity checks
    # ...
)
```

iii. These intermediate values serve a useful purpose for verification and plotting but are not saved to the final output. The memory overhead is modest.
