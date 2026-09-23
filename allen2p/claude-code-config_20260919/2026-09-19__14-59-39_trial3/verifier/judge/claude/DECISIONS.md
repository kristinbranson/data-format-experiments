# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the released `ophys_experiment_table.csv` from the project metadata directory to discover all experiments. It filters to experiments whose NWB file is present locally and whose `project_code == 'VisualBehavior'`. Each selected experiment is loaded individually via `BehaviorOphysExperiment.from_nwb_path()` rather than through the `VisualBehaviorOphysProjectCache` (the SDK cache was unusable because the required `manifests` sub-folder was missing). This loads the same object type the SDK cache would return.

ii.
```python
def select_experiments():
    et = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    present = set()
    for fn in os.listdir(NWB_DIR):
        if fn.endswith('.nwb'):
            present.add(int(fn.split('_')[-1].split('.')[0]))
    sel = et[et.ophys_experiment_id.isin(present)
             & (et.project_code == PROJECT_CODE)
             & (~et.passive.astype(bool))].copy()
    ...

# In process_experiment():
ds = BehaviorOphysExperiment.from_nwb_path(path)
```

iii. The AI documented that `VisualBehaviorOphysProjectCache.from_local_cache` could not be used because the local data lacked the required `manifests` sub-folder. Since `get_behavior_ophys_experiment()` ultimately calls `from_nwb`, loading via `from_nwb_path` produces an identical object. Session selection uses the same `ophys_experiment_table.csv` that the cache's `get_ophys_experiment_table()` returns.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the experiment metadata. After processing, subjects are collected as the sorted set of unique `mouse_id` strings from the kept sessions.

ii.
```python
subjects = sorted({r['subject'] for r in kept})
subject_index = {s: i for i, s in enumerate(subjects)}
# ...
'subject': str(md['mouse_id']),
```

iii. The `mouse_id` field is the SDK's unique identifier for each animal. The AI verified 37 unique mice across the VisualBehavior project.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a single ophys experiment (one NWB file). In the `VisualBehavior` project, every ophys session has exactly one experiment (one imaging plane), so experiment == session. The AI verified this 1:1 mapping in CONVERSION_NOTES.md Step 4.

ii.
```python
def process_experiment(job):
    oeid = job['ophys_experiment_id']
    path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{oeid}.nwb')
    ds = BehaviorOphysExperiment.from_nwb_path(path)
    ...
```

iii. The AI documented that for the VisualBehavior project, every ophys session has exactly 1 experiment (single-plane Scientifica rig), so treating each experiment as a session is correct. This was verified against the experiment table. The AI also explicitly excluded passive sessions (`passive == False`), noting that the paper states passive viewing "was not analyzed here" and that trial outcomes are degenerate in passive sessions (no hits or false alarms).

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `dataset.trials` table. Only go and catch trials are kept (filtering by `go | catch`), which excludes aborted and auto-rewarded trials. The trial window spans from `start_time` to `stop_time`, giving variable-length trials. Ophys frames within `[start_time, stop_time)` are extracted.

ii.
```python
trials = ds.trials
go = trials['go'].values.astype(bool)
catch = trials['catch'].values.astype(bool)
keep = go | catch
tr = trials[keep]
...
starts = tr['start_time'].values.astype(np.float64)
stops = tr['stop_time'].values.astype(np.float64)
i0 = np.searchsorted(ts, starts, side='left')
i1 = np.searchsorted(ts, stops, side='left')
```

iii. The AI noted that in this dataset, `go`, `catch`, `aborted`, and `auto_rewarded` are mutually exclusive, so `go | catch` is equivalent to `~aborted & ~auto_rewarded`. The trial window `[start_time, stop_time)` matches the reference tutorial's approach. Variable-length trials enable time-varying outputs.

## 1-e. How are trials filtered based on quality controls?

i. Beyond excluding aborted and auto-rewarded trials:
- Sessions without eye tracking data are skipped entirely (3 sessions).
- Sessions without valid ROIs are skipped.
- Trials with fewer than 2 ophys frames (`MIN_FRAMES_PER_TRIAL = 2`) are dropped.
- Trials with no valid (non-blink) pupil measurement are dropped (60 trials total).
- Sessions with fewer than 2 usable trials are dropped.
- Passive sessions are excluded at the selection stage.

ii.
```python
if eye is None or len(eye) == 0 or not np.isfinite(eye['pupil_area'].values).any():
    return {'oeid': int(oeid), 'skip': 'no eye tracking data'}
...
valid_roi = cst['valid_roi'].values.astype(bool)
cell_ids = cst.index.values[valid_roi]
if len(cell_ids) == 0:
    return {'oeid': int(oeid), 'skip': 'no valid ROIs'}
...
for k in range(len(tr)):
    if i1[k] - i0[k] < MIN_FRAMES_PER_TRIAL:
        n_drop_short += 1; continue
    if j1[k] - j0[k] < 1:
        n_drop_pupil += 1; continue
    sel.append(k)
if len(sel) < MIN_TRIALS_PER_SESSION:
    return {'oeid': int(oeid), 'skip': f'only {len(sel)} usable trials'}
```

iii. The AI justified excluding sessions without eye tracking because pupil diameter is a required decoder output. Trials without pupil data would have pure extrapolation values. The `valid_roi` filter is a guard (all 29,097 cells passed). The minimum-trials threshold ensures the decoder can be trained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces.dff` (delta F/F calcium fluorescence traces). The AI benchmarked `dff`, `events`, and `filtered_events` and chose dF/F because it decoded better on every output.

ii.
```python
signal = job.get('neural_signal', 'dff')
if signal == 'dff':
    col = ds.dff_traces['dff']
else:
    col = ds.events[signal]
traces = np.empty((len(cell_ids), len(ts)), dtype=np.float32)
for i, cid in enumerate(cell_ids):
    traces[i] = col.loc[cid]
```

iii. The AI documented extensive benchmarking (CONVERSION_NOTES Steps 7-8): dF/F achieved higher validation accuracy than events or filtered_events on every output (e.g., image_identity: 0.470 vs 0.292 for filtered_events on 20 sessions). The AI noted that detected events are ~0.25% non-zero per 32ms frame, making them too sparse for a per-timepoint decoder. dF/F is the primary neural product of the release and is already neuropil-subtracted, demixed, and baseline-normalized.

## 2-b. How is the `neural` data processed?

i. The dF/F traces are used as-is from the NWB files, with only a `valid_roi` filter applied to select cells. No additional processing (no z-scoring, no smoothing, no filtering). The traces are stored as float32.

ii.
```python
cst = ds.cell_specimen_table
valid_roi = cst['valid_roi'].values.astype(bool)
cell_ids = cst.index.values[valid_roi]
...
traces = np.empty((len(cell_ids), len(ts)), dtype=np.float32)
for i, cid in enumerate(cell_ids):
    traces[i] = col.loc[cid]
```

iii. The AI documented that the AllenSDK pipeline already applies motion correction, segmentation, ROI filtering, demixing, neuropil subtraction, dF/F normalization, and detrending. Z-scoring was benchmarked and added nothing (differences <= 0.02).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by `valid_roi == True` from the `cell_specimen_table`. The AI verified that all 29,097 cells in the dataset have `valid_roi == True`, so no neurons are actually removed.

ii.
```python
valid_roi = cst['valid_roi'].values.astype(bool)
cell_ids = cst.index.values[valid_roi]
if len(cell_ids) == 0:
    return {'oeid': int(oeid), 'skip': 'no valid ROIs'}
```

iii. The AI noted that upstream QC (ROI classifier, demixing, container-level QC) has already been applied to the released data, so `valid_roi` is True for all cells. The filter is kept as an explicit guard.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (`trials.start_time`). Ophys frames from `start_time` to `stop_time` are extracted using `np.searchsorted`. The alignment event is the ophys timestamp.

ii.
```python
starts = tr['start_time'].values.astype(np.float64)
stops = tr['stop_time'].values.astype(np.float64)
i0 = np.searchsorted(ts, starts, side='left')
i1 = np.searchsorted(ts, stops, side='left')
...
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The instructions say to "temporally align based on ophys timestamp." The AI uses `ophys_timestamps` as the common time base, extracting frames within each trial's `[start_time, stop_time)` window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native ophys frame rate (~30.94 Hz, ~32.32 ms per frame). No rebinning or resampling of the neural data is applied. The time bin size is computed from the median inter-frame interval.

ii.
```python
frame_periods = np.array([r['info']['frame_period_s'] for r in kept])
data['metadata'] = {
    'time_bin_size': float(np.median(frame_periods) * 1000.0),  # ms
    ...
}
```

iii. The AI verified that the ophys frame period is 0.032319 s (30.9406 Hz) and is identical across all sessions. No rebinning is needed.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name`, `start_time`, and `end_time` columns for each flash. The `stimulus_block_name` column is used to filter to change-detection blocks.

ii.
```python
def stimulus_traces(stimulus_presentations, ts, change_window='flash'):
    sp = stimulus_presentations
    sp = sp[sp['stimulus_block_name'].astype(str).str.contains('change_detection')]
    start = sp['start_time'].values.astype(np.float64)
    end = sp['end_time'].values.astype(np.float64)
    names = sp['image_name'].astype(str).values
    ...
    codes = np.array([IMAGE_CODE.get(n, 0) for n in names], dtype=np.int16)
    idx = np.searchsorted(start, ts, side='right') - 1
    valid = idx >= 0
    idx_c = np.clip(idx, 0, len(start) - 1)
    on_screen = valid & (ts < np.nan_to_num(end[idx_c], nan=-np.inf))
    image = np.where(on_screen, codes[idx_c], 0).astype(np.int16)
```

iii. The AI used the stimulus_presentations table to get precise per-frame image identity, including when the image is on screen vs. during grey inter-stimulus intervals. This gives 17 categories: 0 = grey (ISI and omitted flashes) + 16 natural images. The AI justified including grey as its own category since the spec mentions "image presented during the non-grey screen," implying grey periods need to be distinguished.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each ophys frame is assigned the image code of the flash it falls within (by checking if the frame time is within `[start_time, end_time)` of a flash). If the frame is outside any flash, it gets code 0 (grey). A global mapping of 16 image names to codes 1-16 is used, with code 0 reserved for grey.

ii.
```python
IMAGE_NAMES = ['im000', 'im031', 'im035', ...]  # 16 images
IMAGE_VALUES = ['grey'] + IMAGE_NAMES
IMAGE_CODE = {name: i + 1 for i, name in enumerate(IMAGE_NAMES)}
```

iii. The AI noted that the 16 image names span image sets A and B, and keeping them globally distinct (rather than per-session 0-7 indices) preserves the physical stimulus identity.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The `stimulus_traces` function computes image identity for every ophys timestamp in the session. Per-trial slicing uses the same frame indices as neural data.

ii.
```python
image_all, change_all, sp = stimulus_traces(ds.stimulus_presentations, ts, ...)
# In trial assembly:
out[0] = image_all[a:b]
```

iii. Since image identity is computed at every ophys frame time, alignment with neural data is guaranteed by using the same array indices.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column of `stimulus_presentations`. A frame gets `image_change = 1` only during a flash where `is_change` is True, and only while the image is on screen (the 250ms flash duration).

ii.
```python
is_change = sp['is_change'].values.astype(bool)
...
# default 'flash' mode:
change = (on_screen & is_change[idx_c]).astype(np.int16)
```

iii. The AI noted that catch trials have `is_sham_change` (not `is_change`), so they correctly get 0. The AI also benchmarked a 750ms interval window vs. the 250ms flash window, finding the flash window decoded slightly better (0.640 vs 0.624).

## 4-b. What processing is involved in computing `output` *Image change*?

i. Binary indicator: 1 on frames during a flash with `is_change == True`, 0 everywhere else. No additional processing.

ii. See 4-a code snippet.

iii. The AI used `stimulus_presentations.is_change` directly, which is already computed by the SDK.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is needed. The output values are `['no_change', 'change']`.

ii.
```python
OUTPUT_VALUES = [
    ...
    ['no_change', 'change'],
    ...
]
```

iii. N/A - already categorical.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - computed at every ophys frame time, then sliced per trial using the same indices as neural data.

ii.
```python
out[1] = change_all[a:b]
```

iii. Same frame-level alignment as all other streams.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides `timestamps` and `speed` (cm/s) at 60 Hz.

ii.
```python
def resample_running(running_speed, ts):
    rt = running_speed['timestamps'].values.astype(np.float64)
    rv = running_speed['speed'].values.astype(np.float64)
    ...
    return np.interp(ts, rt[order], rv[order])
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data, already unwrapped, despiked, and 10 Hz low-pass filtered.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native 60 Hz timestamps onto the ophys frame times using `np.interp`. NaN/Inf values are filtered before interpolation. Then discretized into 5 percentile bins computed per-session.

ii.
```python
good = np.isfinite(rt) & np.isfinite(rv)
rt, rv = rt[good], rv[good]
order = np.argsort(rt)
return np.interp(ts, rt[order], rv[order])
...
rb, run_edges = quantile_bins(run_all[frame_idx])
```

iii. The AI used `np.interp` (which handles edge cases by clamping) rather than `scipy.interpolate.interp1d`. Per-session quintile bins ensure the "five equal percentile bins" property holds within each session.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Discretized into 5 equal-percentile (quintile) bins computed per session. Bin edges are the 20th, 40th, 60th, 80th percentiles of running speed values within the session's kept trial timepoints.

ii.
```python
def quantile_bins(values, nbins=N_QUANTILE_BINS):
    edges = np.quantile(values, np.arange(1, nbins) / nbins)
    return np.searchsorted(edges, values, side='right').astype(np.int16), edges

frame_idx = np.concatenate([np.arange(i0[k], i1[k]) for k in sel])
rb, run_edges = quantile_bins(run_all[frame_idx])
```

iii. Per-session bins were chosen to ensure balanced quintiles within each session. The AI noted this prevents session identity from being encoded in the bin values.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the full ophys timestamp vector once per session. Trial slicing uses the same frame indices as neural data.

ii.
```python
run_all = resample_running(ds.running_speed, ts)
...
out[2] = run_bin_all[a:b]
```

iii. Same alignment approach as all other streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, specifically the `pupil_area` column. Blink frames (where `pupil_area` is NaN) are excluded before interpolation. The effective diameter is computed as `2 * sqrt(pupil_area / pi)`.

ii.
```python
def resample_pupil(eye_tracking, ts):
    t = eye_tracking['timestamps'].values.astype(np.float64)
    a = eye_tracking['pupil_area'].values.astype(np.float64)
    good = np.isfinite(t) & np.isfinite(a) & (a > 0)
    t, a = t[good], a[good]
    ...
    diam = 2.0 * np.sqrt(a / np.pi)
    return np.interp(ts, t, diam), t
```

iii. The AI used `pupil_area` (not `pupil_width`) and converted to diameter, noting that since the quintile transform is monotonic, it doesn't matter whether area or diameter is used for binning. `pupil_area` is NaN exactly where `likely_blink` is True.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples (NaN `pupil_area`) are removed. Valid `pupil_area` values are converted to effective diameter via `2*sqrt(area/pi)`. The resulting diameter is linearly interpolated onto the ophys timebase. Then discretized into 5 per-session quintile bins.

ii.
```python
good = np.isfinite(t) & np.isfinite(a) & (a > 0)
t, a = t[good], a[good]
order = np.argsort(t)
t, a = t[order], a[order]
diam = 2.0 * np.sqrt(a / np.pi)
return np.interp(ts, t, diam), t
```

iii. Blink removal before interpolation prevents blink artifacts. The area-to-diameter conversion is a monotonic transform. Per-session quintiles ensure the bins are balanced within each session.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal-percentile (quintile) bins computed per session from the kept trial timepoints.

ii.
```python
pb, pup_edges = quantile_bins(pupil_all[frame_idx])
```

iii. Per-session bins were chosen because pupil size is in camera pixels and depends on zoom/eye position/rig, making it not comparable across sessions. Global bins would mostly encode session identity rather than arousal.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed: interpolated onto the full ophys timestamp vector once per session, then sliced per trial.

ii.
```python
pupil_all, pupil_valid_t = resample_pupil(eye, ts)
...
out[3] = pup_bin_all[a:b]
```

iii. Same alignment approach as all other streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
outcome_cols = np.stack([tr['hit'].values.astype(bool),
                         tr['miss'].values.astype(bool),
                         tr['false_alarm'].values.astype(bool),
                         tr['correct_reject'].values.astype(bool)], axis=1)
assert outcome_cols.sum(axis=1).min() == 1 and outcome_cols.sum(axis=1).max() == 1
outcome = np.argmax(outcome_cols, axis=1).astype(np.int16)
```

iii. The four outcome columns are the SDK's canonical trial outcome labels. The AI verified they are mutually exclusive for go/catch trials (exactly one is True per trial).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four boolean outcome columns are stacked and `argmax` is used to get the integer code (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The code is constant across all timepoints within a trial.

ii.
```python
outcome = np.argmax(outcome_cols, axis=1).astype(np.int16)
...
out[4] = outcome[k]  # broadcast to all frames
```

iii. The AI includes an assertion that every go/catch trial has exactly one outcome. The outcome is stored as time-varying (constant per trial, broadcast to all frames) per the spec's guidance to make outputs time-varying when possible.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- Sessions without eye tracking: skipped entirely (3 sessions).
- Sessions with no valid ROIs: skipped.
- Trials with fewer than 2 ophys frames: dropped.
- Trials with no valid (non-blink) pupil measurement: dropped (60 trials total).
- Sessions with fewer than 2 usable trials after filtering: skipped.
- NaN/Inf in running speed or pupil timestamps: filtered before interpolation.

ii.
```python
if eye is None or len(eye) == 0 or not np.isfinite(eye['pupil_area'].values).any():
    return {'oeid': int(oeid), 'skip': 'no eye tracking data'}
...
if i1[k] - i0[k] < MIN_FRAMES_PER_TRIAL:
    n_drop_short += 1; continue
if j1[k] - j0[k] < 1:
    n_drop_pupil += 1; continue
```

iii. The AI documented all skipped sessions and dropped trials in CONVERSION_NOTES.md. The handling is robust and well-justified.

## 9-a. What are the most time-consuming steps of the code?

i. Opening the NWB files and materializing the neural trace matrices (~3-6 s per session per worker). The AI measured this and documented it in CONVERSION_NOTES.md Step 7.

ii. N/A (runtime measurement, not code logic)

iii. The AI noted that each NWB file is ~0.9 GB and that reading and materializing the dF/F matrix dominates runtime.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cell loop for extracting dF/F traces from the SDK's Series-of-arrays format (`for i, cid in enumerate(cell_ids): traces[i] = col.loc[cid]`) could potentially be vectorized. However, this is minor compared to I/O time.

ii.
```python
traces = np.empty((len(cell_ids), len(ts)), dtype=np.float32)
for i, cid in enumerate(cell_ids):
    traces[i] = col.loc[cid]
```

iii. The AI already vectorized the main bottleneck areas: trial boundary computation uses vectorized `searchsorted`, and all behavioral streams are resampled once per session rather than per trial.

## 9-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each session is processed once in `process_experiment()`. The only potential redundancy is that `quantile_bins` is computed within each session independently, but this is by design (per-session binning).

ii. N/A

iii. The AI used parallel processing (16 workers) to process sessions concurrently, with each session loaded and processed exactly once.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI collects extensive per-session metadata (`info` dict with 25+ fields) that is stored in `metadata['session_info']`. Some of this (e.g., `equipment_name`, `cre_line`, `behavior_session_id`) may not be used by the decoder but is useful for documentation and debugging.

ii.
```python
info = {
    'ophys_experiment_id': int(oeid),
    'ophys_session_id': int(md['ophys_session_id']),
    'behavior_session_id': int(md['behavior_session_id']),
    'mouse_id': str(md['mouse_id']),
    'cre_line': str(md['cre_line']),
    ...
}
```

iii. The extra metadata is small and does not affect processing time or output correctness. The AI also computes pupil blink fraction and other diagnostic statistics that are stored but not used by the decoder.
