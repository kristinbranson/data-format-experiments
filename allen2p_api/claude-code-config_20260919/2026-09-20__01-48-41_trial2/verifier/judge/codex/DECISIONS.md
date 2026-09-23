# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data through the AllenSDK local cache, not the S3 cache. It first enumerates locally present NWB files, filters the SDK experiment table down to those experiment IDs, and then further restricts to active `VisualBehavior` experiments. It then loads each selected experiment independently with `get_behavior_ophys_experiment()`, one experiment per worker process.

ii. 
```python
def get_cache():
    return VisualBehaviorOphysProjectCache.from_local_cache(
        cache_dir=CACHE_DIR, use_static_cache=False)

def locally_available_experiment_ids():
    files = os.listdir(NWB_DIR)
    return sorted(int(re.findall(r'(\d+)', f)[0]) for f in files
                  if f.endswith('.nwb'))

def select_experiments(cache):
    et = cache.get_ophys_experiment_table()
    sub = et.loc[locally_available_experiment_ids()]
    sel = sub[(~sub.passive) & (sub.project_code == 'VisualBehavior')]
    return sel.sort_index()
...
ds = cache.get_behavior_ophys_experiment(ophys_experiment_id)
```

iii. In `CONVERSION_NOTES.md`, the AI justifies this as decision `D1`/`D2`: restrict to the single-plane `VisualBehavior` variant for a constant ophys frame interval and exclude passive sessions because they do not have the active go/catch behavioral structure.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values from the converted experiment results.

ii.
```python
subjects, brain_regions = [], []
for r in ok:
    if r['mouse_id'] not in subjects:
        subjects.append(r['mouse_id'])
...
'subject_idx': np.array([subjects.index(r['mouse_id']) for r in ok],
                        dtype=np.int64),
```

iii. The AI’s notes consistently treat `mouse_id` as the subject identifier and report dataset statistics in terms of mice.

## 1-c. How are the data split into sessions?

i. The AI treats each selected `ophys_experiment_id` as one session. It does not reconstruct multi-experiment sessions by grouping on `ophys_session_id`; instead, it restricts to single-plane `VisualBehavior` experiments so that one experiment effectively serves as one session.

ii.
```python
meta = select_experiments(cache)
eids = list(meta.index)
...
def convert_experiment(ophys_experiment_id, collect_debug=False):
    ds = cache.get_behavior_ophys_experiment(ophys_experiment_id)
```

iii. `CONVERSION_NOTES.md` decision `D1` explicitly says the AI selected the single-plane `VisualBehavior` subset so that “one imaging plane (= one experiment) per session.”

## 1-d. How are the data split into trials?

i. Trials come from the AllenSDK `ds.trials` table. After selecting go/catch trials, the AI assigns each trial all ophys frames with `start_time <= t < stop_time`, so trial lengths are variable.

ii.
```python
trials = ds.trials
sel = trials[trials['go'].to_numpy().astype(bool)
             | trials['catch'].to_numpy().astype(bool)].sort_values('start_time')
...
a_idx = np.searchsorted(ts, sel['start_time'].to_numpy(dtype=float), side='left')
b_idx = np.searchsorted(ts, sel['stop_time'].to_numpy(dtype=float), side='left')
...
for i, (a, b) in enumerate(zip(a_idx, b_idx)):
    neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. `CONVERSION_NOTES.md` decision `D4` says the AllenSDK trial table is the authoritative trial definition and that the trial window should be `[trials.start_time, trials.stop_time)`.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only `go` or `catch` trials, drops trials shorter than 2 ophys frames, and indirectly drops all trials from sessions that fail eye-tracking requirements because those sessions are skipped entirely. It does not explicitly require `change_time.notna()` or a minimum of 2 trials per session.

ii.
```python
sel = trials[trials['go'].to_numpy().astype(bool)
             | trials['catch'].to_numpy().astype(bool)].sort_values('start_time')
...
keep = (b_idx - a_idx) >= 2
n_dropped_short = int((~keep).sum())
a_idx, b_idx = a_idx[keep], b_idx[keep]
...
if eye is None or len(eye) == 0:
    raise ValueError(f'{ophys_experiment_id}: no eye tracking data')
```

iii. The notes justify the go/catch filter using AllenSDK trial-class logic (`D2`, `D4`) and justify dropping sessions with no eye tracking as `D10` because pupil diameter is a required decoder output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data come from `ds.dff_traces['dff']`.

ii.
```python
dff = ds.dff_traces
cell_ids = dff.index.to_numpy()
traces = np.vstack(dff['dff'].to_numpy()).astype(np.float32)
```

iii. In `CONVERSION_NOTES.md`, decision `D3` says the AI deliberately chose `dff_traces` rather than detected events because the event representation was too sparse for per-timepoint decoding.

## 2-b. How is the `neural` data processed?

i. The AI stacks all dF/F traces for the experiment into one `(neurons, frames)` array, checks the trace length against `ophys_timestamps`, optionally applies the `valid_roi` mask, and then slices contiguous trial windows. No additional normalization or re-scaling is applied.

ii.
```python
traces = np.vstack(dff['dff'].to_numpy()).astype(np.float32)
if traces.shape[1] != n_frames:
    raise ValueError(...)
...
if not bool(cst['valid_roi'].all()):
    keep = cst['valid_roi'].to_numpy().astype(bool)
    traces = traces[keep]
...
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The notes describe dF/F as already processed by the Allen pipeline and say no extra neural processing was needed beyond selecting the signal and slicing trial-aligned windows.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI performs no new neuron-quality filter beyond defensively reapplying `cell_specimen_table.valid_roi` if necessary.

ii.
```python
cst = ds.cell_specimen_table
if not bool(cst['valid_roi'].all()):
    keep = cst['valid_roi'].to_numpy().astype(bool)
    traces = traces[keep]
    cell_ids = cell_ids[keep]
```

iii. `CONVERSION_NOTES.md` decision `D11` says the release already applied ROI and session QC upstream, so no additional neuron filtering was required.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to the native ophys timestamp grid. Each trial is the slice of ophys frames between trial `start_time` and `stop_time`; there is no separate re-alignment to `change_time`.

ii.
```python
ts = np.asarray(ds.ophys_timestamps, dtype=float)
a_idx = np.searchsorted(ts, sel['start_time'].to_numpy(dtype=float), side='left')
b_idx = np.searchsorted(ts, sel['stop_time'].to_numpy(dtype=float), side='left')
...
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The notes say all streams are already on the common session clock and define the alignment event as the ophys timestamp grid, with trial ownership defined by `[start_time, stop_time)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native ophys frame resolution, roughly 30.95 Hz (about 32.3 ms per frame), and does not rebin the neural data.

ii.
```python
ts = np.asarray(ds.ophys_timestamps, dtype=float)
...
'ophys_frame_interval_s': float(np.median(np.diff(ts))),
...
'time_bin_size': float(np.mean(dts) * 1000.0),
```

iii. `CONVERSION_NOTES.md` decision `D1` says the single-plane subset was chosen partly to keep a constant native ophys frame interval across sessions, so extra temporal rebinning was unnecessary.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The AI derives image identity from `ds.stimulus_presentations`, specifically the `image_name` values inside the `change_detection_behavior` block.

ii.
```python
sp = stimulus_presentations
sp = sp[sp['stimulus_block_name'] == STIM_BLOCK].sort_values('start_time')
names = sp['image_name'].to_numpy()
raw = np.array([IMAGE_TO_IDX.get(n, -1) if isinstance(n, str) else -1
                for n in names], dtype=np.int64)
```

iii. The notes justify this with decisions `D5` and `D6`: the output should follow the 750 ms image-presentation intervals used in the paper, and labels should be the real 16 image identities rather than per-session aliases.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI maps every ophys frame to the containing stimulus-presentation interval, converts image names to a fixed global 16-class vocabulary, and forward-fills omitted flashes so identity is held through omissions and gray periods.

ii.
```python
valid_pos = np.where(raw >= 0)[0]
fill_src = valid_pos[np.searchsorted(valid_pos,
                                     np.arange(len(raw)), side='right') - 1]
filled = raw[fill_src]
...
k = np.searchsorted(start, ophys_timestamps, side='right') - 1
k = np.clip(k, 0, len(start) - 1)
return filled[k], change[k].astype(np.int64), k
```

iii. `CONVERSION_NOTES.md` decision `D5` says image identity should be held through gray screen and omissions, and `D6` says the categories should be the 16 global image names.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated on the exact same ophys frame timestamps as the neural data. After mapping each frame to a stimulus interval, the trial slices use the same frame indices used for neural slicing.

ii.
```python
image_idx, is_change, interval = build_stimulus_series(ds.stimulus_presentations, ts)
...
out[0] = image_idx[a:b]
...
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The notes emphasize that all streams share the ophys session clock and that stimulus variables are evaluated on the same per-frame grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The AI derives image change from `ds.stimulus_presentations['is_change']` within the `change_detection_behavior` block.

ii.
```python
sp = sp[sp['stimulus_block_name'] == STIM_BLOCK].sort_values('start_time')
change = sp['is_change'].to_numpy().astype(bool)
...
return filled[k], change[k].astype(np.int64), k
```

iii. In `CONVERSION_NOTES.md`, decision `D7` says image change should mark the presentation interval that begins with a real change.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI converts `is_change` to an integer series and broadcasts that label to every ophys frame inside the corresponding stimulus-presentation interval.

ii.
```python
k = np.searchsorted(start, ophys_timestamps, side='right') - 1
k = np.clip(k, 0, len(start) - 1)
return filled[k], change[k].astype(np.int64), k
...
out[1] = is_change[a:b]
```

iii. The notes justify this as matching the paper’s 750 ms “image-presentation interval” unit rather than using only an instantaneous change point.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary and not thresholded from a continuous signal. The categories are `no_change` and `change`.

ii.
```python
CHANGE_NAMES = ['no_change', 'change']
...
OUTPUT_VALUES = [IMAGE_NAMES, CHANGE_NAMES, QUINTILE_NAMES, QUINTILE_NAMES,
                 OUTCOME_NAMES]
```

iii. The AI treats this output as an event label, not a continuous variable needing discretization.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is evaluated per ophys frame and then sliced into trials with the same frame indices as the neural data.

ii.
```python
image_idx, is_change, interval = build_stimulus_series(ds.stimulus_presentations, ts)
...
out[1] = is_change[a:b]
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The notes say the stimulus series is built directly on the ophys timebase, so alignment is inherited automatically.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `ds.running_speed['speed']` and `ds.running_speed['timestamps']`.

ii.
```python
rs = ds.running_speed
run_t = rs['timestamps'].to_numpy(dtype=float)
run_v = rs['speed'].to_numpy(dtype=float)
```

iii. The AI’s notes identify the SDK running-speed stream as the standard locomotion measure.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed onto ophys frame times, computes per-session quintile edges using only frames inside kept trials, and converts the per-frame values to bin indices.

ii.
```python
run_ok = np.isfinite(run_v)
run_frames = np.interp(ts, run_t[run_ok], run_v[run_ok])
...
frame_mask = np.zeros(n_frames, dtype=bool)
for a, b in zip(a_idx, b_idx):
    frame_mask[a:b] = True
run_edges = np.percentile(run_frames[frame_mask], QUANTILE_PCTS)
run_bin = np.searchsorted(run_edges, run_frames, side='right').astype(np.int64)
```

iii. `CONVERSION_NOTES.md` decision `D8` justifies per-session quintiles as a way to avoid between-session scale confounds and to keep all five classes balanced within each session.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The AI thresholds running speed into 5 percentile bins using the 20th, 40th, 60th, and 80th percentiles computed within each session’s kept trial frames.

ii.
```python
N_QUANTILES = 5
QUANTILE_PCTS = [20., 40., 60., 80.]
...
run_edges = np.percentile(run_frames[frame_mask], QUANTILE_PCTS)
run_bin = np.searchsorted(run_edges, run_frames, side='right').astype(np.int64)
```

iii. The notes explicitly call these “quintile bins” and defend the choice as balancing class frequencies session by session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is resampled to the ophys timestamps before any trial slicing, then trial windows index both signals with the same `a:b` frame range.

ii.
```python
run_frames = np.interp(ts, run_t[run_ok], run_v[run_ok])
...
out[2] = run_bin[a:b]
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The AI’s notes say all streams share the hardware-synchronized session clock, so interpolation onto `ophys_timestamps` is the alignment step.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter comes from `ds.eye_tracking['pupil_width']`, `ds.eye_tracking['pupil_height']`, and `ds.eye_tracking['timestamps']`.

ii.
```python
diam = 2.0 * np.maximum(eye_tracking['pupil_width'].to_numpy(dtype=float),
                        eye_tracking['pupil_height'].to_numpy(dtype=float))
t = eye_tracking['timestamps'].to_numpy(dtype=float)
```

iii. `CONVERSION_NOTES.md` decision `D9` says the SDK’s circular-area calculation implies a diameter of `2 * max(width, height)`, so the AI used both width and height rather than width alone.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI computes diameter as `2 * max(width, height)`, drops non-finite frames (including blink/outlier frames set to NaN by the SDK), interpolates the cleaned signal onto ophys frame times, and then discretizes it into per-session quintiles.

ii.
```python
eye_t, eye_d, eye_ok = pupil_diameter_series(eye)
if eye_ok.sum() < 100:
    raise ValueError(f'{ophys_experiment_id}: too few valid pupil frames')
pupil_frames = np.interp(ts, eye_t[eye_ok], eye_d[eye_ok])
...
pupil_edges = np.percentile(pupil_frames[frame_mask], QUANTILE_PCTS)
pupil_bin = np.searchsorted(pupil_edges, pupil_frames, side='right').astype(np.int64)
```

iii. The notes justify blink removal as `D9`, dropping sessions with no eye tracking as `D10`, and per-session quintiles as `D8`.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The AI thresholds pupil diameter into 5 per-session percentile bins using the 20th/40th/60th/80th percentiles of trial-covered frames.

ii.
```python
pupil_edges = np.percentile(pupil_frames[frame_mask], QUANTILE_PCTS)
pupil_bin = np.searchsorted(pupil_edges, pupil_frames, side='right').astype(np.int64)
```

iii. `CONVERSION_NOTES.md` decision `D8` says global pupil bins would leak session identity because pupil size is measured in camera pixels and varies substantially across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto `ophys_timestamps` first, then trial slices use the same ophys frame ranges as neural data.

ii.
```python
pupil_frames = np.interp(ts, eye_t[eye_ok], eye_d[eye_ok])
...
out[3] = pupil_bin[a:b]
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The AI’s notes say all streams share the same session clock, so interpolation onto the ophys grid is the alignment mechanism.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome comes from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome = np.full(len(sel), -1, dtype=np.int64)
for j, name in enumerate(OUTCOME_NAMES):
    outcome[sel[name].to_numpy().astype(bool)] = j
```

iii. The notes rely on AllenSDK trial semantics and state that go/catch trials map cleanly to those four outcome labels.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI encodes the four mutually exclusive outcomes as integers 0-3 and broadcasts the chosen label across all frames in the trial.

ii.
```python
out = np.empty((len(OUTPUT_NAMES), T), dtype=np.int64)
...
out[4] = outcome[i]
```

iii. `CONVERSION_NOTES.md` decision `D12` says trial outcome is static per trial, so it is stored as a time-constant output row.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI is mostly strict rather than imputation-heavy. It skips whole sessions on loader/conversion exceptions, drops sessions with empty eye-tracking tables or too few valid pupil frames, drops trials shorter than 2 frames, and defensively reapplies `valid_roi`. It does not keep NaNs around because it interpolates from finite behavioral samples.

ii.
```python
if eye is None or len(eye) == 0:
    raise ValueError(f'{ophys_experiment_id}: no eye tracking data')
if eye_ok.sum() < 100:
    raise ValueError(f'{ophys_experiment_id}: too few valid pupil frames')
...
keep = (b_idx - a_idx) >= 2
...
try:
    return convert_experiment(eid, collect_debug=collect_debug)
except Exception as exc:
    return {'error': f'{type(exc).__name__}: {exc}', 'eid': int(eid)}
```

iii. The notes explicitly justify skipping no-eye-tracking sessions as `D10`, blink removal as `D9`, and no extra neuron filtering as `D11`.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identifies loading each experiment/NWB through AllenSDK as the dominant cost and therefore parallelizes conversion across experiments.

ii.
```python
with Pool(min(args.workers, max(1, len(jobs)))) as pool:
    for i, res in enumerate(pool.imap(_worker, jobs, chunksize=1)):
        ...
```

iii. `CONVERSION_NOTES.md` Step 6 says “NWB load dominates” and reports multiprocessing as the main performance optimization.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the expensive per-frame stimulus/behavior mapping, but some smaller loops remain: building `frame_mask`, iterating over trials to append outputs, and list scans for subject/brain-region indexing.

ii.
```python
frame_mask = np.zeros(n_frames, dtype=bool)
for a, b in zip(a_idx, b_idx):
    frame_mask[a:b] = True
...
for i, (a, b) in enumerate(zip(a_idx, b_idx)):
    ...
subjects.index(r['mouse_id'])
brain_regions.index(r['brain_region'])
```

iii. The notes say the “first prototype” had Python loops for per-frame mappings and that these were replaced by `np.searchsorted`/`np.interp`; the remaining loops are smaller and secondary.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly reopens the AllenSDK cache inside every `convert_experiment()` call and repeatedly performs linear list searches when building `subject_idx` and `brain_region_idx`.

ii.
```python
def convert_experiment(ophys_experiment_id, collect_debug=False):
    cache = get_cache()
    ds = cache.get_behavior_ophys_experiment(ophys_experiment_id)
...
'subject_idx': np.array([subjects.index(r['mouse_id']) for r in ok],
                        dtype=np.int64),
'brain_region_idx': [
    np.full(r['n_neurons'], brain_regions.index(r['brain_region']),
            dtype=np.int64) for r in ok],
```

iii. The notes do not call these out as a problem, but they are a direct consequence of the AI’s per-experiment multiprocessing design.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter computes and returns `interval` from `build_stimulus_series()` even though it is unused in the conversion path, and it carries intermediate metadata such as `cell_specimen_ids` into per-session results even though those fields are not written into the final dataset consumed by the decoder.

ii.
```python
image_idx, is_change, interval = build_stimulus_series(ds.stimulus_presentations, ts)
...
result = {
    'neural': neural,
    'input': inputs,
    'output': output,
    ...
    'cell_specimen_ids': cell_ids,
    'info': info,
}
...
data = {
    'neural': [r['neural'] for r in ok],
    'input': [r['input'] for r in ok],
    'output': [r['output'] for r in ok],
```

iii. The notes focus on diagnostics and validation, not on trimming every intermediate. The default conversion path avoids the heaviest debug data, but some smaller unused intermediates remain.
