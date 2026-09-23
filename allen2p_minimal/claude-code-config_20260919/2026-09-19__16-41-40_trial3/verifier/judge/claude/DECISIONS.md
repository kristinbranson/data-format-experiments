# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading NWB files directly from disk using `BehaviorOphysExperiment.from_nwb_path()`, rather than using the Allen SDK's S3 cache. It reads the experiment metadata CSV from the project metadata directory, filters to available NWB files on disk, and then filters to the `VisualBehavior` project code. Each experiment is converted independently (and in parallel using `ProcessPoolExecutor`), cached to a per-experiment pickle, and then assembled into the final output.

ii.
```python
def select_experiments():
    exp = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    available = set()
    for fname in os.listdir(NWB_DIR):
        m = re.match(r'behavior_ophys_experiment_(\d+)\.nwb$', fname)
        if m:
            available.add(int(m.group(1)))
    exp = exp[exp.ophys_experiment_id.isin(available)]
    exp = exp[exp.project_code == 'VisualBehavior']
    exp = exp[~exp.passive.astype(bool)]
    return exp.sort_values('ophys_experiment_id').reset_index(drop=True)

# Per-experiment loading:
ds = BehaviorOphysExperiment.from_nwb_path(nwb_path)
```

iii. The AI chose to load NWB files directly rather than using the S3 cache for efficiency and to avoid network dependencies. Experiments were processed in parallel with `ProcessPoolExecutor` for speed. The AI explicitly filters to `VisualBehavior` project code (single-plane) and also filters out passive sessions.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values extracted from experiment metadata. Unique mouse IDs are collected from all converted sessions and sorted.

ii.
```python
subjects = sorted({s['mouse_id'] for s in sessions})
subject_to_idx = {m: i for i, m in enumerate(subjects)}
```

iii. The `mouse_id` field is the SDK's unique identifier for each animal.

## 1-c. How are the data split into sessions?

i. Each experiment (ophys_experiment_id) is treated as a separate session. Since VisualBehavior experiments are single-plane, each session has exactly one experiment/imaging plane, so experiment == session. Passive sessions are explicitly excluded (`~exp.passive`). Sessions are sorted by `ophys_experiment_id` for deterministic ordering.

ii.
```python
exp = exp[exp.project_code == 'VisualBehavior']
exp = exp[~exp.passive.astype(bool)]
# ...
sessions.sort(key=lambda s: s['ophys_experiment_id'])
```

iii. The AI justifies that for the single-plane VisualBehavior variant, there is one imaging plane per ophys session, so "session == experiment" is unambiguous. Passive sessions are excluded because the lick spout is retracted, making trial outcomes undefined. The AI also noted that both familiar and novel image sets are kept, unlike the reference paper which restricted to familiar images.

## 1-d. How are the data split into trials?

i. Trials are defined using the `trials` table from the SDK. Each trial spans from `start_time` to `stop_time` (variable length). Only go and catch trials are kept; aborted and auto-rewarded trials are dropped. Trials with fewer than 2 frames or with `None` image labels (before first flash) are additionally dropped.

ii.
```python
trials = ds.trials
keep = ((trials['go'].astype(bool) | trials['catch'].astype(bool))
        & ~trials['aborted'].astype(bool)
        & ~trials['auto_rewarded'].astype(bool))
trials = trials[keep]
# ...
starts = np.searchsorted(ts, trials['start_time'].values, side='left')
stops = np.searchsorted(ts, trials['stop_time'].values, side='left')
for k in range(len(trials)):
    i0, i1 = int(starts[k]), int(stops[k])
    if i1 - i0 < 2:
        continue
    if np.any(image_name[i0:i1] == None):
        continue
```

iii. The AI follows the instructions to include Go and Catch trials and exclude Aborted and Auto-rewarded. The additional filter on `image_name == None` removes trials that start before the first stimulus flash. The minimum 2-frame requirement prevents degenerate trials.

## 1-e. How are trials filtered based on quality controls?

i. Multiple filters are applied: (1) Only go/catch trials kept (aborted and auto-rewarded dropped); (2) Trials with fewer than 2 ophys frames are dropped; (3) Trials containing frames before the first stimulus flash (image_name is None) are dropped; (4) Trials with invalid outcome codes (not hit/miss/false_alarm/correct_reject) are dropped; (5) Experiments with fewer than 2 valid trials are dropped entirely; (6) Experiments without eye-tracking data are dropped (3 experiments); (7) Passive sessions are excluded.

ii.
```python
keep = ((trials['go'].astype(bool) | trials['catch'].astype(bool))
        & ~trials['aborted'].astype(bool)
        & ~trials['auto_rewarded'].astype(bool))
# ...
if i1 - i0 < 2:
    continue
if np.any(image_name[i0:i1] == None):
    continue
# ...
if np.any(outcome < 0):
    ok = outcome >= 0
    trials = trials[ok]
    outcome = outcome[ok]
    if len(trials) < 2:
        return None
# ...
pt, pv = _pupil_diameter(ds.eye_tracking)
if pt is None:
    return None  # drop experiment without pupil data
```

iii. The AI applies more quality filters than the reference, including dropping experiments without eye-tracking data (required for the pupil output) and dropping trials with frames before the first flash (to ensure every frame has a valid image label). These are reasonable defensive measures.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `dff_traces` (detrended dF/F calcium fluorescence traces) from each experiment. The AI explicitly considered and rejected using deconvolved calcium `events` (which the reference paper uses), because at the per-frame (32ms) resolution required by the decoder, events are too sparse.

ii.
```python
if TRACE == 'dff':
    src = ds.dff_traces
    col = 'dff'
else:
    src = ds.events
    col = TRACE
cell_ids = src.index.values
traces = np.vstack([np.asarray(v, dtype=np.float32)
                    for v in src[col].values])
```

iii. The AI performed an empirical comparison: with deconvolved events, 3.8% of trials contain no neural activity at all, and decoder accuracy drops across all outputs (e.g., image identity 0.40 -> 0.21). The AI justifies this as a legitimate departure from the reference paper required by the task of training a per-frame neural decoder. The environment variable `VB_TRACE` can switch between dff and events.

## 2-b. How is the `neural` data processed?

i. The dF/F traces are loaded and stacked into a (n_neurons, n_timepoints) array. NaN values are replaced with zeros using `np.nan_to_num`. No additional normalization, filtering, or smoothing is applied. Since VisualBehavior has one imaging plane per session, there is no multi-plane merging.

ii.
```python
traces = np.vstack([np.asarray(v, dtype=np.float32)
                    for v in src[col].values])
assert traces.shape[1] == ts.shape[0]
np.nan_to_num(traces, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The Allen SDK pipeline already applies motion correction, segmentation, demixing, neuropil subtraction, dF/F normalization, and detrending. The `nan_to_num` call is a defensive measure against occasional gaps in the traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to neurons. All ROIs in the released NWB files are included (all have `valid_roi == True` from the Allen pipeline).

ii. N/A (no filtering code)

iii. The AI notes that the Allen pipeline has already applied ROI filtering, demixing, and neuropil-correction QC, and that the reference paper applies no further neuron-level curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the ophys timestamps. For each trial, frames from `start_time` to `stop_time` are extracted using `np.searchsorted` on the ophys timestamps. The trial window is variable-length.

ii.
```python
ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
starts = np.searchsorted(ts, trials['start_time'].values, side='left')
stops = np.searchsorted(ts, trials['stop_time'].values, side='left')
for k in range(len(trials)):
    i0, i1 = int(starts[k]), int(stops[k])
    neural_trials.append(np.ascontiguousarray(traces[:, i0:i1]))
```

iii. `np.searchsorted` with `side='left'` finds the first ophys frame at or after the boundary time. The full trial window (start_time to stop_time) is used rather than a fixed window, preserving both pre-change and post-change periods.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Data is kept at the native ophys frame rate (~31 Hz, ~32.32 ms per frame). The time bin size is computed as the median inter-frame interval across all sessions.

ii.
```python
dts = np.array([s['dt'] for s in sessions])
time_bin_size = float(np.median(dts) * 1000.0)
```

Each experiment records its own dt:
```python
'dt': float(np.median(np.diff(ts))),
```

iii. The ophys timestamps have a consistent frame rate determined by the microscope. All data streams are aligned to the ophys timebase, so no resampling of neural data is needed.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name`, `start_time`, and `is_change` columns. The AI uses the `_flash_labels()` function which assigns every ophys frame to the "image presentation interval" it falls in, following the reference paper's convention that an interval is the 750ms beginning with each image onset.

ii.
```python
stim = ds.stimulus_presentations
if 'stimulus_block_name' in stim.columns:
    stim = stim[stim['stimulus_block_name'].astype(str)
                .str.contains('change_detection')]
else:
    stim = stim[stim['active'].astype(bool)]
stim = stim.sort_values('start_time')
image_name, is_change = _flash_labels(stim, ts)
```

iii. The AI chose the stimulus_presentations table over the trials table for image identity because it provides per-flash granularity and handles omitted flashes. The AI explicitly references the paper's convention: "By image presentation interval we refer to the 750 ms interval beginning with each image presentation."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The `_flash_labels()` function uses `np.searchsorted` to find the most recent flash onset before each ophys frame, then labels the frame with that flash's image name. Image names are then mapped to integer codes via a global sorted mapping, with 'omitted' placed last. This creates 17 categories (8 images from set A + 8 from set B + omitted).

ii.
```python
def _flash_labels(stim, ts):
    starts = stim['start_time'].values
    idx = np.searchsorted(starts, ts, side='right') - 1
    valid = idx >= 0
    idx_clipped = np.where(valid, idx, 0)
    names = stim['image_name'].values.astype(object)
    changes = stim['is_change'].values.astype(bool)
    image_name = np.where(valid, names[idx_clipped], None)
    is_change = np.where(valid, changes[idx_clipped], False)
    return image_name, is_change

# Assembly:
images = sorted({name for s in sessions for tr in s['image_name']
                 for name in np.unique(tr)} - {'omitted'})
image_values = images + ['omitted']
image_to_idx = {name: i for i, name in enumerate(image_values)}
```

iii. The approach follows the reference paper's convention for image presentation intervals. 'Omitted' flashes (where the image is suppressed) get their own category and are placed last in the vocabulary. Regular image names are sorted alphabetically for deterministic ordering.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed for every ophys frame in the full session, then sliced to the trial window using the same start/stop indices as the neural data. Alignment is inherent since both use ophys frame indices.

ii.
```python
image_name, is_change = _flash_labels(stim, ts)
# ...
image_trials.append(image_name[i0:i1].copy())
# Assembly:
img = np.array([image_to_idx[n] for n in s['image_name'][k]], dtype=np.int64)
```

iii. Since image labels are computed on the ophys timebase, they are automatically aligned with neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the `stimulus_presentations` table. The `_flash_labels()` function propagates the `is_change` flag to every ophys frame within the corresponding flash interval. Only go trials (real changes) have `is_change == True`; catch trials have sham changes.

ii.
```python
changes = stim['is_change'].values.astype(bool)
is_change = np.where(valid, changes[idx_clipped], False)
# ...
change_trials.append(is_change[i0:i1].astype(np.int64))
```

iii. Using `is_change` from stimulus_presentations directly captures the change event at the flash level. The signal is 1 throughout the 750ms image presentation interval of the changed flash (since `_flash_labels` assigns each frame to its most recent flash, and the change flag persists for that interval).

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` boolean from stimulus_presentations is propagated to ophys frames via the same `_flash_labels()` mechanism used for image identity. It is then cast to int64 (0 or 1). No additional processing or windowing is applied beyond what `_flash_labels` provides.

ii.
```python
is_change = np.where(valid, changes[idx_clipped], False)
# ...
chg = s['is_change'][k].astype(np.int64)
```

iii. The change indicator naturally spans one 750ms flash interval because `_flash_labels` assigns each frame to its most recent flash onset.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed.

ii.
```python
['no_change', 'change']  # output_values for image_change
```

iii. N/A - already binary from the source data.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - computed on the full ophys timebase and sliced to trial windows using the same frame indices as neural data.

ii. See 3-c and 4-a code snippets.

iii. Same frame-level alignment mechanism as all other outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides speed and timestamps from the running wheel encoder.

ii.
```python
run = ds.running_speed
run_t = np.asarray(run['timestamps'].values, dtype=np.float64)
run_v = np.asarray(run['speed'].values, dtype=np.float64)
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps (~60 Hz) to the ophys timebase using `np.interp`. Non-finite values are excluded before interpolation. It is then discretized into 5 equal percentile bins (quintiles) with bin edges computed from the pooled distribution across all sessions.

ii.
```python
good = np.isfinite(run_t) & np.isfinite(run_v)
running = np.interp(ts, run_t[good], run_v[good])
# ...
qs = np.linspace(0, 100, NBINS + 1)[1:-1]  # [20, 40, 60, 80]
run_edges = np.percentile(run_all, qs)
# ...
run = np.searchsorted(run_edges, s['running_speed'][k], side='right').astype(np.int64)
```

iii. Linear interpolation resamples to the ophys timebase. `np.interp` extrapolates edge values rather than producing NaN. Percentile-based binning ensures roughly equal class counts. Bin edges are global across all sessions for consistent categories.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (0-4) using `np.searchsorted` against the 20th, 40th, 60th, and 80th percentile edges computed from the pooled dataset distribution. Values below the 20th percentile go to bin 0, above the 80th percentile to bin 4.

ii.
```python
qs = np.linspace(0, 100, NBINS + 1)[1:-1]  # [20, 40, 60, 80]
run_edges = np.percentile(run_all, qs)
run = np.searchsorted(run_edges, s['running_speed'][k], side='right').astype(np.int64)
```

iii. The 5 bins correspond to quintiles of the pooled distribution. Using `np.searchsorted` with `side='right'` ensures boundary values are handled consistently.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the ophys timebase before trial segmentation, so it shares the same time indices as the neural data. The same trial window (i0:i1) is used to extract both.

ii.
```python
running = np.interp(ts, run_t[good], run_v[good])
# ...
running_trials.append(running[i0:i1].astype(np.float64))
```

iii. By interpolating running speed onto `ophys_timestamps` upfront, alignment is guaranteed.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` in the `eye_tracking` table. The AI computes diameter from the fitted pupil ellipse area as `2*sqrt(area/pi)`. Blink frames are excluded (where `pupil_area` is NaN, corresponding to SDK's `likely_blink` flagging).

ii.
```python
def _pupil_diameter(eye_tracking):
    t = eye_tracking['timestamps'].values
    area = eye_tracking['pupil_area'].values.astype(float)
    good = np.isfinite(area) & np.isfinite(t) & (area > 0)
    if good.sum() < 100:
        return None, None
    return t[good], 2.0 * np.sqrt(area[good] / np.pi)
```

iii. The AI chose to derive diameter from `pupil_area` rather than using `pupil_width` directly, arguing that the area-derived diameter "is more robust than either single ellipse axis." The AI notes that the SDK already sets `pupil_area` to NaN on blink frames.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is converted to diameter (`2*sqrt(area/pi)`), blink frames (NaN/non-positive area) are removed, then the signal is linearly interpolated to the ophys timebase using `np.interp`. It is then discretized into 5 percentile bins with global edges, same as running speed.

ii.
```python
pt, pv = _pupil_diameter(ds.eye_tracking)
pupil = np.interp(ts, pt, pv)
# ...
pup_edges = np.percentile(pup_all, qs)
pup = np.searchsorted(pup_edges, s['pupil_diameter'][k], side='right').astype(np.int64)
```

iii. Same approach as running speed. Blink removal before interpolation prevents artifacts.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 quintile bins using `np.searchsorted` against the 20th/40th/60th/80th percentile edges from the pooled distribution.

ii. See 5-c (same mechanism).

iii. Same as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed - interpolated to the ophys timebase before trial segmentation, sharing the same frame indices as neural data.

ii.
```python
pupil = np.interp(ts, pt, pv)
# ...
pupil_trials.append(pupil[i0:i1].astype(np.float64))
```

iii. Same alignment mechanism as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome = np.full(len(trials), -1, dtype=np.int64)
for i, name in enumerate(OUTCOME_NAMES):
    outcome[trials[name].astype(bool).values] = i
```

iii. These four outcomes are mutually exclusive for go/catch trials. Trials with unresolved outcomes (outcome == -1) are dropped.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The integer code is broadcast across all time bins within the trial (time-varying representation of a per-trial value).

ii.
```python
outcome_trials.append(int(outcome[k]))
# ...
out = np.full(T, s['trial_outcome'][k], dtype=np.int64)
sess_output.append(np.stack([img, chg, run, pup, out], axis=0))
```

iii. The code broadcasts the per-trial outcome across all frames, making it time-varying as the format spec encourages ("If at all possible, make it time-varying").

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Failed experiments**: If `convert_experiment` throws an exception, the experiment is skipped and logged.
- **Missing eye-tracking**: Experiments without eye-tracking data (or with < 100 valid pupil frames) are dropped entirely.
- **NaN in neural traces**: Replaced with 0 via `np.nan_to_num`.
- **NaN in running/pupil**: `np.interp` extrapolates edge values (no NaN produced), so missing data at edges gets nearest valid value.
- **Short trials**: Trials with fewer than 2 frames are dropped.
- **Pre-flash frames**: Trials with frames before the first stimulus flash are dropped.
- **Invalid outcomes**: Trials without a valid hit/miss/FA/CR label are dropped.
- **Few trials**: Experiments with fewer than 2 valid trials are dropped entirely.

ii.
```python
# Failed experiments
try:
    return eid, convert_experiment(eid), None
except Exception:
    return eid, None, traceback.format_exc()
# Missing eye-tracking
if pt is None:
    return None
# NaN in neural
np.nan_to_num(traces, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
# Running/pupil: np.interp extrapolates
running = np.interp(ts, run_t[good], run_v[good])
# Pre-flash frames
if np.any(image_name[i0:i1] == None):
    continue
# Invalid outcomes
if np.any(outcome < 0):
    ok = outcome >= 0
    trials = trials[ok]
```

iii. The AI is defensive about missing data. The most notable decision is dropping entire experiments when eye-tracking is absent (3 experiments), which is necessary because pupil diameter is a required output.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB experiment via `BehaviorOphysExperiment.from_nwb_path()`, which reads large neural and behavioral data arrays from disk. The AI mitigates this with parallel processing (`ProcessPoolExecutor`) and per-experiment caching.

ii.
```python
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    futs = [pool.submit(_worker, e) for e in eids]
```

iii. Each NWB file contains full-session dF/F traces, running speed, eye tracking, and trials data. Parallel processing and caching significantly speed up repeated runs.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_experiment` iterates sequentially over trials to slice neural, image, running, and pupil data. The `np.searchsorted` for trial boundaries is already vectorized across all trials, but the subsequent slicing and filtering loop could potentially be vectorized. The image-to-index mapping in assembly (`[image_to_idx[n] for n in ...]`) could be vectorized with a lookup array.

ii.
```python
for k in range(len(trials)):
    i0, i1 = int(starts[k]), int(stops[k])
    if i1 - i0 < 2:
        continue
    # ... slice and append
```

iii. The trial loop is not a bottleneck since NWB file I/O dominates runtime. The loop is simple and readable.

## 9-c. What processing does the code repeat multiple times?

i. The code loads and processes each experiment only once (cached to disk). In the assembly phase, each cached experiment is loaded once. However, the `_flash_labels` function is called once per experiment, and the image vocabulary is built by iterating over all sessions' image names twice (once for the sorted set, once during assembly).

ii. No significant repeated processing.

iii. The caching design ensures experiments are not re-processed on subsequent runs.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code collects and stores per-session metadata (ophys_experiment_id, behavior_session_id, cre_line, equipment_name, imaging_depth, ophys_frame_rate, etc.) in `session_info` within metadata. While useful for documentation, this is not used by the decoder. The code also stores cell_specimen_ids per experiment in the cache, which are not carried into the final output. The `_edge_labels` function creates human-readable bin labels that are stored in output_values but not used by the decoder.

ii.
```python
session_info.append({
    'ophys_experiment_id': s['ophys_experiment_id'],
    'ophys_session_id': s['ophys_session_id'],
    'behavior_session_id': s['behavior_session_id'],
    # ... etc.
})
```

iii. The extra metadata is useful for provenance and debugging but not consumed by the decoder.
