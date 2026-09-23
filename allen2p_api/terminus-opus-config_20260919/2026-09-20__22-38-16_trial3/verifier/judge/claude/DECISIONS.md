# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data is loaded via the AllenSDK `VisualBehaviorOphysProjectCache.from_local_cache()`. The AI scans the local NWB files on disk to determine which experiments are available, then filters the experiment table to only those present. It includes **all active (non-passive) sessions** from both `VisualBehavior` (single-plane, ~31 Hz) and `VisualBehaviorMultiscope` (multi-plane, ~11 Hz) project codes. Each experiment is loaded via `cache.get_behavior_ophys_experiment(eid)`.

ii.
```python
def get_cache():
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    return VisualBehaviorOphysProjectCache.from_local_cache(
        cache_dir=CACHE_DIR, use_static_cache=False)

def downloaded_experiment_table(cache):
    et = cache.get_ophys_experiment_table()
    ids = sorted(int(os.path.basename(f).split('_')[-1].split('.')[0])
                 for f in glob.glob(NWB_GLOB))
    return et.loc[et.index.isin(ids)].copy()

# In main():
act = et[~et['passive'].astype(bool)].copy()
```

iii. The AI identified that the local cache contains only a subset of the full release (284 of 1936 experiments). It scans for NWB files to restrict to actually available data. It filters by `passive == False` to exclude passive sessions (OPHYS_2/5) where the lick spout is retracted and trial outcomes are undefined. The AI explicitly chose to include both single-plane and multi-plane sessions, noting that they represent the same behavioral task.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `mouse_id` field from the experiment metadata. Each unique mouse_id becomes a subject.

ii.
```python
mouse = info['mouse_id']
if mouse not in subjects:
    subjects.append(mouse)
data['subject_idx'].append(subjects.index(mouse))
```

iii. The `mouse_id` field is the SDK's unique identifier for each animal. The AI assembles subjects during the output assembly phase from per-session metadata.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a unique `ophys_session_id`. Multiple imaging planes (experiments) recorded simultaneously within the same session are grouped together by `ophys_session_id` and their neurons are concatenated along the neuron axis.

ii.
```python
groups = act.groupby('ophys_session_id')
session_ids = sorted(groups.groups.keys())
# ...
for i, sid in enumerate(session_ids):
    exp_ids = list(groups.get_group(sid).index.values)
```

iii. Grouping by `ophys_session_id` ensures that simultaneously recorded imaging planes form a single population recording. This is documented in CONVERSION_NOTES.md as a key decision.

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `trials` table from the Allen SDK. The AI keeps rows where `go | catch` is True (equivalent to non-aborted, non-auto-rewarded). Each trial is aligned to `change_time` with a fixed window of [-3, +3] seconds, divided into 24 bins of 250 ms each. All trials have the same fixed length (24 time bins).

ii.
```python
trials = ref.trials
keep = (trials['go'].astype(bool) | trials['catch'].astype(bool))
tr = trials[keep].copy()
tr = tr[np.isfinite(tr['change_time'].values)]

# Fixed window around change_time
edges = change_times[:, None] + (OFF_START + BIN_SIZE * np.arange(NBINS + 1))[None, :]
```

iii. The AI justified the [-3, +3]s window as encompassing exactly 8 flash cycles (4 before and 4 after the change), verified that the minimum inter-change interval (7.51s) prevents window overlap, and confirmed that the window lies within every trial's start_time to stop_time range (min 3.02s before change, min 4.20s after).

## 1-e. How are trials filtered based on quality controls?

i. Several filters are applied: (1) Only `go | catch` trials are kept (aborted and auto-rewarded excluded). (2) Trials with non-finite `change_time` are dropped. (3) Trials whose [-3, +3]s window falls outside the range of ophys/running/eye/stimulus data are dropped. (4) Trials with no valid outcome (none of hit/miss/false_alarm/correct_reject set) are dropped. (5) Sessions with fewer than 2 valid trials are dropped. (6) Sessions with empty eye-tracking data are dropped (3 sessions).

ii.
```python
keep = (trials['go'].astype(bool) | trials['catch'].astype(bool))
tr = trials[keep].copy()
tr = tr[np.isfinite(tr['change_time'].values)]
# ...
valid = ((change_times + OFF_START) >= t_lo) & ((change_times + OFF_END) <= t_hi)
tr = tr[valid]
# ...
outcome = np.full(ntrials, -1, dtype=np.int64)
# ... assign outcomes ...
ok = outcome >= 0
# ... drop trials with outcome == -1 ...
if ntrials < 2:
    return None
```

iii. The AI documented that 0 trials were dropped due to window clipping in the full run, and 3 sessions were dropped due to empty eye-tracking tables (verified individually). The filtering logic ensures all required data streams are available for every retained trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces` (dF/F calcium fluorescence traces) from each experiment, accessed via `exp.dff_traces`. The AI initially planned to use L0-detected calcium `events` (as stated in the paper) but switched to dF/F after empirical testing showed that the extreme sparsity of event traces (~0.3% nonzero) made an instantaneous per-bin decoder fail.

ii.
```python
tbl = e.events if neural_source in ('events', 'filtered_events') else e.dff_traces
col = {'events': 'events', 'filtered_events': 'filtered_events',
       'dff': 'dff'}[neural_source]
traces = np.vstack(tbl[col].values)  # (ncells, T)
```

iii. The AI conducted a controlled comparison across three neural signals (events, filtered_events, dF/F) on 6 sessions and found dF/F consistently outperformed the others for this decoder architecture. The decision was documented as a deviation from the paper, justified by the mismatch between the paper's random-forest decoder (concatenating 400ms windows) and the provided instantaneous linear decoder.

## 2-b. How is the `neural` data processed?

i. Neural data from multiple imaging planes is concatenated along the neuron axis. The dF/F traces are then averaged within 250 ms time bins using a vectorized cumulative-sum + searchsorted approach (`bin_means` function). The binned values are stored as float32.

ii.
```python
def bin_means(values, timestamps, edges):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    ntrials, nedge = edges.shape
    idx = np.searchsorted(timestamps, edges.ravel(), side='left').reshape(ntrials, nedge)
    counts = np.diff(idx, axis=1)
    csum = np.concatenate([np.zeros((values.shape[0], 1)), np.cumsum(values, axis=1)], axis=1)
    sums = csum[:, idx[:, 1:]] - csum[:, idx[:, :-1]]
    with np.errstate(invalid='ignore', divide='ignore'):
        out = sums / counts[None, :, :]
    out[np.broadcast_to(counts[None, :, :] == 0, out.shape)] = 0.0
    return out, counts
```

iii. The 250 ms bin size was chosen to align with the stimulus structure (1/3 of the 750 ms flash cycle) and to provide a common temporal resolution across the two different frame rates (31 Hz single-plane and 11 Hz multi-plane).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to the neural data beyond the SDK defaults. The SDK's `exclude_invalid_rois=True` (default) already filters out non-valid ROIs.

ii. N/A (relies on SDK defaults)

iii. The AI verified that the Allen pipeline's ROI-filtering classifier and session-level QC (saturation, photobleaching, z-drift, motion, d-prime) have already been applied to the released data. No extra filtering was deemed necessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to `change_time` (the image change on go trials, or the sham change on catch trials). A symmetric window of [-3, +3] seconds around `change_time` is used, divided into 24 bins of 250 ms each. The ophys frames falling within each bin are averaged.

ii.
```python
edges = change_times[:, None] + (OFF_START + BIN_SIZE * np.arange(NBINS + 1))[None, :]
# ...
binned, counts = bin_means(traces, ts, edges)  # (ncells, ntrials, nbins)
```

iii. The AI verified that `change_time` always coincides with a stimulus flash onset (100% of change_times are in `stimulus_presentations.start_time`). The +/-3s window was chosen to capture 4 flash cycles before and after the change.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned to 250 ms time bins (24 bins per trial). This is a temporal rebinning from the native ophys frame rate (~31 Hz for single-plane, ~11 Hz for multi-plane). Neural traces, running speed, and pupil diameter are all averaged within each 250 ms bin.

ii.
```python
BIN_SIZE = 0.25           # s
OFF_START = -3.0          # s relative to the change
OFF_END = 3.0             # s relative to the change
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))  # 24
```

iii. The 250 ms bin size was chosen because: (1) it equals the image presentation duration and is exactly 1/3 of the 750 ms flash cycle, so bin edges align with image onsets/offsets; (2) it is >= 2 ophys frames even on the 11 Hz rig, ensuring every bin contains data; (3) it provides a common temporal resolution for both single-plane and multi-plane sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name` column. The AI uses the `start_time` and `image_name` fields from stimulus presentations in the `change_detection_behavior` stimulus block. Each time bin is labeled with the image of the most recent flash presentation whose onset preceded the bin center.

ii.
```python
sp = ref.stimulus_presentations
beh = sp[sp['stimulus_block_name'] == BEHAVIOR_BLOCK].copy()
pres_start = beh['start_time'].values.astype(np.float64)
pres_image = beh['image_name'].values.astype(object)
# ...
pidx = np.searchsorted(pres_start, centers.ravel(), side='right') - 1
pidx = np.clip(pidx, 0, len(pres_start) - 1)
img_names = pres_image[pidx]
img_identity = np.array([IMAGE_TO_IDX[n] for n in img_names],
                        dtype=np.int64).reshape(ntrials, NBINS)
```

iii. The AI used the flash-level stimulus_presentations table rather than the trial-level initial_image_name/change_image_name columns. This captures the actual image shown at each time point, including during the grey ISI (labeled with the current 750 ms presentation interval's image) and omitted flashes (labeled as 'omitted'). The AI documented this as following the paper's convention of "assigning behavioural events to each image presentation interval."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a fixed global mapping of 17 categories: 16 natural images (from image sets A and B) plus 'omitted'. The mapping is defined as a constant in the code. Each bin center is assigned to the most recent flash onset via searchsorted, and the corresponding image name is converted to its integer code.

ii.
```python
IMAGE_NAMES = ['im000', 'im031', 'im035', 'im045', 'im054', 'im061', 'im062', 'im063',
               'im065', 'im066', 'im069', 'im073', 'im075', 'im077', 'im085', 'im106']
IMAGE_VALUES = IMAGE_NAMES + ['omitted']
IMAGE_TO_IDX = {n: i for i, n in enumerate(IMAGE_VALUES)}
```

iii. The 16 images come from the two image sets (A and B) used across the Visual Behavior project. The 'omitted' category captures omission events (5% of non-change flashes) where the grey screen is shown instead of an image.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the center of each 250 ms bin. The bin center is matched to the most recent stimulus presentation onset via `np.searchsorted(pres_start, centers, side='right') - 1`. This ensures the same temporal grid as the neural data.

ii.
```python
centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
pidx = np.searchsorted(pres_start, centers.ravel(), side='right') - 1
img_names = pres_image[pidx]
```

iii. By computing image identity at bin centers and using the same bin grid as the neural data, temporal alignment is guaranteed. The bin grid is defined relative to `change_time`, which is itself a flash onset.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column of the `stimulus_presentations` table. Each bin is assigned the `is_change` value of the most recent flash presentation preceding the bin center.

ii.
```python
pres_change = beh['is_change'].astype(bool).values
# ...
img_change = pres_change[pidx].astype(np.int64).reshape(ntrials, NBINS)
```

iii. The `is_change` field in stimulus_presentations is True only when the image identity actually changes (i.e., on go trials, not catch/sham changes). This naturally handles the distinction between go and catch trials.

## 4-b. What processing is involved in computing `output` *Image change*?

i. No processing beyond looking up the `is_change` value for the flash presentation corresponding to each bin center. The value is cast to int64 (0 or 1).

ii.
```python
img_change = pres_change[pidx].astype(np.int64).reshape(ntrials, NBINS)
```

iii. The `is_change` field directly provides the binary indicator. For go trials, the 3 bins spanning the 750 ms change-flash interval (bins 12-14, i.e., t=0 to t=0.75s) are labeled 1. For catch trials, all bins are 0.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change) and requires no thresholding.

ii. N/A

iii. The binary nature comes directly from the `is_change` field.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same bin-center-to-flash-onset alignment as image identity, using the same `pidx` index array.

ii. (Same as 3-c)

iii. Alignment is guaranteed by using the same temporal grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, which provides speed (cm/s) and timestamps from the running wheel encoder, at ~60 Hz, low-pass filtered by the Allen pipeline.

ii.
```python
run = ref.running_speed
run_t = run['timestamps'].values.astype(np.float64)
run_v = interpolate_nans(run['speed'].values.astype(np.float64), run_t)
```

iii. The SDK's `running_speed` attribute provides pre-processed running data (10 Hz low-pass Butterworth filtered, per the whitepaper).

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is first NaN-interpolated, then averaged within each 250 ms bin using `bin_means`, and finally discretized into 5 equal-percentile (quintile) bins. Quintile edges are computed **per session**.

ii.
```python
run_v = interpolate_nans(run['speed'].values.astype(np.float64), run_t)
run_binned = bin_means(run_v, run_t, edges)[0][0]  # (ntrials, nbins)
run_q = quantile_bin(run_binned)

def quantile_bin(values, nq=NQUANTILES):
    flat = values.ravel()
    edges = np.percentile(flat, np.linspace(0, 100, nq + 1)[1:-1])
    return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. Per-session quintiles were chosen because running propensity varies greatly between mice/sessions, ensuring exactly 20% of bins per class within every session and preventing the decoder from simply reading out session identity.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins (quintiles) computed per session. The percentile edges are at 20%, 40%, 60%, 80%.

ii.
```python
edges = np.percentile(flat, np.linspace(0, 100, nq + 1)[1:-1])
return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. Equal-percentile binning ensures balanced class counts. Per-session computation accounts for inter-session variability in running behavior.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is averaged over the same 250 ms bins as the neural data, using the same bin edge grid derived from `change_time`.

ii.
```python
run_binned = bin_means(run_v, run_t, edges)[0][0]
```

iii. Using `bin_means` with the same `edges` array as the neural data guarantees temporal alignment. The Allen sync clock ensures the running timestamps are on the same time base.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`, using the `pupil_area` column. The area is converted to an equivalent circular diameter: `diameter = 2 * sqrt(area / pi)`.

ii.
```python
eye = ref.eye_tracking
pupil_d = 2.0 * np.sqrt(eye['pupil_area'].values.astype(np.float64) / np.pi)
pupil_d = interpolate_nans(pupil_d, eye_t)
```

iii. The AI chose `pupil_area` and converted to diameter rather than using `pupil_width` directly. `pupil_area` captures the full ellipse fit (pi * semi_major * semi_minor), and converting to an equivalent circular diameter provides a single scalar measure. Blink frames (NaN in `pupil_area`) are linearly interpolated.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is NaN-interpolated (blink frames removed), averaged within each 250 ms bin, and discretized into 5 per-session quintile bins. Sessions with empty eye-tracking tables are dropped entirely.

ii.
```python
pupil_d = interpolate_nans(pupil_d, eye_t)
if pupil_d is None:
    return None
pupil_binned = bin_means(pupil_d, eye_t, edges)[0][0]
pupil_q = quantile_bin(pupil_binned)
```

iii. Same per-session quintile approach as running speed. Sessions with no eye tracking data (3 sessions) are dropped because a pupil_diameter output cannot be defined.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal-percentile (quintile) bins per session.

ii. (Same `quantile_bin` function as 5-c)

iii. Per-session quintiles account for the camera-dependent pupil measurements (pixel units depend on rig/geometry).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed: bin-averaged over the same 250 ms grid using `bin_means`.

ii.
```python
pupil_binned = bin_means(pupil_d, eye_t, edges)[0][0]
```

iii. Same alignment mechanism as all other data streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
OUTCOME_VALUES = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome = np.full(ntrials, -1, dtype=np.int64)
outcome[tr['hit'].astype(bool).values] = 0
outcome[tr['miss'].astype(bool).values] = 1
outcome[tr['false_alarm'].astype(bool).values] = 2
outcome[tr['correct_reject'].astype(bool).values] = 3
```

iii. These four columns are the SDK's canonical trial outcome labels. They are mutually exclusive for go and catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The code is constant across all time bins within a trial (broadcast over the 24 bins). Trials with no valid outcome (code=-1) are dropped.

ii.
```python
outcome_bins = np.repeat(outcome[:, None], NBINS, axis=1)
# In output assembly:
output_list = [np.stack([img_identity[i], img_change[i], run_q[i], pupil_q[i],
                         outcome_bins[i]], axis=0) for i in range(ntrials)]
```

iii. The static per-trial label is broadcast to match the time-varying output format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiment loads**: Try/except around `get_behavior_ophys_experiment` skips unloadable experiments.
- **Missing eye tracking**: Sessions with `eye is None or len(eye) == 0` are dropped (3 sessions).
- **Blink frames**: NaN values in `pupil_area` (from blinks) are linearly interpolated over time; if fewer than 2 valid frames remain, the session is dropped.
- **NaN running speed**: Linearly interpolated; sessions with fewer than 2 valid values are dropped.
- **Non-finite change_time**: Trials are dropped.
- **Trials outside data range**: Trials whose [-3,+3]s window falls outside available data are dropped.
- **Invalid outcomes**: Trials where none of hit/miss/FA/CR is set are dropped.
- **Empty bins**: If a 250 ms bin contains no ophys frames, it is set to 0.

ii.
```python
run_v = interpolate_nans(run['speed'].values.astype(np.float64), run_t)
if run_v is None:
    return None
# ...
if eye is None or len(eye) == 0:
    return None
pupil_d = interpolate_nans(pupil_d, eye_t)
if pupil_d is None:
    return None
# ...
out[np.broadcast_to(counts[None, :, :] == 0, out.shape)] = 0.0
```

iii. The AI documented all dropped sessions and trials in CONVERSION_NOTES.md with counts (0 trials dropped to window clipping, 3 sessions dropped for missing eye tracking, 0 trials dropped for invalid outcomes in the full run).

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `cache.get_behavior_ophys_experiment()`, which reads large NWB files from disk. This is I/O bound. The AI addressed this by parallelizing with a ProcessPoolExecutor (24 workers).

ii.
```python
with ProcessPoolExecutor(max_workers=args.workers) as ex:
    futs = {ex.submit(_worker, j): j[0] for j in jobs}
```

iii. Full conversion completed in ~55 seconds wall clock using 24 workers.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the main computational bottleneck: the `bin_means` function uses cumulative sums and searchsorted to bin all neurons x trials at once in O(T + nbins) instead of looping over trials and bins. The remaining per-trial loop (assembling output arrays) is lightweight.

ii.
```python
def bin_means(values, timestamps, edges):
    csum = np.concatenate([np.zeros((values.shape[0], 1)), np.cumsum(values, axis=1)], axis=1)
    sums = csum[:, idx[:, 1:]] - csum[:, idx[:, :-1]]
```

iii. The AI identified and implemented the key vectorization. The per-trial output assembly loop is not a bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. Each worker process creates its own `VisualBehaviorOphysProjectCache` instance (the `get_cache()` call inside `process_session`), which involves re-reading the manifest and experiment table. This is repeated for every session processed.

ii.
```python
def process_session(session_id, experiment_ids, ...):
    t0 = time.time()
    cache = get_cache()  # repeated for each worker/session
```

iii. This is a consequence of using multiprocessing (each worker needs its own cache object). The overhead is small compared to NWB loading.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extensive per-session diagnostic information in `metadata['session_info']` (running speed ranges, pupil diameter ranges, load times, blink fractions, trial counts by type, etc.). The processing plots (`--show-processing`) also involve extra computation. Additionally, the code computes `cell_ids` for each session but these are not included in the final output dictionary.

ii.
```python
info = dict(
    ophys_session_id=int(session_id),
    # ... many diagnostic fields ...
    running_speed_range=[float(run_binned.min()), float(run_binned.max())],
    pupil_diameter_range=[float(pupil_binned.min()), float(pupil_binned.max())],
    load_time_s=t_load, total_time_s=time.time() - t0,
)
```

iii. The diagnostic information is useful for quality control and debugging but is not used by the downstream decoder.
