# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `VisualBehaviorOphysProjectCache.from_local_cache()` to open the AllenSDK cache, calls `get_ophys_experiment_table()` to list all experiments, then restricts to experiments whose NWB files exist locally (via glob matching). It filters to active-behavior experiments (excluding passive sessions). For each session, it loads all imaging planes via `cache.get_behavior_ophys_experiment(eid)`.

ii.
```python
def get_cache():
    return VisualBehaviorOphysProjectCache.from_local_cache(
        cache_dir=CACHE_DIR, use_static_cache=False)

def available_experiment_table(cache):
    et = cache.get_ophys_experiment_table()
    avail = sorted(int(os.path.basename(f).split('_')[-1].split('.')[0])
                   for f in glob.glob(NWB_GLOB))
    sub = et.loc[et.index.isin(avail)].copy()
    return sub

# In main():
active = et[~et['passive']].copy()
```

iii. The AI verified that `from_local_cache(use_static_cache=False)` was the correct API for the cache layout (Step 0/2 in trajectory). It filters to active experiments because passive sessions lack behavioral responses/trial outcomes. This is documented in CONVERSION_NOTES.md Step 2 and Step 4.

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values from the experiment metadata. Subjects are collected as sessions are processed and stored as string IDs.

ii.
```python
# In main() assembly loop:
if r['mouse_id'] not in subjects:
    subjects.append(r['mouse_id'])
data['subject_idx'].append(subjects.index(r['mouse_id']))

# In process_session:
mouse_id=str(ds0.metadata['mouse_id'])
```

iii. The `mouse_id` field is the SDK's unique identifier for each animal. The AI confirmed 38 mice in the local cache subset.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a unique `ophys_session_id`. Multiple imaging planes (experiments) sharing the same `ophys_session_id` are grouped together. Sessions are grouped via `active.groupby('ophys_session_id')`.

ii.
```python
groups = active.groupby('ophys_session_id')
session_ids = sorted(groups.groups.keys())
# ...
jobs = [(int(s), list(groups.get_group(s).index.values), region_map, ...)
        for i, s in enumerate(session_ids)]
```

iii. The AI documented in Step 4/5 that all imaging planes of a session share identical trial tables (empirically verified) and are merged into one session. This is consistent with the whitepaper definition of session vs experiment.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `dataset.trials` table. Go and catch trials are kept; aborted and auto-rewarded trials are excluded. Trials must have a valid (non-NaN) `change_time`. Each trial is represented as a fixed-length window of [-2.25, +3.75] seconds around the change time, divided into 24 bins of 250 ms each.

ii.
```python
trials = ds0.trials
sel = ((trials['go'] | trials['catch']) & (~trials['aborted'])
       & (~trials['auto_rewarded']))
trials = trials[sel]
trials = trials[~trials['change_time'].isna()]
if len(trials) < 2:
    out['error'] = 'fewer than 2 go/catch trials'
    return out
change_times = trials['change_time'].values.astype(np.float64)
edges_abs = change_times[:, None] + BIN_EDGES[None, :]  # (n_trials, N+1)
```

iii. The AI justified the window as: min(change_time - start_time) = 2.79 s across all experiments, and min(stop_time - change_time) = 4.20 s, so the [-2.25, +3.75] s window always lies within the trial boundaries. The 250 ms bin size matches the image presentation duration and divides the 750 ms flash cycle exactly.

## 1-e. How are trials filtered based on quality controls?

i. Multiple filtering stages: (1) Only go and catch trials kept (aborted and auto-rewarded excluded). (2) Trials with NaN `change_time` excluded. (3) Sessions with fewer than 2 valid trials excluded. (4) Trials with any NaN in pupil or running speed bins are dropped (blink gaps > 0.5 s that weren't interpolated). (5) Sessions without eye tracking data are excluded entirely.

ii.
```python
# Trial type filter
sel = ((trials['go'] | trials['catch']) & (~trials['aborted'])
       & (~trials['auto_rewarded']))

# Behavior completeness filter
keep = (~np.isnan(pupil).any(axis=1)) & (~np.isnan(running).any(axis=1))
n_dropped = int((~keep).sum())
if keep.sum() < 2:
    out['error'] = 'fewer than 2 trials with complete behaviour'
    return out

# No eye tracking
if len(eye) == 0:
    out['error'] = 'no eye tracking data'
    return out
```

iii. The AI justified dropping trials with incomplete behavior data because "fabricating pupil values for a decoded variable would corrupt the evaluation." This resulted in 2,071 dropped trials (4.6%) and 3 excluded sessions. Trial accounting was verified to sum exactly.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI's final code uses `dff_traces` (dF/F) by default, with an option (`--neural-signal`) to use `events` or `filtered_events`. The default was changed from `events` to `dff` based on empirical decoder comparison.

ii.
```python
if signal == 'dff':
    ev = np.vstack(ds.dff_traces['dff'].values).astype(np.float64)
elif signal == 'filtered_events':
    ev = np.vstack(ds.events['filtered_events'].values).astype(np.float64)
else:
    ev = np.vstack(ds.events['events'].values).astype(np.float64)
```

iii. The AI initially chose `events` (matching the paper: "For all analysis of neural data we used the detected calcium events") but switched to dF/F after empirical testing showed it decoded substantially better (e.g., image identity 0.529 vs 0.336). The AI justified this as: (1) dF/F is what the Allen SWDB reference code uses, (2) events are 98% zeros at 250 ms bins, discarding amplitude information, (3) dF/F retains slow components useful for running/pupil decoding.

## 2-b. How is the `neural` data processed?

i. Neural data from all imaging planes of a session are concatenated along the neuron axis. The data is then binned into 250 ms time bins. For dF/F, the bin value is the mean dF/F across ophys frames in each bin (using cumulative-sum-based vectorized binning). For events, the bin value is the sum.

ii.
```python
def bin_sum(values, timestamps, edges_abs):
    csum = np.concatenate([np.zeros((values.shape[0], 1), dtype=np.float64),
                           np.cumsum(values, axis=1)], axis=1)
    idx = np.searchsorted(timestamps, edges_abs.ravel())
    idx = idx.reshape(edges_abs.shape)
    take = csum[:, idx]
    return (take[:, :, 1:] - take[:, :, :-1]).astype(np.float32)

# For dF/F: average within the bin
if signal == 'dff':
    binned = binned / np.maximum(n_per_bin, 1)
```

iii. The cumulative-sum + searchsorted approach vectorizes all trials in a single call. The AI documented that every 250 ms bin contains at least 2 ophys frames (median 7), so no bins are empty.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. The Allen SDK's default (`exclude_invalid_rois=True`) already ensures only valid ROIs are included. The AI verified that 100% of returned ROIs have `valid_roi == True`.

ii. N/A (no filtering code beyond SDK defaults).

iii. The AI documented: "ROI filtering already applied by the Allen pipeline... the SDK returns only valid ROIs... No further neuron filtering is described in the papers."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the stimulus change time (`trials.change_time`) with a fixed window of [-2.25, +3.75] seconds. The window is divided into 24 bins of 250 ms each. For each trial, absolute bin edges are computed as `change_time + BIN_EDGES`.

ii.
```python
OFF_START = -2.25       # s relative to change time
OFF_END = 3.75          # s relative to change time
BIN_SIZE = 0.25         # s
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 24
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)

change_times = trials['change_time'].values.astype(np.float64)
edges_abs = change_times[:, None] + BIN_EDGES[None, :]
```

iii. The AI chose change_time alignment because it matches the SWDB reference code (`save_trial_response_df.py` aligns traces to `trials['change_time']`). The window [-2.25, +3.75] s was chosen to always lie within trial boundaries (verified: min change-start = 2.79 s, min stop-change = 4.20 s).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins to 250 ms time bins (24 bins per trial). This is different from the reference solution which keeps the native ophys frame rate (~11 Hz, ~93 ms for mesoscope).

ii.
```python
BIN_SIZE = 0.25         # s
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 24
```

iii. The AI justified 250 ms bins as: (1) it equals the image presentation duration, (2) it divides the 750 ms flash cycle exactly into 3 bins, (3) it is larger than the slowest ophys frame interval (93 ms mesoscope), enabling a common bin size across all sessions. The `time_bin_size` in metadata is set to 250.0 ms.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table (filtered to the change_detection block, non-omitted flashes). For each time bin, the image of the most recent non-omitted flash onset at or before the bin center is used.

ii.
```python
sp = ds0.stimulus_presentations
sp = sp[sp['stimulus_block_name'].astype(str).str.contains('change_detection')]
shown = sp[sp['image_name'] != 'omitted']
flash_start = shown['start_time'].values.astype(np.float64)
flash_image = shown['image_name'].values.astype(str)

# index of the most recent shown-image onset at or before each bin centre
j = np.searchsorted(flash_start, centers_abs, side='right') - 1
image_name = flash_image[j]  # (n_trials, N)
```

iii. The AI used stimulus_presentations rather than the trials table's `initial_image_name`/`change_image_name` columns. This approach directly ties each time bin to the actual stimulus shown, handling edge cases like omitted flashes naturally.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global sorted mapping of all unique image names across all sessions (16 images total from two image sets A and B).

ii.
```python
image_values = sorted({str(im) for r in good for im in np.unique(r['image_name'])})
image_to_idx = {im: i for i, im in enumerate(image_values)}
img = np.vectorize(lambda x: image_to_idx[str(x)])(r['image_name']).astype(np.int64)
```

iii. The AI chose 16 global image names rather than per-session 0-7 indices "because index i means different images in image sets A and B."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the center of each 250 ms bin, using the same bin edges derived from the change time. This ensures alignment with the neural data which is binned using the same edges.

ii.
```python
centers_abs = change_times[:, None] + BIN_CENTERS[None, :]
j = np.searchsorted(flash_start, centers_abs, side='right') - 1
image_name = flash_image[j]
```

iii. The bin centers are the same time points used for neural binning, so alignment is inherent.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table's `is_change` column and `start_time`. A bin is marked as 1 if its center falls within 750 ms of a change flash onset and the flash is a real change (`is_change == True`).

ii.
```python
flash_ischange = shown['is_change'].values.astype(bool)
j = np.searchsorted(flash_start, centers_abs, side='right') - 1
since = centers_abs - flash_start[j]
image_change = (flash_ischange[j] & (since < FLASH_CYCLE)).astype(np.int64)
```

iii. The AI noted that catch trials have `is_change == False`, so they naturally get image_change = 0. The 750 ms window corresponds to one full flash cycle (250 ms image + 500 ms grey).

## 4-b. What processing is involved in computing `output` *Image change*?

i. No additional processing beyond the binary indicator computation described in 4-a. The value is 1 during the 750 ms presentation interval of a change flash, 0 otherwise.

ii. See 4-a.

iii. N/A.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is applied. It is 1 in bins where the most recent flash was a real change and the bin center is within 750 ms of that flash onset.

ii. See 4-a.

iii. The AI verified: "image_change is 1 in exactly bins 9-11 (= [0, 0.75) s after the change) for all go trials and 0 for all catch trials."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same bin-center alignment as image identity and neural data, using the same change-time-relative bin structure.

ii. See 4-a.

iii. Same framework as all other outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides speed (cm/s) and timestamps from the running wheel encoder at ~60 Hz.

ii.
```python
run = ds0.running_speed
run_t = run['timestamps'].values.astype(np.float64)
run_v = run['speed'].values.astype(np.float64)
```

iii. The SDK's `running_speed` is the standard interface for locomotion data, already processed (outlier removal + lowpass Butterworth filter).

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each 250 ms bin using `bin_mean_nan` (cumulative sum with NaN handling), then discretized into 5 equal-percentile bins computed globally across all sessions.

ii.
```python
running = bin_mean_nan(run_v, run_t, edges_abs)  # (n_trials, N)

# Global quantile edges
all_run = np.concatenate([r['running'].ravel() for r in good])
run_edges = quantile_bins(all_run)  # interior edges at 20/40/60/80th percentiles

def quantile_bins(values, n_bins=N_QUANTILES):
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    return np.percentile(values, qs)

run_q = digitize_with(r['run_edges'], r['running'])
```

iii. The AI chose global percentile edges (vs per-session) based on both the literal spec ("five equal percentile bins") and empirical testing showing better decoder accuracy with global edges.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using `np.digitize` with 4 interior edges at the 20th, 40th, 60th, and 80th percentiles of all running speed values across the dataset.

ii.
```python
def quantile_bins(values, n_bins=N_QUANTILES):
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    return np.percentile(values, qs)

def digitize_with(edges, values):
    return np.digitize(values, edges).astype(np.int64)
```

iii. The AI verified that the resulting quintiles are each exactly 20% of the data globally.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is binned using the same absolute bin edges as neural data (change_time + BIN_EDGES), so alignment is inherent.

ii.
```python
running = bin_mean_nan(run_v, run_t, edges_abs)  # same edges_abs as neural
```

iii. Both neural and running data use the same bin structure derived from change times.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`'s `pupil_area` column, converted to diameter via `2 * sqrt(area / pi)`.

ii.
```python
eye = ds0.eye_tracking
eye_t = eye['timestamps'].values.astype(np.float64)
pupil_area = eye['pupil_area'].values.astype(np.float64)
pupil_diam = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. The AI chose `pupil_area` and converted to diameter rather than using `pupil_width` directly. The SDK's eye tracking table provides both `pupil_width` and `pupil_area`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) Pupil area is converted to diameter. (2) Blink NaN gaps shorter than 0.5 s are linearly interpolated using `interpolate_short_gaps`. (3) The diameter is averaged within each 250 ms bin. (4) It is discretized into 5 global equal-percentile bins. (5) Trials with any NaN bin (unresolved long blink gaps) are dropped entirely.

ii.
```python
pupil_diam = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diam = interpolate_short_gaps(eye_t, pupil_diam, PUPIL_MAX_INTERP_GAP)
pupil = bin_mean_nan(pupil_diam, eye_t, edges_abs)

# Trial dropping
keep = (~np.isnan(pupil).any(axis=1)) & (~np.isnan(running).any(axis=1))
```

iii. The AI justified: "Blinks are short" (median max gap ~18 s suggests most gaps are very short); 0.5 s threshold interpolates typical blink duration without filling long data gaps. Dropping trials with remaining NaN avoids fabricating pupil values for a decoded variable.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal-percentile bins using global edges computed from all valid pupil values across all sessions.

ii.
```python
all_pup = np.concatenate([r['pupil'].ravel() for r in good])
pup_edges = quantile_bins(all_pup)
pup_q = digitize_with(r['pup_edges'], r['pupil'])
```

iii. Same justification as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same bin-edge alignment as all other outputs, using change-time-relative absolute bin edges.

ii.
```python
pupil = bin_mean_nan(pupil_diam, eye_t, edges_abs)  # same edges_abs
```

iii. Same framework as neural and running data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']

outcome = np.full(n_trials, -1, dtype=np.int64)
for k, name in enumerate(OUTCOMES):
    outcome[trials[name].values.astype(bool)] = k
if np.any(outcome < 0):
    out['error'] = 'trial without an outcome label'
    return out
```

iii. The four outcome columns are mutually exclusive for go/catch trials. The AI verified this holds for all sessions.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) and broadcast as a constant value across all 24 time bins in a trial.

ii.
```python
out_trials.append(np.stack([
    img[t], r['image_change'][t], run_q[t], pup_q[t],
    np.full(N_BINS, r['outcome'][t], dtype=np.int64)]).astype(np.int64))
```

iii. The outcome is static per trial as specified in the instructions. Broadcasting to all time bins is required by the output format (all outputs share the same time dimension).

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Sessions without eye tracking**: 3 sessions excluded entirely (listed in metadata).
- **Blink NaN gaps in pupil**: Gaps <= 0.5 s linearly interpolated; longer gaps left as NaN.
- **Trials with incomplete behavior**: If any 250 ms bin has NaN pupil or running after processing, the trial is dropped (2,071 trials = 4.6%).
- **Sessions with too few trials**: Sessions with < 2 valid trials after filtering are excluded.
- **Session processing errors**: Wrapped in try/except; failed sessions logged and excluded.
- **Trials without outcome labels**: Would trigger an error (none found in practice).

ii.
```python
# Blink interpolation
pupil_diam = interpolate_short_gaps(eye_t, pupil_diam, PUPIL_MAX_INTERP_GAP)

# Behavior completeness filter
keep = (~np.isnan(pupil).any(axis=1)) & (~np.isnan(running).any(axis=1))

# No eye tracking
if len(eye) == 0:
    out['error'] = 'no eye tracking data'

# Exception handling
except Exception as exc:
    out['error'] = f'{exc!r}\n{traceback.format_exc()[-800:]}'
```

iii. The AI verified trial accounting sums exactly: 41,904 kept + 2,071 dropped + 917 in excluded sessions = 44,892 total go+catch trials.

## 9-a. What are the most time-consuming steps of the code?

i. Loading each experiment via `cache.get_behavior_ophys_experiment()` is the most time-consuming step, being I/O bound. The AI measured ~10 s/session wall time (serial) and used 16-way multiprocessing to parallelize, achieving ~0.5 s/session effective throughput.

ii. N/A (timing is reported in conversion output).

iii. The AI documented: "session loading/binning took 90 s (0.50 s/session)" with 16 workers for 174 sessions.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial assembly loop in `main()` iterates over trials to build output arrays. However, the binning itself is already fully vectorized using cumulative sums and searchsorted (all trials of a session binned in one call).

ii.
```python
# Vectorized binning (no per-trial loop):
binned = bin_sum(ev, ts, edges_abs)  # (n_cells, n_trials, N)

# Per-trial assembly loop (could be vectorized but not a bottleneck):
for t in range(n_trials):
    neural_trials.append(np.ascontiguousarray(r['neural'][:, t, :]))
```

iii. Data loading dominates runtime, so the per-trial assembly loop is not a bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. The code does not repeat significant processing. Each session is loaded once, processed once, and the results stored. Global discretization edges are computed once from all accumulated data. The image-to-index mapping is computed once globally.

ii. N/A.

iii. The AI avoided redundant processing by design (one pass through sessions, then one global discretization pass).

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `--show-processing` mode stores debug data (raw ophys timestamps, raw events, raw running/pupil traces, flash times, lick/reward times) which is only used for plotting and not needed for the final output. When `--show-processing` is not used, this data is still collected for debug-enabled sessions (first 2), though it adds minimal overhead since `n_debug = 0` when the flag is off.

ii.
```python
if want_debug:
    out['debug'] = dict(
        ophys_timestamps=..., events0=..., run_t=..., run_v=...,
        eye_t=..., pupil_diam=..., flash_start=..., flash_image=...,
        flash_ischange=..., keep=..., lick_times=..., reward_times=...)
```

iii. This debug data enables visual verification of the conversion but is not included in the final pickle output.
