# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the experiment metadata CSV from disk (`ophys_experiment_table.csv`), filters to available NWB files in the experiment directory, filters by `project_code == 'VisualBehavior'` and excludes passive sessions. Each session's NWB file is loaded directly via `BehaviorOphysExperiment.from_nwb_path()` rather than through the S3 cache. Processing is parallelized across sessions using `ProcessPoolExecutor`.

ii.
```python
et = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
available = set()
for fn in os.listdir(EXPERIMENT_DIR):
    if fn.endswith('.nwb'):
        available.add(int(fn.split('_')[-1].split('.')[0]))
et = et[et.ophys_experiment_id.isin(available)]
et = et[(et.project_code == PROJECT_CODE) & (~et.passive.astype(bool))]
...
ds = BehaviorOphysExperiment.from_nwb_path(path)
```

iii. The AI chose to load NWB files directly from disk rather than using the S3 cache, which is appropriate for local data. Passive sessions were excluded because the lick spout is retracted and trial outcome is behaviorally undefined. Parallel processing was used for efficiency.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata table, sorted as strings.

ii.
```python
subjects = sorted(table.mouse_id.astype(str).unique().tolist())
subject_idx = np.array([subjects.index(str(m)) for m in table.mouse_id],
                        dtype=np.int64)
```

iii. The `mouse_id` field is the SDK's unique identifier for each animal.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a single ophys experiment (one imaging plane). Because the AI filters to `VisualBehavior` project code (single-plane imaging), there is one experiment per session. The AI asserts this: `assert et.ophys_session_id.nunique() == len(et)`.

ii.
```python
et = et[(et.project_code == PROJECT_CODE) & (~et.passive.astype(bool))]
assert et.ophys_session_id.nunique() == len(et)
et = et.sort_values('ophys_experiment_id').reset_index(drop=True)
```

iii. The AI reasoned that for VisualBehavior (single-plane) experiments, each session has exactly one imaging plane, so experiment == session. This avoids the complexity of merging multi-plane data.

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `ds.trials` table. Only go and catch trials are kept; aborted and auto-rewarded trials are excluded. Trials must have a valid `change_time`. Each trial uses a **fixed window of [-2, +2] seconds around `change_time`**, yielding T=124 ophys frames per trial (at ~31 Hz).

ii.
```python
OFF_START = -2.0
OFF_END = 2.0
...
keep = ((trials.go.astype(bool) | trials.catch.astype(bool))
        & ~trials.aborted.astype(bool)
        & ~trials.auto_rewarded.astype(bool)
        & trials.change_time.notna())
trials = trials[keep]
...
change_time = float(tr.change_time)
i0 = int(np.searchsorted(ts, change_time + OFF_START, side='left'))
if i0 + T > len(ts):
    n_dropped_edge += 1
    continue
tt = ts[i0:i0 + T]
```

iii. The AI verified that every go/catch trial has at least 3.0s before and 4.2s after the change time, so the [-2, +2]s window always fits inside the experiment-defined trial. This produces fixed-length trials (T=124 frames), which simplifies downstream batching for the decoder.

## 1-e. How are trials filtered based on quality controls?

i. Aborted trials, auto-rewarded trials, and trials without a valid `change_time` are excluded. Trials that would extend past the recording edge are dropped. Trials without a valid outcome (not hit/miss/false_alarm/correct_reject) are dropped. Sessions with fewer than 2 usable trials are excluded entirely.

ii.
```python
keep = ((trials.go.astype(bool) | trials.catch.astype(bool))
        & ~trials.aborted.astype(bool)
        & ~trials.auto_rewarded.astype(bool)
        & trials.change_time.notna())
...
if i0 + T > len(ts):
    n_dropped_edge += 1
    continue
...
if len(neural) < 2:
    return {'oeid': oeid, 'error': f'only {len(neural)} usable trials'}
```

iii. The AI followed the instructions to exclude aborted and auto-rewarded trials. Edge cases (recording boundary, invalid outcome) are handled by dropping individual trials. Sessions with <2 trials are excluded to enable decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `filtered_events` in the SDK's events table (`ds.events`), which are the L0-regularized deconvolved calcium events with half-Gaussian filtering applied.

ii.
```python
events = ds.events
neural_all = np.stack(events['filtered_events'].values).astype(np.float32)
```

iii. The AI chose `filtered_events` over `dff_traces` based on consistency with the reference paper (Piet et al.), which uses detected calcium events for all neural analysis. The AI ran empirical comparisons showing that `dff_traces` actually yields higher decoding accuracy, but chose `filtered_events` for methodological consistency. The half-Gaussian filtered version was chosen over raw events because raw events are nonzero in only ~0.1% of 32ms bins.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied beyond stacking the `filtered_events` arrays. Each session has a single imaging plane, so no merging across planes is needed.

ii.
```python
neural_all = np.stack(events['filtered_events'].values).astype(np.float32)
...
neural.append(neural_all[:, i0:i0 + T])
```

iii. The AI noted that the SDK pipeline already applies motion correction, neuropil subtraction, event detection, and half-Gaussian filtering. No further processing was deemed necessary.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons present in the released NWB files are included.

ii. N/A (no filtering code)

iii. The AI noted that the released NWB files only contain ROIs that passed the Allen QC pipeline (valid_roi == True), so no further neuron curation is applied. The frame rate is also validated to be within 30-32 Hz.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the **change time** (onset of the changed image flash on go trials, onset of the sham-change flash on catch trials). A fixed window of [-2, +2] seconds around this event is taken, yielding T=124 consecutive ophys frames.

ii.
```python
OFF_START = -2.0
OFF_END = 2.0
...
change_time = float(tr.change_time)
i0 = int(np.searchsorted(ts, change_time + OFF_START, side='left'))
...
tt = ts[i0:i0 + T]
neural.append(neural_all[:, i0:i0 + T])
```

iii. The AI verified that `change_time` coincides exactly with stimulus presentation onset times (difference = 0.0 for both go and catch trials). The fixed window ensures all trials have the same number of time bins.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native ophys frame rate (~31 Hz, ~32.3 ms per bin) is used. All sessions have the same frame rate, so T=124 bins per trial is consistent.

ii.
```python
T = int(round((OFF_END - OFF_START) / dt))
...
'time_bin_size': float(np.mean(dts) * 1000.0),
```

iii. The AI verified that all VisualBehavior sessions run at exactly 31 Hz, so the time bin size is consistent without rebinning.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name` and `start_time` columns of non-omitted stimulus presentations.

ii.
```python
sp = ds.stimulus_presentations
sp = sp[sp.stimulus_block_name == 'change_detection_behavior']
sp = sp.sort_values('start_time')
shown = sp[~sp.omitted.astype(bool)]
flash_start = shown.start_time.values.astype(np.float64)
flash_image = shown.image_name.values.astype(str)
```

iii. The AI used the stimulus_presentations table rather than the trials table to get frame-by-frame image identity. This allows the identity to be held over the entire 750ms image-presentation cycle (250ms flash + 500ms gray) and correctly handles omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each ophys frame in the trial, the most recent flash onset is found via searchsorted, and the corresponding image name is assigned. Image names are then mapped to integer codes via a global sorted mapping.

ii.
```python
k = np.searchsorted(flash_start, tt, side='right') - 1
if np.any(k < 0):
    n_dropped_edge += 1
    continue
image_name.append(flash_image[k])
...
images = sorted({str(im) for s in sessions for im in np.unique(s['image_name'])})
image_to_idx = {im: i for i, im in enumerate(images)}
img = np.vectorize(image_to_idx.__getitem__)(s['image_name'])
```

iii. Using searchsorted against flash onset times correctly assigns the identity even during gray screens and omitted flashes. A global sorted mapping ensures consistent integer codes across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed per ophys frame within the trial window, using the same time points (`tt = ts[i0:i0+T]`) as the neural data.

ii.
```python
tt = ts[i0:i0 + T]
k = np.searchsorted(flash_start, tt, side='right') - 1
image_name.append(flash_image[k])
```

iii. Both neural data and image identity are indexed by the same ophys frame times, ensuring perfect temporal alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table, specifically the `start_time` of presentations where `is_change == True`.

ii.
```python
change_start = sp[sp.is_change.astype(bool)].start_time.values.astype(np.float64)
```

iii. Using `is_change` from stimulus_presentations identifies real image changes. Sham changes on catch trials are not marked as `is_change`, so they correctly get value 0.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each ophys frame, the code checks whether the frame falls within 750ms (one image-presentation cycle) of the most recent image change. If so, the value is 1; otherwise 0.

ii.
```python
k = np.searchsorted(change_start, tt, side='right') - 1
ch = np.zeros(T, dtype=np.int16)
valid = k >= 0
ch[valid] = (tt[valid] - change_start[k[valid]] < IMAGE_CYCLE)
change.append(ch)
```

iii. The 750ms window corresponds to one full image-presentation cycle (250ms image + 500ms gray). The AI noted that labeling only the 250ms flash scored higher in a probe, but chose 750ms for consistency with the image-identity encoding and the paper's convention.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is a binary variable: 1 during the 750ms presentation cycle of a real image change, 0 otherwise. No additional thresholding is applied.

ii.
```python
ch[valid] = (tt[valid] - change_start[k[valid]] < IMAGE_CYCLE)
```

iii. The comparison `< IMAGE_CYCLE` directly produces a boolean that is cast to int16.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed per ophys frame using the same time points as the neural data.

ii. See 4-b code snippet — uses the same `tt` array derived from ophys timestamps.

iii. Alignment is guaranteed by using the same ophys frame times.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed`, specifically the `timestamps` and `speed` columns.

ii.
```python
run = ds.running_speed
run_t = run.timestamps.values.astype(np.float64)
run_v = run.speed.values.astype(np.float64)
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto ophys frame times using `np.interp`, then discretized into 5 equal-count percentile bins computed globally across all sessions.

ii.
```python
running.append(np.interp(tt, run_t, run_v))
...
all_running = np.concatenate([s['running'].ravel() for s in sessions])
run_edges = percentile_bins(all_running, NBINS_BEHAVIOR)
...
run_bin = np.searchsorted(run_edges, s['running'], side='right')
```

iii. Linear interpolation resamples to the ophys timebase. Global percentile binning ensures consistent categories across sessions since the decoder shares one output head.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using percentile-based edges computed globally. The `percentile_bins` function computes only interior edges (4 edges for 5 bins), and `np.searchsorted` assigns bin indices 0-4.

ii.
```python
def percentile_bins(values, nbins):
    qs = np.linspace(0, 100, nbins + 1)[1:-1]
    return np.percentile(values, qs)
...
run_bin = np.searchsorted(run_edges, s['running'], side='right')
```

iii. Equal-count percentile bins ensure balanced class distribution for the decoder.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same ophys frame times used for the neural data within each trial window.

ii.
```python
running.append(np.interp(tt, run_t, run_v))
```

iii. `np.interp` resamples running speed to the exact ophys frame times `tt`, ensuring alignment with the neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` in the `eye_tracking` table, converted to diameter using `2 * sqrt(pupil_area / pi)`. Blink frames (where `likely_blink` is True) and non-finite values are excluded before interpolation.

ii.
```python
eye = ds.eye_tracking
pupil_t = eye.timestamps.values.astype(np.float64)
pupil_d = 2.0 * np.sqrt(eye.pupil_area.values.astype(np.float64) / np.pi)
good = (np.isfinite(pupil_t) & np.isfinite(pupil_d)
        & ~eye.likely_blink.values.astype(bool))
pupil_t, pupil_d = pupil_t[good], pupil_d[good]
```

iii. The AI derived diameter from area rather than using `pupil_width` directly. This gives the diameter of a disc with the fitted pupil area. Blinks are removed before interpolation to prevent artifacts.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. After blink removal, pupil diameter is linearly interpolated onto ophys frame times via `np.interp`, then discretized into 5 equal-count percentile bins computed globally across all sessions.

ii.
```python
pupil.append(np.interp(tt, pupil_t, pupil_d))
...
all_pupil = np.concatenate([s['pupil'].ravel() for s in sessions])
pupil_edges = percentile_bins(all_pupil, NBINS_BEHAVIOR)
pupil_bin = np.searchsorted(pupil_edges, s['pupil'], side='right')
```

iii. Same approach as running speed: linear interpolation to the ophys timebase, then global percentile binning.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 bins using global percentile-based edges, with `np.searchsorted` assigning bin indices 0-4.

ii.
```python
pupil_edges = percentile_bins(all_pupil, NBINS_BEHAVIOR)
pupil_bin = np.searchsorted(pupil_edges, s['pupil'], side='right')
```

iii. Equal-count percentile bins ensure balanced class distribution.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same ophys frame times used for neural data.

ii.
```python
pupil.append(np.interp(tt, pupil_t, pupil_d))
```

iii. Same alignment approach as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
if tr.hit:
    o = 0
elif tr.miss:
    o = 1
elif tr.false_alarm:
    o = 2
elif tr.correct_reject:
    o = 3
else:
    n_dropped_edge += 1
    image_name.pop(); change.pop(); running.pop(); pupil.pop()
    continue
outcome.append(o)
```

iii. These four outcomes are the canonical trial types for the go/no-go change detection task. Trials not matching any outcome are dropped.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The code is constant across all time bins within a trial.

ii.
```python
np.full(T, s['outcome'][i], dtype=np.int16),
```

iii. The mapping order matches `OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed sessions**: If extraction throws an exception, the error is recorded and the session is skipped.
- **Edge trials**: Trials where the [-2,+2]s window would extend past the recording are dropped.
- **Invalid running data**: Non-finite timestamps/values are filtered out before interpolation.
- **Missing eye tracking**: Sessions with no eye tracking data are skipped entirely.
- **Pupil blinks**: `likely_blink` frames and non-finite values are removed, then linearly interpolated over.
- **Few trials**: Sessions with fewer than 2 usable trials are skipped.
- **Invalid outcome**: Trials not matching any of the four outcomes are dropped.

ii.
```python
good = np.isfinite(run_t) & np.isfinite(run_v)
run_t, run_v = run_t[good], run_v[good]
if len(run_t) == 0:
    return {'oeid': oeid, 'error': 'no running data'}
...
good = (np.isfinite(pupil_t) & np.isfinite(pupil_d)
        & ~eye.likely_blink.values.astype(bool))
...
if len(neural) < 2:
    return {'oeid': oeid, 'error': f'only {len(neural)} usable trials'}
```

iii. The AI handles missing data conservatively — sessions or trials with insufficient data are dropped rather than imputed.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each session's NWB file via `BehaviorOphysExperiment.from_nwb_path()`. The AI parallelizes this across sessions using `ProcessPoolExecutor` with configurable number of workers (default 12).

ii.
```python
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for i, res in enumerate(pool.map(extract_and_cache, oeids)):
```

iii. NWB file I/O dominates runtime. The caching strategy (`extract_and_cache`) avoids redundant re-extraction on subsequent runs.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop within `extract_session` iterates over each valid trial sequentially, but the `np.searchsorted`, `np.interp`, and array slicing could theoretically be vectorized since all trials use the same fixed-length window (T=124 frames). The image name assignment via `np.vectorize` is also essentially a loop.

ii.
```python
for _, tr in trials.iterrows():
    ...
    neural.append(neural_all[:, i0:i0 + T])
    ...
```

iii. The per-trial loop is simple and I/O dominates runtime, so vectorization would yield minimal speedup.

## 9-c. What processing does the code repeat multiple times?

i. The code processes each session twice: once during extraction (saving to cache) and once during assembly (loading from cache and applying discretization). The `np.load` step re-reads all per-session data from disk. However, the caching mechanism avoids repeating the expensive NWB loading on subsequent runs.

ii.
```python
# First pass: extract and cache
for i, res in enumerate(pool.map(extract_and_cache, oeids)):
...
# Second pass: load cache and assemble
for oeid in oeids:
    with np.load(cache_path(oeid), allow_pickle=True) as f:
```

iii. The two-pass design separates extraction (parallelizable, cacheable) from assembly (needs global statistics like percentile bin edges).

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores detailed `session_info` metadata for each session (equipment name, indicator, experience level, etc.) that is not used by the decoder. The caching to disk (`.npz` files) is also not needed if the code is run only once.

ii.
```python
session_info.append({
    'ophys_experiment_id': int(s['oeid']),
    'ophys_session_id': int(row.ophys_session_id),
    'behavior_session_id': int(row.behavior_session_id),
    ...
})
```

iii. The extra metadata is useful for debugging and analysis but not required by the decoder format.
