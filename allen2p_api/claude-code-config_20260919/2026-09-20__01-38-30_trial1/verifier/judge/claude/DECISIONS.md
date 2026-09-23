# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data are loaded via the AllenSDK `VisualBehaviorOphysProjectCache.from_s3_cache()`. The experiment table is queried to find all locally available active-behaviour experiments. Sessions are identified by grouping experiments by `ophys_session_id`. For each session, all constituent experiments (imaging planes) are loaded via `cache.get_behavior_ophys_experiment(oeid)`. Data are filtered to only experiments whose NWB file exists locally, and only active-behaviour sessions (`passive == False`).

ii.
```python
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=CACHE_DIR)
et = cache.get_ophys_experiment_table()
ids = local_experiment_ids()
sel = et[et.index.isin(ids) & (~et.passive)].copy()
# ...
datasets = [cache.get_behavior_ophys_experiment(int(o)) for o in oeids]
```

iii. The agent chose to filter to locally available NWB files and to exclude passive sessions (where mice cannot lick and trial outcomes are degenerate). This is documented in CONVERSION_NOTES Steps 2 and 4: "Passive sessions = 'mice view the task stimuli with the lick spout retracted'; the paper's behavioral analyses use 'all active behavioural sessions'."

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `mouse_id` field from experiment metadata. A sorted set of unique mouse IDs across all converted sessions forms the `subjects` list. Subject index per session is determined via a lookup.

ii.
```python
subjects = sorted({r['mouse_id'] for r in results})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_to_idx[r['mouse_id']] for r in results], dtype=np.int64),
```

iii. The `mouse_id` field is the SDK's canonical unique identifier for each animal. The agent confirmed 38 mice with local active NWBs, matching the data.

## 1-c. How are the data split into sessions?

i. Sessions correspond to unique `ophys_session_id` values. Multiple experiments (imaging planes) recorded simultaneously in a single session are grouped together and concatenated along the neuron axis. Sessions are sorted by `ophys_session_id`.

ii.
```python
sel = sel.sort_values(['ophys_session_id', 'ophys_experiment_id'])
session_ids = sorted(sel.ophys_session_id.unique().tolist())
# In convert_session:
datasets = [cache.get_behavior_ophys_experiment(int(o)) for o in oeids]
# Neural data from all planes concatenated:
neural = np.concatenate(neural_blocks, axis=0)   # (n_neurons, n_trials, NBINS)
```

iii. Using `ophys_session_id` groups all simultaneously recorded imaging planes. CONVERSION_NOTES Step 4: "Use ophys_session as the 'session' of the output format: planes recorded simultaneously are concatenated along the neuron axis."

## 1-d. How are the data split into trials?

i. Trials are defined by the SDK's `ds.trials` table. Go and catch trials are included; aborted and auto-rewarded trials are excluded. A fixed window of [-2.25, +3.0] s around `change_time` is used for each trial, yielding 21 bins of 250 ms each (fixed-length trials).

ii.
```python
def select_trials(trials):
    m = (trials.go | trials.catch) & (~trials.aborted) & (~trials.auto_rewarded)
    m &= trials.change_time.notna()
    return trials[m]

# Fixed trial window:
OFF_START = -2.25   # s relative to the change
OFF_END = 3.00      # s relative to the change
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 21
```

iii. CONVERSION_NOTES Step 5 decision 4: "Trial window = [-2.25, +3.0] s around the change, i.e. 7 complete 750 ms flash cycles (3 pre-change flashes, the change flash, 3 post-change flashes). Every included trial contains this window entirely inside its own start_time/stop_time."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) excluding aborted and auto-rewarded trials, (2) requiring valid `change_time`, (3) checking that trial windows don't overlap (strictly increasing bin edges), (4) dropping trials with incomplete neural coverage, (5) dropping trials with behavioural sensor dropouts > 2s, (6) dropping trials where image identity bins fall outside stimulus presentations, (7) requiring at least 2 valid trials per session.

ii.
```python
m = (trials.go | trials.catch) & (~trials.aborted) & (~trials.auto_rewarded)
m &= trials.change_time.notna()
# ...
if np.any(np.diff(edges_flat) <= 0):
    return {'session_id': session_id, 'skip': 'overlapping trial windows'}
# ...
trial_ok = np.isfinite(neural).all(axis=(0, 2))
# ... sensor gap checks ...
trial_ok &= ok
# ...
trial_ok &= ~(image_names == 'none').any(axis=1)
if trial_ok.sum() < 2:
    return {'skip': f'only {int(trial_ok.sum())} trials with complete data'}
```

iii. CONVERSION_NOTES Step 10 documents that 9 trials were dropped for sensor dropout > 2s, and 3 sessions (917 trials) were dropped for missing eye tracking entirely. The agent verified all 43,966 converted trials are accounted for.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces` (dF/F calcium fluorescence traces) from each experiment, accessed via `ds.dff_traces.dff`. The agent initially planned to use `events` (as in the reference paper) but switched to dF/F after empirical comparison showed much better decoding accuracy.

ii.
```python
NEURAL_SIGNAL = 'dff'
# In convert_session:
if signal_name == 'dff':
    traces = np.vstack(ds.dff_traces.dff.values).astype(np.float64)
```

iii. CONVERSION_NOTES Step 5 decision 7: "events are 84-99% zeros and cost roughly half the decodable stimulus information (image identity 0.26 vs 0.51 balanced accuracy). dF/F is the pipeline's primary neural product." The `--signal events` flag reproduces the paper's choice.

## 2-b. How is the `neural` data processed?

i. Neural data is bin-averaged into 250 ms bins. For each session, ophys frames are assigned to bins via `np.searchsorted` on `ophys_timestamps`, and means are computed via `np.add.reduceat`. Multiple imaging planes within a session are concatenated along the neuron axis. No additional filtering, normalization, or z-scoring is applied.

ii.
```python
def bin_mean(values, timestamps, edges_flat):
    idx = np.searchsorted(timestamps, edges_flat, side='left')
    counts = np.diff(idx)
    stop = max(int(idx[-1]), 1)
    starts = np.minimum(idx[:-1], stop - 1)
    sums = np.add.reduceat(v[:, :stop], starts, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        means = sums / np.maximum(counts, 1)[None, :]
    # ...
    return (means[0] if one_d else means), counts

# Applied:
means, counts = bin_mean(traces, ts - neural_lag, edges_flat)
block = means[:, keep].reshape(traces.shape[0], len(sel), NBINS)
neural = np.concatenate(neural_blocks, axis=0)
```

iii. CONVERSION_NOTES Step 5 decision 5: "Bin size = 250 ms ... divides the 750 ms flash cycle exactly into 3 bins (image / gray / gray) and is >= 2 ophys frames even at the 11 Hz Multiscope rate." The vectorized binning approach was chosen for efficiency.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filtering is applied. All neurons in the released NWB files are included (`valid_roi` is True for all cells). Trials with incomplete neural coverage (NaN bins) are dropped.

ii.
```python
trial_ok = np.isfinite(neural).all(axis=(0, 2))
```

iii. CONVERSION_NOTES Step 1: "ROI-quality filtering was already applied upstream (valid_roi all True), so no further neuron curation is justified." Step 10 Check 3 confirms: "Allen pipeline ROI filtering applied before release; neither paper filters further."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to `change_time` (the stimulus change event). A fixed window of [-2.25, +3.0] s around `change_time` is used for each trial. Bin edges are computed as absolute times relative to each trial's `change_time`.

ii.
```python
OFF_START = -2.25
OFF_END = 3.00
BIN_SIZE = 0.25

def bin_edges_for_trials(change_times):
    offsets = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    edges = change_times[:, None] + offsets[None, :]
    return edges
```

iii. CONVERSION_NOTES Step 5 decision 3: "Alignment event = change_time (the actual image change on go trials, the sham change on catch trials). This is the only event common to every included trial, it is exactly a flash onset, and the reference paper aligns its analyses to image presentations/changes."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned into 250 ms bins (21 bins per trial). This is a deliberate rebinning from the native ophys frame rate (~31 Hz single-plane or ~11 Hz Multiscope) into uniform bins.

ii.
```python
BIN_SIZE = 0.25          # s
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 21
# metadata:
'time_bin_size': BIN_SIZE * 1000.0,   # 250 ms
```

iii. CONVERSION_NOTES Step 5 decision 5: "A bin size equal to the native ophys frame period cannot be used because the format requires one bin size for all sessions while the rigs sample at 31 Hz and 11 Hz." 250 ms divides the 750 ms flash cycle into 3 bins and is large enough that no bin is ever empty.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `ds.stimulus_presentations`, specifically the `image_name` and `start_time` columns, filtered to the `change_detection_behavior` stimulus block.

ii.
```python
sp = ds0.stimulus_presentations
spa = sp[sp.stimulus_block_name == STIM_BLOCK]
flash_start = spa.start_time.values.astype(np.float64)
flash_name = spa.image_name.values.astype(object)
j = np.searchsorted(flash_start, centers_flat, side='right') - 1
valid = (j >= 0) & (j < len(flash_start))
j_clipped = np.clip(j, 0, len(flash_start) - 1)
within = valid & (centers_flat - flash_start[j_clipped] < FLASH_CYCLE + 0.05)
image_name_flat = np.where(within, flash_name[j_clipped], 'none')
```

iii. CONVERSION_NOTES Step 5 decision 8: "Image identity is labelled per 750 ms flash interval, exactly the paper's 'image presentation interval' convention." Each bin center is assigned the image from the last flash that started before it.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each bin center is assigned to the most recent stimulus flash. The image name of that flash becomes the bin's identity label. Image names are mapped to integer indices via a global sorted list of all unique image names (16 images + 'omitted' = 17 categories). Omitted flashes are kept as their own category.

ii.
```python
image_values = sorted({n for r in results for n in np.unique(r['image_names']) if n != 'omitted'})
image_values = image_values + ['omitted']
image_to_idx = {n: i for i, n in enumerate(image_values)}
img = np.vectorize(image_to_idx.get)(r['image_names']).astype(np.int64)
```

iii. CONVERSION_NOTES Step 5 decision 9: "Image identity uses global image names (16 across image sets A and B, + omitted = 17 categories) rather than within-session indices, so that a label means the same physical image in every session."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned by assigning each 250 ms bin center (the same bin grid used for neural data) to the appropriate stimulus flash. The bin centers are computed from `change_time` using the same offsets as the neural bins.

ii.
```python
centers_flat = (edges_flat[:-1] + edges_flat[1:]) / 2.0
j = np.searchsorted(flash_start, centers_flat, side='right') - 1
```

iii. The same absolute-time bin grid is used for all data streams, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `trials.is_change` (boolean indicating whether the trial is a go trial with an actual image change) and `trials.change_time` (the time of the change).

ii.
```python
is_change = sel.is_change.values.astype(bool)
rel_centers = centers - change_times[:, None]
image_change = ((rel_centers >= 0) & (rel_centers < CHANGE_WINDOW)
                & is_change[:, None]).astype(np.int64)
```

iii. `is_change` distinguishes go trials (real change) from catch trials (sham change). The agent uses relative bin centers to determine which bins fall within the change window.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Image change is computed as a binary indicator: 1 for bins whose center falls in [0, 0.5s) relative to `change_time` on go trials only, 0 elsewhere and everywhere on catch trials.

ii.
```python
CHANGE_WINDOW = 0.5   # s after the change
image_change = ((rel_centers >= 0) & (rel_centers < CHANGE_WINDOW)
                & is_change[:, None]).astype(np.int64)
```

iii. CONVERSION_NOTES Step 5 decision 8b: "500 ms is the representable window closest to the 400 ms post-presentation window the reference paper decodes in, it sits inside the behavioural response window (150-750 ms)."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary: 0 (no change) or 1 (change). The 500 ms window and the go-trial restriction define the boundary. No additional thresholding is needed.

ii. See 4-b above.

iii. The binary encoding directly satisfies the instruction's requirement of a binary variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change uses the same bin centers as the neural data, computed relative to `change_time`.

ii.
```python
rel_centers = centers - change_times[:, None]
```

iii. Same bin grid as all other data streams.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed`, which provides speed (cm/s) and timestamps from the running wheel encoder.

ii.
```python
rs = ds0.running_speed
run_ts = rs.timestamps.values.astype(np.float64)
run_v = interpolate_nans(rs.speed.values)
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is: (1) interpolated over any NaN values, (2) bin-averaged into 250 ms bins using the same bin grid as neural data, (3) gaps in binned data are filled by linear interpolation if < 2s, (4) discretized into 5 equal-percentile bins computed per session.

ii.
```python
run_v = interpolate_nans(rs.speed.values)
run_binned, _ = bin_mean(run_v, run_ts, edges_flat)
run_binned = run_binned[keep].reshape(len(sel), NBINS)
run_binned, ok = fill_short_gaps(run_binned, centers, MAX_SENSOR_GAP)
# ...
run_bin, run_edges = quantile_bins(run_binned.ravel())
```

iii. CONVERSION_NOTES Step 5 decision 10: "per-session percentiles are the only way the 5 labels carry the same meaning in every session; it also guarantees the 20/20/20/20/20 class balance that 'five equal percentile bins' asks for."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins using `np.percentile` to compute edges at 20th, 40th, 60th, and 80th percentiles, then `np.searchsorted` to assign labels. Bins are computed per session.

ii.
```python
def quantile_bins(values, nq=NQUANTILES):
    edges = np.percentile(values, np.linspace(0, 100, nq + 1)[1:-1])
    labels = np.searchsorted(edges, values, side='right').astype(np.int64)
    return np.clip(labels, 0, nq - 1), edges
```

iii. Per-session computation ensures each quintile contains exactly 20% of that session's data points.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is bin-averaged using the same absolute-time bin grid as the neural data. The same `edges_flat` array (derived from `change_time`) is used for both.

ii.
```python
run_binned, _ = bin_mean(run_v, run_ts, edges_flat)
```

iii. Same bin grid ensures alignment across all data streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking.pupil_area`. The diameter is computed as `2 * sqrt(pupil_area / pi)`.

ii.
```python
et = ds0.eye_tracking
pupil_area = et.pupil_area.values.astype(np.float64)
pupil_diam_raw = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. CONVERSION_NOTES Step 4: "whitepaper: pupil area = area of circle whose diameter is the ellipse major axis; diameter = 2*sqrt(pupil_area/pi)."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is: (1) computed from `pupil_area` as `2*sqrt(area/pi)`, (2) NaN values (from blinks, where `pupil_area` is NaN) are linearly interpolated over, (3) bin-averaged into 250 ms bins, (4) short gaps filled by interpolation, (5) discretized into 5 equal-percentile bins per session.

ii.
```python
pupil_diam_raw = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diam = interpolate_nans(pupil_diam_raw)
pupil_binned, _ = bin_mean(pupil_diam, eye_ts, edges_flat)
pupil_binned, ok = fill_short_gaps(pupil_binned, centers, MAX_SENSOR_GAP)
pupil_bin, pupil_edges = quantile_bins(pupil_binned.ravel())
```

iii. CONVERSION_NOTES Step 5 decision 11: "Blink NaNs in pupil are linearly interpolated over time within the session before binning, rather than dropping trials: blinks are short (3.5% of frames) and dropping them would bias the trial set toward calm periods."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal-percentile bins computed per session using `quantile_bins()`.

ii.
```python
pupil_bin, pupil_edges = quantile_bins(pupil_binned.ravel())
```

iii. Per-session percentile bins guarantee equal class balance within each session.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: bin-averaged using the same absolute-time bin grid as the neural data.

ii.
```python
pupil_binned, _ = bin_mean(pupil_diam, eye_ts, edges_flat)
```

iii. Same bin grid ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
def outcome_labels(sel):
    lab = np.full(len(sel), -1, dtype=np.int64)
    for i, name in enumerate(OUTCOMES):
        lab[sel[name].values.astype(bool)] = i
    return lab
```

iii. These four columns are the SDK's canonical trial outcome labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) via the `OUTCOMES` list ordering. The code is broadcast (repeated) across all 21 time bins within a trial.

ii.
```python
np.full(NBINS, r['outcome'][t], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 decision 12: "Trial outcome is broadcast over time because the format requires a single (d_output, T) array per trial."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: 3 sessions with empty `eye_tracking` tables are dropped entirely.
- **Blink NaNs**: Pupil area NaN values are linearly interpolated over before binning.
- **Short sensor gaps**: Empty bins in running/pupil are filled by linear interpolation if the gap is < 2s.
- **Long sensor gaps**: Trials with sensor dropouts > 2s are dropped (9 trials total).
- **NaN in running speed**: Interpolated over before binning.
- **Incomplete neural coverage**: Trials where binned neural data has NaN values are dropped.
- **Overlapping trial windows**: Sessions with overlapping trial windows are skipped.
- **Failed sessions**: Convert errors result in the session being skipped.
- **Image identity outside stimulus presentations**: Trials with bins that fall outside known flash times are dropped.

ii.
```python
pupil_diam = interpolate_nans(pupil_diam_raw)
run_v = interpolate_nans(rs.speed.values)
run_binned, ok = fill_short_gaps(run_binned, centers, MAX_SENSOR_GAP)
trial_ok = np.isfinite(neural).all(axis=(0, 2))
trial_ok &= ~(image_names == 'none').any(axis=1)
```

iii. CONVERSION_NOTES Step 7 documents that the gap interpolation approach was developed after discovering that 14 sessions were being unnecessarily dropped for sub-second eye-camera dropouts.

## 9-a. What are the most time-consuming steps of the code?

i. Loading NWB files via `cache.get_behavior_ophys_experiment()` is the dominant cost (~3.5s per experiment, 98% of per-session time). The agent mitigated this with multiprocessing (16 workers), reducing total conversion time from ~700s to ~69s.

ii.
```python
if args.nproc > 1 and len(jobs) > 1:
    with Pool(args.nproc) as pool:
        for k, r in enumerate(pool.imap_unordered(convert_session, jobs, chunksize=1)):
            results.append(r)
```

iii. CONVERSION_NOTES Step 6: "Loading an NWB file is ~3.5 s and is the only real cost (98% of the per-session time)."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent already vectorized the binning with `np.searchsorted` + `np.add.reduceat`, avoiding per-trial and per-bin loops. The remaining loop is per-trial in the assembly phase, which iterates over trials to construct output arrays.

ii.
```python
# Vectorized binning:
idx = np.searchsorted(timestamps, edges_flat, side='left')
sums = np.add.reduceat(v[:, :stop], starts, axis=1)

# Remaining loop in assembly:
for t in range(n_trials):
    neural_trials.append(np.ascontiguousarray(r['neural'][:, t, :]))
```

iii. CONVERSION_NOTES Step 6: "One searchsorted + one reduceat per data stream per session (no Python loop over trials or bins)." The per-trial assembly loop is necessary for the output format.

## 9-c. What processing does the code repeat multiple times?

i. Each session opens the AllenSDK cache via `get_cache()` independently (in multiprocessing mode, each worker creates its own cache connection). The session is processed once in a single pass. No data is loaded or processed redundantly within a session.

ii.
```python
def convert_session(args):
    # Each worker opens its own cache
    cache = get_cache()
```

iii. The multiprocessing architecture requires each worker to have its own cache connection. This is a minor overhead compared to the NWB loading time.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores intermediate results (`run_binned`, `pupil_binned`, `run_edges`, `pupil_edges`, `cell_ids`, `change_times`, `trial_ids`, etc.) in the per-session result dict that are used for plotting and diagnostics but not all end up in the final output. Additionally, the code computes `session_info` metadata for every session which includes detailed per-session statistics.

ii.
```python
result = {
    # ... used in final output:
    'neural': neural.astype(np.float32),
    # ... diagnostic/intermediate data:
    'run_binned': run_binned,
    'pupil_binned': pupil_binned,
    'cell_ids': cell_ids,
    'change_times': change_times,
    'trial_ids': sel.index.values,
    'timing': timing,
    # ...
}
```

iii. These intermediate values serve diagnostic and plotting purposes during conversion but add some memory overhead.
