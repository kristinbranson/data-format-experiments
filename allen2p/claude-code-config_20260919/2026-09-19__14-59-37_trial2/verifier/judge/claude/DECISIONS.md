# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the experiment metadata from a CSV file (`ophys_experiment_table.csv`) in the project metadata directory, filters to experiments whose NWB files are locally available and not passive sessions, then loads each experiment's NWB file directly using `BehaviorOphysExperiment.from_nwb_path()`. Sessions are processed in parallel using a `ProcessPoolExecutor` with up to 16 workers.

ii.
```python
def select_sessions():
    exp = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    have = set()
    for f in os.listdir(EXP_DIR):
        if f.endswith('.nwb'):
            have.add(int(f.split('_')[-1].split('.')[0]))
    exp = exp[exp.ophys_experiment_id.isin(have)]
    exp = exp[~exp.passive]
    exp = exp.sort_values(['ophys_session_id', 'ophys_experiment_id'])
    return exp

def load_experiment(eid: int):
    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
        BehaviorOphysExperiment)
    return BehaviorOphysExperiment.from_nwb_path(
        os.path.join(EXP_DIR, 'behavior_ophys_experiment_%d.nwb' % eid))
```

iii. The AI chose to load NWB files directly via `from_nwb_path` rather than through the SDK's cache API, as the data was already downloaded locally. This avoids the overhead of the S3 cache mechanism. The CSV metadata table is used for experiment discovery and filtering. Parallel processing via ProcessPoolExecutor speeds up the I/O-bound NWB loading.

## 1-b. How are the data split into subjects?

i. Subjects are determined from the `mouse_id` field in each successfully converted session's metadata. The unique mouse IDs are collected and sorted.

ii.
```python
subjects = sorted({r['mouse_id'] for r in ok})
# ...
'subject_idx': np.array([subjects.index(r['mouse_id']) for r in ok], dtype=np.int64),
```

iii. The `mouse_id` from the experiment metadata uniquely identifies each animal.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `ophys_session_id` values. Experiments (imaging planes) sharing the same `ophys_session_id` are grouped together as one session. For Multiscope sessions, neurons from multiple simultaneously recorded planes are concatenated.

ii.
```python
groups = exp.groupby('ophys_session_id')
session_ids = list(groups.groups.keys())
# In convert_session:
experiments = [load_experiment(e) for e in eids]
```

iii. The whitepaper defines a session as one continuous recording. Grouping by `ophys_session_id` correctly merges all planes from Multiscope sessions into one session.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `trials` table. Go and Catch trials are kept (filtering via `trials.go | trials.catch`), which automatically excludes Aborted and Auto-rewarded trials. The trial window is `[start_time, stop_time)` with variable length. The window is divided into uniform time bins of BIN_SIZE = 1/31 s.

ii.
```python
sel = trials[(trials.go | trials.catch)].sort_values('start_time')
# ...
for tid, tr in sel.iterrows():
    t_beg, t_stop = float(tr.start_time), float(tr.stop_time)
    nbins = int(np.floor((t_stop - t_beg) / BIN_SIZE))
    if nbins < 2:
        continue
    tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
```

iii. Using `go | catch` as a filter is equivalent to excluding `aborted` and `auto_rewarded`, as required by the task instructions. Variable-length trials preserve the full trial window for time-varying output variables.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) keeping only Go and Catch trials, (2) requiring at least 2 time bins, (3) requiring full ophys coverage of the trial window, (4) requiring a valid trial outcome (hit/miss/FA/CR), (5) sessions with fewer than 2 usable trials or without eye tracking / running data are excluded.

ii.
```python
sel = trials[(trials.go | trials.catch)].sort_values('start_time')
assert not sel.aborted.any() and not sel.auto_rewarded.any()
# ...
if nbins < 2:
    continue
if any(tc[0] < ts[0] or tc[-1] > ts[-1] for ts in plane_ts):
    continue
# ...
if len(neural_trials) < 2:
    return {'ophys_session_id': sid, 'error': 'fewer than 2 usable trials'}
# ...
if len(eye) == 0 or not np.any(np.isfinite(eye['pupil_width'].values)):
    return {'ophys_session_id': sid, 'error': 'no eye tracking data'}
```

iii. Passive sessions are excluded because trial outcome is undefined (no licking/reward). Eye tracking is required because pupil diameter is a decoder output. The ophys coverage check ensures neural data spans the entire trial window.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces.dff` (detrended dF/F calcium fluorescence traces) by default, with an option to use `events` or `filtered_events` via the `--trace` flag.

ii.
```python
if trace_name == 'dff':
    tbl = ds.dff_traces
    mat = np.stack(tbl['dff'].values).astype(np.float32)
else:
    tbl = ds.events
    mat = np.stack(tbl[trace_name].values).astype(np.float32)
```

iii. The AI empirically compared dF/F, filtered_events, and raw events on a sample of sessions and found dF/F gave the best decoding accuracy across all outputs. The whitepaper describes dF/F as the primary processed neural signal. While Piet et al. use detected events, that choice was made for their event-triggered analyses and is not required for a per-timepoint decoder.

## 2-b. How is the `neural` data processed?

i. Neural traces from multiple imaging planes within a session are concatenated along the neuron axis. The traces are cast to float32. For each trial, neural data is sampled at the nearest ophys frame to each bin center (nearest-neighbor lookup).

ii.
```python
plane_traces.append(mat)
plane_ts.append(ts)
# ...
mats = [mat[:, nearest_index(ts, tc)]
        for mat, ts in zip(plane_traces, plane_ts)]
neural = np.concatenate(mats, axis=0) if len(mats) > 1 else mats[0]
```

iii. No additional filtering or normalization is applied. The Allen SDK pipeline already handles motion correction, neuropil subtraction, demixing, and dF/F computation. The nearest-frame approach avoids interpolating neural data (which would be questionable for calcium signals).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to neurons. All ROIs returned by the SDK's `dff_traces` / `events` tables are included. However, sessions without eye tracking data are dropped entirely (3 sessions, affecting ~276 neurons).

ii. N/A — no explicit neuron filtering code.

iii. The Allen pipeline already applies ROI quality control (motion border exclusion, union/duplicate removal, dendrite filtering, low SNR removal). The SDK exposes only valid ROIs, so no further filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to `start_time` from the trials table. Uniform time bins of 1/31 s are created starting at `start_time`, and the neural value at each bin is the ophys frame nearest to the bin center.

ii.
```python
t_beg, t_stop = float(tr.start_time), float(tr.stop_time)
nbins = int(np.floor((t_stop - t_beg) / BIN_SIZE))
tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
# ...
mats = [mat[:, nearest_index(ts, tc)]
        for mat, ts in zip(plane_traces, plane_ts)]
```

iii. The instruction says "Temporally align based on ophys timestamp." The bin centers are placed at regular intervals from trial start, and neural data is looked up at the nearest ophys frame to each bin center. For 31 Hz single-plane sessions, this is essentially a 1:1 mapping to native frames. For 11 Hz Multiscope sessions, this is a zero-order hold upsampling to the common 31 Hz grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses a fixed time bin size of 1/31 s = 32.258 ms for all sessions. This is a rebinning for the 11 Hz Multiscope sessions (native dt ~93.23 ms) but essentially matches the native frame rate for the 31 Hz single-plane sessions.

ii.
```python
BIN_SIZE = 1.0 / 31.0  # s; nominal single-plane 2P frame period
# ...
'time_bin_size': 1000.0 * BIN_SIZE,
```

iii. A common time bin size is needed across all sessions for the decoder. The 31 Hz nominal frame rate was chosen as the common grid because 97% of sessions are already at this rate. For the 6 Multiscope sessions (11 Hz), nearest-frame lookup provides a zero-order hold upsampling.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table (specifically the `change_detection_behavior` block), using `start_time`, `end_time`, `image_name`, and `omitted` columns. This is different from the reference which uses `initial_image_name` and `change_image_name` from the trials table.

ii.
```python
beh = stim[stim.stimulus_block_name == BEHAVIOR_BLOCK].sort_values('start_time')
flash_start = beh.start_time.values.astype(float)
flash_end = beh.end_time.values.astype(float)
flash_img = beh.image_name.values.astype(str)
flash_omitted = beh.omitted.values.astype(bool)
# ...
j = np.searchsorted(flash_start, tc, side='right') - 1
j = np.clip(j, 0, len(flash_start) - 1)
on_screen = (tc >= flash_start[j]) & (tc < flash_end[j])
img = np.zeros(nbins, dtype=np.int64)
if np.any(on_screen):
    img[on_screen] = [img_lookup[n] for n in flash_img[j[on_screen]]]
```

iii. Using `stimulus_presentations` provides finer temporal resolution of image identity, capturing the exact 250ms flash periods and 500ms gray screen intervals. This means image identity is 0 ("gray") during inter-stimulus intervals, rather than being assigned the name of the most recent image. The AI justified this as providing more accurate per-timepoint labels.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes with 0 = "gray" (blank/omitted) and 1-16 for the 16 unique natural images across image sets A and B. A global mapping is built from all sessions. During inter-stimulus intervals (gray screen) and omitted flashes, the label is 0.

ii.
```python
image_names = sorted(set(flash_img[~flash_omitted]) - {'omitted'})
img_lookup = {name: i + 1 for i, name in enumerate(image_names)}  # 0 = gray
# ...
# Global remapping:
global_img = {name: i + 1 for i, name in enumerate(image_names)}
for r in ok:
    remap = np.zeros(len(r['image_names']) + 1, dtype=np.int64)
    for local, name in enumerate(r['image_names']):
        remap[local + 1] = global_img[name]
    for out in r['output']:
        out[0] = remap[out[0]]
```

iii. The two-level mapping (session-local then global) is needed because each session only shows 8 of the 16 images. The global remapping ensures consistent integer codes across all sessions. Including "gray" as a category (0) captures the actual visual stimulus at each timepoint.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same bin centers (`tc`) as the neural data, by looking up which stimulus flash (if any) is on screen at each bin center time. Alignment is guaranteed because both neural and image identity use the same bin time grid.

ii.
```python
tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
# Neural:
mats = [mat[:, nearest_index(ts, tc)] ...]
# Image:
j = np.searchsorted(flash_start, tc, side='right') - 1
on_screen = (tc >= flash_start[j]) & (tc < flash_end[j])
```

iii. Both neural and stimulus data are sampled at the same bin centers, ensuring temporal alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table, specifically the `is_change` flag and the flash `start_time`/`end_time`. It is 1 only during the 250ms flash where `is_change == True` (go trials only), and 0 otherwise.

ii.
```python
flash_is_change = beh.is_change.values.astype(bool)
# ...
chg = np.zeros(nbins, dtype=np.int64)
if bool(tr.go):
    k = int(np.searchsorted(flash_start, float(tr.change_time), side='right') - 1)
    assert flash_is_change[k], 'change_time does not match an is_change flash'
    chg[(tc >= flash_start[k]) & (tc < flash_end[k])] = 1
```

iii. The AI marks image change as 1 only during the 250ms changed-image flash presentation, not the full 750ms cycle (flash + gray). For catch trials, image_change is all zeros since the image doesn't actually change.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Binary indicator (0/1). The change flash is identified by matching `change_time` to the `stimulus_presentations` row with `is_change == True`. The 1 value covers only the 250ms flash duration (approximately 8 bins at 31 Hz).

ii. See 4-a code snippet.

iii. The AI validated that `change_time` exactly matches the `start_time` of the `is_change` flash in `stimulus_presentations`.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary — it is directly computed as 0 or 1 with no thresholding needed.

ii. See 4-a.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same bin-center grid as neural data. The change indicator is 1 for bins whose center falls within the change flash's `[start_time, end_time)` interval.

ii. See 4-a.

iii. Same temporal alignment as all other variables — computed at the shared bin centers.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides `timestamps` and `speed` (cm/s, already 10 Hz low-pass filtered by the SDK).

ii.
```python
running = ds0.running_speed
run_t = running['timestamps'].values.astype(float)
run_v = running['speed'].values.astype(float)
```

iii. The `running_speed` attribute (filtered, not `raw_running_speed`) is the SDK's standard processed running speed.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated (ignoring NaN values) onto the bin center times, then discretized into 5 equal-percentile bins computed **per session**.

ii.
```python
run_trials.append(interp_nonan(run_t, run_v, tc))
# ...
run_all = np.concatenate(run_trials)
run_lab, run_edges = quantile_bin(run_all)
```

iii. Linear interpolation preserves the signal shape. Per-session quintile binning ensures balanced bins within each session, accounting for between-session variability in running behavior.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Discretized into 5 bins (0-4) using percentile-based edges computed per session. Edges are at the 20th, 40th, 60th, and 80th percentiles of the session's running speed values.

ii.
```python
def quantile_bin(values, nbins=N_QUANTILE_BINS):
    edges = np.quantile(values, np.arange(1, nbins) / nbins)
    labels = np.searchsorted(edges, values, side='right').astype(np.int64)
    return np.clip(labels, 0, nbins - 1), edges
```

iii. Per-session quantization yields balanced 5-class bins within each session. The AI justified this because pupil width varies in camera pixels across rigs and running distributions differ across mice, so pooled quintiles would produce degenerate per-session distributions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same bin center times as neural data, ensuring perfect temporal alignment.

ii.
```python
run_trials.append(interp_nonan(run_t, run_v, tc))
```

iii. The running speed timestamps are hardware-synced with the ophys clock.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using the `pupil_width` column. `timestamps` provide the time axis.

ii.
```python
eye = ds0.eye_tracking
eye_t = eye['timestamps'].values.astype(float)
eye_v = eye['pupil_width'].values.astype(float)
```

iii. `pupil_width` is the SDK's measure of pupil diameter. Tutorials use it as "pupil diameter."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil width is linearly interpolated onto bin center times, ignoring NaN values (which correspond to blinks/outliers — the SDK sets `pupil_width` to NaN on `likely_blink` frames). Then discretized into 5 per-session percentile bins.

ii.
```python
pup_trials.append(interp_nonan(eye_t, eye_v, tc))
# ...
pup_all = np.concatenate(pup_trials)
pup_lab, pup_edges = quantile_bin(pup_all)
```

iii. The `interp_nonan` function filters out NaN/non-finite values before interpolation, effectively interpolating across blinks. Sessions where all pupil values are non-finite after interpolation are dropped entirely.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed — discretized into 5 per-session percentile bins (0-4).

ii. See `quantile_bin` function in 5-c.

iii. Per-session binning is justified because pupil width is measured in camera pixels which vary across rigs/sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated onto the shared bin center times.

ii.
```python
pup_trials.append(interp_nonan(eye_t, eye_v, tc))
```

iii. Same temporal alignment mechanism as all other variables.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
if bool(tr.hit):
    outcome = 0
elif bool(tr.miss):
    outcome = 1
elif bool(tr.false_alarm):
    outcome = 2
elif bool(tr.correct_reject):
    outcome = 3
else:
    continue  # skip trial with no outcome flag
```

iii. These four columns are the SDK's canonical trial outcome labels. They are mutually exclusive for Go/Catch trials. Trials without any outcome flag are skipped (should not occur for valid Go/Catch trials).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The code is constant across all time bins within a trial (static per-trial variable broadcast over time).

ii.
```python
out[4] = outcome_trials[i]  # scalar broadcast to all bins
```

iii. The mapping follows the order of `OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions with no eye tracking data or all-NaN pupil values are excluded entirely (3 sessions).
- **Missing running data**: Sessions without running data are excluded.
- **Blink/NaN in eye tracking**: The `interp_nonan` function interpolates through NaN values by fitting only on finite samples.
- **Non-finite running/pupil after interpolation**: Sessions with any non-finite values after interpolation are excluded.
- **Trials not fully covered by ophys**: Trials whose bin centers fall outside the ophys timestamp range are skipped.
- **Very short trials**: Trials with fewer than 2 bins are skipped.
- **Sessions with too few trials**: Sessions with fewer than 2 usable trials are excluded.
- **Inconsistent behavior tables across planes**: An assertion checks that trial tables agree across planes of a Multiscope session.
- **General exceptions**: A try/except in `convert_session` catches any unexpected errors and logs them without crashing the pipeline.

ii.
```python
if len(eye) == 0 or not np.any(np.isfinite(eye['pupil_width'].values)):
    return {'ophys_session_id': sid, 'error': 'no eye tracking data'}
# ...
if any(tc[0] < ts[0] or tc[-1] > ts[-1] for ts in plane_ts):
    continue
# ...
if not np.all(np.isfinite(pup_all)):
    return {'ophys_session_id': sid, 'error': 'non-finite pupil after interpolation'}
```

iii. The AI chose to drop sessions/trials with missing data rather than impute, which is conservative but ensures data quality. Excluded sessions are logged in `metadata['excluded_sessions']`.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files via `BehaviorOphysExperiment.from_nwb_path()`, which is I/O bound (~4-5 s per single-plane experiment, ~17 s for a 7-plane Multiscope session). This is mitigated by parallel processing with up to 16 workers.

ii. N/A

iii. The AI logged timing information and found the total wall-clock time was ~79 seconds for all 171 sessions with 16 workers.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop within `convert_session` iterates over each valid trial sequentially, performing `searchsorted`, frame index lookup, and output construction for each trial. The image identity lookup (`[img_lookup[n] for n in flash_img[j[on_screen]]]`) uses a Python list comprehension inside the trial loop.

ii.
```python
for tid, tr in sel.iterrows():
    # ... per-trial processing
    if np.any(on_screen):
        img[on_screen] = [img_lookup[n] for n in flash_img[j[on_screen]]]
```

iii. The per-trial loop is not a bottleneck compared to NWB loading. The list comprehension for image lookup could theoretically be vectorized with a precomputed integer array, but the savings would be minimal.

## 9-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each NWB file is opened once per session, and all data streams are extracted from that single object. The global image mapping remapping is applied once after all sessions are processed.

ii. N/A

iii. The AI designed the code to avoid redundant work — each session is processed independently in a worker process.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extensive per-session metadata (`info` dict with trial counts, quantile edges, frame rates, etc.) and `cell_specimen_ids` that are carried through processing but not all fields may be used by the decoder. The per-session quantile binning computation is done within each session worker, meaning quantile edges are computed per-session rather than globally — this is a design choice rather than unnecessary processing, but it does mean the quintile computation happens 171 times rather than once.

ii.
```python
info = dict(
    ophys_session_id=int(sid),
    behavior_session_id=int(md0['behavior_session_id']),
    # ... many more fields
)
result = dict(
    # ...
    cell_specimen_ids=np.concatenate(plane_cellid),
    info=info,
    trial_info=trial_info,
    timing=timing,
)
```

iii. The extra metadata is useful for debugging and validation but is not used by the decoder.
