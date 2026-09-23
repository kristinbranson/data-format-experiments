# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads data through the AllenSDK `VisualBehaviorOphysProjectCache` local cache at `/app/data`, not from S3. It starts from the experiment table, restricts to experiment IDs that actually have local NWB files, excludes passive sessions, groups experiments by `ophys_session_id`, and then loads each experiment with `get_behavior_ophys_experiment()`.

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
```

```python
for k, eid in enumerate(experiment_ids):
    ds = bc.get_behavior_ophys_experiment(int(eid))
```

iii. In `CONVERSION_NOTES.md` the agent says it used `from_local_cache('/app/data')` because the provided dataset is already a local AllenSDK cache, and that it intentionally restricted processing to active, locally present sessions.

## 1-b. How are the data split into subjects?

i. Subjects are split by `mouse_id`. The final `subjects` list is the sorted set of mouse IDs among kept sessions.

ii.
```python
subjects = sorted(set(r['mouse_id'] for r in results))
...
data['subject_idx'].append(subjects.index(r['mouse_id']))
```

iii. `CONVERSION_NOTES.md` treats `metadata['mouse_id']` as the canonical animal identifier and reports subject counts from that field.

## 1-c. How are the data split into sessions?

i. Sessions are split by unique `ophys_session_id`. All experiments/imaging planes with the same session ID are grouped into one session.

ii.
```python
for sid, sub in et.groupby('ophys_session_id'):
    groups.append((int(sid), sorted(int(i) for i in sub.index)))
```

```python
session_definition='one ophys_session_id; all active imaging planes concatenated as neurons'
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent states that one session should be one `ophys_session_id`, with all active planes concatenated.

## 1-d. How are the data split into trials?

i. Trials are not taken as the full AllenSDK trial windows. Instead, the agent keeps `go` and `catch` rows from `dataset.trials`, aligns each one to `change_time`, and creates a fixed `[-2.0 s, +3.0 s]` window around the change, binned into twenty 250 ms bins.

ii.
```python
BIN_SIZE = 0.25
OFF_START = -2.0
OFF_END = 3.0
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))

keep = trials[(trials.go | trials.catch)]
change_times = keep.change_time.values.astype(float)
edges = bin_edges_for_trials(change_times)
```

```python
def bin_edges_for_trials(change_times):
    offsets = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    return change_times[:, None] + offsets[None, :]
```

iii. The notes say the agent chose fixed windows around `change_time` because `change_time` exactly matches a stimulus flash onset, 250 ms bins match the flash structure, and the measured trial timing guaranteed the `[-2, +3]` window stayed inside each trial.

## 1-e. How are trials filtered based on quality controls?

i. The code keeps only `go` or `catch` trials, drops sessions with fewer than two such trials, and drops whole sessions with no usable eye-tracking data. It does not separately clip each trial to `start_time`/`stop_time` because it assumes the fixed `change_time` window is always covered.

ii.
```python
keep = trials[(trials.go | trials.catch)]
if len(keep) < 2:
    return None
```

```python
if pupil_binned is None:
    return dict(skip=True, ophys_session_id=int(ophys_session_id),
                reason='no eye tracking')
```

iii. `CONVERSION_NOTES.md` says the agent followed the task’s “keep go+catch” instruction, excluded passive sessions, and dropped the three sessions with no eye-tracking because the pupil output could not be defined.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. By default the final `neural` data come from `dataset.dff_traces['dff']`. The script also exposes `events` and `filtered_events` as options, but the shipped conversion uses the default `dff`.

ii.
```python
ap.add_argument('--neural-signal', choices=['events', 'filtered_events', 'dff'], default='dff')
...
if signal == 'dff':
    ev = ds.dff_traces
    E = np.vstack(ev['dff'].values).astype(np.float32)
else:
    ev = ds.events
    E = np.vstack(ev[signal].values).astype(np.float32)
```

iii. The notes explain that the paper used events, but the agent switched to dF/F after a decoder comparison and kept the events-based variants as command-line options.

## 2-b. How is the `neural` data processed?

i. Neural traces are loaded plane by plane, binned into 250 ms windows around each trial’s `change_time`, and then concatenated across planes along the neuron axis. For dF/F, the code computes the mean within each bin; for events, it uses bin sums.

ii.
```python
binned = bin_sum(E, ots, edges)
if signal == 'dff':
    counts = bin_sum(np.ones((1, E.shape[1]), dtype=np.float32), ots, edges)
    binned = binned / np.maximum(counts, 1.0)
neural_planes.append(binned)
...
neural = np.concatenate(neural_planes, axis=0)
```

iii. In the notes, the agent justifies 250 ms bins as flash-locked and says dF/F should be averaged, not summed, because the number of ophys frames per bin differs across rigs.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no additional neuron-level filtering beyond what the AllenSDK already returns. The code uses all cells in `dff_traces`/`events` for the kept sessions.

ii.
```python
if signal == 'dff':
    ev = ds.dff_traces
    E = np.vstack(ev['dff'].values).astype(np.float32)
else:
    ev = ds.events
    E = np.vstack(ev[signal].values).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says ROI validity filtering is already applied by the SDK (`exclude_invalid_rois=True`), so the agent did not add extra per-cell QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `trials.change_time`, not to `start_time`. The neural window spans `[-2.0, +3.0]` seconds around that event.

ii.
```python
change_times = keep.change_time.values.astype(float)
edges = bin_edges_for_trials(change_times)
...
temporal_alignment_event=('stimulus change time (trials.change_time): the onset of the '
                          'changed image on go trials, of the sham change on catch trials')
```

iii. The notes repeatedly state that `change_time` exactly matches the onset of the real or sham change flash, so the agent treated it as the natural alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data are rebinned to 250 ms bins. The original frame-rate resolution is not preserved in the final dataset.

ii.
```python
BIN_SIZE = 0.25
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
...
data['metadata'] = dict(
    ...
    time_bin_size=BIN_SIZE * 1000.0,
```

iii. The notes justify 250 ms because it equals the image-on duration and divides the 750 ms flash cycle into three stimulus-locked bins.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `dataset.stimulus_presentations.image_name`, restricted to the change-detection block. Omitted flashes are forward/back-filled so every flash interval has an image label.

ii.
```python
def flash_table(dataset):
    sp = dataset.stimulus_presentations
    sp = sp[sp.stimulus_block_name.str.contains('change_detection')]
    sp = sp.sort_values('start_time')
    names = sp.image_name.astype(str).replace('omitted', np.nan).ffill().bfill().values
    return (sp.start_time.values.astype(float), names,
            sp.is_change.values.astype(bool), sp)
```

iii. The notes say image identity should be attached to the 750 ms image-presentation interval, and omitted flashes should inherit the previous image identity.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code looks up the flash containing each 250 ms bin centre, assigns that flash’s image name, then maps global image names to integer codes shared across sessions.

ii.
```python
centres = edges[:, :-1] + BIN_SIZE / 2.0
fidx = np.searchsorted(starts, centres.ravel(), side='right') - 1
fidx = np.clip(fidx, 0, len(starts) - 1)
image_name = names[fidx].reshape(ntrials, NBINS)
```

```python
images = sorted(set(np.concatenate([r['image_name'].ravel() for r in results])))
img_index = {n: i for i, n in enumerate(images)}
...
img_idx = np.vectorize(img_index.get)(r['image_name']).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says the agent wanted a global 16-image vocabulary and a time-varying image label defined by the flash interval containing each bin centre.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is aligned to the same 250 ms trial bins as the neural data. Each bin’s image identity is read out at that bin’s centre time.

ii.
```python
centres = edges[:, :-1] + BIN_SIZE / 2.0
fidx = np.searchsorted(starts, centres.ravel(), side='right') - 1
image_name = names[fidx].reshape(ntrials, NBINS)
...
out = np.stack([img_idx[t], r['image_change'][t], run_q[i][t], pup_q[i][t],
                np.full(T, r['outcome'][t], dtype=np.int64)], axis=0)
```

iii. The notes describe all outputs as being built on the same fixed-bin time base as the binned neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The code derives image change from `dataset.stimulus_presentations.is_change` in the change-detection stimulus table, using the flash associated with each bin centre.

ii.
```python
starts, names, is_change, sp = flash_table(ds)
...
if change_window == 'interval':
    change_flag = is_change[fidx].reshape(ntrials, NBINS).astype(np.int64)
else:
    change_flag = np.zeros((ntrials, NBINS), dtype=np.int64)
    flash_start = starts[fidx].reshape(ntrials, NBINS)
    first_bin = (centres - flash_start) < BIN_SIZE
    change_flag[is_change[fidx].reshape(ntrials, NBINS) & first_bin] = 1
```

iii. The notes say the agent moved away from a trial-table-only definition and instead used the stimulus table so the change output would refer to the actual change flash.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The processing finds which stimulus flash each 250 ms bin centre falls into, checks whether that flash is marked `is_change`, and then emits either a single-bin or full-interval indicator depending on `--change-window`. The default is the single-bin version.

ii.
```python
ap.add_argument('--change-window', choices=['interval', 'bin'], default='bin')
...
fidx = np.searchsorted(starts, centres.ravel(), side='right') - 1
...
first_bin = (centres - flash_start) < BIN_SIZE
change_flag[is_change[fidx].reshape(ntrials, NBINS) & first_bin] = 1
```

iii. The notes say the default was changed to a single 250 ms bin because that matched the agent’s reading of “right after a change” and slightly improved decoding.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is represented as a binary categorical variable: `0` for no change and `1` for change.

ii.
```python
output_values=[images, ['no_change', 'change'],
               [f'q{i+1}' for i in range(NQUANTILES)],
               [f'q{i+1}' for i in range(NQUANTILES)],
               OUTCOMES]
```

iii. The notes describe this output as a binary event label with `no_change` and `change` categories.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned to the same 250 ms bins as neural activity, using the change status of the flash containing each bin centre.

ii.
```python
centres = edges[:, :-1] + BIN_SIZE / 2.0
...
change_flag = np.zeros((ntrials, NBINS), dtype=np.int64)
...
neural_trials.append(np.ascontiguousarray(r['neural'][:, t, :], dtype=np.float32))
...
out = np.stack([img_idx[t], r['image_change'][t], run_q[i][t], pup_q[i][t],
                np.full(T, r['outcome'][t], dtype=np.int64)], axis=0)
```

iii. The notes treat all outputs as living on the same fixed bin grid as the neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, specifically its `speed` and `timestamps` columns.

ii.
```python
rs = ds.running_speed
speed = np.asarray(rs['speed'].values, dtype=float)
speed_t = np.asarray(rs['timestamps'].values, dtype=float)
```

iii. The notes identify `dataset.running_speed` as the AllenSDK locomotion stream and describe it as the correct source for this output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code linearly interpolates NaNs in the raw running signal if any exist, averages running speed within each 250 ms trial bin, and then discretizes each session’s binned values into five equal-percentile bins.

ii.
```python
speed = interpolate_nans(speed)
running_binned = bin_mean_1d(speed, speed_t, edges)
```

```python
run_q, run_edges = discretize([r['running'] for r in results], scope=args.quantile_scope)
...
def discretize(values_list, nq=NQUANTILES, scope='session'):
    allv = np.concatenate([v.ravel() for v in values_list])
    global_qs = np.percentile(allv, np.linspace(0, 100, nq + 1))
    if scope == 'global':
        ...
    else:
        out = []
        for v in values_list:
            qs = np.percentile(v.ravel(), np.linspace(0, 100, nq + 1))
            out.append(np.digitize(v, qs[1:-1], right=False).astype(np.int64))
```

iii. The notes say the agent explicitly changed from global to per-session quintiles because global thresholds partly encoded session identity and performed slightly worse.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five quantile bins, with the default thresholds computed separately within each session.

ii.
```python
ap.add_argument('--quantile-scope', choices=['session', 'global'], default='session')
...
out.append(np.digitize(v, qs[1:-1], right=False).astype(np.int64))
```

iii. In the notes, the agent justifies per-session quintiles by saying running baselines differ across animals and sessions, so session-wise bins are more interpretable than one global threshold set.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by averaging the running samples that fall inside each neural trial bin. The final running output has the same twenty 250 ms bins as the neural matrix.

ii.
```python
def bin_mean_1d(values, sample_times, edges):
    v = values[None, :]
    ones = np.ones_like(v)
    s = bin_sum(v, sample_times, edges)[0]
    n = bin_sum(ones, sample_times, edges)[0]
    ...
    return m
```

```python
running_binned = bin_mean_1d(speed, speed_t, edges)
...
out = np.stack([img_idx[t], r['image_change'][t], run_q[i][t], pup_q[i][t],
                np.full(T, r['outcome'][t], dtype=np.int64)], axis=0)
```

iii. The notes say all streams were intentionally projected into the same flash-locked 250 ms bins.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using both `pupil_width` and `pupil_height`. The code defines diameter as `2 * max(width, height)` at each sample.

ii.
```python
et = ds.eye_tracking
...
diam = 2.0 * np.nanmax(
    np.vstack([et['pupil_width'].values, et['pupil_height'].values]), axis=0)
```

iii. The notes say the agent chose this definition from the whitepaper’s “major axis reflects the pupil diameter” description.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code computes `2 * max(width, height)`, linearly interpolates missing values, averages within each 250 ms trial bin, and discretizes the binned values into quintiles, by default separately for each session.

ii.
```python
diam = 2.0 * np.nanmax(
    np.vstack([et['pupil_width'].values, et['pupil_height'].values]), axis=0)
diam = interpolate_nans(diam)
if diam is not None:
    pupil_binned = bin_mean_1d(diam, np.asarray(et['timestamps'].values, float),
                               edges)
```

```python
pup_q, pup_edges = discretize([r['pupil'] for r in results], scope=args.quantile_scope)
```

iii. The notes justify interpolation as blink handling and session-wise quintiles as protection against rig/session differences in camera-pixel scale.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five quantile bins, with session-wise quantiles by default.

ii.
```python
ap.add_argument('--quantile-scope', choices=['session', 'global'], default='session')
...
out.append(np.digitize(v, qs[1:-1], right=False).astype(np.int64))
```

iii. The notes argue that per-session percentile bins are better justified because raw pupil values are in pixels and are not directly comparable across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by averaging the eye-tracking samples inside each of the same twenty 250 ms neural bins.

ii.
```python
pupil_binned = bin_mean_1d(diam, np.asarray(et['timestamps'].values, float),
                           edges)
...
out = np.stack([img_idx[t], r['image_change'][t], run_q[i][t], pup_q[i][t],
                np.full(T, r['outcome'][t], dtype=np.int64)], axis=0)
```

iii. The notes explicitly frame pupil as another continuous stream resampled onto the common trial-bin time base.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the AllenSDK trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome = np.full(ntrials, -1, dtype=np.int64)
for i, name in enumerate(OUTCOMES):
    outcome[keep[name].values.astype(bool)] = i
```

iii. The notes say these are the four valid go/catch outcomes after excluding aborted and auto-rewarded trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps the four outcome booleans to integer codes `0..3` and then broadcasts each trial’s code across all twenty time bins of that trial.

ii.
```python
outcome = np.full(ntrials, -1, dtype=np.int64)
for i, name in enumerate(OUTCOMES):
    outcome[keep[name].values.astype(bool)] = i
...
out = np.stack([img_idx[t], r['image_change'][t], run_q[i][t], pup_q[i][t],
                np.full(T, r['outcome'][t], dtype=np.int64)], axis=0)
```

iii. The notes describe trial outcome as static per trial and broadcast over bins to fit the decoder format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code fills internal NaNs in continuous behavioral traces by linear interpolation, fills empty behavior bins by interpolation at the bin centre, skips sessions with no usable eye-tracking, skips sessions with fewer than two go/catch trials, and silently filters out `None`/`skip` results after multiprocessing.

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
```

```python
if np.any(n == 0):
    centres = edges[:, :-1] + BIN_SIZE / 2.0
    fill = np.interp(centres, sample_times, values)
    m = np.where(n == 0, fill, m)
```

```python
if len(keep) < 2:
    return None
...
if pupil_binned is None:
    return dict(skip=True, ophys_session_id=int(ophys_session_id),
                reason='no eye tracking')
...
results = [r for r in results if r is not None and not r.get('skip')]
```

iii. The notes say blink NaNs were interpolated, completely missing eye-tracking caused session exclusion, and the fixed trial window was only adopted after checking that it always had ophys coverage.

## 9-a. What are the most time-consuming steps of the code?

i. The agent treats AllenSDK session loading as the dominant runtime cost, not the binning logic. It added multiprocessing across sessions because NWB loading was the bottleneck.

ii.
```python
for k, eid in enumerate(experiment_ids):
    t0 = time.time()
    ds = bc.get_behavior_ophys_experiment(int(eid))
    timings['load'] = timings.get('load', 0) + time.time() - t0
```

```python
if args.workers > 1 and len(jobs) > 1:
    with Pool(min(args.workers, len(jobs))) as p:
        results = p.map(process_session, jobs)
```

iii. `CONVERSION_NOTES.md` Step 6 says NWB loading takes roughly 3-4 s per plane and dominates the runtime; it explicitly describes binning as relatively cheap.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent already vectorized the expensive temporal binning with cumulative sums and `searchsorted`. The remaining obvious Python loops are mostly over sessions, experiments, and trials when assembling outputs and metadata; those could be reduced further, but the agent treats them as secondary because I/O dominates.

ii.
```python
def bin_sum(values_2d, sample_times, edges, chunk=256):
    idx = np.searchsorted(sample_times, edges.ravel(), side='left')
    out = np.empty((nrows, edges.shape[0], NBINS), dtype=np.float64)
    for a in range(0, nrows, chunk):
        ...
        out[a:b] = np.diff(v, axis=2)
```

```python
for i, r in enumerate(results):
    ...
    for t in range(ntrials):
        neural_trials.append(np.ascontiguousarray(r['neural'][:, t, :], dtype=np.float32))
        input_trials.append(np.zeros((0, T), dtype=np.float32))
        out = np.stack([img_idx[t], r['image_change'][t], run_q[i][t], pup_q[i][t],
                        np.full(T, r['outcome'][t], dtype=np.int64)], axis=0)
```

iii. In the notes, the agent emphasizes that it already replaced per-trial temporal slicing with vectorized binning and that extra vectorization would have limited effect relative to file loading.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats some binning-related work: `bin_mean_1d()` calls `bin_sum()` twice for every 1-D signal, and dF/F processing computes one `bin_sum()` for the data and another for frame counts. It also does repeated list-index lookups when building `subject_idx` and `brain_region_idx`.

ii.
```python
def bin_mean_1d(values, sample_times, edges):
    v = values[None, :]
    ones = np.ones_like(v)
    s = bin_sum(v, sample_times, edges)[0]
    n = bin_sum(ones, sample_times, edges)[0]
```

```python
if signal == 'dff':
    counts = bin_sum(np.ones((1, E.shape[1]), dtype=np.float32), ots, edges)
    binned = binned / np.maximum(counts, 1.0)
```

```python
data['subject_idx'].append(subjects.index(r['mouse_id']))
...
np.array([regions.index(x) for x in r['regions']], dtype=np.int64)
```

iii. The agent’s notes do not foreground these as problems, but the code itself shows repeated binning and repeated linear lookups as part of the implementation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code keeps and processes continuous running and pupil arrays only long enough to bin and discretize them, then discards the continuous values from the final pickle. It also builds diagnostic timing/sanity information and optional plotting support that the downstream decoder does not use.

ii.
```python
result = dict(
    ...
    running=running_binned.astype(np.float64),
    pupil=pupil_binned.astype(np.float64),
    ...
    timings=timings,
)
```

```python
run_q, run_edges = discretize([r['running'] for r in results], scope=args.quantile_scope)
pup_q, pup_edges = discretize([r['pupil'] for r in results], scope=args.quantile_scope)
...
out = np.stack([img_idx[t], r['image_change'][t], run_q[i][t], pup_q[i][t],
                np.full(T, r['outcome'][t], dtype=np.int64)], axis=0)
```

```python
if args.show_processing:
    ...
    make_plots(r, images, run_q[i], pup_q[i], out)
```

iii. The notes frame this extra work as validation and interpretability support rather than part of the final decoder payload.
