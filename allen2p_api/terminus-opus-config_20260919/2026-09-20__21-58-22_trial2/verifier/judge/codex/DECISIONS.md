# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent uses `VisualBehaviorOphysProjectCache.from_local_cache('/app/data')`, gets the ophys experiment table, intersects it with locally present experiment files, excludes passive sessions, groups experiments by `ophys_session_id`, and loads every experiment/plane in each selected group through `get_behavior_ophys_experiment`. Session jobs are normally processed by a 16-process pool.

ii.
```python
bc = VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)
et = bc.get_ophys_experiment_table()
local_ids = set(int(f.split('_')[-1].split('.')[0]) for f in os.listdir(
    os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0',
                 'behavior_ophys_experiments')))
et = et.loc[sorted(set(et.index).intersection(local_ids))]
et = et[~et.passive]
...
ds = bc.get_behavior_ophys_experiment(int(eid))
```

iii. The notes say SDK-only loading obeys the requirement and avoids direct NWB access. Restricting to local IDs avoids network fetches; passive sessions are excluded because the requested task is active Visual Behavior. Multiprocessing addresses the dominant file-loading cost.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique string-valued `mouse_id`s among successfully processed sessions; each session gets the index of its mouse in that list.

ii.
```python
subjects = sorted(set(r['mouse_id'] for r in results))
...
data['subject_idx'].append(subjects.index(r['mouse_id']))
```

iii. The notes treat the SDK `mouse_id` as the animal identifier and report 38 locally represented mice.

## 1-c. How are the data split into sessions?

i. A session is one `ophys_session_id`. All locally available active experiments (imaging planes) sharing that ID are combined; neural rows are concatenated across planes. Sessions with no usable eye tracking, or fewer than two go/catch trials, are skipped.

ii.
```python
for sid, sub in et.groupby('ophys_session_id'):
    groups.append((int(sid), sorted(int(i) for i in sub.index)))
...
neural = np.concatenate(neural_planes, axis=0)
```

iii. The agent correctly recognized that an ophys experiment is a plane and an `ophys_session_id` is the simultaneous behavioral recording. It excluded passive sessions and three eye-tracking-free sessions so every retained session supports all requested outputs.

## 1-d. How are the data split into trials?

i. Trials come from `dataset.trials`; rows where `go` or `catch` is true are retained. Rather than retaining the SDK trial from `start_time` to `stop_time`, every trial becomes a fixed window from 2 seconds before to 3 seconds after `change_time`, divided into 20 bins.

ii.
```python
trials = ds.trials
keep = trials[(trials.go | trials.catch)]
change_times = keep.change_time.values.astype(float)
edges = bin_edges_for_trials(change_times)
```

iii. The agent chose `change_time` because it exists for both real go changes and sham catch changes, gives equal-length examples, and supports a common alignment. It states that the window supplies a pre-change baseline and remains inside every trial.

## 1-e. How are trials filtered based on quality controls?

i. Keeping only `go | catch` implicitly removes aborted and auto-rewarded trials. A session is rejected if it has fewer than two retained trials; sessions without usable pupil data are also rejected wholesale.

ii.
```python
keep = trials[(trials.go | trials.catch)]
if len(keep) < 2:
    return None
...
if pupil_binned is None:
    return dict(skip=True, ophys_session_id=int(ophys_session_id),
                reason='no eye tracking')
```

iii. This directly follows the requested trial classes. The notes justify dropping no-eye-tracking sessions because the pupil output otherwise cannot be defined.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. By default neural data comes from each experiment’s SDK `dff_traces['dff']`. Command-line alternatives can use `events['events']` or `events['filtered_events']`.

ii.
```python
if signal == 'dff':
    ev = ds.dff_traces
    E = np.vstack(ev['dff'].values).astype(np.float32)
else:
    ev = ds.events
    E = np.vstack(ev[signal].values).astype(np.float32)
```

iii. The agent acknowledges that the paper used detected calcium events, but selected dF/F after decoder comparisons showed better performance on four of five outputs and eliminated all-zero sparse trials.

## 2-b. How is the `neural` data processed?

i. Each plane is binned using timestamp-based cumulative sums. Default dF/F is averaged per 250-ms bin; event alternatives are summed. Binned planes are concatenated along the neuron axis and stored as float32.

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

iii. Averaging dF/F avoids artificial scaling when 250-ms bins contain different frame counts on 11-Hz versus 31-Hz rigs. Cumulative-sum binning was chosen for speed and well-defined empty-bin behavior.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional cell-level filtering is performed. The code relies on SDK-delivered traces, which already exclude invalid ROIs. It excludes passive sessions and sessions lacking eye tracking.

ii.
```python
E = np.vstack(ev['dff'].values).astype(np.float32)
...
et = et[~et.passive]
```

iii. The notes state that AllenSDK’s default `exclude_invalid_rois=True` embodies pipeline ROI QC and that the paper adds no further cell filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are aligned to `trials.change_time`, including sham change time for catch trials, using bin edges spanning [-2, +3] seconds.

ii.
```python
def bin_edges_for_trials(change_times):
    offsets = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    return change_times[:, None] + offsets[None, :]
```

iii. The notes call `change_time` the natural shared event for go and catch trials and verify it coincides with a flash onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 250 ms. Native ophys frames are explicitly rebinned into 20 bins per fixed 5-second window.

ii.
```python
BIN_SIZE = 0.25
OFF_START = -2.0
OFF_END = 3.0
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

iii. The agent chose 250 ms to equal the image-on duration and one third of the 750-ms flash cycle, making bins phase-locked to the stimulus.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `dataset.stimulus_presentations`: `start_time`, `image_name`, and `stimulus_block_name`. Omitted image names are replaced by neighboring image identity.

ii.
```python
sp = dataset.stimulus_presentations
sp = sp[sp.stimulus_block_name.str.contains('change_detection')]
sp = sp.sort_values('start_time')
names = sp.image_name.astype(str).replace('omitted', np.nan).ffill().bfill().values
```

iii. The notes prefer the flash table because it represents the actual stimulus stream, including omissions, over reconstructing identity solely from trial columns.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For every output-bin center, the most recent change-detection flash is found. Its forward/back-filled image name is assigned, then all dataset-wide names are sorted and encoded as integers.

ii.
```python
centres = edges[:, :-1] + BIN_SIZE / 2.0
fidx = np.searchsorted(starts, centres.ravel(), side='right') - 1
fidx = np.clip(fidx, 0, len(starts) - 1)
image_name = names[fidx].reshape(ntrials, NBINS)
...
img_idx = np.vectorize(img_index.get)(r['image_name']).astype(np.int64)
```

iii. The agent defines identity across the full 750-ms presentation interval, including gray periods; omitted flashes inherit the prior identity.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is sampled at the centers of the exact same change-aligned 250-ms bins used for neural aggregation, producing an `(ntrials, 20)` array.

ii.
```python
centres = edges[:, :-1] + BIN_SIZE / 2.0
...
image_name = names[fidx].reshape(ntrials, NBINS)
```

iii. The agent reports plots and independent checks showing the image switches in the change bin without a temporal shift.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from `stimulus_presentations.start_time` and `is_change` within the change-detection block.

ii.
```python
return (sp.start_time.values.astype(float), names,
        sp.is_change.values.astype(bool), sp)
```

iii. The notes identify `is_change` as the SDK’s canonical indicator for an actual new image and distinguish it from a catch trial’s sham change time.

## 4-b. What processing is involved in computing `output` *Image change*?

i. In the default `bin` mode, the code locates the flash containing each bin center and sets 1 only if it is a true change flash and the bin center is within its first 250 ms. An optional mode marks all three bins of the 750-ms interval.

ii.
```python
change_flag = np.zeros((ntrials, NBINS), dtype=np.int64)
flash_start = starts[fidx].reshape(ntrials, NBINS)
first_bin = (centres - flash_start) < BIN_SIZE
change_flag[is_change[fidx].reshape(ntrials, NBINS) & first_bin] = 1
```

iii. A controlled decoder comparison favored the single change bin, and the agent viewed it as the literal meaning of “right after a change.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is directly binary: 0 (`no_change`) or 1 (`change`); there is no numeric threshold beyond the boolean SDK `is_change` and first-bin timing test.

ii.
```python
output_values=[images, ['no_change', 'change'], ...]
```

iii. The agent treats `is_change` as already categorical and verifies go trials have one positive bin while catch trials have none.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It has one value at each shared bin center. With the default window, the positive value is expected at bin 8 (0 to 0.25 seconds) for every go trial.

ii.
```python
expect[int((0 - OFF_START) / BIN_SIZE):
       int((0 - OFF_START) / BIN_SIZE) + nchange_bins] = 1
```

iii. The notes validate the expected pattern against raw SDK presentations and require catch trials to remain zero.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from `dataset.running_speed['speed']` and its `timestamps`.

ii.
```python
rs = ds.running_speed
speed = np.asarray(rs['speed'].values, dtype=float)
speed_t = np.asarray(rs['timestamps'].values, dtype=float)
```

iii. The notes use the SDK’s processed wheel speed in cm/s, already low-pass filtered and cleaned by the release pipeline.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. NaNs are linearly interpolated, speed samples are averaged within each 250-ms trial bin, and those means are discretized into quintiles.

ii.
```python
speed = interpolate_nans(speed)
running_binned = bin_mean_1d(speed, speed_t, edges)
...
qs = np.percentile(v.ravel(), np.linspace(0, 100, nq + 1))
out.append(np.digitize(v, qs[1:-1], right=False).astype(np.int64))
```

iii. Bin means respect the higher-rate behavior clock. Interpolation is a fallback for missing samples; quintiles create the requested categorical output.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Default thresholds are the 0/20/40/60/80/100 percentiles computed separately within each session over all retained trial-bin means. A global option exists.

ii.
```python
run_q, run_edges = discretize([r['running'] for r in results],
                              scope=args.quantile_scope)
```

iii. The agent argues per-session bins avoid encoding session-specific running baselines and gave slightly better decoder results, while guaranteeing approximately balanced classes per session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running samples are averaged directly within the same absolute timestamp edges used for each neural bin. Empty behavior bins fall back to linear interpolation at the bin center.

ii.
```python
running_binned = bin_mean_1d(speed, speed_t, edges)
```

iii. Shared hardware-clock timestamps and identical edges provide alignment without first resampling the complete stream.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It comes from `dataset.eye_tracking['pupil_width']`, `['pupil_height']`, and `['timestamps']`. The diameter proxy is twice the larger ellipse radius/axis value.

ii.
```python
et = ds.eye_tracking
diam = 2.0 * np.nanmax(
    np.vstack([et['pupil_width'].values, et['pupil_height'].values]), axis=0)
```

iii. The agent says `2 * max(width, height)` follows the SDK circular-area convention and is a more diameter-like measure than width alone.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaNs, including blink-related NaNs, are linearly interpolated with nearest-value edge filling. The result is averaged in 250-ms bins and converted to quintiles. A session with no usable finite pupil series is skipped.

ii.
```python
diam = interpolate_nans(diam)
if diam is not None:
    pupil_binned = bin_mean_1d(diam, np.asarray(et['timestamps'].values, float), edges)
```

iii. Interpolation avoids blink artifacts becoming missing outputs. The notes report a median 2.9% blink-frame rate and exclude only three sessions with no eye tracking.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Default thresholds are session-specific quintiles of all pupil bin means; the code can instead use global thresholds.

ii.
```python
pup_q, pup_edges = discretize([r['pupil'] for r in results],
                              scope=args.quantile_scope)
```

iii. The agent argues pupil values are camera pixels and therefore not comparable across rig geometry/session; per-session quintiles balance categories and reduce session-identity leakage.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are averaged over the exact same timestamp edges as neural bins, with center interpolation only if a bin has no eye samples.

ii.
```python
pupil_binned = bin_mean_1d(diam, np.asarray(et['timestamps'].values, float),
                           edges)
```

iii. The notes cite synchronized clocks and raw-versus-binned plots as alignment checks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the retained trial table’s mutually exclusive `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
for i, name in enumerate(OUTCOMES):
    outcome[keep[name].values.astype(bool)] = i
```

iii. These are the four canonical outcomes for go and catch trials; an assertion ensures every retained trial receives one.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Booleans are mapped to fixed integer codes 0–3, then each scalar code is broadcast across all 20 time bins of its trial.

ii.
```python
np.full(T, r['outcome'][t], dtype=np.int64)
```

iii. Broadcasting makes the static label compatible with the common time-varying output matrix while retaining its per-trial meaning.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. NaNs in running and pupil streams are linearly interpolated, with nearest finite values at edges; empty temporal bins use center interpolation. Sessions with no finite pupil data or too few trials are skipped. Local IDs prevent attempts to load absent experiments. Assertions and sanity checks detect invalid outcomes and temporal patterns.

ii.
```python
good = np.isfinite(v)
if not np.any(good):
    return None
v[~good] = np.interp(x[~good], x[good], v[good])
...
m = np.where(n == 0, fill, m)
```

iii. The agent favors interpolation for short blink/missing stretches and exclusion when a required output is wholly unavailable. It reports independent raw-data checks and clean format verification.

## 9-a. What are the most time-consuming steps of the code?

i. Loading the large SDK experiment/NWB-backed objects is dominant (about 3–4 seconds per plane); binning and behavior processing are reported below 0.2 seconds per session.

ii.
```python
ds = bc.get_behavior_ophys_experiment(int(eid))
timings['load'] = timings.get('load', 0) + time.time() - t0
```

iii. The notes measured an approximately 83–87 second full run with 16 workers and identify I/O as the bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Most costly trial/bin work is already vectorized. Remaining small loops include per-plane loading, chunking neuron rows, assigning four outcome classes, converting each session into trial lists, and repeated list-based index lookup for subjects/regions.

ii.
```python
for a in range(0, nrows, chunk):
    ...
for t in range(ntrials):
    neural_trials.append(...)
```

iii. The agent explicitly replaced per-trial neural binning with cumulative sums and parallelized sessions. It considers loading, not these residual loops, the limiting cost.

## 9-c. What processing does the code repeat multiple times?

i. `bin_sum` is invoked twice for each one-dimensional bin mean (once for values and once for counts), and dF/F binning invokes it again for frame counts. Each worker independently constructs an SDK cache. Assembly repeatedly performs `subjects.index` and `regions.index`. Optional plotting reloads a dataset and recomputes signals.

ii.
```python
s = bin_sum(v, sample_times, edges)[0]
n = bin_sum(ones, sample_times, edges)[0]
...
counts = bin_sum(np.ones((1, E.shape[1]), dtype=np.float32), ots, edges)
```

iii. The notes emphasize a single main processing pass per plane; repeated count calculations are used to obtain correct bin means, and plotting recomputation is optional validation work.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It retains diagnostic fields such as continuous binned running/pupil, change times, go/catch flags, detailed timing information, equipment metadata, and experiment IDs until assembly, though only categorized outputs and selected metadata are saved. It also calculates global quantile edges even in per-session mode, solely for reporting/metadata. Optional plotting loads event traces even when the delivered neural signal is dF/F.

ii.
```python
allv = np.concatenate([v.ravel() for v in values_list])
global_qs = np.percentile(allv, np.linspace(0, 100, nq + 1))
...
ev = ds.events
E = np.vstack(ev['events'].values).astype(np.float32)
```

iii. These discarded intermediates support sanity checks, plots, provenance, and quantile reporting rather than decoder input. The agent prioritizes auditability, while noting loading dominates runtime.
