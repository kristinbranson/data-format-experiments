# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent opens the local AllenSDK cache, intersects the experiment table with the NWB files actually present, removes passive experiments, groups the remaining experiments by `ophys_session_id`, and loads every experiment/plane in each group. It processes sessions in a multiprocessing pool. Thus “all” means all locally available active experiments, including both `VisualBehavior` and `VisualBehaviorMultiscope`, not every row in the release metadata.

ii.
```python
cache = get_cache()
exp_table = cache.get_ophys_experiment_table()
local_ids = sorted(int(f.split('_')[-1].split('.')[0])
                   for f in os.listdir(nwb_dir) if f.endswith('.nwb'))
et = exp_table.loc[exp_table.index.intersection(local_ids)]
et = et[~et.passive]
datasets = [cache.get_behavior_ophys_experiment(int(e)) for e in exp_ids]
```

iii. The notes say the local cache contains only 284 of 1,936 release experiments, passive sessions have no licks or rewards and were not analyzed in the paper, and active sessions best represent the requested behavioral task. Local-file intersection prevents attempted downloads; parallel loading addresses the main I/O bottleneck.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id`s from successfully converted sessions; each session receives an index into the sorted subject list.

ii.
```python
subjects = sorted({r['mouse_id'] for r in good})
subject_idx = np.array([subjects.index(r['mouse_id']) for r in good], dtype=np.int64)
```

iii. The agent identifies `mouse_id` as the SDK’s animal identifier and validates the resulting mouse count against local metadata.

## 1-c. How are the data split into sessions?

i. A session is one `ophys_session_id`. All experiments/imaging planes with that ID are loaded together and their neurons are concatenated; outputs and behavior come from the first plane after trial-table consistency checks.

ii.
```python
session_ids = sorted(et.ophys_session_id.unique())
rows = et[et.ophys_session_id == sid].sort_index()
jobs.append((int(sid), list(rows.index), rows, ...))
datasets = [cache.get_behavior_ophys_experiment(int(e)) for e in exp_ids]
```

iii. The notes explain that simultaneously acquired Multiscope planes share behavior, stimulus, and clock, so `ophys_session_id` is the correct grouping key.

## 1-d. How are the data split into trials?

i. Trials come from the SDK `trials` table. Each retained row is represented by a fixed 6 s window from 2 s before through 4 s after `change_time` (the sham change on catch trials), yielding 60 bins rather than the full `start_time`–`stop_time` interval.

ii.
```python
keep = (trials.go | trials.catch) & trials.change_time.notna()
tr = trials[keep]
change_times = tr.change_time.values.astype(np.float64)
for ct in change_times:
    edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
```

iii. The agent argues that change is the event defining the task, that every observed window remains inside its trial and all data streams, and that a common fixed length simplifies decoding.

## 1-e. How are trials filtered based on quality controls?

i. Only rows explicitly labeled `go` or `catch` with nonmissing `change_time` are kept, which excludes aborted and auto-rewarded trials. Sessions with fewer than two such trials are dropped. Sessions lacking eye tracking (or usable pupil data) are also dropped at session level.

ii.
```python
keep = (trials.go | trials.catch) & trials.change_time.notna()
tr = trials[keep]
if len(tr) < 2:
    return {'session_id': ophys_session_id, 'skipped': 'fewer than 2 go/catch trials'}
```

iii. This directly follows the task’s go/catch inclusion rule. The notes cite the SDK warning that auto rewards bias choice and explain that pupil is a required output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. By default, neural data are derived from each experiment’s Allen-pipeline `dff_traces.dff`. The CLI also permits `events.events` or `events.filtered_events`.

ii.
```python
if neural_signal == 'dff':
    ev = np.vstack(d.dff_traces.dff.values).astype(np.float32)
elif neural_signal == 'filtered_events':
    ev = np.vstack(d.events.filtered_events.values).astype(np.float32)
else:
    ev = np.vstack(d.events.events.values).astype(np.float32)
```

iii. Although the paper analyzed detected events, the agent benchmarked all three streams and chose dF/F because it decoded four of five targets better; the notes call this a deliberate decoder-oriented deviation.

## 2-b. How is the `neural` data processed?

i. For every plane and trial, dF/F samples are averaged into 100 ms bins using that plane’s own ophys timestamps, then planes are vertically stacked. The released traces otherwise receive no normalization or filtering.

ii.
```python
b, counts = bin_average(ev, ts, edges)
parts.append(b)
neural_trials.append(np.vstack(parts).astype(np.float32))
```

iii. The notes say Allen already performs motion/ROI processing, demixing, neuropil subtraction, and dF/F computation. Bin averaging creates one common resolution without upsampling the slower 11 Hz recordings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level filter is applied; all released trace rows are used. Sessions with no neurons are skipped, and empty-bin/finite-value checks are recorded.

ii.
```python
n_neurons = sum(e.shape[0] for e in plane_events)
if n_neurons == 0:
    return {'session_id': ophys_session_id, 'skipped': 'no neurons'}
```

iii. The agent states that released cells already satisfy `valid_roi == True` and upstream QC has removed invalid ROIs and artifacts.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are aligned to `trials.change_time`; catch trials use the SDK sham-change time. Raw samples enter bins according to each plane’s `ophys_timestamps`.

ii.
```python
edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
b, counts = bin_average(ev, ts, edges)
```

iii. The notes say all clocks are hardware-synchronized, the paper aligns event-triggered responses to image changes, and the use of ophys timestamps satisfies the requested ophys-time alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is 100 ms. Neural, running, and pupil samples are mean-binned into 60 bins per trial.

ii.
```python
BIN_SIZE = 0.1
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

iii. The agent chose 100 ms as the smallest round interval at least as long as the slowest 11 Hz frame interval, avoiding artificial upsampling while preserving useful decoder resolution.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `stimulus_presentations.image_name`, after restricting rows to the change-detection stimulus block; the global class list contains the 16 A/B-set images plus `omitted`.

ii.
```python
sp = d0.stimulus_presentations
sp = sp[sp.stimulus_block_name.str.contains('change_detection')]
stim_img = sp.image_name.values.astype(str)
stim_code = np.array([img_to_code[n] for n in stim_img], dtype=np.int64)
```

iii. The notes cite the paper’s image-presentation-interval convention and the tutorials’ warning that newer SDK tables also contain gray-screen and movie blocks.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each stimulus flash is mapped to a global integer code. Its interval ends at the next flash onset, capped at one second. Each output bin receives the interval containing its center, or `omitted` outside an interval; identity is held through the gray portion of the 750 ms cycle.

ii.
```python
stim_end[:-1] = stim_start[1:]
stim_end[-1] = stim_start[-1] + FLASH_INTERVAL
stim_end = np.minimum(stim_end, stim_start + MAX_FLASH_INTERVAL)
j = np.searchsorted(stim_start, centers, side='right') - 1
within = centers < stim_end[j]
img = np.where(within, stim_code[j], img_to_code['omitted'])
```

iii. The agent says using actual next-flash onset prevents unlabeled slivers, holding identity across gray matches the paper’s 750 ms interval definition, and an omitted class faithfully represents omission trials.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The image class is evaluated at the center of each of the same change-aligned 100 ms bins used for neural binning, so both arrays have 60 columns.

ii.
```python
centers = 0.5 * (edges[:-1] + edges[1:])
j = np.searchsorted(stim_start, centers, side='right') - 1
img_trials.append(img.astype(np.int64))
```

iii. The notes report exact checks around time zero and visual comparisons against the raw flash table.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from `stimulus_presentations.is_change` and the corresponding presentation timing in the filtered change-detection block.

ii.
```python
stim_change = sp.is_change.values.astype(bool)
chg = np.where(within, stim_change[j], False).astype(np.int64)
```

iii. The agent uses the SDK’s canonical per-presentation change flag so catch sham changes remain zero and real go changes are one.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The presentation’s boolean change flag is held over its full image-presentation interval (actual next onset, capped at one second), then converted to integer labels.

ii.
```python
within = centers < stim_end[j]
chg = np.where(within, stim_change[j], False).astype(np.int64)
```

iii. The notes interpret “right after a change” as the paper’s 750 ms presentation interval and validate approximately 7–8 positive 100 ms bins per go trial and none per catch trial.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is estimated: the raw boolean `is_change` becomes category 1, and all other bins become 0.

ii.
```python
change_trials.append(chg)
# output_values: ['no_change', 'change']
```

iii. The source is already binary, so additional thresholding is unnecessary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is sampled at the centers of the same 100 ms bins as image identity and neural activity.

ii.
```python
centers = 0.5 * (edges[:-1] + edges[1:])
chg = np.where(within, stim_change[j], False).astype(np.int64)
```

iii. Shared bin edges guarantee column-wise alignment; raw-data validations confirmed onset at the change bin.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from `running_speed.speed` and `running_speed.timestamps` on the first experiment in the session.

ii.
```python
rs = d0.running_speed
run_ts = rs.timestamps.values.astype(np.float64)
run_sp = interpolate_nans(rs.speed.values)
```

iii. The notes identify this as the SDK’s standard 10 Hz-low-pass-filtered wheel-speed stream in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Nonfinite values are linearly interpolated, raw samples are averaged within each 100 ms trial bin, empty bins are filled by interpolation at the bin center, and values are later discretized.

ii.
```python
rb, rc = bin_average(run_sp[None, :], run_ts, edges)
rb = rb[0]
if (rc == 0).any():
    rb[rc == 0] = np.interp(centers[rc == 0], run_ts, run_sp)
```

iii. The agent chose bin means to match the common temporal grid and center interpolation so dropped samples do not become false zeros.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. All extracted running values within each session are divided by that session’s 20th, 40th, 60th, and 80th percentiles into labels 0–4.

ii.
```python
run_mat = np.vstack(run_trials)
run_lab, run_q = quantile_bins(run_mat.ravel())
```

iii. Per-session quintiles satisfy “five equal percentile bins” and normalize large mouse/rig differences so labels describe relative speed.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running samples are binned directly on their synchronized timestamps using exactly the neural trial edges, producing one value per 100 ms neural bin.

ii.
```python
rb, rc = bin_average(run_sp[None, :], run_ts, edges)
run_trials.append(rb)
```

iii. The notes rely on upstream hardware synchronization and shared edges rather than first resampling onto ophys frames.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It is computed from `eye_tracking.pupil_area` and `eye_tracking.timestamps`; blink/failed-fit samples are represented as NaN in `pupil_area`.

ii.
```python
eye_ts = eye.timestamps.values.astype(np.float64)
pupil_area = eye.pupil_area.values.astype(np.float64)
pupil_diam_raw = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. The agent treats the diameter of the area-equivalent circle as pupil diameter and notes that the SDK already masks likely blinks in this field.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area is converted to equivalent-circle diameter, NaNs are linearly interpolated with nearest valid edge behavior, samples are averaged within 100 ms bins, and empty bins are filled at their centers before discretization.

ii.
```python
pupil_diam = interpolate_nans(pupil_diam_raw)
pb, pc = bin_average(pupil_diam[None, :], eye_ts, edges)
if (pc == 0).any():
    pb[pc == 0] = np.interp(centers[pc == 0], eye_ts, pupil_diam)
```

iii. The notes call blink-gap interpolation standard practice and use it to avoid propagating missingness or inserting zeros.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. All binned pupil values in a session are split into session-specific quintiles labeled 0–4.

ii.
```python
pupil_mat = np.vstack(pupil_trials)
pupil_lab, pupil_q = quantile_bins(pupil_mat.ravel())
```

iii. The agent argues that camera-pixel diameter is not comparable across rigs and mice; within-session quintiles provide consistent relative-arousal categories and equal occupancy.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Eye samples are binned on native synchronized eye timestamps using the identical change-relative edges used for neural data.

ii.
```python
pb, pc = bin_average(pupil_diam[None, :], eye_ts, edges)
pupil_trials.append(pb)
```

iii. Shared trial edges produce exact column alignment while retaining all available eye-camera samples.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from the mutually exclusive `hit`, `miss`, `false_alarm`, and `correct_reject` columns of the retained trial rows.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
for k, name in enumerate(OUTCOMES):
    outcome[tr[name].values.astype(bool)] = k
```

iii. The notes identify these as the SDK’s canonical outcome flags and validate their counts against metadata.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Each true flag maps to a fixed integer 0–3. The code asserts every retained trial has a label and broadcasts that static label across all 60 time bins.

ii.
```python
assert (outcome >= 0).all(), 'trial with no outcome label'
np.full(NBINS, outcome[i], dtype=np.int64)
```

iii. Broadcasting makes all outputs share a time dimension while preserving the per-trial meaning.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing local experiments are excluded before loading. Sessions with no eye table, all-NaN pupil, fewer than two trials, or no neurons are skipped. NaNs in running/pupil are interpolated; bins with dropped samples are center-interpolated. Per-session exceptions are returned and reported without terminating the run. Plot failures are explicitly nonfatal.

ii.
```python
if eye is None or len(eye) == 0:
    return {'session_id': ophys_session_id, 'skipped': 'no eye tracking'}
if bad.all():
    return None
except Exception as exc:
    return {'session_id': int(ophys_session_id), 'error': f'{exc}', ...}
```

iii. The rationale is to retain valid sessions, avoid fabricating zero behavior during dropouts, and ensure one corrupt session or optional diagnostic cannot abort the full conversion.

## 9-a. What are the most time-consuming steps of the code?

i. NWB loading is identified as the bottleneck (about three seconds per experiment); trial binning is secondary. Sessions are therefore processed with up to 16 worker processes, and per-stage timing is recorded.

ii.
```python
with Pool(min(args.workers, len(jobs))) as pool:
    for i, r in enumerate(pool.imap_unordered(process_session, jobs)):
        ...
timing['load'] = time.time() - t0
```

iii. The notes state that each large NWB is loaded once and that multiprocessing materially reduces wall time.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent already vectorized the expensive loop over bins and neurons using search indices and cumulative sums. Remaining Python loops are over sessions, trials, and imaging planes; trials could be batched further, though variable windows and modest cost make the current structure reasonable. Outcome assignment and region-index lookups could also be vectorized but are negligible.

ii.
```python
idx = np.searchsorted(sub_ts, edges, side='left')
csum = np.concatenate([np.zeros((sub.shape[0], 1)),
                       np.cumsum(sub.astype(np.float64), axis=1)], axis=1)
sums = csum[:, idx[1:]] - csum[:, idx[:-1]]
```

iii. The notes report roughly a 20-fold binning speedup and say I/O, not these residual loops, dominates runtime.

## 9-c. What processing does the code repeat multiple times?

i. Within each trial it repeats bin-edge construction and separate calls to the same binning routine for each neural plane, running, and pupil. It also creates a new cache object in every worker job. It does not reload an NWB within a session, and final assembly reuses returned arrays.

ii.
```python
for ct in change_times:
    edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    for ev, ts in zip(plane_events, plane_ts):
        b, counts = bin_average(ev, ts, edges)
    rb, rc = bin_average(run_sp[None, :], run_ts, edges)
    pb, pc = bin_average(pupil_diam[None, :], eye_ts, edges)
```

iii. The notes emphasize that each NWB and neural stream is read once; the repeated calls are required for streams with different timestamps.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The default conversion retains some diagnostic-only work and metadata—timing counters, cell IDs, per-session QC counts, and quantile thresholds—that the decoder does not consume. With `--show-processing`, it additionally reloads/plots example raw traces and recomputes display-only binned running/pupil values; those plots are optional and not stored in the pickle’s decoder tensors.

ii.
```python
if show_processing:
    make_processing_plot(...)
result = {..., 'cell_specimen_ids': ..., 'timing': timing, ...}
```

iii. These values support auditability, validation, and visualization rather than decoding. The agent deliberately made plotting opt-in and nonfatal, so little costly processing is unnecessarily discarded in the normal full run.
