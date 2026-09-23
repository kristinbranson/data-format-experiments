# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK project cache. It reads the local metadata CSV `ophys_experiment_table.csv`, restricts to locally present NWB files, keeps only `project_code == 'VisualBehavior'`, further restricts to `passive == False`, and then loads each selected experiment directly from disk with `BehaviorOphysExperiment.from_nwb_path`. Trials are then read from each experiment’s `ds.trials` table.

ii. 
```python
et = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
present = set()
for fn in os.listdir(NWB_DIR):
    if fn.endswith('.nwb'):
        present.add(int(fn.split('_')[-1].split('.')[0]))
sel = et[et.ophys_experiment_id.isin(present)
         & (et.project_code == PROJECT_CODE)
         & (~et.passive.astype(bool))].copy()
...
path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{oeid}.nwb')
ds = BehaviorOphysExperiment.from_nwb_path(path)
...
trials = ds.trials
```

iii. In `CONVERSION_NOTES.md`, the AI says it switched away from `VisualBehaviorOphysProjectCache` because the local cache layout was missing the SDK manifest directory, and that `from_nwb_path` returns the same object type. It also justifies the `passive == False` filter by citing the paper’s focus on active behavior and the need for uniform frame rate.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values from each kept session’s metadata, stored as strings and sorted globally when assembling the final dataset.

ii. 
```python
'subject': str(md['mouse_id']),
...
subjects = sorted({r['subject'] for r in kept})
subject_index = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_index[r['subject']] for r in kept], dtype=np.int64),
```

iii. The notes say subject identity comes from `metadata['mouse_id']` and is recorded as unique sorted IDs.

## 1-c. How are the data split into sessions?

i. The AI treats each selected `ophys_experiment_id` as one session. It explicitly relies on the choice of the single-plane `VisualBehavior` project, where it claims experiment and session are effectively the same for this local dataset.

ii. 
```python
def process_experiment(job):
    """Convert one ophys experiment (== one session in this project)."""
    oeid = job['ophys_experiment_id']
...
jobs.append({'ophys_experiment_id': int(row.ophys_experiment_id), ...})
...
'ophys_session_id': int(md['ophys_session_id']),
```

iii. `CONVERSION_NOTES.md` states, “For this project, experiment ≡ session; a converted ‘session’ is one NWB file,” and justifies that by restricting to the single-plane `VisualBehavior` project.

## 1-d. How are the data split into trials?

i. Trials are taken from `ds.trials`. The AI keeps only rows where `go` or `catch` is true, then uses the experiment-defined window `[start_time, stop_time)` and slices the ophys-indexed arrays with `np.searchsorted`.

ii. 
```python
trials = ds.trials
go = trials['go'].values.astype(bool)
catch = trials['catch'].values.astype(bool)
keep = go | catch
tr = trials[keep]

starts = tr['start_time'].values.astype(np.float64)
stops = tr['stop_time'].values.astype(np.float64)
i0 = np.searchsorted(ts, starts, side='left')
i1 = np.searchsorted(ts, stops, side='left')
```

iii. The notes say this reproduces the tutorial’s use of the natural trial window from `start_time` to `stop_time`, rather than a fixed window around `change_time`.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes all non-`go` and non-`catch` trials by construction, drops trials with fewer than 2 ophys frames, drops trials with no valid non-blink pupil sample, and drops whole sessions left with fewer than 2 usable trials.

ii. 
```python
keep = go | catch
tr = trials[keep]
...
for k in range(len(tr)):
    if i1[k] - i0[k] < MIN_FRAMES_PER_TRIAL:
        n_drop_short += 1
        continue
    if j1[k] - j0[k] < 1:
        n_drop_pupil += 1
        continue
    sel.append(k)
...
if len(sel) < MIN_TRIALS_PER_SESSION:
    return {'oeid': int(oeid), 'skip': f'only {len(sel)} usable trials'}
```

iii. The notes justify the pupil-based exclusion by saying pupil diameter is a required decoder output, so a trial with no real pupil sample would otherwise be pure extrapolation. They also mention the decoder requirement for at least two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final code uses `ds.dff_traces['dff']` as the default neural signal. The code also supports `events` and `filtered_events` behind a flag, but the documented final choice is dF/F.

ii. 
```python
signal = job.get('neural_signal', 'dff')
if signal == 'dff':
    col = ds.dff_traces['dff']
else:
    col = ds.events[signal]
```

iii. In Step 7 and Step 10 of `CONVERSION_NOTES.md`, the AI says it benchmarked `events`, `filtered_events`, and `dff`, and kept dF/F because it decoded better on every output in this per-timepoint task.

## 2-b. How is the `neural` data processed?

i. The AI filters to `valid_roi` cells, extracts each selected cell’s full-session trace into a `float32` matrix, optionally z-scores only if a non-default flag is set, and then slices that matrix into per-trial arrays without further temporal smoothing or rebinning.

ii. 
```python
cst = ds.cell_specimen_table
valid_roi = cst['valid_roi'].values.astype(bool)
cell_ids = cst.index.values[valid_roi]
...
traces = np.empty((len(cell_ids), len(ts)), dtype=np.float32)
for i, cid in enumerate(cell_ids):
    traces[i] = col.loc[cid]
if job.get('zscore', False):
    sd = traces.std(axis=1, keepdims=True)
    traces = (traces - traces.mean(axis=1, keepdims=True)) / np.maximum(sd, 1e-6)
...
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The notes say dF/F is already released as a processed AllenSDK product, z-scoring was benchmarked and not chosen, and no extra smoothing/rebinning was needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only cells with `valid_roi == True` and skips a session if no valid ROIs remain. It does not apply any additional activity-based or trial-based neuron filtering.

ii. 
```python
cst = ds.cell_specimen_table
valid_roi = cst['valid_roi'].values.astype(bool)
cell_ids = cst.index.values[valid_roi]
if len(cell_ids) == 0:
    return {'oeid': int(oeid), 'skip': 'no valid ROIs'}
```

iii. The notes say this is mainly an explicit guard, because the local sessions reportedly already have all cells marked `valid_roi == True`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start. The code converts each trial’s `start_time` and `stop_time` into indices on the ophys timestamp clock, then uses that slice for the neural matrix.

ii. 
```python
starts = tr['start_time'].values.astype(np.float64)
stops = tr['stop_time'].values.astype(np.float64)
i0 = np.searchsorted(ts, starts, side='left')
i1 = np.searchsorted(ts, stops, side='left')
...
a, b = i0[k], i1[k]
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The notes explicitly record `trial_start` / `trials.start_time` as the alignment event and say all streams are put on the ophys timestamp clock within `[start_time, stop_time)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay on the native ophys frame grid. The script records the median ophys frame period as the time bin size and does not apply any temporal rebinning.

ii. 
```python
'time_bin_size': float(np.median(frame_periods) * 1000.0),
'sampling_rate_hz': float(1.0 / np.median(frame_periods)),
...
assert np.isclose(frame_periods.min(), frame_periods.max(), atol=1e-4)
```

iii. The notes justify selecting the single-plane `VisualBehavior` project partly to preserve one uniform native frame rate across all sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The AI derives image identity from `ds.stimulus_presentations`, using `start_time`, `end_time`, `image_name`, `omitted`, and `stimulus_block_name`, rather than from the trial table’s `initial_image_name` / `change_image_name`.

ii. 
```python
sp = stimulus_presentations
sp = sp[sp['stimulus_block_name'].astype(str).str.contains('change_detection')]
start = sp['start_time'].values.astype(np.float64)
end = sp['end_time'].values.astype(np.float64)
names = sp['image_name'].astype(str).values
...
codes = np.array([IMAGE_CODE.get(n, 0) for n in names], dtype=np.int16)
```

iii. The notes say this was chosen so the label reflects the image actually shown during each flash, with grey screen represented explicitly outside flashed intervals and during omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI hard-codes the 16 known natural-image names, reserves code 0 for grey, and assigns each ophys frame the flashed image code only if the frame falls inside a non-omitted presentation interval. Frames outside a flash get code 0.

ii. 
```python
IMAGE_VALUES = ['grey'] + IMAGE_NAMES
IMAGE_CODE = {name: i + 1 for i, name in enumerate(IMAGE_NAMES)}
...
idx = np.searchsorted(start, ts, side='right') - 1
valid = idx >= 0
idx_c = np.clip(idx, 0, len(start) - 1)
on_screen = valid & (ts < np.nan_to_num(end[idx_c], nan=-np.inf))

image = np.where(on_screen, codes[idx_c], 0).astype(np.int16)
```

iii. The notes justify the grey class by the task wording “image presented during the non-grey screen” and by the 250 ms flash / 500 ms grey structure described in the paper.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed for the full ophys timestamp vector of the session, then trial windows slice the same `[a:b]` indices used for neural data.

ii. 
```python
image_all, change_all, sp = stimulus_traces(
    ds.stimulus_presentations, ts, job.get('change_window', 'flash'))
...
out[0] = image_all[a:b]
...
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The notes repeatedly state that all streams are aligned on `ophys_timestamps`, so the label arrays and neural arrays share identical frame indices.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The AI derives image change from `stimulus_presentations.is_change` on the flashed stimulus table, not from `trials.change_time` plus `trials.go`.

ii. 
```python
names = sp['image_name'].astype(str).values
is_change = sp['is_change'].values.astype(bool)
...
if change_window == 'interval':
    change = (valid & is_change[idx_c]).astype(np.int16)
else:
    change = (on_screen & is_change[idx_c]).astype(np.int16)
```

iii. The notes say this directly marks the changed flash, while catch trials have `is_sham_change` and therefore naturally remain zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. By default, the AI labels `image_change = 1` only on frames that fall inside the changed 250 ms flash itself. It also includes an alternate `interval` mode for a 750 ms label window, but does not use that by default.

ii. 
```python
if change_window == 'interval':
    change = (valid & is_change[idx_c]).astype(np.int16)
else:
    change = (on_screen & is_change[idx_c]).astype(np.int16)
```

iii. `CONVERSION_NOTES.md` says the 250 ms flash was benchmarked against a 750 ms interval label and kept because it decoded slightly better and matched the image-on-screen interpretation.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary: `0` for no change and `1` for change.

ii. 
```python
OUTPUT_VALUES = [
    IMAGE_VALUES,
    ['no_change', 'change'],
    ...
]
...
change = (on_screen & is_change[idx_c]).astype(np.int16)
```

iii. The notes describe this as a binary output derived from whether the current flash is the changed flash.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is first computed on the full ophys time base and then sliced per trial using the same `[a:b]` frame boundaries used for neural data.

ii. 
```python
image_all, change_all, sp = stimulus_traces(...)
...
out[1] = change_all[a:b]
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The notes justify all output alignment through the common `ophys_timestamps` clock.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `ds.running_speed['speed']` and its corresponding timestamps.

ii. 
```python
rt = running_speed['timestamps'].values.astype(np.float64)
rv = running_speed['speed'].values.astype(np.float64)
```

iii. The notes describe this as the SDK’s processed wheel-speed stream, already low-pass filtered by Allen preprocessing.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed to the ophys timestamps, then discretizes only the kept in-trial timepoints into 5 equal-quantile bins on a per-session basis.

ii. 
```python
def resample_running(running_speed, ts):
    ...
    return np.interp(ts, rt[order], rv[order])
...
frame_idx = np.concatenate([np.arange(i0[k], i1[k]) for k in sel])
run_bin_all = np.zeros(len(ts), dtype=np.int16)
rb, run_edges = quantile_bins(run_all[frame_idx])
run_bin_all[frame_idx] = rb
```

iii. The notes explicitly say the implementation uses “5 equal-percentile bins from the session’s own within-trial values,” motivated by the task requirement for five equal percentile bins and by wanting balanced labels within each session.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into 5 quantile bins, with thresholds computed separately within each session from the frames that belong to kept trials.

ii. 
```python
def quantile_bins(values, nbins=N_QUANTILE_BINS):
    edges = np.quantile(values, np.arange(1, nbins) / nbins)
    return np.searchsorted(edges, values, side='right').astype(np.int16), edges
...
rb, run_edges = quantile_bins(run_all[frame_idx])
```

iii. The notes justify this as exactly matching the “five equal percentile bins” instruction, but at session scope rather than globally.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The code interpolates running speed to the session’s ophys timestamps first, then slices the running-bin vector and neural matrix with the same trial boundaries.

ii. 
```python
run_all = resample_running(ds.running_speed, ts)
...
out[2] = run_bin_all[a:b]
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The notes say all streams share the session clock and are resampled once per session onto `ophys_timestamps`.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI derives pupil diameter from `ds.eye_tracking['pupil_area']` plus `eye_tracking['timestamps']`, not from `pupil_width`. It treats finite positive `pupil_area` samples as valid.

ii. 
```python
t = eye_tracking['timestamps'].values.astype(np.float64)
a = eye_tracking['pupil_area'].values.astype(np.float64)
good = np.isfinite(t) & np.isfinite(a) & (a > 0)
```

iii. The notes justify this by saying blink periods already appear as `NaN` in `pupil_area`, and that effective diameter can be computed as `2*sqrt(area/pi)`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI removes non-finite / non-positive pupil-area samples, converts remaining area values to an effective diameter, linearly interpolates that diameter to the ophys time base, then bins only kept in-trial timepoints into 5 per-session quantile bins.

ii. 
```python
t, a = t[good], a[good]
order = np.argsort(t)
t, a = t[order], a[order]
diam = 2.0 * np.sqrt(a / np.pi)
return np.interp(ts, t, diam), t
...
pb, pup_edges = quantile_bins(pupil_all[frame_idx])
pup_bin_all[frame_idx] = pb
```

iii. The notes say this is “standard blink handling,” and that per-session quintiles satisfy the categorical-output requirement.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded into 5 quantile bins, computed separately per session from the pupil values on frames belonging to kept trials.

ii. 
```python
pb, pup_edges = quantile_bins(pupil_all[frame_idx])
pup_bin_all[frame_idx] = pb
...
[f'pupil_quintile_{i}' for i in range(N_QUANTILE_BINS)]
```

iii. The notes justify the five-bin discretization from the task specification and use session-level edges for equal occupancy.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the ophys timestamps before trial segmentation, and then sliced per trial with the same boundaries as neural data.

ii. 
```python
pupil_all, pupil_valid_t = resample_pupil(eye, ts)
...
out[3] = pup_bin_all[a:b]
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The notes say eye-tracking timestamps are already on the session clock and therefore only require resampling, not any additional time shift.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome comes from the four boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. 
```python
outcome_cols = np.stack([tr['hit'].values.astype(bool),
                         tr['miss'].values.astype(bool),
                         tr['false_alarm'].values.astype(bool),
                         tr['correct_reject'].values.astype(bool)], axis=1)
```

iii. The notes say these four outcomes are exactly the union of `go | catch` trials in the active task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI asserts that each kept trial has exactly one of the four outcomes, converts that one-hot boolean pattern into an integer label with `np.argmax`, and broadcasts that integer across every timepoint in the trial.

ii. 
```python
assert outcome_cols.sum(axis=1).min() == 1 and outcome_cols.sum(axis=1).max() == 1
outcome = np.argmax(outcome_cols, axis=1).astype(np.int16)
...
out[4] = outcome[k]
```

iii. The notes justify keeping trial outcome static within trial because the task says it is a static per-trial output.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable eye-tracking data causes an entire session to be skipped. Trials with no valid pupil sample are dropped. Trials that are too short are dropped. For running speed and pupil interpolation, the code first removes invalid samples and then uses `np.interp`, which fills intermediate and edge values rather than preserving NaNs.

ii. 
```python
if eye is None or len(eye) == 0 or not np.isfinite(eye['pupil_area'].values).any():
    return {'oeid': int(oeid), 'skip': 'no eye tracking data'}
...
if i1[k] - i0[k] < MIN_FRAMES_PER_TRIAL:
    n_drop_short += 1
    continue
if j1[k] - j0[k] < 1:
    n_drop_pupil += 1
    continue
...
return np.interp(ts, rt[order], rv[order])
...
return np.interp(ts, t, diam), t
```

iii. The notes justify the session and trial drops as necessary because pupil diameter is a mandatory output. They also say diagnostic and sanity-check code was added to verify that no edge extrapolation affected kept trials.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identifies opening NWB files and materializing the `(n_cells x n_frames)` neural trace matrix as the dominant runtime cost.

ii. 
```python
ds = BehaviorOphysExperiment.from_nwb_path(path)
...
traces = np.empty((len(cell_ids), len(ts)), dtype=np.float32)
for i, cid in enumerate(cell_ids):
    traces[i] = col.loc[cid]
```

iii. In Step 6 and Step 7, the notes say “Opening the NWB and materialising the trace matrix dominates runtime.”

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code already vectorizes trial-boundary `searchsorted` and resamples each full-session stream only once. The remaining obvious non-vectorized loops are the per-cell trace extraction loop and the per-trial filtering / assembly loops.

ii. 
```python
for i, cid in enumerate(cell_ids):
    traces[i] = col.loc[cid]
...
for k in range(len(tr)):
    if i1[k] - i0[k] < MIN_FRAMES_PER_TRIAL:
        ...
...
for k in sel:
    a, b = i0[k], i1[k]
    ...
    outputs.append(out)
```

iii. The notes explicitly call out that naive per-trial interpolation and per-trial `searchsorted` would have been wasteful, and that those parts were already vectorized. They do not defend the remaining loops beyond noting that file I/O dominates.

## 9-c. What processing does the code repeat multiple times?

i. Within a normal conversion run, the code avoids repeating most heavy processing by resampling once per session and reusing those arrays for every trial. The only repeated work is the per-trial slicing itself and, optionally, repeated full reruns done during benchmarking with different flags.

ii. 
```python
run_all = resample_running(ds.running_speed, ts)
pupil_all, pupil_valid_t = resample_pupil(eye, ts)
image_all, change_all, sp = stimulus_traces(...)
...
for k in sel:
    a, b = i0[k], i1[k]
    neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. Step 6 of the notes explicitly says the implementation was designed to avoid repeated per-trial interpolation and repeated `searchsorted` calls.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The final dataset keeps only per-trial arrays, but the code still computes several session-level products that are used only for diagnostics, metadata, or logging: full-session `image_all` / `change_all` / `run_all` / `pupil_all`, plotting support via `sp`, timing measurements, and an `allout` concatenation used only for the printed summary.

ii. 
```python
image_all, change_all, sp = stimulus_traces(...)
...
timing['open'] = time.time() - t0
...
if show:
    make_processing_plot(...)
...
allout = np.concatenate([o for r in kept for o in r['output']], axis=1)
for i, name in enumerate(OUTPUT_NAMES):
    cnt = np.bincount(allout[i], minlength=len(OUTPUT_VALUES[i]))
```

iii. There is no strong explicit justification for these computations beyond validation and observability. The notes present them as sanity-check and diagnostic support rather than part of the core exported representation.
