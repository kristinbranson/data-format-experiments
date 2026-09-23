# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use `VisualBehaviorOphysProjectCache`. It reads the local metadata CSV (`ophys_experiment_table.csv`), keeps only locally present NWB files, filters to active non-passive `VisualBehavior` experiments, then loads each NWB directly with `BehaviorOphysExperiment.from_nwb_path`. Each selected experiment is processed independently in a worker process.

ii.
```python
et = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
local = {int(re.findall(r'(\d+)', f)[0]): os.path.join(NWB_DIR, f)
         for f in os.listdir(NWB_DIR) if f.endswith('.nwb')}
et = et[et.ophys_experiment_id.isin(local.keys())].copy()
et['path'] = et.ophys_experiment_id.map(local)
et = et[~et.session_type.str.contains('passive')]
et = et[et.project_code == 'VisualBehavior']
...
ds = BehaviorOphysExperiment.from_nwb_path(path)
```

iii. In `CONVERSION_NOTES.md`, the AI says the provided data are not in AllenSDK cache layout, so the cache API fails; therefore it used the NWB loader that returns the same `BehaviorOphysExperiment` object. It also justifies restricting to active single-plane `VisualBehavior` experiments as the subset compatible with one common ophys time-bin size.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the filtered experiment table. The final `subjects` list is the sorted set of mouse IDs from retained sessions, and `subject_idx` maps each session back to that list.

ii.
```python
et = et.sort_values(['mouse_id', 'date_of_acquisition']).reset_index(drop=True)
...
subjects = sorted(set(r['mouse'] for r in good))
subj_to_idx = {s: i for i, s in enumerate(subjects)}
...
subject_idx.append(subj_to_idx[r['mouse']])
```

iii. The justification in the notes is straightforward: `mouse_id` is the SDK subject identifier, and the single-plane active subset contains 37 mice after dropping 3 sessions without eye tracking.

## 1-c. How are the data split into sessions?

i. Each retained NWB file is treated as one session. Because the AI keeps only single-plane `VisualBehavior` data, it treats experiment and session as equivalent for conversion, while still recording `ophys_session_id` in metadata.

ii.
```python
et = et[et.project_code == 'VisualBehavior']
...
path, want_raw, signal, normalize = args
ds = BehaviorOphysExperiment.from_nwb_path(path)
md = ds.metadata
res = {'eid': eid, 'mouse': str(md['mouse_id']),
       ...
       'ophys_session_id': int(md['ophys_session_id'])}
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly argues that after excluding Multiscope data, single-plane experiments have `experiment == session`, which avoids duplicated behavioral labels across multiple planes and keeps one native frame rate.

## 1-d. How are the data split into trials?

i. Trials come from `ds.trials`. The AI keeps only `go` or `catch` rows, converts each trial’s `[start_time, stop_time)` into ophys-frame indices with `np.searchsorted`, and slices full variable-length trial windows from the session arrays.

ii.
```python
trials = ds.trials
sel = trials[(trials.go.values | trials.catch.values)].copy()
...
i0 = np.searchsorted(ots, sel.start_time.values, side='left')
i1 = np.searchsorted(ots, sel.stop_time.values, side='left')
...
for k in range(n_trials):
    a, b = i0[k], i1[k]
    neural.append(np.ascontiguousarray(neural_full[:, a:b]))
```

iii. The notes justify this as using the SDK’s canonical trial table and the experiment’s own trial definition, with trial start as the alignment event and variable trial length preserved on the native ophys grid.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by keeping only `go` and `catch` trials, dropping any with fewer than 2 ophys frames, and dropping sessions with fewer than 2 usable trials. At the session level, sessions without usable eye tracking are skipped because pupil diameter is a required output.

ii.
```python
try:
    eye = ds.eye_tracking
    if eye is None or len(eye) == 0:
        raise ValueError('empty')
except Exception as exc:
    res['skip'] = f'no eye tracking ({exc})'
    return res
...
sel = trials[(trials.go.values | trials.catch.values)].copy()
...
keep = (i1 - i0) > 1
...
if n_trials < 2:
    res['skip'] = f'only {n_trials} usable trials'
    return res
```

iii. The AI’s notes say `go | catch` is exactly the non-aborted, non-auto-rewarded subset defined by the SDK trial taxonomy. It also says whole sessions without eye tracking are removed because pupil diameter cannot be recovered for a required decoder target.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. By default, neural data come from `ds.dff_traces.dff`. The script also exposes `events` and `filtered_events` as optional alternatives, but the chosen default is dF/F.

ii.
```python
NEURAL_SIGNAL = 'dff'
...
if signal == 'dff':
    neural_full = np.vstack(ds.dff_traces.dff.values).astype(np.float32)
else:
    neural_full = np.vstack(ds.events[signal].values).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says the paper used detected events, but the AI switched the default to dF/F after benchmarking decoder performance and to avoid all-zero event trials in this per-timepoint decoder.

## 2-b. How is the `neural` data processed?

i. The AI casts the full neural matrix to `float32`, optionally applies per-neuron normalization if requested, verifies ROI/timestamp consistency, and then slices trial windows directly from the full-session matrix. No temporal smoothing or rebinning is applied in the default path.

ii.
```python
if signal == 'dff':
    neural_full = np.vstack(ds.dff_traces.dff.values).astype(np.float32)
...
if normalize == 'zscore':
    sd = neural_full.std(axis=1, keepdims=True)
    neural_full = neural_full / np.maximum(sd, 1e-6)
elif normalize == 'noise_std':
    sd = ds.events['noise_std'].values.astype(np.float32)[:, None]
    neural_full = neural_full / np.maximum(sd, 1e-6)
...
neural.append(np.ascontiguousarray(neural_full[:, a:b]))
```

iii. The notes justify keeping `--normalize none` because dF/F is already normalized by the Allen pipeline and extra scaling changed accuracy only slightly. They also describe the float32 cast as a memory optimization.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply an additional neuron-level filter. It asserts that all released ROIs are `valid_roi` and then includes every neuron in the NWB.

ii.
```python
cst = ds.cell_specimen_table
assert bool(cst.valid_roi.all()), 'invalid ROIs present in released data'
assert len(cst) == neural_full.shape[0]
res['cell_specimen_ids'] = cst.index.values.astype(np.int64)
```

iii. The notes explicitly say ROI filtering, demixing, and neuropil subtraction were already handled by the Allen pipeline, and that only valid ROIs are present in the released files.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start, with each trial spanning `[start_time, stop_time)` on the ophys timestamp grid. The alignment is implemented by converting trial boundaries to indices in `ophys_timestamps` and slicing the neural array with those indices.

ii.
```python
i0 = np.searchsorted(ots, sel.start_time.values, side='left')
i1 = np.searchsorted(ots, sel.stop_time.values, side='left')
...
for k in range(n_trials):
    a, b = i0[k], i1[k]
    neural.append(np.ascontiguousarray(neural_full[:, a:b]))
```

iii. The metadata and notes state that trial start is the alignment event and the task-required “temporal alignment based on ophys timestamp” is satisfied by sampling everything on native `ophys_timestamps`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay on the native ophys frame grid, about 32.3 ms per sample for the retained single-plane sessions. No temporal rebinning is applied.

ii.
```python
res['dt_median'] = float(np.median(np.diff(ots)))
...
dt = float(np.mean([r['dt_median'] for r in good]))
...
'time_bin_size': dt * 1000.0,
```

iii. The AI’s notes repeatedly justify excluding 11 Hz Multiscope sessions so the final dataset can keep one common native ophys bin size without resampling across rigs.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `ds.stimulus_presentations`, specifically `start_time`, `omitted`, `is_change`, and `image_name`, not from the trial table. The AI labels each ophys frame by the most recent non-omitted flash identity.

ii.
```python
stim = ds.stimulus_presentations
flashes = stim[stim.stimulus_block_name == 'change_detection_behavior'].copy()
f_start = flashes.start_time.values.astype(np.float64)
f_omitted = flashes.omitted.values.astype(bool)
f_name = flashes.image_name.values.astype(str)
...
local_idx = np.array([img_to_local.get(n, -1) for n in f_name], dtype=np.int16)
carry = local_idx.copy()
```

iii. The notes justify this as matching the paper’s 750 ms “image presentation interval” framing and handling omissions correctly, rather than treating image identity as a single pre/post-change switch from the trial table.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI builds a per-session image vocabulary from non-omitted flashes, forward-fills omitted flashes so identity carries through omissions, maps every ophys frame to a flash interval, and then remaps the per-session image labels into a global image vocabulary during final assembly.

ii.
```python
images = sorted(str(x) for x in set(f_name[~f_omitted]))
img_to_local = {n: i for i, n in enumerate(images)}
...
for i in range(1, carry.size):
    if f_omitted[i]:
        carry[i] = carry[i - 1]
...
fi = np.searchsorted(f_start, ots, side='right') - 1
image_per_frame = carry[np.clip(fi, 0, f_start.size - 1)]
...
remap = np.array([image_vocab.index(n) for n in r['images']], dtype=np.int8)
for o in r['output']:
    o[0] = remap[o[0]]
```

iii. The AI’s notes say the forward-fill keeps identity defined across omissions and grey intervals so the identity signal changes exactly when an image change occurs. The global remap ensures consistent class labels across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is first computed per ophys frame for the full session, then trial slices use the same `[a:b]` frame indices as the neural data. That makes the image identity and neural traces frame-aligned by construction.

ii.
```python
fi = np.searchsorted(f_start, ots, side='right') - 1
image_per_frame = carry[fi_clipped]
...
for k in range(n_trials):
    a, b = i0[k], i1[k]
    neural.append(np.ascontiguousarray(neural_full[:, a:b]))
    out = np.empty((5, b - a), dtype=np.int8)
    out[0] = image_per_frame[a:b]
```

iii. The notes explicitly say all streams are represented on `ophys_timestamps` before trial cutting, so trial slicing gives shared alignment across neural and output arrays.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change` and `stimulus_presentations.start_time`, mapped onto ophys frames. The AI does not derive it from `trials.change_time` and `trials.go`.

ii.
```python
f_start = flashes.start_time.values.astype(np.float64)
f_change = flashes.is_change.values.astype(bool)
...
fi = np.searchsorted(f_start, ots, side='right') - 1
change_per_frame = f_change[fi_clipped].astype(np.int8)
change_per_frame[fi < 0] = 0
```

iii. The notes justify this with the paper’s definition of the 750 ms image-presentation interval: the change label should mark the full flash interval that begins at a real change.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI converts the flash-level `is_change` flag into a per-frame binary vector by assigning every ophys frame to its flash interval. It then slices that binary session-long vector into trials.

ii.
```python
fi = np.searchsorted(f_start, ots, side='right') - 1
fi_clipped = np.clip(fi, 0, f_start.size - 1)
change_per_frame = f_change[fi_clipped].astype(np.int8)
change_per_frame[fi < 0] = 0
...
out[1] = change_per_frame[a:b]
```

iii. The notes say this is intended to label the entire 750 ms interval beginning at a real image change, and to keep `image_change` consistent with the framewise image-identity signal.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary: `0` for no change and `1` for change.

ii.
```python
change_per_frame = f_change[fi_clipped].astype(np.int8)
...
'output_values': [
    image_vocab,
    ['no_change', 'change'],
    ...
]
```

iii. The justification is just the decoder specification: image change is a binary target.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned exactly like image identity: full-session frames are labeled on `ophys_timestamps`, then the same trial frame bounds are applied to both neural and output arrays.

ii.
```python
change_per_frame = f_change[fi_clipped].astype(np.int8)
...
for k in range(n_trials):
    a, b = i0[k], i1[k]
    neural.append(np.ascontiguousarray(neural_full[:, a:b]))
    out[1] = change_per_frame[a:b]
```

iii. The notes justify alignment by putting every output stream on the ophys frame clock before cutting trials.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `ds.running_speed.timestamps` and `ds.running_speed.speed`.

ii.
```python
run = ds.running_speed
run_t = run.timestamps.values.astype(np.float64)
run_v = run.speed.values.astype(np.float64)
```

iii. The notes describe `running_speed` as the AllenSDK’s already filtered running-wheel signal and use it as the source for the running-speed output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI filters finite running-speed samples, linearly interpolates them onto ophys timestamps, computes percentile bins separately for each session using only frames retained inside kept trials, and stores the resulting bin label per frame.

ii.
```python
good = np.isfinite(run_v)
speed_per_frame = np.interp(ots, run_t[good], run_v[good])
...
def percentile_bins(x):
    edges = np.percentile(x, np.linspace(0, 100, N_BEHAVIOR_BINS + 1)[1:-1])
    return np.digitize(x, edges).astype(np.int8), edges
...
speed_bin_all, speed_edges = percentile_bins(speed_per_frame[frame_idx])
speed_bin = np.zeros(ots.size, dtype=np.int8)
speed_bin[frame_idx] = speed_bin_all
```

iii. In the notes, the AI argues for per-session quintiles because running propensity differs strongly across mice; the goal is to make the labels mean “slowest to fastest fifth within this session” rather than encode session identity.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into 5 percentile bins, computed per session, using `np.percentile` and `np.digitize`.

ii.
```python
N_BEHAVIOR_BINS = 5
...
def percentile_bins(x):
    edges = np.percentile(x, np.linspace(0, 100, N_BEHAVIOR_BINS + 1)[1:-1])
    return np.digitize(x, edges).astype(np.int8), edges
...
'output_values': [
    image_vocab,
    ['no_change', 'change'],
    [f'speed_pct_{i*20}_{(i+1)*20}' for i in range(N_BEHAVIOR_BINS)],
```

iii. The justification in the notes is that the task asks for five equal-percentile bins, and per-session edges avoid cross-session scale confounds.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the ophys frame clock before trial segmentation, then trial slices use the same frame indices as neural data.

ii.
```python
speed_per_frame = np.interp(ots, run_t[good], run_v[good])
...
for k in range(n_trials):
    a, b = i0[k], i1[k]
    neural.append(np.ascontiguousarray(neural_full[:, a:b]))
    out[2] = speed_bin[a:b]
```

iii. The AI’s notes say all behavioral streams are put on `ophys_timestamps` first, so alignment is shared automatically.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking.pupil_area` and `ds.eye_tracking.timestamps`. Frames with non-finite pupil area are treated as blink/missing frames and excluded before interpolation.

ii.
```python
eye = ds.eye_tracking
eye_t = eye.timestamps.values.astype(np.float64)
pupil_area = eye.pupil_area.values.astype(np.float64)
good_eye = np.isfinite(pupil_area)
```

iii. The notes explain that the SDK has already marked blink-corrupted eye samples as NaN, so filtering finite `pupil_area` values effectively drops blinks without directly using the `likely_blink` column.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI converts pupil area to an equivalent diameter `2*sqrt(area/pi)`, linearly interpolates that diameter across non-blink samples onto the ophys frame clock, and then bins it into 5 per-session percentile bins over retained trial frames.

ii.
```python
good_eye = np.isfinite(pupil_area)
...
pupil_diam = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_per_frame = np.interp(ots, eye_t[good_eye], pupil_diam[good_eye])
...
pupil_bin_all, pupil_edges = percentile_bins(pupil_per_frame[frame_idx])
pupil_bin = np.zeros(ots.size, dtype=np.int8)
pupil_bin[frame_idx] = pupil_bin_all
```

iii. The notes justify the area-to-diameter transform as a monotone conversion that does not change percentile labels, and per-session bins as protection against session/rig scale confounds in eye-tracking pixels.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into 5 percentile bins, computed separately for each session from retained trial frames.

ii.
```python
N_BEHAVIOR_BINS = 5
...
pupil_bin_all, pupil_edges = percentile_bins(pupil_per_frame[frame_idx])
...
'output_values': [
    image_vocab,
    ['no_change', 'change'],
    [f'speed_pct_{i*20}_{(i+1)*20}' for i in range(N_BEHAVIOR_BINS)],
    [f'pupil_pct_{i*20}_{(i+1)*20}' for i in range(N_BEHAVIOR_BINS)],
```

iii. The notes justify the same per-session percentile strategy as for running speed, especially because pupil measurements are in session-specific camera pixels.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is converted to a full-session per-frame signal on `ophys_timestamps`, then the same trial frame boundaries are used for both pupil and neural arrays.

ii.
```python
pupil_per_frame = np.interp(ots, eye_t[good_eye], pupil_diam[good_eye])
...
for k in range(n_trials):
    a, b = i0[k], i1[k]
    neural.append(np.ascontiguousarray(neural_full[:, a:b]))
    out[3] = pupil_bin[a:b]
```

iii. The notes again justify this as common-clock alignment via `ophys_timestamps`.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject` in `ds.trials`.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
oc = np.full(len(sel), -1, dtype=np.int8)
for k, name in enumerate(OUTCOME_NAMES):
    oc[sel[name].values.astype(bool)] = k
```

iii. The notes cite the SDK trial taxonomy and say these four labels are mutually exclusive and exhaustive on the retained `go|catch` trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI converts the four mutually exclusive boolean columns into an integer code `0..3` once per trial, then broadcasts that single code across every timepoint in the trial output array.

ii.
```python
oc = np.full(len(sel), -1, dtype=np.int8)
for k, name in enumerate(OUTCOME_NAMES):
    oc[sel[name].values.astype(bool)] = k
...
for k in range(n_trials):
    ...
    out[4] = oc[k]
```

iii. The notes justify this as the static per-trial decoder target required by the task.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles problematic sessions by skipping them and handles problematic trials by dropping them. Missing eye tracking or effectively all-NaN pupil traces cause a session skip; trials with fewer than 2 frames are removed; sessions with fewer than 2 usable trials are removed. Within kept sessions, blink/missing pupil samples are bridged by interpolation after removing non-finite points, and numerous assertions are used as fail-fast sanity checks.

ii.
```python
try:
    eye = ds.eye_tracking
    if eye is None or len(eye) == 0:
        raise ValueError('empty')
except Exception as exc:
    res['skip'] = f'no eye tracking ({exc})'
    return res
...
if good_eye.sum() < 100:
    res['skip'] = 'pupil data all NaN'
    return res
...
keep = (i1 - i0) > 1
...
assert np.all(np.diff(frame_idx) > 0), 'trials overlap in time'
```

iii. The notes frame this as quality control rather than imputation: required outputs must exist, and invariants like non-overlapping trials and consistent image labels are asserted so format or alignment bugs fail immediately.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading each NWB session and performing per-session preprocessing/trial cutting. The AI parallelizes this work across worker processes and reports that session processing dominates wall time.

ii.
```python
ds = BehaviorOphysExperiment.from_nwb_path(path)
...
if args.workers > 1 and len(jobs) > 1:
    with Pool(min(args.workers, len(jobs))) as pool:
        for i, r in enumerate(pool.imap(process_session, jobs)):
            _report(i, len(jobs), r, t_proc)
            results.append(r)
```

iii. In Step 7 of the notes, the AI reports about 2.4 s/session for NWB loading and 0.8 s/session for trace/behavior processing, with the parallel worker pool reducing full-run wall time to under a minute.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorizes most session-level work. The remaining obvious Python loops are the forward-fill over omitted flashes, the per-trial slicing loop, the per-trial `_check_session` loop, and minor list-comprehension/counting loops in plotting and assembly.

ii.
```python
for i in range(1, carry.size):
    if f_omitted[i]:
        carry[i] = carry[i - 1]
...
for k in range(n_trials):
    a, b = i0[k], i1[k]
    neural.append(np.ascontiguousarray(neural_full[:, a:b]))
    ...
for k in range(len(sel)):
    t = ots[i0[k]:i1[k]]
    ident = output[k][0]
```

iii. The AI’s notes explicitly say it pushed all per-frame computations to vectorized whole-session arrays and left only trial slicing in Python because that was not the bottleneck after parallelization.

## 9-c. What processing does the code repeat multiple times?

i. In the main conversion path, little heavy processing is repeated: full-session framewise outputs are computed once, then reused by slicing. Repeated work is mostly minor bookkeeping, such as per-session remapping from local image codes to global image codes and the optional sanity-check/plotting passes.

ii.
```python
image_per_frame = carry[fi_clipped]
change_per_frame = f_change[fi_clipped].astype(np.int8)
speed_bin_all, speed_edges = percentile_bins(speed_per_frame[frame_idx])
pupil_bin_all, pupil_edges = percentile_bins(pupil_per_frame[frame_idx])
...
remap = np.array([image_vocab.index(n) for n in r['images']], dtype=np.int8)
for o in r['output']:
    o[0] = remap[o[0]]
```

iii. The notes explicitly emphasize that per-frame quantities are computed once per session and then only sliced, which is the AI’s main anti-repetition design choice.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does optional extra processing for diagnostics and validation that is not needed by downstream decoder training: it can save a large `raw` dictionary for plotting, runs extensive `_check_session` assertions, computes `speed_range`/`pupil_range` only for notes, and stores rich session metadata beyond what the decoder itself consumes.

ii.
```python
res['speed_range'] = (float(speed_per_frame[frame_idx].min()),
                      float(speed_per_frame[frame_idx].max()))
res['pupil_range'] = (float(pupil_per_frame[frame_idx].min()),
                      float(pupil_per_frame[frame_idx].max()))
...
_check_session(res, sel, oc, i0, i1, ots, output, images)
...
if want_raw:
    res['raw'] = dict(
        ots=ots, neural_full=neural_full, ...
        speed_bin_all=speed_bin_all, pupil_bin_all=pupil_bin_all)
```

iii. The notes justify these extras as sanity checks, validation plots, and traceability rather than decoder inputs. They are intentionally outside the minimal downstream data path.
