# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment metadata CSV, maps locally present NWB filenames to experiment IDs, retains active `VisualBehavior` rows, and loads each NWB directly with `BehaviorOphysExperiment.from_nwb_path`. It processes the selected files in a multiprocessing pool. Thus “all” means all locally available active single-plane experiments with usable eye tracking, not all experiments exposed by the SDK cache.

ii.
```python
et = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
local = {int(re.findall(r'(\d+)', f)[0]): os.path.join(NWB_DIR, f)
         for f in os.listdir(NWB_DIR) if f.endswith('.nwb')}
et = et[et.ophys_experiment_id.isin(local.keys())].copy()
et = et[~et.session_type.str.contains('passive')]
et = et[et.project_code == 'VisualBehavior']
ds = BehaviorOphysExperiment.from_nwb_path(path)
```

iii. The notes justify limiting to local files because the local release is a subset, excluding passive recordings because they have no trials, and excluding multiscope data because its 11 Hz sampling conflicts with a common native bin size and its planes duplicate behavior from only six sessions.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique string-valued `mouse_id` values among sessions that survive processing; each session receives the corresponding `subject_idx`.

ii.
```python
subjects = sorted(set(r['mouse'] for r in good))
subj_to_idx = {s: i for i, s in enumerate(subjects)}
subject_idx.append(subj_to_idx[r['mouse']])
```

iii. The agent relies on the SDK/NWB `mouse_id` as the canonical animal identifier and reports that the retained local subset contains 37 mice.

## 1-c. How are the data split into sessions?

i. Each retained single-plane NWB/`ophys_experiment_id` becomes one output session. The code does not group rows by `ophys_session_id`; it argues that for the retained single-plane rig, one experiment equals one recording session.

ii.
```python
jobs = [(p, args.show_processing and (args.sample or i < n_plot),
         args.neural, args.normalize) for i, p in enumerate(paths)]
res = {'eid': eid, ...,
       'ophys_session_id': int(md['ophys_session_id'])}
neural.append(r['neural'])
```

iii. The notes explicitly distinguish experiment, session, and container, then justify the equivalence only after excluding multiscope experiments, where several planes would otherwise share one session.

## 1-d. How are the data split into trials?

i. Trials come from `ds.trials`; retained trial windows are `[start_time, stop_time)` converted to native ophys indices with `searchsorted`. Each resulting trial is variable length.

ii.
```python
sel = trials[(trials.go.values | trials.catch.values)].copy()
i0 = np.searchsorted(ots, sel.start_time.values, side='left')
i1 = np.searchsorted(ots, sel.stop_time.values, side='left')
for k in range(n_trials):
    a, b = i0[k], i1[k]
    neural.append(np.ascontiguousarray(neural_full[:, a:b]))
```

iii. The agent says the SDK trials table is authoritative and that full trial windows retain both pre-change and post-change activity and permit time-varying stimulus outputs.

## 1-e. How are trials filtered based on quality controls?

i. Only `go | catch` trials are kept, which excludes aborted and auto-rewarded trials. Trials with at most one ophys frame are removed; sessions with fewer than two usable trials are skipped. The code asserts outcome exhaustiveness and nonoverlap.

ii.
```python
sel = trials[(trials.go.values | trials.catch.values)].copy()
keep = (i1 - i0) > 1
...
if n_trials < 2:
    res['skip'] = f'only {n_trials} usable trials'
```

iii. The notes verify that trial types are mutually exclusive and that `go|catch` exactly implements the requested exclusion. Minimum lengths protect decoder validation and percentile binning.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The default neural signal is each NWB's `dff_traces.dff`; optional CLI choices permit `events` or `filtered_events`.

ii.
```python
if signal == 'dff':
    neural_full = np.vstack(ds.dff_traces.dff.values).astype(np.float32)
else:
    neural_full = np.vstack(ds.events[signal].values).astype(np.float32)
```

iii. The agent chose released dF/F because it is the Allen pipeline's normalized/detrended primary trace and empirically decoded better, while documenting that the paper used detected events.

## 2-b. How is the `neural` data processed?

i. Df/F rows are stacked into a neuron-by-frame float32 array and sliced into contiguous trial arrays. The default applies no additional normalization, although optional z-score and noise-standard-deviation scaling exist.

ii.
```python
NEURAL_NORMALIZE = 'none'
neural_full = np.vstack(ds.dff_traces.dff.values).astype(np.float32)
...
neural.append(np.ascontiguousarray(neural_full[:, a:b]))
```

iii. The notes argue that dF/F already has baseline normalization, detrending, demixing, and neuropil correction, and that extra scaling is unnecessary by default.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional cells are removed. The code asserts that all released cells have `valid_roi=True`, that the cell count matches the trace count, and that timestamps are increasing.

ii.
```python
cst = ds.cell_specimen_table
assert bool(cst.valid_roi.all()), 'invalid ROIs present in released data'
assert len(cst) == neural_full.shape[0]
```

iii. The agent states that Allen's upstream pipeline already removed invalid, duplicate, border, and nonsomatic ROIs and found 100% of cells in retained files valid.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start and extends until trial stop. Both boundaries are mapped to the first ophys frame at or after the timestamp.

ii.
```python
i0 = np.searchsorted(ots, sel.start_time.values, side='left')
i1 = np.searchsorted(ots, sel.stop_time.values, side='left')
neural.append(np.ascontiguousarray(neural_full[:, i0[k]:i1[k]]))
```

iii. The metadata calls `trials.start_time` the temporal alignment event and explains that all streams are sampled on the native ophys clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is performed. Data remain at the retained rigs' native ~31 Hz ophys sampling, with the stored bin size equal to the mean of session median intervals (32.3193 ms in the full run).

ii.
```python
res['dt_median'] = float(np.median(np.diff(ots)))
dt = float(np.mean([r['dt_median'] for r in good]))
'time_bin_size': dt * 1000.0,
```

iii. The agent excluded 11 Hz multiscope recordings specifically to preserve one effectively common native bin size without resampling.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `stimulus_presentations` fields `stimulus_block_name`, `start_time`, `image_name`, and `omitted`.

ii.
```python
stim = ds.stimulus_presentations
flashes = stim[stim.stimulus_block_name == 'change_detection_behavior'].copy()
f_start = flashes.start_time.values.astype(np.float64)
f_omitted = flashes.omitted.values.astype(bool)
f_name = flashes.image_name.values.astype(str)
```

iii. The agent chose the actual stimulus timeline rather than reconstructing identity only from trial columns, so omissions and flash intervals are represented explicitly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Non-omitted image names are assigned per-session integer codes. Omitted flashes forward-fill the preceding identity. Every ophys frame is assigned the most recent flash interval, then local codes are remapped to a sorted global 16-image vocabulary.

ii.
```python
carry = local_idx.copy()
for i in range(1, carry.size):
    if f_omitted[i]:
        carry[i] = carry[i - 1]
fi = np.searchsorted(f_start, ots, side='right') - 1
image_per_frame = carry[np.clip(fi, 0, f_start.size - 1)]
```

iii. The notes justify carrying identity through omissions because no new image appears and this makes identity change exactly when a real image change occurs.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Each ophys timestamp is mapped to its containing flash interval; the resulting whole-session array is sliced with the exact same `[a:b]` indices as neural data.

ii.
```python
fi = np.searchsorted(f_start, ots, side='right') - 1
image_per_frame = carry[fi_clipped]
...
out[0] = image_per_frame[a:b]
```

iii. The shared ophys grid guarantees frame-for-frame alignment; per-session checks compare pre/post-change identities with trial metadata.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from `stimulus_presentations.is_change` and `start_time` within the change-detection block.

ii.
```python
f_change = flashes.is_change.values.astype(bool)
fi = np.searchsorted(f_start, ots, side='right') - 1
change_per_frame = f_change[fi_clipped].astype(np.int8)
```

iii. The stimulus table directly identifies real change flashes, avoiding falsely marking catch/sham changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The change flag of the current flash interval is copied to every ophys frame in that 750 ms interval; frames before the first behavior flash are forced to zero.

ii.
```python
change_per_frame = f_change[fi_clipped].astype(np.int8)
change_per_frame[fi < 0] = 0
```

iii. The agent interprets “right after” as the full 250 ms flash plus 500 ms gray interval beginning at a real change.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is estimated: the SDK boolean is cast directly to categorical `int8` values 0 and 1.

ii.
```python
change_per_frame = f_change[fi_clipped].astype(np.int8)
```

iii. The source is already binary, and output labels are `['no_change', 'change']`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is first evaluated at every ophys frame and then sliced with the same trial indices used for neural activity.

ii.
```python
out[1] = change_per_frame[a:b]
```

iii. Assertions check that changes occur only after `change_time`, last no more than about 0.8 s, and are absent on catch trials.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `ds.running_speed.timestamps` and `ds.running_speed.speed`.

ii.
```python
run = ds.running_speed
run_t = run.timestamps.values.astype(np.float64)
run_v = run.speed.values.astype(np.float64)
```

iii. The agent describes this as the SDK's already-filtered wheel-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite speed samples are linearly interpolated onto ophys timestamps, then discretized using retained trial frames. The resulting session-length bin array is sliced per trial.

ii.
```python
good = np.isfinite(run_v)
speed_per_frame = np.interp(ots, run_t[good], run_v[good])
speed_bin_all, speed_edges = percentile_bins(speed_per_frame[frame_idx])
```

iii. Interpolation places behavior on the neural clock; the notes say upstream SDK filtering means no extra smoothing is needed.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four 20th/40th/60th/80th percentile edges are computed separately in each session over retained trial frames, producing five approximately equally occupied integer bins.

ii.
```python
edges = np.percentile(x, np.linspace(0, 100, N_BEHAVIOR_BINS + 1)[1:-1])
return np.digitize(x, edges).astype(np.int8), edges
```

iii. The agent argues per-session bins handle large differences in locomotion propensity and ensure balanced classes, though this differs from the reference's global edges.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated onto every ophys timestamp before both signals are sliced with identical trial bounds.

ii.
```python
speed_per_frame = np.interp(ots, run_t[good], run_v[good])
out[2] = speed_bin[a:b]
```

iii. The common ophys grid provides direct frame-level alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses `ds.eye_tracking.timestamps` and `pupil_area`; nonfinite area values, including SDK-masked blinks, are excluded.

ii.
```python
eye_t = eye.timestamps.values.astype(np.float64)
pupil_area = eye.pupil_area.values.astype(np.float64)
good_eye = np.isfinite(pupil_area)
```

iii. The agent chose area-derived equivalent-circle diameter and notes that this monotonic transform leaves percentile ranks unchanged.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area is converted to `2*sqrt(area/pi)`, finite samples are linearly interpolated across blinks and onto ophys timestamps, and the result is percentile-discretized per session.

ii.
```python
pupil_diam = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_per_frame = np.interp(ots, eye_t[good_eye], pupil_diam[good_eye])
pupil_bin_all, pupil_edges = percentile_bins(pupil_per_frame[frame_idx])
```

iii. The notes justify interpolating blink gaps and using a diameter-like measure; sessions lacking enough eye data are discarded because pupil is required.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four within-session percentile edges over retained frames define five approximately equal-occupancy categories.

ii.
```python
pupil_bin_all, pupil_edges = percentile_bins(pupil_per_frame[frame_idx])
```

iii. The agent argues camera-pixel scale is not comparable across sessions and reports essentially 20% occupancy per class, but this differs from the reference's global thresholds.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to ophys timestamps, placed into a session-length bin array, and sliced with the same trial indices as neural data.

ii.
```python
pupil_per_frame = np.interp(ots, eye_t[good_eye], pupil_diam[good_eye])
out[3] = pupil_bin[a:b]
```

iii. This makes each pupil category correspond to the same ophys frame as each neural column.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`, after selecting go/catch trials.

ii.
```python
for k, name in enumerate(OUTCOME_NAMES):
    oc[sel[name].values.astype(bool)] = k
```

iii. The agent treats these as the canonical mutually exclusive outcomes and asserts exactly one is true for every retained trial.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes map in fixed order to codes 0–3 and each code is broadcast across every frame of its trial.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
out[4] = oc[k]
```

iii. A time-constant row preserves the static trial label while satisfying the common output matrix shape.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/empty eye tracking or fewer than 100 valid pupil samples skips a session; nonfinite running and pupil samples are omitted before interpolation; too-short trials and sessions are dropped. Extensive assertions fail on malformed timestamps, ROIs, outcomes, overlaps, or stimulus inconsistencies. Unlike the reference, worker exceptions are not caught and would abort the run.

ii.
```python
if eye is None or len(eye) == 0:
    raise ValueError('empty')
...
if good_eye.sum() < 100:
    res['skip'] = 'pupil data all NaN'
    return res
keep = (i1 - i0) > 1
```

iii. The notes say required pupil output cannot be responsibly imputed for an entire session and describe interpolation as appropriate for blink gaps. Assertions are used to avoid silent corruption.

## 9-a. What are the most time-consuming steps of the code?

i. Loading/parsing large NWBs and materializing full dF/F arrays dominate per-session work; writing the 8.34 GB pickle is also substantial. The code records load, processing, assembly, and write times.

ii.
```python
ds = BehaviorOphysExperiment.from_nwb_path(path)
neural_full = np.vstack(ds.dff_traces.dff.values).astype(np.float32)
...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes report about 3.2 CPU-seconds per session, 49 seconds wall time with 24 workers, and roughly 13 seconds to write the pickle.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The omission forward-fill loop, per-trial slicing/output assembly loop, outcome-label loop, validation loop, and global image remapping loop could be further vectorized. Most expensive timeline operations already use `searchsorted`, `interp`, and array indexing.

ii.
```python
for i in range(1, carry.size):
    if f_omitted[i]: carry[i] = carry[i - 1]
for k in range(n_trials):
    ...
for o in r['output']:
    o[0] = remap[o[0]]
```

iii. The agent emphasizes vectorized whole-session timelines and parallel sessions; it retains trial loops because trials are variable length and must ultimately be stored as separate arrays.

## 9-c. What processing does the code repeat multiple times?

i. Each session independently repeats NWB loading, interpolation, flash-to-frame mapping, percentile computation, trial slicing, and checks. Trial indices are concatenated more than once (processing and optional plotting), and image arrays are remapped in a later global assembly pass.

ii.
```python
frame_idx = np.concatenate([np.arange(a, b) for a, b in zip(i0, i1)])
...
for r in good:
    remap = np.array([image_vocab.index(n) for n in r['images']], dtype=np.int8)
```

iii. The notes characterize per-frame quantities as computed once per session and reused, so repetition is mainly the necessary application of the same pipeline to independent files plus a later local-to-global mapping.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes and retains extensive diagnostics and metadata not consumed by decoder training (cell IDs, raw ranges, blink fractions, trial times, counts, timing, and optional raw plotting arrays). `--show-processing` additionally computes figures and triggered averages solely for validation; raw plotting payloads are then removed.

ii.
```python
if want_raw:
    res['raw'] = dict(ots=ots, neural_full=neural_full, ...)
...
for r in results:
    r.pop('raw', None)
```

iii. The agent intentionally performs these checks to document correctness and catch alignment errors; they are diagnostic rather than required by downstream decoder inputs.
