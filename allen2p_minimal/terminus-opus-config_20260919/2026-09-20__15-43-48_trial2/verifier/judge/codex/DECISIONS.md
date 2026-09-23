# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK project cache. It reads the local metadata CSV and the locally downloaded NWB files under `/app/data/visual-behavior-ophys-1.1.0`, filters them before loading to `behavior_type == 'active_behavior'`, `experience_level == 'Familiar'`, and `project_code == 'VisualBehavior'`, then loads each retained experiment directly from disk with `BehaviorOphysExperiment.from_nwb_path`. Processing is parallelized with `multiprocessing.Pool`.

ii.
```python
def select_experiments():
    meta = pd.read_csv(META)
    have = set()
    for f in os.listdir(EXP_DIR):
        if f.endswith('.nwb'):
            have.add(int(f.split('_')[-1].split('.')[0]))
    sel = meta[meta.ophys_experiment_id.isin(have)
               & (meta.behavior_type == 'active_behavior')
               & (meta.experience_level == 'Familiar')
               & (meta.project_code == 'VisualBehavior')].copy()
    sel = sel.sort_values('ophys_experiment_id').reset_index(drop=True)
    return sel
...
exp = BehaviorOphysExperiment.from_nwb_path(
    os.path.join(EXP_DIR, f'behavior_ophys_experiment_{eid}.nwb'))
...
with Pool(args.workers) as pool:
    results = pool.map(process_experiment, [(e, args.signal) for e in eids])
```

iii. The trajectory says the agent deliberately restricted the dataset to "active-behavior, familiar, single-plane experiments" after finding that the local download mixed 31 Hz single-plane sessions with 11 Hz multiscope sessions, and it preferred direct NWB loading over the cache because it had already confirmed the local files loaded cleanly.

## 1-b. How are the data split into subjects?

i. Subjects are split by `mouse_id`. After session processing, the output `subjects` list is the sorted set of retained `mouse_id` values, and each session gets a `subject_idx` pointing into that list.

ii.
```python
subjects = sorted({r['mouse_id'] for r in results})
...
data['subject_idx'].append(subjects.index(r['mouse_id']))
```

iii. The trajectory repeatedly summarizes the dataset in terms of numbers of mice and sessions, and always uses `mouse_id` as the unit of subject identity.

## 1-c. How are the data split into sessions?

i. The AI treats each retained ophys experiment NWB file as one session. It does not reconstruct sessions by grouping multiple experiments under one `ophys_session_id`; instead it prefilters to the single-plane `VisualBehavior` project where it expects one experiment per session.

ii.
```python
sel = meta[meta.ophys_experiment_id.isin(have)
           & (meta.behavior_type == 'active_behavior')
           & (meta.experience_level == 'Familiar')
           & (meta.project_code == 'VisualBehavior')].copy()
...
eids = [int(x) for x in sel.ophys_experiment_id.values]
...
exp = BehaviorOphysExperiment.from_nwb_path(
    os.path.join(EXP_DIR, f'behavior_ophys_experiment_{eid}.nwb'))
```

iii. In the trajectory, the agent states that the retained "single-plane VisualBehavior" experiments are effectively one session each, and that it excluded multiscope data because those sessions used a different frame rate.

## 1-d. How are the data split into trials?

i. Trials are taken from `exp.trials`, but only the `go` and `catch` rows are used. For each such row, the code converts `start_time` and `stop_time` to ophys-frame indices with `np.searchsorted` and slices all streams over that variable-length interval.

ii.
```python
tr = exp.trials
gc = tr[(tr.go | tr.catch)]
trials = []
for tid, row in gc.iterrows():
    a = int(np.searchsorted(ts, row.start_time, side='left'))
    b = int(np.searchsorted(ts, row.stop_time, side='left'))
    if b - a < 2:
        continue
    ...
    trials.append(dict(
        neural=neural[:, a:b].copy(),
        image_id=image_id[a:b].copy(),
        change=change[a:b].copy(),
        running=running[a:b].copy(),
        pupil=pupil[a:b].copy(),
        outcome=OUTCOMES.index(oc[0]),
        trial_id=int(tid),
        change_time=float(row.change_time),
        start_time=float(row.start_time),
        stop_time=float(row.stop_time),
    ))
```

iii. The trajectory initially considered a fixed change-centered window, but the final plan in step 34 explicitly says it would "segment go/catch trials by trials-table start/stop", and the final code follows that plan.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `go` or `catch` trials only. Additional QC excludes trials with fewer than 2 ophys frames, trials where less than 50% of frames have nearby valid pupil samples, and trials whose outcome is not exactly one of `hit`, `miss`, `false_alarm`, or `correct_reject`. Entire sessions are skipped if eye tracking is absent, if there are too few valid pupil samples, or if fewer than 2 trials survive.

ii.
```python
if len(et) == 0:
    return dict(eid=eid, skip='no eye tracking')
...
if good.sum() < 100:
    return dict(eid=eid, skip='no valid pupil samples')
...
gc = tr[(tr.go | tr.catch)]
...
if b - a < 2:
    continue
if np.mean(pupil_valid[a:b]) < 0.5:
    continue
oc = [o for o in OUTCOMES if bool(row[o])]
if len(oc) != 1:
    continue
...
results = [r for r in results if r.get('skip') is None and len(r['trials']) >= 2]
```

iii. The trajectory says the agent added the pupil-coverage rule after deciding that the pupil output was required and after finding one session with no eye tracking at all. It also notes that the dropped session was experiment `806456687`, which it confirmed had zero eye-tracking rows.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `exp.events`, using either the `events` or `filtered_events` column; the default and final choice is `filtered_events`.

ii.
```python
ap.add_argument('--signal', default='filtered_events',
                choices=['events', 'filtered_events'])
...
ev_table = exp.events
cell_ids = list(ev_table.index.values)
neural = np.stack([np.asarray(x, dtype=np.float32)
                   for x in ev_table[signal].values])
```

iii. The trajectory says the agent compared raw `events` against `filtered_events` in pilot runs and chose `filtered_events` because raw events were too sparse for per-timepoint decoding.

## 2-b. How is the `neural` data processed?

i. The code stacks all event traces into a neuron-by-time matrix, uses the `filtered_events` trace by default, and divides each neuron's full-session trace by its own standard deviation so high-variance neurons do not dominate the decoder.

ii.
```python
neural = np.stack([np.asarray(x, dtype=np.float32)
                   for x in ev_table[signal].values])
assert neural.shape[1] == nframes
sd = neural.std(axis=1, keepdims=True)
sd[sd == 0] = 1.0
neural = neural / sd
```

iii. The trajectory gives two justifications: `filtered_events` improved pilot decoder accuracy relative to raw events, and per-neuron SD normalization further improved accuracy while preventing a few high-variance cells from dominating the per-session projection.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no direct neuron-level quality filter. The code includes every row in `exp.events` for a retained session. The only relevant safeguards are session-level skips driven by missing pupil data, not neural QC.

ii.
```python
ev_table = exp.events
cell_ids = list(ev_table.index.values)
neural = np.stack([np.asarray(x, dtype=np.float32)
                   for x in ev_table[signal].values])
...
results = [r for r in results if r.get('skip') is None and len(r['trials']) >= 2]
```

iii. The trajectory never describes any explicit cell filtering step. Its QC discussion is instead about session selection, eye tracking availability, and trial validity.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the ophys timestamp clock and segmented by trial start and stop. Within each trial, the neural slice is `neural[:, a:b]`, where `a` and `b` are the ophys-frame indices corresponding to `start_time` and `stop_time`.

ii.
```python
ts = np.asarray(exp.ophys_timestamps, dtype=np.float64)
...
a = int(np.searchsorted(ts, row.start_time, side='left'))
b = int(np.searchsorted(ts, row.stop_time, side='left'))
...
neural=neural[:, a:b].copy(),
```

iii. The trajectory says the agent wanted every stream on the common ophys frame clock, and the final implementation uses trial-table `start_time` and `stop_time` as the extraction boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is one native ophys frame from the retained single-plane sessions, about 32.3 ms. No additional temporal binning is applied. The AI enforces this by excluding the 11 Hz multiscope sessions rather than resampling them.

ii.
```python
dt=float(np.median(np.diff(ts))),
...
data['metadata'] = dict(
    ...
    time_bin_size=dt * 1000.0,
    ...
    project_code='VisualBehavior (single-plane, ~31 Hz)',
)
```

iii. The trajectory explicitly says the agent excluded multiscope data because 11 Hz sessions would break its requirement that every session share the same time-bin size, and that one time bin should equal one ophys frame.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation table, specifically `exp.stimulus_presentations.image_name` for rows where the flash is active and not omitted.

ii.
```python
sp = exp.stimulus_presentations
flashes = sp[(sp.active == True) & (sp.omitted == False)]
image_names = sorted(flashes.image_name.unique())
```

iii. The trajectory says the agent sanity-checked stimulus flash timing and wanted image identity to follow the flashed stimuli on the actual ophys timebase.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code creates a full-session framewise `image_id` vector on the ophys clock. It initializes all frames to `0` for gray screen / omitted periods, then fills each active flash interval with the per-session image index `1..N`. During assembly it remaps those per-session IDs into a global image vocabulary for the final output matrix.

ii.
```python
image_id = np.zeros(nframes, dtype=np.int64)   # 0 == gray screen / omitted
starts = np.searchsorted(ts, flashes.start_time.values, side='left')
stops = np.searchsorted(ts, flashes.end_time.values, side='left')
idx = np.array([image_names.index(n) + 1 for n in flashes.image_name.values])
for a, b, i, c in zip(starts, stops, idx, is_change):
    image_id[a:b] = i
...
image_names = sorted({n for r in results for n in r['image_names']})
...
img_map = np.array([0] + [image_names.index(n) + 1
                          for n in r['image_names']])
...
out[0] = img_map[t['image_id']]
```

iii. The trajectory says the agent wanted a time-varying output tied to the real flash structure and later spot-checked that the output changed at the expected flash boundaries.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is first represented on the full-session ophys frame clock and then sliced with the same trial boundaries as neural data, so the final `out[0]` row is frame-aligned to `t['neural']`.

ii.
```python
image_id = np.zeros(nframes, dtype=np.int64)
...
trials.append(dict(
    neural=neural[:, a:b].copy(),
    image_id=image_id[a:b].copy(),
    ...
))
...
T = t['neural'].shape[1]
out = np.zeros((5, T), dtype=np.int64)
out[0] = img_map[t['image_id']]
```

iii. The trajectory says the agent aligned every stream to the same ophys timestamps and later verified on a raw trial that the image-identity output changed at the expected frame.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change` on the active, non-omitted flash table.

ii.
```python
flashes = sp[(sp.active == True) & (sp.omitted == False)]
...
is_change = flashes.is_change.values.astype(bool)
```

iii. The trajectory says the agent wanted the change signal tied to the actual flashed stimulus timing rather than only to trial metadata.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code creates a framewise binary `change` vector on the ophys clock. It sets frames to 1 only during flash intervals where `is_change` is true; all other frames stay 0.

ii.
```python
change = np.zeros(nframes, dtype=np.int64)
...
for a, b, i, c in zip(starts, stops, idx, is_change):
    image_id[a:b] = i
    if c:
        change[a:b] = 1
```

iii. The trajectory reports that the agent spot-checked a converted trial and confirmed that the change flag landed exactly on the changed flash.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already categorical: `0` means no change and `1` means change.

ii.
```python
change = np.zeros(nframes, dtype=np.int64)
...
if c:
    change[a:b] = 1
...
out[1] = t['change']
```

iii. The trajectory does not discuss any extra thresholding because the agent treated image change as an intrinsically binary variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is computed on the ophys frame clock and then sliced with the exact same trial indices as the neural matrix.

ii.
```python
trials.append(dict(
    neural=neural[:, a:b].copy(),
    ...
    change=change[a:b].copy(),
    ...
))
...
out[1] = t['change']
```

iii. The trajectory justification is the same as for image identity: all streams live on the same ophys frame timeline, and the agent later checked that the change flag matched the raw trial timing.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, using its `timestamps` and `speed` columns.

ii.
```python
rs = exp.running_speed
running = np.interp(ts, rs.timestamps.values, rs.speed.values)
```

iii. The trajectory treats running speed as one of the standard behavioral streams that should be resampled to the ophys clock.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the ophys timestamps with `np.interp`. Later, during dataset assembly, the continuous values are converted into discrete bins using session-specific or global percentile edges; the default is per-session.

ii.
```python
running = np.interp(ts, rs.timestamps.values, rs.speed.values)
...
if args.binning == 'global':
    run_all = np.concatenate([t['running'] for r in results for t in r['trials']])
    ...
else:
    ...
    run_all = np.concatenate([t['running'] for t in r['trials']])
    ...
out[2] = np.searchsorted(run_edges, t['running'], side='right')
```

iii. The trajectory says the agent originally tried global bins, then switched to per-session quintiles because absolute running levels differed across mice and global bins became nearly constant within some sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five percentile bins. By default those percentiles are computed separately within each session (`--binning session`), though the script keeps an optional `--binning global` mode.

ii.
```python
ap.add_argument('--binning', default='session', choices=['session', 'global'])
...
qs = np.linspace(0, 100, NBINS + 1)[1:-1]
if args.binning == 'global':
    ...
else:
    edges = {}
    for r in results:
        run_all = np.concatenate([t['running'] for t in r['trials']])
        ...
        edges[r['eid']] = (np.percentile(run_all, qs), np.percentile(pup_all, qs))
...
out[2] = np.searchsorted(run_edges, t['running'], side='right')
```

iii. The trajectory explicitly justifies this as a decoder-driven choice: because sessions differ strongly in absolute running levels, per-session equal-occupancy quintiles better preserve within-session variation.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first interpolated onto `ts = exp.ophys_timestamps`, then trial slices use the same `[a:b]` bounds as the neural data.

ii.
```python
ts = np.asarray(exp.ophys_timestamps, dtype=np.float64)
running = np.interp(ts, rs.timestamps.values, rs.speed.values)
...
trials.append(dict(
    neural=neural[:, a:b].copy(),
    ...
    running=running[a:b].copy(),
    ...
))
```

iii. The trajectory repeatedly states that all data streams should be put on the common ophys frame clock.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the eye-tracking table, using `pupil_area`, `timestamps`, and `likely_blink`. The code converts area to an estimated diameter.

ii.
```python
et = exp.eye_tracking
...
area = et.pupil_area.values.astype(np.float64).copy()
area[et.likely_blink.values.astype(bool)] = np.nan
diam = 2.0 * np.sqrt(area / np.pi)
et_ts = et.timestamps.values.astype(np.float64)
```

iii. The trajectory says the agent inspected eye tracking, saw blink-related NaNs, and later justified dropping the one experiment with no eye-tracking rows because the pupil output was required.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code removes blink frames by setting them to NaN, converts pupil area to diameter, interpolates valid samples to the ophys timebase, estimates how far each ophys frame is from the nearest valid pupil sample, uses that to define a validity mask, and later discretizes the resulting continuous pupil values into percentile bins.

ii.
```python
area = et.pupil_area.values.astype(np.float64).copy()
area[et.likely_blink.values.astype(bool)] = np.nan
diam = 2.0 * np.sqrt(area / np.pi)
good = np.isfinite(diam)
...
pupil = np.interp(ts, et_ts[good], diam[good])
nearest = np.abs(ts - et_ts[good][np.clip(
    np.searchsorted(et_ts[good], ts), 0, good.sum() - 1)])
nearest_prev = np.abs(ts - et_ts[good][np.clip(
    np.searchsorted(et_ts[good], ts) - 1, 0, good.sum() - 1)])
gap = np.minimum(nearest, nearest_prev)
pupil_valid = gap < 0.5
...
out[3] = np.searchsorted(pup_edges, t['pupil'], side='right')
```

iii. The trajectory says the agent added the 50%-coverage rule after checking pupil gaps and wanted to avoid treating long missing stretches as reliable pupil measurements.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five percentile bins, with the same `--binning session` default and `--binning global` option used for running speed.

ii.
```python
ap.add_argument('--binning', default='session', choices=['session', 'global'])
...
edges[r['eid']] = (np.percentile(run_all, qs), np.percentile(pup_all, qs))
...
out[3] = np.searchsorted(pup_edges, t['pupil'], side='right')
```

iii. The trajectory uses the same justification as for running speed: absolute pupil scale varied substantially across sessions, so per-session quintiles were chosen to preserve within-session variation.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the ophys timestamps before trial segmentation, and trial slices use the same `[a:b]` ophys-frame bounds as neural data.

ii.
```python
pupil = np.interp(ts, et_ts[good], diam[good])
...
trials.append(dict(
    neural=neural[:, a:b].copy(),
    ...
    pupil=pupil[a:b].copy(),
    ...
))
```

iii. The trajectory consistently states that the conversion should place every stream onto the ophys frame clock.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome comes from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
oc = [o for o in OUTCOMES if bool(row[o])]
if len(oc) != 1:
    continue
...
outcome=OUTCOMES.index(oc[0]),
```

iii. The trajectory says the agent verified that every retained go/catch trial had exactly one outcome label among those four columns.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps the one true outcome column to its index in `OUTCOMES` and stores that integer as a constant row across all time bins of the trial.

ii.
```python
oc = [o for o in OUTCOMES if bool(row[o])]
if len(oc) != 1:
    continue
trials.append(dict(
    ...
    outcome=OUTCOMES.index(oc[0]),
    ...
))
...
out[4] = t['outcome']
```

iii. The trajectory says the agent kept trial outcome as a static per-trial quantity but represented it as a constant time series so it could live in the same `(n_output, T)` array as the other outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or problematic data are mostly handled by exclusion rather than imputation. Sessions with no eye tracking or too few valid pupil samples are skipped, trials with too little pupil coverage are skipped, zero-variance neurons are left unchanged by forcing their SD divisor to 1, and sessions with fewer than 2 surviving trials are dropped. Pupil blink gaps are interpolated only when there are nearby valid measurements.

ii.
```python
if len(et) == 0:
    return dict(eid=eid, skip='no eye tracking')
...
if good.sum() < 100:
    return dict(eid=eid, skip='no valid pupil samples')
...
sd = neural.std(axis=1, keepdims=True)
sd[sd == 0] = 1.0
...
if np.mean(pupil_valid[a:b]) < 0.5:
    continue
...
results = [r for r in results if r.get('skip') is None and len(r['trials']) >= 2]
```

iii. The trajectory says the agent explicitly chose to drop the one no-eye-tracking experiment and to require at least 50% valid pupil coverage for each retained trial.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading every NWB file, extracting full-session arrays (`events`, stimulus presentations, running, eye tracking, trials), and then iterating over every trial to build copied trial slices. The code uses a process pool to reduce wall-clock time.

ii.
```python
with Pool(args.workers) as pool:
    results = pool.map(process_experiment, [(e, args.signal) for e in eids])
...
exp = BehaviorOphysExperiment.from_nwb_path(
    os.path.join(EXP_DIR, f'behavior_ophys_experiment_{eid}.nwb'))
...
for tid, row in gc.iterrows():
    ...
    trials.append(dict(
        neural=neural[:, a:b].copy(),
        ...
    ))
```

iii. The trajectory focuses on data loading, pilot runs, worker counts, and total dataset size (up to 4 GB), which indicates the AI understood I/O and full-session processing to be the main runtime costs.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the Python loop over flashes that writes `image_id` and `change`, the per-trial loop over `gc.iterrows()`, and the per-session/trial assembly loops that repeatedly call `np.searchsorted` and `.append`. The AI left these as explicit loops.

ii.
```python
for a, b, i, c in zip(starts, stops, idx, is_change):
    image_id[a:b] = i
    if c:
        change[a:b] = 1
...
for tid, row in gc.iterrows():
    ...
    trials.append(dict(...))
...
for r in results:
    ...
    for t in r['trials']:
        ...
        output_s.append(out)
```

iii. The trajectory does not contain an explicit vectorization discussion. The code suggests the AI prioritized straightforward implementation and used multiprocessing for the coarser-grained speedup instead.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats several lookup and conversion steps: it uses `image_names.index(...)` for each flash when building per-session IDs and again when building the global `img_map`; it recomputes percentile edges session by session; and it repeatedly converts Python lists of trials into concatenated arrays for bin computation and output assembly.

ii.
```python
idx = np.array([image_names.index(n) + 1 for n in flashes.image_name.values])
...
img_map = np.array([0] + [image_names.index(n) + 1
                          for n in r['image_names']])
...
for r in results:
    run_all = np.concatenate([t['running'] for t in r['trials']])
    pup_all = np.concatenate([t['pupil'] for t in r['trials']])
    edges[r['eid']] = (np.percentile(run_all, qs), np.percentile(pup_all, qs))
```

iii. The trajectory does not justify these repeated computations explicitly; they appear to be a consequence of the simple per-session assembly strategy the AI chose.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code keeps some information only transiently and discards it before the final pickle, such as `trial_id`, `change_time`, `start_time`, and `stop_time` for each segmented trial. It also computes detailed `session_info` metadata and stores `cell_ids`, `cre_line`, and imaging metadata that the downstream decoder itself does not use.

ii.
```python
trials.append(dict(
    ...
    trial_id=int(tid),
    change_time=float(row.change_time),
    start_time=float(row.start_time),
    stop_time=float(row.stop_time),
))
...
session_info.append(dict(
    ophys_experiment_id=r['eid'], ophys_session_id=r['ophys_session_id'],
    mouse_id=r['mouse_id'], cre_line=r['cre_line'],
    session_type=r['session_type'], targeted_structure=r['structure'],
    imaging_depth=r['imaging_depth'], frame_rate=r['frame_rate'],
    n_neurons=len(r['cell_ids']), n_trials=len(r['trials']),
    n_trials_go_catch=r['ntrials_total'],
    running_speed_bin_edges=[float(x) for x in run_edges],
    pupil_diameter_bin_edges=[float(x) for x in pup_edges],
    cell_specimen_ids=[int(c) for c in r['cell_ids']]))
```

iii. The trajectory does not call these out as wasteful; it presents them as sanity-check and bookkeeping information rather than decoder-essential outputs.
