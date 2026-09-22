# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment metadata CSV, intersects it with locally present NWB files, and keeps active, familiar, single-plane `VisualBehavior` experiments. Each selected NWB is loaded directly with `BehaviorOphysExperiment.from_nwb_path`, in parallel.

ii.
```python
sel = meta[meta.ophys_experiment_id.isin(have)
           & (meta.behavior_type == 'active_behavior')
           & (meta.experience_level == 'Familiar')
           & (meta.project_code == 'VisualBehavior')].copy()
...
exp = BehaviorOphysExperiment.from_nwb_path(
    os.path.join(EXP_DIR, f'behavior_ophys_experiment_{eid}.nwb'))
...
with Pool(args.workers) as pool:
    results = pool.map(process_experiment, [(e, args.signal) for e in eids])
```

iii. The trajectory says the agent restricted to active sessions because passive sessions lack behavioral reports, to familiar images because the paper restricted analyses to familiar stimuli, and to single-plane data because those files share a ~31 Hz frame rate while local multiscope data are ~11 Hz.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique string-valued `mouse_id` metadata among retained experiments; each session stores the corresponding index.

ii.
```python
subjects = sorted({r['mouse_id'] for r in results})
...
data['subject_idx'].append(subjects.index(r['mouse_id']))
```

iii. The agent treated the SDK `mouse_id` as the animal identifier and reported 37 retained mice after filtering.

## 1-c. How are the data split into sessions?

i. Every selected `ophys_experiment_id` becomes one output session. The code records `ophys_session_id` but does not group multiple experiments/planes sharing it.

ii.
```python
eids = [int(x) for x in sel.ophys_experiment_id.values]
...
for r in results:
    ...
    data['neural'].append(neural_s)
```

iii. The trajectory justifies this by selecting the single-plane `VisualBehavior` project, where the examined experiments effectively correspond one-to-one with sessions; multiscope experiments were excluded due to their different sampling rate.

## 1-d. How are the data split into trials?

i. Trials come from the Allen trials table. Rows marked `go` or `catch` are sliced from `start_time` (inclusive at the first ophys frame at/after it) to `stop_time` (exclusive).

ii.
```python
tr = exp.trials
gc = tr[(tr.go | tr.catch)]
for tid, row in gc.iterrows():
    a = int(np.searchsorted(ts, row.start_time, side='left'))
    b = int(np.searchsorted(ts, row.stop_time, side='left'))
```

iii. The agent found that go/catch rows have valid timing and exactly one canonical outcome, and chose the full variable-length trial interval so the outputs remain time-varying across the trial.

## 1-e. How are trials filtered based on quality controls?

i. Selecting `go | catch` implicitly excludes aborted and auto-rewarded trials. Trials shorter than two frames, trials with under 50% pupil-valid frames, and trials without exactly one outcome label are dropped. Experiments with missing/insufficient pupil data and sessions with fewer than two retained trials are also dropped.

ii.
```python
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

iii. The instruction explicitly required go and catch while excluding aborted and auto-rewarded trials. The agent added pupil coverage checks to avoid assigning pupil categories where tracking was mostly absent; it reports retaining 99.4% of go/catch trials and dropping one experiment with no eye tracking.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `exp.events`, using `filtered_events` by default (or raw `events` through a command-line option).

ii.
```python
ev_table = exp.events
neural = np.stack([np.asarray(x, dtype=np.float32)
                   for x in ev_table[signal].values])
...
ap.add_argument('--signal', default='filtered_events',
                choices=['events', 'filtered_events'])
```

iii. The paper stated that analyses used detected calcium events. Pilot decoder tests showed filtered events outperformed very sparse raw events, so the agent selected the SDK's half-normal-filtered event traces.

## 2-b. How is the `neural` data processed?

i. Each cell's full-session event trace is divided by its own standard deviation; zero standard deviations are replaced with one. Trial slices are later cast to `float32`.

ii.
```python
sd = neural.std(axis=1, keepdims=True)
sd[sd == 0] = 1.0
neural = neural / sd
...
neural_s.append(t['neural'].astype(np.float32))
```

iii. The agent reasoned that normalization prevents a few high-variance cells from dominating PCA/projection and reported improved pilot decoder accuracy after adding it.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code takes all cells present in `exp.events`; it applies no additional ROI/cell filtering. It asserts trace length equals the ophys timestamp count. Sessions are indirectly excluded if pupil/trial requirements fail.

ii.
```python
cell_ids = list(ev_table.index.values)
...
assert neural.shape[1] == nframes
```

iii. The trajectory notes that all-zero trial slices in low-cell-count sessions were considered genuine sparse-event data rather than errors. No further neural QC was added.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples already lie on `ophys_timestamps`. Each trial is aligned to trial `start_time` and sliced through `stop_time` using the corresponding ophys-frame indices.

ii.
```python
a = int(np.searchsorted(ts, row.start_time, side='left'))
b = int(np.searchsorted(ts, row.stop_time, side='left'))
...
neural=neural[:, a:b].copy()
```

iii. The agent interpreted “temporally align based on ophys timestamp” as putting all streams on the ophys clock, with trial start as the per-trial alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One time bin is one native single-plane ophys frame, about 32.32 ms (~31 Hz). Neural data are not temporally rebinned; metadata uses the mean of per-session median frame intervals.

ii.
```python
dt=float(np.median(np.diff(ts)))
...
dt = float(np.mean([r['dt'] for r in results]))
...
time_bin_size=dt * 1000.0
```

iii. The agent excluded 11 Hz multiscope data to preserve a common native bin size and reported a final 32.32 ms resolution.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from active, non-omitted rows of `exp.stimulus_presentations`, specifically `image_name`, `start_time`, and `end_time`.

ii.
```python
sp = exp.stimulus_presentations
flashes = sp[(sp.active == True) & (sp.omitted == False)]
starts = np.searchsorted(ts, flashes.start_time.values, side='left')
stops = np.searchsorted(ts, flashes.end_time.values, side='left')
```

iii. The agent chose the stimulus table to represent the actual 250 ms flashes and omitted/gray periods on the ophys clock, rather than treating the trial's initial/change names as continuously displayed.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A session-local sorted image list is coded from 1 upward; zero represents gray or omitted frames. At assembly, local codes are remapped into a globally sorted image vocabulary, also with zero for gray.

ii.
```python
image_id = np.zeros(nframes, dtype=np.int64)
idx = np.array([image_names.index(n) + 1 for n in flashes.image_name.values])
for a, b, i, c in zip(starts, stops, idx, is_change):
    image_id[a:b] = i
...
img_map = np.array([0] + [image_names.index(n) + 1
                          for n in r['image_names']])
out[0] = img_map[t['image_id']]
```

iii. The justification was to encode the truly displayed image only during flashes and reserve an explicit gray-screen class between flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Flash start/end times are mapped to ophys indices, a full-session image array is created on that clock, and the same `[a:b]` trial slice used for neural data is applied.

ii.
```python
image_id[a:b] = i
...
image_id=image_id[a:b].copy(),
neural=neural[:, a:b].copy(),
```

iii. The trajectory reports a raw-versus-converted spot check confirming the image code and frame timing.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It comes from `stimulus_presentations.is_change` together with each presentation's start and end times, after restricting to active, non-omitted flashes.

ii.
```python
is_change = flashes.is_change.values.astype(bool)
for a, b, i, c in zip(starts, stops, idx, is_change):
    ...
    if c:
        change[a:b] = 1
```

iii. The agent used the SDK's explicit stimulus-level change annotation so catch/sham trials remain zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero-valued full-session integer array is filled with one over every presentation interval whose `is_change` flag is true, then sliced into trials.

ii.
```python
change = np.zeros(nframes, dtype=np.int64)
...
if c:
    change[a:b] = 1
```

iii. The agent intended “right after a change” to mean the duration of the actual change flash (~250 ms), not the entire post-change trial.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is estimated. The boolean `is_change` field directly produces category 1 during a change flash and 0 otherwise; labels are `no_change` and `change`.

ii.
```python
is_change = flashes.is_change.values.astype(bool)
...
['no_change', 'change']
```

iii. The source field is already binary, so the agent considered further thresholding unnecessary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Presentation boundaries are converted to ophys-frame indices, and the full-session change vector is cut with the same trial boundaries as neural data.

ii.
```python
starts = np.searchsorted(ts, flashes.start_time.values, side='left')
stops = np.searchsorted(ts, flashes.end_time.values, side='left')
...
change=change[a:b].copy(),
```

iii. A trajectory spot check found the change flag beginning exactly at the `searchsorted(change_time)` frame and lasting about eight 31 Hz frames.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `exp.running_speed.speed` and `exp.running_speed.timestamps`.

ii.
```python
rs = exp.running_speed
running = np.interp(ts, rs.timestamps.values, rs.speed.values)
```

iii. The agent used the SDK's standard wheel-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated to the ophys clock. By default, four 20/40/60/80 percentile edges are calculated separately within each retained session's trial samples, then `searchsorted` assigns five bins. Global binning is optional.

ii.
```python
qs = np.linspace(0, 100, NBINS + 1)[1:-1]
run_all = np.concatenate([t['running'] for t in r['trials']])
edges[r['eid']] = (np.percentile(run_all, qs), ...)
...
out[2] = np.searchsorted(run_edges, t['running'], side='right')
```

iii. Pilot global bins were highly imbalanced within sessions. Because the decoder has a session-specific projection, the agent chose per-session equal-occupancy quintiles and reported better interpretability/decodability.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Values are assigned categories 0–4 using the session's four percentile boundaries with `side='right'`; equality moves into the higher bin.

ii.
```python
out[2] = np.searchsorted(run_edges, t['running'], side='right')
```

iii. Five percentile bins were explicitly requested; per-session percentiles were selected to make occupancy approximately equal within each recording.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The full running stream is interpolated at every ophys timestamp before the same trial slice `[a:b]` is applied.

ii.
```python
running = np.interp(ts, rs.timestamps.values, rs.speed.values)
...
running=running[a:b].copy(),
```

iii. The agent used the ophys timestamps as the common clock required by the instructions.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It derives diameter from `exp.eye_tracking.pupil_area`, rejects `likely_blink` samples, and uses eye-tracking timestamps.

ii.
```python
area = et.pupil_area.values.astype(np.float64).copy()
area[et.likely_blink.values.astype(bool)] = np.nan
diam = 2.0 * np.sqrt(area / np.pi)
et_ts = et.timestamps.values.astype(np.float64)
```

iii. The agent interpreted pupil diameter geometrically as the diameter of a circle with the measured pupil area and removed SDK-flagged blinks.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink/nonfinite samples are removed; sessions with under 100 good samples are rejected. Remaining samples are linearly interpolated to ophys timestamps. A nearest-valid-sample distance marks coverage, and default per-session percentile edges discretize retained trial values into quintiles.

ii.
```python
good = np.isfinite(diam)
if good.sum() < 100:
    return dict(eid=eid, skip='no valid pupil samples')
pupil = np.interp(ts, et_ts[good], diam[good])
...
pupil_valid = gap < 0.5
...
out[3] = np.searchsorted(pup_edges, t['pupil'], side='right')
```

iii. The agent wanted to interpolate ordinary blink gaps while rejecting trials where the nearest real measurement was too distant; per-session bins address camera-position and apparent-size differences across recordings.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four session-specific 20/40/60/80 percentile edges yield integer categories 0–4 via right-sided `searchsorted`.

ii.
```python
pup_all = np.concatenate([t['pupil'] for t in r['trials']])
np.percentile(pup_all, qs)
...
out[3] = np.searchsorted(pup_edges, t['pupil'], side='right')
```

iii. The five requested categories are made approximately equal-occupancy within each session.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Valid eye samples are interpolated directly at ophys timestamps, after which pupil and neural arrays use the same `[a:b]` trial slice.

ii.
```python
pupil = np.interp(ts, et_ts[good], diam[good])
...
pupil=pupil[a:b].copy(),
neural=neural[:, a:b].copy(),
```

iii. This places the eye stream on the hardware-synchronized ophys clock before segmentation.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It uses the trials-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`, requiring exactly one true value.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
oc = [o for o in OUTCOMES if bool(row[o])]
if len(oc) != 1:
    continue
```

iii. The agent found these labels complete and mutually exclusive for go/catch trials and used the canonical four-outcome scheme.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The sole true outcome is converted to its index in `OUTCOMES` (0–3), then broadcast across every time point of that trial.

ii.
```python
outcome=OUTCOMES.index(oc[0]),
...
out[4] = t['outcome']
```

iii. The output is conceptually static per trial, but broadcasting lets it coexist in the common `(5, T)` output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing eye tables or fewer than 100 finite pupil samples cause experiment exclusion. Blink/nonfinite pupil samples are interpolated, but trials with under 50% frames within 0.5 s of a valid pupil sample are dropped. Very short or ambiguously labeled trials and sessions with fewer than two trials are removed. Zero-variance neural traces are safely divided by one.

ii.
```python
if len(et) == 0:
    return dict(eid=eid, skip='no eye tracking')
...
sd[sd == 0] = 1.0
...
if np.mean(pupil_valid[a:b]) < 0.5:
    continue
```

iii. The trajectory confirms one zero-row eye-tracking experiment was excluded and emphasizes retaining almost all otherwise valid go/catch trials while preventing fabricated pupil labels.

## 9-a. What are the most time-consuming steps of the code?

i. Reading and parsing many large NWB files and extracting full-session event/behavior tables are the main conversion cost. The code parallelizes experiment processing. Full decoder training was also time-consuming but is outside conversion.

ii.
```python
with Pool(args.workers) as pool:
    results = pool.map(process_experiment, [(e, args.signal) for e in eids])
```

iii. The trajectory repeatedly measured NWB loading and used 22 workers for the full 88-experiment run; it also reports a 4 GB output and a later full GPU decoder run.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The stimulus-presentation fill loop, per-trial extraction loop, global-image remapping list comprehension, and repeated `subjects.index`/`regions.index` lookups could be vectorized or replaced with dictionaries. Variable trial lengths make full trial extraction less straightforward.

ii.
```python
for a, b, i, c in zip(starts, stops, idx, is_change):
    image_id[a:b] = i
...
for tid, row in gc.iterrows():
    ...
for t in r['trials']:
    ...
```

iii. The trajectory focused optimization on process-level parallelism because file loading dominated; it did not explicitly discuss these loop-level alternatives.

## 9-c. What processing does the code repeat multiple times?

i. It repeatedly concatenates each session's trial arrays to compute running and pupil percentiles, repeatedly searches global image/subject/region lists, and loops over trials once during extraction and again during assembly. It also stores copied trial arrays before copying/casting them into final output structures.

ii.
```python
run_all = np.concatenate([t['running'] for t in r['trials']])
pup_all = np.concatenate([t['pupil'] for t in r['trials']])
...
subjects.index(r['mouse_id'])
regions.index(r['structure'])
```

iii. The trajectory does not explicitly justify these repetitions; they reflect a clear two-stage design separating raw extraction from global coding/binning.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes and retains extensive `session_info` and trial timing/ID metadata used mainly for provenance, calculates pupil-valid distance for QC only, and copies arrays during both trial extraction and final assembly. The decoder does not consume many metadata fields. The optional raw-events/global-binning branches are unused in the default output.

ii.
```python
trials.append(dict(... trial_id=int(tid), change_time=float(row.change_time),
                   start_time=float(row.start_time), stop_time=float(row.stop_time)))
...
session_info.append(dict(... cell_specimen_ids=[int(c) for c in r['cell_ids']]))
```

iii. The trajectory used timing and identifiers for sanity checks and reproducibility, so they helped validation even though downstream decoder training discards them.
