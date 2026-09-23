# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Local NWB IDs are joined to metadata; only active, familiar, single-plane `VisualBehavior` experiments are loaded directly with `BehaviorOphysExperiment.from_nwb_path`, in parallel and with per-experiment caches. Thus the agent intentionally does not load every VisualBehavior session.

ii.
```python
sel = tbl[tbl.ophys_experiment_id.isin(available)
          & (tbl.project_code == 'VisualBehavior')
          & (~tbl.passive) & (tbl.experience_level == 'Familiar')]
expt = BehaviorOphysExperiment.from_nwb_path(path)
results = pool.map(extract_cached, rows, chunksize=1)
```

iii. The trajectory says active sessions have meaningful outcomes, familiar sessions follow the paper and share image set A, and single-plane sessions share a 31 Hz time base; local discovery avoids fetching absent files.

## 1-b. How are the data split into subjects?

i. Unique metadata `mouse_id` strings define subjects; retained experiments get an index into their first-seen order.

ii.
```python
mouse = str(row['mouse_id'])
if mouse not in subject_list: subject_list.append(mouse)
subject_idx.append(subject_list.index(mouse))
```

iii. The agent treated `mouse_id` as the canonical animal identifier.

## 1-c. How are the data split into sessions?

i. Each retained `ophys_experiment_id` becomes one output session; experiments are not grouped by `ophys_session_id`.

ii.
```python
for s, row in zip(sessions, meta_rows):
    neural.append(n_sess)
```

iii. The agent observed that its selected single-plane subset has one experiment per session; it excluded multiscope sessions where planes would need grouping.

## 1-d. How are the data split into trials?

i. SDK `trials` rows define variable-length trials over `[start_time, stop_time)`.

ii.
```python
i0 = np.searchsorted(ts, tr['start_time'], side='left')
i1 = np.searchsorted(ts, tr['stop_time'], side='left')
out['neural'].append(np.ascontiguousarray(traces[:, i0:i1]))
```

iii. The agent used the experiment's native trial definition to preserve pre-change and post-change periods.

## 1-e. How are trials filtered based on quality controls?

i. Only go/catch trials remain; aborted and auto-rewarded trials, windows under two frames, and trials without exactly one recognized outcome are removed. Sessions need at least two trials.

ii.
```python
keep = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
if i1 - i0 < 2: continue
if sum(bool(x) for x in outcome) != 1: continue
```

iii. The main filter directly follows the instructions; extra checks ensure scoreable trials and decoder-compatible sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. It uses `expt.events['filtered_events']`, not dF/F.

ii.
```python
events = expt.events
traces = np.stack([np.asarray(x, dtype=np.float32)
                   for x in events['filtered_events'].values])
```

iii. The agent followed the paper's detected-events statement. Raw events were too sparse per frame; filtered events retain inferred timing/magnitude. It kept this choice although its test found dF/F decoded better.

## 2-b. How is the `neural` data processed?

i. Filtered event vectors are stacked neuron-by-frame, cast to float32, shape-checked, and sliced contiguously by trial; no further normalization occurs.

ii.
```python
assert traces.shape == (len(cell_ids), nframes)
np.ascontiguousarray(traces[:, i0:i1])
```

iii. The agent relied on Allen preprocessing and considered filtered events a way to reduce indicator decay.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron/activity filtering is applied; all event-table ROIs and even all-zero trial slices remain.

ii.
```python
cell_ids = list(events.index)
traces = np.stack([... for x in events['filtered_events'].values])
```

iii. The trajectory found all released ROIs had `valid_roi=True`; zero-activity trials were considered genuine.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial start is the alignment event. Start/stop times map to the first ophys frames at or after each boundary, and all streams use the same slice.

ii.
```python
i0 = np.searchsorted(ts, tr['start_time'], side='left')
i1 = np.searchsorted(ts, tr['stop_time'], side='left')
```

iii. Ophys timestamps are the required common time base; metadata records `off_start=0` and a variable end.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One native single-plane ophys frame is one bin, about 32.3 ms; no temporal rebinning is applied.

ii.
```python
out['dt'] = float(np.median(np.diff(ts)))
dt_ms = float(np.mean([s['dt'] for s in sessions]) * 1000.0)
```

iii. The agent selected sessions with a common frame rate specifically to avoid neural resampling.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Active, non-omitted `stimulus_presentations` rows supply `image_name`, `start_time`, and `end_time`.

ii.
```python
stim = expt.stimulus_presentations
shown = stim[stim['active'].astype(bool) & ~stim['omitted'].astype(bool)]
```

iii. The agent aimed to represent the image actually visible at each frame, including gray periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Codes 1..N are painted across shown presentations; 0 means gray. Session codes are remapped to one sorted global vocabulary.

ii.
```python
image_id = np.zeros(nframes, dtype=np.int8)
for i0, i1, c in zip(starts, stops, codes): image_id[i0:i1] = c
s['image_id'] = [remap[a] for a in s['image_id']]
```

iii. Global codes keep meanings consistent; gray covers inter-stimulus and omitted intervals.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation boundaries are mapped onto ophys timestamps, then the full-session image vector is sliced with the neural trial indices.

ii.
```python
starts = np.searchsorted(ts, shown['start_time'].to_numpy(float), side='left')
stops = np.searchsorted(ts, shown['end_time'].to_numpy(float), side='left')
out['image_id'].append(image_id[i0:i1])
```

iii. Ophys time is the master clock for every stream.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses active, shown presentations' `is_change`, `start_time`, and `end_time`.

ii.
```python
is_chg = shown['is_change'].astype(bool).to_numpy()
for i0, i1 in zip(starts[is_chg], stops[is_chg]): change[i0:i1] = 1
```

iii. This marks real identity changes while catch-trial sham changes remain zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is set to one only over each changed-image flash (roughly 250 ms).

ii.
```python
change = np.zeros(nframes, dtype=np.int8)
change[i0:i1] = 1
```

iii. The agent initially tested 750 ms, then chose the flash itself because “right after” seemed more literal and decoder accuracy improved.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is inherently binary: changed-flash frames are 1 and all others 0; no fitted threshold is used.

ii.
```python
output_values = [..., ['no_change', 'change'], ...]
```

iii. These are the two requested categories.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change presentation boundaries are converted to ophys frames and sliced using the same trial boundaries as neural data.

ii.
```python
out['change'].append(change[i0:i1])
```

iii. The shared ophys clock provides framewise alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `expt.running_speed.speed` and `timestamps`.

ii.
```python
run = expt.running_speed
running = np.interp(ts, run['timestamps'].to_numpy(float),
                    run['speed'].to_numpy(float))
```

iii. The agent used the SDK's processed running stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated to ophys time, cast to float32, pooled over retained trial frames, then discretized.

ii.
```python
running = np.interp(...).astype(np.float32)
all_run = np.concatenate([np.concatenate(s['running']) for s in sessions])
```

iii. Pooling makes categories comparable across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Global 20/40/60/80 percentile edges form five equal-count bins; `np.digitize` assigns 0–4.

ii.
```python
qs = np.linspace(0, 100, nbins + 1)[1:-1]
run_edges = np.percentile(all_run, qs)
out[2] = np.digitize(s['running'][k], run_edges)
```

iii. This directly implements five global percentile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Full-session speed is interpolated to every ophys timestamp and sliced with neural trial indices.

ii.
```python
running = np.interp(ts, ...)
out['running'].append(running[i0:i1])
```

iii. Ophys timestamps are the common alignment base.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking `pupil_area`, `likely_blink`, and `timestamps`, converting area to an equal-circle diameter.

ii.
```python
area[eye_tracking['likely_blink'].to_numpy(bool)] = np.nan
diam = 2.0 * np.sqrt(area[good] / np.pi)
```

iii. The agent excluded blink frames before interpolation.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink/nonfinite samples are removed, area becomes diameter, and valid samples are linearly interpolated to ophys time with constant edge fill. Sessions with under 50% usable samples are rejected.

ii.
```python
if good.sum() < 0.5 * len(area): return None, None
pupil = np.interp(ts, et, ed).astype(np.float32)
```

iii. This guarantees finite pupil labels; sessions without adequate eye tracking are dropped because pupil is required.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Global pooled 20/40/60/80 percentile edges define five bins.

ii.
```python
pupil_edges = quantile_bins(all_pupil, NQUANTILES)
out[3] = np.digitize(s['pupil'][k], pupil_edges)
```

iii. These are the requested equal-percentile categories.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Clean pupil values are interpolated to ophys timestamps and sliced with neural trial indices.

ii.
```python
pupil = np.interp(ts, et, ed)
out['pupil'].append(pupil[i0:i1])
```

iii. The agent also numerically spot-checked alignment in the trajectory.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It uses trials-table Booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
outcome = [tr['hit'], tr['miss'], tr['false_alarm'], tr['correct_reject']]
```

iii. These are the SDK's scoreable go/catch outcomes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Exactly one flag must be true; its fixed-order index is broadcast across every frame of the trial.

ii.
```python
if sum(bool(x) for x in outcome) != 1: continue
out['outcome'].append(int(np.argmax(outcome)))
out[4] = s['outcome'][k]
```

iii. Broadcasting fits the static label into the common output-by-time matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Blink/nonfinite pupil samples are interpolated; poor/missing eye sessions, invalid trials, and sessions under two trials are skipped. Corrupt caches are rebuilt. Running NaNs receive no explicit handling.

ii.
```python
if et is None: return {'oeid': oeid, 'skip': 'no eye tracking'}
if res.get('skip') or len(res.get('neural', [])) < 2: continue
except Exception: pass  # retry cache extraction
```

iii. The agent prioritized complete output matrices and reported dropping one no-eye session.

## 9-a. What are the most time-consuming steps of the code?

i. Reading/parsing large NWBs and extracting event/behavior tables dominate conversion; multiprocessing and disk caching mitigate this.

ii.
```python
with Pool(min(workers, len(rows))) as pool:
    results = pool.map(extract_cached, rows, chunksize=1)
```

iii. The trajectory timed full conversions and used 16–22 workers, indicating NWB I/O/parsing was the practical bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Presentation painting, `trials.iterrows()`, image remapping with repeated `list.index`, and final assembly could be reduced/vectorized, though variable-length slicing naturally requires iteration.

ii.
```python
for i0, i1, c in zip(starts, stops, codes): image_id[i0:i1] = c
for tid, tr in trials.iterrows(): ...
for i, name in enumerate(s['image_names']): remap[i + 1] = image_names.index(name) + 1
```

iii. The agent did not discuss vectorization; it instead parallelized independent experiments.

## 9-c. What processing does the code repeat multiple times?

i. Trial streams are traversed for pooled bins, image remapping, and final assembly. Image and change intervals are painted separately. Subject/region lookup repeatedly scans lists.

ii.
```python
all_run = np.concatenate([np.concatenate(s['running']) for s in sessions])
for s in sessions: ...
for s, row in zip(sessions, meta_rows):
    for k in range(len(s['neural'])): ...
```

iii. Multiple passes permit global bins/categories after all usable sessions are known; caching avoids repeated NWB loads.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It builds extensive diagnostic metadata/bin labels and full-session aligned behavior/stimulus vectors although decoding uses trial slices. These aid reproducibility but are not required by training.

ii.
```python
session_info.append({'full_genotype': ..., 'equipment_name': ..., 'trial_ids': ...})
image_id = np.zeros(nframes, dtype=np.int8)
change = np.zeros(nframes, dtype=np.int8)
```

iii. The agent did not call this waste; the full-session representation simplifies alignment and the metadata documents provenance.

