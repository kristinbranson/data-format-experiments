# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment metadata CSV, inventories available NWB files, retains available active single-plane `VisualBehavior` experiments, and loads each with `BehaviorOphysExperiment.from_nwb_path`. It parallelizes experiment conversion and caches per-experiment results.

ii.
```python
exp = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
exp = exp[exp.ophys_experiment_id.isin(available)]
exp = exp[exp.project_code == 'VisualBehavior']
exp = exp[~exp.passive.astype(bool)]
ds = BehaviorOphysExperiment.from_nwb_path(nwb_path)
```

iii. The trajectory shows the agent inspected the release and chose the single-plane project to obtain a common ~31 Hz clock and one plane per experiment. It excluded passive sessions because behavioral outcome is not meaningful there, used local NWBs to avoid repeated SDK downloads, and used process workers plus caches because NWB loading dominates runtime.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id`s among successfully converted experiments; each session receives the corresponding index into the sorted subject list.

ii.
```python
subjects = sorted({s['mouse_id'] for s in sessions})
subject_to_idx = {m: i for i, m in enumerate(subjects)}
subject_idx.append(subject_to_idx[s['mouse_id']])
```

iii. The agent used the Allen metadata’s stable animal identifier and a deterministic sorted vocabulary.

## 1-c. How are the data split into sessions?

i. Each selected single-plane ophys experiment becomes one output session, sorted by `ophys_experiment_id`; its `ophys_session_id` is retained as metadata.

ii.
```python
sessions.sort(key=lambda s: s['ophys_experiment_id'])
for s in sessions:
    neural.append(sess_neural)
```

iii. The agent found that the selected `VisualBehavior` variant has one imaging plane per ophys session, making experiment and session equivalent; this also avoids mixing frame rates with multiplane data.

## 1-d. How are the data split into trials?

i. Trials come from `ds.trials` and are sliced from the first ophys frame at or after `start_time` to the first at or after `stop_time`, producing variable-length trials.

ii.
```python
starts = np.searchsorted(ts, trials['start_time'].values, side='left')
stops = np.searchsorted(ts, trials['stop_time'].values, side='left')
neural_trials.append(np.ascontiguousarray(traces[:, i0:i1]))
```

iii. The built-in trial table is the experiment’s canonical trial definition, and full start-to-stop windows retain pre-change and response periods.

## 1-e. How are trials filtered based on quality controls?

i. Only Go or Catch trials are kept; Aborted and Auto-rewarded trials are removed. Trials with no valid outcome, fewer than two frames, or an unlabeled frame are removed, and experiments with fewer than two remaining trials are dropped.

ii.
```python
keep = ((trials['go'].astype(bool) | trials['catch'].astype(bool))
        & ~trials['aborted'].astype(bool)
        & ~trials['auto_rewarded'].astype(bool))
if i1 - i0 < 2 or np.any(image_name[i0:i1] == None):
    continue
```

iii. This directly follows the requested Go/Catch inclusion and Aborted/Auto-rewarded exclusion, while the additional checks protect format validity.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. By default neural data comes from `ds.dff_traces['dff']`. Environment variable `VB_TRACE` can instead select `events` or `filtered_events`.

ii.
```python
if TRACE == 'dff':
    src, col = ds.dff_traces, 'dff'
else:
    src, col = ds.events, TRACE
```

iii. The agent tested dF/F against event variants and reported materially better decoder accuracy and fewer all-zero trials at 32 ms resolution, so it deliberately selected dF/F despite the paper’s event-based analyses.

## 2-b. How is the `neural` data processed?

i. Released traces are stacked neuron-by-time, converted to float32, checked against the ophys timestamp length, made finite, then sliced into contiguous trial matrices. No normalization or temporal filtering is added.

ii.
```python
traces = np.vstack([np.asarray(v, dtype=np.float32) for v in src[col].values])
np.nan_to_num(traces, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The agent relied on the Allen pipeline’s motion correction, demixing, neuropil subtraction, dF/F baseline normalization, and detrending rather than duplicating them.

## 2-c. How is the `neural` data filtered based on quality controls?

i. All released ROIs are retained; the only extra repair is replacing non-finite trace samples with zero.

ii.
```python
cell_ids = src.index.values
np.nan_to_num(traces, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The agent states that released ROIs already passed Allen ROI QC (`valid_roi`) and that the paper applied no further neuron-level curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples use the ophys clock and are aligned to trial `start_time`; each matrix spans `start_time` through (but not including) the frame at `stop_time`.

ii.
```python
i0, i1 = int(starts[k]), int(stops[k])
neural_trials.append(np.ascontiguousarray(traces[:, i0:i1]))
```

iii. Ophys timestamps are the required master timebase, and selecting the first frame at or after each boundary gives explicit frame-level alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native single-plane ophys frame is retained, about 32.32 ms; no neural rebinning is applied. Metadata uses the median session `dt` in milliseconds.

ii.
```python
'dt': float(np.median(np.diff(ts)))
time_bin_size = float(np.median(dts) * 1000.0)
```

iii. The agent selected only a common-rate dataset variant so native bins could be preserved consistently across sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from change-detection `stimulus_presentations`, specifically `start_time` and `image_name`.

ii.
```python
starts = stim['start_time'].values
names = stim['image_name'].values.astype(object)
```

iii. The agent preferred the presentation table over trial-level initial/change fields because it represents every actual flash and omission.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Every ophys frame is labeled with the most recent flash’s image for the full 750 ms presentation interval; omissions form a category. Global names are deterministically integer-encoded.

ii.
```python
idx = np.searchsorted(starts, ts, side='right') - 1
image_name = np.where(valid, names[idx_clipped], None)
image_to_idx = {name: i for i, name in enumerate(image_values)}
```

iii. The agent cited the paper’s definition of an image-presentation interval as the 750 ms beginning at a flash, including the expected interval for omissions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Labels are first evaluated on every ophys timestamp and then sliced with exactly the same `[i0:i1]` bounds as neural activity.

ii.
```python
image_name, is_change = _flash_labels(stim, ts)
image_trials.append(image_name[i0:i1].copy())
```

iii. A shared ophys timebase and identical slices guarantee samplewise alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It comes from `stimulus_presentations.start_time` and `stimulus_presentations.is_change` in the active change-detection block.

ii.
```python
changes = stim['is_change'].values.astype(bool)
is_change = np.where(valid, changes[idx_clipped], False)
```

iii. This distinguishes real Go changes from Catch sham changes using the presentation record itself.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The most recent flash’s `is_change` flag is held across that entire 750 ms flash cycle and converted to integer 0/1.

ii.
```python
is_change = np.where(valid, changes[idx_clipped], False)
change_trials.append(is_change[i0:i1].astype(np.int64))
```

iii. The agent used the same paper-defined presentation interval as for image identity.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is estimated: the raw boolean flag maps directly to 0 (`no_change`) or 1 (`change`).

ii.
```python
['no_change', 'change']
```

iii. The source flag is already binary, so further thresholding would be unnecessary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The flag is projected onto ophys timestamps and sliced using the same trial indices as neural activity.

ii.
```python
change_trials.append(is_change[i0:i1].astype(np.int64))
```

iii. This yields one change label per neural frame.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `ds.running_speed['timestamps']` and the Allen-processed `speed` in cm/s.

ii.
```python
run_t = np.asarray(run['timestamps'].values, dtype=np.float64)
run_v = np.asarray(run['speed'].values, dtype=np.float64)
```

iii. The SDK’s processed speed includes the documented wheel corrections and filtering.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite samples are linearly interpolated to ophys timestamps. Values pooled from all retained trial frames define global 20/40/60/80 percentile edges, then each sample is digitized.

ii.
```python
running = np.interp(ts, run_t[good], run_v[good])
run_edges = np.percentile(run_all, qs)
run = np.searchsorted(run_edges, s['running_speed'][k], side='right')
```

iii. Global quintiles provide the requested five equal-percentile categories with consistent physical ranges across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four global percentile edges produce integer categories 0 through 4 via right-sided insertion.

ii.
```python
qs = np.linspace(0, 100, NBINS + 1)[1:-1]
np.searchsorted(run_edges, values, side='right')
```

iii. This is a direct implementation of five equal percentile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated to all ophys timestamps before being sliced at the same trial bounds as neural activity.

ii.
```python
running = np.interp(ts, run_t[good], run_v[good])
running_trials.append(running[i0:i1].astype(np.float64))
```

iii. Hardware-synchronized timestamps and shared ophys samples provide framewise alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses `ds.eye_tracking['timestamps']` and `pupil_area`, accepting finite positive area samples.

ii.
```python
t = eye_tracking['timestamps'].values
area = eye_tracking['pupil_area'].values.astype(float)
good = np.isfinite(area) & np.isfinite(t) & (area > 0)
```

iii. The agent notes that the SDK has already made blink-contaminated area samples NaN and chose area-derived equivalent diameter as more robust than either ellipse axis.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Equivalent circular diameter `2*sqrt(area/pi)` is computed, interpolated to ophys time, pooled over retained trials for global quintile edges, and digitized.

ii.
```python
return t[good], 2.0 * np.sqrt(area[good] / np.pi)
pupil = np.interp(ts, pt, pv)
pup_edges = np.percentile(pup_all, qs)
```

iii. This removes invalid/blink points, fills gaps on the master clock, and supplies the required categorical output.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four pooled 20/40/60/80 percentile edges produce categories 0 through 4.

ii.
```python
pup = np.searchsorted(pup_edges, s['pupil_diameter'][k],
                      side='right').astype(np.int64)
```

iii. The agent used one global scale so each class has the same interpretation across experiments.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Clean diameter samples are linearly interpolated onto ophys timestamps and sliced with the neural trial bounds.

ii.
```python
pupil = np.interp(ts, pt, pv)
pupil_trials.append(pupil[i0:i1].astype(np.float64))
```

iii. This supplies one pupil category for every retained neural frame.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the trial table’s `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome[trials[name].astype(bool).values] = i
```

iii. These are the canonical mutually exclusive outcomes of valid Go and Catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes map to fixed codes 0–3; unclassified trials are discarded, and the per-trial code is repeated across all trial frames.

ii.
```python
ok = outcome >= 0
out = np.full(T, s['trial_outcome'][k], dtype=np.int64)
```

iii. Repetition makes this static attribute compatible with the time-varying output matrix and the preference for time-varying outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Non-finite neural values become zero; non-finite running samples are excluded before interpolation; invalid/blink pupil areas are omitted and experiments with fewer than 100 valid eye samples are dropped. Short/unlabeled trials, unknown outcomes, and sessions with fewer than two trials are dropped. Worker exceptions are reported and that experiment is omitted. Interpolation uses endpoint values outside the sampled range.

ii.
```python
np.nan_to_num(traces, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
if good.sum() < 100: return None, None
if pt is None: return None
```

iii. The agent prioritized a complete categorical output at every retained frame and dropped the few experiments unable to supply pupil data rather than inventing a pupil class.

## 9-a. What are the most time-consuming steps of the code?

i. Loading and parsing large NWB experiments and extracting full-session traces are dominant; full conversions and decoder comparisons were also expensive. The implementation parallelizes conversion and caches each experiment.

ii.
```python
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    futs = [pool.submit(_worker, e) for e in eids]
if os.path.exists(cache_path):
    return cache_path
```

iii. The trajectory includes long conversion/training runs and repeated waiting, supporting the agent’s conclusion that I/O/extraction and validation training dominate.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial slicing must remain ragged, but image integer encoding (a Python comprehension per frame), the four-outcome assignment loop, and some vocabulary/metadata loops could be vectorized or mapped categorically. Trial start/stop lookup is already vectorized.

ii.
```python
img = np.array([image_to_idx[n] for n in s['image_name'][k]])
for i, name in enumerate(OUTCOME_NAMES):
    outcome[trials[name].astype(bool).values] = i
```

iii. The agent emphasized that these loops are small relative to NWB loading; it already vectorized timestamp search and interpolation.

## 9-c. What processing does the code repeat multiple times?

i. Trial arrays are traversed once to pool running/pupil values, again to build image vocabularies, and again for final assembly. Cache files are also written, then reopened during assembly.

ii.
```python
run_all = np.concatenate([np.concatenate(s['running_speed']) for s in sessions])
images = sorted({name for s in sessions for tr in s['image_name'] for name in np.unique(tr)})
for s in sessions:
    for k in range(ntrials):
```

iii. Multiple passes keep global vocabularies and percentile thresholds deterministic while limiting the complexity of the per-experiment workers.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `cell_specimen_ids` are loaded and cached but not placed in the final dataset. Cache serialization duplicates intermediate arrays, and extensive `session_info` metadata is not required by the decoder. Optional event-trace branches and label strings do not affect the default numerical conversion.

ii.
```python
cell_ids = src.index.values
'cell_specimen_ids': np.asarray(cell_ids),
# assemble() never consumes s['cell_specimen_ids']
```

iii. These items aid provenance, reproducibility, and alternate-trace experiments, but the downstream decoder does not consume them.
