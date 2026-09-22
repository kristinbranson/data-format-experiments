# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent opens the local AllenSDK `VisualBehaviorOphysProjectCache`, gets the ophys experiment table, restricts it to experiment IDs with local NWB files, excludes passive experiments, groups the remainder into sessions, and loads each experiment through `get_behavior_ophys_experiment`. Sessions are processed in a multiprocessing pool.

ii.
```python
return VisualBehaviorOphysProjectCache.from_local_cache(
    cache_dir=CACHE_DIR, use_static_cache=False)
...
et = cache.get_ophys_experiment_table()
avail = sorted(int(os.path.basename(f).split('_')[-1].split('.')[0])
               for f in glob.glob(NWB_GLOB))
sub = et.loc[et.index.isin(avail)].copy()
...
active = et[~et['passive']].copy()
datasets = [cache.get_behavior_ophys_experiment(int(e)) for e in eids]
```

iii. The notes say the local cache contains only a subset of the release, so checking locally available IDs avoids attempted downloads. All scientific data are still accessed through AllenSDK, never by opening NWB files directly. Passive sessions were excluded because they have no behavioral report/outcome.

## 1-b. How are the data split into subjects?

i. Subjects are unique mouse IDs encountered among retained sessions; each session receives an index into that ordered list.

ii.
```python
mouse_id=str(ds0.metadata['mouse_id'])
...
if r['mouse_id'] not in subjects:
    subjects.append(r['mouse_id'])
data['subject_idx'].append(subjects.index(r['mouse_id']))
```

iii. The mouse ID is the AllenSDK animal identifier. The notes report 38 mice in the locally available active subset.

## 1-c. How are the data split into sessions?

i. A session is one `ophys_session_id`. All experiments/imaging planes sharing that ID are combined along the neuron dimension.

ii.
```python
groups = active.groupby('ophys_session_id')
jobs = [(int(s), list(groups.get_group(s).index.values), region_map, i < n_debug,
         args.neural_signal) for i, s in enumerate(session_ids)]
...
neural = np.concatenate(neural_planes, axis=0)
```

iii. The notes state that these planes were recorded simultaneously and have identical trial tables, so combining them reconstructs the behavioral recording session.

## 1-d. How are the data split into trials?

i. Trial rows come from `ds0.trials`. Each retained row is represented by a fixed window from 2.25 s before through 3.75 s after `change_time`, divided into 24 bins.

ii.
```python
trials = ds0.trials
sel = ((trials['go'] | trials['catch']) & (~trials['aborted'])
       & (~trials['auto_rewarded']))
trials = trials[sel]
trials = trials[~trials['change_time'].isna()]
change_times = trials['change_time'].values.astype(np.float64)
edges_abs = change_times[:, None] + BIN_EDGES[None, :]
```

iii. The agent chose a fixed change-centered window that its scans showed was inside every trial. It gives equal-length examples, includes three flash cycles before and five after the change, and avoids contamination from neighboring trials.

## 1-e. How are trials filtered based on quality controls?

i. Only go or catch trials are retained; aborted, auto-rewarded, and missing-change-time trials are removed. Trials lacking complete binned running or pupil values are also removed, and a session must retain at least two trials. Passive sessions and sessions with no eye tracking are excluded.

ii.
```python
sel = ((trials['go'] | trials['catch']) & (~trials['aborted'])
       & (~trials['auto_rewarded']))
...
keep = (~np.isnan(pupil).any(axis=1)) & (~np.isnan(running).any(axis=1))
if keep.sum() < 2:
    out['error'] = 'fewer than 2 trials with complete behaviour'
```

iii. The first exclusions follow the task. The notes argue that fabricating a required decoded behavioral variable would corrupt evaluation; consequently, incomplete trials and three sessions without eye tracking were dropped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. In the delivered default run, neural data are derived from each experiment's Allen-pipeline `dff_traces['dff']` and `ophys_timestamps`. The script optionally supports raw or filtered detected events.

ii.
```python
if signal == 'dff':
    ev = np.vstack(ds.dff_traces['dff'].values).astype(np.float64)
elif signal == 'filtered_events':
    ev = np.vstack(ds.events['filtered_events'].values).astype(np.float64)
else:
    ev = np.vstack(ds.events['events'].values).astype(np.float64)
```

iii. Although the paper says it used detected events, the agent selected dF/F after sample decoder tests showed higher accuracy and because an Allen reference trial-analysis script and tutorials use dF/F.

## 2-b. How is the `neural` data processed?

i. Traces are binned by timestamp using cumulative sums. Events are summed; the delivered dF/F signal is divided by frame count to obtain a mean per 250 ms bin. Planes are concatenated, and stored trial arrays are float32.

ii.
```python
binned = bin_sum(ev, ts, edges_abs)
if signal == 'dff':
    binned = binned / np.maximum(n_per_bin, 1)
neural_planes.append(binned)
...
neural = np.concatenate(neural_planes, axis=0)
```

iii. Averaging prevents different microscope frame rates from scaling dF/F. Fixed bins were chosen to support a common resolution across 31 Hz and 11 Hz recordings and to coincide with the 250 ms image duration.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No extra cell-level filter is applied beyond AllenSDK's released/default valid-ROI curation. Sessions can be lost because behavioral data are incomplete, but neurons within a retained experiment are not selectively removed.

ii.
```python
cell_ids += list(ds.events.index.values)
```

iii. The agent verified that all returned ROIs were marked valid and concluded further neuron filtering was neither needed nor described by the source work.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to each trial's real or sham `change_time`; absolute bin edges are offsets from that time.

ii.
```python
change_times = trials['change_time'].values.astype(np.float64)
edges_abs = change_times[:, None] + BIN_EDGES[None, :]
```

iii. The notes cite Allen trial-response code that aligns activity to change time and emphasize that all streams use the same absolute edges.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 250 ms. Native ophys samples are explicitly rebinned into 24 bins spanning [-2.25, 3.75) s.

ii.
```python
BIN_SIZE = 0.25
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
```

iii. The bin matches the image-on duration, divides the 750 ms flash cycle evenly, and contains at least two frames even for the slowest recordings.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `stimulus_presentations.start_time` and `image_name`, restricted to the change-detection block and excluding rows named `omitted`.

ii.
```python
sp = ds0.stimulus_presentations
sp = sp[sp['stimulus_block_name'].astype(str).str.contains('change_detection')]
shown = sp[sp['image_name'] != 'omitted']
flash_start = shown['start_time'].values.astype(np.float64)
flash_image = shown['image_name'].values.astype(str)
```

iii. The agent used presentation records rather than only trial initial/change columns so that repeated flashes and omissions within the window are represented.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. At every bin center, the most recent non-omitted image onset is found. Its name is held through the gray interval and omissions, then mapped through one sorted, dataset-wide 16-image vocabulary.

ii.
```python
j = np.searchsorted(flash_start, centers_abs, side='right') - 1
image_name = flash_image[j]
...
image_values = sorted({str(im) for r in good for im in np.unique(r['image_name'])})
image_to_idx = {im: i for i, im in enumerate(image_values)}
```

iii. Global names avoid conflating the same per-session image index across image sets A and B. Holding the identity implements the paper's 750 ms presentation interval.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at the centers of the same 250 ms bins used for neural activity.

ii.
```python
centers_abs = change_times[:, None] + BIN_CENTERS[None, :]
j = np.searchsorted(flash_start, centers_abs, side='right') - 1
```

iii. Shared change times and bin geometry make the output and neural columns correspond exactly.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses `stimulus_presentations.start_time` and `is_change` after the same change-detection/non-omitted filtering.

ii.
```python
flash_start = shown['start_time'].values.astype(np.float64)
flash_ischange = shown['is_change'].values.astype(bool)
```

iii. `is_change` distinguishes a real go-trial image change from a catch-trial sham event.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The most recent flash is found for each bin center, the elapsed time since its onset is calculated, and a binary label marks the 750 ms presentation interval after real change flashes.

ii.
```python
since = centers_abs - flash_start[j]
image_change = (flash_ischange[j] & (since < FLASH_CYCLE)).astype(np.int64)
```

iii. The notes define one presentation interval as a 250 ms image plus 500 ms gray; the label is zero throughout catch trials.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is intrinsically binary: class 1 requires the most recent shown flash to have `is_change=True` and its onset to be less than 0.75 s before the bin center; otherwise class 0.

ii.
```python
image_change = (flash_ischange[j] & (since < 0.75)).astype(np.int64)
```

iii. This produces exactly three change bins per go trial and none per catch trial, which the agent explicitly validated.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is evaluated at the same change-aligned 250 ms bin centers as image identity and the binned neural signal.

ii.
```python
j = np.searchsorted(flash_start, centers_abs, side='right') - 1
```

iii. The sample and full-data checks confirmed that the change flash begins at the left edge of bin 9 and labels bins 9-11.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from the SDK `running_speed` table's `speed` and `timestamps` columns.

ii.
```python
run = ds0.running_speed
run_t = run['timestamps'].values.astype(np.float64)
run_v = run['speed'].values.astype(np.float64)
```

iii. The notes state this is already the SDK-processed/filtered wheel-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. NaN-aware means are computed within each 250 ms trial bin, then converted to dataset-wide quintile labels.

ii.
```python
running = bin_mean_nan(run_v, run_t, edges_abs)
...
run_edges = quantile_bins(all_run)
run_q = digitize_with(r['run_edges'], r['running'])
```

iii. Global equal-percentile edges give consistent labels and approximately equal class populations across the complete dataset.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four global edges at the 20th, 40th, 60th, and 80th percentiles define integer categories 0-4 via `np.digitize`.

ii.
```python
qs = np.linspace(0, 100, n_bins + 1)[1:-1]
return np.percentile(values, qs)
...
return np.digitize(values, edges).astype(np.int64)
```

iii. This is the agent's literal interpretation of “five equal percentile bins”; it also outperformed per-session percentiles in its comparison.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running samples are averaged between the exact absolute edges used to bin neural activity.

ii.
```python
running = bin_mean_nan(run_v, run_t, edges_abs)
```

iii. No separate resampled time grid is introduced, so each running label corresponds to the same 250 ms interval as its neural column.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It is computed from SDK `eye_tracking.pupil_area` and `timestamps`; blink-contaminated samples are already NaN in the SDK table.

ii.
```python
eye_t = eye['timestamps'].values.astype(np.float64)
pupil_area = eye['pupil_area'].values.astype(np.float64)
pupil_diam = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. The geometric conversion produces a diameter rather than using area as the requested target.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area is converted to equivalent-circle diameter, internal NaN gaps no longer than 0.5 s are linearly interpolated, values are averaged per 250 ms bin, and the means are globally discretized.

ii.
```python
pupil_diam = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diam = interpolate_short_gaps(eye_t, pupil_diam, PUPIL_MAX_INTERP_GAP)
pupil = bin_mean_nan(pupil_diam, eye_t, edges_abs)
```

iii. The agent considered short blink interpolation reasonable but left long gaps missing and dropped affected trials rather than inventing pupil targets.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global 20/40/60/80 percentile edges define labels 0-4 with `np.digitize`.

ii.
```python
pup_edges = quantile_bins(all_pup)
pup_q = digitize_with(r['pup_edges'], r['pupil'])
```

iii. Global edges make the five labels consistent across sessions and exactly balanced over retained bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil samples are averaged inside the same absolute 250 ms edges as neural activity.

ii.
```python
pupil = bin_mean_nan(pupil_diam, eye_t, edges_abs)
```

iii. Independent spot checks in the notes reproduced all pupil bins and labels from SDK values.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It comes from the four boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
for k, name in enumerate(OUTCOMES):
    outcome[trials[name].values.astype(bool)] = k
```

iii. These are the SDK's mutually exclusive outcome definitions for go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The true boolean column is mapped to class 0-3, missing labels cause session rejection, and the per-trial class is broadcast across all 24 time bins.

ii.
```python
if np.any(outcome < 0):
    out['error'] = 'trial without an outcome label'
...
np.full(N_BINS, r['outcome'][t], dtype=np.int64)
```

iii. Broadcasting satisfies the output matrix shape while preserving a static per-trial target.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent explicitly handles absent NWBs, missing change times, absent eye tables, blink NaNs, incomplete behavior bins, too-few trials, invalid outcome labels, and per-session loading exceptions. Short pupil gaps are interpolated; long gaps cause trial removal; excluded sessions are recorded.

ii.
```python
if len(eye) == 0:
    out['error'] = 'no eye tracking data'
    return out
...
keep = (~np.isnan(pupil).any(axis=1)) & (~np.isnan(running).any(axis=1))
...
except Exception as exc:
    out['error'] = f'{exc!r}\n{traceback.format_exc()[-800:]}'
```

iii. The notes emphasize complete accounting: 917 trials belonged to three no-eye sessions and 2,071 more had incomplete behavior; no missing target was silently fabricated.

## 9-a. What are the most time-consuming steps of the code?

i. AllenSDK loading of one or more NWBs per session dominates, followed by trace binning, global assembly, statistics, and writing a large pickle.

ii.
```python
with Pool(min(args.nproc, len(jobs))) as pool:
    results = pool.map(process_session, jobs)
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes measured roughly 3 s per plane serially and 90 s total with 16 workers; pickling and statistics took under 10 s.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial-wise binning was already vectorized with cumulative sums and `searchsorted`. Remaining assembly loops over sessions/trials and region/name mappings could be partially vectorized, though the required nested-list output limits gains. `np.vectorize` for image mapping is still Python-level work.

ii.
```python
idx = np.searchsorted(timestamps, edges_abs.ravel()).reshape(edges_abs.shape)
return (take[:, :, 1:] - take[:, :, :-1]).astype(np.float32)
...
img = np.vectorize(lambda x: image_to_idx[str(x)])(r['image_name'])
for t in range(n_trials):
```

iii. The agent specifically identified naive per-trial trace slicing as wasteful and replaced it with vectorized session-level binning.

## 9-c. What processing does the code repeat multiple times?

i. Every worker reopens the cache and loads a session's experiment objects. Frame-bin counts are recomputed for each imaging plane even though planes in a session share bin edges (and often timestamps). Membership/index construction for subjects and regions repeatedly uses linear list searches.

ii.
```python
cache = get_cache()
datasets = [cache.get_behavior_ophys_experiment(int(e)) for e in eids]
...
for ds, eid in zip(datasets, eids):
    n_per_bin = bin_sum(np.ones((1, ts.shape[0])), ts, edges_abs)
...
subjects.index(r['mouse_id'])
```

iii. Per-worker caches enable multiprocessing isolation. The notes acknowledge cache/NWB loading as expensive; repeated small mappings were not identified as a practical bottleneck.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It retains bookkeeping such as cell IDs and change/go/catch arrays during conversion although these are not placed in the final decoder fields. Debug mode additionally loads raw events, lick/reward times, and plotting data. Global quantile edges are computed even when optional session-scoped quantiles are requested.

ii.
```python
cell_ids += list(ds.events.index.values)
...
change_times=change_times[keep], go=..., catch=..., cell_ids=np.array(cell_ids)
...
all_run = np.concatenate([r['running'].ravel() for r in good])
run_edges = quantile_bins(all_run)
if args.quantile_scope == 'session':
```

iii. Most bookkeeping supports validation, plots, and session metadata, then becomes eligible for garbage collection after final assembly. Debug-only work is intentionally optional.
