# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent opens the local AllenSDK cache with `VisualBehaviorOphysProjectCache`, gets the ophys experiment table, restricts it to experiment IDs whose NWB files are locally present, excludes passive sessions, groups experiments by ophys session, and loads every experiment in each group through `get_behavior_ophys_experiment`. Thus it processes the locally downloaded active single- and multi-plane data, not every row advertised by the cache.

ii.
```python
return VisualBehaviorOphysProjectCache.from_local_cache(
    cache_dir=CACHE_DIR, use_static_cache=False)
et = cache.get_ophys_experiment_table()
ids = sorted(int(os.path.basename(f).split('_')[-1].split('.')[0])
             for f in glob.glob(NWB_GLOB))
return et.loc[et.index.isin(ids)].copy()
...
act = et[~et['passive'].astype(bool)].copy()
...
exps.append((eid, cache.get_behavior_ophys_experiment(eid)))
```

iii. The notes say the local file restriction avoids attempting absent data, SDK-only access satisfies the instruction, and active sessions are selected because passive sessions lack meaningful licking, reward, and trial outcome. Simultaneous planes are all loaded to form a population.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` strings encountered while assembling retained sessions. Each session stores the index of its mouse in that ordered list.

ii.
```python
mouse = info['mouse_id']
if mouse not in subjects:
    subjects.append(mouse)
data['subject_idx'].append(subjects.index(mouse))
```

iii. The notes identify `mouse_id` as the animal identifier and require `subject_idx` to map each ophys session to that animal.

## 1-c. How are the data split into sessions?

i. One output session is one `ophys_session_id`. All experiments/imaging planes sharing that ID are processed together and concatenated along the neuron dimension. Session IDs are sorted numerically for deterministic output.

ii.
```python
groups = act.groupby('ophys_session_id')
session_ids = sorted(groups.groups.keys())
...
exp_ids = list(groups.get_group(sid).index.values)
...
neural = np.concatenate(neural_planes, axis=0)
```

iii. The notes justify this as the physically correct simultaneous population recording; an ophys experiment is a plane, whereas an ophys session comprises all planes recorded together.

## 1-d. How are the data split into trials?

i. Trials come from the SDK `trials` table. Only go/catch rows with finite `change_time` are retained. Each trial is a fixed window from 3 seconds before to 3 seconds after `change_time`, divided into 24 bins.

ii.
```python
keep = (trials['go'].astype(bool) | trials['catch'].astype(bool))
tr = trials[keep].copy()
tr = tr[np.isfinite(tr['change_time'].values)]
...
edges = change_times[:, None] + (
    OFF_START + BIN_SIZE * np.arange(NBINS + 1))[None, :]
```

iii. The agent reports that every retained trial has at least 3.02 seconds before and 4.20 seconds after the change, that adjacent change windows do not overlap, and that six seconds covers eight flash cycles.

## 1-e. How are trials filtered based on quality controls?

i. `go | catch` implicitly excludes aborted and auto-rewarded rows. Trials lacking a finite change time, lying outside the common coverage of neural/running/eye/stimulus streams, or lacking one of the four outcomes are dropped. Sessions with fewer than two remaining trials are dropped.

ii.
```python
tr = trials[trials['go'].astype(bool) | trials['catch'].astype(bool)].copy()
tr = tr[np.isfinite(tr['change_time'].values)]
valid = ((change_times + OFF_START) >= t_lo) & ((change_times + OFF_END) <= t_hi)
tr = tr[valid]
...
ok = outcome >= 0
...
if ntrials < 2:
    return None
```

iii. The notes equate go/catch with the reference `contingent_trials` mask and the explicit instruction to exclude aborted/auto-rewarded trials. Coverage and outcome checks prevent incomplete arrays or invalid labels.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. By default, neural data come from each experiment’s SDK `dff_traces['dff']`; command-line options can instead select `events['events']` or `events['filtered_events']`.

ii.
```python
tbl = e.events if neural_source in ('events', 'filtered_events') else e.dff_traces
col = {'events': 'events', 'filtered_events': 'filtered_events',
       'dff': 'dff'}[neural_source]
traces = np.vstack(tbl[col].values)
...
ap.add_argument('--neural', default='dff',
                choices=['events', 'filtered_events', 'dff'])
```

iii. Although the paper used detected events, the agent tested all three streams and chose dF/F because sparse raw events performed poorly with the supplied instantaneous decoder; dF/F gave higher held-out accuracy.

## 2-b. How is the `neural` data processed?

i. Optionally each full-session neuron trace is z-scored (off by default). Samples are averaged within every 250 ms trial bin using cumulative sums and `searchsorted`; planes are concatenated, and trial matrices are stored as float32.

ii.
```python
if zscore:
    mu = traces.mean(axis=1, keepdims=True)
    sd = traces.std(axis=1, keepdims=True)
    traces = (traces - mu) / np.maximum(sd, 1e-9)
binned, counts = bin_means(traces, ts, edges)
neural_planes.append(binned.astype(np.float32))
neural = np.concatenate(neural_planes, axis=0)
```

iii. The notes argue that 250 ms integrates activity for an instantaneous decoder, corresponds to image presentation duration, and supplies data from both ~31 Hz and ~11 Hz rigs. Vectorized cumulative-sum binning is used for speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No extra cell-level filter is applied. Empty tables/planes are skipped; a session is dropped if none remains. The agent relies on AllenSDK’s default valid-ROI filtering.

ii.
```python
if len(tbl) == 0:
    continue
...
if not neural_planes:
    return None
```

iii. The notes state that the released SDK cell tables already apply ROI/cell quality control (`exclude_invalid_rois=True` by default), so further filtering was unnecessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All trials are aligned to `trials.change_time` (the sham change time for catch trials) and cover `[-3,+3)` seconds on the common synchronized clock.

ii.
```python
change_times = tr['change_time'].values.astype(np.float64)
edges = change_times[:, None] + (
    OFF_START + BIN_SIZE * np.arange(NBINS + 1))[None, :]
binned, counts = bin_means(traces, ts, edges)
```

iii. The agent verified change times coincide with flash onsets and chose this common task event so go and catch trials have comparable stimulus-centered windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The default resolution is 250 ms, producing 24 bins per six-second trial. Native ophys samples are averaged into those bins, so temporal rebinning is applied. A CLI flag can change the bin size.

ii.
```python
BIN_SIZE = 0.25
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
...
binned, counts = bin_means(traces, ts, edges)
...
time_bin_size=BIN_SIZE * 1000.0
```

iii. The notes tie 250 ms to the illuminated-image duration, one third of the 750 ms cycle, and sufficient ophys samples even on the multi-plane rig.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from behavior-block `stimulus_presentations.image_name` and `start_time`, rather than the trial table’s initial/change image fields.

ii.
```python
beh = sp[sp['stimulus_block_name'] == BEHAVIOR_BLOCK].copy()
pres_start = beh['start_time'].values.astype(np.float64)
pres_image = beh['image_name'].values.astype(object)
```

iii. The notes say stimulus presentations represent the actual time-varying image stream, including omissions, and match the paper’s image-presentation-interval convention.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For every bin center, the most recent stimulus onset is found. Its image name is mapped through a fixed global 17-class map (16 images plus `omitted`). The label is therefore held through the associated 750 ms image/gray interval.

ii.
```python
pidx = np.searchsorted(pres_start, centers.ravel(), side='right') - 1
pidx = np.clip(pidx, 0, len(pres_start) - 1)
img_names = pres_image[pidx]
img_identity = np.array([IMAGE_TO_IDX[n] for n in img_names],
                        dtype=np.int64).reshape(ntrials, NBINS)
```

iii. The agent says this gives every time bin an identity while following the paper’s 750 ms interval convention; omission intervals receive their own class.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at the centers of exactly the same change-relative 250 ms bins used for neural averages.

ii.
```python
centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
pidx = np.searchsorted(pres_start, centers.ravel(), side='right') - 1
```

iii. The notes emphasize that trials, stimulus presentations, and ophys timestamps share the Allen synchronization clock.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from behavior-block `stimulus_presentations.is_change` and presentation start times (consistent with go-trial `change_time`).

ii.
```python
pres_start = beh['start_time'].values.astype(np.float64)
pres_change = beh['is_change'].astype(bool).values
```

iii. The agent verified that all trial change times coincide with stimulus-presentation onsets and uses the SDK’s explicit change flag.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The presentation containing each bin center is selected, and its Boolean `is_change` flag is converted to integer. Thus the changed presentation labels three consecutive 250 ms bins.

ii.
```python
pidx = np.searchsorted(pres_start, centers.ravel(), side='right') - 1
img_change = pres_change[pidx].astype(np.int64).reshape(ntrials, NBINS)
```

iii. The notes define “right after” as the 750 ms changed-image presentation interval (image plus following gray period), matching the paper’s interval convention.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is estimated: the SDK Boolean is directly encoded as 0 (`no_change`) or 1 (`change`).

ii.
```python
img_change = pres_change[pidx].astype(np.int64).reshape(ntrials, NBINS)
...
['no_change', 'change']
```

iii. The raw field is already categorical; go trials have a changed presentation, while catch trials remain zero.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change flag is sampled at the same bin centers whose edges are used to average neural activity.

ii.
```python
centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
pidx = np.searchsorted(pres_start, centers.ravel(), side='right') - 1
```

iii. Shared synchronized timestamps and the common bin grid provide alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from `running_speed['speed']` and `running_speed['timestamps']` of the first experiment in the session.

ii.
```python
run = ref.running_speed
run_t = run['timestamps'].values.astype(np.float64)
run_v = interpolate_nans(run['speed'].values.astype(np.float64), run_t)
```

iii. Behavior streams are shared across simultaneous planes, and the SDK table is the standard processed wheel-speed source.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Missing samples are linearly interpolated (nearest valid value at the edges), speed is averaged within each 250 ms bin, and the binned values are discretized into quintiles separately within each session.

ii.
```python
run_v = interpolate_nans(run['speed'].values.astype(np.float64), run_t)
run_binned = bin_means(run_v, run_t, edges)[0][0]
run_q = quantile_bin(run_binned)
```

iii. The notes say bin averaging aligns behavior with neural data, and per-session quintiles balance classes while avoiding rig/mouse/session scale or propensity differences.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 20th, 40th, 60th, and 80th percentiles of all binned running values in a session form five integer categories 0–4.

ii.
```python
edges = np.percentile(flat, np.linspace(0, 100, nq + 1)[1:-1])
return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. The agent chose session-specific equal-percentile classes to give approximately 20% occupancy per class in every session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running samples are averaged over the identical absolute-time bin edges used for each trial’s neural traces.

ii.
```python
run_binned = bin_means(run_v, run_t, edges)[0][0]
...
binned, counts = bin_means(traces, ts, edges)
```

iii. The notes state all streams use the common hardware synchronization clock; using identical edges aligns them despite different native rates.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It is derived from `eye_tracking.pupil_area` and eye-tracking timestamps. Area is converted to equivalent circular diameter.

ii.
```python
eye_t = eye['timestamps'].values.astype(np.float64)
pupil_d = 2.0 * np.sqrt(eye['pupil_area'].values.astype(np.float64) / np.pi)
```

iii. The notes describe `2*sqrt(area/pi)` as a diameter in pixels and report checking the area against pupil width/height.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Non-finite blink/missing frames are linearly interpolated, diameter is averaged within each 250 ms bin, then discretized into per-session quintiles. Sessions with no eye table or fewer than two finite points are dropped.

ii.
```python
pupil_d = interpolate_nans(pupil_d, eye_t)
if pupil_d is None:
    return None
pupil_binned = bin_means(pupil_d, eye_t, edges)[0][0]
pupil_q = quantile_bin(pupil_binned)
```

iii. The notes justify interpolation for short blink gaps, dropping sessions where the required output cannot be defined, and session-specific percentiles because pixel scale varies by camera geometry.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The 20th, 40th, 60th, and 80th percentiles of all binned pupil values in each session yield labels 0–4.

ii.
```python
pupil_q = quantile_bin(pupil_binned)
...
edges = np.percentile(flat, np.linspace(0, 100, nq + 1)[1:-1])
```

iii. The agent sought equal class occupancy within every session and robustness to session-specific camera scale.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated eye samples are averaged over the same absolute-time bin edges as neural data.

ii.
```python
pupil_binned = bin_means(pupil_d, eye_t, edges)[0][0]
...
binned, counts = bin_means(traces, ts, edges)
```

iii. The common Allen sync clock and shared bin edges are the stated alignment mechanism.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from the Boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
outcome[tr['hit'].astype(bool).values] = 0
outcome[tr['miss'].astype(bool).values] = 1
outcome[tr['false_alarm'].astype(bool).values] = 2
outcome[tr['correct_reject'].astype(bool).values] = 3
```

iii. The notes identify these as the four canonical mutually exclusive outcomes for go and catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four flags are mapped to codes 0–3. Rows with no recognized outcome are removed, and each retained static code is broadcast across all 24 time bins.

ii.
```python
ok = outcome >= 0
...
outcome_bins = np.repeat(outcome[:, None], NBINS, axis=1)
```

iii. Broadcasting satisfies the decoder’s time-series output layout while preserving a static per-trial target; invalid outcome rows are not assigned an artificial class.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Running and pupil NaNs are interpolated when at least two valid samples exist. Sessions without usable behavior, eye, neural, or at least two valid trials are dropped. Candidate windows must lie inside all streams. Individual experiment-load failures are logged and other planes may continue; worker exceptions drop the session. Empty time bins are filled with zero.

ii.
```python
if good.sum() < 2:
    return None
x[~good] = np.interp(t[~good], t[good], x[good])
...
valid = ((change_times + OFF_START) >= t_lo) & ((change_times + OFF_END) <= t_hi)
...
out[np.broadcast_to(counts[None, :, :] == 0, out.shape)] = 0.0
```

iii. The notes favor interpolation for short blink gaps, complete-window filtering to avoid extrapolation, and session dropping when a required output cannot be defined.

## 9-a. What are the most time-consuming steps of the code?

i. Loading/decoding experiment NWB content through AllenSDK and binning large full-session neural arrays are the principal costs. The code parallelizes sessions with 24 worker processes.

ii.
```python
exps.append((eid, cache.get_behavior_ophys_experiment(eid)))
...
with ProcessPoolExecutor(max_workers=args.workers) as ex:
    futs = {ex.submit(_worker, j): j[0] for j in jobs}
```

iii. The notes specifically identify naive trial slicing and reopening NWBs as costly, and report large gains from vectorized binning, one SDK load per experiment, and process-level session parallelism.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Core trial/bin averaging is already vectorized with cumulative sums and `searchsorted`. Remaining avoidable Python loops include per-image name mapping, per-session region lookup with repeated list searches, and construction of per-trial output/list objects; experiment loading itself cannot simply be vectorized.

ii.
```python
img_identity = np.array([IMAGE_TO_IDX[n] for n in img_names], dtype=np.int64)
...
np.array([regions.index(r) for r in res['region_names']], dtype=np.int64)
...
output_list = [np.stack([...], axis=0) for i in range(ntrials)]
```

iii. The agent’s stated optimization focus was the formerly expensive trial/bin loop, replaced by `bin_means`; it considered I/O dominant after that change.

## 9-c. What processing does the code repeat multiple times?

i. `bin_means` is run independently for every neural plane and again for running and pupil. Each worker also creates a separate cache object. During assembly, list membership/index scans are repeated for subjects and regions. Optional plotting re-reads/reshapes traces already accessed for conversion.

ii.
```python
cache = get_cache()
...
binned, counts = bin_means(traces, ts, edges)
...
run_binned = bin_means(run_v, run_t, edges)[0][0]
pupil_binned = bin_means(pupil_d, eye_t, edges)[0][0]
```

iii. The notes say each experiment is intentionally loaded only once and reused within its session; repetition across streams is necessary because they have distinct timestamps and values.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It retains/constructs diagnostics not consumed by decoder tensors, including `cell_ids`, raw binned running/pupil arrays used only for ranges/plots, detailed timing metadata, and optional large processing plots. The conversion also calculates per-bin sample counts mainly for diagnostics. These do not alter final decoder inputs or targets.

ii.
```python
cell_ids += list(tbl.index.values)
...
running_speed_range=[float(run_binned.min()), float(run_binned.max())],
pupil_diameter_range=[float(pupil_binned.min()), float(pupil_binned.max())],
...
return dict(neural=neural_list, output=output_list, region_names=region_names,
            cell_ids=cell_ids, info=info)
```

iii. The agent presents these as validation/provenance aids. `cell_ids` in particular are returned by workers but never inserted into the final pickle, while plots are only generated when explicitly requested.
