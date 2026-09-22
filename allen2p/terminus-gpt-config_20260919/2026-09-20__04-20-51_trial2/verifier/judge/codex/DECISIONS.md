# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively finds local `behavior_ophys_experiment_*.nwb` files, joins them to the local experiment CSV, removes passive experiments and experiments without eye tracking, and streams each retained NWB directly with `h5py`. It does not use the AllenSDK cache loader.

ii.
```python
files = sorted(glob.glob(str(DATA_ROOT / '**' / 'behavior_ophys_experiment_*.nwb'), recursive=True))
meta = metadata_table('ophys_experiment_table.csv')
meta = meta[meta.ophys_experiment_id.isin(files)].copy()
meta = meta[~meta.session_type.str.contains('passive', case=False, na=False)].copy()
meta = meta[meta.ophys_experiment_id.map(eye_map)].copy()
with h5py.File(path, 'r') as h:
```

iii. The notes say direct HDF5 reads reproduce the SDK paths while avoiding expensive PyNWB construction and unnecessary objects. Passive data were considered inappropriate for active-task outcomes, and files lacking mandatory pupil data were excluded rather than fabricated.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique mouse IDs among retained experiments; every experiment receives the corresponding integer subject index.

ii.
```python
subjects=sorted({z['mouse_id'] for z in infos}); smap={x:i for i,x in enumerate(subjects)}
'subject_idx':np.asarray([smap[z['mouse_id']] for z in infos],dtype=np.int64),
```

iii. Mouse ID is the released animal identifier. The notes report 38 retained mice and exact agreement with the independently scanned local files.

## 1-c. How are the data split into sessions?

i. Each ophys experiment/NWB imaging plane is treated as one decoder session, even when several experiments share an `ophys_session_id`.

ii.
```python
for k,row in meta.iterrows():
    eid=int(row.ophys_experiment_id)
    n,x,y,info=process_experiment(row,files[eid],imap,...)
    neural.append(n); inputs.append(x); outputs.append(y); infos.append(info)
```

iii. The agent argued that each plane is a distinct simultaneously recorded neural population and that this matches paper decoding by plane; duplicated behavioral trials across sister planes were intentional.

## 1-d. How are the data split into trials?

i. The NWB `intervals/trials` table supplies trial start and stop times. Each retained trial uses `floor((stop-start)/0.1)` bins whose centers begin 50 ms after trial start, so trials remain variable length.

ii.
```python
starts = np.asarray(tr['start_time'], float); stops = np.asarray(tr['stop_time'], float)
nbin = int(np.floor((stops[ti] - starts[ti]) / BIN_S + 1e-9))
centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
```

iii. The notes justify retaining the complete native trial interval, with half-open 100 ms bins and centers guaranteed to precede stop time.

## 1-e. How are trials filtered based on quality controls?

i. Trials must be go or catch and must not be aborted or auto-rewarded. Trials with no bins or nonfinite aligned running/pupil values are dropped; an experiment must retain at least two trials. Passive experiments and experiments lacking eye tracking are removed earlier.

ii.
```python
keep = (flags['go'] | flags['catch']) & ~flags['aborted'] & ~flags['auto_rewarded']
if nbin < 1: continue
if not (np.isfinite(r).all() and np.isfinite(p).all()): continue
if len(neural) < 2: raise ValueError(f'{eid}: fewer than 2 usable trials')
```

iii. The required go/catch and aborted/auto-rewarded rules are explicit. The notes justify excluding passive data as label artifacts and dropping missing mandatory behavior instead of inventing a category.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the NWB detected calcium-event magnitudes and their timestamps, not from dF/F.

ii.
```python
ev = np.asarray(h['processing/ophys/event_detection/data'], dtype=np.float32)
ots = np.asarray(h['processing/ophys/event_detection/timestamps'], dtype=float)
```

iii. The agent says the paper's neural analyses used detected events and therefore selected this released representation rather than recomputing or using dF/F.

## 2-b. How is the `neural` data processed?

i. Time-by-cell events are averaged within half-open 100 ms trial bins and transposed into cell-by-time output. Empty bins are linearly interpolated from adjacent event frames; values are stored as float32.

ii.
```python
lo = np.searchsorted(ts, edges[:-1], side='left')
hi = np.searchsorted(ts, edges[1:], side='left')
out[:, j] = np.asarray(events[a:b], dtype=np.float32).mean(axis=0)
# empty-bin fallback
out[:, j] = (1-w)*events[k-1] + w*events[k]
```

iii. The notes say 100 ms supports both 31 Hz and 11 Hz recordings, preserves the 250 ms flashes and 400 ms paper window, and creates a common bin size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional cell/activity threshold is applied; every ROI in each retained released event array is kept. Experiments are filtered as described in 1-e.

ii.
```python
ev = np.asarray(h['processing/ophys/event_detection/data'], dtype=np.float32)
# no subsequent neuron mask
```

iii. The agent considered the released NWBs already post-QC (ROI validity, duplicates, boundaries, neuropil failures) and avoided unjustified double filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural bins are aligned to trial start on the ophys/session clock; bin zero covers `[start,start+0.1)` and has center `start+0.05`. The full start-to-stop trial is retained.

ii.
```python
centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
n = event_bin_means(ev, ots, starts[ti], centers)
'temporal_alignment_event':'trial start time on the ophys session clock'
```

iii. The notes identify trial start as the alignment event and report direct raw-versus-converted timestamp checks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses fixed 100 ms bins. Native event samples at approximately 11 or 31 Hz are averaged or, for empty bins, interpolated.

ii.
```python
BIN_S = 0.100
'time_bin_size':100.0
```

iii. A common 100 ms grid was chosen to reconcile the two acquisition rates while retaining relevant stimulus timing.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from NWB stimulus-presentation timestamps and template `control_description` values/indices.

ii.
```python
t = np.asarray(g['timestamps'], float)
idx = np.asarray(g['data']).astype(int)
desc = decode_strings(templates[gname]['control_description'][:])
onsets.extend(t[good]); names.extend(desc[idx[good]])
```

iii. The notes say exact presentation timing is preferable to inferring identity solely from trial columns.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A deterministic global vocabulary assigns gray code 0 and sorted image-name codes thereafter. A bin receives the most recent image if its center is within the 250 ms visible interval; gray gaps and omissions remain zero.

ii.
```python
return ['gray'] + sorted(n for n in names if n and n.lower() not in {'gray', 'omitted'})
image = np.zeros(nbin, dtype=np.int64)
if ft <= centers[j] < ft + 0.250:
    image[j] = image_to_idx.get(str(flash_name[b-1]), 0)
```

iii. The agent interpreted the requested identity as the image visible during non-gray periods and explicitly represented the 500 ms gray gaps.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Stimulus visibility is evaluated at the same 100 ms bin centers used for neural data.

ii.
```python
left = np.searchsorted(flash_t, centers - 0.250, side='left')
right = np.searchsorted(flash_t, centers, side='right')
```

iii. The notes report raw stimulus comparisons and expected one-third image occupancy, supporting shared-clock alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change uses trial `go` and `change_time`; catch trials remain zero.

ii.
```python
changes = np.asarray(tr['change_time'], float)
if flags['go'][ti] and np.isfinite(changes[ti]):
```

iii. The agent used explicit go/change metadata so sham catch changes are not labeled as real identity changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Exactly one 100 ms bin—the first bin at or after a go trial's change onset—is set to one.

ii.
```python
j = int(np.ceil((changes[ti] - starts[ti]) / BIN_S - 1e-12))
if 0 <= j < nbin:
    change[j] = 1
```

iii. The agent interpreted “right after” as a one-bin impulse rather than a sustained post-change label.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is intrinsically binary: 0 for no change and 1 for the single true-change bin.

ii.
```python
change = np.zeros(nbin, dtype=np.int64)
'output_values': [..., ['no_change','change'], ...]
```

iii. No numeric threshold is needed because go/change metadata directly define the two categories.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change time is converted to an index on the same trial-start 100 ms grid as neural activity.

ii.
```python
j = int(np.ceil((changes[ti] - starts[ti]) / BIN_S - 1e-12))
```

iii. The notes report direct comparisons and temporal plots showing impulses coincident with image transitions.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed uses NWB running-speed data and timestamps.

ii.
```python
run_t = np.asarray(h['processing/running/speed/timestamps'], float)
run_x = np.asarray(h['processing/running/speed/data'], float)
```

iii. This is the processed timestamped speed stream exposed by the SDK.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite samples are linearly interpolated to 100 ms centers, with nearest valid endpoints outside support. Values are then discretized using quintile edges fitted separately within each experiment across retained trials.

ii.
```python
r = interp_valid(centers, run_t, run_x)
run_lab, run_edges = quintile_labels([q[2] for q in prelim])
edges = np.quantile(allv, [0.2, 0.4, 0.6, 0.8])
```

iii. The agent says session-specific quintiles avoid trial-specific rank leakage and produce balanced categories.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four experiment-specific 20th/40th/60th/80th percentile edges produce labels 0–4 via right-sided `searchsorted`.

ii.
```python
edges = np.quantile(allv, [0.2, 0.4, 0.6, 0.8])
labels = [np.searchsorted(edges, v, side='right').astype(np.int64) for v in values]
```

iii. Five equal-percentile bins were required; the notes emphasize approximate within-session class balance.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly at the same 100 ms centers as neural bins.

ii.
```python
centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
r = interp_valid(centers, run_t, run_x)
```

iii. Common session-clock timestamps and shared centers provide alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from both axes of the NWB pupil-tracking ellipse, its timestamps, and `likely_blink`.

ii.
```python
axes = np.asarray(h['acquisition/EyeTracking/pupil_tracking/data'], float)
blink = np.asarray(h['acquisition/EyeTracking/likely_blink/data']).astype(bool)
pupil = np.sqrt(np.maximum(axes[:, 0] * axes[:, 1], 0.0))
```

iii. The agent chose equivalent circular diameter `sqrt(width*height)` to incorporate both ellipse axes and treated blinks/nonpositive values as invalid.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink, nonfinite, and nonpositive measurements are removed; valid equivalent diameter is linearly interpolated to bin centers, then discretized into experiment-specific quintiles.

ii.
```python
pupil_good = (~blink) & np.isfinite(pupil) & (pupil > 0)
p = interp_valid(centers, eye_t, pupil, pupil_good)
pup_lab, pup_edges = quintile_labels([q[3] for q in prelim])
```

iii. The notes say blink artifacts should not become a category and that interpolation plus quintiles parallels running-speed processing.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Experiment-specific 20th/40th/60th/80th percentiles produce integer labels 0–4.

ii.
```python
pup_lab, pup_edges = quintile_labels([q[3] for q in prelim])
```

iii. This implements the requested five equal-percentile bins and yields nearly equal within-experiment occupancy.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Clean pupil samples are interpolated at the identical trial-centered 100 ms timestamps used for neural data.

ii.
```python
p = interp_valid(centers, eye_t, pupil, pupil_good)
```

iii. The agent relied on hardware-synchronized session-clock timestamps and confirmed alignment in plots/raw checks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
flags = {k: np.asarray(tr[k]).astype(bool) for k in
         ['go','catch','aborted','auto_rewarded','hit','miss','false_alarm','correct_reject']}
```

iii. These SDK/NWB flags are the canonical mutually exclusive outcomes for retained active trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes map to 0=hit, 1=miss, 2=false alarm, 3=correct reject and are broadcast across all bins of a trial; a retained trial without one raises an error.

ii.
```python
if flags['hit'][ti]: outcome = 0
elif flags['miss'][ti]: outcome = 1
elif flags['false_alarm'][ti]: outcome = 2
elif flags['correct_reject'][ti]: outcome = 3
else: raise ValueError(...)
np.full(len(centers), outcome, dtype=np.int64)
```

iii. Broadcasting accommodates the homogeneous time-varying output matrix while preserving a static trial label.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing eye streams cause experiment exclusion. Invalid pupil samples are omitted before interpolation; interpolation uses nearest valid endpoints outside support. Trials with remaining nonfinite running/pupil values are dropped. Shape/range assertions fail loudly, and trials without valid outcomes or experiments with fewer than two trials raise errors.

ii.
```python
meta = meta[meta.ophys_experiment_id.map(eye_map)].copy()
if good.sum() < 2: return np.full(len(t_new), np.nan, dtype=np.float32)
if not (np.isfinite(r).all() and np.isfinite(p).all()): continue
else: raise ValueError(f'{eid} trial {ti}: retained trial lacks outcome')
```

iii. The notes prefer exclusion to fabricated pupil categories and document independent finite-value, dimension, and raw-data checks.

## 9-a. What are the most time-consuming steps of the code?

i. Reading event/stimulus/behavior datasets from 199 large NWBs and constructing all per-trial neural arrays dominate. The optional plotting and initial image-vocabulary scan add smaller costs.

ii.
```python
for k,row in meta.iterrows():
    n,x,y,info=process_experiment(row,files[eid],imap,...)
ev = np.asarray(h['processing/ophys/event_detection/data'], dtype=np.float32)
```

iii. The agent identified full SDK/PyNWB construction and large-file I/O as likely bottlenecks; the measured full conversion took 149.3 seconds.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin neural loop in `event_bin_means`, the per-bin image-visibility loop, and parts of the per-trial loop could be vectorized or grouped. File and variable-length trial loops are less naturally vectorizable.

ii.
```python
for j, (a, b) in enumerate(zip(lo, hi)):
for ti in tids:
for j, (a, b) in enumerate(zip(left, right)):
```

iii. The notes say `searchsorted` already avoids costly frame-by-bin boolean masks and that I/O dominates enough that further vectorization was unnecessary.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB is opened once during `image_vocabulary` and again during `process_experiment`; `select_experiments` also opens each file to test eye availability. Thus retained files may be opened three times and stimulus templates are reread. Per-experiment quintile processing separately concatenates/labels running and pupil.

ii.
```python
with h5py.File(files[eid], 'r') as h:  # image_vocabulary
with h5py.File(path, 'r') as h:        # process_experiment
eye_map = {eid: has_eye(files[eid]) for eid in ...}
```

iii. The agent nevertheless described core conversion as one-file streaming with no redundant full-file reads; the preliminary metadata/vocabulary passes are comparatively light.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Trial bin-center arrays are retained in `prelim` only for assembly/plots and are not saved; continuous running and pupil arrays are discarded after discretization. Optional plots also compute display-only scaled traces. The global vocabulary scan reads templates before normal processing.

ii.
```python
prelim.append((image, change, r, p, outcome, centers))
image, change, _, _, outcome, centers = q
ax[1].step(t, out[1]*out[0].max(), ...)
```

iii. Continuous values are necessary to fit quintiles and centers are useful for validation, but neither is needed by downstream decoder training. Plotting is optional diagnostic work.
