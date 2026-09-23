# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the supplied experiment metadata CSV and directly opens every locally present `behavior_ophys_experiment_*.nwb` with `h5py`. It intersects metadata IDs with file IDs, rather than using the AllenSDK cache.

ii.
```python
meta=pd.read_csv(META); fmap={int(f.stem.rsplit('_',1)[1]):f for f in NWBDIR.glob('*.nwb')}
meta=meta[meta.ophys_experiment_id.isin(fmap)].sort_values('ophys_experiment_id')
for j,row in enumerate(meta.itertuples(index=False),1):
    result,reason=process_experiment(fmap[eid],row,args.show_processing and j<=2)
```

iii. The notes say direct HDF5 reads only required datasets, avoids high-level SDK overhead, and converts every supplied usable experiment rather than metadata entries whose files are absent.

## 1-b. How are the data split into subjects?

i. Subjects are unique metadata `mouse_id` strings encountered in experiment order; each retained experiment gets the corresponding subject index.

ii.
```python
mouse=str(row.mouse_id)
if mouse not in subjects: subjects.append(mouse)
subject_idx.append(subjects.index(mouse))
```

iii. The notes identify `mouse_id` as the animal identifier and report 38 supplied mice.

## 1-c. How are the data split into sessions?

i. Each NWB ophys experiment/imaging plane is made one output session. Experiments sharing an `ophys_session_id` are not merged.

ii.
```python
for j,row in enumerate(meta.itertuples(index=False),1):
    result,reason=process_experiment(fmap[eid],row,...)
    neural.append(nn); inputs.append(ii); outputs.append(oo)
```

iii. The agent says plane clocks may be staggered, the paper decoded by imaging plane, and keeping planes separate avoids invalid concatenation.

## 1-d. How are the data split into trials?

i. Native trial-table `start_time`/`stop_time` boundaries define variable-duration trials. Each retained interval is clipped to the ophys recording and sampled on a trial portion of the experiment-wide 30 Hz grid.

ii.
```python
starts = np.asarray(tr['start_time'][:], float)
stops = np.asarray(tr['stop_time'][:], float)
a = max(starts[ti], ots[0]); b = min(stops[ti], ots[-1] + DT)
k0 = int(np.ceil((a-origin)*HZ - 1e-9)); k1 = int(np.ceil((b-origin)*HZ - 1e-9))
grid = origin + np.arange(k0, k1, dtype=np.float64)*DT
```

iii. The notes justify preserving native trial boundaries while imposing a common bin duration anchored to ophys timestamps.

## 1-e. How are trials filtered based on quality controls?

i. Only rows with `go` or `catch` true are retained; impossible non-overlap and fewer-than-two-bin trials are removed. Sessions without pupil tracking are excluded, and the code asserts at least two retained trials.

ii.
```python
elig = np.asarray(tr['go'][:], bool) | np.asarray(tr['catch'][:], bool)
trial_idx = np.flatnonzero(elig)
trial_idx = trial_idx[(stops[trial_idx] >= ots[0]) & (starts[trial_idx] <= ots[-1])]
if len(grid) < 2: continue
assert len(neural)==len(outputs)==len(inputs)>=2
```

iii. The agent concluded `go | catch` exactly excludes aborted and auto-rewarded trials and documented three pupil-missing experiment exclusions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from NWB `processing/ophys/event_detection/data` and its timestamps, with ROI validity from `cell_specimen_table/valid_roi`.

ii.
```python
evg = h['processing/ophys/event_detection']
ots = np.asarray(evg['timestamps'][:], dtype=np.float64)
events = np.asarray(evg['data'][:], dtype=np.float32)
valid = np.asarray(cells['valid_roi'][:], bool)
```

iii. The notes say the paper used detected calcium events for neural analyses, so events were preferred over dF/F.

## 2-b. How is the `neural` data processed?

i. Valid ROI columns are selected, all cells are linearly interpolated together onto the fixed 30 Hz grid, transposed to cells by time, and stored as float32. No smoothing or normalization is added.

ii.
```python
events = events[:, valid]
n = interp_rows(ots, events, grid).T.astype(np.float32, copy=False)
```

iii. The agent cites the paper's common 30 Hz interpolation and vectorizes interpolation for speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `valid_roi` cells are included. No activity/amplitude filter is applied; even all-zero trial slices are retained.

ii.
```python
valid = np.asarray(cells['valid_roi'][:], bool) if 'valid_roi' in cells else np.ones(events.shape[1], bool)
events = events[:, valid]
```

iii. The notes rely on release QC and argue that removing genuine zero-event intervals would bias the data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to native trial start: the first grid sample is at or after trial start on a 30 Hz grid anchored to the experiment's first ophys timestamp, ending before trial stop.

ii.
```python
origin = ots[0]
k0 = int(np.ceil((a-origin)*HZ - 1e-9))
grid = origin + np.arange(k0, k1, dtype=np.float64)*DT
```

iii. The agent says this is explicitly tied to the ophys clock and avoids pre-trial samples.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is exactly 1/30 s (33.333 ms). Native event traces are linearly resampled to this grid.

ii.
```python
HZ = 30.0
DT = 1.0 / HZ
'time_bin_size':1000.0/HZ
```

iii. The agent chose 30 Hz to match the paper and guarantee a common bin size despite differing native plane rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from the active stimulus-presentation table's `image_name`, `start_time`, `stop_time`, and `omitted` fields.

ii.
```python
if all(c in g for c in ('image_name','start_time','stop_time','is_change','omitted')):
    return g
simg = np.array([dec(x) for x in sg['image_name'][:]], object)
```

iii. The agent used actual presentation intervals to distinguish non-gray image periods from gray/omitted periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each bin defaults to class 0 (`gray`). During a non-omitted known-image presentation it receives one of 16 fixed global codes 1–16.

ii.
```python
image = np.zeros(len(grid), dtype=np.int16)
if not somit[pi] and simg[pi] in IMAGE_TO_CODE:
    mask=(grid >= ss[pi]) & (grid < se[pi])
    image[mask] = IMAGE_TO_CODE[simg[pi]]
```

iii. The notes interpret the requested identity as the image only during the non-gray screen and make gray an explicit class.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation intervals are masked directly against the same `grid` used to interpolate neural data.

ii.
```python
mask=(grid >= ss[pi]) & (grid < se[pi])
image[mask] = IMAGE_TO_CODE[simg[pi]]
```

iii. A shared trial grid gives bin-for-bin alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It comes from stimulus-presentation `is_change`, `omitted`, and presentation start time, constrained to the current trial.

ii.
```python
schange = np.asarray(sg['is_change'][:],float) == 1
somit = np.asarray(sg['omitted'][:],float) == 1
```

iii. The agent treats a true, non-omitted changed presentation as the change event and excludes sham catch changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created and a single bin is set at the first grid sample at or after each true change onset.

ii.
```python
change = np.zeros(len(grid), dtype=np.int16)
q=np.searchsorted(grid,ss[pi],side='left')
if q < len(grid): change[q]=1
```

iii. The notes say a one-bin pulse most literally represents “right after” and avoids labeling the whole changed-image period.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: 0 is no change and 1 is a qualifying change-onset bin; no numeric threshold is learned.

ii.
```python
'output_values':[...,['no_change','change'],...]
```

iii. Categorization follows the Boolean `is_change` field plus omitted/trial-boundary checks.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. `searchsorted` places the onset on the same 30 Hz grid used for neural interpolation.

ii.
```python
q=np.searchsorted(grid,ss[pi],side='left')
```

iii. The agent independently audited grid boundaries and reported no failures.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `processing/running/speed` timestamps and data.

ii.
```python
rg = h['processing/running/speed']
rts, rsp = np.asarray(rg['timestamps'][:],float), np.asarray(rg['data'][:],float)
```

iii. The notes identify this as the SDK-filtered running-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite samples are linearly interpolated to each trial grid; all retained trial samples in that experiment are pooled to compute quintile edges and discretized with right-sided `searchsorted`.

ii.
```python
run_cont.append(interp_1d_finite(rts,rsp,grid))
edges = np.percentile(allv, [20,40,60,80])
np.searchsorted(edges, x, side='right')
```

iii. The agent chose within-experiment quintiles to avoid rig and mouse-size differences dominating categories.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four within-experiment percentile thresholds (20, 40, 60, 80) create integer classes 0–4.

ii.
```python
edges = np.percentile(allv, [20,40,60,80])
return [np.searchsorted(edges, x, side='right').astype(np.int16) for x in values], edges
```

iii. This produces five nearly equally populated classes per experiment.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated directly to the identical per-trial 30 Hz grid.

ii.
```python
run_cont.append(interp_1d_finite(rts,rsp,grid))
```

iii. The streams have synchronized timestamps; common-grid interpolation provides alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses filtered pupil-tracking `area` and eye-tracking timestamps.

ii.
```python
ets = np.asarray(eg['eye_tracking/timestamps'][:],float)
area = np.asarray(eg['pupil_tracking/area'][:],float)
```

iii. The notes identify filtered ellipse area as the reference-processed pupil-size measurement.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area is converted to equivalent-circle diameter, finite samples are linearly interpolated to the trial grid, then within-experiment quintiles are computed.

ii.
```python
diameter = 2.0*np.sqrt(np.maximum(area,0.0)/np.pi)
pupil_cont.append(interp_1d_finite(ets,diameter,grid))
pupil_bins, pedges = quintile(pupil_cont)
```

iii. The monotonic area conversion gives diameter semantics; interpolation avoids making missing values a class.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The same 20/40/60/80 within-experiment percentile thresholds create classes 0–4.

ii.
```python
pupil_bins, pedges = quintile(pupil_cont)
```

iii. The goal is five balanced pupil-size categories per experiment.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Diameter is interpolated from eye timestamps to the same trial grid as neural data.

ii.
```python
pupil_cont.append(interp_1d_finite(ets,diameter,grid))
```

iii. The shared ophys-anchored grid supplies temporal alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from trial-table Boolean fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
out = next((oi for oi,name in enumerate(OUTCOMES) if bool(tr[name][ti])), None)
```

iii. The notes describe these as mutually exclusive canonical outcomes for eligible go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The first true outcome field maps to fixed code 0–3; absence of one raises an error. The static label is repeated across all time bins.

ii.
```python
if out is None: raise ValueError(...)
outcome=np.full(len(g), outcomes[i], dtype=np.int16)
```

iii. Repetition accommodates the validator's single 2-D output matrix while preserving per-trial semantics.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Experiments with no pupil tracking are explicitly excluded. Nonfinite continuous samples are ignored during interpolation, which requires two finite samples. Invalid ROI/event dimensions and missing outcomes raise errors; trial boundaries are clipped. A broad per-experiment caller does not suppress processing errors, but the explicit missing-pupil case returns an exclusion reason.

ii.
```python
if 'acquisition/EyeTracking/pupil_tracking/area' not in h:
    return None, 'missing pupil tracking'
ok = np.isfinite(times) & np.isfinite(values)
if ok.sum() < 2: raise ValueError(...)
```

iii. The notes reject missing-value categories or unjustified pupil imputation and document all three excluded files.

## 9-a. What are the most time-consuming steps of the code?

i. Reading large event arrays, interpolating them for all cells/trials, retaining a 13.7 GB result, and serializing the pickle dominate. The agent reports the full conversion took 282.2 seconds.

ii.
```python
events = np.asarray(evg['data'][:], dtype=np.float32)
n = interp_rows(ots, events, grid).T.astype(np.float32, copy=False)
pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes say output size is inherently large and direct HDF5 plus vectorization kept the full conversion under five minutes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial iteration, per-trial stimulus-presentation iteration, output assembly, and list-based subject/region lookup remain loops. The costly per-cell interpolation is already vectorized.

ii.
```python
for ti in trial_idx:
    for pi in range(max(0,p0), min(len(ss),p1+1)):
for i,g in enumerate(grids):
```

iii. The notes specifically credit vectorized all-cell interpolation and sorted interval searches as speedups.

## 9-c. What processing does the code repeat multiple times?

i. For planes sharing a behavioral session, the duplicated trial, running, pupil, and stimulus streams are read and processed independently. Within an experiment, each trial separately searches/interpolates streams and stimulus overlaps.

ii.
```python
for j,row in enumerate(meta.itertuples(index=False),1):
    result,reason=process_experiment(fmap[eid],row,...)
```

iii. The agent intentionally accepts cross-plane behavioral duplication because each plane is treated as an independent decoding session.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It optionally builds plot payloads and plots; otherwise little converted processing is discarded. It nevertheless loads whole-session event matrices although only eligible-trial intervals are saved, and retains per-trial grids temporarily only to assemble outputs.

ii.
```python
events = np.asarray(evg['data'][:], dtype=np.float32)
plot_payload=(grids[0],neural[0],run_cont[0],pupil_cont[0],outputs[0]) if show else None
```

iii. The notes defend direct HDF5 as selective compared with SDK loading and describe plots as optional alignment validation.
