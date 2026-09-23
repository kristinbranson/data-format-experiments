# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the dataset directly from local NWB/HDF5 files plus the experiment metadata CSV, rather than through the AllenSDK project cache. It hard-codes a dataset root, builds a map from `ophys_experiment_id` to NWB path, iterates over every experiment row present in the metadata, and loads each experiment by opening the NWB with `h5py`.

ii.
```python
ROOT = Path('/app/data/visual-behavior-ophys-1.1.0')
NWBDIR = ROOT / 'behavior_ophys_experiments'
META = ROOT / 'project_metadata' / 'ophys_experiment_table.csv'
...
meta=pd.read_csv(META); fmap={int(f.stem.rsplit('_',1)[1]):f for f in NWBDIR.glob('*.nwb')}
meta=meta[meta.ophys_experiment_id.isin(fmap)].sort_values('ophys_experiment_id')
...
with h5py.File(path, 'r') as h:
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 6, the AI justified this as a speed and simplicity decision: use direct HDF5 access, avoid AllenSDK overhead, and convert every supplied usable experiment file rather than reconstructing the project through the SDK cache.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by the `mouse_id` column from the experiment metadata CSV. The script keeps a list of unique mouse IDs in first-seen order and stores one `subject_idx` per converted session.

ii.
```python
mouse=str(row.mouse_id)
if mouse not in subjects: subjects.append(mouse)
subject_idx.append(subjects.index(mouse))
```

iii. Step 5 of `CONVERSION_NOTES.md` states that `mouse_id` is the subject identifier used for `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` NWB file, i.e. each imaging plane, as one session. It does not group multiple experiments belonging to the same `ophys_session_id`; instead each experiment becomes its own entry in `data['neural']`, `data['input']`, and `data['output']`.

ii.
```python
for j,row in enumerate(meta.itertuples(index=False),1):
    eid=int(row.ophys_experiment_id)
    result,reason=process_experiment(fmap[eid],row,args.show_processing and j<=2)
...
neural.append(nn); inputs.append(ii); outputs.append(oo); infos.append(info)
...
'session_unit':'one ophys experiment/imaging plane'
```

iii. The AI’s justification appears in Step 4 and Step 5 of `CONVERSION_NOTES.md`: multiscope planes can have staggered timestamps, so direct concatenation would be awkward, and the paper’s analysis unit was described as an imaging plane. On that basis it chose plane-level sessions.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table. For each eligible trial, the script uses that trial’s `start_time` and `stop_time` to form a per-trial time grid, then extracts neural and output variables on that grid. Trial duration is variable.

ii.
```python
tr = h['intervals/trials']
elig = np.asarray(tr['go'][:], bool) | np.asarray(tr['catch'][:], bool)
trial_idx = np.flatnonzero(elig)
starts = np.asarray(tr['start_time'][:], float)
stops = np.asarray(tr['stop_time'][:], float)
...
for ti in trial_idx:
    a = max(starts[ti], ots[0]); b = min(stops[ti], ots[-1] + DT)
    k0 = int(np.ceil((a-origin)*HZ - 1e-9)); k1 = int(np.ceil((b-origin)*HZ - 1e-9))
    grid = origin + np.arange(k0, k1, dtype=np.float64)*DT
```

iii. Step 5 says the AI wanted to preserve native trial boundaries while still forcing all trials onto a fixed 30 Hz bin size. The notes also emphasize that go and catch trials are the eligible categories.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by keeping only rows where `go` or `catch` is true. Then the script drops trials that do not overlap the ophys recording and trials whose derived 30 Hz grid has fewer than 2 bins. At the experiment level it requires at least 2 retained trials.

ii.
```python
elig = np.asarray(tr['go'][:], bool) | np.asarray(tr['catch'][:], bool)
trial_idx = np.flatnonzero(elig)
...
trial_idx = trial_idx[(stops[trial_idx] >= ots[0]) & (starts[trial_idx] <= ots[-1])]
...
if len(grid) < 2: continue
...
assert len(neural)==len(outputs)==len(inputs)>=2
```

iii. In Step 4 and Step 5, the AI states that trial categories are mutually exclusive in this dataset, so `go | catch` already excludes aborted and auto-rewarded rows. It also justified overlap checks and the minimum-trial requirement as safeguards against impossible or degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural activity from the event-detection output in the NWB: `processing/ophys/event_detection/data`, using the paired event timestamps from `processing/ophys/event_detection/timestamps`.

ii.
```python
evg = h['processing/ophys/event_detection']
ots = np.asarray(evg['timestamps'][:], dtype=np.float64)
events = np.asarray(evg['data'][:], dtype=np.float32)
```

iii. Step 3, Step 4, and Step 5 of `CONVERSION_NOTES.md` explicitly justify this choice by saying the paper’s analyses used detected calcium events rather than dF/F traces, so event magnitudes are the reference-consistent neural representation.

## 2-b. How is the `neural` data processed?

i. The AI filters the event matrix to `valid_roi` cells, then linearly interpolates the event trace from native event timestamps onto a fixed 30 Hz trial grid. Each trial stores a `cells x time` float32 array.

ii.
```python
cells = h['processing/ophys/image_segmentation/cell_specimen_table']
valid = np.asarray(cells['valid_roi'][:], bool) if 'valid_roi' in cells else np.ones(events.shape[1], bool)
events = events[:, valid]
...
n = interp_rows(ots, events, grid).T.astype(np.float32, copy=False)
```

iii. The notes justify this in two parts: `valid_roi` is treated as release QC for cells, and a common 30 Hz time series was chosen because the paper discussed interpolation to 30 Hz and the AI wanted a fixed bin size across variable-duration trials and planes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data are filtered by the `valid_roi` flag from the cell specimen table. No additional amplitude thresholding, de-noising, or session-specific neural QC is applied.

ii.
```python
valid = np.asarray(cells['valid_roi'][:], bool) if 'valid_roi' in cells else np.ones(events.shape[1], bool)
...
events = events[:, valid]
```

iii. Step 4 and Step 5 say the AI chose to trust release QC and use `valid_roi`, but not add any extra neural filtering because the files had already passed release checks and the paper did not motivate another amplitude-based threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start. For each trial, the grid starts at the first 30 Hz sample at or after `start_time` and ends before `stop_time`, with the grid anchored to the experiment’s first ophys timestamp.

ii.
```python
origin = ots[0]
...
a = max(starts[ti], ots[0]); b = min(stops[ti], ots[-1] + DT)
k0 = int(np.ceil((a-origin)*HZ - 1e-9)); k1 = int(np.ceil((b-origin)*HZ - 1e-9))
grid = origin + np.arange(k0, k1, dtype=np.float64)*DT
...
n = interp_rows(ots, events, grid).T.astype(np.float32, copy=False)
```

iii. Step 5 says the grid was designed to be “ophys-aligned” while maintaining equal bin size. The AI explicitly documented trial start, not image change, as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 30 Hz bins, i.e. `DT = 1/30 s` or `33.333... ms`. Yes: the neural data are rebinned by linear interpolation from native timestamps onto this new fixed grid.

ii.
```python
HZ = 30.0
DT = 1.0 / HZ
...
n = interp_rows(ots, events, grid).T.astype(np.float32, copy=False)
...
'time_bin_size':1000.0/HZ
```

iii. The AI repeatedly justified this in `CONVERSION_NOTES.md` Step 3 through Step 6 as matching a “common 30 Hz timeseries” discussed in the paper and as the easiest way to enforce one fixed bin size across all converted trials and sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the natural-image stimulus presentation table, specifically its `image_name`, `start_time`, `stop_time`, and `omitted` fields, not from the trials table.

ii.
```python
def stimulus_table(h):
    for name, g in h['intervals'].items():
        if all(c in g for c in ('image_name','start_time','stop_time','is_change','omitted')):
            return g
...
sg = stimulus_table(h)
ss = np.asarray(sg['start_time'][:],float); se = np.asarray(sg['stop_time'][:],float)
simg = np.array([dec(x) for x in sg['image_name'][:]], object)
somit = np.asarray(sg['omitted'][:],float) == 1
```

iii. In Step 4 and Step 5, the AI says stimulus presentations are the correct source because they define the actual on-screen image intervals, including omitted/gray periods, whereas trial-level image columns are coarser.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI initializes each trial as gray (`0`), then writes an image code only during overlapping non-omitted stimulus presentation intervals. It uses a fixed global mapping of 16 image names to classes `1..16`, with `0` reserved for gray.

ii.
```python
IMAGE_NAMES = ['im000','im031','im035','im045','im054','im061','im062','im063',
               'im065','im066','im069','im073','im075','im077','im085','im106']
IMAGE_TO_CODE = {x:i+1 for i,x in enumerate(IMAGE_NAMES)}
...
image = np.zeros(len(grid), dtype=np.int16)
...
if not somit[pi] and simg[pi] in IMAGE_TO_CODE:
    mask=(grid >= ss[pi]) & (grid < se[pi])
    image[mask] = IMAGE_TO_CODE[simg[pi]]
```

iii. The notes justify this as following the decoder instruction literally: predict the image shown during the non-gray screen. Gray bins and omitted presentations should therefore not be forced into one of the natural-image categories.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the exact same per-trial 30 Hz grid as the neural data. For each stimulus presentation overlapping the trial, the code marks those bins whose timestamps fall between that presentation’s `start_time` and `stop_time`.

ii.
```python
for pi in range(max(0,p0), min(len(ss),p1+1)):
    if not somit[pi] and simg[pi] in IMAGE_TO_CODE:
        mask=(grid >= ss[pi]) & (grid < se[pi])
        image[mask] = IMAGE_TO_CODE[simg[pi]]
...
outputs.append(np.vstack((image_rows[i],change_rows[i],run_bins[i],pupil_bins[i],outcome)))
```

iii. Step 5 states that all continuous and interval outputs were placed on the same ophys-anchored 30 Hz trial grid, so alignment is handled by shared timestamps.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentation table’s `is_change` field together with presentation `start_time`, with omitted presentations suppressed.

ii.
```python
schange = np.asarray(sg['is_change'][:],float) == 1
somit = np.asarray(sg['omitted'][:],float) == 1
...
if schange[pi] and not somit[pi] and starts[ti] <= ss[pi] < stops[ti]:
```

iii. The notes justify this by saying true stimulus changes should be read from the actual presentation table, not inferred indirectly from trial metadata.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI represents image change as a one-bin pulse. When a change presentation begins inside a trial, it finds the first trial-grid bin at or after that presentation’s start and sets only that bin to `1`.

ii.
```python
change = np.zeros(len(grid), dtype=np.int16)
...
if schange[pi] and not somit[pi] and starts[ti] <= ss[pi] < stops[ti]:
    q=np.searchsorted(grid,ss[pi],side='left')
    if q < len(grid): change[q]=1
```

iii. Step 5 explicitly says the AI chose a one-bin event because the instructions said image change should be `1` “right after” a change, and it wanted to mark the onset rather than the whole post-change epoch.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary: `0` means no change bin, `1` means the change-onset bin.

ii.
```python
change = np.zeros(len(grid), dtype=np.int16)
...
if q < len(grid): change[q]=1
...
'output_values':[['gray']+IMAGE_NAMES,['no_change','change'],
```

iii. The justification in the notes is simply that the decoder specification called for a binary time-varying change variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image-change pulses are placed on the same 30 Hz trial grid as the neural data, using `np.searchsorted` to map each presentation start time to the first corresponding grid bin.

ii.
```python
q=np.searchsorted(grid,ss[pi],side='left')
if q < len(grid): change[q]=1
...
n = interp_rows(ots, events, grid).T.astype(np.float32, copy=False)
```

iii. Step 5 says shared trial-grid timestamps are the alignment mechanism for interval-based outputs and neural traces.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the NWB running-speed stream: `processing/running/speed/data` and `processing/running/speed/timestamps`.

ii.
```python
rg = h['processing/running/speed']
rts, rsp = np.asarray(rg['timestamps'][:],float), np.asarray(rg['data'][:],float)
```

iii. In Step 1 and Step 5, the AI treated the SDK-filtered running-speed stream as the intended source and then read the equivalent dataset directly from the NWB for performance.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The script linearly interpolates running speed onto each trial’s 30 Hz grid, stores the continuous value per bin, and after all trials in that experiment computes within-experiment quintile edges from the concatenated running values.

ii.
```python
run_cont.append(interp_1d_finite(rts,rsp,grid))
...
def quintile(values):
    allv = np.concatenate(values)
    edges = np.percentile(allv, [20,40,60,80])
    return [np.searchsorted(edges, x, side='right').astype(np.int16) for x in values], edges
...
run_bins, redges = quintile(run_cont)
```

iii. Step 5 says interpolation is needed because running timestamps differ from ophys timestamps, and session-wise or experiment-wise quintiles were chosen to balance classes within each converted session.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into 5 classes using the 20th, 40th, 60th, and 80th percentiles computed from all retained running samples in that experiment.

ii.
```python
edges = np.percentile(allv, [20,40,60,80])
return [np.searchsorted(edges, x, side='right').astype(np.int16) for x in values], edges
...
'output_values':[['gray']+IMAGE_NAMES,['no_change','change'],
                 ['Q1_slowest','Q2','Q3','Q4','Q5_fastest'],
```

iii. The notes explicitly justify per-experiment quintiles as a way to avoid cross-session calibration differences dominating the labels.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation onto the same per-trial 30 Hz grid used for neural activity. The aligned series for each trial has the same number of time bins as the neural matrix.

ii.
```python
run_cont.append(interp_1d_finite(rts,rsp,grid))
...
outputs.append(np.vstack((image_rows[i],change_rows[i],run_bins[i],pupil_bins[i],outcome)))
...
assert n.shape[1]==o.shape[1]==inp.shape[1]
```

iii. Step 5 says the AI intentionally aligned all streams to the common ophys-anchored grid instead of retaining separate native clocks.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the eye-tracking acquisition stream, specifically `acquisition/EyeTracking/pupil_tracking/area` and `acquisition/EyeTracking/eye_tracking/timestamps`.

ii.
```python
eg = h['acquisition/EyeTracking']
ets = np.asarray(eg['eye_tracking/timestamps'][:],float)
area = np.asarray(eg['pupil_tracking/area'][:],float)
```

iii. The notes justify using pupil area because it is the pupil-size measurement available in the NWB, and they interpret equivalent-circle diameter as a more semantically direct “diameter” output for the decoder.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI converts area to equivalent-circle diameter using `2*sqrt(area/pi)`, interpolates it to the trial grid using only finite samples, and then discretizes those values into within-experiment quintiles.

ii.
```python
diameter = 2.0*np.sqrt(np.maximum(area,0.0)/np.pi)
...
pupil_cont.append(interp_1d_finite(ets,diameter,grid))
...
pupil_bins, pedges = quintile(pupil_cont)
```

iii. Step 3 through Step 5 say this was motivated by the paper’s pupil-size processing and by the need to avoid turning missing or nonfinite pupil values into a biological category. The AI also excluded experiments with no pupil stream at all.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Like running speed, pupil diameter is binned into 5 classes using the per-experiment 20th, 40th, 60th, and 80th percentiles over all retained trial bins.

ii.
```python
def quintile(values):
    allv = np.concatenate(values)
    edges = np.percentile(allv, [20,40,60,80])
    return [np.searchsorted(edges, x, side='right').astype(np.int16) for x in values], edges
...
pupil_bins, pedges = quintile(pupil_cont)
```

iii. Step 5 explicitly says the AI used experiment-wise quintiles for the same reason as running speed: balanced categories within each converted session.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation to the same per-trial 30 Hz grid used for the neural traces.

ii.
```python
pupil_cont.append(interp_1d_finite(ets,diameter,grid))
...
n = interp_rows(ots, events, grid).T.astype(np.float32, copy=False)
...
assert n.shape[1]==o.shape[1]==inp.shape[1]
```

iii. Step 5 says all streams were projected onto the shared grid so that every output row and the neural trial matrix have equal time length.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject` in `intervals/trials`.

ii.
```python
OUTCOMES = ['hit','miss','false_alarm','correct_reject']
...
out = next((oi for oi,name in enumerate(OUTCOMES) if bool(tr[name][ti])), None)
if out is None: raise ValueError(f'no outcome for eligible trial {ti} in {path.name}')
```

iii. The notes cite these as the canonical non-aborted outcome fields for go and catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is encoded as integers `0..3` using the fixed order in `OUTCOMES`, then repeated across all time bins in the trial so it can live in the same 2-D output matrix as the time-varying outputs.

ii.
```python
out = next((oi for oi,name in enumerate(OUTCOMES) if bool(tr[name][ti])), None)
...
outcome=np.full(len(g), outcomes[i], dtype=np.int16)
outputs.append(np.vstack((image_rows[i],change_rows[i],run_bins[i],pupil_bins[i],outcome)))
```

iii. Step 5 says the validator/trainer expected a uniform 2-D per-trial output matrix when other outputs are time-varying, so the static outcome label was repeated over time.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several data problems by exclusion or clipping: experiments missing pupil tracking are dropped entirely; trials are clipped to the overlapping ophys time range; trials with fewer than 2 bins are skipped; interpolation fails if fewer than two finite samples exist; and ROI/event count mismatches raise an error.

ii.
```python
if 'acquisition/EyeTracking/pupil_tracking/area' not in h:
    return None, 'missing pupil tracking'
...
if len(valid) != events.shape[1]:
    raise ValueError(f'ROI/event mismatch in {path.name}')
...
trial_idx = trial_idx[(stops[trial_idx] >= ots[0]) & (starts[trial_idx] <= ots[-1])]
...
if len(grid) < 2: continue
...
if ok.sum() < 2:
    raise ValueError('fewer than two finite samples for interpolation')
```

iii. In Step 4 through Step 6, the AI justifies this as conservative handling: do not fabricate missing required outputs, do not silently accept inconsistent neural dimensions, and only retain trials with a minimally valid aligned time series.

## 9-a. What are the most time-consuming steps of the code?

i. The code’s expensive steps are opening each large NWB file, reading the full event matrix, and interpolating neural data trial-by-trial onto the 30 Hz grid. The AI explicitly viewed high-level SDK loading as avoidable overhead and replaced it with direct HDF5 reads.

ii.
```python
with h5py.File(path, 'r') as h:
...
events = np.asarray(evg['data'][:], dtype=np.float32)
...
for ti in trial_idx:
    ...
    n = interp_rows(ots, events, grid).T.astype(np.float32, copy=False)
```

iii. Step 6 says direct HDF5 loading and vectorized interpolation were used as speedups because the dataset is very large and the AI expected SDK object construction to be unnecessarily slow.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining non-vectorized loops are the per-trial loop over `trial_idx` and the inner per-presentation loop that assigns image identities and change pulses. Neural interpolation itself was already vectorized across cells.

ii.
```python
for ti in trial_idx:
    ...
    for pi in range(max(0,p0), min(len(ss),p1+1)):
        if not somit[pi] and simg[pi] in IMAGE_TO_CODE:
            mask=(grid >= ss[pi]) & (grid < se[pi])
            image[mask] = IMAGE_TO_CODE[simg[pi]]
```

iii. The notes frame this as a pragmatic compromise: vectorize the large neuron-by-time interpolation work, but keep trial/presentation logic explicit because it is harder to batch and was not identified as the dominant cost.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly recomputes per-trial 30 Hz grids, `searchsorted` stimulus-overlap bounds, and boolean masks for stimulus presentations. It also computes running and pupil quintiles separately for each experiment rather than once globally.

ii.
```python
for ti in trial_idx:
    ...
    k0 = int(np.ceil((a-origin)*HZ - 1e-9)); k1 = int(np.ceil((b-origin)*HZ - 1e-9))
    grid = origin + np.arange(k0, k1, dtype=np.float64)*DT
    ...
    p0 = np.searchsorted(se, grid[0], side='right')
    p1 = np.searchsorted(ss, grid[-1], side='right')
...
run_bins, redges = quintile(run_cont); pupil_bins, pedges = quintile(pupil_cont)
```

iii. The trajectory and notes suggest these repetitions were accepted because they preserve a simple per-experiment conversion flow and because the AI wanted experiment-specific behavioral bin edges.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code keeps continuous running and pupil arrays only to convert them immediately into quintile labels, and optionally packages plotting payloads solely for diagnostic figures. It also stores detailed `session_info` metadata that downstream decoding does not use.

ii.
```python
run_cont.append(interp_1d_finite(rts,rsp,grid))
pupil_cont.append(interp_1d_finite(ets,diameter,grid))
...
plot_payload=(grids[0],neural[0],run_cont[0],pupil_cont[0],outputs[0]) if show else None
...
info=dict(... running_quintile_edges=redges.tolist(), pupil_diameter_quintile_edges=pedges.tolist(), ...)
```

iii. Step 6 and Step 7 show these were intentional diagnostics and metadata conveniences rather than required decoder inputs. The AI kept them for validation and documentation, not because the trained decoder needs them.
