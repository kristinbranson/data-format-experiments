# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent glob-loads every NWB in the release directory directly with `h5py`. It first scans all files for global subject, region, and image vocabularies, then streams each file again to convert it. It does not use the AllenSDK project cache/table.

ii.
```python
FILES=sorted(glob.glob(os.path.join(ROOT,'*.nwb')))
for p in FILES:
    with h5py.File(p,'r') as f:
        ...
for fi,p in enumerate(FILES):
    with h5py.File(p,'r') as f:
        ...
```

iii. The trajectory says direct HDF5 access was chosen because 284 NWBs total about 247 GB and SDK/PyNWB loading was judged slower and more memory-heavy. The agent believed the HDF5 datasets were equivalent to the released SDK products.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique `general/subject/subject_id` values found across NWBs; each retained file is mapped to that list.

ii.
```python
sid=dec(f['general/subject/subject_id'][()])
if sid not in subjects: subjects.append(sid)
subjects=sorted(subjects)
...
subject_idx.append(subjects.index(sid))
```

iii. The agent identified 38 mice and treated the NWB subject identifier as the canonical animal ID.

## 1-c. How are the data split into sessions?

i. Every behavior-ophys experiment NWB (one imaging plane/population) is treated as one output session. Experiments sharing an `ophys_session_id` are not combined.

ii.
```python
for fi,p in enumerate(FILES):
    with h5py.File(p,'r') as f:
        ...
        neural.append(ns)
        outputs.append(outs)
```

iii. The module docstring explicitly says “one behavior_ophys_experiment NWB ... is a session.” In the trajectory the agent recognized that files might share behavioral sessions, but concluded plane-level sessions were permitted because each file has its own neural population and timestamps.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials`; retained trial rows are sliced from `start_time` through `stop_time` using ophys-frame indices, producing variable-length trials.

ii.
```python
starts=tr['start_time'][:]; stops=tr['stop_time'][:]
for j in tids:
    a=np.searchsorted(ot,starts[j],side='left')
    b=np.searchsorted(ot,stops[j],side='right')
    bounds.append((j,a,b))
```

iii. The agent used the SDK-defined trial table and full trial interval so pre-change and post-change task periods and time-varying outputs are retained.

## 1-e. How are trials filtered based on quality controls?

i. It retains explicit Go or Catch rows, excludes aborted and auto-rewarded rows, skips empty trial slices, requires at least two trials, and excludes entire recordings without usable pupil tracking or stimulus presentations.

ii.
```python
keep=(tr['go'][:].astype(bool)|tr['catch'][:].astype(bool))
keep &= ~tr['aborted'][:].astype(bool)
keep &= ~tr['auto_rewarded'][:].astype(bool)
...
if len(tids)<2: continue
if b<=a: continue
```

iii. The Go/Catch and exclusion rules follow the prompt. After conversion failed on missing eye tracking, the agent decided excluding those recordings was preferable to inventing a required pupil output; three recordings were ultimately excluded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from the released dF/F trace data and its ophys timestamps.

ii.
```python
ot=find_dset(f,['processing/ophys/dff/traces/timestamps'])[:]
dff=find_dset(f,['processing/ophys/dff/traces/data'])
```

iii. The agent chose released, ROI-curated dF/F instead of recomputing fluorescence or using inferred events, because the NWBs already contain pipeline-processed traces.

## 2-b. How is the `neural` data processed?

i. Time-by-cell HDF5 slices are converted to `float32` and transposed to cell-by-time. Nonfinite values are replaced with zero. No planes are stacked because each plane is a separate output session.

ii.
```python
x=np.asarray(dff[a:b,:],dtype=np.float32).T
if not np.isfinite(x).all():
    x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
```

iii. The trajectory confirms trace orientation and says the released NWBs already contain curated ROIs. Compact `float32` storage was chosen to control the large output size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit ROI/cell filter. All stored dF/F columns are kept; only rare nonfinite samples are zero-filled.

ii.
```python
ncell=dff.shape[1]
...
x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
```

iii. The agent inspected the release and concluded every stored ROI was already valid/curated, so another `valid_roi` filter was unnecessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are indexed on the native ophys clock from the first frame at/after `start_time` through the frame at/before `stop_time` (`side='right'` for the end). There is no fixed event-centered window.

ii.
```python
a=np.searchsorted(ot,starts[j],side='left')
b=np.searchsorted(ot,stops[j],side='right')
x=np.asarray(dff[a:b,:],dtype=np.float32).T
```

iii. The agent selected native ophys timestamps as the common timebase and full SDK trial bounds as the temporal alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Plane-level recordings remain at their native approximately 31 Hz rate; metadata hard-codes `1000/31` ms.

ii.
```python
'time_bin_size':1000.0/31.0,
'temporal_alignment_event':'Native ophys timestamps; each trial spans SDK trial start_time through stop_time.'
```

iii. The agent observed an ophys sampling rate near 31 Hz and chose native frames to avoid discarding temporal information.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from stimulus-presentation onset timestamps and integer indices plus template `control_description` labels. Trial `initial_image_name`/`change_image_name` values define the global vocabulary and serve as a fallback label list.

ii.
```python
pon=series['timestamps'][:]; pind=series['data'][:].astype(int)
labels=[dec(x) for x in templ['control_description'][:]]
...
for col in ('initial_image_name','change_image_name'):
    images.update(dec(v) for v in tr[col][:] ...)
```

iii. The agent inspected the NWB stimulus representation and concluded it must represent 250 ms flashes and intervening gray periods, including omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global deterministic codebook contains `gray` plus sorted task images. The full-session vector defaults to gray, and each valid presentation paints its image code for exactly 250 ms; unknown/out-of-range labels are ignored.

ii.
```python
image_values=['gray']+images
image_code={v:i for i,v in enumerate(image_values)}
img=np.zeros(ot.size,dtype=np.int16)
for onset,ii in zip(pon,pind):
    a=np.searchsorted(ot,onset,'left')
    b=np.searchsorted(ot,onset+0.250,'left')
    img[a:b]=image_code[label]
```

iii. The trajectory explicitly reasons that images appear for 250 ms followed by 500 ms gray and that omitted presentations should remain gray instead of carrying the last image forward.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation times are converted to ophys indices, and the same `[a:b]` trial slice used for dF/F is taken from the full-session image vector.

ii.
```python
a=np.searchsorted(ot,onset,'left'); b=np.searchsorted(ot,onset+0.250,'left')
img[a:b]=image_code[label]
...
y=np.vstack((img[a:b], ...))
```

iii. The common ophys-frame clock was chosen to guarantee output/neural timepoint correspondence.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change uses trial `change_time` and the `go` flag. Catch sham changes remain zero.

ii.
```python
ct=float(tr['change_time'][j])
if np.isfinite(ct) and bool(tr['go'][j]):
    k=np.searchsorted(ot[a:b],ct,'left')
```

iii. A sanity check initially found impulses on Catch trials; the agent corrected this because the prompt asks for actual identity changes, and verified the impulse count exactly matched Go trials.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created for each trial and a single one-frame impulse is inserted at the first ophys frame at/after a finite Go-trial change time.

ii.
```python
change=np.zeros(T,dtype=np.int8)
...
if k<T: change[k]=1
```

iii. The agent interpreted “right after a change” as a transient impulse, not a state lasting through the next flash cycle.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: 0 means no change and 1 is the one-frame actual-change impulse. No numeric threshold is applied.

ii.
```python
'output_values':[image_values,['no change','change'], ...]
```

iii. The source flags and event timing directly define the two requested categories.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. `np.searchsorted` locates the first trial ophys frame at/after `change_time`; the resulting vector has exactly the dF/F trial length.

ii.
```python
k=np.searchsorted(ot[a:b],ct,'left')
if k<T: change[k]=1
```

iii. This uses the display-lag-corrected SDK trial time on the native ophys clock.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses the released processed running-speed timestamps and speed data.

ii.
```python
rt=f['processing/running/speed/timestamps'][:]
rv=f['processing/running/speed/data'][:]
```

iii. The agent chose the Allen-filtered running stream rather than recomputing wheel speed.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite speed samples are linearly interpolated to ophys timestamps. Quintile edges are then computed separately for each recording using finite samples in retained trial frames.

ii.
```python
run=interp_finite(ot,rt,rv)
runbin,redges=quintile(run,retained)
```

iii. Interpolation provides a shared neural clock. Per-recording percentile bins were chosen to meet the equal-percentile request and produce balanced local classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The recording's 20th, 40th, 60th, and 80th percentiles define codes 0–4; ties at an edge go into the higher bin.

ii.
```python
edges=np.quantile(vals,[.2,.4,.6,.8])
return np.searchsorted(edges,x,side='right').astype(np.int8), edges.tolist()
```

iii. The agent used quintiles as the literal implementation of five equal percentile bins and stored each recording's edges in metadata.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated to every ophys timestamp before trial slicing; `runbin[a:b]` uses the exact neural bounds.

ii.
```python
run=interp_finite(ot,rt,rv)
...
y=np.vstack((...,runbin[a:b],...))
```

iii. The agent used native ophys timestamps as the common clock for all time-varying streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses released filtered pupil `area` plus eye-tracking timestamps, rather than the SDK `pupil_width` column.

ii.
```python
pg=f['acquisition/EyeTracking/pupil_tracking']
pt=pg['timestamps'][:] if 'timestamps' in pg else ...
pa=pg['area'][:]
```

iii. The trajectory says canonical filtered pupil area contains blink/outlier NaNs and converts it to an equivalent circular diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Negative area is clamped to zero, diameter is calculated as `2*sqrt(area/pi)`, finite points are linearly interpolated to ophys time, and recording-specific quintiles are computed over retained trial frames.

ii.
```python
diam=interp_finite(ot,pt,2*np.sqrt(np.maximum(pa,0)/np.pi))
...
pupbin,pedges=quintile(diam,retained)
```

iii. The agent viewed linear interpolation as a way to fill filtered blink/outlier gaps without inventing a categorical value, and used the same percentile scheme as running speed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Per-recording finite diameter values in retained frames are split at the 20th, 40th, 60th, and 80th percentiles into codes 0–4.

ii.
```python
pupbin,pedges=quintile(diam,retained)
```

iii. This directly implements five percentile bins and balances categories within each recording.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Diameter is interpolated onto the ophys timestamp vector and sliced with the same `[a:b]` indices as dF/F.

ii.
```python
diam=interp_finite(ot,pt,...)
...
y=np.vstack((...,pupbin[a:b],...))
```

iii. The agent relied on synchronized timestamps and a shared ophys timebase.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome is derived from the mutually exclusive trial booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
if bool(tr['hit'][j]): outcome=0
elif bool(tr['miss'][j]): outcome=1
elif bool(tr['false_alarm'][j]): outcome=2
elif bool(tr['correct_reject'][j]): outcome=3
else: raise ValueError(...)
```

iii. The agent used the SDK trial annotations as the canonical four outcomes and treats an unclassified retained trial as an error.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes are coded 0–3 in the declared order and repeated at every timepoint of their trial.

ii.
```python
np.full(T,outcome,dtype=np.int8)
...
['hit','miss','false alarm','correct reject']
```

iii. Repetition satisfies the converter's two-dimensional, time-aligned output convention even though outcome is static per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The interpolation helper ignores nonfinite samples, returns zeros if none exist, and a constant if only one exists. Recordings are skipped for absent eye tracking, fewer than two finite pupil samples, absent stimuli, or fewer than two trials. Empty bounds are skipped, bounds are clipped, and nonfinite dF/F is zero-filled. Output is written atomically via a temporary file.

ii.
```python
if ok.sum()==0: return np.zeros(len(tnew),dtype=np.float32)
if ok.sum()==1: return np.full(len(tnew),x[ok][0],dtype=np.float32)
...
if np.isfinite(pa).sum() < 2: continue
...
with open(OUT+'.tmp','wb') as fh: pickle.dump(data,fh,...)
os.replace(OUT+'.tmp',OUT)
```

iii. The trajectory explicitly prefers excluding recordings missing a required stream over fabricating pupil values. It also reports successful validation and no conversion errors after these guards were added.

## 9-a. What are the most time-consuming steps of the code?

i. Reading and slicing the very large dF/F arrays across 284 NWBs, followed by serializing the multi-gigabyte pickle, dominate. The code also opens every NWB twice.

ii.
```python
for p in FILES:                         # vocabulary scan
    with h5py.File(p,'r') as f: ...
for fi,p in enumerate(FILES):           # conversion pass
    with h5py.File(p,'r') as f: ...
x=np.asarray(dff[a:b,:],dtype=np.float32).T
```

iii. The trajectory repeatedly notes the 247 GB source footprint and that full conversion and serialization take several minutes; direct HDF5 was selected for speed.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The stimulus-onset loop, retained-trial bounds loop, per-trial extraction/output loop, and Python membership/index lookups could be partly vectorized or replaced with dictionaries. Variable trial lengths make full vectorization less straightforward.

ii.
```python
for onset,ii in zip(pon,pind): ...
for j in tids: ...
for j,a,b in bounds: ...
subject_idx.append(subjects.index(sid))
```

iii. The trajectory focused on I/O rather than loop optimization; it did not provide a specific vectorization justification.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB is opened during the global vocabulary scan and again during conversion. Trial tables and identifiers are reread; each trial also causes a separate HDF5 dF/F slice. Subjects/regions are repeatedly searched with list membership and `.index`.

ii.
```python
for p in FILES:
    with h5py.File(p,'r') as f: ...
...
for fi,p in enumerate(FILES):
    with h5py.File(p,'r') as f: ...
```

iii. The agent accepted a cheap metadata pre-scan to create deterministic global vocabularies before streaming conversion.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes full-session running, pupil, image, and bin vectors although only retained trial slices are saved. It gathers vocabulary entries from files later excluded for missing pupil/stimuli. It also creates a full-session retained mask and stores verbose per-session metadata/percentile edges not required by the decoder.

ii.
```python
run=interp_finite(ot,rt,rv)
diam=interp_finite(ot,pt,...)
img=np.zeros(ot.size,dtype=np.int16)
retained=np.zeros(ot.size,bool)
```

iii. The agent prioritized simple shared-clock alignment and auditability over minimizing these temporary computations; no explicit trajectory discussion identifies them as waste.
