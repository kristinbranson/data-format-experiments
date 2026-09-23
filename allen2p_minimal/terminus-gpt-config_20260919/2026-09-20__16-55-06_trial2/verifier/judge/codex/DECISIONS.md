# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads data by globbing every local NWB file under `/app/data`, opening each file directly with `h5py`, and keeping only files whose `session_description` is one of four "active" session types. It does not use the Allen SDK cache or experiment table. Trials are later read from each kept file's `intervals/trials` table.

ii. 
```python
DATA_GLOB='/app/data/**/*.nwb'
ACTIVE={'OPHYS_1_images_A','OPHYS_3_images_A','OPHYS_4_images_B','OPHYS_6_images_B'}

files=[]
for f in sorted(glob.glob(DATA_GLOB,recursive=True)):
    with h5py.File(f,'r') as h:
        if text(h['session_description']) in ACTIVE: files.append(f)
...
for fi,f in enumerate(files):
  with h5py.File(f,'r') as h:
      ...
      tr=h['intervals/trials']
```

iii. In the trajectory, the agent said the AllenSDK was too slow because it eagerly loaded large arrays, so it switched to direct HDF5 access. It also justified restricting to the four active session types because it considered passive sessions outside the active change-detection task.

## 1-b. How are the data split into subjects?

i. Subjects are defined as the unique NWB field `general/subject/subject_id` across the retained files.

ii. 
```python
subjects=sorted({text(h5py.File(f,'r')['general/subject/subject_id']) for f in files})
subjmap={v:i for i,v in enumerate(subjects)}
...
mouse=text(h['general/subject/subject_id'])
...
subject_idx.append(subjmap[mouse])
```

iii. The trajectory treats each unique mouse `subject_id` as the canonical animal identifier and reports summary counts in terms of retained files and retained mice.

## 1-c. How are the data split into sessions?

i. Each retained NWB file is treated as one session. The code does not merge files that belong to the same `ophys_session_id` or same simultaneous multiplane acquisition; one `ophys_experiment_id` becomes one output session.

ii. 
```python
for fi,f in enumerate(files):
  with h5py.File(f,'r') as h:
    stype=text(h['session_description']); mouse=text(h['general/subject/subject_id'])
    expid=int(text(h['identifier']))
    ...
    if len(sn)>=2:
        neural.append(sn); inputs.append(si); outputs.append(so)
        subject_idx.append(subjmap[mouse])
        session_info.append({'ophys_experiment_id':expid,'mouse_id':mouse,'session_type':stype,
                             'n_cells':ncell,'n_trials':len(sn),'source_file':os.path.basename(f)})
```

iii. In the trajectory, the agent initially considered grouping planes, but later decided each NWB experiment should remain its own decoder session because each file had its own neuronal population and timestamps.

## 1-d. How are the data split into trials?

i. Trials come from rows of `intervals/trials`, but instead of using each trial's full `start_time` to `stop_time` interval, the agent converts every retained trial into a fixed 72-bin window from 3.0 s before `change_time` to 4.2 s after `change_time`.

ii. 
```python
DT=0.100
OFF0,OFF1=-3.0,4.2
NB=int(round((OFF1-OFF0)/DT))
...
change=np.asarray(tr['change_time'],dtype=float)
keep=np.flatnonzero((go|catch)&~abort&~auto&np.isfinite(change))
...
for j in keep:
    edges=change[j]+OFF0+np.arange(NB+1)*DT
    centers=(edges[:-1]+edges[1:])/2
```

iii. The trajectory says trial starts varied too much for a rectangular dataset, so the agent chose the "common portion" of all valid trials around the scheduled image change to keep every trial the same length.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained only if they are marked `go` or `catch`, are not `aborted`, are not `auto_rewarded`, and have finite `change_time`. Entire file-sessions with fewer than two retained trials are dropped.

ii. 
```python
go=np.asarray(tr['go']); catch=np.asarray(tr['catch'])
abort=np.asarray(tr['aborted']); auto=np.asarray(tr['auto_rewarded'])
change=np.asarray(tr['change_time'],dtype=float)
keep=np.flatnonzero((go|catch)&~abort&~auto&np.isfinite(change))
...
if len(sn)>=2:
    neural.append(sn); inputs.append(si); outputs.append(so)
```

iii. In the trajectory, the agent explicitly said it would retain Go and Catch trials, exclude aborted and auto-rewarded trials, and keep a minimum of two trials per session because the decoder requires at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the NWB dataset `processing/ophys/event_detection/data`, with timestamps from `processing/ophys/event_detection/timestamps`.

ii. 
```python
ev=h['processing/ophys/event_detection/data']
ot=np.asarray(h['processing/ophys/event_detection/timestamps'],dtype=float)
```

iii. The trajectory states that the agent chose detected calcium events rather than dF/F because the paper methods described event-based analyses, and it viewed that as the more faithful neural signal.

## 2-b. How is the `neural` data processed?

i. For each trial window, the code reads the event matrix rows whose timestamps fall inside the fixed peri-change interval and averages event magnitudes within each 100 ms bin. The result is a `(n_cells, 72)` float32 matrix per trial.

ii. 
```python
a=np.searchsorted(ot,edges[0]); b=np.searchsorted(ot,edges[-1])
slab=np.asarray(ev[a:b,:],dtype=np.float32)
tt=ot[a:b]; mat=np.zeros((ncell,NB),dtype=np.float32)
bi=np.floor((tt-edges[0])/DT).astype(int)
for k in range(NB):
    z=slab[bi==k]
    if len(z): mat[:,k]=z.mean(axis=0)
...
sn.append(mat)
```

iii. The trajectory justifies this as a tractable way to convert the full dataset while using change-aligned fixed-length trials. It explicitly says 100 ms bins were chosen to keep the dataset size manageable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit cell-level quality filter in the conversion code. All columns in `event_detection/data` are used.

ii. 
```python
ev=h['processing/ophys/event_detection/data']
ot=np.asarray(h['processing/ophys/event_detection/timestamps'],dtype=float)
ncell=ev.shape[1]
...
mat=np.zeros((ncell,NB),dtype=np.float32)
```

iii. The trajectory assumes the NWB files already contain curated cells and does not describe any additional neuron rejection step beyond using the released event-detection traces.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to the trial's scheduled `change_time` (or sham-change time for catch trials), not to trial start. The fixed window spans `[-3.0, 4.2]` seconds around that event.

ii. 
```python
OFF0,OFF1=-3.0,4.2
...
for j in keep:
    edges=change[j]+OFF0+np.arange(NB+1)*DT
    centers=(edges[:-1]+edges[1:])/2
```

iii. In the trajectory, the agent said a peri-change window preserved the common part of all trials and gave a uniform representation for decoding.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 100 ms bins. Yes: native event timestamps are rebinned by averaging all event samples within each 100 ms bin.

ii. 
```python
DT=0.100
NB=int(round((OFF1-OFF0)/DT))
...
bi=np.floor((tt-edges[0])/DT).astype(int)
for k in range(NB):
    z=slab[bi==k]
    if len(z): mat[:,k]=z.mean(axis=0)
...
'time_bin_size':DT*1000,
'neural_signal':'Allen event_detection calcium events, mean per 100-ms bin',
```

iii. The trajectory explicitly says 100 ms bins were chosen to keep the full active-task dataset tractable while preserving calcium-event dynamics.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the first dataset under `stimulus/presentation`, specifically its `timestamps` and `data` arrays, together with the session type (`images_A` vs `images_B`) to choose an offset in the output label space.

ii. 
```python
stype=text(h['session_description']); ...; setoff=1 if 'images_A' in stype else 9
...
preskeys=list(h['stimulus/presentation'].keys())
pg=h['stimulus/presentation/'+preskeys[0]]
stim_t=np.asarray(pg['timestamps'],dtype=float); stim_i=np.asarray(pg['data'])
```

iii. The trajectory says the agent preferred raw stimulus-presentation timing over trial-table image labels so it could represent gray periods and exact flash timing.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code defines a global label set `['gray', 'A_image_0'..'A_image_7', 'B_image_0'..'B_image_7']`. For each trial bin center, it finds the most recent stimulus onset and assigns a non-gray image only if the center lies within the next 250 ms and the stimulus index is `< 8`; otherwise it assigns gray.

ii. 
```python
image_values=['gray']+[f'{s}_image_{i}' for s in ('A','B') for i in range(8)]
...
pos=np.searchsorted(stim_t,centers,side='right')-1
img=np.zeros(NB,dtype=np.int16)
ok=(pos>=0)
pp=np.clip(pos,0,len(stim_t)-1)
shown=ok&(centers<stim_t[pp]+.25)&(stim_i[pp]<8)
img[shown]=(setoff+stim_i[pp[shown]].astype(int)).astype(np.int16)
```

iii. The trajectory justifies this as a way to encode the true stimulus on screen at each time bin and to reserve category 0 for the gray inter-stimulus period or omissions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated at the same 100 ms bin centers used for neural trial matrices, so it is time-locked to the same change-aligned trial grid.

ii. 
```python
centers=(edges[:-1]+edges[1:])/2
...
pos=np.searchsorted(stim_t,centers,side='right')-1
...
img[shown]=(setoff+stim_i[pp[shown]].astype(int)).astype(np.int16)
...
out=np.vstack([img,ch,runbin,pupbin,np.full(NB,outcome,dtype=np.int16)])
```

iii. The trajectory says all behavioral and stimulus outputs were aligned by absolute time onto the same ophys-clock bin centers as the neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the trial-table `change_time` column after trial filtering. Although the code reads `go` and `catch`, the final `image change` output is based only on `change_time` and is applied to every retained trial.

ii. 
```python
go=np.asarray(tr['go']); catch=np.asarray(tr['catch'])
...
change=np.asarray(tr['change_time'],dtype=float)
keep=np.flatnonzero((go|catch)&~abort&~auto&np.isfinite(change))
...
ch=np.zeros(NB,dtype=np.int16); ch[np.argmin(abs(centers-change[j]))]=1
```

iii. The trajectory says the agent wanted a pulse at the scheduled change time for both Go and Catch trials in the fixed peri-change representation.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code creates a zero vector for each trial and sets exactly one bin, the bin whose center is nearest `change_time`, to 1.

ii. 
```python
ch=np.zeros(NB,dtype=np.int16)
ch[np.argmin(abs(centers-change[j]))]=1
```

iii. The trajectory frames this as a compact event marker on the common 100 ms trial grid.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is encoded as a binary categorical variable: `0` for `no change` and `1` for `image change`.

ii. 
```python
'output_values':[image_values,['no change','image change'],
                 ['0-20%','20-40%','40-60%','60-80%','80-100%'],
                 ['0-20%','20-40%','40-60%','60-80%','80-100%'],
                 ['hit','miss','false alarm','correct reject']],
```

iii. The trajectory treats image change as a binary event channel on the aligned time grid.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned to the same 100 ms change-centered bin grid as the neural data; the bin nearest `change_time` is marked as the change bin.

ii. 
```python
edges=change[j]+OFF0+np.arange(NB+1)*DT
centers=(edges[:-1]+edges[1:])/2
...
ch=np.zeros(NB,dtype=np.int16); ch[np.argmin(abs(centers-change[j]))]=1
```

iii. The trajectory says all outputs were placed on the same peri-change time axis as the neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/timestamps` and `processing/running/speed/data` in each NWB file.

ii. 
```python
rt=np.asarray(h['processing/running/speed/timestamps'])
rv=np.asarray(h['processing/running/speed/data'])
```

iii. The trajectory describes this as the native synchronized running-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto the trial's 100 ms bin centers using `interp_clean`, then discretized with `np.digitize`.

ii. 
```python
def interp_clean(t,x,q):
    t=np.asarray(t); x=np.asarray(x,dtype=float)
    good=np.isfinite(t)&np.isfinite(x)
    if good.sum()<2: return np.zeros(len(q),dtype=np.float32)
    return np.interp(q,t[good],x[good]).astype(np.float32)
...
run=interp_clean(rt,rv,centers)
runbin=np.digitize(run,rq).astype(np.int16)
```

iii. The trajectory says all continuous streams were interpolated to the common trial grid after the agent decided on 100 ms change-aligned bins.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five within-session quintile bins computed from that file's full running-speed stream.

ii. 
```python
def quintile_reference(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if len(x)==0: return np.array([-np.inf]*4)
    return np.quantile(x,[.2,.4,.6,.8])
...
rq=quintile_reference(rv)
...
runbin=np.digitize(run,rq).astype(np.int16)
```

iii. The trajectory explicitly described this as "session-wide percentile-binned" or "within-session quintiles."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is evaluated at the same 100 ms bin centers used to form the neural trial matrix.

ii. 
```python
centers=(edges[:-1]+edges[1:])/2
...
run=interp_clean(rt,rv,centers)
...
out=np.vstack([img,ch,runbin,pupbin,np.full(NB,outcome,dtype=np.int16)])
```

iii. The trajectory says alignment was done by interpolating each stream onto the common ophys-clock bin centers.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width`, using timestamps from `acquisition/EyeTracking/eye_tracking/timestamps`.

ii. 
```python
pbase='acquisition/EyeTracking/pupil_tracking'
if pbase in h:
    pt=np.asarray(h['acquisition/EyeTracking/eye_tracking/timestamps'])
    pv=np.asarray(h[pbase+'/width'])
else: pt=np.array([]); pv=np.array([])
```

iii. The trajectory identifies `pupil_tracking/width` as the pupil-size signal after it searched the NWB paths directly.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil width is linearly interpolated to the 100 ms trial-bin centers using `interp_clean`, then discretized with `np.digitize`. No blink filtering is applied.

ii. 
```python
pup=interp_clean(pt,pv,centers)
pupbin=np.digitize(pup,pq).astype(np.int16)
```

iii. The trajectory says the agent directly used the raw pupil-width path discovered in the NWB file and aligned it to the common change-centered bins.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five within-session quintile bins computed from all finite values in the file's `pupil_tracking/width` array.

ii. 
```python
pq=quintile_reference(pv)
...
pupbin=np.digitize(pup,pq).astype(np.int16)
```

iii. The trajectory grouped pupil processing with running-speed processing and described both as within-session quintile binning on the common trial grid.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation to the same 100 ms bin centers used for neural data.

ii. 
```python
centers=(edges[:-1]+edges[1:])/2
...
pup=interp_clean(pt,pv,centers)
...
out=np.vstack([img,ch,runbin,pupbin,np.full(NB,outcome,dtype=np.int16)])
```

iii. The trajectory says all streams share the same aligned trial grid after interpolation.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. 
```python
hit=np.asarray(tr['hit']); miss=np.asarray(tr['miss'])
fa=np.asarray(tr['false_alarm']); cr=np.asarray(tr['correct_reject'])
...
outcome=0 if hit[j] else 1 if miss[j] else 2 if fa[j] else 3
```

iii. The trajectory explicitly says trial outcome would be encoded from the standard Go/Catch outcome columns.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps each retained trial to one integer category (`hit`=0, `miss`=1, `false_alarm`=2, `correct_reject`=3) and repeats that label across every time bin of the trial.

ii. 
```python
outcome=0 if hit[j] else 1 if miss[j] else 2 if fa[j] else 3
out=np.vstack([img,ch,runbin,pupbin,np.full(NB,outcome,dtype=np.int16)])
```

iii. The trajectory says a static per-trial outcome was expanded across time so every output trial remained a rectangular `(n_outputs, n_timepoints)` matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or non-finite behavioral samples are ignored during interpolation by `interp_clean`; if fewer than two finite samples exist, the code substitutes all-zero trajectories. Missing pupil tracking also falls back to empty arrays, which then become zero-valued interpolants. Sessions with fewer than two kept trials are skipped. There is no try/except around file processing, no blink removal, and no special clipping logic for truncated trial windows because every trial uses the same fixed peri-change window.

ii. 
```python
def interp_clean(t,x,q):
    t=np.asarray(t); x=np.asarray(x,dtype=float)
    good=np.isfinite(t)&np.isfinite(x)
    if good.sum()<2: return np.zeros(len(q),dtype=np.float32)
    return np.interp(q,t[good],x[good]).astype(np.float32)
...
if pbase in h:
    pt=np.asarray(h['acquisition/EyeTracking/eye_tracking/timestamps'])
    pv=np.asarray(h[pbase+'/width'])
else: pt=np.array([]); pv=np.array([])
...
if len(sn)>=2:
    neural.append(sn); inputs.append(si); outputs.append(so)
```

iii. The trajectory emphasizes robustness through direct HDF5 access and simple interpolation, but it does not mention any more conservative missing-data handling beyond zero-fill fallbacks and dropping sessions with too few trials.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are opening every retained NWB file and, inside each file, iterating over every retained trial to read the event slab and average it over 72 bins.

ii. 
```python
for fi,f in enumerate(files):
  with h5py.File(f,'r') as h:
      ...
      for j in keep:
          a=np.searchsorted(ot,edges[0]); b=np.searchsorted(ot,edges[-1])
          slab=np.asarray(ev[a:b,:],dtype=np.float32)
          ...
          for k in range(NB):
              z=slab[bi==k]
              if len(z): mat[:,k]=z.mean(axis=0)
```

iii. In the trajectory, the agent repeatedly called out data loading as the bottleneck, first with the SDK and then with the direct file-by-file conversion over 202 retained NWBs.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization targets are the per-trial loop over `keep` and especially the inner `for k in range(NB)` loop that slices `slab[bi==k]` one bin at a time. The repeated file scans for filtering and subject discovery could also be collapsed into one pass.

ii. 
```python
for f in sorted(glob.glob(DATA_GLOB,recursive=True)):
    with h5py.File(f,'r') as h:
        if text(h['session_description']) in ACTIVE: files.append(f)
...
subjects=sorted({text(h5py.File(f,'r')['general/subject/subject_id']) for f in files})
...
for j in keep:
    ...
    for k in range(NB):
        z=slab[bi==k]
        if len(z): mat[:,k]=z.mean(axis=0)
```

iii. The trajectory focuses on getting the direct-HDF5 pipeline working quickly rather than on vectorizing the bin-aggregation code.

## 9-c. What processing does the code repeat multiple times?

i. The code scans the file list multiple times: once to collect active files, again to build the subject set, and again to do the full conversion. Inside conversion it repeats the same searchsorted/bin-allocation procedure independently for every trial.

ii. 
```python
files=[]
for f in sorted(glob.glob(DATA_GLOB,recursive=True)):
    with h5py.File(f,'r') as h:
        if text(h['session_description']) in ACTIVE: files.append(f)
...
subjects=sorted({text(h5py.File(f,'r')['general/subject/subject_id']) for f in files})
...
for j in keep:
    edges=change[j]+OFF0+np.arange(NB+1)*DT
    centers=(edges[:-1]+edges[1:])/2
    ...
    pos=np.searchsorted(stim_t,centers,side='right')-1
```

iii. The trajectory reflects this tradeoff: the agent optimized away the SDK bottleneck first and accepted repeated simpler passes over the local files.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores an empty input array for every trial even though there are no decoder inputs, and it repeats the static trial-outcome label across all 72 time bins instead of storing it once per trial. It also does an extra subject-discovery pass over all active files before the real conversion pass.

ii. 
```python
subjects=sorted({text(h5py.File(f,'r')['general/subject/subject_id']) for f in files})
...
sn.append(mat); si.append(np.empty((0,NB),dtype=np.float32)); so.append(out)
...
out=np.vstack([img,ch,runbin,pupbin,np.full(NB,outcome,dtype=np.int16)])
```

iii. The trajectory mainly justified these choices as compatibility decisions for the decoder's expected rectangular array format rather than as analytically necessary processing.
