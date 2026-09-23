# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache or experiment table. It directly scans the local Visual Behavior NWB release under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, opens each `.nwb` file with `h5py`, and does an initial pass over all files to collect subject IDs, brain regions, and image labels before a second pass that performs conversion.

ii.
```python
ROOT='/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments'
FILES=sorted(glob.glob(os.path.join(ROOT,'*.nwb')))

subjects=[]; regions=[]; images=set()
for p in FILES:
    with h5py.File(p,'r') as f:
        sid=dec(f['general/subject/subject_id'][()])
        ...

for fi,p in enumerate(FILES):
    with h5py.File(p,'r') as f:
        tr=f['intervals/trials']
        ...
```

iii. In the trajectory, the AI said the dataset consisted of “many large per-experiment NWB files” totaling 247 GB and concluded that conversion should “stream one file at a time” and use direct HDF5 reads because loading all 284 files through the SDK would be “unnecessarily slow and memory-heavy” (steps 3, 4, 13, 15).

## 1-b. How are the data split into subjects?

i. Subjects are split by the NWB field `general/subject/subject_id`. The code builds a unique sorted `subjects` list from that field, then uses `subjects.index(sid)` for each retained session.

ii.
```python
subjects=[]; regions=[]; images=set()
for p in FILES:
    with h5py.File(p,'r') as f:
        sid=dec(f['general/subject/subject_id'][()])
        if sid not in subjects: subjects.append(sid)
...
subjects=sorted(subjects)
...
sid=dec(f['general/subject/subject_id'][()])
subject_idx.append(subjects.index(sid))
```

iii. The trajectory says the released NWBs already contain the needed metadata, so the AI chose to pre-scan them directly rather than reconstruct subject lists from SDK metadata (steps 13 and 15).

## 1-c. How are the data split into sessions?

i. Each individual `behavior_ophys_experiment_*.nwb` file is treated as one session. The AI does not group multiple experiments by `ophys_session_id`; one imaging plane / one NWB becomes one decoder session.

ii.
```python
"""Convert Allen Visual Behavior Ophys NWBs to decoder format.

Decisions follow the released AllenSDK/NWB products:
* one behavior_ophys_experiment NWB (one imaging plane/population) is a session;
...
"""
...
for fi,p in enumerate(FILES):
    with h5py.File(p,'r') as f:
        ...
        neural.append(ns); inputs.append(ins); outputs.append(outs)
```

iii. The AI explicitly justified this in the trajectory: “Treating each NWB experiment as one decoder session is likely appropriate because each has its own neuron population and timestamps” (step 8). Earlier it noted that the target format “permits each recording experiment as a session” (step 6).

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. For each retained trial, the code uses `start_time` and `stop_time`, finds the corresponding ophys frame indices with `np.searchsorted`, and slices that variable-length ophys window as one trial.

ii.
```python
tr=f['intervals/trials']
...
starts=tr['start_time'][:]; stops=tr['stop_time'][:]
for j in tids:
    a=np.searchsorted(ot,starts[j],side='left')
    b=np.searchsorted(ot,stops[j],side='right')
    a=max(0,a); b=min(ot.size,b)
    if b>a: retained[a:b]=True
    bounds.append((j,a,b))
...
for j,a,b in bounds:
    if b<=a: continue
    x=np.asarray(dff[a:b,:],dtype=np.float32).T
```

iii. In the trajectory the AI noted that the trials table already contains “variable-duration trials” and later summarized the plan as “slice each trial from SDK `start_time` through `stop_time` on ophys timestamps” (steps 5, 12, 15).

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials where `go` or `catch` is true, excludes `aborted` and `auto_rewarded`, requires at least two retained trials per NWB, and skips empty trial windows. At the session level it also skips files lacking pupil tracking, files lacking stimulus presentations, and files with fewer than two finite pupil samples. It does not explicitly filter trials by non-null `change_time`.

ii.
```python
if 'acquisition/EyeTracking/pupil_tracking' not in f:
    print(f'{fi+1}/{len(FILES)} skip: no pupil tracking {os.path.basename(p)}', flush=True)
    continue
if 'stimulus/presentation' not in f or len(f['stimulus/presentation']) == 0:
    print(f'{fi+1}/{len(FILES)} skip: no stimulus presentations {os.path.basename(p)}', flush=True)
    continue
keep=(tr['go'][:].astype(bool)|tr['catch'][:].astype(bool))
keep &= ~tr['aborted'][:].astype(bool)
keep &= ~tr['auto_rewarded'][:].astype(bool)
tids=np.flatnonzero(keep)
if len(tids)<2:
    print('skip <2 trials',p,flush=True); continue
...
if np.isfinite(pa).sum() < 2:
    print(f'{fi+1}/{len(FILES)} skip: insufficient pupil samples {os.path.basename(p)}', flush=True)
    continue
...
for j,a,b in bounds:
    if b<=a: continue
```

iii. The AI’s stated reasoning was that the task “should retain go/catch trials while explicitly removing auto-rewarded ones” (step 9), and later that sessions without pupil tracking should be excluded because “pupil diameter is a required output” and “fabricating values would be inappropriate” (step 19). The final metadata also documents “sessions lacking required pupil tracking excluded” (step 19 and final script).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the released dF/F traces in `processing/ophys/dff/traces/data`, using the corresponding ophys timestamps in `processing/ophys/dff/traces/timestamps`.

ii.
```python
ot=find_dset(f,['processing/ophys/dff/traces/timestamps'])[:]
dff=find_dset(f,['processing/ophys/dff/traces/data'])
...
x=np.asarray(dff[a:b,:],dtype=np.float32).T
```

iii. The AI repeatedly described the source as “released, ROI-curated dF/F” and said it would use “dF/F as neural activity” rather than recomputing fluorescence or using inferred events (steps 12 and 15).

## 2-b. How is the `neural` data processed?

i. The AI uses the released dF/F traces as-is, transposes them from time-by-cell to cell-by-time, casts them to `float32`, and replaces rare non-finite values with zeros. Because each NWB is treated as its own session, it does not merge multiple imaging planes into one session.

ii.
```python
* use released, ROI-curated dF/F (not recomputed fluorescence or inferred events);
...
# h5py reads time x cells; target is cells x time.
x=np.asarray(dff[a:b,:],dtype=np.float32).T
# Very rare nonfinite dF/F values are invalid for the decoder.
if not np.isfinite(x).all():
    x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
```

iii. In the trajectory the AI said the released NWBs “already contain curated dF/F traces” and that a direct HDF5 converter should preserve those “processed streams” while keeping memory low (steps 13 and 15).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No extra ROI-level cell filtering is applied beyond trusting the released NWB content as already curated. The only neural cleanup is to zero-fill non-finite dF/F values after slicing.

ii.
```python
* use released, ROI-curated dF/F (not recomputed fluorescence or inferred events);
...
if not np.isfinite(x).all():
    x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
...
'neural_measure':'Allen released detrended dF/F from valid curated ROIs',
```

iii. The AI justified this by concluding from its NWB inspection that “Every stored ROI is already valid, indicating the released NWBs contain only curated ROIs” (step 8).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to native ophys timestamps and then segmented by trial `start_time` and `stop_time`. The code does not realign neural traces to `change_time`; it slices the ophys frames belonging to the full trial window.

ii.
```python
* use native ophys frames (~31 Hz) as the common clock and trial bins;
...
a=np.searchsorted(ot,starts[j],side='left')
b=np.searchsorted(ot,stops[j],side='right')
...
x=np.asarray(dff[a:b,:],dtype=np.float32).T
...
'temporal_alignment_event':'Native ophys timestamps; each trial spans SDK trial start_time through stop_time.',
```

iii. The trajectory states that the implementation would “align all outputs to ophys timestamps” and that each trial would “span SDK `start_time` through `stop_time`” (steps 12 and 15).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native ophys frame rate with no temporal rebinning. It assumes the recordings are about 31 Hz and hardcodes `time_bin_size` as `1000.0/31.0` ms.

ii.
```python
* use native ophys frames (~31 Hz) as the common clock and trial bins;
...
'time_bin_size':1000.0/31.0,
```

iii. In the trajectory the AI recorded that “Ophys sampling is about 31 Hz” and later planned to “use native ophys frames (~31 Hz) as the common clock and trial bins” (steps 9 and 15).

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The AI derives image identity from the stimulus presentation stream, specifically the presentation onset timestamps and image indices in `stimulus/presentation`, decoded through `stimulus/templates/.../control_description`. It only uses trial image names during the global vocabulary pre-scan, not for per-frame trial construction.

ii.
```python
for col in ('initial_image_name','change_image_name'):
    images.update(dec(v) for v in tr[col][:] if dec(v) not in ('','nan','None'))
...
pres=f['stimulus/presentation']
series=next(iter(pres.values()))
pon=series['timestamps'][:]; pind=series['data'][:].astype(int)
templ=next(iter(f['stimulus/templates'].values()))
if 'control_description' in templ:
    labels=[dec(x) for x in templ['control_description'][:]]
```

iii. The AI said in the trajectory that “Image names can be decoded from the template’s control descriptions” and that the raw stimulus series provides “image indices and onset timestamps at the 750 ms flash cadence” (steps 14 and 15).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code builds a global image vocabulary, prepends a synthetic `gray` category, then reconstructs a full-session image-identity time series on the ophys clock. For each stimulus onset it fills only the next 250 ms with the flashed image code, leaving the inter-stimulus 500 ms and omitted flashes as `gray`.

ii.
```python
subjects=sorted(subjects); regions=sorted(regions); images=sorted(images)
image_values=['gray']+images
image_code={v:i for i,v in enumerate(image_values)}
...
img=np.zeros(ot.size,dtype=np.int16)
for onset,ii in zip(pon,pind):
    if ii>=len(labels): continue
    label=labels[ii]
    if label not in image_code: continue
    a=np.searchsorted(ot,onset,'left'); b=np.searchsorted(ot,onset+0.250,'left')
    img[a:b]=image_code[label]
```

iii. The AI’s trajectory justification was explicit: “Image identity must account for the experiment’s 250 ms image flashes separated by 500 ms gray periods rather than holding each image for the full 750 ms cadence,” and “Omitted flashes therefore remain gray” (steps 15 and final script header).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is first reconstructed on the full ophys timebase, then trial outputs simply take `img[a:b]`, which uses the same ophys frame bounds as the neural slice `dff[a:b, :]`.

ii.
```python
img=np.zeros(ot.size,dtype=np.int16)
...
for j,a,b in bounds:
    if b<=a: continue
    x=np.asarray(dff[a:b,:],dtype=np.float32).T
    ...
    y=np.vstack((img[a:b],change,runbin[a:b],pupbin[a:b],
                 np.full(T,outcome,dtype=np.int8))).astype(np.int16)
```

iii. The trajectory repeatedly says the converter will “align all outputs to ophys timestamps” and “use processed dF/F and processed behavior streams, all aligned to ophys timestamps” (steps 12, 14, and 15).

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `image change` is derived from `change_time` in the trials table plus the `go` trial flag.

ii.
```python
change=np.zeros(T,dtype=np.int8)
ct=float(tr['change_time'][j])
# Catch trials have a sham change_time but no identity change.
if np.isfinite(ct) and bool(tr['go'][j]):
    k=np.searchsorted(ot[a:b],ct,'left')
    if k<T: change[k]=1
```

iii. The trajectory says the AI would use “change impulses” based on `change_time`, and later corrected the implementation so that only Go trials, not Catch trials, carry an actual image-change impulse (steps 15, 28, 35, 37).

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI computes `image change` as a single-frame impulse at the first ophys frame at or after `change_time`, only on Go trials. Catch trials keep zeros for the whole trial even though they have a sham `change_time`.

ii.
```python
change=np.zeros(T,dtype=np.int8)
ct=float(tr['change_time'][j])
# Catch trials have a sham change_time but no identity change.
if np.isfinite(ct) and bool(tr['go'][j]):
    k=np.searchsorted(ot[a:b],ct,'left')
    if k<T: change[k]=1
```

iii. The AI’s justification was that the specification required a value of 1 “only after an actual image identity change,” so it revised the code when it noticed sham Catch changes were being marked (step 28). The final trajectory confirms the impulses “exactly match the Go trials” (steps 35 to 37).

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric thresholding is applied. The variable is binary by construction: `0` for `no change` and `1` for `change`.

ii.
```python
change=np.zeros(T,dtype=np.int8)
...
if np.isfinite(ct) and bool(tr['go'][j]):
    ...
    change[k]=1
...
'output_values':[image_values,['no change','change'],['Q1','Q2','Q3','Q4','Q5'],
```

iii. The AI treated this as a semantic binary event rather than something to discretize, and its trajectory discussion focused on ensuring that only actual Go-trial changes got the positive category (steps 28 and 37).

## 4-d. How is `output` *Image change* aligned with the neural data?

i. `change_time` is converted to the first matching ophys frame inside the current trial slice with `np.searchsorted(ot[a:b], ct, 'left')`, so the change indicator shares the exact same trial frame indices as the neural data.

ii.
```python
for j,a,b in bounds:
    ...
    x=np.asarray(dff[a:b,:],dtype=np.float32).T
    ...
    if np.isfinite(ct) and bool(tr['go'][j]):
        k=np.searchsorted(ot[a:b],ct,'left')
        if k<T: change[k]=1
```

iii. The trajectory consistently describes all outputs as being aligned on the ophys clock, with image-change impulses placed “at the first ophys frame at/after” the trial `change_time` (steps 12, 15, and the final script header).

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the NWB running stream `processing/running/speed/data` with timestamps from `processing/running/speed/timestamps`.

ii.
```python
rt=f['processing/running/speed/timestamps'][:]
rv=f['processing/running/speed/data'][:]
run=interp_finite(ot,rt,rv)
```

iii. The AI described this as the “Processed running speed supplied by AllenSDK/NWB” and said it would use the “official filtered running speed” rather than re-deriving locomotion (steps 12 and 15).

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto the ophys timestamps with the helper `interp_finite`. If there are no finite samples it returns zeros; if there is one finite sample it repeats that constant. The interpolated trace is then discretized.

ii.
```python
def interp_finite(tnew,t,x):
    t=np.asarray(t,float); x=np.asarray(x,float)
    ok=np.isfinite(t)&np.isfinite(x)
    if ok.sum()==0: return np.zeros(len(tnew),dtype=np.float32)
    if ok.sum()==1: return np.full(len(tnew),x[ok][0],dtype=np.float32)
    return np.interp(tnew,t[ok],x[ok]).astype(np.float32)
...
rt=f['processing/running/speed/timestamps'][:]
rv=f['processing/running/speed/data'][:]
run=interp_finite(ot,rt,rv)
```

iii. In the trajectory the AI said it would “interpolate official filtered running speed ... to ophys time” as part of putting every output on the same clock (step 12).

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is binned into five within-recording quintiles. The quantile edges are computed from all retained-trial frames for that session only, and `np.searchsorted` assigns bins `0..4`, which are labeled `Q1` to `Q5`.

ii.
```python
def quintile(x, mask):
    vals=x[mask & np.isfinite(x)]
    if vals.size==0: return np.zeros(x.size,dtype=np.int8), [np.nan]*4
    edges=np.quantile(vals,[.2,.4,.6,.8])
    return np.searchsorted(edges,x,side='right').astype(np.int8), edges.tolist()
...
runbin,redges=quintile(run,retained)
...
'output_values':[image_values,['no change','change'],['Q1','Q2','Q3','Q4','Q5'],
...
'continuous_discretization':'Per-recording quintiles over retained trial frames; filtered missing pupil samples linearly interpolated.',
```

iii. The trajectory explicitly says the AI would “compute five percentile bins per experiment from finite retained-trial values” and later documents “Per-recording quintiles over retained trial frames” in metadata (steps 12 and 15).

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running is interpolated to the full ophys timebase first, then the discretized running trace is sliced with the same trial bounds used for neural data.

ii.
```python
run=interp_finite(ot,rt,rv)
...
for j in tids:
    a=np.searchsorted(ot,starts[j],side='left')
    b=np.searchsorted(ot,stops[j],side='right')
...
y=np.vstack((img[a:b],change,runbin[a:b],pupbin[a:b],
             np.full(T,outcome,dtype=np.int8))).astype(np.int16)
```

iii. The AI said it would “align all outputs to ophys timestamps” before trial slicing, including running speed (steps 12 and 15).

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the released pupil area stream in `acquisition/EyeTracking/pupil_tracking/area`, with timestamps from the same group if present, otherwise from `acquisition/EyeTracking/eye_tracking/timestamps`.

ii.
```python
pg=f['acquisition/EyeTracking/pupil_tracking']
pt=pg['timestamps'][:] if 'timestamps' in pg else f['acquisition/EyeTracking/eye_tracking/timestamps'][:]
pa=pg['area'][:]
```

iii. The trajectory says the AI determined that “Processed pupil area is in `acquisition/EyeTracking/pupil_tracking/area`” and decided to use that released stream directly (steps 14 and 15).

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI converts pupil area to diameter with `2*sqrt(area/pi)`, then interpolates finite values to the ophys clock with `interp_finite`. Sessions are skipped if the pupil stream is missing or has fewer than two finite samples. Missing blink/outlier periods are effectively bridged by interpolation across the finite samples.

ii.
```python
if 'acquisition/EyeTracking/pupil_tracking' not in f:
    ...
pa=pg['area'][:]
if np.isfinite(pa).sum() < 2:
    ...
diam=interp_finite(ot,pt,2*np.sqrt(np.maximum(pa,0)/np.pi))
```

iii. In the trajectory the AI concluded that “Canonical eye tracking provides filtered `pupil_area` with blink/outlier periods represented as NaN, which should be converted to diameter as `2*sqrt(area/pi)`,” and later said that sessions missing this required output should be excluded rather than imputed (steps 12, 15, and 19).

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is binned exactly like running speed: five within-recording quintiles computed from retained-trial frames, with labels `Q1` through `Q5`.

ii.
```python
pupbin,pedges=quintile(diam,retained)
...
'output_values':[image_values,['no change','change'],['Q1','Q2','Q3','Q4','Q5'],
                 ['Q1','Q2','Q3','Q4','Q5'],['hit','miss','false alarm','correct reject']],
...
'percentile_edges':percentile_edges,
```

iii. The AI’s plan was to “compute five percentile bins per experiment from finite retained-trial values” for both running and pupil (step 12), and the final metadata stores per-session edges (script lines 167 to 170 and 188 to 190).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is converted to the ophys clock before trial segmentation, and the discretized pupil trace is then sliced with the same `[a:b]` trial bounds used for neural data.

ii.
```python
diam=interp_finite(ot,pt,2*np.sqrt(np.maximum(pa,0)/np.pi))
...
pupbin,pedges=quintile(diam,retained)
...
y=np.vstack((img[a:b],change,runbin[a:b],pupbin[a:b],
             np.full(T,outcome,dtype=np.int8))).astype(np.int16)
```

iii. The trajectory repeatedly states that all streams would be aligned to ophys timestamps before trial extraction (steps 12 and 15).

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the four boolean trial flags `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
if bool(tr['hit'][j]): outcome=0
elif bool(tr['miss'][j]): outcome=1
elif bool(tr['false_alarm'][j]): outcome=2
elif bool(tr['correct_reject'][j]): outcome=3
else: raise ValueError(f'unclassified retained trial {j} in {p}')
```

iii. The AI stated in the trajectory that trials have “all required classification flags” and that it would encode a “four-class outcome” (steps 12 and 15).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four outcome flags are mapped to integer classes 0 to 3 in a fixed `if/elif` order, and the chosen outcome code is repeated across every time bin of the trial so that the output has shape `(d_output, T)`.

ii.
```python
if bool(tr['hit'][j]): outcome=0
elif bool(tr['miss'][j]): outcome=1
elif bool(tr['false_alarm'][j]): outcome=2
elif bool(tr['correct_reject'][j]): outcome=3
...
y=np.vstack((img[a:b],change,runbin[a:b],pupbin[a:b],
             np.full(T,outcome,dtype=np.int8))).astype(np.int16)
```

iii. The trajectory explicitly notes that “even the static trial outcome should be repeated across time” because the decoder expects each output trial to have shape `(d_output, T)` (step 10).

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or bad data mainly by fallback interpolation and by skipping sessions. `interp_finite` fills all-missing streams with zeros and one-sample streams with a constant, non-finite neural values are converted to zero, trial indices are clipped to recording bounds, empty trial windows are skipped, and entire sessions are skipped if they lack pupil tracking, lack stimulus presentations, have too few finite pupil samples, or have fewer than two retained trials. The pickle is written atomically through a temporary file.

ii.
```python
def interp_finite(tnew,t,x):
    ...
    if ok.sum()==0: return np.zeros(len(tnew),dtype=np.float32)
    if ok.sum()==1: return np.full(len(tnew),x[ok][0],dtype=np.float32)
...
if 'acquisition/EyeTracking/pupil_tracking' not in f:
    ...
if 'stimulus/presentation' not in f or len(f['stimulus/presentation']) == 0:
    ...
if np.isfinite(pa).sum() < 2:
    ...
if len(tids)<2:
    ...
a=max(0,a); b=min(ot.size,b)
...
if b<=a: continue
...
if not np.isfinite(x).all():
    x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
...
with open(OUT+'.tmp','wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
os.replace(OUT+'.tmp',OUT)
```

iii. The trajectory justification is that required outputs should not be fabricated, so sessions missing pupil data are excluded (step 19), while the general implementation is built to stream large files robustly and avoid failures from sparse or malformed samples (steps 3, 15, 19, 25, 37).

## 9-a. What are the most time-consuming steps of the code?

i. The code’s most time-consuming work is reading 284 very large NWB files, making a full pre-scan over all files, processing every retained trial and stimulus onset, and serializing the large final pickle. The implementation is intentionally I/O-heavy but memory-conscious.

ii.
```python
FILES=sorted(glob.glob(os.path.join(ROOT,'*.nwb')))
...
for p in FILES:
    with h5py.File(p,'r') as f:
        ...
for fi,p in enumerate(FILES):
    with h5py.File(p,'r') as f:
        ...
with open(OUT+'.tmp','wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The AI repeatedly described the dataset as 247 GB across 284 NWBs and said the converter had to “stream one file at a time” because this would take several minutes and be dominated by bulk data reading and writing (steps 3, 4, 13, 16 to 25, and 29 to 37).

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code leaves several Python loops that could have been vectorized or otherwise batched: the initial full-file pre-scan, the per-trial bounds loop that builds `retained` and `bounds`, the per-stimulus-onset loop that writes the image identity trace, and the per-trial loop that slices neural/output arrays.

ii.
```python
for p in FILES:
    with h5py.File(p,'r') as f:
        ...
for j in tids:
    a=np.searchsorted(ot,starts[j],side='left')
    b=np.searchsorted(ot,stops[j],side='right')
    ...
for onset,ii in zip(pon,pind):
    ...
for j,a,b in bounds:
    ...
```

iii. The trajectory does not propose vectorizing these loops; instead it emphasizes streaming and direct HDF5 access as the main performance strategy because the source data are extremely large and I/O dominates runtime (steps 3, 13, and 15).

## 9-c. What processing does the code repeat multiple times?

i. The code repeats a full pass over all NWB files: once to collect global subjects, regions, and image labels, then again to perform conversion. Within each file it also iterates over retained trials once to build the retained-frame mask and `bounds`, and then again to build the actual trial matrices.

ii.
```python
subjects=[]; regions=[]; images=set()
for p in FILES:
    with h5py.File(p,'r') as f:
        ...
for fi,p in enumerate(FILES):
    with h5py.File(p,'r') as f:
        ...
for j in tids:
    ...
    bounds.append((j,a,b))
...
for j,a,b in bounds:
    ...
```

iii. The trajectory frames this as a deliberate pre-scan for global vocabularies and then a second streaming conversion pass (steps 13 and 15), rather than as an optimization target.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does extra work that the decoder itself does not use: it reconstructs full-session image, running, and pupil traces before slicing only the trial segments; it pre-scans global subjects/regions/images for sessions that may later be skipped; and it stores `session_info` and `percentile_edges` metadata that are not consumed by the downstream decoder.

ii.
```python
subjects=[]; regions=[]; images=set()
...
retained=np.zeros(ot.size,bool); bounds=[]
...
img=np.zeros(ot.size,dtype=np.int16)
for onset,ii in zip(pon,pind):
    ...
session_info=[]; percentile_edges=[]
...
session_info.append({'ophys_experiment_id':eid,'mouse_id':sid,
  'targeted_structure':reg,'n_cells':ncell,'n_trials':len(ns),
  'ophys_frame_rate_hz':float(plane['imaging_rate'][()])})
percentile_edges.append({'running_speed_cm_per_s':redges,'pupil_diameter_pixels':pedges})
```

iii. The trajectory presents these as pragmatic implementation choices for documenting and reconstructing the data, not as outputs needed by the decoder. It specifically says the script will “document every alignment, filtering, interpolation, categorical encoding, and binning decision in comments and metadata” (step 13).
