# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly from disk using `h5py`, bypassing the AllenSDK's higher-level API. It globs all `.nwb` files under the `visual-behavior-ophys-1.1.0/behavior_ophys_experiments` directory and iterates over each file. A pre-scan pass collects global subjects, brain regions, and image names; then a second pass loads and processes each NWB individually.

ii.
```python
ROOT='/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments'
FILES=sorted(glob.glob(os.path.join(ROOT,'*.nwb')))
# Pre-scan for global vocabularies
for p in FILES:
    with h5py.File(p,'r') as f:
        sid=dec(f['general/subject/subject_id'][()])
        ...
# Main processing loop
for fi,p in enumerate(FILES):
    with h5py.File(p,'r') as f:
        ...
```

iii. The AI chose h5py over the AllenSDK to avoid heavy memory usage and slow SDK loading overhead for 284 NWB files (247 GB total). The trajectory notes: "Loading all 284 files through pynwb/AllenSDK would be unnecessarily slow and memory-heavy, so the converter should read the equivalent released datasets directly with h5py while preserving the SDK-defined trial flags and processed streams."

## 1-b. How are the data split into subjects?

i. Subjects are identified by reading `general/subject/subject_id` from each NWB file during the pre-scan. Unique subject IDs are collected across all files and sorted.

ii.
```python
subjects=[]
for p in FILES:
    with h5py.File(p,'r') as f:
        sid=dec(f['general/subject/subject_id'][()])
        if sid not in subjects: subjects.append(sid)
subjects=sorted(subjects)
```

iii. Each NWB file contains a subject ID; collecting unique IDs across all files recovers the full subject list.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session (one imaging plane = one session). There is no grouping of multiple imaging planes into a single behavioral session.

ii.
```python
for fi,p in enumerate(FILES):
    with h5py.File(p,'r') as f:
        # Each file is processed as an independent session
        ...
        neural.append(ns); inputs.append(ins); outputs.append(outs)
```

iii. The AI's trajectory notes that "each NWB file represents one imaging plane within one behavior session" but chose to treat each plane as an independent session. The converter header states "Each session has one imaging plane."

## 1-d. How are the data split into trials?

i. Trials are defined using the `intervals/trials` table in each NWB. The `go` and `catch` boolean columns are used to select valid trials. Aborted and auto-rewarded trials are excluded. Trial boundaries use `start_time` and `stop_time` to define variable-length windows.

ii.
```python
keep=(tr['go'][:].astype(bool)|tr['catch'][:].astype(bool))
keep &= ~tr['aborted'][:].astype(bool)
keep &= ~tr['auto_rewarded'][:].astype(bool)
tids=np.flatnonzero(keep)
...
for j in tids:
    a=np.searchsorted(ot,starts[j],side='left')
    b=np.searchsorted(ot,stops[j],side='right')
```

iii. The instructions specify "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials." The AI implemented this by filtering on the `go`, `catch`, `aborted`, and `auto_rewarded` flags.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) requiring `go` or `catch` to be true; (2) excluding `aborted`; (3) excluding `auto_rewarded`; (4) skipping trials where `b <= a` (empty window); (5) requiring at least 2 valid trials per session. Sessions lacking pupil tracking or stimulus presentations are also excluded entirely.

ii.
```python
keep=(tr['go'][:].astype(bool)|tr['catch'][:].astype(bool))
keep &= ~tr['aborted'][:].astype(bool)
keep &= ~tr['auto_rewarded'][:].astype(bool)
...
if len(tids)<2:
    print('skip <2 trials',p,flush=True); continue
...
if 'acquisition/EyeTracking/pupil_tracking' not in f:
    print(f'{fi+1}/{len(FILES)} skip: no pupil tracking ...'); continue
```

iii. The AI excluded sessions without pupil tracking because "pupil diameter is a required decoder output; fabricating values would be inappropriate." The minimum 2-trial threshold ensures decoder evaluation is possible.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/dff/traces/data` in the NWB, which contains pre-computed dF/F calcium traces.

ii.
```python
dff=find_dset(f,['processing/ophys/dff/traces/data'])
...
x=np.asarray(dff[a:b,:],dtype=np.float32).T
```

iii. The AI used the released dF/F traces, noting that these are the SDK's standard neural activity measure for two-photon imaging with neuropil correction already applied.

## 2-b. How is the `neural` data processed?

i. The dF/F data is read from HDF5 (stored as time x cells), transposed to cells x time, and cast to float32. Non-finite values (NaN, inf) are replaced with 0.0.

ii.
```python
x=np.asarray(dff[a:b,:],dtype=np.float32).T
if not np.isfinite(x).all():
    x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
```

iii. The NWB stores traces in time x cells format; transposing gives the required cells x time layout. Non-finite cleanup avoids decoder failures from rare corrupted samples.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level filtering is applied. All ROIs present in the released NWB files are included. The AI verified during exploration that "every stored ROI is already valid, indicating the released NWBs contain only curated ROIs."

ii. N/A (no filtering code)

iii. The AI confirmed that the released NWBs already contain only curated (valid) ROIs, so no additional `valid_roi` filtering was necessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start time. For each trial, ophys frames from `start_time` to `stop_time` are extracted using `np.searchsorted`, producing variable-length trial windows.

ii.
```python
starts=tr['start_time'][:]; stops=tr['stop_time'][:]
for j in tids:
    a=np.searchsorted(ot,starts[j],side='left')
    b=np.searchsorted(ot,stops[j],side='right')
    a=max(0,a); b=min(ot.size,b)
    ...
    x=np.asarray(dff[a:b,:],dtype=np.float32).T
```

iii. The instructions say "Temporally align based on ophys timestamp." The AI uses ophys timestamps as the common clock and extracts trial windows based on SDK-defined start/stop times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native ophys frame rate (~31 Hz). The time bin size is hardcoded as `1000.0/31.0` ms (~32.26 ms). No rebinning is applied.

ii.
```python
'time_bin_size':1000.0/31.0,
```

iii. The AI chose to use the native frame rate directly. However, the actual ophys frame rate varies by experiment; the AI hardcoded an approximate value rather than computing it from timestamps.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation series (`stimulus/presentation`) timestamps and data indices, combined with the template's `control_description` to map indices to image names. During gray periods (inter-stimulus intervals, omitted flashes), the code assigns value 0 ("gray").

ii.
```python
pres=f['stimulus/presentation']
series=next(iter(pres.values()))
pon=series['timestamps'][:]; pind=series['data'][:].astype(int)
templ=next(iter(f['stimulus/templates'].values()))
if 'control_description' in templ:
    labels=[dec(x) for x in templ['control_description'][:]]
...
img=np.zeros(ot.size,dtype=np.int16)  # default: gray (code 0)
for onset,ii in zip(pon,pind):
    if ii>=len(labels): continue
    label=labels[ii]
    if label not in image_code: continue
    a=np.searchsorted(ot,onset,'left'); b=np.searchsorted(ot,onset+0.250,'left')
    img[a:b]=image_code[label]
```

iii. The AI used the raw stimulus presentation data to reconstruct frame-by-frame image identity. Each flash lasts 250 ms, after which the screen returns to gray. This captures the actual visual stimulus timing including omitted flashes (which remain gray).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global image vocabulary is built from all NWB files during the pre-scan. Image codes are assigned as: 0 = "gray", then sorted unique image names. For each ophys frame, the code checks which (if any) 250ms image flash is active, assigning the corresponding code or defaulting to gray (0).

ii.
```python
images=set()
for p in FILES:
    with h5py.File(p,'r') as f:
        tr=f['intervals/trials']
        for col in ('initial_image_name','change_image_name'):
            images.update(dec(v) for v in tr[col][:] if dec(v) not in ('','nan','None'))
images=sorted(images)
image_values=['gray']+images
image_code={v:i for i,v in enumerate(image_values)}
```

iii. Including "gray" as an explicit category (code 0) captures the inter-stimulus interval and omitted flashes, representing the actual visual experience during those periods.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the ophys timebase by assigning each 250ms flash window to the corresponding ophys frames. The same frame indices (`a:b`) are used for both neural data and image identity.

ii.
```python
img=np.zeros(ot.size,dtype=np.int16)
for onset,ii in zip(pon,pind):
    a=np.searchsorted(ot,onset,'left'); b=np.searchsorted(ot,onset+0.250,'left')
    img[a:b]=image_code[label]
...
y=np.vstack((img[a:b], ...))
```

iii. Image identity is pre-computed for every ophys frame, then sliced using the same trial boundaries as neural data, guaranteeing alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `change_time` and `go` columns in the trials table. It is a binary impulse (single-frame 1) placed at the first ophys frame at or after `change_time`, but only for Go trials.

ii.
```python
ct=float(tr['change_time'][j])
if np.isfinite(ct) and bool(tr['go'][j]):
    k=np.searchsorted(ot[a:b],ct,'left')
    if k<T: change[k]=1
```

iii. The AI initially marked change impulses for all trials but corrected this after noticing that "catch trials have a sham change_time but no identity change." The final version only marks Go trials.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero-initialized array is created per trial. For Go trials with a finite `change_time`, a single frame is set to 1 at the change point. No processing beyond this binary indicator.

ii. See 4-a code snippet.

iii. The single-frame impulse marks the exact moment of image change. Catch trials get all zeros since no actual identity change occurs.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 or 1), so no thresholding is needed. Output values are `['no change', 'change']`.

ii.
```python
'output_values':[image_values,['no change','change'], ...]
```

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change impulse is computed within the trial's ophys frame window (`ot[a:b]`), directly aligned with neural data using the same indices.

ii.
```python
k=np.searchsorted(ot[a:b],ct,'left')
if k<T: change[k]=1
```

iii. Using the same trial-sliced ophys timestamps ensures frame-level alignment with neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and its timestamps in the NWB.

ii.
```python
rt=f['processing/running/speed/timestamps'][:]
rv=f['processing/running/speed/data'][:]
run=interp_finite(ot,rt,rv)
```

iii. This is the SDK's pre-filtered running speed signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the ophys timebase using `np.interp` (via the `interp_finite` helper, which filters out NaN values before interpolation). It is then discretized into 5 quintile bins computed per-session from retained trial frames only.

ii.
```python
def interp_finite(tnew,t,x):
    t=np.asarray(t,float); x=np.asarray(x,float)
    ok=np.isfinite(t)&np.isfinite(x)
    ...
    return np.interp(tnew,t[ok],x[ok]).astype(np.float32)

def quintile(x, mask):
    vals=x[mask & np.isfinite(x)]
    edges=np.quantile(vals,[.2,.4,.6,.8])
    return np.searchsorted(edges,x,side='right').astype(np.int8), edges.tolist()

run=interp_finite(ot,rt,rv)
retained=np.zeros(ot.size,bool)
# ... mark retained trial frames ...
runbin,redges=quintile(run,retained)
```

iii. The AI computed quintile bins per-session (per-NWB file) using only frames from retained trials, rather than globally across all sessions. The `interp_finite` helper handles missing values by excluding them before interpolation.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (codes 0-4) using quantile edges at [0.2, 0.4, 0.6, 0.8] computed per session from retained trial frames.

ii.
```python
edges=np.quantile(vals,[.2,.4,.6,.8])
return np.searchsorted(edges,x,side='right').astype(np.int8), edges.tolist()
```

iii. Quintile-based binning produces approximately equal bin counts within each session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the ophys timebase before trial segmentation, then sliced using the same frame indices as neural data.

ii.
```python
run=interp_finite(ot,rt,rv)
...
y=np.vstack((..., runbin[a:b], ...))
```

iii. Interpolation to the ophys clock followed by shared indexing ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` in the NWB. The raw pupil area is converted to diameter using `2*sqrt(area/pi)`.

ii.
```python
pg=f['acquisition/EyeTracking/pupil_tracking']
pa=pg['area'][:]
diam=interp_finite(ot,pt,2*np.sqrt(np.maximum(pa,0)/np.pi))
```

iii. The AI chose to convert pupil area to diameter because the task asks for "pupil diameter." The formula `2*sqrt(area/pi)` assumes a circular pupil.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is converted to diameter, non-finite values are excluded during interpolation to the ophys timebase, and the result is discretized into 5 quintile bins per session from retained trial frames.

ii.
```python
diam=interp_finite(ot,pt,2*np.sqrt(np.maximum(pa,0)/np.pi))
pupbin,pedges=quintile(diam,retained)
```

iii. Same quintile-based discretization as running speed, applied per-session.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 quintile bins (codes 0-4) with edges at [0.2, 0.4, 0.6, 0.8] quantiles per session.

ii. See `quintile()` function in 5-c.

iii. N/A

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: interpolated to ophys timebase, then sliced using the same frame indices.

ii.
```python
diam=interp_finite(ot,pt,2*np.sqrt(np.maximum(pa,0)/np.pi))
...
y=np.vstack((..., pupbin[a:b], ...))
```

iii. Shared ophys frame indices ensure alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns in the trials table.

ii.
```python
if bool(tr['hit'][j]): outcome=0
elif bool(tr['miss'][j]): outcome=1
elif bool(tr['false_alarm'][j]): outcome=2
elif bool(tr['correct_reject'][j]): outcome=3
else: raise ValueError(f'unclassified retained trial {j} in {p}')
```

iii. These four outcomes are the canonical SDK trial classifications for the change detection task. They are mutually exclusive for non-aborted, non-auto-rewarded trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is encoded as an integer (0-3) mapping to hit/miss/false_alarm/correct_reject. It is replicated across all time frames in the trial to make it time-varying.

ii.
```python
y=np.vstack((..., np.full(T,outcome,dtype=np.int8)))
```

iii. Replicating the static outcome across time makes it compatible with the time-varying output format expected by the decoder.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing pupil tracking**: Sessions without `acquisition/EyeTracking/pupil_tracking` are skipped entirely.
- **Missing stimulus presentations**: Sessions without stimulus data are skipped.
- **Insufficient pupil samples**: Sessions with fewer than 2 finite pupil values are skipped.
- **Non-finite dF/F**: Rare NaN/inf values in dF/F are replaced with 0.0.
- **Non-finite behavioral data**: The `interp_finite` helper excludes non-finite values before interpolation.
- **Few trials**: Sessions with fewer than 2 valid trials are skipped.
- **Empty trial windows**: Trials where `b <= a` are skipped.

ii.
```python
if 'acquisition/EyeTracking/pupil_tracking' not in f:
    continue
if np.isfinite(pa).sum() < 2:
    continue
if not np.isfinite(x).all():
    x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
def interp_finite(tnew,t,x):
    ok=np.isfinite(t)&np.isfinite(x)
    ...
```

iii. The AI's approach skips sessions that lack required data streams rather than fabricating data, which is a conservative and justified approach.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is reading the dF/F traces from each NWB file. Each file is ~300 MB and contains full-session neural data that must be read from disk. With 284 files totaling 247 GB, I/O dominates runtime.

ii. N/A

iii. The AI's trajectory shows multiple polling steps waiting for the full conversion to complete, indicating it took several minutes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The image identity construction loop iterates over every stimulus onset to assign 250ms flash windows. This could potentially be vectorized using array operations. The per-trial loop for extracting neural/behavioral data is sequential but involves HDF5 reads that are inherently sequential.

ii.
```python
for onset,ii in zip(pon,pind):
    ...
    a=np.searchsorted(ot,onset,'left'); b=np.searchsorted(ot,onset+0.250,'left')
    img[a:b]=image_code[label]
```

iii. The loop is not a bottleneck compared to I/O.

## 9-c. What processing does the code repeat multiple times?

i. The code performs two passes over all NWB files: a pre-scan to collect global vocabularies (subjects, regions, images) and a main pass for processing. This means each file is opened twice.

ii.
```python
# Pre-scan
for p in FILES:
    with h5py.File(p,'r') as f:
        sid=dec(f['general/subject/subject_id'][()])
        ...
# Main processing
for fi,p in enumerate(FILES):
    with h5py.File(p,'r') as f:
        ...
```

iii. The pre-scan only reads small metadata fields, so the overhead is minimal compared to the main processing pass.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes per-session percentile edges and stores them in metadata (`percentile_edges`), which is informational but not used by the decoder. The session_info metadata is also extra bookkeeping.

ii.
```python
percentile_edges.append({'running_speed_cm_per_s':redges,'pupil_diameter_pixels':pedges})
session_info.append({'ophys_experiment_id':eid,'mouse_id':sid, ...})
```

iii. This metadata is useful for debugging and interpretation but not consumed by the decoder.
