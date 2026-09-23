# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads only a hard-coded 12-session subset from `/app/data/Ephys_Behavior`. For each session it opens `data_structure_<subject>_<date>.mat` with `h5py`, and it separately opens `motionEnergy_<subject>_<date>.mat` with `scipy.io.loadmat` when available.

ii. 
```python
ROOT=Path('/app/data/Ephys_Behavior')
SESSIONS=[('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),('JGR2','2021-11-16',1),
 ('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),('JEB19','2023-04-18',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-20',1),('JEB19','2023-04-21',1)]
...
with h5py.File(path,'r') as f:
...
m=loadmat(p,squeeze_me=True,struct_as_record=False)['me']
```

iii. `CONVERSION_NOTES.md` Step 4-5 says the agent intentionally chose the 12 Figure-8 two-context ALM/video sessions because WC/DR labels were required, and Step 6 says it used targeted HDF5 traversal to avoid full MAT deserialization.

## 1-b. How are the data split into subjects?

i. The subject is taken from the first field of each hard-coded `(subject, date, probe)` tuple. `subjects` is built in first-seen order, and `subject_idx` stores that order for each session.

ii. 
```python
for i,(sub,date,probe) in enumerate(sessions):
    ...
    if sub not in subjects:subjects.append(sub)
    sidx.append(subjects.index(sub))
```

iii. The notes repeatedly describe the chosen dataset as the 12 Figure-8 sessions from seven released subject IDs. No separate justification for preserving first-seen order instead of sorting was documented.

## 1-c. How are the data split into sessions?

i. Each hard-coded tuple in `SESSIONS` is treated as one session. One `data_structure_<subject>_<date>.mat` file becomes one element of `neural`, `input`, and `output`.

ii. 
```python
def convert_session(subject,date,probe,show=False):
    t0=time.time(); path=ROOT/f'data_structure_{subject}_{date}.mat'
...
for i,(sub,date,probe) in enumerate(sessions):
    n,x,y,info,cont=convert_session(sub,date,probe,args.show_processing)
    neural.append(n);inputs.append(x);outputs.append(y);infos.append(info)
```

iii. `CONVERSION_NOTES.md` Step 4-5 justifies the 12-session choice by matching the paper's Figure 8 two-context subset rather than the broader ephys dataset.

## 1-d. How are the data split into trials?

i. The number of trials comes from `bp['Ntrials']`. Trial-level behavioral arrays are read at that length, and the kept trial indices in `keep_trials` define the final trial list. Spike data use per-spike trial IDs (`clu.trial`), and video/motion-energy data are indexed trial-by-trial with the same retained trial indices.

ii. 
```python
b=f['obj/bp']; ntr=int(np.asarray(b['Ntrials'][()]).squeeze())
...
keep_trials=np.flatnonzero((~stim)&np.isfinite(go))
...
tr=np.asarray(f[rr][()]).ravel(order='F').astype(int)-1
...
for j,tr in enumerate(keep_trials):
```

iii. No special trial-boundary reconstruction was documented; the notes treat the native behavioral trial table as authoritative.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered only by excluding photostimulation trials and trials with non-finite go-cue times. Early-lick trials are retained. There is no later cutoff for trials that extend past the end of the recording.

ii. 
```python
stim=np.zeros(ntr,bool)
if isinstance(b.get('stim'),h5py.Group) and 'enable' in b['stim']:stim=vec(b['stim/enable'],bool)
keep_trials=np.flatnonzero((~stim)&np.isfinite(go))
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says to exclude stimulation trials but retain hit, miss, no-response, and early trials with valid go cues because the decoder required incorrect/ignore outputs. The notes do not justify omitting the reference's early-lick exclusion or recording-end cutoff.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from the selected probe's cluster table: `quality`, `trial`, and `trialtm`, together with `bp.ev.goCue` for alignment.

ii. 
```python
go=vec(b['ev/goCue'],float)
clu=f['obj/clu']; cg=f[clu[()].ravel(order='F')[probe-1]]
qualities=np.array([chars(f,r).lower() for r in cg['quality'][()].ravel(order='F')])
trialrefs=cg['trial'][()].ravel(order='F'); tmrefs=cg['trialtm'][()].ravel(order='F')
```

iii. The notes describe the converter as using the selected probe, source quality labels, and go-cue alignment, matching the agent's interpretation of the Figure 8 ALM pipeline.

## 2-b. How is the `neural` data processed?

i. For each retained unit, spike times are aligned by subtracting the trial's go cue, counted into 10 ms bins from -2.5 s to 2.5 s, converted to Hz by dividing by `DT`, stacked as trial-by-neuron-by-time, and smoothed with a centered 15-bin boxcar (`uniform_filter1d`).

ii. 
```python
DT=.01; TMIN=-2.5; TMAX=2.5
EDGES=np.arange(TMIN,TMAX+DT/2,DT); TIME=(EDGES[:-1]+EDGES[1:])/2
...
z=tm[(tr==old)]-go[old]
mats[remap[old]]=np.histogram(z,EDGES)[0]/DT
...
neural=np.stack(unit_counts,axis=1).astype(np.float32)
neural=uniform_filter1d(neural,size=15,axis=2,mode='nearest').astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 4-5 says the agent deliberately chose the Figure 8a-c population settings: 10 ms bins, no time warping, and smoothing width 15, instead of the 5 ms single-unit Figure 8d settings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by cluster quality and mean firing rate. The drop set is `{'garbage', '', 'noisy', 'real?'}`. Surviving units must have mean aligned firing rate strictly greater than 1 Hz over all native valid trials in the [-2.5, 2.5) s window. The code keeps `poor` units and does not handle the typo label `gabrga`.

ii. 
```python
ARTIFACT={'garbage','','noisy','real?'}
...
aligned_all=tm-go[tr]
rate=np.sum((aligned_all>=TMIN)&(aligned_all<TMAX))/(ntr*(TMAX-TMIN))
if q not in ARTIFACT and rate>1:
    ...
```

iii. `CONVERSION_NOTES.md` Step 5 says the agent intended to exclude only "clear artifacts" and keep poor/fair/good/great/excellent/multi units because it interpreted the population analysis as using all units above 1 Hz.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting `goCue` from the same trial.

ii. 
```python
z=tm[(tr==old)]-go[old]
```

iii. The notes explicitly cite `alignSpikes` and say the converter should subtract `bp.ev.goCue` from within-trial spike times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins over a 5 s window, yielding 500 time points. There is no extra temporal rebinning after the initial histogramming; only smoothing is applied.

ii. 
```python
DT=.01; TMIN=-2.5; TMAX=2.5
EDGES=np.arange(TMIN,TMAX+DT/2,DT); TIME=(EDGES[:-1]+EDGES[1:])/2
...
mats[remap[old]]=np.histogram(z,EDGES)[0]/DT
```

iii. `CONVERSION_NOTES.md` Step 4-5 defends 10 ms bins as the Figure 8 population setting.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a separate raw variable. It is constructed analytically from the chosen time grid around the go cue.

ii. 
```python
DT=.01; TMIN=-2.5; TMAX=2.5
EDGES=np.arange(TMIN,TMAX+DT/2,DT); TIME=(EDGES[:-1]+EDGES[1:])/2
```

iii. The notes say the input should be the bin-center vector for the go-cue-aligned analysis window.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script computes the bin centers once and copies the same `TIME[None, :]` array into every trial.

ii. 
```python
TIME=(EDGES[:-1]+EDGES[1:])/2
...
return [x.astype(np.float32) for x in neural], [TIME[None,:].astype(np.float32).copy() for _ in keep_trials], outputs, info, (tongue,paw,me)
```

iii. No extra processing was justified beyond using the analysis time base itself.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the same `TIME` grid as the neural binning, so its columns correspond one-to-one with the neural time bins.

ii. 
```python
EDGES=np.arange(TMIN,TMAX+DT/2,DT); TIME=(EDGES[:-1]+EDGES[1:])/2
...
mats[remap[old]]=np.histogram(z,EDGES)[0]/DT
...
[TIME[None,:].astype(np.float32).copy() for _ in keep_trials]
```

iii. The notes say the bin centers are meant to match the same go-cue-aligned neural window exactly.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from trial flags `R`, `hit`, `miss`, and `no`.

ii. 
```python
R,L,hit,miss,no,early,aw=map(get,['R','L','hit','miss','no','early','autowater'])
```

iii. `CONVERSION_NOTES.md` Step 5 says R/L encode instructed side, while hit/miss/no determine actual behavioral response.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The script maps no-response trials to `2` (`none`), hit trials to the instructed side, and miss trials to the opposite side.

ii. 
```python
if no[tr]:lick=2
elif hit[tr]:lick=1 if R[tr] else 0
elif miss[tr]:lick=0 if R[tr] else 1
else:lick=2
```

iii. The notes explicitly justify inverting the instructed side on miss trials so the label reflects actual lick direction rather than trial instruction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from `autowater`.

ii. 
```python
R,L,hit,miss,no,early,aw=map(get,['R','L','hit','miss','no','early','autowater'])
```

iii. The notes say `autowater` is the code/data indicator of WC versus DR context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The script maps `autowater=True` to WC (`0`) and `False` to DR (`1`), then repeats that value across time bins.

ii. 
```python
context=0 if aw[tr] else 1
...
y=np.vstack([np.full(len(TIME),lick),np.full(len(TIME),context),np.full(len(TIME),outcome),td[j],pd[j],md[j]]).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 states that Figure 8 contrasts `autowater` against `~autowater`, so this direct relabeling was intentional.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `hit` and `miss`; trials that are neither are treated as ignore.

ii. 
```python
R,L,hit,miss,no,early,aw=map(get,['R','L','hit','miss','no','early','autowater'])
...
outcome=1 if hit[tr] else (0 if miss[tr] else 2)
```

iii. The notes describe outcome as a direct relabeling of the behavioral result flags into incorrect/correct/ignore.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The mapping is `miss -> 0 (incorrect)`, `hit -> 1 (correct)`, and all other retained trials -> `2 (ignore)`. The value is repeated across all time bins of the trial.

ii. 
```python
outcome=1 if hit[tr] else (0 if miss[tr] else 2)
...
y=np.vstack([np.full(len(TIME),lick),np.full(len(TIME),context),np.full(len(TIME),outcome),td[j],pd[j],md[j]]).astype(np.int64)
```

iii. The notes say ignore trials were kept because the decoder specification required an ignore class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the first camera view's DeepLabCut feature named `tongue`, using that trial's `ts`, `frameTimes`, and likelihood values, plus the trial go cue. The script does not use the bottom-camera tongue feature.

ii. 
```python
def feature_trial(f,view,trial,name):
    names_obj=f[view['featNames'][()].ravel(order='F')[trial]]
    names=[chars(f,r) for r in names_obj[()].ravel(order='F')]
    if name not in names:return None,None,None
    a=deref_array(f,view['ts'],trial)
...
for j,tr in enumerate(keep_trials):
    ft,pos,lk=feature_trial(f,views[0],tr,'tongue');tongue[j]=velocity_on_grid(ft,pos,lk,go[tr])
```

iii. `CONVERSION_NOTES.md` Step 5 mentions "side-view DLC tongue landmark positions" and preserving missing visibility. No explicit justification was documented for dropping the bottom-camera tongue view.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The script marks frames visible when x/y, frame time, and likelihood are finite and `likelihood > 0.9`. It linearly interpolates missing coordinates along the full frame index, smooths each coordinate with a 21-frame boxcar, computes speed from frame-to-frame displacement divided by frame-time differences, interpolates that speed onto the neural time grid, restores invisibility gaps by nearest-frame visibility, and finally median-splits the finite session-wide values.

ii. 
```python
visible=np.isfinite(pos).all(1)&np.isfinite(ft)&np.isfinite(likelihood)&(likelihood>0.9)
pp=pos.copy()
for j in range(2):
    good=np.isfinite(pp[:,j])
    if good.sum()>=2:
        fill=np.interp(np.arange(len(pp)),np.flatnonzero(good),pp[good,j])
        pp[:,j]=uniform_filter1d(fill,size=21,mode='nearest')
dt=np.diff(ft); speed=np.r_[np.nan,np.sqrt(np.sum(np.diff(pp,axis=0)**2,axis=1))/np.where(dt>0,dt,np.nan)]
...
out[inside]=np.interp(TIME[inside],rel[good],speed[good])
```

iii. The notes justify per-session medians and preserving missingness as category 2. No explicit justification was found for replacing the reference two-view, run-wise Gaussian/binning pipeline with one-view interpolation.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. All finite tongue-velocity samples from the session are split at the session median. Values below the median become `0`, values at or above it become `1`, and NaNs become `2`.

ii. 
```python
def disc(a):
    out=np.full(a.shape,2,np.int64); finite=np.isfinite(a)
    if finite.any():
        med=float(np.median(a[finite]));out[finite]=(a[finite]>=med).astype(np.int64)
    else:med=np.nan
    return out,med
td,tmed=disc(tongue)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says thresholds should be per-session medians over finite available samples and that missing visibility should map to class 2.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The code aligns video by subtracting `goCue` from raw `frameTimes` and then interpolating speed onto the neural `TIME` grid. It does not apply a session-wide video/behavior clock offset first.

ii. 
```python
speed[~visible]=np.nan; rel=ft-go
...
inside=(TIME>=rel[good].min())&(TIME<=rel[good].max())
out[inside]=np.interp(TIME[inside],rel[good],speed[good])
```

iii. The notes emphasize go-cue alignment and preserving missingness, but no explicit justification was recorded for omitting the reference `bitStart`/`sglx.bitcode` video offset correction.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the second camera view's DeepLabCut feature `top_paw`, using its `ts`, `frameTimes`, likelihood values, and the trial go cue.

ii. 
```python
for j,tr in enumerate(keep_trials):
    ...
    ft,pos,lk=feature_trial(f,views[1],tr,'top_paw');paw[j]=velocity_on_grid(ft,pos,lk,go[tr])
```

iii. The trajectory notes explicitly mention that representative DLC data used view-2 `top_paw`, and the mapping notes identify paw velocity with that feature.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw velocity uses the same `velocity_on_grid` pipeline as tongue velocity: visibility threshold at 0.9, coordinate interpolation, 21-frame boxcar smoothing, speed by finite differencing, interpolation onto the neural time grid, and session-median discretization.

ii. 
```python
ft,pos,lk=feature_trial(f,views[1],tr,'top_paw');paw[j]=velocity_on_grid(ft,pos,lk,go[tr])
...
pd,pmed=disc(paw)
```

iii. The notes justify using named DLC features, per-session medians, and preserving missingness. No more specific justification for this smoothing/interpolation choice was documented.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Finite paw-velocity samples are split at the session median into classes `0` and `1`; NaNs become `2`.

ii. 
```python
pd,pmed=disc(paw)
```

iii. `CONVERSION_NOTES.md` Step 5 says session medians should be computed from finite samples only, with missing values mapped to class 2.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The code subtracts each trial's `goCue` from the raw frame times and interpolates the paw speed onto the neural `TIME` grid. It does not apply a video-clock offset.

ii. 
```python
speed[~visible]=np.nan; rel=ft-go
...
out[inside]=np.interp(TIME[inside],rel[good],speed[good])
```

iii. No explicit justification was found for skipping the reference video-offset correction.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from a separate `motionEnergy_<subject>_<date>.mat` file in the same folder. The loader expects `me.data` to be a trial-wise cell-like array.

ii. 
```python
def load_motion(subject,date,ntrials):
    p=ROOT/f'motionEnergy_{subject}_{date}.mat'
    ...
    m=loadmat(p,squeeze_me=True,struct_as_record=False)['me']
    raw=np.asarray(m.data,dtype=object).ravel(order='F')
```

iii. The notes say motion energy should come from the dedicated per-session motion-energy files and be preserved as a separate stream with a "no video" class when absent.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each retained trial, the script pairs the motion-energy trace with the first view's frame times, truncates both to the same length, converts frame times to time-from-go-cue, interpolates the trace onto the neural `TIME` grid wherever at least two finite samples exist, and then median-splits the finite session-wide values.

ii. 
```python
if me_raw is not None:
    for j,tr in enumerate(keep_trials):
        y=me_raw[tr]; ft=video_times[j]; n=min(len(y),len(ft)); y=y[:n]; rel=ft[:n]-go[tr]
        good=np.isfinite(y)&np.isfinite(rel)
        if good.sum()>1:
            inside=(TIME>=rel[good].min())&(TIME<=rel[good].max())
            me[j,inside]=np.interp(TIME[inside],rel[good],y[good])
```

iii. The notes justify interpolating motion energy using actual camera timestamps and using per-session medians. No separate justification was documented for interpolation instead of bin-averaging.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Finite motion-energy values are split at the session median into `0` and `1`; NaNs remain `2`, interpreted as no video/data.

ii. 
```python
md,mmed=disc(me)
```

iii. `CONVERSION_NOTES.md` Step 5 says motion energy should be split at a per-session median and that absent video should remain class 2.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using the first camera's raw frame times minus the trial go cue, and then interpolated onto the same `TIME` grid as the neural data. No session-level video offset is applied.

ii. 
```python
video_times.append(deref_array(f,views[0]['frameTimes'],tr).ravel(order='F').astype(float))
...
y=me_raw[tr]; ft=video_times[j]; n=min(len(y),len(ft)); y=y[:n]; rel=ft[:n]-go[tr]
...
me[j,inside]=np.interp(TIME[inside],rel[good],y[good])
```

iii. The notes say motion energy samples were aligned using actual camera frame timestamps, but they do not document a reason for omitting the reference video-offset correction.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed inputs are mostly turned into NaNs and then into class `2`. If a feature is absent, `feature_trial` returns `None` and `velocity_on_grid` returns all-NaN. If fewer than two frame samples exist, the trial stays all-NaN. Motion-energy load failures or trial-count mismatches return `None`, which makes the whole motion-energy stream class `2`. Mismatched frame and position lengths are truncated to the shorter one. Within visible streams, missing coordinates are interpolated before smoothing, but bins nearest invisible frames are reset to NaN afterward.

ii. 
```python
if name not in names:return None,None,None
...
if ft is None or len(ft)<2:return out
...
n=min(len(ft),len(x)); return ft[:n],np.c_[x[:n],y[:n]],likelihood[:n]
...
if len(raw)!=ntrials:return None
...
fill=np.interp(np.arange(len(pp)),np.flatnonzero(good),pp[good,j])
...
out[np.flatnonzero(inside)[~visible[near]]]=np.nan
```

iii. The notes explicitly justify preserving missing visibility/video as category 2 rather than converting it to low movement. They do not separately justify interpolating missing coordinates before smoothing.

## 11-a. What are the most time-consuming steps of the code?

i. The expensive work in this script is the per-unit spike processing loop and the per-trial video/motion-energy interpolation loop inside each session, plus the targeted HDF5 reads that feed them. The code was written to reduce whole-file MAT deserialization cost.

ii. 
```python
for q,rr,rt in zip(qualities,trialrefs,tmrefs):
    ...
    for old in np.unique(tr[use]):
        z=tm[(tr==old)]-go[old]
        mats[remap[old]]=np.histogram(z,EDGES)[0]/DT
...
for j,tr in enumerate(keep_trials):
    ft,pos,lk=feature_trial(f,views[0],tr,'tongue');tongue[j]=velocity_on_grid(ft,pos,lk,go[tr])
    ft,pos,lk=feature_trial(f,views[1],tr,'top_paw');paw[j]=velocity_on_grid(ft,pos,lk,go[tr])
```

iii. `CONVERSION_NOTES.md` Step 6 says the agent switched to targeted HDF5 reference traversal because full MAT loading stalled, and that conversion time was about 3 seconds per sample session after this change.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunity left in the code is the trial-by-trial spike histogram loop inside each unit. The video loops are harder to vectorize because trials have ragged frame counts, but the spike binning could have been closer to a single `histogram2d` style pass.

ii. 
```python
for q,rr,rt in zip(qualities,trialrefs,tmrefs):
    ...
    for old in np.unique(tr[use]):
        z=tm[(tr==old)]-go[old]
        mats[remap[old]]=np.histogram(z,EDGES)[0]/DT
```

iii. `CONVERSION_NOTES.md` Step 6 says the agent "vectorized histogram construction where possible," implying it saw this as a partial optimization rather than a fully vectorized implementation.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several session-local operations: it rebuilds `remap` and `np.isin(tr, keep_trials)` inside every unit loop, it interpolates tongue, paw, and motion energy in three separate per-trial passes, and it recreates constant per-trial label vectors with `np.full(len(TIME), ...)` for every output row.

ii. 
```python
use=np.isin(tr,keep_trials)
...
remap=np.full(ntr,-1,int);remap[keep_trials]=np.arange(len(keep_trials))
...
for j,tr in enumerate(keep_trials):
    ft,pos,lk=feature_trial(f,views[0],tr,'tongue');tongue[j]=velocity_on_grid(ft,pos,lk,go[tr])
    ft,pos,lk=feature_trial(f,views[1],tr,'top_paw');paw[j]=velocity_on_grid(ft,pos,lk,go[tr])
...
y=np.vstack([np.full(len(TIME),lick),np.full(len(TIME),context),np.full(len(TIME),outcome),td[j],pd[j],md[j]]).astype(np.int64)
```

iii. No explicit justification for these repeated steps was documented beyond the general focus on getting a fast-enough converter working.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code is relatively targeted, but it still reads some behavioral arrays that are unused in the final computation (`L` and `early`), defines an unused helper (`smooth_rates`), and keeps a `show` argument that `convert_session` itself does not use. The much larger design choice was to avoid unnecessary full MAT deserialization in the first place.

ii. 
```python
R,L,hit,miss,no,early,aw=map(get,['R','L','hit','miss','no','early','autowater'])
...
def smooth_rates(x,width=15):
    # Reference mySmooth is a centered boxcar; nearest avoids artificial zero edges.
    return uniform_filter1d(x,size=width,axis=1,mode='nearest').astype(np.float32)
...
def convert_session(subject,date,probe,show=False):
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly says the agent moved away from full recursive MAT loading because it stalled and would materialize large unused branches. No separate note justified the smaller unused reads that remain.
