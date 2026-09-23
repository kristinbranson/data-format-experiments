# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI only loads a hard-coded 12-session subset from `/app/data/Ephys_Behavior`, using one selected probe per session from a fixed `SESSIONS` list. Main session files are opened with `h5py` as HDF5, while motion-energy companions are opened separately with `scipy.io.loadmat`. It does not search both task folders, does not support the full 44-session cohort, and does not implement the v5/v7.3 dual MAT loader used by the human reference.

ii. ```python
ROOT=Path('/app/data/Ephys_Behavior')
SESSIONS=[
 ('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),
 ('JGR2','2021-11-16',1),('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),
 ('JEB19','2023-04-21',1),('JEB19','2023-04-20',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-18',1)]
...
with h5py.File(p,'r') as f:
    b=behavior(f)
```
```python
def motion_continuous(subject,date,trial_ids,go,frame_times):
    p=ROOT/f'motionEnergy_{subject}_{date}.mat'
    ...
    me=loadmat(p,simplify_cells=True)['me']
```

iii. In `CONVERSION_NOTES.md` and `README.md`, the AI justifies this as using the “exact 12 sessions selected by the reference Figure 8 metadata loaders” for the two-context subset, rather than the full paper cohort.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the first element of each hard-coded session tuple, and `subjects` is built in first-seen order while iterating through sessions. `subject_idx` stores the index of each session’s subject in that list.

ii. ```python
for si,(sub,date,probe) in enumerate(sessions):
    ...
    if sub not in subjects:subjects.append(sub)
    ...
    subject_idx.append(subjects.index(sub))
```

iii. The AI’s notes justify this by treating the released subject IDs in the Figure 8 subset as the authoritative subject split for the converted dataset.

## 1-c. How are the data split into sessions?

i. Each tuple in `SESSIONS` is treated as one session, and each tuple maps to one `data_structure_<subject>_<date>.mat` file inside `/app/data/Ephys_Behavior`. Each session becomes one element of `neural`, `input`, and `output`.

ii. ```python
def process_session(subject,date,probe,show=False):
    t0=time.time(); p=ROOT/f'data_structure_{subject}_{date}.mat'
    with h5py.File(p,'r') as f:
        ...
```
```python
for si,(sub,date,probe) in enumerate(sessions):
    n,x,y,info=process_session(sub,date,probe,...)
    neural.append(n);inputs.append(x);outputs.append(y)
```

iii. The justification in the notes is again that these 12 sessions are the intended two-context subset from Figure 8, so fixed session enumeration is preferable to broader discovery.

## 1-d. How are the data split into trials?

i. Trials are represented by indices into per-trial arrays from `obj.bp` and `obj.trials.bp`. After building a boolean `valid` mask, the AI uses `ids = np.flatnonzero(valid)` and then indexes neural, video, motion-energy, and label streams by each original trial index.

ii. ```python
b=behavior(f); n=len(b['go'])
valid=b['haveEphys']&np.isfinite(b['go'])&(np.asarray(b['early'])==0)&(np.asarray(b['stim'])==0)
outcome_sum=np.asarray(b['hit'])+np.asarray(b['miss'])+np.asarray(b['no'])
valid &= outcome_sum==1
ids=np.flatnonzero(valid)
```
```python
for old in ids:
    l=first_post_go(b['lickL'][old],b['go'][old]);r=first_post_go(b['lickR'][old],b['go'][old])
```

iii. The AI’s notes argue that the Bpod-style per-trial fields already define the trial structure directly, so trial reconstruction is unnecessary.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if `haveEphys` is true, the go cue is finite, the trial is not early, stimulation is off, and exactly one of `hit`, `miss`, or `no` is true. The code does not apply the reference’s explicit post-recording cutoff based on the last trial containing spikes.

ii. ```python
valid=b['haveEphys']&np.isfinite(b['go'])&(np.asarray(b['early'])==0)&(np.asarray(b['stim'])==0)
outcome_sum=np.asarray(b['hit'])+np.asarray(b['miss'])+np.asarray(b['no'])
valid &= outcome_sum==1
ids=np.flatnonzero(valid)
```

iii. `CONVERSION_NOTES.md` says the AI chose to exclude early and stimulated trials to follow the paper, but intentionally retained no-response trials because the decoder task explicitly requires `ignore` and `none` classes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from the selected probe inside `obj.clu`, specifically cluster `trialtm`, `trial`, and `quality`, together with per-trial `bp.ev.goCue` for alignment.

ii. ```python
def load_clusters(f, probe, go, trial_keep):
    pref=np.asarray(f['obj/clu']).ravel()[probe-1]; g=f[pref]
    ...
    for rt,ri,rq in zip(np.asarray(g['trialtm']).ravel(),np.asarray(g['trial']).ravel(),np.asarray(g['quality']).ravel()):
        q=h5str(f,rq).lower().strip()
        ...
        tm=np.atleast_1d(h5arr(f,rt)).astype(float)
        tr=np.atleast_1d(h5arr(f,ri)).astype(int)-1
```

iii. The AI’s notes say this matches the reference electrophysiology pathway: spike times already expressed per trial, plus go-cue timestamps for alignment.

## 2-b. How is the `neural` data processed?

i. For each retained cluster, spikes are aligned by subtracting go cue, histogrammed into 10 ms bins from -3.0 s to +2.5 s, converted to Hz by dividing by `DT`, then smoothed with a custom causal Gaussian intended to match MATLAB `mySmooth(x,15)`.

ii. ```python
TMIN,TMAX,DT=-3.0,2.5,0.01
EDGES=np.arange(TMIN,TMAX+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
...
z=tm[tr==old]-go[old]
mat[pos[old]]=np.histogram(z,EDGES)[0]/DT
```
```python
def causal_smooth(x, n=15):
    idx=np.linspace(-1.0,1.0,int(n),dtype=float)
    k=np.exp(-0.5*(2.5*idx)**2)
    k[:len(k)//2]=0.0; k/=k.sum()
    ...
```

iii. The notes justify this by saying Figure 8’s two-context analysis uses -3..+2.5 s, 10 ms bins, and causal `gausswin(15)` smoothing, and that this subset is the most directly applicable reference path.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters are kept only if their lower-cased quality label is one of `poor`, `fair`, `good`, `great`, `excellent`, or `multi`. After smoothing, units are further filtered to mean firing rate strictly greater than 1 Hz, based on a trial-averaged aligned PSTH.

ii. ```python
VALID_QUAL={'poor','fair','good','great','excellent','multi'}
...
q=h5str(f,rq).lower().strip()
if q not in VALID_QUAL: continue
```
```python
psth=np.histogram(aligned,EDGES)[0].astype(np.float32)/(len(go)*DT)
rates.append(float(np.mean(causal_smooth(psth[None,:])[0])))
use=np.asarray(rates)>1.0
```

iii. In the notes, the AI argues that this reproduces the Figure 8 single/multi-unit curation logic and the paper’s `>1 Hz` rule, while explicitly excluding `garbage`, `noisy`, `unknown`, and related labels.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to go-cue onset by subtracting the trial’s go-cue time from each spike’s within-trial time before histogramming.

ii. ```python
z=tm[tr==old]-go[old]
mat[pos[old]]=np.histogram(z,EDGES)[0]/DT
```

iii. The notes explicitly cite the reference `alignSpikes` rule as `trialtm_aligned = trialtm - event`, with `event = goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins over a -3.0 s to +2.5 s window, producing 550 time bins. Spikes are directly histogrammed onto that grid; no further rebinning is applied after histogramming.

ii. ```python
TMIN,TMAX,DT=-3.0,2.5,0.01
EDGES=np.arange(TMIN,TMAX+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

iii. The AI justifies 10 ms bins as the Figure 8 timing setup for the two-context subset.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This input is not read from a raw array. It is generated from the fixed bin-center grid defined by `TMIN`, `TMAX`, and `DT`, interpreted as time relative to go cue.

ii. ```python
TMIN,TMAX,DT=-3.0,2.5,0.01
EDGES=np.arange(TMIN,TMAX+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

iii. The AI’s notes describe this as the task-required signed time axis shared by all trials, rather than as a native behavioral variable.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code computes the centers of the fixed neural bin edges and then reuses the same `CENTERS` vector for every trial as a single-row input array.

ii. ```python
EDGES=np.arange(TMIN,TMAX+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
...
inputs.append(CENTERS[None,:].astype(np.float32))
```

iii. The notes justify this as aligning the decoder input exactly to the chosen neural time grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time input is exactly the same bin-center grid used to build the neural histograms, so each input timepoint corresponds to the same interval used for neural counts.

ii. ```python
mat[pos[old]]=np.histogram(z,EDGES)[0]/DT
...
inputs.append(CENTERS[None,:].astype(np.float32))
```

iii. The AI’s notes explicitly state that the signed-time input is analytically generated from the same 10 ms grid used for spike binning.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction from the actual lick-event lists `bp.ev.lickL` and `bp.ev.lickR`, together with `bp.ev.goCue`, by looking for the first post-go lick on each side. It does not derive lick direction from instructed side plus hit/miss.

ii. ```python
d['lickL']=cell_arrays(f,ev['lickL']); d['lickR']=cell_arrays(f,ev['lickR'])
...
l=first_post_go(b['lickL'][old],b['go'][old]);r=first_post_go(b['lickR'][old],b['go'][old])
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as using the animal’s emitted behavior directly: “target side cannot stand in for emitted behavior. No post-go lick is `none`.”

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each retained trial, the code finds the earliest left lick and earliest right lick at or after go cue. If the left lick occurs first the class is `left` (0), if the right lick occurs first the class is `right` (1), and if neither exists the class is `none` (2). That trial label is then repeated across all time bins.

ii. ```python
def first_post_go(a,go):
    x=np.atleast_1d(a).astype(float); x=x[np.isfinite(x)&(x>=go)]
    return np.min(x) if x.size else np.inf
```
```python
lick.append(0 if l<r else (1 if r<l else 2))
...
o=np.vstack([np.full(CENTERS.size,lick[i]), ...])
```

iii. The AI’s notes say this was chosen because the decoder target should reflect actual licks rather than inferred choice from task variables.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `bp.autowater` flag.

ii. ```python
d={k:get(k) for k in ['L','R','hit','miss','no','early','autowater']}
...
context.append(1 if b['autowater'][old] else 0)
```

iii. The notes justify this as following the figure-script logic where `autowater` denotes the WC context and its complement denotes DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code directly maps `autowater == 0` to class 0 and `autowater != 0` to class 1, with output labels `['DR', 'WC']`. The context label is repeated across all time bins within a trial.

ii. ```python
context.append(1 if b['autowater'][old] else 0)
...
'output_values':[['left','right','none'],['DR','WC'],['incorrect','correct','ignore'], ...]
```

iii. The AI’s notes say this ordering was chosen so the stored codes read as DR/WC in the output metadata, even though the paper text refers to WC and DR.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the per-trial `hit`, `miss`, and `no` flags from `obj.bp`.

ii. ```python
d={k:get(k) for k in ['L','R','hit','miss','no','early','autowater']}
...
outcome.append(0 if b['miss'][old] else (1 if b['hit'][old] else 2))
```

iii. The notes justify this as a direct mapping of the mutually exclusive raw trial-outcome flags.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps `miss` to `incorrect` (0), `hit` to `correct` (1), and all remaining valid trials to `ignore` (2). That label is repeated across all time bins.

ii. ```python
outcome.append(0 if b['miss'][old] else (1 if b['hit'][old] else 2))
...
o=np.vstack([..., np.full(CENTERS.size,outcome[i]), ...]).astype(np.int64)
```

iii. The AI’s notes say it intentionally keeps ignore trials because the decoder task explicitly requires an ignore class even though some paper analyses omitted them.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj` tracking arrays and `frameTimes` for camera 0, using any feature name containing `tongue`, plus per-trial go cue. The AI does not use the second camera’s tongue feature or the video/ephys offset from `obj.sglx`.

ii. ```python
cams=np.asarray(f['obj/traj']).ravel()
...
arr=np.asarray(f[tsrefs[old]],float)
ft=np.asarray(f[ftrefs[old]],float).squeeze()-go[old]
names=[x.lower() for x in feat_names(f,nrefs[old])]
...
tongue=speed_for(0,['tongue'])
```

iii. The notes justify this as “align video-derived speed ... to the same bins” while preserving missing visibility as a separate class, but they do not document a two-camera combination in the final code.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each matching tongue feature in camera 0, the code computes framewise speed as the Euclidean norm of `np.gradient(x, ft)` and `np.gradient(y, ft)`, masks out non-visible frames (`likelihood <= 0.9` or nonfinite coordinates), then resamples that speed to neural bins with nearest-neighbor assignment. If multiple tongue-like features match, it averages them per bin. It does not smooth positions first, split contiguous visible runs, normalize the two camera views separately, or average across views.

ii. ```python
if arr.shape[1]>2: visible &= np.isfinite(arr[j,2])&(arr[j,2]>0.9)
dx=np.gradient(x,ft);dy=np.gradient(y,ft); sp=np.hypot(dx,dy);sp[~visible]=np.nan
speeds.append(nearest_bin(ft,sp))
...
return np.divide(np.nansum(z,axis=0),count,out=np.full(z.shape[1],np.nan),where=count>0)
```

iii. The AI’s notes justify median-thresholded velocity outputs and explicit missing-data classes, but the trajectory and notes do not give a separate defense for dropping the reference’s smoothing and two-view combination.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After continuous tongue speed is produced for a session, all finite values across trials and time bins are split at the session 50th percentile. Values below the median become class 0, values at or above it become class 1, and NaNs become class 2 (`not visible`).

ii. ```python
def discretize_session(a):
    a=np.asarray(a,float); valid=np.isfinite(a); med=float(np.nanpercentile(a,50)) if valid.any() else np.nan
    y=np.full(a.shape,2,np.int64); y[valid & (a<med)]=0; y[valid & (a>=med)]=1
    return y,med
```

iii. The notes explicitly justify this as following the decoder specification’s per-session median split rather than the paper’s original movement thresholds.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are aligned only by subtracting the trial’s go-cue time. The resulting framewise speed is then mapped to the neural time axis with nearest-neighbor resampling onto `CENTERS`. No session-wide video offset correction is applied.

ii. ```python
ft=np.asarray(f[ftrefs[old]],float).squeeze()-go[old]
...
speeds.append(nearest_bin(ft,sp))
```

iii. The AI’s notes justify using a shared go-cue-centered grid for all streams, but the implemented code omits the reference offset-correction step.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj` camera 1 tracking and frame times, using any feature name containing `paw`, together with go cue. This can pool multiple paw-like features in camera 1.

ii. ```python
arr=np.asarray(f[tsrefs[old]],float)
ft=np.asarray(f[ftrefs[old]],float).squeeze()-go[old]
names=[x.lower() for x in feat_names(f,nrefs[old])]
...
paw=speed_for(1,['paw'])
```

iii. The notes justify using aligned video-derived paw speed with explicit missing-visibility handling, but do not preserve the reference’s specific `top_paw` selection.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing is the same as for tongue speed except it uses camera 1 and paw-matching features: raw coordinate gradients over frame time, visibility masking, nearest-bin resampling to neural bins, and averaging across any matched paw features before session-median discretization.

ii. ```python
def speed_for(cam,patterns):
    ...
    inds=[i for i,n in enumerate(names) if any(p in n for p in patterns)]
    ...
    dx=np.gradient(x,ft);dy=np.gradient(y,ft); sp=np.hypot(dx,dy);sp[~visible]=np.nan
    speeds.append(nearest_bin(ft,sp))
```

iii. The AI’s notes justify preserving time-varying paw speed and missing bins, but do not defend combining multiple paw features or skipping the reference smoothing.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is discretized with the same per-session 50th-percentile split used for tongue speed: below median is 0, at/above median is 1, and NaN is 2 (`not visible`).

ii. ```python
pv,pmed=discretize_session(paw)
```

iii. The notes justify this directly from the decoder prompt’s required discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw frame times are aligned by subtracting go cue and then resampled to the neural bin centers with nearest-neighbor assignment. No separate video-clock offset is used.

ii. ```python
ft=np.asarray(f[ftrefs[old]],float).squeeze()-go[old]
...
paw=speed_for(1,['paw'])
```

iii. The notes again justify a shared go-cue-centered time axis, but the code does not implement the reference video-offset correction.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from a separate `motionEnergy_<subject>_<date>.mat` file loaded from the same `Ephys_Behavior` root, specifically `me['data']`, together with the frame times passed in from the first camera of `video_continuous`.

ii. ```python
def motion_continuous(subject,date,trial_ids,go,frame_times):
    p=ROOT/f'motionEnergy_{subject}_{date}.mat'
    ...
    me=loadmat(p,simplify_cells=True)['me']; data=np.atleast_1d(me['data'])
```

iii. The notes justify using the standalone motion-energy files and treating missing files as a separate `no video` class.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI does not recompute motion energy from video pixels. It loads the precomputed per-frame trace, trims the two arrays to a common length, and resamples framewise values to neural bins with nearest-neighbor assignment before session-median discretization.

ii. ```python
v=np.asarray(data[old],float).squeeze(); t=np.asarray(ft,float).squeeze()
n=min(len(v),len(t)); out.append(nearest_bin(t[:n],v[:n]))
```

iii. The notes justify keeping the released motion-energy values and only changing the thresholding rule to the decoder-required median split.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy uses the same session-level median split as the kinematic outputs: below median is 0, at/above median is 1, and NaN becomes class 2 (`no video`).

ii. ```python
mv,mmed=discretize_session(motion)
...
'output_values':[..., ['below_session_median','at_or_above_session_median','no_video']]
```

iii. The notes explicitly state that the decoder’s per-session 50th percentile replaces the paper’s manual motion threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy samples are aligned using the frame-time vectors returned from `video_continuous` for the first camera, with go cue already subtracted there, and are then resampled to the neural bin centers with `nearest_bin`.

ii. ```python
results.append((tongue,paw,vals[0][1]))
...
motion=np.stack(motion_continuous(subject,date,ids,b['go'],fts))
```
```python
n=min(len(v),len(t)); out.append(nearest_bin(t[:n],v[:n]))
```

iii. The AI’s notes justify a common go-cue-centered alignment for all outputs, but the implementation again omits the reference’s video-offset correction and bin averaging.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing motion-energy files produce all-NaN traces that later become class 2 (`no video`). Missing or non-visible tracked points become NaN and later class 2 (`not visible`). Bins outside the support of available frame times also remain NaN because `nearest_bin` only fills the range between the earliest and latest finite frame times. The code does not impute across gaps.

ii. ```python
if not p.exists(): return [np.full(CENTERS.size,np.nan) for _ in trial_ids]
```
```python
visible=np.isfinite(x)&np.isfinite(y)
if arr.shape[1]>2: visible &= np.isfinite(arr[j,2])&(arr[j,2]>0.9)
...
out=np.full(CENTERS.size,np.nan)
...
inside=(CENTERS>=t[0])&(CENTERS<=t[-1]); out[inside]=v[pick[inside]]
```

iii. The notes justify this as preserving honest missingness for required class-2 outputs instead of fabricating kinematic or motion-energy values.

## 11-a. What are the most time-consuming steps of the code?

i. The likely runtime bottlenecks are the per-cluster spike histogramming/smoothing in `load_clusters`, the second full pass over all clusters to recompute firing-rate eligibility, and the per-trial video loops in `video_continuous` and `motion_continuous`. Unlike the human reference, this implementation also does substantial Python-level looping after file load.

ii. ```python
for rt,ri,rq in zip(...):
    ...
    for old in np.unique(tr):
        ...
        mat[pos[old]]=np.histogram(z,EDGES)[0]/DT
...
sm=np.stack([causal_smooth(x) for x in raw],axis=0)
```
```python
for old in trial_ids:
    ...
    tongue=speed_for(0,['tongue'])
    paw=speed_for(1,['paw'])
```

iii. The notes emphasize vectorized histograms and per-session timing, but the code structure shows that cluster and video loops are still major work items beyond file I/O.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain vectorizable: the inner `for old in np.unique(tr)` loop per cluster, the entire second cluster loop used only to recompute rate thresholds, the per-trial `video_continuous` loop, and the per-feature nearest-bin resampling. The code also rebuilds the `pos` trial-index mapping inside each cluster loop.

ii. ```python
for rt,ri,rq in zip(...):
    ...
    pos={int(t):i for i,t in enumerate(trial_keep)}
    for old in np.unique(tr):
        ...
```
```python
for rt,ri,rq in zip(...):
    ...
    psth=np.histogram(aligned,EDGES)[0].astype(np.float32)/(len(go)*DT)
    rates.append(float(np.mean(causal_smooth(psth[None,:])[0])))
```

iii. The notes claim “vectorized per-neuron histograms,” but the final code still retains several avoidable Python loops that the reference implementation avoided or minimized.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats cluster traversal twice in `load_clusters`: once to build trial-by-time matrices for every unit and again to compute the firing-rate inclusion mask. It also decodes feature names and frame-time arrays trial by trial in `video_continuous`, and rebuilds trial-position mappings inside the per-cluster loop.

ii. ```python
for rt,ri,rq in zip(...):
    ...
    mats.append(mat); qualities.append(q)
...
for rt,ri,rq in zip(...):
    ...
    psth=np.histogram(aligned,EDGES)[0].astype(np.float32)/(len(go)*DT)
    rates.append(...)
```

iii. The notes do not acknowledge this duplicated cluster pass; they mainly justify the design as matching reference curation and timing.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads several behavior fields it never uses in downstream outputs (`L`, `R`, `haveVid`). It also computes and returns the `use` mask from `load_clusters` even though the caller ignores it. More broadly, it computes continuous tongue, paw, and motion-energy arrays only to discretize them immediately, but that continuous stage is still necessary for the requested median split.

ii. ```python
d={k:get(k) for k in ['L','R','hit','miss','no','early','autowater']}
...
d['haveVid']=np.asarray(trials['haveVid']).squeeze().astype(bool)
```
```python
use=np.asarray(rates)>1.0
return sm[:,use,:],np.asarray(qualities)[use],use
```

iii. The notes focus on correctness rather than minimizing unused loading; the one clear AI justification is that preserving the continuous intermediate values is necessary to compute the per-session thresholds before categorization.
