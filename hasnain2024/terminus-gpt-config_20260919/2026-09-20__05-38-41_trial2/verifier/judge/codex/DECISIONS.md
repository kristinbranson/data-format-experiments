# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes 12 subject/date/probe records from the Figure 8 two-context cohort, all under `Ephys_Behavior`, and opens each `data_structure_*.mat` with `h5py`. It therefore does not load the reference solution's full 44-session cohort (including randomized-delay sessions), nor support classic MAT session files.

ii.
```python
ROOT=Path('/app/data/Ephys_Behavior')
SESSIONS=[('JEB6','2021-04-18',2), ...]
with h5py.File(p,'r') as f:
    b=behavior(f)
```

iii. The notes say the Figure 8 cohort is the closest match because behavioral context is required and those exact 12 sessions contain both WC and DR; the AI intentionally chose it instead of all neural-bearing recordings.

## 1-b. How are the data split into subjects?

i. Subject IDs are taken from the hard-coded session tuples. A first-seen-order `subjects` list is built and each session receives its index. The output has seven released IDs.

ii.
```python
if sub not in subjects:subjects.append(sub)
subject_idx.append(subjects.index(sub))
```

iii. The AI states that filenames/metadata provide released IDs and declines to merge them merely to force the paper's six-mouse count.

## 1-c. How are the data split into sessions?

i. Each hard-coded `(subject,date,probe)` tuple and corresponding MAT file becomes one session-level element of `neural`, `input`, and `output`.

ii.
```python
for si,(sub,date,probe) in enumerate(sessions):
    n,x,y,info=process_session(sub,date,probe,...)
    neural.append(n);inputs.append(x);outputs.append(y)
```

iii. The notes justify the 12 records as the exact Figure 8/two-context metadata selection and the relevant probes.

## 1-d. How are the data split into trials?

i. Trial rows come directly from Bpod-length arrays. `ids=np.flatnonzero(valid)` retains source indices; spikes, video, motion, and scalar labels are indexed with those IDs, and each retained row is emitted as one trial.

ii.
```python
n=len(b['go'])
ids=np.flatnonzero(valid)
for i in range(len(ids)):
    neur.append(neural[i]); inputs.append(CENTERS[None,:]); outputs.append(o)
```

iii. The AI reports verifying equal trial-field lengths and cross-stream mappings, avoiding reconstruction of boundaries.

## 1-e. How are trials filtered based on quality controls?

i. It keeps trials with `haveEphys`, finite go cue, no early lick, no photostimulation, and exactly one of hit/miss/no. It retains ignore trials because required outputs include ignore/none. It does not use the reference's last-spiking-trial recording cutoff.

ii.
```python
valid=b['haveEphys']&np.isfinite(b['go'])&(np.asarray(b['early'])==0)&(np.asarray(b['stim'])==0)
outcome_sum=np.asarray(b['hit'])+np.asarray(b['miss'])+np.asarray(b['no'])
valid &= outcome_sum==1
```

iii. Early/stim exclusion follows the paper; ignore retention intentionally follows the decoder specification. The extra exclusivity check is presented as validation of technically valid outcomes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected probe's `obj/clu` fields `trialtm`, `trial`, and `quality`, plus `bp.ev.goCue` for alignment.

ii.
```python
for rt,ri,rq in zip(g['trialtm'],g['trial'],g['quality']):
    tm=...; tr=...-1
    z=tm[tr==old]-go[old]
```

iii. The notes identify these as the same fields used by `alignSpikes` and the reference cluster-curation functions.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed per trial into 10 ms bins, divided by `DT` to Hz, and processed by a causal half-`gausswin(15)` convolution with reflected prepending. No normalization or baseline subtraction is applied.

ii.
```python
mat[pos[old]]=np.histogram(z,EDGES)[0]/DT
sm=np.stack([causal_smooth(x) for x in raw],axis=0)
k[:len(k)//2]=0.0; k/=k.sum()
```

iii. The AI says this exactly translates Figure 8's `mySmooth(x,15)` after correcting an initial mistaken sigma interpretation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only labels in `{poor,fair,good,great,excellent,multi}` are admitted, then units with a reference-style all-trial mean smoothed PSTH strictly above 1 Hz are kept. Sessions must retain at least ten units.

ii.
```python
VALID_QUAL={'poor','fair','good','great','excellent','multi'}
if q not in VALID_QUAL: continue
use=np.asarray(rates)>1.0
if n[0].shape[0]<10: raise ValueError(...)
```

iii. This is justified from manual single/multiunit curation and the paper's >1 Hz/session-size criteria; the AI notes 515 retained versus 522 reported.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time has that trial's go-cue time subtracted before histogramming.

ii.
```python
z=tm[tr==old]-go[old]
mat[pos[old]]=np.histogram(z,EDGES)[0]/DT
```

iii. The AI cites `alignSpikes` as specifying this direct subtraction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. It uses 10 ms bins from −3.0 to +2.5 s (550 bins). Raw spikes are binned; video streams are nearest-frame resampled to the same centers.

ii.
```python
TMIN,TMAX,DT=-3.0,2.5,0.01
EDGES=np.arange(TMIN,TMAX+DT/2,DT)
```

iii. The AI chose Figure 8 context-analysis timing over the repository's generic 5 ms, −2.5-to-2.5 s defaults.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived from the configured bin edges/centers rather than a varying raw field; raw `goCue` defines the zero used for all streams.

ii.
```python
CENTERS=(EDGES[:-1]+EDGES[1:])/2
inputs.append(CENTERS[None,:].astype(np.float32))
```

iii. The notes describe signed bin centers as the natural continuous decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Adjacent edge midpoints are computed once, cast to float32, and copied for every trial; values run from −2.995 to +2.495 s.

ii.
```python
CENTERS=(EDGES[:-1]+EDGES[1:])/2
inputs.append(CENTERS[None,:].astype(np.float32))
```

iii. The AI reports independently comparing every stored input to analytically generated centers with `np.allclose`.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the exact centers of the edges used to histogram go-cue-subtracted spikes, so columns correspond directly.

ii.
```python
mat[pos[old]]=np.histogram(z,EDGES)[0]/DT
inputs.append(CENTERS[None,:].astype(np.float32))
```

iii. The common grid was explicitly selected to keep all decoder streams aligned.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.ev.lickL`, `bp.ev.lickR`, and `bp.ev.goCue`, selecting actual post-go lick events rather than instructed side/outcome.

ii.
```python
d['lickL']=cell_arrays(f,ev['lickL']); d['lickR']=cell_arrays(f,ev['lickR'])
l=first_post_go(b['lickL'][old],b['go'][old])
```

iii. The AI argues that emitted lick behavior is preferable to substituting the target direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. It finds the first finite left and right event at/after go cue; earlier left gives class 0, earlier right class 1, and ties/no event class 2, then repeats the scalar over time.

ii.
```python
x=x[np.isfinite(x)&(x>=go)]
lick.append(0 if l<r else (1 if r<l else 2))
np.full(CENTERS.size,lick[i])
```

iii. The notes say this directly measures lick direction and honestly represents no post-go response as `none`.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived directly from per-trial `bp.autowater`.

ii.
```python
d={k:get(k) for k in [...,'autowater']}
context.append(1 if b['autowater'][old] else 0)
```

iii. Figure code equates non-autowater/AFC with DR and autowater/AW with WC.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It maps false to DR class 0 and true to WC class 1, repeating the trial scalar across bins.

ii.
```python
context.append(1 if b['autowater'][old] else 0)
np.full(CENTERS.size,context[i])
```

iii. The AI describes this as a direct relabeling consistent with the paper's task terminology.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It loads `bp.miss`, `bp.hit`, and `bp.no` (with `no` also participating in the exclusivity filter).

ii.
```python
d={k:get(k) for k in ['L','R','hit','miss','no',...]}
outcome_sum=b['hit']+b['miss']+b['no']
```

iii. These are the mutually exclusive behavioral outcome flags used by the reference conditions.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss maps to incorrect 0, hit to correct 1, otherwise to ignore 2, and the scalar is repeated over time.

ii.
```python
outcome.append(0 if b['miss'][old] else (1 if b['hit'][old] else 2))
```

iii. The mapping follows the requested class order; ignore trials are retained specifically because the decoder requests that class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses camera 0 `obj.traj.ts`, `frameTimes`, `featNames`, and every feature name containing `tongue`, plus go cue.

ii.
```python
arr=np.asarray(f[tsrefs[old]],float)
names=[x.lower() for x in feat_names(f,nrefs[old])]
tongue=speed_for(0,['tongue'])
```

iii. The notes characterize camera 0 as the reference side-camera tongue view and use likelihood/NaNs for visibility.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For matching landmarks it differentiates unsmoothed x/y using frame times, takes Euclidean speed, masks coordinates/likelihood ≤0.9, nearest-resamples each speed, and averages all available matching features.

ii.
```python
dx=np.gradient(x,ft);dy=np.gradient(y,ft); sp=np.hypot(dx,dy);sp[~visible]=np.nan
speeds.append(nearest_bin(ft,sp))
return np.divide(np.nansum(z,axis=0),count,...)
```

iii. The AI says this preserves unavailable bins and applies the paper's confidence cutoff, though it does not implement the reference solution's per-run smoothing or two-camera tongue combination.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One median is computed over all finite tongue samples in a session: below is 0, at/above is 1, and nonfinite is 2.

ii.
```python
med=float(np.nanpercentile(a,50))
y=np.full(a.shape,2,np.int64); y[valid & (a<med)]=0; y[valid & (a>=med)]=1
```

iii. This directly follows the prompt's per-session 50th-percentile rule and retains missing visibility.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI subtracts go cue directly from `frameTimes` and nearest-resamples onto neural bin centers. It does not apply the session video/behavior clock offset used by the reference.

ii.
```python
ft=np.asarray(f[ftrefs[old]],float).squeeze()-go[old]
speeds.append(nearest_bin(ft,sp))
```

iii. The notes claim native frame/event timing can be aligned directly and report spot checks, but do not justify omitting `findVideoOffset`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses camera 1 trajectory arrays and all feature names containing `paw`, with their frame times and confidence channel.

ii.
```python
paw=speed_for(1,['paw'])
```

iii. The AI identifies camera 1 as the bottom view containing paw landmarks.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. It applies the same unsmoothed gradient, speed magnitude, visibility mask, nearest resampling, and averaging across every paw landmark as for tongue.

ii.
```python
inds=[i for i,n in enumerate(names) if any(p in n for p in patterns)]
dx=np.gradient(x,ft);dy=np.gradient(y,ft); sp=np.hypot(dx,dy)
```

iii. The notes say native coordinates/confidence should determine speed and absence; unlike the reference, it averages both paws rather than choosing reliable `top_paw`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Finite session-wide values are split at their median into 0/1 and nonfinite bins become 2.

ii.
```python
pv,pmed=discretize_session(paw)
```

iii. This follows the explicit per-session percentile requirement.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Camera-1 frame times have go cue directly subtracted and the values are nearest-resampled to the common centers, without correcting the video clock offset.

ii.
```python
ft=np.asarray(f[ftrefs[old]],float).squeeze()-go[old]
return nearest_bin(ft,sp)
```

iii. The AI relies on the shared target grid and native timestamps, but does not address the reference offset correction.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It loads `me.data[trial]` from `motionEnergy_<subject>_<date>.mat` and pairs it with camera-0 frame times/go cue.

ii.
```python
me=loadmat(p,simplify_cells=True)['me']; data=np.atleast_1d(me['data'])
v=np.asarray(data[old],float).squeeze()
```

iii. The AI notes motion energy is already computed upstream according to the paper, so it does not recompute pixels.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. It truncates value/time vectors to their shorter length and nearest-frame resamples. Missing files or out-of-range trials yield all NaNs. It assumes a single `me['data']` wrapper.

ii.
```python
if not p.exists(): return [np.full(CENTERS.size,np.nan) for _ in trial_ids]
n=min(len(v),len(t)); out.append(nearest_bin(t[:n],v[:n]))
```

iii. The AI says upstream spatial processing should not be repeated and unavailable video should remain class 2.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. All finite motion-energy samples in each session are split at the session median; missing/nonfinite samples are class 2.

ii.
```python
mv,mmed=discretize_session(motion)
```

iii. It intentionally replaces the stored manual movement threshold with the prompt's required 50th percentile.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It reuses camera-0 frame times already reduced by go cue, truncates to the motion trace length, and nearest-resamples to neural centers; again no video clock offset is applied.

ii.
```python
motion_continuous(subject,date,ids,b['go'],fts)
out.append(nearest_bin(t[:n],v[:n]))
```

iii. The AI treats motion energy as frame-synchronous with camera 0 and the common decoder grid.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/nonfinite video stays NaN until discretization then becomes class 2; absent motion files/out-of-range trials become all-NaN; unequal motion/time lengths are truncated; too-short streams return NaNs; invalid behavioral trials are dropped. No interpolation outside source support is done.

ii.
```python
if t.size<2:return out
inside=(CENTERS>=t[0])&(CENTERS<=t[-1]); out[inside]=v[pick[inside]]
n=min(len(v),len(t))
y=np.full(a.shape,2,np.int64)
```

iii. The AI explicitly prefers preserving unavailable categories over fabricating values and reports validation of shapes and finite ranges.

## 11-a. What are the most time-consuming steps of the code?

i. Per the AI's printed timings and notes, session HDF5 decoding plus per-unit spike histogramming/smoothing and per-trial video reference decoding/resampling dominate; each selected session took about 2–4 seconds.

ii.
```python
for rt,ri,rq in zip(...): ... np.histogram(...)
for old in trial_ids:
    arr=np.asarray(f[tsrefs[old]],float)
```

iii. The notes emphasize direct HDF5 decoding and vectorized stacking to keep the complete conversion short.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Loops remain over clusters, each cluster's unique trials, retained video trials/cameras/features, neural rows in smoothing, output trials, and repeated low-FR cluster traversal. Some rectangular output assembly and spike counting could be vectorized; ragged video references are less amenable.

ii.
```python
for old in np.unique(tr): ... np.histogram(z,EDGES)
for i,row in enumerate(x): ... np.convolve(...)
for i in range(len(ids)): ...
```

iii. The AI claims per-neuron histograms/vectorized stacking and trial dictionaries already avoid repeated index searches; it does not specifically discuss all remaining loops.

## 11-c. What processing does the code repeat multiple times?

i. Curated clusters are traversed twice: once to build trial firing arrays and again to compute the low-rate PSTH. `CENTERS.astype(np.float32)` and scalar full-length arrays are rebuilt for every trial; `nearest_bin` sorting/searching repeats for each landmark/stream.

ii.
```python
for rt,ri,rq in zip(...): # build mats
...
for rt,ri,rq in zip(...): # compute rates
...
inputs.append(CENTERS[None,:].astype(np.float32))
```

iii. The AI justifies the separate rate pass as matching reference curation, while noting stacking/index maps reduce other repetition.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes/stores quality counts, continuous-stream medians, timing metadata, and optional figures; it also smooths firing arrays for units later removed and loads full trajectory arrays although only tongue/paw channels are used. These intermediates/plots do not enter decoder tensors.

ii.
```python
sm=np.stack([causal_smooth(x) for x in raw],axis=0)
return sm[:,use,:],...
info={'quality_counts':..., 'thresholds':..., 'seconds':...}
```

iii. The extra metadata and optional plots are intended for auditability and sanity checking; the pre-filter smoothing is an implementation convenience rather than a documented scientific need.
