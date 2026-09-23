# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI only loads `data_structure_*.mat` files from `/app/data/Ephys_Behavior`, not from both electrophysiology folders. It discovers files by `glob`, then keeps only sessions whose `(mouse, day)` pair appears in the hard-coded `PROBES` map. Session `.mat` files are loaded with `mat73.loadmat`, and motion energy is loaded separately from `motionEnergy_<mouse>_<day>.mat` with `scipy.io.loadmat`.

ii.
```python
ROOT='/app/data/Ephys_Behavior'
...
PROBES={
 ('EKH3','2021-08-11'):[2], ('JEB6','2021-04-18'):[2],
 ...
 ('JEB19','2023-04-20'):[1], ('JEB19','2023-04-21'):[1]}
...
files=[]
for f in glob.glob(ROOT+'/data_structure_*.mat'):
    base=os.path.basename(f)[15:-4]; mouse,day=base.split('_',1)
    if (mouse,day) in PROBES: files.append((mouse,day,f))
...
o=mat73.loadmat(f)['obj']
...
mef=os.path.join(ROOT,f'motionEnergy_{mouse}_{day}.mat'); me=None
if os.path.exists(mef): me=loadmat(mef,squeeze_me=True,struct_as_record=False)['me'].data
```

iii. The trajectory says the AI decided Figure 8's six two-context mice and their 11 released sessions were the "authoritative subset" and planned to use "the 11 Figure-8 two-context sessions" rather than the full released dataset (steps 26-28). It also says this was done to "keep runtime and pickle size manageable" (step 28).

## 1-b. How are the data split into subjects?

i. The AI treats the filename prefix before the first underscore as the subject id (`mouse`). It also hard-codes the allowed subject set as the six Figure 8 mice and stores `subject_idx` by looking up each session's mouse in `sorted(MICE)`.

ii.
```python
MICE={'JEB6','JEB7','EKH3','JGR2','JGR3','JEB19'}
...
base=os.path.basename(f)[15:-4]; mouse,day=base.split('_',1)
...
subjects=sorted(MICE); subject_idx=[]
...
subject_idx.append(subjects.index(mouse))
```

iii. The trajectory repeatedly refers to "Figure 8's six two-context mice" and says the converter should use that subset only (steps 26-28). There is no additional subject-splitting logic beyond parsing the filename prefix.

## 1-c. How are the data split into sessions?

i. The AI treats each selected `data_structure_<mouse>_<day>.mat` file as one session. Session membership is determined entirely by whether `(mouse, day)` is present in the hard-coded `PROBES` dictionary, and every kept file becomes one entry in `neural`, `input`, and `output`.

ii.
```python
for f in glob.glob(ROOT+'/data_structure_*.mat'):
    base=os.path.basename(f)[15:-4]; mouse,day=base.split('_',1)
    if (mouse,day) in PROBES: files.append((mouse,day,f))
files.sort()
...
for mouse,day,f in files:
    ...
    neural.append(ns);inputs.append(ins);outputs.append(outs)
```

iii. The trajectory says the session subset should follow the sessions explicitly loaded by the repository's Figure 8 code, which the AI interpreted as 11 released sessions from six mice (steps 26-28).

## 1-d. How are the data split into trials?

i. Trials are indexed directly from the Bpod arrays in `o['bp']`. The AI reads `Ntrials`, builds a kept-trial index array from `~early`, and then iterates over those trial indices. Spikes are assigned to trials through `clu.trial`, and video/motion-energy streams are read trial-by-trial with the same trial index.

ii.
```python
b=o['bp']; n=int(b['Ntrials'])
early=np.asarray(b['early']).astype(bool); keep=np.flatnonzero(~early)
go=np.asarray(b['ev']['goCue'],float).ravel()
...
for tr,tm in zip(c['trial'],c['trialtm']):
    units.append((np.asarray(tr,int).ravel()-1,np.asarray(tm,float).ravel()))
...
for t in keep:
    tv,tvok,pv,pvok,ft=trial_kin(o['traj'],int(t),go[t])
```

iii. The trajectory notes that cluster spike times are stored per unit as `trial` and `trialtm`, and that video coordinates and frame times are available per trial, so the AI chose to use those stored trial assignments directly rather than reconstructing trial boundaries (step 28).

## 1-e. How are trials filtered based on quality controls?

i. The only trial-level filter is removal of early-lick trials. All non-early trials are kept, including ignore/no-response trials. The AI does not remove photostimulation trials or trials that continue past the end of electrophysiology recording.

ii.
```python
early=np.asarray(b['early']).astype(bool); keep=np.flatnonzero(~early)
...
# ... use all non-early trials ...
for j,t in enumerate(keep):
    ...
```

iii. The docstring says "Early-lick trials are omitted as in the paper, while no-response trials are retained because ignore is a required class." The trajectory repeats that it would "exclude early trials but retain hit/miss/no trials" because ignore was needed for the decoder task (steps 28 and 32).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the selected probe entries in `o['clu']`, specifically each cluster's `trial` and `trialtm` arrays, together with `bp.ev.goCue` to realign spike times. The code does not use the raw cluster `quality` labels.

ii.
```python
go=np.asarray(b['ev']['goCue'],float).ravel()
...
for p in PROBES[(mouse,day)]:
    c=o['clu'][p-1]
    for tr,tm in zip(c['trial'],c['trialtm']):
        units.append((np.asarray(tr,int).ravel()-1,np.asarray(tm,float).ravel()))
...
h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT
```

iii. The trajectory says the important neural raw variables were `trial`, `trialtm`, and per-trial `bp.ev.goCue`, because `trialtm` is relative to trial start and therefore must be shifted by go cue for this task (step 28).

## 2-b. How is the `neural` data processed?

i. For each kept trial of each unit, spikes are histogrammed into 5 ms bins over `[-2.5, 2.5]`, divided by `DT` to convert to Hz, and then smoothed with a causal 15-sample Gaussian kernel whose first 7 taps are zeroed. Rates are stored as `float32`.

ii.
```python
DT=.005; T0=-2.5; T1=2.5
EDGES=np.arange(T0,T1+DT/2,DT)
...
def smooth_rates(x):
    # MATLAB mySmooth: gausswin(15), first floor(N/2) samples set to zero.
    k=gaussian(15, std=(15-1)/(2*2.5), sym=True); k[:7]=0; k/=k.sum()
    return np.apply_along_axis(lambda z:np.convolve(z,k,mode='same'),1,x)
...
h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT
rates[ui,j]=smooth_rates(h[None,:])[0]
```

iii. The trajectory says the converter would reproduce "5 ms binning, causal Gaussian smoothing, and >1 Hz filter" from the repository, and later claims the final script "reproduces the repository's go-cue alignment, 5 ms bins, causal Gaussian smoothing" (steps 26, 28, and 32).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural quality-control filter applied is a mean firing-rate threshold: units whose mean binned-and-smoothed firing rate over kept trials and all bins is not greater than 1 Hz are dropped. No manual cluster-quality labels are used.

ii.
```python
use=rates.mean(axis=(1,2))>1.0; rates=rates[use]
```

iii. The trajectory says the AI inspected the repository's low-firing-rate rule and concluded that units should pass a `>1 Hz` mean-rate criterion. It does not mention using cluster quality labels, and the final summary only mentions the `>1 Hz` filter (steps 27, 28, and 32).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to go cue by subtracting the go-cue time of the corresponding trial before histogramming into the common bin edges.

ii.
```python
go=np.asarray(b['ev']['goCue'],float).ravel()
...
h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT
```

iii. The trajectory explicitly says `trialtm` is relative to trial start rather than go cue, so "for this task we must subtract each trial's `bp.ev.goCue`" (step 28).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 5 ms bins over a fixed `[-2.5, 2.5]` window, giving 1000 time bins. No later temporal rebinning is applied.

ii.
```python
DT=.005; T0=-2.5; T1=2.5
EDGES=np.arange(T0,T1+DT/2,DT); TIME=EDGES[:-1]+DT/2
```

iii. The trajectory repeatedly states that the converter should use "5 ms bins over `[-2.5, 2.5)`" and a shared 1000-point grid (steps 27, 28, and 32).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a raw data field; it is constructed as the fixed bin-center time axis for the common go-cue-aligned window. It is therefore only indirectly tied to raw `goCue`, through the choice of alignment event.

ii.
```python
DT=.005; T0=-2.5; T1=2.5
EDGES=np.arange(T0,T1+DT/2,DT); TIME=EDGES[:-1]+DT/2
...
ins.append(TIME[None,:].astype(np.float32))
```

iii. The trajectory says the converter would use the common go-cue-centered grid itself as the decoder input and keep everything aligned to that 1000-point axis (steps 27 and 28).

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes the input by taking bin centers from the fixed edge vector and repeating that 1-by-time array for every trial.

ii.
```python
EDGES=np.arange(T0,T1+DT/2,DT); TIME=EDGES[:-1]+DT/2
...
ins.append(TIME[None,:].astype(np.float32))
```

iii. No deeper justification is given beyond using the same fixed grid as the aligned neural data (steps 27-28).

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the same go-cue-aligned bin grid used to histogram spikes, so its timepoints correspond directly to the neural bins.

ii.
```python
EDGES=np.arange(T0,T1+DT/2,DT); TIME=EDGES[:-1]+DT/2
...
h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT
...
ins.append(TIME[None,:].astype(np.float32))
```

iii. The trajectory says the converter would "interpolate to the common go-cue-centered grid" and use that same grid for neural data and decoder inputs (steps 26-28).

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the per-trial Bpod flags `L`, `R`, `hit`, `miss`, and `no`.

ii.
```python
L=np.asarray(b['L']).astype(bool); R=np.asarray(b['R']).astype(bool)
hit=np.asarray(b['hit']).astype(bool); miss=np.asarray(b['miss']).astype(bool)
no=np.asarray(b['no']).astype(bool)
```

iii. The trajectory says the AI would "map lick direction from actual outcome" while retaining hit/miss/no trials (step 28).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI encodes lick direction per trial as `2` for no response / ignore, otherwise `0` for left and `1` for right. On hit trials it uses the instructed side; on miss trials it flips the instructed side.

ii.
```python
# Actual lick: instructed direction on hit, opposite on miss, none on no/ignore.
lick=2 if no[t] or (not hit[t] and not miss[t]) else (
    0 if (hit[t] and L[t]) or (miss[t] and R[t]) else 1
)
```

iii. The docstring and trajectory emphasize keeping ignore as an explicit class because the decoder task requires it (docstring, steps 28 and 32).

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `bp.autowater` flag.

ii.
```python
aw=np.asarray(b['autowater']).astype(bool)
```

iii. The trajectory first considered block-based context inference, then concludes that repository Figure 8 code "confirms that `bp.autowater` is directly used as the WC-versus-DR trial label" (step 27).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI relabels `autowater=True` as `WC` code `0` and `autowater=False` as `DR` code `1`, and then repeats that value across all time bins in the trial.

ii.
```python
const=np.vstack([
    np.full(TIME.size,lick),
    np.full(TIME.size,int(not aw[t])),
    np.full(TIME.size,outcome)
])
...
'output_values':[['left','right','none'],['WC','DR'],['incorrect','correct','ignore'], ...]
```

iii. The trajectory notes an initial encoding mismatch and says it must be corrected so that "WC is 0 and DR is 1 to match the declared `['WC','DR']` value order" (steps 29-30).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the per-trial Bpod flags `hit`, `miss`, and `no`.

ii.
```python
hit=np.asarray(b['hit']).astype(bool); miss=np.asarray(b['miss']).astype(bool)
no=np.asarray(b['no']).astype(bool)
```

iii. The trajectory says the AI would "retain hit/miss/no trials" and map trial outcomes directly from those behavioral flags (step 28).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is encoded per trial as `2` for no response / ignore, `1` for hit / correct, and `0` for miss / incorrect, then repeated across all time bins.

ii.
```python
outcome=2 if no[t] or (not hit[t] and not miss[t]) else (1 if hit[t] else 0)
```

iii. The trajectory justifies retaining ignore trials because the decoder task requires an explicit ignore class rather than dropping them (steps 28 and 32).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `o['traj']`, but only from one camera entry (`traj[1]`). The AI uses the first four landmark indices in that camera's `ts` array, plus that camera's `frameTimes`, and aligns them with per-trial `goCue`. It does not use feature names, the second camera, or the session video-offset fields in `sglx`.

ii.
```python
def trial_kin(traj, trial, go):
    # Side camera (cam 1): first four tongue landmarks, top/bottom paw landmarks.
    ...
    cam=traj[1]; ts=np.asarray(cam['ts'][trial],float)
    ft=np.asarray(cam['frameTimes'][trial],float).ravel()-go
    ...
    tongue,tvis=speed([0,1,2,3])
```

iii. The trajectory says the AI planned to "compute tongue/paw speed from camera-1 landmarks and visibility" and align them using frame times relative to go cue (step 28).

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each frame, the AI marks tongue visibility if any chosen landmark has likelihood `>= 0.9`, averages visible landmark coordinates into a centroid, computes speed as `sqrt(gradient(x)^2 + gradient(y)^2)` in frame units, sets non-visible frames to `NaN`, and then linearly interpolates both speed and visibility onto the common neural time grid. No Gaussian smoothing, per-run processing, per-time-step derivative, or cross-camera normalization is applied.

ii.
```python
xy=ts[:,0:2,ids]; lk=ts[:,2,ids] if ts.shape[1]>2 else np.ones((len(ft),len(ids)))
vis=np.any(lk>=.9,axis=1)
xy=np.where((lk>=.9)[:,None,:],xy,np.nan)
pos=np.nanmean(xy,axis=2)
# MATLAB findVelocity uses gradient per frame (pixel/frame), not /dt.
vx=np.gradient(pos[:,0]); vy=np.gradient(pos[:,1]); sp=np.hypot(vx,vy)
sp[~vis]=np.nan
return interp_nearest(ft,sp,TIME), interp_nearest(ft,vis.astype(float),TIME,0)>=.5
```

iii. The trajectory says the AI chose explicit visibility classes and intended to "preserve invisibility rather than paper's zero fill"; it also states it would "derive velocity classes from aligned video landmarks with explicit visibility classes" (step 27-29 and the in-code comment on line 53).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI computes a single session-wide median over all visible tongue-velocity samples, then assigns category `0` below median, `1` at or above median, and `2` where the tongue is not visible.

ii.
```python
def classes(x,vis):
    th=np.nanpercentile(x[vis],50) if np.any(vis) else np.nan
    z=np.full(x.shape,2,np.int8); z[vis]=(x[vis]>=th).astype(np.int8); return z,float(th)
tc,tth=classes(tongue,tvis)
```

iii. The trajectory says movement variables would be "discretized by per-session medians" with explicit visibility classes (steps 26-28).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are shifted only by subtracting the per-trial go cue and then interpolated directly to the common `TIME` vector. No session-wide video-to-behavior clock offset is estimated or applied.

ii.
```python
cam=traj[1]; ts=np.asarray(cam['ts'][trial],float)
ft=np.asarray(cam['frameTimes'][trial],float).ravel()-go
...
return interp_nearest(ft,sp,TIME), interp_nearest(ft,vis.astype(float),TIME,0)>=.5
```

iii. The trajectory says the available video coordinates and frame times "can be shifted by go cue and interpolated" to the common grid, and that the AI would align motion variables using camera frame times relative to go cue (step 28).

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the same single-camera `o['traj'][1]` stream as tongue velocity, using landmark indices `[4, 5]` from `ts`, along with that camera's `frameTimes` and per-trial go cue.

ii.
```python
cam=traj[1]
...
tongue,tvis=speed([0,1,2,3]); paw,pvis=speed([4,5])
```

iii. The trajectory says the AI would compute "tongue/paw speed from camera-1 landmarks and visibility" (step 28).

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw velocity uses the same centroid, likelihood-thresholding, framewise gradient, NaN-masking, and interpolation procedure as tongue velocity, with a different landmark index set.

ii.
```python
def speed(ids):
    ids=[i for i in ids if i<ts.shape[2]]
    xy=ts[:,0:2,ids]; lk=ts[:,2,ids] if ts.shape[1]>2 else np.ones((len(ft),len(ids)))
    vis=np.any(lk>=.9,axis=1)
    xy=np.where((lk>=.9)[:,None,:],xy,np.nan)
    pos=np.nanmean(xy,axis=2)
    vx=np.gradient(pos[:,0]); vy=np.gradient(pos[:,1]); sp=np.hypot(vx,vy)
    sp[~vis]=np.nan
    return interp_nearest(ft,sp,TIME), interp_nearest(ft,vis.astype(float),TIME,0)>=.5
```

iii. The trajectory gives the same justification as tongue velocity: use aligned camera-1 landmarks, explicit visibility classes, and per-session median discretization (step 28).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI uses the same `classes` helper as for tongue velocity: category `0` below the session median over visible paw samples, `1` at or above median, and `2` when the paw is not visible.

ii.
```python
pc,pth=classes(paw,pvis)
```

iii. The trajectory says paw velocity would also be discretized by per-session medians with an explicit not-visible class (steps 26-28).

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw frame times are aligned by subtracting go cue only and interpolating to the shared `TIME` grid. No video offset is applied.

ii.
```python
ft=np.asarray(cam['frameTimes'][trial],float).ravel()-go
...
return interp_nearest(ft,sp,TIME), interp_nearest(ft,vis.astype(float),TIME,0)>=.5
```

iii. The trajectory says paw velocity would be aligned using camera frame times relative to go cue and then projected to the common grid (step 28).

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from a separate `motionEnergy_<mouse>_<day>.mat` file, specifically `loadmat(... )['me'].data`, and then indexed trial-by-trial. It is paired with the frame times returned from `trial_kin`.

ii.
```python
mef=os.path.join(ROOT,f'motionEnergy_{mouse}_{day}.mat'); me=None
if os.path.exists(mef): me=loadmat(mef,squeeze_me=True,struct_as_record=False)['me'].data
...
mv=np.asarray(me[t],float).ravel()
```

iii. The trajectory says the converter would "align motion energy using camera frame times" and treat the standalone motion-energy files as part of the kept Figure 8 sessions (step 28).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI does not smooth or differentiate motion energy. It truncates the motion-energy trace and frame-time vector to the same length, then interpolates motion-energy values directly onto the common `TIME` grid.

ii.
```python
mv=np.asarray(me[t],float).ravel(); m=min(len(mv),len(ft))
z=interp_nearest(ft[:m],mv[:m],TIME); mes.append(z);mvis.append(np.isfinite(z))
```

iii. The trajectory says motion energy would be aligned using frame times and then discretized by session median on the common grid (step 28). No separate justification for interpolation rather than bin averaging is given.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI thresholds motion energy exactly like tongue and paw velocity: category `0` below the session median over valid samples, `1` at or above median, and `2` when no video is available.

ii.
```python
mc,mth=classes(mes,mvis)
```

iii. The trajectory says motion energy, like the other movement outputs, would use per-session median discretization with an explicit missing-video class (step 28).

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by taking the frame-time vector returned from `trial_kin`, subtracting go cue there, and interpolating the motion-energy trace onto the common `TIME` axis. No session-wide video offset is applied.

ii.
```python
tv,tvok,pv,pvok,ft=trial_kin(o['traj'],int(t),go[t])
...
z=interp_nearest(ft[:m],mv[:m],TIME)
```

iii. The trajectory says motion energy would be aligned "using camera frame times" and placed on the same common grid as neural data (step 28).

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed video data are not dropped; instead they are converted to all-`NaN` traces with all-false visibility masks, which later become category `2` ("not visible" or "no video"). If motion-energy and frame-time vectors differ in length, the AI truncates both to the shorter length. Interpolation outside the valid support returns `NaN`.

ii.
```python
if len(traj)<2 or trial>=len(traj[1]['ts']):
    return np.full(TIME.size,np.nan),np.zeros(TIME.size,bool),np.full(TIME.size,np.nan),np.zeros(TIME.size,bool),None
...
if ts.ndim!=3 or ts.shape[0]!=ft.size:
    return np.full(TIME.size,np.nan),np.zeros(TIME.size,bool),np.full(TIME.size,np.nan),np.zeros(TIME.size,bool),ft
...
if ok.sum()<2:return np.full(target.shape,fill,float)
...
mv=np.asarray(me[t],float).ravel(); m=min(len(mv),len(ft))
z=interp_nearest(ft[:m],mv[:m],TIME)
```

iii. The trajectory calls the `Mean of empty slice` warning benign, says invisible samples are intentionally kept as an explicit not-visible class, and emphasizes "preserve invisibility rather than paper's zero fill" (steps 29 and 32 plus the in-code comment on line 53).

## 11-a. What are the most time-consuming steps of the code?

i. The likely dominant costs are the nested Python loops that build neural rates (`unit -> trial -> histogram -> smoothing`) and the per-trial kinematic/motion-energy interpolation loops. The code loads each session once, but most expensive computation is done after loading in Python rather than vectorized library calls.

ii.
```python
for ui,(tr,tm) in enumerate(units):
    for t in np.unique(tr):
        j=keepmap.get(int(t));
        if j is None or not np.isfinite(go[t]): continue
        h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT
        rates[ui,j]=smooth_rates(h[None,:])[0]
...
for t in keep:
    tv,tvok,pv,pvok,ft=trial_kin(o['traj'],int(t),go[t]); ...
```

iii. The trajectory says the AI restricted the dataset "to keep runtime and pickle size manageable" (step 28), but it does not give a deeper profiling analysis.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The unit-by-trial spike loop could have been replaced with a more vectorized counting strategy across all spikes or all clusters. The per-trial loops over `keep` that call `trial_kin` and separately interpolate motion energy could also be partially vectorized or batched. `smooth_rates` also repeatedly invokes `np.apply_along_axis` and `np.convolve` on one trial vector at a time.

ii.
```python
for ui,(tr,tm) in enumerate(units):
    for t in np.unique(tr):
        ...
        h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT
        rates[ui,j]=smooth_rates(h[None,:])[0]
...
for t in keep:
    tv,tvok,pv,pvok,ft=trial_kin(o['traj'],int(t),go[t]); ...
```

iii. The trajectory only mentions runtime manageability in broad terms and does not explicitly justify leaving these loops unvectorized (step 28).

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly converts the same `TIME[None, :]` input array for every trial, repeatedly calls `smooth_rates` separately for each unit/trial histogram, and repeatedly interpolates frame-level traces to the neural grid trial by trial. It also computes tongue and paw visibility/interpolation independently even though both come from the same camera/frame-time stream.

ii.
```python
rates[ui,j]=smooth_rates(h[None,:])[0]
...
for t in keep:
    tv,tvok,pv,pvok,ft=trial_kin(o['traj'],int(t),go[t]); ...
...
ins.append(TIME[None,:].astype(np.float32))
```

iii. The trajectory does not explicitly discuss repeated computations; it only states the converter should stay manageable in runtime and file size (step 28).

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores some values only for metadata or intermediate masking rather than for decoder inputs/outputs themselves: `n_trials_raw`, `n_units_raw`, and session median thresholds are recorded in `session_info`; `trial_kin` returns raw frame times mainly to support later motion-energy interpolation; and the code repeatedly constructs per-trial copies of the same input time axis. It also imports `convolve1d` but never uses it.

ii.
```python
from scipy.ndimage import convolve1d
...
session_info.append({
    'subject':mouse,'date':day,'n_trials_raw':n,'n_trials':len(keep),
    'n_units_raw':len(units),'n_units':int(use.sum()),
    'tongue_median':tth,'paw_median':pth,'motion_energy_median':mth
})
...
ins.append(TIME[None,:].astype(np.float32))
```

iii. The trajectory gives no explicit justification for these extra allocations or metadata computations beyond wanting compact validation summaries and manageable runtime/size (steps 28-32).
