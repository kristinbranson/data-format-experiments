# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 12 Figure 8 WC/DR sessions and one selected probe per session, all from `/app/data/Ephys_Behavior`. It loads each HDF5 MATLAB file with `mat73`; it does not load the other eligible fixed-delay sessions, any randomized-delay sessions, separate motion-energy files, or MATLAB v5 files.

ii.
```python
DATA='/app/data/Ephys_Behavior'
SESS=[('JEB6','2021-04-18',2), ... ('JEB19','2023-04-18',1)]
o=mat73.loadmat(path)['obj']
```

iii. The agent said the 12 sessions and selected probes were the exact set used by the paper's released Figure 8 pipeline. It prioritized that analysis subset over converting all relevant released sessions.

## 1-b. How are the data split into subjects?

i. Subject IDs are the hard-coded animal strings in `SESS`. Unique IDs are appended in encounter order, and each retained session gets the corresponding index. This produces seven subjects.

ii.
```python
for anm,date,probe in SESS:
    if anm not in D['subjects']: D['subjects'].append(anm)
    D['subject_idx'].append(D['subjects'].index(anm))
```

iii. The trajectory treated the animal component of each selected session as the subject identifier and reported seven subjects after validation.

## 1-c. How are the data split into sessions?

i. Each tuple in `SESS` maps to one `data_structure_<animal>_<date>.mat` and becomes one entry in `neural`, `input`, and `output`, unless it has fewer than two trials or no neurons. Exactly 12 sessions were retained.

ii.
```python
p=f'{DATA}/data_structure_{anm}_{date}.mat'
neu,inp,out,thr=process(p,probe)
if len(neu)<2 or not neu or neu[0].shape[0]==0: continue
```

iii. The agent justified session boundaries and selection as following the Figure 8 session/probe list.

## 1-d. How are the data split into trials?

i. Trial indices come directly from Bpod arrays. `valid` contains eligible zero-based Bpod trial indices; spikes are assigned using the cluster's one-based `trial` field, and camera arrays are indexed by the same trial index.

ii.
```python
valid=np.flatnonzero(np.isfinite(go)&(stim==0)&(early==0))
tri=arr(clu['trial'][u]).astype(int)-1
for t in valid:
    v=video_trial(o,t,go[t])
```

iii. The trajectory relied on the released per-trial Bpod, spike, and trajectory structures rather than reconstructing trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained when the go cue is finite, photostimulation is off, and there is no early lick. Incorrect and ignore trials are deliberately retained. There is no filter for trials after electrophysiology recording ended.

ii.
```python
valid=np.flatnonzero(np.isfinite(go)&(stim==0)&(early==0))
```

iii. The agent said non-stim/non-early filtering follows the paper, while all outcomes must remain because outcome is a requested decoder target.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected `obj.clu` probe's `trialtm`, `trial`, and `quality` fields, plus `bp.ev.goCue` for alignment.

ii.
```python
clu=o['clu'][probe-1] if isinstance(o['clu'],list) else o['clu']
st=arr(clu['trialtm'][u]); tri=arr(clu['trial'][u]).astype(int)-1
z=st[tri==t]-go[t]
```

iii. The agent identified the paper's selected probe and trial-relative spike times as the relevant Figure 8 inputs.

## 2-b. How is the `neural` data processed?

i. Per unit and trial, aligned spikes are histogrammed, divided by 0.005 s to obtain Hz, and convolved with a 15-sample one-sided Gaussian whose first seven coefficients are zero. No normalization or baseline subtraction is applied.

ii.
```python
k=np.exp(-.5*(2.5*n/(14/2))**2)
k[:7]=0; k/=k.sum()
return np.stack([np.convolve(row,k,'same') for row in x],axis=0)
neural[ii,jj]=matlab_smooth((np.histogram(z,edges)[0]/DT)[None,:])[0]
```

iii. The agent explicitly interpreted the released `mySmooth`/Figure 8 processing as a causal 15-bin Gaussian smoother.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It drops quality strings exactly equal to `garbage`, `noise`, `nan`, `none`, or empty, then retains units whose mean converted firing rate is strictly above 1 Hz.

ii.
```python
if q not in ('garbage','noise','nan','none',''): keep0.append(u)
unitkeep=np.nanmean(neural,axis=(0,2))>1
```

iii. The agent said `quality='all'` means all non-garbage curated clusters and cited the paper's >1 Hz general-analysis criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The trial's go-cue time is subtracted from each spike's `trialtm` before histogramming.

ii.
```python
z=st[tri==t]-go[t]
```

iii. The agent selected go-cue onset as required and regarded both quantities as being on the behavioral trial clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The nominal resolution is 5 ms. The agent creates 1001 centers from -2.5 through +2.5 s inclusive and histograms spikes using half-bin-offset edges, so it rebins spikes into 1001 5-ms bins.

ii.
```python
DT=.005; TMIN=-2.5; TMAX=2.5
TIME=np.arange(TMIN,TMAX+DT/2,DT)
edges=np.r_[TIME-DT/2,TIME[-1]+DT/2]
```

iii. It reasoned that MATLAB's inclusive colon expression and `getSeq` imply 1001 sample centers.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a synthetic common time axis, defined from constants rather than directly read from raw data; raw go cues determine alignment conceptually.

ii.
```python
TIME=np.arange(TMIN,TMAX+DT/2,DT)
inp.append(TIME[None,:].astype(np.float32))
```

iii. The agent used the requested go-cue-relative window and paper time step.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. `np.arange` constructs values from -2.5 to +2.5 s inclusive at 0.005-s increments; the same 1-by-1001 float32 array is appended for every trial.

ii.
```python
TIME=np.arange(TMIN,TMAX+DT/2,DT)
inp.append(TIME[None,:].astype(np.float32))
```

iii. The trajectory justified the inclusive endpoints by its interpretation of the MATLAB grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. `TIME` supplies both the input values and the centers used to create neural histogram edges, so indices correspond exactly within the agent's representation.

ii.
```python
edges=np.r_[TIME-DT/2,TIME[-1]+DT/2]
inp.append(TIME[None,:].astype(np.float32))
```

iii. The common grid was deliberately used for neural and behavioral streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.L`, `bp.R`, `bp.hit`, and `bp.miss`; `bp.no` is loaded but not used.

ii.
```python
L=arr(b['L']); R=arr(b['R'])
hit=arr(b['hit']); miss=arr(b['miss']); no=arr(b['no'])
```

iii. After inspecting output distributions, the agent corrected an earlier instructed-side encoding to represent actual response direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Ignore trials are class 2; hits use the instructed side; misses use the opposite side. Codes are left 0, right 1, none 2 and are repeated across time.

ii.
```python
if outcome==2: lick=2
elif hit[t]>0: lick=0 if L[t]>0 else 1
else: lick=1 if L[t]>0 else 0
```

iii. The trajectory states that actual lick direction is instructed side for hits, opposite for misses, and none for ignores.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived solely from `bp.autowater`.

ii.
```python
aw=arr(b['autowater'])
```

iii. The agent treated autowater as the released indicator distinguishing WC and DR trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater-positive trials become WC (0), otherwise DR (1), repeated across all time points.

ii.
```python
outs.append([lick,int(not (aw[t]>0)),outcome])
rows=[np.full(TIME.size,x,dtype=np.int16) for x in base]
```

iii. This direct Boolean recoding follows the requested class order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses `bp.hit` and `bp.miss`; `bp.no` is loaded but the fallback category is used instead.

ii.
```python
hit=arr(b['hit']); miss=arr(b['miss']); no=arr(b['no'])
```

iii. The agent regarded neither-hit-nor-miss as an ignore trial.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit maps to correct (1), miss to incorrect (0), otherwise ignore (2), repeated over time.

ii.
```python
outcome=1 if hit[t]>0 else (0 if miss[t]>0 else 2)
```

iii. Ignore trials were retained because outcome is a decoder target, even though correct-only paper panels exclude them.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses the side camera (`obj.traj[0]`) landmark named `tongue`: per-frame x/y coordinates, likelihood, and frame times. The bottom-camera `top_tongue` landmark is not used.

ii.
```python
for cam, feats in [(0,['tongue']),(1,['top_paw','bottom_paw'])]:
    tr=o['traj'][cam]; ts=np.asarray(trial_item(tr['ts'],i),float)
```

iii. The agent described the side-camera tongue landmark as the relevant Figure 8 feature and preserved low-confidence tracking as missing.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Coordinates with likelihood below 0.9 become NaN. The code takes x/y gradients per frame, forms speed in pixels/frame, marks frames without both coordinates missing, and linearly interpolates finite speeds onto `TIME`. It does not actually smooth positions despite comments suggesting interpolation/smoothing.

ii.
```python
xy[lk<.9]=np.nan
vx=np.gradient(xy[:,0]); vy=np.gradient(xy[:,1])
speed=np.hypot(vx,vy); speed[~visible]=np.nan
out.append(interp_stream(ft,speed))
```

iii. The agent said this matched `findVelocity`'s gradient of coordinates and that explicit missingness should be preserved.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median is computed over every finite, interpolated tongue sample. Finite values below the median are 0 and values at/above it are 1; NaNs are 2.

ii.
```python
z=np.concatenate([x[k][np.isfinite(x[k])] for x in continuous])
thresholds.append(float(np.percentile(z,50)) if z.size else np.nan)
y=np.full(TIME.size,2); y[ok]=(z[ok]>=thresholds[k]).astype(np.int16)
```

iii. The agent followed the requested per-session 50th-percentile threshold and explicit not-visible class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. It subtracts the Bpod go cue directly from camera `frameTimes`, then interpolates speed onto the neural `TIME` centers. It does not estimate or subtract the video-to-behavior clock offset.

ii.
```python
ft=arr(trial_item(tr['frameTimes'],i))-go
return np.interp(TIME,xx,yy,left=np.nan,right=np.nan)
```

iii. The agent assumed camera frame times could be made go-relative by direct subtraction; the trajectory does not document a clock-offset check.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses both `top_paw` and `bottom_paw` landmarks from bottom-camera tracking, including x/y, likelihood, and frame times.

ii.
```python
for cam, feats in [(0,['tongue']),(1,['top_paw','bottom_paw'])]:
```

iii. The agent described paw as the mean of top and bottom paw in the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Low-confidence coordinates are set to NaN, the two paw positions are averaged framewise, x/y gradients yield pixels/frame speed, and finite speed is linearly interpolated to the common grid.

ii.
```python
xy=np.nanmean(np.stack(coords),axis=0)
vx=np.gradient(xy[:,0]); vy=np.gradient(xy[:,1])
speed=np.hypot(vx,vy)
```

iii. The agent intended to combine the two paw landmarks into one position and apply the same tracking pipeline as tongue.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The per-session median of all finite interpolated paw samples is used; below is 0, at/above is 1, and NaN is 2.

ii.
```python
thresholds.append(float(np.percentile(z,50)) if z.size else np.nan)
y[ok]=(z[ok]>=thresholds[k]).astype(np.int16)
```

iii. This directly implements the requested per-session split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times have the go cue directly subtracted, then paw speed is interpolated onto `TIME`; no cross-clock video offset is applied.

ii.
```python
ft=arr(trial_item(tr['frameTimes'],i))-go
out.append(interp_stream(ft,speed))
```

iii. The agent assumed direct go-cue subtraction was sufficient and emphasized a shared grid.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It reads only embedded `obj.me` (either a dictionary's `data` or a list) and side-camera frame times. It does not load the standalone `motionEnergy_<animal>_<date>.mat` files.

ii.
```python
meobj=o.get('me',{})
me=meobj.get('data',[]) if isinstance(meobj,dict) else (meobj if isinstance(meobj,list) else [])
```

iii. The agent discovered that some JEB19 objects embed motion energy and older selected objects do not; it chose class 2 for the latter, despite also finding that repository code normally loads separate files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Nested singleton lists are unwrapped and finite raw per-frame values are linearly interpolated onto the common grid. No smoothing or recomputation from video occurs.

ii.
```python
while isinstance(m,list) and len(m)==1: m=m[0]
out.append(interp_stream(ft,m))
```

iii. The agent regarded embedded motion energy as already processed and decided absent streams should remain missing rather than be synthesized.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. It uses the session median of all finite interpolated values; below is 0, at/above is 1, and missing is 2. Sessions without embedded `obj.me` consequently contain only class 2.

ii.
```python
thresholds.append(float(np.percentile(z,50)) if z.size else np.nan)
y=np.full(TIME.size,2); y[ok]=(z[ok]>=thresholds[k]).astype(np.int16)
```

iii. The agent followed the requested median rule and interpreted absent embedded motion energy as “no video.”

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times have the go cue directly subtracted and motion energy is interpolated onto `TIME`, without the session video clock offset.

ii.
```python
ft=arr(trial_item(o['traj'][0]['frameTimes'],i))-go
out.append(interp_stream(ft,m))
```

iii. The agent followed the repository's broad idea of interpolation to the neural axis but omitted its clock shift.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Low-likelihood tracking and streams with fewer than two finite samples become NaN, later category 2. Missing features yield all-NaN trials; motion-energy extraction errors are broadly caught and also become category 2. Sessions with fewer than two trials or zero neurons are dropped. Finite gaps are bridged by interpolation, while extrapolation remains NaN.

ii.
```python
if ok.sum()<2:return np.full(TIME.size,np.nan)
except Exception: out.append(np.full(TIME.size,np.nan))
if len(neu)<2 or not neu or neu[0].shape[0]==0: continue
```

iii. The agent emphasized preserving missing video as an explicit target category instead of filling it. It used defensive handling to let conversion continue.

## 11-a. What are the most time-consuming steps of the code?

i. The code does not profile or state this explicitly. Its likely dominant work is loading large `mat73` objects and the nested unit-by-trial spike histogram/smoothing loop; video processing also loops through every retained trial.

ii.
```python
o=mat73.loadmat(path)['obj']
for jj,u in enumerate(keep0):
    for ii,t in enumerate(valid):
```

iii. The trajectory focused on correctness and end-to-end decoder runtime, not conversion profiling, so no measured justification was provided.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested unit-by-trial spike selection/histogram loop could be replaced by a per-unit 2-D trial/time histogram. Trial label construction and threshold categorization could also be array operations; ragged camera data makes full video vectorization harder.

ii.
```python
for jj,u in enumerate(keep0):
    for ii,t in enumerate(valid):
        z=st[tri==t]-go[t]
```

iii. The agent did not discuss vectorization choices in the trajectory.

## 11-c. What processing does the code repeat multiple times?

i. It recreates/casts the identical input `TIME` for every trial, calls the smoothing function separately for every unit-trial pair (rebuilding its kernel each call), and repeatedly extracts trajectory metadata and frame times inside `video_trial`.

ii.
```python
inp.append(TIME[None,:].astype(np.float32))
neural[ii,jj]=matlab_smooth(...)
```

iii. No rationale for these repeated computations appears in the trajectory.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `mat73` materializes the whole `obj` although only selected fields are used; `R` and `no` are loaded but not used in label logic; `scipy.signal.windows.gaussian` creates `k` before that value is immediately overwritten; and continuous camera streams are retained temporarily only to compute thresholds and categorical outputs.

ii.
```python
k=gaussian(15,std=(15-1)/6, sym=True)
n=np.arange(15)-(14/2); k=np.exp(...)
L=arr(b['L']); R=arr(b['R']); ... no=arr(b['no'])
```

iii. The trajectory provides no explicit justification; these appear to be implementation leftovers or convenient intermediate data.
