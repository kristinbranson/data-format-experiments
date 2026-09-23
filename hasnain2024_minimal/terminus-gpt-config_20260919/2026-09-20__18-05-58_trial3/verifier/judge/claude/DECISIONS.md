# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only 12 sessions from the `Ephys_Behavior` folder, using `mat73.loadmat()` to read each MATLAB v7.3 file. The 12 sessions are hard-coded in the `SESS` list, identified from the Figure 8 MATLAB script. No motion energy files are loaded separately; instead `obj.me` (embedded in the data structure) is used when available.

ii.
```python
DATA='/app/data/Ephys_Behavior'
SESS=[
 ('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),
 ('JGR2','2021-11-16',1),('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),
 ('JEB19','2023-04-21',1),('JEB19','2023-04-20',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-18',1)]

for anm,date,probe in SESS:
    p=f'{DATA}/data_structure_{anm}_{date}.mat'
    o=mat73.loadmat(path)['obj']
```

iii. The agent identified the Figure 8 MATLAB script as the authoritative source for the two-context (WC/DR) cohort. It reasoned that since the decoder needs to predict behavioral context (WC vs DR), only two-context sessions should be included. It explicitly ruled out randomized-delay and inhibition sessions.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the animal name string (`anm`) from the hard-coded `SESS` list. Each unique animal name is added to the `subjects` list on first occurrence, and sessions get an index into this list.

ii.
```python
if anm not in D['subjects']:D['subjects'].append(anm)
D['subject_idx'].append(D['subjects'].index(anm))
```

iii. The agent extracted the animal names directly from the session tuples. This yields 7 subjects: JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19.

## 1-c. How are the data split into sessions?

i. Each entry in the `SESS` list is one session. The AI processes only 12 sessions from the `Ephys_Behavior` folder, corresponding to the two-context (WC/DR alternating) cohort. No randomized-delay sessions are included.

ii.
```python
SESS=[
 ('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),
 ('JGR2','2021-11-16',1),('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),
 ('JEB19','2023-04-21',1),('JEB19','2023-04-20',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-18',1)]
```

iii. The agent justified this by referencing the Figure 8 analysis script, which loads these specific 12 sessions for the two-context task. The paper reports "12 sessions from six mice" for two-context sessions (though the agent found 7 animals in the code).

## 1-d. How are the data split into trials?

i. Trials are indexed by their position in the behavioral arrays (0-based). For each session, `Ntrials` is read from `bp.Ntrials` to determine the total number of trials, and each trial is one entry in the behavioral arrays.

ii.
```python
n=int(float(np.asarray(b['Ntrials'])))
go=arr(b['ev']['goCue']); stim=arr(b['stim']['enable']); early=arr(b['early'])
valid=np.flatnonzero(np.isfinite(go)&(stim==0)&(early==0))
```

iii. Trials are defined by the Bpod behavioral table entries, consistent with the reference approach.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) trials with photostimulation (`stim.enable == 1`) are excluded, (2) trials with early licks (`early == 1`) are excluded, (3) trials where go-cue time is not finite are excluded. Unlike the paper's figure panels which use only correct trials, all non-stim, non-early trials are retained (including incorrect and ignore trials) because outcome is a decoder target.

ii.
```python
valid=np.flatnonzero(np.isfinite(go)&(stim==0)&(early==0))
```

iii. The agent explicitly noted: "Unlike correct-trial paper panels, all non-stim, non-early trials are retained because outcome is a requested decoder target." No recording-length filter is applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from `obj.clu[probe-1]`, the spike-sorted clusters for the selected probe. Each cluster has `trialtm` (spike times relative to trial start), `trial` (which trial each spike belongs to), and `quality` (curation label). Go cue times come from `bp.ev.goCue`.

ii.
```python
clu=o['clu'][probe-1] if isinstance(o['clu'],list) else o['clu']
st=arr(clu['trialtm'][u]); tri=arr(clu['trial'][u]).astype(int)-1
z=st[tri==t]-go[t]
```

iii. The agent identified `clu.trialtm` as the spike times on the behavior clock and `clu.trial` as the trial assignment, consistent with the reference approach.

## 2-b. How is the `neural` data processed?

i. Four steps: (1) Spike times are aligned to go cue by subtracting `goCue[trial]` from `trialtm`. (2) Spikes are histogrammed into 5 ms bins from -2.5 to +2.5 s (1001 bins). (3) Counts are divided by bin width (0.005 s) to get firing rates in Hz. (4) A causal Gaussian smoother is applied: a `gausswin(15)` kernel with its first 7 elements zeroed and renormalized, convolved with `'same'` mode.

ii.
```python
edges=np.r_[TIME-DT/2,TIME[-1]+DT/2]
neural[ii,jj]=matlab_smooth((np.histogram(z,edges)[0]/DT)[None,:])[0]
```

```python
def matlab_smooth(x):
    n=np.arange(15)-(14/2); k=np.exp(-.5*(2.5*n/(14/2))**2)
    k[:7]=0; k/=k.sum()
    return np.stack([np.convolve(row,k,'same') for row in x],axis=0)
```

iii. The agent based its smoother on `mySmooth.m` from the reference code, which uses `gausswin(15)` with its first half zeroed to create a causal filter. The agent explicitly noted this matches the paper's processing pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Clusters whose quality label (lowercased) is in `{'garbage', 'noise', 'nan', 'none', ''}` are excluded; all others including multi-units are kept. (2) After smoothing, units with mean firing rate <= 1 Hz (averaged across all time bins and trials) are excluded.

ii.
```python
for u,q in enumerate(quals):
    q=str(q).lower()
    if q not in ('garbage','noise','nan','none',''): keep0.append(u)
...
unitkeep=np.nanmean(neural,axis=(0,2))>1
neural=neural[:,unitkeep,:]
```

iii. The agent cited `findClusters.m` with `params.quality = {'all'}` meaning all non-garbage clusters, and the paper's explicit statement "All units with firing rates exceeding 1 Hz were included in all other analyses."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to go cue is done by subtracting `bp.ev.goCue[trial]` from each spike's `trialtm`. This places spike times in seconds relative to go cue onset.

ii.
```python
z=st[tri==t]-go[t]
neural[ii,jj]=matlab_smooth((np.histogram(z,edges)[0]/DT)[None,:])[0]
```

iii. This matches `alignSpikes.m` from the reference code and the task specification.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 5 ms (DT=0.005 s). The time axis spans -2.5 to +2.5 s, yielding 1001 time points (using `np.arange(TMIN, TMAX+DT/2, DT)`). No rebinning is applied after the initial histogram.

ii.
```python
DT=.005; TMIN=-2.5; TMAX=2.5
TIME=np.arange(TMIN,TMAX+DT/2,DT)
```

iii. The agent matched `getDefaultParams.m`: `params.dt = 1/200`, `params.tmin = -2.5`, `params.tmax = 2.5`. The 1001 time points come from replicating MATLAB's colon operator behavior.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself (`TIME`), defined as bin centers from -2.5 to +2.5 s in 5 ms steps, tiled for each trial.

ii.
```python
inp.append(TIME[None,:].astype(np.float32))
```

iii. The input is defined by the binning grid, not derived from raw data variables.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing — the time axis is a fixed grid defined at module level and tiled identically for each trial.

ii.
```python
TIME=np.arange(TMIN,TMAX+DT/2,DT)
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is the same grid used for spike binning, so alignment is by construction.

ii.
```python
edges=np.r_[TIME-DT/2,TIME[-1]+DT/2]
```

iii. The bin edges for spike histogramming are derived from `TIME`, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `bp.L` (left instruction), `bp.R` (right instruction), `bp.hit` (correct response), and `bp.miss` (incorrect response).

ii.
```python
L=arr(b['L']); R=arr(b['R'])
hit=arr(b['hit']); miss=arr(b['miss'])
```

iii. The agent inferred lick direction from the combination of instructed side and outcome, since actual lick direction is not directly recorded.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. On hit trials, the lick direction matches the instructed side (left=0 if L, right=1 if R). On miss trials, the lick direction is the opposite of the instructed side. On ignore trials (neither hit nor miss), the lick direction is "none" (2).

ii.
```python
if outcome==2:
    lick=2
elif hit[t]>0:
    lick=0 if L[t]>0 else 1
else:
    lick=1 if L[t]>0 else 0
```

iii. The agent corrected an initial implementation to reflect actual lick direction rather than instructed side. Encoding: left=0, right=1, none=2.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. When autowater > 0, the context is WC (water-cued); otherwise DR (delayed-response).

ii.
```python
aw=arr(b['autowater'])
...
outs.append([lick,int(not (aw[t]>0)),outcome])
```

iii. The agent identified autowater as the flag distinguishing the two behavioral contexts.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct mapping: autowater > 0 maps to WC (0), otherwise DR (1). The encoding is WC=0, DR=1.

ii.
```python
int(not (aw[t]>0))  # 0 if autowater (WC), 1 if not (DR)
```

iii. Matches the prompt's WC, DR ordering.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit` and `bp.miss`. If hit > 0, outcome is correct (1). If miss > 0, outcome is incorrect (0). Otherwise, outcome is ignore (2).

ii.
```python
hit=arr(b['hit']); miss=arr(b['miss'])
outcome=1 if hit[t]>0 else (0 if miss[t]>0 else 2)
```

iii. The agent mapped hit/miss/neither to the three outcome categories specified in the instructions.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A direct relabelling: hit -> correct (1), miss -> incorrect (0), neither -> ignore (2).

ii.
```python
outcome=1 if hit[t]>0 else (0 if miss[t]>0 else 2)
```

iii. Encoding matches the prompt: incorrect=0, correct=1, ignore=2.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `obj.traj[0]` (side camera, index 0), specifically the feature named `'tongue'`. The tracking data includes `ts` (x, y, likelihood per feature per frame), `frameTimes`, and `featNames`.

ii.
```python
for cam, feats in [(0,['tongue']),(1,['top_paw','bottom_paw'])]:
    tr=o['traj'][cam]; ts=np.asarray(trial_item(tr['ts'],i),float)
    ft=arr(trial_item(tr['frameTimes'],i))-go
    nm=names_at(tr,i)
    for feat in feats:
        if feat not in nm: continue
        j=nm.index(feat); xy=ts[:,0:2,j].copy(); lk=ts[:,2,j]
        xy[lk<.9]=np.nan; coords.append(xy)
```

iii. The agent used only the side camera for tongue tracking. The reference solution uses both side and bottom camera views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps: (1) Frames with DLC likelihood < 0.9 are set to NaN. (2) Velocity is computed as `np.gradient` of x and y separately, then speed = `hypot(vx, vy)`. No positional smoothing is applied before differentiation. (3) The speed is interpolated onto the 5 ms time grid using `np.interp`. (4) The speed is discretized at the session 50th percentile: below=0, at or above=1, not visible=2.

ii.
```python
xy[lk<.9]=np.nan; coords.append(xy)
...
vx=np.gradient(xy[:,0]); vy=np.gradient(xy[:,1])
speed=np.hypot(vx,vy); speed[~visible]=np.nan
out.append(interp_stream(ft,speed))
```

```python
def interp_stream(ft,val):
    ft=arr(ft); val=arr(val)
    ok=np.isfinite(ft)&np.isfinite(val)
    if ok.sum()<2:return np.full(TIME.size,np.nan)
    order=np.argsort(ft[ok]); xx=ft[ok][order]; yy=val[ok][order]
    return np.interp(TIME,xx,yy,left=np.nan,right=np.nan)
```

iii. The agent did not apply positional smoothing before differentiation. The reference applies a 5 ms Gaussian to the positions within contiguous runs of valid frames before computing the gradient. The agent also uses `np.interp` (linear interpolation) to map onto the time grid, whereas the reference bins by averaging frames within each 5 ms bin.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Discretized at the session-wide 50th percentile of all finite (visible) samples: values below the threshold get 0, at or above get 1, NaN (not visible) gets 2.

ii.
```python
z=np.concatenate([x[k][np.isfinite(x[k])] for x in continuous])
thresholds.append(float(np.percentile(z,50)) if z.size else np.nan)
...
y=np.full(TIME.size,2,dtype=np.int16); ok=np.isfinite(z)
y[ok]=(z[ok]>=thresholds[k]).astype(np.int16)
```

iii. Matches the instructions: per-session 50th percentile threshold, with "not visible" as class 2.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are aligned to go cue by subtracting `go[t]` from `frameTimes`. The resulting time-speed pairs are then linearly interpolated onto the same 1001-point time grid as the neural data using `np.interp`. No video-to-behavior clock offset correction is applied.

ii.
```python
ft=arr(trial_item(tr['frameTimes'],i))-go
...
out.append(interp_stream(ft,speed))
```

iii. The agent subtracts the go cue time from frame times directly. The reference solution computes a video offset from bitcode timestamps (`sglx.bitcode.bitstart / sglx.fs` vs `bp.ev.bitStart`) before subtracting the go cue.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from `obj.traj[1]` (bottom camera, index 1), using features `'top_paw'` and `'bottom_paw'`. Both paw landmarks are averaged.

ii.
```python
for cam, feats in [(0,['tongue']),(1,['top_paw','bottom_paw'])]:
```

iii. The agent used both paw landmarks from the bottom camera, averaging their coordinates. The reference uses only `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue velocity: likelihood thresholding at 0.9, averaging of both paw coordinates, gradient-based velocity computation (no positional smoothing), linear interpolation onto the time grid, and discretization at the session 50th percentile.

ii.
```python
xy=np.nanmean(np.stack(coords),axis=0)
vx=np.gradient(xy[:,0]); vy=np.gradient(xy[:,1])
speed=np.hypot(vx,vy); speed[~visible]=np.nan
out.append(interp_stream(ft,speed))
```

iii. The agent averaged both paw landmarks (top_paw and bottom_paw) rather than using only top_paw.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same approach as tongue velocity: per-session 50th percentile of all visible samples; below=0, at or above=1, not visible=2.

ii.
```python
thresholds.append(float(np.percentile(z,50)) if z.size else np.nan)
```

iii. Consistent with the instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue velocity: frame times minus go cue, then linear interpolation onto the time grid. No video offset correction.

ii.
```python
ft=arr(trial_item(tr['frameTimes'],i))-go
out.append(interp_stream(ft,speed))
```

iii. Same alignment approach as tongue, without video offset.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Derived from `obj.me` (embedded in the data structure object), when available. For sessions without `me` (most of the 12 sessions), all time points are assigned class 2 ("no video").

ii.
```python
meobj=o.get('me',{})
me=meobj.get('data',[]) if isinstance(meobj,dict) else (meobj if isinstance(meobj,list) else [])
try:
    m=trial_item(me,i); ft=arr(trial_item(o['traj'][0]['frameTimes'],i))-go
    while isinstance(m,list) and len(m)==1: m=m[0]
    out.append(interp_stream(ft,m))
except Exception: out.append(np.full(TIME.size,np.nan))
```

iii. The agent noted that motion energy is genuinely absent from `obj.me` for most sessions. The reference loads separate `motionEnergy_*.mat` files which exist for all 44 sessions.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. When available, the per-trial motion energy array is interpolated onto the time grid using `np.interp`, then discretized at the session 50th percentile.

ii.
```python
out.append(interp_stream(ft,m))
```

iii. No additional processing beyond interpolation and discretization.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same per-session 50th percentile approach as tongue and paw velocity: below=0, at or above=1, no video=2.

ii.
```python
y=np.full(TIME.size,2,dtype=np.int16); ok=np.isfinite(z)
y[ok]=(z[ok]>=thresholds[k]).astype(np.int16)
```

iii. Consistent with the instructions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera (`traj[0].frameTimes`) minus go cue, then linear interpolation onto the time grid. No video offset correction.

ii.
```python
ft=arr(trial_item(o['traj'][0]['frameTimes'],i))-go
out.append(interp_stream(ft,m))
```

iii. Same alignment approach as tongue and paw, without video offset.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing video data (NaN tongue/paw positions or absent motion energy) are preserved as NaN through velocity computation and interpolation, then mapped to class 2 ("not visible" / "no video") in the discretized output. Sessions without `obj.me` get all-NaN motion energy, which becomes all class 2. Sessions with fewer than 2 valid trials or 0 neurons after filtering are dropped with a warning. Trials with non-finite go-cue times are excluded.

ii.
```python
if len(neu)<2 or not neu or neu[0].shape[0]==0:
    warnings.warn('dropping unusable session '+p); continue
```

iii. The agent intentionally preserved video missingness as a separate category rather than filling it, to support the decoder's "not visible" class requirement.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the MAT files with `mat73.loadmat()` dominates runtime. The per-unit, per-trial loop for spike binning and smoothing is also computationally expensive due to nested Python loops.

ii.
```python
o=mat73.loadmat(path)['obj']
...
for jj,u in enumerate(keep0):
    for ii,t in enumerate(valid):
        z=st[tri==t]-go[t]
        neural[ii,jj]=matlab_smooth((np.histogram(z,edges)[0]/DT)[None,:])[0]
```

iii. File I/O is inherently slow. The nested unit-by-trial loop is the main computational bottleneck.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The double loop over units and trials for spike counting and smoothing could be vectorized using `np.histogram2d` (as the reference does), binning all spikes for a unit across all trials in one call. The per-trial video processing loop could also potentially be vectorized.

ii.
```python
for jj,u in enumerate(keep0):
    st=arr(clu['trialtm'][u]); tri=arr(clu['trial'][u]).astype(int)-1
    for ii,t in enumerate(valid):
        z=st[tri==t]-go[t]
        neural[ii,jj]=matlab_smooth((np.histogram(z,edges)[0]/DT)[None,:])[0]
```

iii. The reference uses `np.histogram2d` to count all spikes for a cluster across all trials at once, avoiding the inner trial loop entirely.

## 11-c. What processing does the code repeat multiple times?

i. The smoothing kernel (`matlab_smooth`) is recomputed on every call — the `gausswin` kernel is reconstructed each time, though this is a negligible cost. The `interp_stream` function is called separately for each video feature per trial.

ii.
```python
def matlab_smooth(x):
    k=gaussian(15,std=(15-1)/6, sym=True)
    n=np.arange(15)-(14/2); k=np.exp(-.5*(2.5*n/(14/2))**2)
    k[:7]=0; k/=k.sum()
    return np.stack([np.convolve(row,k,'same') for row in x],axis=0)
```

iii. The kernel could be precomputed once, but the cost is negligible compared to I/O and spike binning.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads the entire data object with `mat73.loadmat`, materializing many fields never used (spike waveforms, other tracking features, etc.). Additionally, computing video velocity for all frames including ones with NaN coordinates leads to NaN velocities that are immediately discarded in the discretization step.

ii.
```python
o=mat73.loadmat(path)['obj']
```

iii. No selective field reading is performed, so the full data structure is loaded into memory.
