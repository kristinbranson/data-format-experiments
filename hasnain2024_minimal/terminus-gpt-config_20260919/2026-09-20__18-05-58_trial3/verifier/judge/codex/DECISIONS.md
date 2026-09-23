# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 12-session cohort from `/app/data/Ephys_Behavior` and loads each session with `mat73.loadmat`. It does not search both electrophysiology folders, does not use the separate `motionEnergy_*.mat` files, and does not support MATLAB v5 files. In the trajectory it justified this as following the paper's Figure 8 "two-context" cohort rather than the full released set.

ii.
```python
DATA='/app/data/Ephys_Behavior'
SESS=[
 ('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),
 ('JGR2','2021-11-16',1),('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),
 ('JEB19','2023-04-21',1),('JEB19','2023-04-20',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-18',1)]
...
o=mat73.loadmat(path)['obj']
```

iii. The trajectory repeatedly says the requested WC/DR outputs "restrict conversion to the two-context electrophysiology sessions" and that the "exact expert cohort" is 12 sessions from Figure 8, so the agent intentionally narrowed the dataset.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the hard-coded animal name in each `(animal, date, probe)` tuple. A subject is added the first time its animal string appears, so `subjects` preserves first-seen order rather than being derived from session filenames and then sorted.

ii.
```python
for anm,date,probe in SESS:
    ...
    if anm not in D['subjects']:D['subjects'].append(anm)
    D['subject_idx'].append(D['subjects'].index(anm))
```

iii. The trajectory treats the Figure 8 session list as authoritative and reports the resulting cohort as seven mice, so subject splitting follows that same hard-coded cohort.

## 1-c. How are the data split into sessions?

i. Each tuple in `SESS` becomes one session. The script builds one path per tuple under `/app/data/Ephys_Behavior`, processes it once, and appends one entry to `neural`, `input`, `output`, and `subject_idx`.

ii.
```python
for anm,date,probe in SESS:
    p=f'{DATA}/data_structure_{anm}_{date}.mat'
    neu,inp,out,thr=process(p,probe)
    ...
    D['neural'].append(neu)
    D['input'].append(inp)
    D['output'].append(out)
```

iii. The trajectory explicitly says it will "process the exact 12 sessions/probes sequentially" because it believes Figure 8 defines the correct cohort.

## 1-d. How are the data split into trials?

i. Trials are represented by integer trial indices in the Bpod arrays. After loading `Ntrials`, the script identifies "valid" trial numbers using `goCue`, `stim.enable`, and `early`, then iterates over those trial indices. Neural and behavioral values are pulled for each retained trial using the same integer trial id.

ii.
```python
o=mat73.loadmat(path)['obj']; b=o['bp']; n=int(float(np.asarray(b['Ntrials'])))
go=arr(b['ev']['goCue']); stim=arr(b['stim']['enable']); early=arr(b['early'])
valid=np.flatnonzero(np.isfinite(go)&(stim==0)&(early==0))
...
for ii,t in enumerate(valid):
    z=st[tri==t]-go[t]
...
for t in valid:
    inp.append(TIME[None,:].astype(np.float32))
```

iii. The trajectory justification is implicit: it says the session contains the needed Bpod fields, including `goCue`, `stim`, and `early`, and that it will "retain all valid non-stimulation/non-early trials."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to those with finite `goCue`, `stim.enable == 0`, and `early == 0`. The script keeps correct, incorrect, and ignore trials, but it does not implement the reference's extra drop of trials that continue after the recording ends.

ii.
```python
go=arr(b['ev']['goCue']); stim=arr(b['stim']['enable']); early=arr(b['early'])
valid=np.flatnonzero(np.isfinite(go)&(stim==0)&(early==0))
...
'trial_filter':'no photostimulation, no early lick; includes correct, incorrect and ignore trials',
```

iii. The trajectory explicitly says the agent retained "all non-stim, non-early trials" because outcome is a decoder target and correct-only subsets would discard the requested ignore/incorrect classes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `obj['clu']` on one selected probe per session, specifically each unit's `trialtm`, `trial`, and `quality`, plus `bp.ev.goCue` for alignment.

ii.
```python
clu=o['clu'][probe-1] if isinstance(o['clu'],list) else o['clu']
quals=clu['quality']
...
st=arr(clu['trialtm'][u]); tri=arr(clu['trial'][u]).astype(int)-1
z=st[tri==t]-go[t]
```

iii. In the trajectory the agent says the cluster schema is "per-unit quality and spike timestamps plus 1-based trial assignments" and that go-cue alignment should reproduce the paper's neural processing.

## 2-b. How is the `neural` data processed?

i. For each kept unit and each kept trial, spikes are histogrammed into 5 ms bins around the go cue, converted to firing rate by dividing by `DT`, and smoothed with a causal 15-sample Gaussian kernel implemented by `matlab_smooth`.

ii.
```python
DT=.005; TMIN=-2.5; TMAX=2.5
TIME=np.arange(TMIN,TMAX+DT/2,DT)
...
def matlab_smooth(x):
    n=np.arange(15)-(14/2); k=np.exp(-.5*(2.5*n/(14/2))**2)
    k[:7]=0; k/=k.sum()
    return np.stack([np.convolve(row,k,'same') for row in x],axis=0)
...
neural[ii,jj]=matlab_smooth((np.histogram(z,edges)[0]/DT)[None,:])[0]
```

iii. The trajectory says it is reproducing "5 ms go-cue-aligned firing rates and causal Gaussian smoothing" from Figure 8, and the module docstring repeats that rationale.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps units whose quality string is not one of `garbage`, `noise`, `nan`, `none`, or empty, then drops units whose average rate across retained trials and all time bins is not strictly above 1 Hz.

ii.
```python
quals=clu['quality']; keep0=[]
for u,q in enumerate(quals):
    q=str(q).lower()
    if q not in ('garbage','noise','nan','none',''): keep0.append(u)
...
unitkeep=np.nanmean(neural,axis=(0,2))>1
neural=neural[:,unitkeep,:]
```

iii. The trajectory says it wanted "all curated units followed by >1 Hz filtering" and that Figure 8 appeared to use `quality='all'` with a meaningful garbage exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting the trial's go-cue time from each spike's per-trial timestamp.

ii.
```python
st=arr(clu['trialtm'][u]); tri=arr(clu['trial'][u]).astype(int)-1
...
z=st[tri==t]-go[t]
```

iii. The trajectory explicitly says "go-cue alignment" is the authoritative default and that trial delays do not matter once spikes are aligned to `goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 5 ms spacing from `-2.5` to `+2.5` seconds. The script constructs a 1001-sample `TIME` vector, defines histogram edges around those sample centers, and produces 1001 firing-rate values per trial.

ii.
```python
DT=.005; TMIN=-2.5; TMAX=2.5
TIME=np.arange(TMIN,TMAX+DT/2,DT)
...
edges=np.r_[TIME-DT/2,TIME[-1]+DT/2]
...
neural=np.zeros((len(valid),len(keep0),TIME.size),np.float32)
```

iii. The trajectory says the agent is matching the paper's "5 ms bins from -2.5 to +2.5 seconds," but its implementation uses sample centers rather than the reference's 1000-bin edge grid.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This input is not read directly from a raw variable. It is defined from the chosen alignment window around `goCue`: the script creates a fixed `TIME` vector representing seconds from go cue.

ii.
```python
DT=.005; TMIN=-2.5; TMAX=2.5
TIME=np.arange(TMIN,TMAX+DT/2,DT)
...
inp.append(TIME[None,:].astype(np.float32))
```

iii. The trajectory describes this as part of the go-cue-aligned 5 ms analysis grid rather than as a source variable taken from the file.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No per-trial computation is done beyond repeating the same `TIME` vector for each retained trial and casting it to `float32`.

ii.
```python
for t in valid:
    inp.append(TIME[None,:].astype(np.float32))
```

iii. The trajectory gives no separate justification here beyond using the common go-cue-aligned grid for all streams.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the exact same `TIME` grid that defines the neural histogram edges, so the input and neural arrays are aligned by construction.

ii.
```python
TIME=np.arange(TMIN,TMAX+DT/2,DT)
edges=np.r_[TIME-DT/2,TIME[-1]+DT/2]
...
inp.append(TIME[None,:].astype(np.float32))
...
neural[ii,jj]=matlab_smooth((np.histogram(z,edges)[0]/DT)[None,:])[0]
```

iii. The trajectory repeatedly states that all streams are put on one common go-cue-aligned 5 ms axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The final code derives lick direction from the trial side (`L`) together with `hit` and `miss`. The script also reads `R` and `no`, but it does not use them after the final patch.

ii.
```python
L=arr(b['L']); R=arr(b['R']); aw=arr(b['autowater'])
hit=arr(b['hit']); miss=arr(b['miss']); no=arr(b['no'])
...
if outcome==2:
    lick=2
elif hit[t]>0:
    lick=0 if L[t]>0 else 1
else:
    lick=1 if L[t]>0 else 0
```

iii. The trajectory explicitly notes a late correction: lick direction "must be based on actual response: instructed side on hits, opposite side on misses, none on ignores."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI encodes left as `0`, right as `1`, and none as `2`. Ignore trials get `2`; hit trials get the instructed side; miss trials get the opposite side.

ii.
```python
outcome=1 if hit[t]>0 else (0 if miss[t]>0 else 2)
if outcome==2:
    lick=2
elif hit[t]>0:
    lick=0 if L[t]>0 else 1
else:
    lick=1 if L[t]>0 else 0
```

iii. The trajectory says this was a deliberate fix after validation showed the earlier version mislabeled ignore and miss trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp.autowater`.

ii.
```python
aw=arr(b['autowater'])
...
outs.append([lick,int(not (aw[t]>0)),outcome])
```

iii. The trajectory identifies `autowater` as the context flag and later patches the encoding so that WC maps to `0` and DR to `1`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code relabels `autowater > 0` trials as WC (`0`) and the rest as DR (`1`) by storing `int(not (aw[t] > 0))`.

ii.
```python
outs.append([lick,int(not (aw[t]>0)),outcome])
```

iii. The trajectory explicitly notes this patch: the original coding used `autowater` directly, but the agent changed it because `output_values` required `WC=0, DR=1`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp.hit` and `bp.miss`; anything that is neither is treated as ignore.

ii.
```python
hit=arr(b['hit']); miss=arr(b['miss']); no=arr(b['no'])
...
outcome=1 if hit[t]>0 else (0 if miss[t]>0 else 2)
```

iii. The trajectory describes outcome classes as correct, incorrect, and ignore and says these trials were retained because outcome is a requested decoder target.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps miss to `0`, hit to `1`, and all other retained trials to `2`.

ii.
```python
outcome=1 if hit[t]>0 else (0 if miss[t]>0 else 2)
```

iii. The trajectory justification is that the decoder requires all three outcome classes instead of the paper's correct-trial-only analysis subsets.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj['traj'][0]`, specifically the side-camera feature named `tongue`, along with that camera's `ts` and `frameTimes`. Alignment uses the trial's `goCue`. The bottom-camera tongue feature is not used in the final code.

ii.
```python
for cam, feats in [(0,['tongue']),(1,['top_paw','bottom_paw'])]:
    tr=o['traj'][cam]; ts=np.asarray(trial_item(tr['ts'],i),float)
    ft=arr(trial_item(tr['frameTimes'],i))-go
    nm=names_at(tr,i)
```

iii. The trajectory initially intended to use frame-aligned behavioral streams and explicit missing classes, but the final code does not include the two-view tongue combination described in the reference.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Low-confidence frames (`likelihood < 0.9`) are set to NaN. The code then computes per-frame speed as `sqrt(gradient(x)^2 + gradient(y)^2)` on the side-camera coordinates, preserves NaNs where the tongue is not visible, and linearly interpolates the resulting speed trace onto the common `TIME` grid.

ii.
```python
j=nm.index(feat); xy=ts[:,0:2,j].copy(); lk=ts[:,2,j]
xy[lk<.9]=np.nan
...
visible=np.all(np.isfinite(xy),axis=1)
vx=np.gradient(xy[:,0]); vy=np.gradient(xy[:,1])
speed=np.hypot(vx,vy); speed[~visible]=np.nan
out.append(interp_stream(ft,speed))
```

iii. The trajectory justification is only partial: it says video missingness should be preserved and behavioral streams interpolated to the common axis, but it does not explicitly justify omitting the reference's per-run smoothing, time-derivative scaling, and two-camera normalization.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After continuous tongue speeds are computed for all trials in a session, the AI pools all finite tongue samples across that session, computes the 50th percentile, and discretizes each time point as below-threshold (`0`), at-or-above-threshold (`1`), or not visible (`2`).

ii.
```python
thresholds=[]
for k in range(3):
    z=np.concatenate([x[k][np.isfinite(x[k])] for x in continuous])
    thresholds.append(float(np.percentile(z,50)) if z.size else np.nan)
...
y=np.full(TIME.size,2,dtype=np.int16); ok=np.isfinite(z)
y[ok]=(z[ok]>=thresholds[k]).astype(np.int16)
```

iii. The trajectory repeatedly says the movement outputs are discretized using per-session medians and explicit missing classes.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The side-camera `frameTimes` are shifted only by subtracting the trial's `goCue`, then the speed trace is interpolated onto the common `TIME` vector. The code does not apply the reference's per-session video-to-behavior clock offset.

ii.
```python
ft=arr(trial_item(tr['frameTimes'],i))-go
...
out.append(interp_stream(ft,speed))
```

iii. The trajectory says the behavioral streams should be "frame-aligned" and on the "common go-aligned grid," but the final implementation stops at direct `frameTimes - goCue`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom camera, but the code averages both `top_paw` and `bottom_paw` landmarks rather than using a single reliable paw feature.

ii.
```python
for cam, feats in [(0,['tongue']),(1,['top_paw','bottom_paw'])]:
    ...
    for feat in feats:
        if feat not in nm: continue
        j=nm.index(feat); xy=ts[:,0:2,j].copy(); lk=ts[:,2,j]
```

iii. The trajectory does not explicitly justify averaging the two paw landmarks; this choice is only visible in the final code comment and loop structure.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The two paw coordinates that are present are averaged with `np.nanmean`, low-confidence frames are blanked by NaNs, per-frame speed is computed with coordinate gradients, and the result is linearly interpolated onto `TIME`.

ii.
```python
xy[lk<.9]=np.nan; coords.append(xy)
...
xy=np.nanmean(np.stack(coords),axis=0)
visible=np.all(np.isfinite(xy),axis=1)
vx=np.gradient(xy[:,0]); vy=np.gradient(xy[:,1])
speed=np.hypot(vx,vy); speed[~visible]=np.nan
out.append(interp_stream(ft,speed))
```

iii. The trajectory again only gives the high-level rationale of putting behavioral streams on the common axis with explicit missingness; it does not mention the reference's run-wise smoothing or time-derivative scaling.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity uses the same per-session median split as tongue velocity: `0` below median, `1` at or above median, `2` not visible.

ii.
```python
thresholds=[]
for k in range(3):
    z=np.concatenate([x[k][np.isfinite(x[k])] for x in continuous])
    thresholds.append(float(np.percentile(z,50)) if z.size else np.nan)
...
y=np.full(TIME.size,2,dtype=np.int16); ok=np.isfinite(z)
y[ok]=(z[ok]>=thresholds[k]).astype(np.int16)
```

iii. The trajectory explicitly says the movement outputs will be "session-median discretized."

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. As with tongue velocity, the bottom-camera `frameTimes` are reduced only by the trial `goCue` and the resulting speed trace is interpolated to `TIME`. No video clock offset is applied.

ii.
```python
ft=arr(trial_item(tr['frameTimes'],i))-go
...
out.append(interp_stream(ft,speed))
```

iii. The trajectory justification is only that the streams should share one go-cue-aligned axis.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is taken from the `obj['me']` field embedded inside the session file when present. The code unwraps either `me['data']` or a list-like `me` object. It does not load the separate `motionEnergy_<session>.mat` files.

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

iii. The trajectory shows the agent discovered late that some sessions had embedded `me` while older sessions did not; it then decided to unwrap embedded JEB19 motion-energy arrays and otherwise preserve "no video."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The script does not recompute motion energy from video. It unwraps the per-trial stored trace if possible and linearly interpolates that trace onto the common `TIME` grid; on failure it returns all-NaN for that trial.

ii.
```python
while isinstance(m,list) and len(m)==1: m=m[0]
out.append(interp_stream(ft,m))
except Exception: out.append(np.full(TIME.size,np.nan))
```

iii. The trajectory says it considered reproducing motion energy from video, but the final implementation uses embedded `me` when available and otherwise falls back to the missing-data category.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded exactly like the other continuous outputs: per-session median split of finite values, with `2` for missing video.

ii.
```python
for k in range(3):
    z=np.concatenate([x[k][np.isfinite(x[k])] for x in continuous])
    thresholds.append(float(np.percentile(z,50)) if z.size else np.nan)
...
y=np.full(TIME.size,2,dtype=np.int16); ok=np.isfinite(z)
y[ok]=(z[ok]>=thresholds[k]).astype(np.int16)
```

iii. The trajectory explicitly says motion energy should use the requested per-session median discretization and preserve a "no video" class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses camera-0 `frameTimes`, subtracts the trial `goCue`, and interpolates the raw trace to the common `TIME` grid. The code does not apply the reference's bitcode-derived video offset before alignment.

ii.
```python
m=trial_item(me,i); ft=arr(trial_item(o['traj'][0]['frameTimes'],i))-go
...
out.append(interp_stream(ft,m))
```

iii. The trajectory justification is that motion energy should be resampled onto the common neural time axis, but the final code omits the reference's clock-correction step.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable behavioral samples are represented as NaN in the continuous streams and then converted to category `2` during discretization. Missing motion-energy structures are handled by a broad `try/except` that emits an all-NaN trace, which then becomes `no video`. The code does not fill missing values.

ii.
```python
if ok.sum()<2:return np.full(TIME.size,np.nan)
...
xy[lk<.9]=np.nan
...
except Exception: out.append(np.full(TIME.size,np.nan))
...
y=np.full(TIME.size,2,dtype=np.int16); ok=np.isfinite(z)
y[ok]=(z[ok]>=thresholds[k]).astype(np.int16)
```

iii. The trajectory explicitly says video missingness should be an explicit category rather than being filled, and later says "older sessions still have tracked video but no motion-energy stream; class 2 is the only target-supported missing category."

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive steps in this script are the nested neural loops over units and trials, plus the repeated per-trial video processing. The code bins and smooths spikes separately for every unit-trial pair and runs `video_trial` for every retained trial.

ii.
```python
for jj,u in enumerate(keep0):
    st=arr(clu['trialtm'][u]); tri=arr(clu['trial'][u]).astype(int)-1
    for ii,t in enumerate(valid):
        z=st[tri==t]-go[t]
        neural[ii,jj]=matlab_smooth((np.histogram(z,edges)[0]/DT)[None,:])[0]
...
for t in valid:
    ...
    v=video_trial(o,t,go[t]); continuous.append(v)
```

iii. The trajectory does not explicitly profile runtime. This conclusion is inferred from the code structure and from the agent repeatedly re-running a conversion that prints per-session trial and unit counts.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious vectorization targets are the nested `for jj` / `for ii` neural loop, and the per-trial `for t in valid` loop that repeatedly calls `video_trial`. The inner feature loop inside `video_trial` could also be reduced for the fixed feature sets used here.

ii.
```python
for jj,u in enumerate(keep0):
    ...
    for ii,t in enumerate(valid):
        ...
for t in valid:
    ...
    v=video_trial(o,t,go[t]); continuous.append(v)
...
for feat in feats:
    if feat not in nm: continue
```

iii. The trajectory does not explicitly discuss vectorization. This answer is inferred from the implementation.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats some work in two passes. It first computes all continuous behavioral streams and stores them in `continuous`, then scans the stored streams again to compute session thresholds, and then scans them a third time when building discretized outputs. It also computes smoothed firing rates for every curated unit before discarding low-rate units.

ii.
```python
for t in valid:
    ...
    v=video_trial(o,t,go[t]); continuous.append(v)
...
for k in range(3):
    z=np.concatenate([x[k][np.isfinite(x[k])] for x in continuous])
...
for base,v in zip(outs,continuous):
    ...
unitkeep=np.nanmean(neural,axis=(0,2))>1
neural=neural[:,unitkeep,:]
```

iii. The trajectory does not call this out explicitly; it is visible in the final code organization.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is the computation of full smoothed neural traces for units that are later removed by the `>1 Hz` filter. The script also loads or computes values it does not use downstream, such as `R`, `no`, and some imported functions.

ii.
```python
neural=np.zeros((len(valid),len(keep0),TIME.size),np.float32)
...
unitkeep=np.nanmean(neural,axis=(0,2))>1
neural=neural[:,unitkeep,:]
...
L=arr(b['L']); R=arr(b['R']); aw=arr(b['autowater'])
hit=arr(b['hit']); miss=arr(b['miss']); no=arr(b['no'])
```

iii. The trajectory does not explicitly justify these inefficiencies; they are artifacts of the final implementation rather than stated design goals.
