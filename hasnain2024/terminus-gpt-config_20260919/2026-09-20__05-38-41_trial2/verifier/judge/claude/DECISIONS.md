# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only 12 sessions from the `Ephys_Behavior` folder (the "Figure 8 two-context subset"), not all 44 sessions across both `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`. Sessions are hardcoded in a `SESSIONS` list with tuples of (subject, date, probe). Each session is opened as an HDF5 file using `h5py.File` only -- the code does not handle MATLAB v5 format files. Motion energy is loaded separately via `scipy.io.loadmat`.

ii.
```python
ROOT=Path('/app/data/Ephys_Behavior')
SESSIONS=[
 ('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),
 ('JGR2','2021-11-16',1),('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),
 ('JEB19','2023-04-21',1),('JEB19','2023-04-20',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-18',1)]
# ...
with h5py.File(p,'r') as f:
```

iii. The AI justified this by choosing the "Figure 8 two-context subset" from the paper, arguing that context is a required output and this cohort matches the paper's 12-session WC/DR analysis. From CONVERSION_NOTES.md: "Use the exact 12-session Figure 8/two-context cohort."

## 1-b. How are the data split into subjects?

i. Subject is extracted from the session tuple's first element (e.g., `'JEB6'`). Unique subjects are accumulated in order of first appearance.

ii.
```python
for si,(sub,date,probe) in enumerate(sessions):
    # ...
    if sub not in subjects:subjects.append(sub)
    subject_idx.append(subjects.index(sub))
```

iii. Subject identity comes from the hardcoded session metadata. The AI preserves released IDs without merging.

## 1-c. How are the data split into sessions?

i. Each tuple in `SESSIONS` is one session, identified by (subject, date, probe). The AI only looks in the `Ephys_Behavior` folder (hardcoded `ROOT`), not `RandomizedDelay_Ephys_Behavior`. This means the 19 randomized-delay sessions are excluded entirely.

ii.
```python
ROOT=Path('/app/data/Ephys_Behavior')
# ...
p=ROOT/f'data_structure_{subject}_{date}.mat'
```

iii. The AI decided to use only the two-context subset sessions that appear in the Figure 8 analysis scripts.

## 1-d. How are the data split into trials?

i. Trials are identified by index into the behavior arrays. The AI reads behavior fields (`L`, `R`, `hit`, `miss`, `no`, `early`, `autowater`, `stim`, `goCue`, `haveEphys`, `haveVid`) and creates a boolean validity mask. Valid trial indices are extracted with `np.flatnonzero(valid)`.

ii.
```python
b=behavior(f); n=len(b['go'])
valid=b['haveEphys']&np.isfinite(b['go'])&(np.asarray(b['early'])==0)&(np.asarray(b['stim'])==0)
outcome_sum=np.asarray(b['hit'])+np.asarray(b['miss'])+np.asarray(b['no'])
valid &= outcome_sum==1
ids=np.flatnonzero(valid)
```

iii. The AI uses the native per-trial indexing from the behavior structure. Each trial is one entry in the behavior arrays.

## 1-e. How are trials filtered based on quality controls?

i. Four filters are applied: (1) `haveEphys` must be true, (2) `goCue` must be finite, (3) `early` and `stim` must be zero, (4) `hit + miss + no` must equal exactly 1 (mutually exclusive outcome check). Unlike the reference, the AI does not check whether trials fall within the actual recording range (i.e., whether spikes exist for that trial).

ii.
```python
valid=b['haveEphys']&np.isfinite(b['go'])&(np.asarray(b['early'])==0)&(np.asarray(b['stim'])==0)
outcome_sum=np.asarray(b['hit'])+np.asarray(b['miss'])+np.asarray(b['no'])
valid &= outcome_sum==1
```

iii. The AI follows the paper's exclusion of early-lick and photostimulation trials. The `haveEphys` filter is an additional validity check. The outcome-sum check ensures mutually exclusive outcomes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `obj.clu` (spike-sorted clusters) via HDF5 references. For each cluster: `trialtm` (spike times relative to trial start), `trial` (trial indices), and `quality` (curation labels). `bp.ev.goCue` provides alignment.

ii.
```python
pref=np.asarray(f['obj/clu']).ravel()[probe-1]; g=f[pref]
for rt,ri,rq in zip(np.asarray(g['trialtm']).ravel(),np.asarray(g['trial']).ravel(),np.asarray(g['quality']).ravel()):
    tm=np.atleast_1d(h5arr(f,rt)).astype(float); tr=np.atleast_1d(h5arr(f,ri)).astype(int)-1
```

iii. Same source variables as the reference code's `alignSpikes` and cluster loading.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into 10 ms bins from -3.0 to +2.5 s, divided by bin width (0.01 s) to get Hz, then smoothed with a **causal** Gaussian kernel. The causal smoothing exactly replicates the reference MATLAB `mySmooth` function: a `gausswin(15)` with the first half zeroed, reflect-padded, and convolved.

ii.
```python
TMIN,TMAX,DT=-3.0,2.5,0.01
EDGES=np.arange(TMIN,TMAX+DT/2,DT)
# ...
mat[pos[old]]=np.histogram(z,EDGES)[0]/DT
# ...
def causal_smooth(x, n=15):
    idx=np.linspace(-1.0,1.0,int(n),dtype=float)
    k=np.exp(-0.5*(2.5*idx)**2)
    k[:len(k)//2]=0.0; k/=k.sum()
    out=np.empty_like(x,dtype=np.float32)
    for i,row in enumerate(x):
        padded=np.concatenate([row[:n][::-1],row])
        y=np.convolve(padded,k,mode='same')
        out[i]=y[n:]
    return out
```

iii. The AI explicitly verified it was matching the reference MATLAB code's `mySmooth` function and corrected its initial implementation during Step 10 review.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI uses a **whitelist** approach: quality labels are lower-cased and stripped, and only units whose quality is in `{'poor','fair','good','great','excellent','multi'}` are retained. Then a firing-rate threshold of strictly >1 Hz is applied. The firing rate for this threshold is computed from a PSTH over all valid trials (not just the decoder-retained trials), smoothed with the causal kernel, and averaged over time.

ii.
```python
VALID_QUAL={'poor','fair','good','great','excellent','multi'}
# ...
q=h5str(f,rq).lower().strip()
if q not in VALID_QUAL: continue
# ...
# Low-FR curation:
rates=[]
for rt,ri,rq in zip(...):
    q=h5str(f,rq).lower().strip()
    if q not in VALID_QUAL: continue
    psth=np.histogram(aligned,EDGES)[0].astype(np.float32)/(len(go)*DT)
    rates.append(float(np.mean(causal_smooth(psth[None,:])[0])))
use=np.asarray(rates)>1.0
```

iii. The AI documented that it retains "poor" quality units, unlike the reference solution which drops them. The AI's rationale: "Exclude garbage, noisy, null/unknown and real?; retain curated poor/fair/good/great/excellent/multi labels."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Go cue times are subtracted from spike times before histogramming: `trialtm - goCue[trial]`.

ii.
```python
z=tm[tr==old]-go[old]
mat[pos[old]]=np.histogram(z,EDGES)[0]/DT
```

iii. Matches the reference `alignSpikes` approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins spanning -3.0 to +2.5 s, giving 550 time bins. This differs from the reference solution which uses 5 ms bins from -2.5 to +2.5 s (1000 bins).

ii.
```python
TMIN,TMAX,DT=-3.0,2.5,0.01
EDGES=np.arange(TMIN,TMAX+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

iii. The AI chose Figure 8's temporal parameters (-3..2.5 s, 10 ms) over the generic defaults (-2.5..2.5 s, 5 ms), arguing these are "directly applicable to WC/DR context analyses."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Derived from the bin centers of the temporal grid, which are themselves defined by the go-cue-aligned time window.

ii.
```python
CENTERS=(EDGES[:-1]+EDGES[1:])/2
# ...
inputs.append(CENTERS[None,:].astype(np.float32))
```

iii. The input is the time axis itself, not derived from raw data variables.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The bin centers are computed as the midpoints of the bin edges. No further processing.

ii.
```python
CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the same bin-center array used for the neural data's temporal grid, so it is inherently aligned.

ii.
```python
inputs.append(CENTERS[None,:].astype(np.float32))
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI uses actual lick event times: `bp.ev.lickL` and `bp.ev.lickR`, which are per-trial arrays of lick timestamps. The first post-go-cue lick from each side determines direction.

ii.
```python
d['lickL']=cell_arrays(f,ev['lickL']); d['lickR']=cell_arrays(f,ev['lickR'])
# ...
l=first_post_go(b['lickL'][old],b['go'][old])
r=first_post_go(b['lickR'][old],b['go'][old])
lick.append(0 if l<r else (1 if r<l else 2))
```

iii. The AI argues: "Decoder asks for lick direction, so target side cannot stand in for emitted behavior. No post-go lick is `none`." This differs from the reference which derives direction from hit/miss + instructed side (R/L).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, the AI finds the first post-go-cue lick on each side. If the left lick comes first, direction is left (0); if right comes first, direction is right (1); if neither or tied, direction is none (2).

ii.
```python
def first_post_go(a,go):
    x=np.atleast_1d(a).astype(float); x=x[np.isfinite(x)&(x>=go)]
    return np.min(x) if x.size else np.inf

l=first_post_go(b['lickL'][old],b['go'][old])
r=first_post_go(b['lickR'][old],b['go'][old])
lick.append(0 if l<r else (1 if r<l else 2))
```

iii. The AI uses actual behavioral events rather than inferring from outcome and instructed side.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The `bp.autowater` field, read from the behavior structure.

ii.
```python
d['autowater']=get('autowater')
# ...
context.append(1 if b['autowater'][old] else 0)
```

iii. Same source as the reference.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct mapping: `autowater=0` maps to DR (code 0), `autowater=1` maps to WC (code 1). This is the **opposite** encoding from the reference (which uses WC=0, DR=1).

ii.
```python
context.append(1 if b['autowater'][old] else 0)
# output_values: ['DR','WC']  -- so index 0=DR, 1=WC
```

iii. The AI's encoding is internally consistent with its output_values declaration.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit`, `bp.miss`, and `bp.no` fields.

ii.
```python
d={k:get(k) for k in ['L','R','hit','miss','no','early','autowater']}
# ...
outcome.append(0 if b['miss'][old] else (1 if b['hit'][old] else 2))
```

iii. Same source variables as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. miss=0 (incorrect), hit=1 (correct), neither=2 (ignore). Matches the reference encoding.

ii.
```python
outcome.append(0 if b['miss'][old] else (1 if b['hit'][old] else 2))
```

iii. Direct relabelling of the flags.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI uses `obj.traj` tracking data from camera 0 (side camera only), matching features with pattern `'tongue'`. Unlike the reference which uses both side (`tongue`) and bottom (`top_tongue`) cameras.

ii.
```python
tongue=speed_for(0,['tongue'])
```

iii. The AI only uses the side camera for tongue tracking, not both cameras.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each matching feature: extract x, y coordinates; mark frames with likelihood <= 0.9 as invisible; compute `np.gradient(x, ft)` and `np.gradient(y, ft)` **without smoothing first**; compute speed as hypot; set invisible frames to NaN; resample to bins using **nearest-frame** interpolation (not mean binning). If multiple tongue features match, their nearest-bin speeds are averaged.

ii.
```python
dx=np.gradient(x,ft);dy=np.gradient(y,ft); sp=np.hypot(dx,dy);sp[~visible]=np.nan
speeds.append(nearest_bin(ft,sp))
```

iii. The AI does not smooth x,y positions before differentiating, unlike the reference which applies a 5ms Gaussian within contiguous runs of valid frames. The AI also uses nearest-frame resampling rather than mean binning.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Session-wide 50th percentile of valid (non-NaN) values; below = 0, at-or-above = 1, NaN = 2.

ii.
```python
def discretize_session(a):
    a=np.asarray(a,float); valid=np.isfinite(a); med=float(np.nanpercentile(a,50)) if valid.any() else np.nan
    y=np.full(a.shape,2,np.int64); y[valid & (a<med)]=0; y[valid & (a>=med)]=1
    return y,med
```

iii. Matches the instruction's per-session 50th percentile threshold.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are read from `traj.frameTimes` and the go cue is subtracted directly: `frameTimes - goCue`. The AI does **not** compute or apply a video-to-behavior clock offset (bitcode correction), unlike the reference which computes the offset from `sglx.bitcode.bitstart`.

ii.
```python
ft=np.asarray(f[ftrefs[old]],float).squeeze()-go[old]
```

iii. The AI does not mention or address the video clock offset. This is a potential temporal misalignment between video-derived outputs and neural data.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI uses `obj.traj` from camera 1 (bottom camera), matching features with pattern `'paw'`. This pattern would match both `top_paw` and `bottom_paw` if both are present.

ii.
```python
paw=speed_for(1,['paw'])
```

iii. The reference uses only `top_paw` from the bottom camera. The AI's pattern matching could include both paw features.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue: extract x,y; visibility filter at likelihood > 0.9; compute gradient without smoothing; compute speed magnitude; nearest-frame resampling. If multiple paw features match, their speeds are averaged.

ii.
```python
dx=np.gradient(x,ft);dy=np.gradient(y,ft); sp=np.hypot(dx,dy);sp[~visible]=np.nan
speeds.append(nearest_bin(ft,sp))
```

iii. Same processing differences as tongue (no smoothing, nearest-bin resampling).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: session 50th percentile split; below=0, at-or-above=1, not-visible=2.

ii.
```python
pv,pmed=discretize_session(paw)
```

iii. Matches instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frameTimes - goCue, with **no** video clock offset correction.

ii.
```python
ft=np.asarray(f[ftrefs[old]],float).squeeze()-go[old]
```

iii. Missing video offset, same issue as tongue.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `motionEnergy_<subject>_<date>.mat` files from the `Ephys_Behavior` folder. Loaded via `scipy.io.loadmat`.

ii.
```python
p=ROOT/f'motionEnergy_{subject}_{date}.mat'
me=loadmat(p,simplify_cells=True)['me']; data=np.atleast_1d(me['data'])
```

iii. Same source as reference but only in Ephys_Behavior folder (matches 12-session subset).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. No processing beyond resampling. The per-trial motion energy values are resampled to the bin grid using **nearest-frame** interpolation (not mean binning as in the reference). Frame times from the side camera provide the temporal reference.

ii.
```python
v=np.asarray(data[old],float).squeeze(); t=np.asarray(ft,float).squeeze()
n=min(len(v),len(t)); out.append(nearest_bin(t[:n],v[:n]))
```

iii. The AI notes motion energy is already computed upstream.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same discretization: session 50th percentile; below=0, at-or-above=1, no-video/NaN=2.

ii.
```python
mv,mmed=discretize_session(motion)
```

iii. Matches instructions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the side camera's frame times minus the go cue. Same issue: **no** video clock offset correction applied.

ii.
```python
def motion_continuous(subject,date,trial_ids,go,frame_times):
    # frame_times already computed as: ft = frameTimes - go[old]
    v=np.asarray(data[old],float).squeeze(); t=np.asarray(ft,float).squeeze()
    n=min(len(v),len(t)); out.append(nearest_bin(t[:n],v[:n]))
```

iii. Missing video offset, same concern.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) If a session lacks a motion energy file, all trials get NaN which becomes class 2 (no video). (2) If trial index exceeds motion energy length, that trial gets NaN. (3) The `nearest_bin` function returns NaN for bins outside the frame time range or where input values are non-finite. (4) The `outcome_sum == 1` check filters out trials with ambiguous outcomes. (5) Frame times that are non-finite are filtered in `nearest_bin`.

ii.
```python
if not p.exists(): return [np.full(CENTERS.size,np.nan) for _ in trial_ids]
# ...
if old>=len(data):out.append(np.full(CENTERS.size,np.nan));continue
# ...
ok=np.isfinite(t); t=t[ok]; v=v[ok]
out=np.full(CENTERS.size,np.nan)
if t.size<2:return out
```

iii. The AI keeps trials even when video data is missing and assigns the "not visible" / "no video" class.

## 11-a. What are the most time-consuming steps of the code?

i. Loading files dominates. Per the conversion output, each session takes 1.9-4.3 seconds, with full conversion in ~31 seconds for 12 sessions.

ii.
```python
with h5py.File(p,'r') as f:
    b=behavior(f)
    neural,qualities,_=load_clusters(f,probe,b['go'],ids)
    vid=video_continuous(f,ids,b['go'])
```

iii. File I/O and HDF5 dereferencing are the bottleneck.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron spike histogramming loops over clusters individually. The per-trial video processing loops over trials individually due to variable frame counts. The causal smoothing loops over rows. The reference solution vectorized spike counting with a single `histogram2d` call.

ii.
```python
for old in np.unique(tr):
    if old not in pos or old<0 or old>=len(go): continue
    z=tm[tr==old]-go[old]
    mat[pos[old]]=np.histogram(z,EDGES)[0]/DT
# ...
for i,row in enumerate(x):
    padded=np.concatenate([row[:n][::-1],row])
    y=np.convolve(padded,k,mode='same')
    out[i]=y[n:]
```

iii. These loops exist due to irregular trial/frame sizes but could potentially be vectorized.

## 11-c. What processing does the code repeat multiple times?

i. The cluster iteration is done **twice** in `load_clusters`: once to build the spike-count matrices and once to compute firing rates for the low-FR threshold. Both loops iterate over all clusters, read the same HDF5 references, decode quality strings, and filter by quality. This is inefficient.

ii.
```python
mats=[]; qualities=[]
for rt,ri,rq in zip(np.asarray(g['trialtm']).ravel(),np.asarray(g['trial']).ravel(),np.asarray(g['quality']).ravel()):
    # ... first pass: build spike count matrices
# ...
rates=[]
for rt,ri,rq in zip(np.asarray(g['trialtm']).ravel(),np.asarray(g['trial']).ravel(),np.asarray(g['quality']).ravel()):
    # ... second pass: compute firing rates for threshold
```

iii. The double pass is because the firing rate threshold is computed differently (using all trials, not just decoder-retained trials), but the cluster loading and quality checking is repeated.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `lickL` and `lickR` event arrays for all trials (including filtered ones). The `haveVid` flag is loaded but not used in the final filtering. All behavior fields are loaded even for early/stim trials that are filtered out. The neural data is built for all valid-quality clusters before the firing-rate filter removes some.

ii.
```python
d['lickL']=cell_arrays(f,ev['lickL']); d['lickR']=cell_arrays(f,ev['lickR'])
d['haveVid']=np.asarray(trials['haveVid']).squeeze().astype(bool)
```

iii. These are minor inefficiencies inherent in loading all data before filtering.
