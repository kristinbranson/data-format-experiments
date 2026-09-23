# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only 12 sessions from the `Ephys_Behavior` folder, corresponding to the Figure 8 "two-context" subset identified in the authors' `load<ANM>_ALMVideo.m` loader scripts. Sessions are hard-coded in a `SESSIONS` list as `(subject, date, probe)` tuples. Each session's `.mat` file is opened with `h5py` (v7.3 HDF5 format only). Motion energy is loaded from separate `motionEnergy_*.mat` files via `scipy.io.loadmat`. The AI does not include the 25 fixed-delay-only sessions or the 19 randomized-delay sessions.

ii.
```python
ROOT=Path('/app/data/Ephys_Behavior')
SESSIONS=[('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),('JGR2','2021-11-16',1),
 ('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),('JEB19','2023-04-18',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-20',1),('JEB19','2023-04-21',1)]
...
with h5py.File(path,'r') as f:
    b=f['obj/bp']; ntr=int(np.asarray(b['Ntrials'][()]).squeeze())
```

iii. The AI's CONVERSION_NOTES explain that the 12 sessions are the "Figure 8 two-context subset" where both WC and DR contexts are present, matching the paper's stated 12 two-context sessions and 522 units. The AI argues that including randomized-delay sessions would not provide valid WC/DR context labels.

## 1-b. How are the data split into subjects?

i. Subject ID is extracted from the session tuple (the first element, e.g., `'JEB6'`). Unique subjects are accumulated in order of first appearance. `subject_idx` maps each session to its position in the subjects list. The AI identifies 7 unique subjects across the 12 sessions.

ii.
```python
if sub not in subjects:subjects.append(sub)
sidx.append(subjects.index(sub))
```

iii. The AI notes that while the paper states 6 mice for two-context sessions, the released data/loaders have 7 animal IDs. This discrepancy is documented but not resolved by dropping an animal.

## 1-c. How are the data split into sessions?

i. Each session is one `(subject, date, probe)` tuple in the hard-coded `SESSIONS` list. One `.mat` file corresponds to one session. The AI only accesses `Ephys_Behavior` (not `RandomizedDelay_Ephys_Behavior`). This yields 12 sessions total.

ii.
```python
sessions=SESSIONS[:2] if args.sample else SESSIONS
for i,(sub,date,probe) in enumerate(sessions):
    n,x,y,info,cont=convert_session(sub,date,probe,args.show_processing)
```

iii. The AI selected these 12 sessions based on the Figure 8 ALM/video loader scripts in the reference code, which define the two-context analysis subset.

## 1-d. How are the data split into trials?

i. Trials are indexed 0-based from 0 to `Ntrials-1`. The trial count comes from `bp['Ntrials']`. Behavioral flags (`R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`) are read as boolean vectors of length `Ntrials`. Spike data carries trial indices that map each spike to its trial.

ii.
```python
b=f['obj/bp']; ntr=int(np.asarray(b['Ntrials'][()]).squeeze())
get=lambda k:vec(b[k],bool)
R,L,hit,miss,no,early,aw=map(get,['R','L','hit','miss','no','early','autowater'])
```

iii. The AI uses the Bpod structure's `Ntrials` field and per-trial arrays to define trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by two criteria: (1) photostimulation trials are excluded (`~stim`), and (2) trials must have a finite go cue (`np.isfinite(go)`). Notably, early-lick trials are NOT excluded. The AI retains early-lick, miss, and no-response trials because the decoder output explicitly requires `ignore` as an outcome class.

ii.
```python
stim=np.zeros(ntr,bool)
if isinstance(b.get('stim'),h5py.Group) and 'enable' in b['stim']:stim=vec(b['stim/enable'],bool)
keep_trials=np.flatnonzero((~stim)&np.isfinite(go))
```

iii. The AI's CONVERSION_NOTES Step 5 states: "Retain hit, miss, no-response, and early trials with valid go cues because the requested output explicitly includes incorrect and ignore." The AI explicitly decided to keep early-lick trials as part of the `ignore` outcome class. However, the AI's trajectory (Step 38) also noted that the Figure 8 conditions exclude early-lick trials (`~early`).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `obj.clu` spike-sorted clusters from the selected probe. Each cluster has `trial` (1-based trial indices) and `trialtm` (within-trial spike times), plus a `quality` label. The go cue times `bp.ev.goCue` provide alignment.

ii.
```python
clu=f['obj/clu']; cg=f[clu[()].ravel(order='F')[probe-1]]
qualities=np.array([chars(f,r).lower() for r in cg['quality'][()].ravel(order='F')])
trialrefs=cg['trial'][()].ravel(order='F'); tmrefs=cg['trialtm'][()].ravel(order='F')
```

iii. The AI reads clusters from the specific probe designated in the session list, matching the reference loader scripts.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue by subtraction, then histogrammed into 10 ms bins from -2.5 to +2.5 s (500 bins). Counts are divided by the bin width (0.01 s) to convert to Hz. The resulting firing rates are then smoothed with a uniform (boxcar) filter of width 15 bins (150 ms at 10 ms resolution) using `uniform_filter1d(size=15, mode='nearest')`.

ii.
```python
DT=.01; TMIN=-2.5; TMAX=2.5
EDGES=np.arange(TMIN,TMAX+DT/2,DT); TIME=(EDGES[:-1]+EDGES[1:])/2
...
z=tm[(tr==old)]-go[old]
mats[remap[old]]=np.histogram(z,EDGES)[0]/DT
...
neural=uniform_filter1d(neural,size=15,axis=2,mode='nearest').astype(np.float32)
```

iii. The AI cites the Figure 8a-c population context analysis parameters (`params.dt=1/100` for 10 ms) and interprets `params.smooth = 15` as a boxcar filter of width 15 bins. The AI's code comment says "Reference mySmooth is a centered boxcar."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied. First, clusters whose quality label (lowercased) matches the artifact set `{'garbage', '', 'noisy', 'real?'}` are excluded. Second, clusters must have a mean firing rate strictly >1 Hz over the full analysis window. The firing rate is computed across ALL native trials (not just kept trials), as `total_spikes_in_window / (ntr * (TMAX - TMIN))`.

ii.
```python
ARTIFACT={'garbage','','noisy','real?'}
...
aligned_all=tm-go[tr]
rate=np.sum((aligned_all>=TMIN)&(aligned_all<TMAX))/(ntr*(TMAX-TMIN))
if q not in ARTIFACT and rate>1:
```

iii. The AI's CONVERSION_NOTES explain that the artifact labels follow `findClusters.m`, and the rate threshold follows the paper's statement about units >1 Hz. The AI notes that the rate is computed over all native trials "before the decoder-specific exclusion of stimulation trials."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time (`trialtm`) has the trial's go cue time (`bp.ev.goCue`) subtracted, putting spike times in seconds relative to go cue onset. The aligned times are then histogrammed into the bin grid.

ii.
```python
go=vec(b['ev/goCue'],float)
...
z=tm[(tr==old)]-go[old]
mats[remap[old]]=np.histogram(z,EDGES)[0]/DT
```

iii. This matches the reference code's `alignSpikes.m` which does `trialtm_aligned = trialtm - event`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins, producing 500 time points per trial over -2.5 to +2.5 s. No rebinning is applied; spikes are directly histogrammed into this grid.

ii.
```python
DT=.01; TMIN=-2.5; TMAX=2.5
EDGES=np.arange(TMIN,TMAX+DT/2,DT); TIME=(EDGES[:-1]+EDGES[1:])/2
```

iii. The AI chose 10 ms based on the Figure 8a-c population analysis (`params.dt=1/100`), distinguishing this from the 5 ms used in Figure 8d's single-unit analysis.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the array of bin centers derived from the time grid constants, not from any raw data variable. It is the same for every trial.

ii.
```python
EDGES=np.arange(TMIN,TMAX+DT/2,DT); TIME=(EDGES[:-1]+EDGES[1:])/2
...
[TIME[None,:].astype(np.float32).copy() for _ in keep_trials]
```

iii. The time vector is defined analytically from the bin edges.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing. The bin centers are computed once from the edge array and replicated for every trial.

ii. Same as 3-a.

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time vector is the same bin-center grid onto which neural spike counts are histogrammed, so alignment is inherent.

ii.
```python
TIME=(EDGES[:-1]+EDGES[1:])/2
```

iii. Both neural and input share the same grid by construction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `bp.R` (instructed right), `bp.hit`, `bp.miss`, and `bp.no` (no-response). These are boolean per-trial flags.

ii.
```python
R,L,hit,miss,no,early,aw=map(get,['R','L','hit','miss','no','early','autowater'])
```

iii. Lick direction is not directly recorded; it must be inferred from the instructed side and the outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. On hit trials, the animal licked the instructed side (right if R, left otherwise). On miss trials, the animal licked the opposite side. On no-response trials, lick direction is "none" (code 2). The encoding is left=0, right=1, none=2.

ii.
```python
if no[tr]:lick=2
elif hit[tr]:lick=1 if R[tr] else 0
elif miss[tr]:lick=0 if R[tr] else 1
else:lick=2
```

iii. This correctly inverts the instructed side for miss trials to get the actual lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater` boolean flag. Autowater=True indicates WC (water-cued) context; autowater=False indicates DR (delayed-response) context.

ii.
```python
aw=get('autowater')
...
context=0 if aw[tr] else 1
```

iii. This follows the paper's definition where WC blocks have autowater enabled.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater=True -> WC (code 0), autowater=False -> DR (code 1). The value is repeated across all time bins.

ii.
```python
context=0 if aw[tr] else 1
...
y=np.vstack([...,np.full(len(TIME),context),...])
```

iii. Consistent with the decoder specification's WC=0, DR=1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit` and `bp.miss` boolean flags. The AI also reads `bp.no` but uses it only for lick direction; outcome uses the same hit/miss flags with everything else becoming ignore.

ii.
```python
outcome=1 if hit[tr] else (0 if miss[tr] else 2)
```

iii. Hit maps to correct (1), miss to incorrect (0), everything else to ignore (2).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct conditional mapping: hit -> correct (1), miss -> incorrect (0), all other -> ignore (2). The value is constant across time bins.

ii.
```python
outcome=1 if hit[tr] else (0 if miss[tr] else 2)
...
np.full(len(TIME),outcome)
```

iii. Matches the decoder specification's encoding.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DLC tracking data in `obj.traj`, specifically the side camera view (`views[0]`), feature name `'tongue'`. The tracking provides x, y coordinates, likelihood, and frame timestamps per trial.

ii.
```python
ft,pos,lk=feature_trial(f,views[0],tr,'tongue')
tongue[j]=velocity_on_grid(ft,pos,lk,go[tr])
```

iii. The AI uses only the side camera for the tongue. The reference solution uses both the side camera (`tongue`) and bottom camera (`top_tongue`) and averages them after normalizing by each view's 90th percentile.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Five steps: (1) Frames with likelihood <= 0.9 or non-finite positions are marked invisible. (2) Positions (including NaN positions) are filled by linear interpolation, then smoothed with a uniform (boxcar) filter of size 21 frames. (3) Speed is computed via finite differences (`np.diff`) divided by frame time intervals. (4) Speed at invisible frames is set to NaN. (5) The speed is interpolated onto the analysis time grid using `np.interp`, then visibility gaps are restored by nearest-frame lookup. Per-session median threshold splits into below (0), at-or-above (1), and not-visible (2).

ii.
```python
pp=pos.copy()
for j in range(2):
    good=np.isfinite(pp[:,j])
    if good.sum()>=2:
        fill=np.interp(np.arange(len(pp)),np.flatnonzero(good),pp[good,j])
        pp[:,j]=uniform_filter1d(fill,size=21,mode='nearest')
dt=np.diff(ft); speed=np.r_[np.nan,np.sqrt(np.sum(np.diff(pp,axis=0)**2,axis=1))/np.where(dt>0,dt,np.nan)]
speed[~visible]=np.nan; rel=ft-go
...
out[inside]=np.interp(TIME[inside],rel[good],speed[good])
```

iii. The AI comment says "Reference smooths positions over 21 frames before finite differencing." The interpolation of NaN positions before smoothing differs from the reference, which only smoothes within contiguous valid runs.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The session median of all finite tongue velocity values is computed. Values below the median are class 0, at or above are class 1, and NaN (not visible) is class 2.

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

iii. Uses median (50th percentile) as specified in the instructions. The reference uses `np.nanpercentile(tongue, 50)`, which is equivalent.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are directly subtracted by the go cue time (`rel = ft - go`) and then interpolated onto the analysis time grid. No video-to-behavior clock offset correction (bitcode offset) is applied.

ii.
```python
rel=ft-go
...
out[inside]=np.interp(TIME[inside],rel[good],speed[good])
```

iii. The AI does not compute or apply the video offset that the reference code derives from bitcode timing (`findVideoOffset.m`). The reference subtracts both the video offset and the go cue time from frame timestamps.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DLC tracking from the bottom camera (`views[1]`), feature `'top_paw'`. The tracking provides x, y, likelihood, and frame timestamps.

ii.
```python
ft,pos,lk=feature_trial(f,views[1],tr,'top_paw')
paw[j]=velocity_on_grid(ft,pos,lk,go[tr])
```

iii. Uses the bottom camera view for the paw, matching the reference which also uses `top_paw` from the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same processing as tongue velocity: interpolate NaN positions, smooth with 21-frame boxcar, compute speed via finite differences, interpolate onto the time grid, restore visibility gaps.

ii. Same `velocity_on_grid` function as tongue (see 7-b).

iii. Same processing pipeline for all velocity-based features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: session median of finite values, below=0, at-or-above=1, not-visible=2.

ii.
```python
pd,pmed=disc(paw)
```

iii. Consistent with instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times minus go cue, interpolated onto the time grid. No video offset correction.

ii. Same `velocity_on_grid` function (see 7-d).

iii. Same alignment approach as tongue.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_<subject>_<date>.mat` files loaded via `scipy.io.loadmat`. The AI looks for these files only in `ROOT = /app/data/Ephys_Behavior`. The motion energy data is a per-trial array of values, one per camera frame.

ii.
```python
def load_motion(subject,date,ntrials):
    p=ROOT/f'motionEnergy_{subject}_{date}.mat'
    if not p.exists():return None
    ...
    m=loadmat(p,squeeze_me=True,struct_as_record=False)['me']
    raw=np.asarray(m.data,dtype=object).ravel(order='F')
```

iii. The AI loads motion energy from a separate file, unwrapping the MATLAB struct to get per-trial arrays. If the file doesn't exist, the session gets all-NaN motion energy (class 2 = "no video").

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-trial motion energy trace is aligned to go cue using the side camera's frame timestamps (no video offset correction). Values are interpolated onto the time grid using `np.interp`. Then the session median splits into categories.

ii.
```python
y=me_raw[tr]; ft=video_times[j]; n=min(len(y),len(ft)); y=y[:n]; rel=ft[:n]-go[tr]
good=np.isfinite(y)&np.isfinite(rel)
if good.sum()>1:
    inside=(TIME>=rel[good].min())&(TIME<=rel[good].max())
    me[j,inside]=np.interp(TIME[inside],rel[good],y[good])
```

iii. Motion energy values are already computed upstream (one per frame), so only alignment and discretization are needed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Session median of finite values, same discretization as velocity outputs.

ii.
```python
md,mmed=disc(me)
```

iii. Consistent with instructions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The side camera's frame timestamps are used (`video_times[j]`), with go cue subtracted but no video offset correction. Values are interpolated onto the time grid.

ii.
```python
rel=ft[:n]-go[tr]
...
me[j,inside]=np.interp(TIME[inside],rel[good],y[good])
```

iii. Same alignment as velocity streams, no bitcode offset correction.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) If a feature is not found in a camera view, `feature_trial` returns `None` and `velocity_on_grid` returns all-NaN (class 2). (2) If motion energy file is missing, the session gets all-NaN motion energy. (3) If motion energy trial count doesn't match `Ntrials`, all-NaN is returned. (4) Frames with likelihood <= 0.9 or non-finite positions are marked invisible (NaN speed, class 2). (5) If `stim` field doesn't exist in `bp`, it defaults to all-False.

ii.
```python
if ft is None or len(ft)<2:return out  # all NaN
...
if not p.exists():return None  # no motion energy file
if len(raw)!=ntrials:return None  # trial count mismatch
...
stim=np.zeros(ntr,bool)
if isinstance(b.get('stim'),h5py.Group) and 'enable' in b['stim']:stim=vec(b['stim/enable'],bool)
```

iii. Missing data is preserved as NaN and mapped to the "not visible" / "no video" class rather than filled or interpolated.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the HDF5 files via h5py with targeted reference traversal. The AI's CONVERSION_NOTES report ~3 seconds per session for the sample, estimating under 1 minute for all 12 sessions. The full conversion completed quickly.

ii.
```python
with h5py.File(path,'r') as f:
    ...  # all processing happens within this context
```

iii. The AI uses targeted HDF5 access rather than full deserialization, which was initially tried but stalled.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop over clusters reads each unit's spike times and histograms them individually. Each trial's velocity computation is also done in a per-trial loop. These could potentially be vectorized but the ragged array sizes make it non-trivial.

ii.
```python
for q,rr,rt in zip(qualities,trialrefs,tmrefs):
    ...
    for old in np.unique(tr[use]):
        z=tm[(tr==old)]-go[old]
        mats[remap[old]]=np.histogram(z,EDGES)[0]/DT
```

iii. The reference solution vectorizes spike counting into a single `np.histogram2d` call over all trials, which is more efficient.

## 11-c. What processing does the code repeat multiple times?

i. The side camera frame times (`video_times`) are loaded both for the tongue velocity computation (via `feature_trial`) and stored separately for motion energy alignment. This means frame times are read twice for the side camera per trial.

ii.
```python
ft,pos,lk=feature_trial(f,views[0],tr,'tongue')  # reads frameTimes
...
video_times.append(deref_array(f,views[0]['frameTimes'],tr)...)  # reads again
```

iii. This redundancy is minor since the file is already open.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI interpolates NaN positions before smoothing for velocity computation. This interpolated data is then masked back to NaN at invisible frames, so the interpolation across gaps is discarded for those frames. Additionally, the `early` flag is loaded but never used for filtering.

ii.
```python
fill=np.interp(np.arange(len(pp)),np.flatnonzero(good),pp[good,j])
pp[:,j]=uniform_filter1d(fill,size=21,mode='nearest')
...
speed[~visible]=np.nan  # restored to NaN
```

iii. The interpolation step is used to allow smoothing across the full trace, but the invisible portions are discarded afterward.
