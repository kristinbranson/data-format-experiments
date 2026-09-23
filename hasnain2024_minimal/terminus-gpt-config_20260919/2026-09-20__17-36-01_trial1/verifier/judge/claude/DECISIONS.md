# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only 12 sessions from the `Ephys_Behavior` folder (fixed-delay task only), selected to match the paper's Figure 8 two-context analysis subset. Each session is loaded as a MATLAB v7.3 HDF5 file using `h5py`. The 12 sessions and their probe selections are hardcoded in the `SESSIONS` list. The `RandomizedDelay_Ephys_Behavior` folder is not used. Only a single `ROOT` path is defined pointing to `Ephys_Behavior`.

ii.
```python
ROOT='/app/data/Ephys_Behavior'
SESSIONS=[
 ('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),
 ('JGR2','2021-11-16',1),('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),
 ('JEB19','2023-04-21',1),('JEB19','2023-04-20',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-18',1)]

# Loading:
path=f'{ROOT}/data_structure_{animal}_{date}.mat'
with h5py.File(path,'r') as f:
```

iii. The AI identified these 12 sessions by reading the paper's Figure 8 analysis scripts (`Figure8a_thru_c.m`) and their associated data loading scripts, which specify the curated two-context subset of sessions. The AI's trajectory (steps 20-24) shows it systematically identified these sessions from the paper's metadata loaders.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the animal name in the session tuple (first element). The AI builds the `subjects` list by iterating through sessions in order and adding new animal names as encountered.

ii.
```python
subjects=[]; subject_idx=[]
for animal,_,_ in SESSIONS:
    if animal not in subjects:subjects.append(animal)
    subject_idx.append(subjects.index(animal))
```

iii. The animal ID is taken directly from the session metadata tuples. This results in 7 unique subjects from the 12 sessions (EKH1, EKH3, JEB6, JEB7, JEB19, JGR2, JGR3 — though ordered by first appearance).

## 1-c. How are the data split into sessions?

i. Each entry in the `SESSIONS` list corresponds to one session. Each session is one `.mat` file loaded with `h5py`. Sessions are processed sequentially and each becomes one element in the `neural`, `input`, and `output` lists. Only the `Ephys_Behavior` folder is searched; the `RandomizedDelay_Ephys_Behavior` folder is excluded entirely.

ii.
```python
for s in SESSIONS:
    a,b,c,i=convert_session(*s)
    neural.append(a); inp.append(b); out.append(c); infos.append(i)
```

iii. The AI selected these 12 sessions as the paper's curated two-context subset from Figure 8 analysis code, which requires sessions with both WC and DR contexts.

## 1-d. How are the data split into trials?

i. Trials are identified from `obj.bp.Ntrials`, which gives the total number of trials. The AI then filters to keep trials where `hit | miss | no` is true and the go cue time is finite. Trial indices are tracked as `tids = np.flatnonzero(keep)`.

ii.
```python
bp=f['obj/bp']; n=int(np.asarray(bp['Ntrials']).squeeze())
label={k:np.asarray(bp[k]).ravel().astype(bool) for k in ['R','L','hit','miss','no','early','autowater']}
go=np.asarray(bp['ev/goCue']).ravel().astype(float)
keep=(label['hit']|label['miss']|label['no']) & np.isfinite(go)
tids=np.flatnonzero(keep)
```

iii. The AI explicitly references "Figure 8 condition 1: hit|miss|no" in the code comment, which retains all outcome types (hits, misses, and no-response/ignore trials).

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials based on outcome type (hit, miss, or no-response must be true) and requires finite go cue times. Early-lick trials are NOT explicitly excluded — the code reads `early` into labels but does not use it in the `keep` mask. Photostimulation trials (`stim.enable`) are also NOT filtered. The comment says "Keep early trials because that exact all-trial condition does not exclude them."

ii.
```python
keep=(label['hit']|label['miss']|label['no']) & np.isfinite(go)
tids=np.flatnonzero(keep)
```

iii. The AI's trajectory (step 22) notes that "Trial condition 1 includes hit, miss, and no outcomes, so ignore/no trials should indeed be retained." The AI chose not to exclude early-lick trials, reasoning that the Figure 8 condition 1 code does not explicitly exclude them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu` — specifically from `clu.trial` (which trial each spike belongs to, 1-based), `clu.trialtm` (spike time relative to trial start), and `clu.tm` (used for counting total units). The go cue times `bp.ev.goCue` are used for alignment. Only one probe per session is used, selected by the probe index in `SESSIONS`.

ii.
```python
clu=f[refs(f['obj/clu'])[probe-1]]
trial_cells=refs(clu['trial']); tm_cells=refs(clu['trialtm'])
alltm=refs(clu['tm'])
# ...
tr=arr(f,trial_cells[ui]).astype(int)-1
rel=arr(f,tm_cells[ui]).astype(float)
sp=rel[tr==ti]-go[ti]
```

iii. The AI identified these fields by inspecting the HDF5 structure and cross-referencing with the paper's processing code.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to go cue onset, binned at 10 ms into 550 bins spanning [-3.0, 2.5) seconds, converted to firing rate (Hz) by dividing by bin width, then smoothed with a causal Gaussian kernel. The causal kernel is MATLAB's `gausswin(15)` with the first 7 (floor(15/2)) entries zeroed out, then normalized to sum to 1. This matches the paper's `MySmooth`/`getPSTHs.m` causal smoothing.

ii.
```python
DT=.01; TMIN=-3.; TMAX=2.5
TIME=np.arange(TMIN,TMAX,DT); EDGES=np.arange(TMIN,TMAX+DT/2,DT)

def causal_smooth(x, n=15):
    k=gaussian(n, std=(n-1)/2/2.5, sym=True)
    k[:n//2]=0; k/=k.sum()
    return convolve1d(x, k, axis=-1, mode='constant', origin=0)

# Per unit, per trial:
counts=np.histogram(sp,bins=EDGES)[0].astype(np.float32)/DT
neural_trials[oj][ni]=causal_smooth(counts)
```

iii. The AI's trajectory (steps 22, 25) explicitly references `getPSTHs.m` and `MySmooth` for the causal Gaussian smoothing approach, and uses the paper's exact parameters: 10 ms bins, [-3, 2.5) time window, and width-15 causal Gaussian.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps ALL cluster qualities (no quality-based filtering). Units are filtered only by firing rate: units with mean rate > 1 Hz over the analysis window are kept. The firing rate is computed as the number of spikes in the [-3.0, 2.5) window across all trials divided by (n_trials × 5.5 seconds).

ii.
```python
for ui in range(len(alltm)):
    tr_all=arr(f,trial_cells[ui]).astype(int)-1
    rel_all=arr(f,tm_cells[ui]).astype(float)
    aligned=rel_all-go[tr_all]
    nwin=np.count_nonzero((aligned>=TMIN)&(aligned<TMAX))
    rate=nwin/(n*(TMAX-TMIN))
    if rate>1.: good.append(ui)
```

iii. The AI's trajectory (step 22) states: "remove units below 1 Hz, include all unit qualities" based on reading the Figure 8 parameters where `params.quality={'all'}`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is done by subtracting the go cue time from each spike's trial-relative time: `sp = rel[tr==ti] - go[ti]`. This puts each spike in seconds from go cue onset. The aligned spike times are then binned into the time grid.

ii.
```python
go=np.asarray(bp['ev/goCue']).ravel().astype(float)
# Per unit, per trial:
sp=rel[tr==ti]-go[ti]
counts=np.histogram(sp,bins=EDGES)[0].astype(np.float32)/DT
```

iii. This follows the same approach as the reference code's `alignSpikes.m`: subtract go cue time from trial-relative spike times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10 ms (DT=0.01), spanning [-3.0, 2.5) seconds, yielding 550 time bins. No rebinning is applied — spikes are directly binned at this resolution.

ii.
```python
DT=.01; TMIN=-3.; TMAX=2.5
TIME=np.arange(TMIN,TMAX,DT); EDGES=np.arange(TMIN,TMAX+DT/2,DT)
```

iii. The AI chose 10 ms bins based on the paper's `params.dt = 1/200` (which is 5 ms), but the Figure 8 code uses `params.dt = 0.01` (10 ms). The time window [-3.0, 2.5) matches `params.tmin = -3` and `params.tmax = 2.5` from the Figure 8 code.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself — the centers of the 550 time bins spanning [-3.0, 2.5) seconds at 10 ms resolution. It is not derived from any raw data variable but is constructed from the binning parameters.

ii.
```python
TIME=np.arange(TMIN,TMAX,DT)
time_row=TIME.astype(np.float32)[None,:]
inputs.append(time_row.copy())
```

iii. This is a constructed variable representing time from go cue onset, identical for every trial.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing — the time axis is directly constructed from the binning parameters using `np.arange`.

ii.
```python
TIME=np.arange(TMIN,TMAX,DT)
```

iii. N/A — constructed variable.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis IS the neural binning grid. Both neural data and this input share the same time bins, so they are inherently aligned.

ii.
```python
TIME=np.arange(TMIN,TMAX,DT)
# Neural uses the same edges:
EDGES=np.arange(TMIN,TMAX+DT/2,DT)
counts=np.histogram(sp,bins=EDGES)[0]
```

iii. N/A — same time grid by construction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `bp.L`, `bp.hit`, `bp.miss`, and `bp.no` — the instructed side and outcome flags. The AI uses `L` (left-instructed) to determine the instructed side, then combines with hit/miss/no to infer lick direction.

ii.
```python
label={k:np.asarray(bp[k]).ravel().astype(bool) for k in ['R','L','hit','miss','no','early','autowater']}
# ...
if label['no'][ti]: lick=2
elif label['hit'][ti]: lick=0 if label['L'][ti] else 1
else: lick=1 if label['L'][ti] else 0
```

iii. The AI derives lick direction from the combination of instructed side and outcome, same logic as the reference.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit means the animal licked the instructed side (so lick = instructed side). A miss means the animal licked the opposite side. A "no" trial means no lick (coded as 2). Left = 0, right = 1, none = 2. The logic is: if no-response, lick=2; if hit and L-instructed, lick=left(0); if hit and R-instructed, lick=right(1); if miss and L-instructed, lick=right(1, opposite); if miss and R-instructed, lick=left(0, opposite).

ii.
```python
if label['no'][ti]: lick=2
elif label['hit'][ti]: lick=0 if label['L'][ti] else 1
else: lick=1 if label['L'][ti] else 0
```

iii. Same derivation logic as the reference — lick direction is inferred from instructed side + outcome.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. Autowater trials are the water-cued (WC) context; non-autowater trials are the delayed-response (DR) context.

ii.
```python
label={k:np.asarray(bp[k]).ravel().astype(bool) for k in ['R','L','hit','miss','no','early','autowater']}
context=1 if label['autowater'][ti] else 0 # DR=0, WC=1
```

iii. The AI correctly identifies `autowater` as the context flag.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater=True maps to WC=1, autowater=False maps to DR=0. Note the coding is DR=0, WC=1, which is the OPPOSITE of the reference (WC=0, DR=1).

ii.
```python
context=1 if label['autowater'][ti] else 0 # DR=0, WC=1
```

iii. The AI chose DR=0, WC=1 coding. The output_values list confirms: `['DR','WC']`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit`, `bp.miss`, and `bp.no`. These three boolean flags are mutually exclusive and exhaustive.

ii.
```python
label={k:np.asarray(bp[k]).ravel().astype(bool) for k in ['R','L','hit','miss','no','early','autowater']}
outcome=2 if label['no'][ti] else (1 if label['hit'][ti] else 0)
```

iii. The AI reads all three outcome flags from the behavioral data.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct mapping: no-response → ignore(2), hit → correct(1), miss → incorrect(0). This matches the reference's coding scheme.

ii.
```python
outcome=2 if label['no'][ti] else (1 if label['hit'][ti] else 0)
```

iii. Matches the instruction's specification: incorrect=0, correct=1, ignore=2.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `obj.traj` — the DLC tracking data. The AI searches for features with 'tongue' in the name across both cameras. The tracking data includes x, y coordinates and likelihood per frame per feature.

ii.
```python
tong=[j for j,n in enumerate(names) if 'tongue' in n]
s,_=landmark_speed(ts,tong)
```

iii. The AI uses all features matching 'tongue' in each camera's feature list.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI computes tongue velocity using `landmark_speed`: (1) Filter frames by DLC likelihood >= 0.9. (2) For visible frames, average the x,y coordinates across all tongue landmarks. (3) Compute speed as frame-to-frame displacement times 400 (assumed frame rate). (4) Interpolate to the 10ms time grid using `interp_trace`. (5) Discretize at session median.

ii.
```python
def landmark_speed(ts, indices):
    xy=ts[indices,:2,:]
    lk=ts[indices,2,:]
    vis=lk>=0.9
    xy=np.where(vis[:,None,:],xy,np.nan)
    cx=np.nanmean(xy[:,0,:],axis=0); cy=np.nanmean(xy[:,1,:],axis=0)
    visible=np.any(vis,axis=0)
    speed=np.r_[np.nan, np.hypot(np.diff(cx),np.diff(cy))*400.]
    speed[~visible]=np.nan
    return speed,visible
```

iii. The AI computes velocity as displacement between consecutive frames multiplied by a fixed 400 Hz frame rate, then interpolates to the time grid. This differs from the reference which smooths x,y with a Gaussian before differentiating and uses actual frame timestamps for the gradient.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Discretized at the 50th percentile (session median) of all finite values. Below threshold = 0, at or above = 1, NaN/not visible = 2.

ii.
```python
def discretize_session(stream):
    finite=np.isfinite(stream)
    threshold=float(np.nanpercentile(stream,50)) if finite.any() else np.nan
    out=np.full(stream.shape,2,dtype=np.int16)
    out[finite & (stream<threshold)]=0; out[finite & (stream>=threshold)]=1
    return out,threshold
```

iii. Matches the instruction's specification for thresholding.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI uses a hardcoded 0.5 s offset subtracted from frame times: `old = ft - 0.5 - go[ti]`. The velocity values are then interpolated to the neural time grid using `np.interp`.

ii.
```python
old=ft-0.5-go[ti] # exact offset used by paper loadMotionEnergy/getPosition
# ...
def interp_trace(old_t, val):
    good=np.isfinite(old_t)&np.isfinite(val)
    out=np.full(TIME.size,np.nan,dtype=np.float32)
    if good.sum()>=2:
        out[:]=np.interp(TIME,old_t[good],val[good]).astype(np.float32)
        out[TIME < old_t[good].min()]=np.nan; out[TIME > old_t[good].max()]=np.nan
    return out
```

iii. The AI uses a fixed 0.5 s video offset based on the paper's `loadMotionEnergy` code comment. The reference computes the offset dynamically from bitcode timestamps.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from `obj.traj` — all features matching 'paw' in the feature names, across both cameras.

ii.
```python
paws=[j for j,n in enumerate(names) if 'paw' in n]
s,_=landmark_speed(ts,paws)
```

iii. The AI uses all paw-named landmarks from each camera. The reference uses only `top_paw` from the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same `landmark_speed` function as tongue: average x,y of all paw landmarks with likelihood >= 0.9, compute frame-to-frame displacement × 400, interpolate to time grid.

ii.
```python
s,_=landmark_speed(ts,paws)
if s is not None:
    pv=interp_trace(old,s)
```

iii. Same processing pipeline as tongue velocity.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same `discretize_session` function: 50th percentile threshold, below=0, above=1, NaN=2.

ii.
```python
d,th=discretize_session(x)
```

iii. Same as tongue velocity thresholding.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: hardcoded 0.5 s offset subtracted from frame times, then interpolated to the neural time grid.

ii.
```python
old=ft-0.5-go[ti]
pv=interp_trace(old,s)
```

iii. Same alignment approach as tongue velocity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI does NOT use the dedicated `motionEnergy_*.mat` files. Instead, it computes a motion energy proxy from aggregate visible-landmark motion across both cameras — averaging the frame-to-frame speed of ALL visible DLC landmarks.

ii.
```python
# Aggregate all confidently visible landmarks as image-motion proxy.
s,_=landmark_speed(ts,list(range(len(names))))
if s is not None: components.append(interp_trace(old,s))
# ...
me=np.nanmean(np.stack(components),axis=0).astype(np.float32)
```

iii. The AI's trajectory (step 28) states: "No raw video or precomputed pixel motion-energy files are supplied, so true paper motion energy cannot be reconstructed. A justified substitute is aggregate visible-landmark motion from both cameras." The code comments also note: "Motion energy is an aggregate landmark-motion proxy because released files contain no raw video/pixel-motion stream."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each camera and each trial: (1) compute `landmark_speed` over ALL features (all DLC landmarks), (2) interpolate to time grid, (3) average all components (tongue speeds, paw speeds, and all-landmark speeds across cameras) using `nanmean`. This produces an aggregate motion proxy.

ii.
```python
if components:
    with np.errstate(invalid='ignore'):
        me=np.nanmean(np.stack(components),axis=0).astype(np.float32)
else:
    me=np.full(TIME.size,np.nan,np.float32)
```

iii. This is a proxy rather than the true pixel-based motion energy the paper computes.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same `discretize_session` function: 50th percentile threshold, below=0, above=1, NaN=2.

ii.
```python
d,th=discretize_session(x)
```

iii. Same threshold approach as other velocity outputs.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same as other camera outputs: hardcoded 0.5 s offset from frame times, interpolated to neural time grid.

ii.
```python
old=ft-0.5-go[ti]
components.append(interp_trace(old,s))
```

iii. Same alignment as tongue and paw.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is handled by: (1) The `camera_trial` function returns `None` if trajectory data cannot be parsed (wrong dimensions, missing fields). (2) `interp_trace` returns all-NaN if fewer than 2 valid points. (3) If no camera components are available, motion energy defaults to all-NaN, which becomes class 2 after discretization. (4) The go cue finite check (`np.isfinite(go)`) in trial filtering removes trials with missing go cue times.

ii.
```python
def camera_trial(f,traj,ti):
    try:
        ts=arr(f,refs(traj['ts'])[ti]).astype(float)
        ft=arr(f,refs(traj['frameTimes'])[ti]).astype(float)
    except Exception:return None,None,None
    if ts.ndim!=3:return None,None,None

def interp_trace(old_t, val):
    out=np.full(TIME.size,np.nan,dtype=np.float32)
    if good.sum()>=2:
        out[:]=np.interp(TIME,old_t[good],val[good]).astype(np.float32)
```

iii. The AI handles missing data by propagating NaN values that are then converted to the "not visible" / "no video" category (class 2) during discretization.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the HDF5 files and processing video/trajectory data are the most time-consuming steps. Each session requires reading a large MATLAB file and processing DLC tracking for every trial across both cameras.

ii.
```python
with h5py.File(path,'r') as f:
    # ... all processing happens within this context manager
```

iii. The AI processes everything within the HDF5 file context, reading fields on-demand via h5py references.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The neural data processing has a triple-nested loop: over units, over trials, and over time bins. This could be vectorized using `np.histogram2d` (as the reference does) instead of per-trial `np.histogram` calls.

ii.
```python
for ni,ui in enumerate(good):
    tr=arr(f,trial_cells[ui]).astype(int)-1
    rel=arr(f,tm_cells[ui]).astype(float)
    for ti,oj in tid_to_out.items():
        sp=rel[tr==ti]-go[ti]
        counts=np.histogram(sp,bins=EDGES)[0].astype(np.float32)/DT
        neural_trials[oj][ni]=causal_smooth(counts)
```

iii. The per-unit, per-trial spike binning loop is the most obvious vectorization candidate.

## 11-c. What processing does the code repeat multiple times?

i. The trajectory data is accessed multiple times per trial — once for tongue features, once for paw features, and once for all-landmark motion energy. Each call to `camera_trial` re-reads frame times and tracking data from the HDF5 file. The `landmark_speed` function is called 3 times per camera per trial (tongue, paw, all landmarks).

ii.
```python
for cami,tr in enumerate(traj):
    ts,names,ft=camera_trial(f,tr,ti)
    # Called 3 times with different landmark subsets:
    s,_=landmark_speed(ts,tong)
    s,_=landmark_speed(ts,paws)
    s,_=landmark_speed(ts,list(range(len(names))))
```

iii. The trajectory data could be read once per trial and reused for all feature computations.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes the aggregate landmark-motion proxy for motion energy by combining ALL landmark speeds including tongue and paw, then also separately computes tongue and paw velocities. The all-landmark speed computation duplicates work already done for tongue and paw individually. Additionally, the `components` list mixes tongue, paw, and all-landmark speeds, which means the motion energy proxy includes redundant information from the same landmarks.

ii.
```python
# These are all added to components and averaged:
s,_=landmark_speed(ts,tong)     # tongue landmarks
s,_=landmark_speed(ts,paws)     # paw landmarks
s,_=landmark_speed(ts,list(range(len(names))))  # ALL landmarks (includes tongue+paw again)
me=np.nanmean(np.stack(components),axis=0)
```

iii. The motion energy proxy double-counts tongue and paw landmarks by including them both separately and in the all-landmark aggregate.
