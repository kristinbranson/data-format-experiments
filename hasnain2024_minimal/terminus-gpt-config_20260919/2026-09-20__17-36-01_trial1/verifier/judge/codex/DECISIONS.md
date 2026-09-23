# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-coded a 12-session subset from `/app/data/Ephys_Behavior` and opened each `data_structure_<animal>_<date>.mat` file directly with `h5py`. It did not search both ephys folders, did not load MATLAB v5 files, and did not load separate `motionEnergy_<...>.mat` files. Instead, all behavioral, neural, and video-derived outputs came from the single HDF5 `data_structure` file for each chosen session.

ii.
```python
ROOT='/app/data/Ephys_Behavior'
SESSIONS=[
 ('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),
 ('JGR2','2021-11-16',1),('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),
 ('JEB19','2023-04-21',1),('JEB19','2023-04-20',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-18',1)]

def convert_session(animal,date,probe):
    path=f'{ROOT}/data_structure_{animal}_{date}.mat'
    with h5py.File(path,'r') as f:
```

iii. In the trajectory, the AI said Figure 8 defined a curated 12-session two-context subset and that those sessions should be used instead of all available files. It also said the release lacked raw videos or separate motion-energy assets, so it decided to derive a proxy from the trajectory data instead of loading standalone motion-energy files.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the first element of each hard-coded session tuple. The script builds `subjects` in first-appearance order and creates `subject_idx` by looking up each session's animal in that list.

ii.
```python
subjects=[]; subject_idx=[]
for animal,_,_ in SESSIONS:
    if animal not in subjects:subjects.append(animal)
    subject_idx.append(subjects.index(animal))
```

iii. In the trajectory, the AI repeatedly described the dataset as the 12 curated sessions from the Figure 8 code and treated the animal name in those tuples as the subject identifier.

## 1-c. How are the data split into sessions?

i. One hard-coded `(animal, date, probe)` tuple is one session. Each session becomes one element of `neural`, `input`, and `output`, and only one probe per session is used.

ii.
```python
SESSIONS=[
 ('JEB6','2021-04-18',2), ... ('JEB19','2023-04-18',1)]

def main():
    neural=[]; inp=[]; out=[]; infos=[]
    for s in SESSIONS:
        a,b,c,i=convert_session(*s); neural.append(a); inp.append(b); out.append(c); infos.append(i)
```

iii. In the trajectory, the AI said it recovered the "exact expert two-context subset" from the Figure 8 loader code and that each selected session used one specified probe rather than combining all recordings.

## 1-d. How are the data split into trials?

i. Trials are the indices in the behavioral arrays that satisfy the AI's `keep` mask. The script uses those trial indices everywhere else: behavioral outputs are read from those indices, neural spikes are selected by matching `trial==ti`, and video traces are computed for each kept trial index.

ii.
```python
bp=f['obj/bp']; n=int(np.asarray(bp['Ntrials']).squeeze())
label={k:np.asarray(bp[k]).ravel().astype(bool) for k in ['R','L','hit','miss','no','early','autowater']}
go=np.asarray(bp['ev/goCue']).ravel().astype(float)
keep=(label['hit']|label['miss']|label['no']) & np.isfinite(go)
tids=np.flatnonzero(keep)
```

```python
for j,ti in enumerate(tids):
    inputs.append(time_row.copy())
    ...
```

iii. In the trajectory, the AI said the paper code used the boolean trial fields `R`, `L`, `hit`, `miss`, `no`, `early`, and `autowater`, and it concluded that a trial could be indexed directly from those per-trial arrays without reconstructing boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered only by requiring one of `hit`, `miss`, or `no` to be true and requiring a finite go-cue time. Early-lick trials are explicitly kept, photostimulation is not checked, and there is no post hoc removal of trials after the recording stops.

ii.
```python
# Figure 8 condition 1: hit|miss|no. Keep early trials because that exact
# all-trial condition does not exclude them and outcome decoding needs no.
keep=(label['hit']|label['miss']|label['no']) & np.isfinite(go)
tids=np.flatnonzero(keep)
```

iii. In the trajectory, the AI said the Figure 8 condition included hit, miss, and no-response trials, and it reasoned that ignore trials had to be retained because the decoder required an ignore class. It also explicitly decided to keep early trials because it believed the exact Figure 8 all-trial condition did not exclude them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data come from the selected probe inside `obj/clu`. The script uses each cluster's `trial` and `trialtm` to place spikes into trial-aligned bins, uses `tm` only to count the available units, and uses `bp.ev.goCue` to align spikes to go cue.

ii.
```python
clu=f[refs(f['obj/clu'])[probe-1]]
trial_cells=refs(clu['trial']); tm_cells=refs(clu['trialtm'])
alltm=refs(clu['tm'])
...
go=np.asarray(bp['ev/goCue']).ravel().astype(float)
```

iii. In the trajectory, the AI said the cluster fields were `quality`, `site`, `tm`, `trial`, and `trialtm`, and that `trialtm` was already relative to trial start, so go-cue alignment should be a subtraction against `bp.ev.goCue`.

## 2-b. How is the `neural` data processed?

i. For each retained unit and each kept trial, the AI bins go-cue-aligned spike times into 10 ms bins over `[-3.0, 2.5)` seconds, converts counts to rates by dividing by `DT`, and applies a causal 15-bin Gaussian kernel with the first half zeroed out.

ii.
```python
DT=.01; TMIN=-3.; TMAX=2.5
TIME=np.arange(TMIN,TMAX,DT); EDGES=np.arange(TMIN,TMAX+DT/2,DT)

def causal_smooth(x, n=15):
    k=gaussian(n, std=(n-1)/2/2.5, sym=True)
    k[:n//2]=0; k/=k.sum()
    return convolve1d(x, k, axis=-1, mode='constant', origin=0)
```

```python
sp=rel[tr==ti]-go[ti]
counts=np.histogram(sp,bins=EDGES)[0].astype(np.float32)/DT
neural_trials[oj][ni]=causal_smooth(counts)
```

iii. In the trajectory, the AI said it was following the Figure 8 defaults: go-cue alignment, a `[-3, 2.5)` window, 10 ms bins, and the causal 15-bin Gaussian smoothing used by `getPSTHs.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not use cluster quality labels at all. It keeps all units from the selected probe whose average firing rate exceeds 1 Hz, where rate is computed as the number of spikes from that unit falling in the aligned analysis window across all source trials divided by `n * (TMAX - TMIN)`.

ii.
```python
good=[]
for ui in range(len(alltm)):
    tr_all=arr(f,trial_cells[ui]).astype(int)-1
    rel_all=arr(f,tm_cells[ui]).astype(float)
    aligned=rel_all-go[tr_all]
    nwin=np.count_nonzero((aligned>=TMIN)&(aligned<TMAX))
    rate=nwin/(n*(TMAX-TMIN))
    if rate>1.: good.append(ui)
```

iii. In the trajectory, the AI initially used wall-clock recording duration, then corrected itself after reading `getFiringRate.m`. It justified the final filter by saying Figure 8 used `quality={'all'}` and a >1 Hz mean rate over the aligned trial window, so all qualities were eligible before this firing-rate cutoff.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial spikes are aligned by subtracting the trial's go-cue time from `trialtm`, then histogramming the result on the common bin grid.

ii.
```python
tr=arr(f,trial_cells[ui]).astype(int)-1; rel=arr(f,tm_cells[ui]).astype(float)
for ti,oj in tid_to_out.items():
    sp=rel[tr==ti]-go[ti]
    counts=np.histogram(sp,bins=EDGES)[0].astype(np.float32)/DT
```

iii. In the trajectory, the AI explicitly said `trialtm` was relative to trial start and that go-cue alignment therefore required subtracting `bp.ev.goCue`, "exactly as paper preprocessing does."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins and a `[-3.0, 2.5)` second window, yielding 550 time bins per trial. There is no later rebinning of the neural signal after histogramming.

ii.
```python
DT=.01; TMIN=-3.; TMAX=2.5
TIME=np.arange(TMIN,TMAX,DT); EDGES=np.arange(TMIN,TMAX+DT/2,DT)
```

iii. In the trajectory, the AI said the Figure 8 script used 10 ms bins over `-3` to `2.5` seconds and that it wanted all streams on that grid.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a dedicated raw variable. It is the synthetic time grid defined by the script's `TIME`, which is intended to represent seconds from go-cue onset for every trial.

ii.
```python
DT=.01; TMIN=-3.; TMAX=2.5
TIME=np.arange(TMIN,TMAX,DT)
```

iii. In the trajectory, the AI said the common time axis should follow the Figure 8 aligned window, so it constructed the decoder input directly from that analysis grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script builds a 1D time vector with `np.arange`, casts it to `float32`, adds a singleton feature dimension, and copies that same row into every kept trial.

ii.
```python
time_row=TIME.astype(np.float32)[None,:]
for j,ti in enumerate(tids):
    inputs.append(time_row.copy())
```

iii. In the trajectory, the AI justified this as the simplest way to put the requested decoder input on the same grid as the neural and video outputs.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is exactly the same analysis grid used to bin the neural data, so each input time point corresponds to the same bin index used for spike histogramming.

ii.
```python
TIME=np.arange(TMIN,TMAX,DT); EDGES=np.arange(TMIN,TMAX+DT/2,DT)
...
counts=np.histogram(sp,bins=EDGES)[0].astype(np.float32)/DT
...
time_row=TIME.astype(np.float32)[None,:]
```

iii. In the trajectory, the AI said all streams should share one go-cue-aligned grid, with the input being that grid itself.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the per-trial behavioral flags `L`, `R`, `hit`, `miss`, and `no`.

ii.
```python
label={k:np.asarray(bp[k]).ravel().astype(bool) for k in ['R','L','hit','miss','no','early','autowater']}
```

```python
if label['no'][ti]: lick=2
elif label['hit'][ti]: lick=0 if label['L'][ti] else 1
else: lick=1 if label['L'][ti] else 0
```

iii. In the trajectory, the AI said the paper code exposed boolean trial labels `R`, `L`, `hit`, `miss`, and `no`, and it described lick direction as being recoverable from instructed side plus outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The script encodes no-response trials as class 2 (`none`), hit trials as the instructed lick side, and miss trials as the opposite side. It then repeats that scalar class across all time bins for the trial.

ii.
```python
if label['no'][ti]: lick=2
elif label['hit'][ti]: lick=0 if label['L'][ti] else 1
else: lick=1 if label['L'][ti] else 0
...
const=np.array([lick,context,outcome],dtype=np.int16)[:,None]
const=np.repeat(const,TIME.size,axis=1)
```

iii. In the trajectory, the AI said "correct follows instructed side; incorrect is opposite; ignored/no-response has no lick" and chose the class order left, right, none.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived directly from the `autowater` trial flag in `obj/bp`.

ii.
```python
label={k:np.asarray(bp[k]).ravel().astype(bool) for k in ['R','L','hit','miss','no','early','autowater']}
...
context=1 if label['autowater'][ti] else 0
```

iii. In the trajectory, the AI said the context code revealed that `obj.bp.autowater` was the context label distinguishing delayed-response versus autowater trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI relabels `autowater=True` as class 1 and all other trials as class 0, and its metadata names these classes `['DR', 'WC']`. In effect, the code uses DR=0 and WC=1 and repeats that label across all time bins in the trial.

ii.
```python
context=1 if label['autowater'][ti] else 0 # DR=0, WC=1
...
'output_values':[['left','right','none'],['DR','WC'],['incorrect','correct','ignore'],
```

iii. In the trajectory, the AI said `autowater` denoted WC context and everything else DR, but it ultimately encoded the classes in the reverse numeric order from the prompt/reference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the trial flags `hit`, `no`, and implicitly `miss` via the `else` branch.

ii.
```python
label={k:np.asarray(bp[k]).ravel().astype(bool) for k in ['R','L','hit','miss','no','early','autowater']}
...
outcome=2 if label['no'][ti] else (1 if label['hit'][ti] else 0)
```

iii. In the trajectory, the AI said outcome should be formed from hit, miss, and no-response trial labels because the decoder required an ignore class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script maps no-response trials to 2 (`ignore`), hit trials to 1 (`correct`), and all remaining kept trials to 0 (`incorrect`). It then repeats that scalar across all time bins.

ii.
```python
outcome=2 if label['no'][ti] else (1 if label['hit'][ti] else 0)
...
const=np.array([lick,context,outcome],dtype=np.int16)[:,None]
const=np.repeat(const,TIME.size,axis=1)
```

iii. In the trajectory, the AI explicitly said the decoder outputs should include incorrect, correct, and ignore, and that ignore/no-response trials had to stay in the dataset.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the trajectory data in `obj/traj`: `ts`, `frameTimes`, and `featNames`, together with per-trial `goCue`. The script uses any feature whose name contains `"tongue"` from either camera.

ii.
```python
def camera_trial(f,traj,ti):
    ts=arr(f,refs(traj['ts'])[ti]).astype(float)
    ft=arr(f,refs(traj['frameTimes'])[ti]).astype(float)
    ...
    names_obj=f[refs(traj['featNames'])[ti]]
    names=[matlab_text(f[r]) for r in refs(names_obj)]
```

```python
tong=[j for j,n in enumerate(names) if 'tongue' in n]
```

iii. In the trajectory, the AI said the side camera had tongue landmarks, the top camera also had tongue landmarks, and both should contribute because true raw-video motion energy was unavailable.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each camera and trial, the script keeps only coordinates with DLC likelihood `>=0.9`, averages all selected tongue landmarks into a centroid, computes frame-to-frame speed as `hypot(diff(x), diff(y)) * 400`, shifts frame times by `0.5` s and the trial go cue, linearly interpolates onto the decoder grid, and uses the first available camera (or the second if the first is all NaN) as the tongue-velocity trace. No Gaussian smoothing, per-run differentiation, or camera-scale normalization is applied.

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

```python
old=ft-0.5-go[ti]
...
s,_=landmark_speed(ts,tong)
if s is not None:
    q=interp_trace(old,s); components.append(q)
    if cami==0 or np.all(~np.isfinite(tv)): tv=q
```

iii. In the trajectory, the AI said it would derive velocity from DLC x/y/likelihood and frame times, align it the way it believed the paper did, and use session-median discretization. It also justified simplifying to landmark-based motion because it believed raw videos or standalone motion-energy streams were unavailable.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After continuous tongue traces are built, the AI computes the 50th percentile over all finite session values, sets class 0 below threshold, class 1 at or above threshold, and class 2 wherever the value is NaN.

ii.
```python
def discretize_session(stream):
    finite=np.isfinite(stream)
    threshold=float(np.nanpercentile(stream,50)) if finite.any() else np.nan
    out=np.full(stream.shape,2,dtype=np.int16)
    out[finite & (stream<threshold)]=0; out[finite & (stream>=threshold)]=1
    return out,threshold
```

iii. In the trajectory, the AI said the prompt required session-median thresholding and a special class for bins where the feature was not visible.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The script aligns each camera frame by subtracting a fixed `0.5` s offset and the trial's go-cue time from `frameTimes`, then interpolates the resulting continuous trace onto the same `TIME` vector used for neural bins.

ii.
```python
old=ft-0.5-go[ti] # exact offset used by paper loadMotionEnergy/getPosition
...
q=interp_trace(old,s)
```

```python
def interp_trace(old_t, val):
    good=np.isfinite(old_t)&np.isfinite(val)
    out=np.full(TIME.size,np.nan,dtype=np.float32)
    if good.sum()>=2:
        out[:]=np.interp(TIME,old_t[good],val[good]).astype(np.float32)
```

iii. In the trajectory, the AI said motion-energy code used a 0.5 s video offset and decided to apply that same constant offset to the DLC streams before interpolation to the common time grid.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj/traj` `ts`, `frameTimes`, and `featNames`, together with `goCue`. The code uses any feature whose name contains `"paw"`.

ii.
```python
paws=[j for j,n in enumerate(names) if 'paw' in n]
...
s,_=landmark_speed(ts,paws)
if s is not None:
    pv=interp_trace(old,s); components.append(pv)
```

iii. In the trajectory, the AI said the decoder requested paw velocity even though Figure 8 emphasized other features, so it chose to derive paw movement from the available DLC landmarks.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw velocity uses the same `landmark_speed` and `interp_trace` functions as tongue velocity: likelihood thresholding at `0.9`, averaging available paw landmarks to a centroid, frame-to-frame Euclidean speed times 400, frame-time shifting by `0.5` s and go cue, and interpolation onto the `TIME` grid. The final continuous paw signal is whichever paw trace was last assigned during the camera loop.

ii.
```python
def landmark_speed(ts, indices):
    ...
    speed=np.r_[np.nan, np.hypot(np.diff(cx),np.diff(cy))*400.]
```

```python
s,_=landmark_speed(ts,paws)
if s is not None:
    pv=interp_trace(old,s); components.append(pv)
```

iii. In the trajectory, the AI said it would "implement velocity from DLC x/y derivatives with visibility from likelihood" for paw as well as tongue.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The paw stream is discretized the same way as tongue velocity: session median over finite values, 0 below median, 1 at or above median, 2 for NaNs.

ii.
```python
for x in rawvid:
    d,th=discretize_session(x); disc.append(d); thresholds.append(th)
```

iii. In the trajectory, the AI said each valid continuous movement signal would be thresholded at the session median and assigned class 2 when not visible.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity uses the same alignment as tongue velocity: `frameTimes - 0.5 - goCue` followed by interpolation onto the neural `TIME` grid.

ii.
```python
old=ft-0.5-go[ti]
...
pv=interp_trace(old,s)
```

iii. In the trajectory, the AI treated all DLC-derived kinematic streams as sharing the same constant-offset alignment to the common go-cue-centered grid.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is not loaded from a motion-energy file. Instead, the AI derives a proxy from the same trajectory data used for kinematics by computing landmark motion over all visible landmarks from both cameras.

ii.
```python
# Aggregate all confidently visible landmarks as image-motion proxy.
s,_=landmark_speed(ts,list(range(len(names))))
if s is not None: components.append(interp_trace(old,s))
```

iii. In the trajectory, the AI said the release lacked raw videos or precomputed pixel motion-energy files, so "a justified substitute is aggregate visible-landmark motion from both cameras."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial, the script builds a list of interpolated component traces that includes tongue motion, paw motion, and all-landmark motion from each available camera, then averages those component traces with `np.nanmean` to create a continuous motion-energy proxy. That proxy is later discretized at the session median.

ii.
```python
tv=np.full(TIME.size,np.nan,np.float32); pv=tv.copy(); components=[]
...
if s is not None:
    q=interp_trace(old,s); components.append(q)
...
if s is not None:
    pv=interp_trace(old,s); components.append(pv)
...
s,_=landmark_speed(ts,list(range(len(names))))
if s is not None: components.append(interp_trace(old,s))
if components:
    with np.errstate(invalid='ignore'): me=np.nanmean(np.stack(components),axis=0).astype(np.float32)
```

iii. In the trajectory, the AI justified this as an approximation to motion energy because it believed no true raw-video motion-energy stream was available in the released data.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The motion-energy proxy is discretized with the same session-level median thresholding used for tongue and paw: 0 below median, 1 at or above median, 2 when the proxy is NaN.

ii.
```python
def discretize_session(stream):
    finite=np.isfinite(stream)
    threshold=float(np.nanpercentile(stream,50)) if finite.any() else np.nan
    out=np.full(stream.shape,2,dtype=np.int16)
    out[finite & (stream<threshold)]=0; out[finite & (stream>=threshold)]=1
```

iii. In the trajectory, the AI said all three movement-like outputs would be discretized with per-session median splits and a special missing-data class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy proxy components are aligned with the same `frameTimes - 0.5 - goCue` transformation used for tongue and paw, then interpolated onto the neural `TIME` grid before averaging.

ii.
```python
old=ft-0.5-go[ti]
...
if s is not None: components.append(interp_trace(old,s))
```

iii. In the trajectory, the AI said it would place all video-derived streams on the same common time axis as the neural data by using the same constant-offset, go-cue-centered alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI keeps trials even when some camera information is missing. If `camera_trial` fails, if array shapes are unexpected, or if there are fewer than two valid samples for interpolation, the corresponding video-derived trace becomes all `NaN`; after discretization those bins become class 2. Missing landmark visibility also becomes `NaN` because the code masks points with likelihood `<0.9`.

ii.
```python
def camera_trial(f,traj,ti):
    try:
        ts=arr(f,refs(traj['ts'])[ti]).astype(float)
        ft=arr(f,refs(traj['frameTimes'])[ti]).astype(float)
    except Exception:return None,None,None
    if ts.ndim!=3:return None,None,None
    if ts.shape[1]!=3:return None,None,None
```

```python
def interp_trace(old_t, val):
    good=np.isfinite(old_t)&np.isfinite(val)
    out=np.full(TIME.size,np.nan,dtype=np.float32)
    if good.sum()>=2:
        out[:]=np.interp(TIME,old_t[good],val[good]).astype(np.float32)
        out[TIME < old_t[good].min()]=np.nan; out[TIME > old_t[good].max()]=np.nan
    return out
```

iii. In the trajectory, the AI described class 2 as the way to represent not-visible or no-video bins and said it would preserve missing values as `NaN` until discretization, rather than dropping whole trials.

## 11-a. What are the most time-consuming steps of the code?

i. The likely runtime bottlenecks are the nested loops over units and trials for spike histogramming/smoothing and the per-trial/per-camera video interpolation work in `session_video`, not file discovery. The code performs a separate spike selection and histogram for every retained unit on every kept trial and interpolates video streams trial by trial.

ii.
```python
for ui in range(len(alltm)):
    ...
for ni,ui in enumerate(good):
    tr=arr(f,trial_cells[ui]).astype(int)-1; rel=arr(f,tm_cells[ui]).astype(float)
    for ti,oj in tid_to_out.items():
        sp=rel[tr==ti]-go[ti]
        counts=np.histogram(sp,bins=EDGES)[0].astype(np.float32)/DT
        neural_trials[oj][ni]=causal_smooth(counts)
```

```python
for ti in tids:
    ...
    for cami,tr in enumerate(traj):
        ...
        q=interp_trace(old,s)
```

iii. In the trajectory, the AI noted long conversion times during these loops and repeatedly polled session-by-session progress while the converter ran. It also described the video-processing stage as part of the main conversion workload.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidates are the inner neural loops over trials within each unit, the session-level rate-filter loop over units, and the repeated interpolation/discretization passes over `tongue`, `paw`, and motion components. Those parts repeatedly slice one unit or one trial at a time rather than batching across trials or units.

ii.
```python
for ui in range(len(alltm)):
    tr_all=arr(f,trial_cells[ui]).astype(int)-1
    rel_all=arr(f,tm_cells[ui]).astype(float)
    aligned=rel_all-go[tr_all]
    nwin=np.count_nonzero((aligned>=TMIN)&(aligned<TMAX))
```

```python
for ni,ui in enumerate(good):
    ...
    for ti,oj in tid_to_out.items():
        sp=rel[tr==ti]-go[ti]
        counts=np.histogram(sp,bins=EDGES)[0].astype(np.float32)/DT
```

iii. In the trajectory, the AI focused on reproducing the paper's logic and validating output structure rather than optimizing these loops. It did not claim any vectorization strategy.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several per-trial operations: for each unit it reselects spikes trial by trial inside the nested loop; for each camera stream it calls `interp_trace` separately for tongue, paw, and all-landmark motion; and for every trial it reconstructs the repeated constant output rows for lick/context/outcome.

ii.
```python
for ni,ui in enumerate(good):
    tr=arr(f,trial_cells[ui]).astype(int)-1; rel=arr(f,tm_cells[ui]).astype(float)
    for ti,oj in tid_to_out.items():
        sp=rel[tr==ti]-go[ti]
```

```python
s,_=landmark_speed(ts,tong)
if s is not None:
    q=interp_trace(old,s); components.append(q)
...
s,_=landmark_speed(ts,paws)
if s is not None:
    pv=interp_trace(old,s); components.append(pv)
...
s,_=landmark_speed(ts,list(range(len(names))))
if s is not None: components.append(interp_trace(old,s))
```

iii. In the trajectory, the AI justified these repetitions as a direct implementation of its chosen per-stream processing rather than as an efficiency-oriented design.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores intermediate continuous video traces only to discard them after discretization; it also builds `visible` in `landmark_speed` but does not use it downstream except to set NaNs, and it reads the `early` label even though that flag is not used in the final keep mask. The motion-energy proxy also reuses tongue/paw-derived components even though only the final categorical output is kept.

ii.
```python
label={k:np.asarray(bp[k]).ravel().astype(bool) for k in ['R','L','hit','miss','no','early','autowater']}
```

```python
visible=np.any(vis,axis=0)
speed=np.r_[np.nan, np.hypot(np.diff(cx),np.diff(cy))*400.]
speed[~visible]=np.nan
return speed,visible
```

```python
rawvid=session_video(f,tids,go)
disc=[]; thresholds=[]
for x in rawvid:
    d,th=discretize_session(x); disc.append(d); thresholds.append(th)
```

iii. In the trajectory, the AI described these continuous traces and thresholds as intermediate steps needed to create categorical decoder outputs, but the continuous versions are not preserved in the final dataset.
