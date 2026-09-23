# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 12 curated fixed-delay sessions and one probe per session, all under `Ephys_Behavior`, and reads only HDF5 MATLAB files with `h5py`. It does not load the randomized-delay folder or all 44 reference sessions.

ii.
```python
ROOT='/app/data/Ephys_Behavior'
SESSIONS=[('JEB6','2021-04-18',2), ..., ('JEB19','2023-04-18',1)]
for s in SESSIONS:
    a,b,c,i=convert_session(*s)
```

iii. The trajectory says these are the 12 active sessions and probe selections in `Figure8a_thru_c.m`, which the agent treated as the curated two-context dataset.

## 1-b. How are the data split into subjects?

i. The animal string in each hard-coded tuple defines the subject. Unique subjects are accumulated in first-seen order and each session receives its index.

ii.
```python
for animal,_,_ in SESSIONS:
    if animal not in subjects: subjects.append(animal)
    subject_idx.append(subjects.index(animal))
```

iii. No explicit trajectory justification was given beyond adopting the session list, whose filenames encode animal IDs.

## 1-c. How are the data split into sessions?

i. Every `(animal, date, probe)` tuple is one session and maps to one `data_structure_<animal>_<date>.mat` file and one outer-list entry.

ii.
```python
path=f'{ROOT}/data_structure_{animal}_{date}.mat'
neural.append(a); inp.append(b); out.append(c)
```

iii. The agent followed the Figure 8 loading list and described these as the paper's curated sessions.

## 1-d. How are the data split into trials?

i. Trials are indexed from the Bpod arrays. Kept zero-based indices (`tids`) select behavioral labels, spike `trial` values, and video cell entries.

ii.
```python
n=int(np.asarray(bp['Ntrials']).squeeze())
tids=np.flatnonzero(keep)
tr=arr(f,trial_cells[ui]).astype(int)-1
```

iii. The agent relied on the raw per-trial Bpod and cluster organization; no additional boundary reconstruction was considered necessary.

## 1-e. How are trials filtered based on quality controls?

i. It keeps trials marked hit, miss, or no-response and having a finite go cue. It deliberately retains early-lick trials and does not read or exclude photostimulation trials or trials after ephys ends.

ii.
```python
keep=(label['hit']|label['miss']|label['no']) & np.isfinite(go)
```

iii. Comments and trajectory reasoning say Figure 8 “condition 1” includes hit/miss/no, and ignore trials were needed for outcome decoding; the agent explicitly argued that this condition did not exclude early trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected probe's cluster `trial` and `trialtm` spike arrays and Bpod `ev/goCue`. Cluster `tm` is referenced only to count units.

ii.
```python
trial_cells=refs(clu['trial']); tm_cells=refs(clu['trialtm'])
tr=arr(f,trial_cells[ui]).astype(int)-1
rel=arr(f,tm_cells[ui]).astype(float)
sp=rel[tr==ti]-go[ti]
```

iii. The trajectory identified the paper pipeline as operating on trial-aligned firing rates around the go cue.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into 10 ms bins, divided by 0.01 to obtain Hz, and convolved with a one-sided 15-bin Gaussian kernel. No normalization or baseline correction is applied.

ii.
```python
counts=np.histogram(sp,bins=EDGES)[0].astype(np.float32)/DT
neural_trials[oj][ni]=causal_smooth(counts)
```

iii. The agent interpreted `getPSTHs.m` as using MATLAB `gausswin(15)` with its first half zeroed, hence causal smoothing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. All cluster quality labels are accepted. A unit is retained only if spikes within the full source-trial window imply a mean rate greater than 1 Hz. Only the single hard-coded probe is used.

ii.
```python
nwin=np.count_nonzero((aligned>=TMIN)&(aligned<TMAX))
rate=nwin/(n*(TMAX-TMIN))
if rate>1.: good.append(ui)
```

iii. After initially using wall-clock duration, the agent inspected `getFiringRate.m`, corrected the denominator to analyzed trial time, and reasoned that `quality={'all'}` made every quality eligible.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time is shifted by that trial's go-cue time before binning.

ii.
```python
sp=rel[tr==ti]-go[ti]
```

iii. The agent stated that all streams should be aligned to go-cue onset, consistent with the task and paper code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses 10 ms bins from -3.0 through 2.5 seconds (550 samples). Raw spikes are rebinned directly onto this grid.

ii.
```python
DT=.01; TMIN=-3.; TMAX=2.5
TIME=np.arange(TMIN,TMAX,DT)
EDGES=np.arange(TMIN,TMAX+DT/2,DT)
```

iii. The agent attributed this grid to the Figure 8 code and recorded 10 ms in metadata.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a synthetic fixed `TIME` vector based on the chosen window and bin size, conceptually relative to raw `bp.ev.goCue` but not recomputed from each trial's timestamps.

ii.
```python
TIME=np.arange(TMIN,TMAX,DT)
time_row=TIME.astype(np.float32)[None,:]
```

iii. The agent chose the same go-cue-centered grid used for neural binning.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. `np.arange` produces seconds from -3.0 in 0.01-second increments; the row is copied for every trial.

ii.
```python
inputs.append(time_row.copy())
```

iii. No separate justification was given; it directly represents elapsed time from the alignment event.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It has one value for each neural histogram bin and uses the same `TMIN`, `TMAX`, and `DT`; values denote left bin edges.

ii.
```python
counts=np.histogram(sp,bins=EDGES)[0]
TIME=np.arange(TMIN,TMAX,DT)
```

iii. The trajectory says all streams share go-cue alignment and the same 550-bin timing.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses Bpod `L`, `hit`, and `no` (with the remaining kept trials treated as misses).

ii.
```python
label={k:np.asarray(bp[k]).ravel().astype(bool)
       for k in ['R','L','hit','miss','no','early','autowater']}
```

iii. The agent reasoned that hits follow the instructed side, misses lick the opposite side, and no-response trials have no lick.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Codes are left=0, right=1, none=2. A hit uses the instructed `L` side, a miss reverses it, and `no` becomes none. The per-trial code is repeated across time.

ii.
```python
if label['no'][ti]: lick=2
elif label['hit'][ti]: lick=0 if label['L'][ti] else 1
else: lick=1 if label['L'][ti] else 0
```

iii. This mapping is explained in the inline comment as “actual lick direction.”

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived solely from Bpod `autowater`.

ii.
```python
context=1 if label['autowater'][ti] else 0
```

iii. The agent treated autowater as WC and all other trials as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It maps non-autowater to DR=0 and autowater to WC=1, then repeats the value through time.

ii.
```python
context=1 if label['autowater'][ti] else 0 # DR=0, WC=1
const=np.repeat(const,TIME.size,axis=1)
```

iii. No additional trajectory justification was supplied.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses Bpod `no` and `hit`; remaining selected trials are misses.

ii.
```python
outcome=2 if label['no'][ti] else (1 if label['hit'][ti] else 0)
```

iii. The agent retained no-response trials specifically because the requested output includes an ignore class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. It maps miss/other to incorrect=0, hit to correct=1, and `no` to ignore=2, repeating the category through time.

ii.
```python
const=np.array([lick,context,outcome],dtype=np.int16)[:,None]
const=np.repeat(const,TIME.size,axis=1)
```

iii. This directly follows the requested three outcome labels.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses `obj.traj` for both cameras: per-trial `ts` x/y/likelihood, `featNames`, and `frameTimes`, selecting every landmark name containing `tongue`; Bpod go cues provide alignment.

ii.
```python
tong=[j for j,n in enumerate(names) if 'tongue' in n]
s,_=landmark_speed(ts,tong)
old=ft-0.5-go[ti]
```

iii. The agent said the DLC coordinates and timing should be handled like `getKinematicsFromVideo`, combining available camera-derived components.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Coordinates with likelihood below 0.9 become NaN. Visible tongue landmarks are averaged to a centroid, framewise Euclidean displacement is multiplied by 400, and the trace is linearly interpolated to the 10 ms grid. The first camera is preferred rather than averaging both tongue views.

ii.
```python
vis=lk>=0.9
cx=np.nanmean(xy[:,0,:],axis=0); cy=np.nanmean(xy[:,1,:],axis=0)
speed=np.r_[np.nan, np.hypot(np.diff(cx),np.diff(cy))*400.]
q=interp_trace(old,s)
if cami==0 or np.all(~np.isfinite(tv)): tv=q
```

iii. The trajectory describes likelihood filtering, framewise x/y displacement, and session-median discretization, but gives no evidence for the fixed 400-Hz rate or first-camera preference.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single median is computed over all finite tongue samples in the session. Values below it are 0, values at/above it are 1, and NaNs are 2.

ii.
```python
threshold=float(np.nanpercentile(stream,50))
out[finite & (stream<threshold)]=0
out[finite & (stream>=threshold)]=1
```

iii. The agent followed the prompt's per-session 50th-percentile threshold and not-visible category.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are shifted by a hard-coded 0.5 seconds and the trial go cue, then interpolated onto `TIME`.

ii.
```python
old=ft-0.5-go[ti]
q=interp_trace(old,s)
```

iii. The code calls 0.5 “the exact offset used by paper loadMotionEnergy/getPosition”; the trajectory broadly claimed paper-consistent alignment.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses all `obj.traj` landmark names containing `paw`, their x/y/likelihood arrays, frame times, and go cues, across both cameras.

ii.
```python
paws=[j for j,n in enumerate(names) if 'paw' in n]
s,_=landmark_speed(ts,paws)
```

iii. The agent grouped paw DLC landmarks as the available paw signal; no selection-quality rationale was recorded.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. It applies the same 0.9 likelihood mask, landmark-centroid displacement, fixed 400 multiplier, and interpolation as tongue. If multiple camera paw traces exist, later cameras overwrite earlier ones.

ii.
```python
s,_=landmark_speed(ts,paws)
if s is not None:
    pv=interp_trace(old,s); components.append(pv)
```

iii. The trajectory summarizes this as velocity from framewise x/y displacement, without explaining overwriting or the frame-rate assumption.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It uses the finite-sample session median: below=0, at/above=1, NaN/not visible=2.

ii.
```python
d,th=discretize_session(x)
```

iii. This follows the requested per-session median rule.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw frame times are shifted by 0.5 seconds and the trial go cue, then interpolated to the same 10 ms `TIME` samples.

ii.
```python
old=ft-0.5-go[ti]
pv=interp_trace(old,s)
```

iii. The agent considered the fixed offset paper-exact, though it did not compute the session-specific clock offset used by the reference.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It is not derived from the raw/precomputed motion-energy stream. Instead, it combines interpolated speeds of tongue, paw, and all confidently visible DLC landmarks from both cameras.

ii.
```python
s,_=landmark_speed(ts,list(range(len(names))))
if s is not None: components.append(interp_trace(old,s))
me=np.nanmean(np.stack(components),axis=0)
```

iii. The agent believed the release lacked raw videos and separate motion-energy files, so it explicitly chose aggregate visible-landmark motion as a proxy.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. All collected tongue, paw, and whole-landmark speed traces are averaged with `nanmean`; thus some components are duplicated (tongue/paw appear both separately and in all-landmark motion).

ii.
```python
if components:
    me=np.nanmean(np.stack(components),axis=0).astype(np.float32)
```

iii. The proxy was a fallback intended to represent image motion when the agent thought pixel motion energy was unavailable.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The proxy is split at its session-wide finite median; below=0, at/above=1, and NaN/no proxy data=2.

ii.
```python
out=np.full(stream.shape,2,dtype=np.int16)
out[finite & (stream<threshold)]=0
out[finite & (stream>=threshold)]=1
```

iii. The median and category 2 follow the task specification, although they are applied to the wrong underlying signal.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Each constituent landmark-speed trace is aligned with `frameTimes - 0.5 - goCue` and interpolated to `TIME` before averaging.

ii.
```python
components.append(interp_trace(old,s))
```

iii. The agent intended the proxy to share the exact output grid with neural data.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. `camera_trial` catches any read error or unexpected shape and returns missing values. Interpolation requires two finite samples and otherwise leaves NaNs; categorical discretization maps NaN to class 2. `nanmean` tolerates partially missing landmarks, but can emit warnings. Finite go cues are required.

ii.
```python
except Exception:return None,None,None
if good.sum()>=2: ...
out=np.full(stream.shape,2,dtype=np.int16)
```

iii. The trajectory explicitly considered warnings expected where DLC landmarks were invisible and said NaNs would intentionally become class 2.

## 11-a. What are the most time-consuming steps of the code?

i. The code does not time individual stages. Its likely expensive work is nested HDF5 dereferencing plus unit-by-trial spike histograms and trial/camera video interpolation.

ii.
```python
for ni,ui in enumerate(good):
    for ti,oj in tid_to_out.items():
        counts=np.histogram(sp,bins=EDGES)[0]
```

iii. The trajectory shows conversion requiring repeated polling over many seconds, but provides no formal profiling.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested unit-by-trial neural loop could use a 2-D histogram over trial and aligned time. Some video landmark and session-discretization loops could also be consolidated, though ragged camera data limits vectorization.

ii.
```python
for ni,ui in enumerate(good):
    ...
    for ti,oj in tid_to_out.items():
```

iii. The agent did not discuss vectorization; it prioritized matching processing and successful validation.

## 11-c. What processing does the code repeat multiple times?

i. Each unit's spike arrays are dereferenced once during rate filtering and again during neural construction. Landmark speed is separately computed for tongue, paw, and all landmarks, duplicating work and contributions. The same time row and constant outputs are copied/repeated for every trial.

ii.
```python
tr_all=arr(f,trial_cells[ui]); rel_all=arr(f,tm_cells[ui])
...
tr=arr(f,trial_cells[ui]); rel=arr(f,tm_cells[ui])
```

iii. No trajectory justification addresses these repetitions.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes `visible` in `landmark_speed` but callers discard it; reads `R`, `early`, and `tm` without using their values in final processing; builds proxy component traces that are only averaged; and repeats trial-level categories across all 550 time bins, increasing storage.

ii.
```python
s,_=landmark_speed(ts,tong)
alltm=refs(clu['tm'])
const=np.repeat(const,TIME.size,axis=1)
```

iii. The agent did not identify these as waste; its final reasoning focused on schema validity and dataset dimensions.
