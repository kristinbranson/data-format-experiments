# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI programmatically parses the authors' MATLAB loader scripts (`load<ANM>_ALMVideo.m`) using regex to discover which sessions and probes are active. It then matches those session keys against `.mat` files found by globbing the two ephys directories (`Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`). Each session file is loaded via either `h5py` (HDF5/v7.3) or `scipy.io.loadmat` (v5), auto-detected by `h5py.is_hdf5()`. Motion energy is loaded from companion `motionEnergy_*.mat` files using `scipy.io.loadmat`.

ii. Session discovery:
```python
def parse_sessions():
    rows=[]
    p=CODE/'DataLoadingScripts'/'Recording and video'
    for f in sorted(p.glob('*.m')):
        cur={}; active=False
        for raw in f.read_text(errors='ignore').splitlines():
            line=raw.strip()
            ...
            if re.match(r'meta\(end\+1\)',line):
                if active and {'anm','date'}<=cur.keys(): rows.append(...)
                cur=cur.copy(); active=True
    ...
    out=[]
    for f in sorted(DATA.glob('*Ephys_Behavior/data_structure_*.mat')):
        key=f.stem.removeprefix('data_structure_')
        if key in active and active[key]: out.append((key,f,active[key]))
    return out
```

Loading per format:
```python
holder,B,units,traj=load_h5(f,probes) if is_h5 else load_v5(f,probes)
```

iii. The AI's CONVERSION_NOTES.md documents: "Repository is MATLAB code... Loader scripts contain 16 recording animals and 50 active session-date entries; commented entries are excluded by the authors." The programmatic parsing approach avoids hardcoding but achieves the same 44 sessions.

## 1-b. How are the data split into subjects?

i. The animal ID is extracted from the session key by splitting on `'_'` and taking the first part. `subjects` is the sorted set of unique animal IDs, and `subject_idx` maps each session to its index.

ii.
```python
animals.append(key.split('_')[0])
subjects=sorted(set(animals))
subject_idx=np.asarray([subjects.index(a) for a in animals],dtype=np.int64)
```

iii. The AI notes in CONVERSION_NOTES.md that there are 14 neural subjects: "EKH1, EKH3, JEB11, JEB12, JEB13, JEB14, JEB15, JEB19, JEB23, JEB24, JEB6, JEB7, JGR2, JGR3."

## 1-c. How are the data split into sessions?

i. One session is one `data_structure_<anm>_<date>.mat` file from either ephys directory, filtered to only those sessions active in the author loader scripts. The result is 44 sessions (25 fixed-delay + 19 randomized-delay).

ii.
```python
for f in sorted(DATA.glob('*Ephys_Behavior/data_structure_*.mat')):
    key=f.stem.removeprefix('data_structure_')
    if key in active and active[key]: out.append((key,f,active[key]))
```

iii. The AI documents: "Use the 44 sessions that are both present and active in reference recording loaders. This exactly gives 25 fixed-delay and 19 randomized-delay sessions."

## 1-d. How are the data split into trials?

i. Trials are defined by the Bpod behavior structure. The number of trials is determined by `B['n'] = len(B['go'])` (the length of the go cue array). Each trial index is used to access behavior flags, spike data, and trajectory data.

ii.
```python
B['go']=scalar('obj/bp/ev/goCue').astype(float); B['n']=len(B['go'])
```

Trial-level behavior flags:
```python
B={k:scalar('obj/bp/'+k).ravel().astype(bool) for k in ('R','L','hit','miss','no','early','autowater')}
```

iii. The AI's CONVERSION_NOTES.md states trial counts come from "raw data contain hit, miss, no, early flags" and the go cue array length.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple filters: (1) requires `haveEphys` flag to be true, (2) requires finite positive go cue time, (3) excludes early-lick trials, (4) excludes stimulation trials, and (5) after neural data construction, removes trials where ALL retained neurons have zero firing across all time bins. This last step catches trailing trials past the end of the recording.

ii.
```python
valid=B['haveE']&np.isfinite(B['go'])&(B['go']>0)&(~B['early'])&(~B['stim'])
tids=np.flatnonzero(valid)
neural,rates=neural_arrays(B,units,tids)
neural_valid=np.any(neural!=0,axis=(1,2))
tids=tids[neural_valid]; neural=neural[neural_valid]
```

iii. The AI documents: "Exclude early trials. Retain ignore because requested target explicitly requires an ignore class. Do not balance/drop valid trials for the converted dataset." Also: "Sixty-one trailing trials in two JEB24 sessions had no neural samples despite incorrect native validity flags; these invalid periods were excluded."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The cluster spike data from `obj.clu{probe}`: `trial` (which trial each spike belongs to, 1-based), `trialtm` (spike time relative to trial start), and `quality` (manual curation label). The go cue times from `bp.ev.goCue` are used for alignment.

ii.
```python
# HDF5 loading:
tr=np.asarray(h[g['trial'][i,0]],int).ravel()-1
tm=np.asarray(h[g['trialtm'][i,0]],float).ravel()
units.append((q,tr,tm))
```

```python
ok=(tr>=0)&(tr<n); tr=tr[ok]; al=tm[ok]-B['go'][tr]
```

iii. The AI identified these as the key neural fields in Step 1 and Step 5 of CONVERSION_NOTES.md.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue by subtracting `goCue[trial]` from `trialtm`. Spikes are binned into 5ms bins on [-2.5, 2.5)s and converted to firing rates (spikes/s) by dividing by DT. Rates are then smoothed with a **causal** Gaussian kernel: a `gausswin(15)` with alpha=2.5, where the first half is zeroed, normalized, and convolved with `mode='same'`. This matches the reference MATLAB `mySmooth` function.

ii.
```python
# Per-unit, per-trial histogram:
mat[oi]=np.histogram(vals,EDGES)[0].astype(np.float32)/DT

# Causal Gaussian smoothing (exact mySmooth equivalent):
N=15; alpha=2.5
nn=np.arange(N,dtype=float)-(N-1)/2
kern=np.exp(-0.5*(alpha*nn/((N-1)/2))**2)
kern[:N//2]=0; kern/=kern.sum()
mat=np.stack([np.convolve(row,kern,mode='same') for row in mat]).astype(np.float32)
```

iii. The AI's trajectory shows it initially used an incorrect symmetric Gaussian filter, then corrected to the exact causal kernel after inspecting the MATLAB `mySmooth` function: "Reference `mySmooth` uses a 15-point causal `gausswin`, zeros the first half of the kernel, normalizes it, and convolves with `same`."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) clusters with quality labels `garbage` or `gabrga` are excluded; all other quality labels are retained (this is interpreted as the MATLAB default `quality='all'`). (2) Units with mean firing rate <= 0.5 Hz are excluded, following the reference code's `params.lowFR = 0.5`. This yields 2,498 neurons across 44 sessions.

ii.
```python
BADQ={'garbage','gabrga'}
...
q=hchars(h,g['quality'][i,0]).strip().lower()
if q in BADQ: continue
...
rate=np.count_nonzero((al>=TMIN)&(al<TMAX))/(n*(TMAX-TMIN))
if rate<=0.5: continue
```

iii. The AI documents: "Exclude explicit garbage labels; retain curated multi/fair/poor/good/great/excellent labels as reference 'all units.'" And: "Use >0.5 Hz because it reproduces reported aggregate within 2 units; >1 Hz yields only 2,459."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's `trialtm` (time relative to trial start) has the trial-specific go cue time subtracted, putting all spikes in seconds from go cue onset.

ii.
```python
ok=(tr>=0)&(tr<n); tr=tr[ok]; al=tm[ok]-B['go'][tr]
```

iii. The AI notes this matches `alignSpikes.m`: "raw `trialtm - goCue[trial]`, exactly as `alignSpikes`."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Spikes are binned into 1000 non-overlapping 5ms bins spanning -2.5 to +2.5s from the go cue, matching the reference `params.dt = 1/200`.

ii.
```python
DT=0.005; TMIN=-2.5; TMAX=2.5
EDGES=np.arange(TMIN,TMAX+DT/2,DT,dtype=np.float64)
TIME=((EDGES[:-1]+EDGES[1:])/2).astype(np.float32)
```

iii. The AI documents: "5-ms `histc`-equivalent bins on [-2.5,2.5), rates in spikes/s."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is not derived from raw data variables. It is defined as the bin centers of the time axis, which is determined by the alignment event (go cue), the window (-2.5 to 2.5s), and the bin size (5ms).

ii.
```python
TIME=((EDGES[:-1]+EDGES[1:])/2).astype(np.float32)
trialsI.append(TIME[None,:].copy())
```

iii. The AI documents in Step 5: "Broadcast signed seconds from go cue across every trial."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is defined analytically from the bin edges. No processing beyond computing midpoints of each bin.

ii.
```python
TIME=((EDGES[:-1]+EDGES[1:])/2).astype(np.float32)
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis IS the neural binning grid itself. The bin centers are the same intervals used for spike counting, so alignment is automatic.

ii.
```python
# Neural bins use EDGES:
mat[oi]=np.histogram(vals,EDGES)[0].astype(np.float32)/DT
# Input uses the centers of those same EDGES:
trialsI.append(TIME[None,:].copy())
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Three behavior flags from `obj.bp`: `R` (right-instructed trial), `L` (left-instructed trial), `hit` (correct response), and `miss` (incorrect response).

ii.
```python
B={k:scalar('obj/bp/'+k).ravel().astype(bool) for k in ('R','L','hit','miss','no','early','autowater')}
```

iii. Documented in Step 5 mapping: "right=`R&hit` or `L&miss`; left=`L&hit` or `R&miss`; none=`no`."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Lick direction is inferred from the combination of instructed side and outcome: a hit on a right trial means the animal licked right; a miss on a right trial means it licked left; and vice versa for left trials. No-response trials get `none` (class 2). Codes: left=0, right=1, none=2.

ii.
```python
lick=0 if (B['L'][tid] and B['hit'][tid]) or (B['R'][tid] and B['miss'][tid]) else 1 if (B['R'][tid] and B['hit'][tid]) or (B['L'][tid] and B['miss'][tid]) else 2
```

iii. The AI documents this as the standard mapping in Step 5 and the conversion notes.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The `autowater` flag from `obj.bp`. When true, the trial is in the water-cued (WC) context; when false, the delayed-response (DR) context.

ii.
```python
B['autowater']=np.asarray(getattr(bp,'autowater')).ravel().astype(bool)
```

iii. Documented in Step 5: "`bp.autowater`".

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct conversion of the boolean flag: `int(autowater)` gives 0 for DR, 1 for WC. The output_values are `['DR', 'WC']` to match.

ii.
```python
np.full(len(TIME),int(B['autowater'][tid]),np.int8)
```

Output values definition:
```python
'output_values':[['left','right','none'],['DR','WC'],...]
```

iii. The encoding DR=0, WC=1 is the natural mapping from the boolean flag.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Two behavior flags from `obj.bp`: `hit` (correct response) and `miss` (incorrect response). Trials that are neither are classified as ignore.

ii.
```python
outcome=1 if B['hit'][tid] else 0 if B['miss'][tid] else 2
```

iii. Documented in Step 5: "`bp.hit/miss/no` source flags."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct relabelling: hit -> correct (1), miss -> incorrect (0), everything else -> ignore (2). Codes: incorrect=0, correct=1, ignore=2.

ii.
```python
outcome=1 if B['hit'][tid] else 0 if B['miss'][tid] else 2
```

Output values:
```python
['incorrect','correct','ignore']
```

iii. The AI retains ignore trials because the task explicitly requires an ignore class, despite the paper omitting them from most analyses.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, specifically the `tongue` feature from the side camera (view 0). The tracked x, y coordinates and likelihood/NaN masking are used. Frame times from `traj.frameTimes` and go cue times from `bp.ev.goCue` provide temporal alignment.

ii.
```python
tongue=feature_speed(traj,tids,0,'tongue',B)
```

iii. The AI uses only the side camera's tongue feature. The CONVERSION_NOTES.md Step 5 mentions "use a consistently named tongue feature" but doesn't discuss combining multiple camera views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. (1) Tracked coordinates are checked for finite values (NaN marks invisible frames). (2) `np.gradient` is applied to raw (unsmoothed) x and y coordinates for ALL frames (including NaN). (3) Speed is computed as `hypot(dx, dy)` only for visible frames. (4) The frame-resolution velocities are **linearly interpolated** to the 5ms time grid using `interp_visible`. (5) A visibility mask is interpolated to determine which bins are tracked vs not visible. (6) The continuous velocity is discretized at the session median.

ii.
```python
vis=np.isfinite(xy).all(1)
vel=np.full(m,np.nan); dx=np.gradient(xy[:,0]); dy=np.gradient(xy[:,1])
vel[vis]=np.hypot(dx[vis],dy[vis])
rel=ft-B['go'][tid]
speed=interp_visible(rel,vel,TIME)
# Visibility mask interpolation:
vi=np.interp(TIME,rr,vv,left=0,right=0)>=0.5; speed[~vi]=np.nan
```

iii. The AI does not smooth the coordinates before differentiation (unlike the reference which uses a 5ms Gaussian). The AI uses linear interpolation to the target time grid rather than binning/averaging frames per bin.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The session median (nanmedian) of all finite velocity values across all trials and time bins is computed. Values below the median get class 0, at or above get class 1, and NaN/not-visible bins get class 2.

ii.
```python
def discretize(x,missing_class=True):
    med=float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
    y=np.full(x.shape,2 if missing_class else 0,np.int8); ok=np.isfinite(x)
    if np.isfinite(med): y[ok]=(x[ok]>=med).astype(np.int8)
    return y,med
```

iii. The 50th percentile split matches the task instructions.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times from the trajectory data are subtracted by the go cue time of that trial: `ft - B['go'][tid]`. The resulting time-from-go-cue values are used to interpolate velocities to the 5ms time grid. **No video-to-behavior clock offset is computed.**

ii.
```python
rel=ft-B['go'][tid]
speed=interp_visible(rel,vel,TIME)
```

iii. The AI does not compute the bitcode-based video offset that the reference MATLAB code (`findVideoOffset.m`) uses to correct for the difference between the video clock and the behavior clock.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The `top_paw` feature from the bottom camera (view 1) in `obj.traj`.

ii.
```python
paw=feature_speed(traj,tids,1,'top_paw',B)
```

iii. The AI correctly identifies `top_paw` as the paw feature from the bottom view.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue velocity: raw `np.gradient` on unsmoothed coordinates, speed from `hypot`, visibility masking, and linear interpolation to the 5ms grid.

ii.
```python
paw=feature_speed(traj,tids,1,'top_paw',B)
```
(Uses the same `feature_speed` function as tongue.)

iii. No smoothing before differentiation. Uses interpolation rather than binning.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: session median split. Below median = 0, at/above = 1, not visible = 2.

ii.
```python
pd,pmed=discretize(paw)
```

iii. Matches the task instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: `frameTimes - goCue[trial]`, then interpolation to the 5ms grid. No video offset correction.

ii.
```python
rel=ft-B['go'][tid]
speed=interp_visible(rel,vel,TIME)
```

iii. Same video offset omission as tongue.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Companion `motionEnergy_<anm>_<date>.mat` files, which contain one trace per trial. The nested struct is unwrapped (handling legacy `me.data.data` nesting). Frame times from the side camera (view 0) in `traj` provide the temporal reference.

ii.
```python
me=loadmat(mf,squeeze_me=True,struct_as_record=False)['me']
raw=me.data
while hasattr(raw,'data'):
    raw=raw.data
```

iii. The AI documents: "Reference `loadMotionEnergy` unwraps legacy struct-valued `me.data`."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are aligned to go cue time using the side camera's frame times, then **linearly interpolated** to the 5ms grid, and **nearest-neighbor filled** to close any remaining NaN gaps. The continuous values are then discretized at the session median.

ii.
```python
rel=np.asarray(ft[:m])-B['go'][tid]
z=interp_visible(rel,y[:m],TIME); z=nearest_fill(z).astype(np.float32)
```

```python
def nearest_fill(x):
    x=np.asarray(x,float).copy(); good=np.isfinite(x)
    if not good.any(): return x
    ix=np.arange(x.size); x[~good]=np.interp(ix[~good],ix[good],x[good]); return x
```

iii. The nearest-fill step ensures almost no bins are NaN, meaning almost no bins get the "no video" class. The AI's verification shows motion energy no-video fraction is essentially 0.000.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue and paw: session median split. Below = 0, at/above = 1, no video = 2.

ii.
```python
md,mmed=discretize(motion)
```

iii. Matches the task instructions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side camera frame times minus go cue time, then linear interpolation to the 5ms grid, then nearest-fill. No video offset correction.

ii.
```python
ft=traj[0][tid][0]
rel=np.asarray(ft[:m])-B['go'][tid]
z=interp_visible(rel,y[:m],TIME); z=nearest_fill(z).astype(np.float32)
```

iii. Same video offset omission as the other camera-derived outputs.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Trials without valid ephys (`haveEphys=false`) are excluded. (2) Trials with all-zero neural activity (trailing recording periods) are excluded. (3) Missing trajectory data or features return NaN arrays which become "not visible" class 2. (4) Motion energy gaps are nearest-neighbor filled. (5) Velocity visibility is interpolated to determine which bins are tracked. (6) The `haveVid` flag is checked before processing video features.

ii.
```python
# haveVid check in feature_speed:
if z is None or not B['haveV'][tid]: out.append(np.full(len(TIME),np.nan,np.float32)); continue

# nearest_fill for motion energy:
z=nearest_fill(z).astype(np.float32)

# All-zero neural trial exclusion:
neural_valid=np.any(neural!=0,axis=(1,2))
tids=tids[neural_valid]; neural=neural[neural_valid]
```

iii. The AI documents these edge cases in Steps 9-10 of CONVERSION_NOTES.md.

## 11-a. What are the most time-consuming steps of the code?

i. File I/O dominates. Loading each session's MATLAB file (especially HDF5 files with lazy dereferencing) takes 2-5 seconds per session. The full conversion of 44 sessions takes about 2-3 minutes total.

ii. Per-session timing from conversion output:
```
EKH1_2021-08-07: trials 252/305, neurons 48, 2.80s
JEB14_2022-08-22: trials 474/517, neurons 83, 5.59s
```

iii. The AI documents: "Lazy HDF5 reads, session-wise processing, compact dtypes."

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike histogram loop is per-unit, per-trial, computing one `np.histogram` call per trial per unit. This could be vectorized with a single `np.histogram2d` per unit (binning across trials and time simultaneously). The smoothing convolution also loops over trials with `np.stack([np.convolve(row, kern, mode='same') for row in mat])`. The velocity computation loops per trial with `feature_speed`.

ii.
```python
# Per-trial spike histogram (could be vectorized):
for oi,tid in enumerate(trial_ids):
    vals=al[tr==tid]; mat[oi]=np.histogram(vals,EDGES)[0].astype(np.float32)/DT

# Per-trial smoothing (could be vectorized):
mat=np.stack([np.convolve(row,kern,mode='same') for row in mat]).astype(np.float32)
```

iii. The AI notes: "Vectorized alignment" as a speedup but the actual implementation still loops per trial.

## 11-c. What processing does the code repeat multiple times?

i. The `parse_sessions()` function reads the MATLAB loader scripts once at startup. Within each session, behavior flags are read once. The `feature_speed` function is called three times (tongue, paw from two different views) but each call processes a different feature, so nothing is truly redundant. However, the trajectory data for each view is accessed multiple times per trial (once for each feature from that view).

ii.
```python
tongue=feature_speed(traj,tids,0,'tongue',B)
paw=feature_speed(traj,tids,1,'top_paw',B)
```

iii. No significant repeated computation identified.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `nearest_fill` interpolation for motion energy produces continuous values that are immediately discretized, so the precise interpolated values are discarded. (2) The full trajectory data (all tracked features) is loaded into memory even though only `tongue` and `top_paw` are used. (3) The `haveVid` array is loaded but only used as a filter, and some sessions have all-true values.

ii.
```python
# nearest_fill values are immediately discretized:
z=nearest_fill(z).astype(np.float32)
...
md,mmed=discretize(motion)
```

iii. The file loading is the main source of unnecessary processing, as the entire trajectory object is read regardless of which features are needed.
