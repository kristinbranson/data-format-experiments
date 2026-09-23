# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script parses the authors' MATLAB loader files to recover active session names and probe selections, then globs `data_structure_*.mat` files from both `Ephys_Behavior` directories and keeps only files whose `<animal>_<date>` key appears in the parsed loader output. Each selected session is then opened with either an HDF5 reader (`h5py`) or a v5 MATLAB reader (`scipy.io.loadmat` via `loadmat`), depending on file format.

ii. 
```python
def parse_sessions():
    rows=[]
    p=CODE/'DataLoadingScripts'/'Recording and video'
    for f in sorted(p.glob('*.m')):
        ...
    active={f'{a}_{d}':pr for a,d,pr in rows}
    out=[]
    for f in sorted(DATA.glob('*Ephys_Behavior/data_structure_*.mat')):
        key=f.stem.removeprefix('data_structure_')
        if key in active and active[key]: out.append((key,f,active[key]))
    return out
```

```python
def load_h5(f,probes):
    h=h5py.File(f,'r')
    ...

def load_v5(f,probes):
    o=loadmat(f,squeeze_me=True,struct_as_record=False,variable_names=['obj'])['obj']
    ...
```

iii. In `CONVERSION_NOTES.md`, the AI says it wanted to follow the active author loader entries while also handling the mixed MATLAB formats present in the supplied data.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are derived from the session key by splitting on the underscore and taking the prefix, e.g. `JEB19_2023-04-19 -> JEB19`. At assembly time the script sorts the unique subject ids and builds `subject_idx` by lookup into that sorted list.

ii.
```python
animals.append(key.split('_')[0])
subjects=sorted(set(animals))
subject_idx=np.asarray([subjects.index(a) for a in animals],dtype=np.int64)
```

iii. The notes justify this by relying on filenames as the stable source of animal identity.

## 1-c. How are the data split into sessions?

i. One session is one `data_structure_<animal>_<date>.mat` file whose key survives `parse_sessions()`. Fixed-delay and randomized-delay sessions are both included because the glob spans both `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`.

ii.
```python
for f in sorted(DATA.glob('*Ephys_Behavior/data_structure_*.mat')):
    key=f.stem.removeprefix('data_structure_')
    if key in active and active[key]: out.append((key,f,active[key]))
```

```python
for i,(key,f,probes) in enumerate(sessions):
    n,x,y,info=convert_session(key,f,probes,args.show_processing and i<2)
```

iii. In the notes, the AI says session scope should follow the active loader scripts and reproduce the `25 + 19` session split.

## 1-d. How are the data split into trials?

i. Trials are handled as shared 0-based indices into the per-trial behavior arrays. `B['go']` defines the trial count, `tids` are chosen from a boolean mask over trial-level fields, spike times are assigned by each unit's `trial` array, and video entries are indexed trial-by-trial from `traj`.

ii.
```python
B['go']=scalar('obj/bp/ev/goCue').astype(float); B['n']=len(B['go'])
...
valid=B['haveE']&np.isfinite(B['go'])&(B['go']>0)&(~B['early'])&(~B['stim'])
tids=np.flatnonzero(valid)
```

```python
tr=np.asarray(h[g['trial'][i,0]],int).ravel()-1
...
z=trials[tid] if tid<len(trials) else None
```

iii. No separate justification was recorded beyond the assumption that the raw behavior, spike, and video structures are already trial-indexed.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if `haveEphys` is true, go-cue is finite and positive, and the trial is neither early-lick nor stimulation. After neural construction, the script also drops any included trial whose retained neural matrix is all zeros, which it uses to remove trailing invalid recording periods.

ii.
```python
valid=B['haveE']&np.isfinite(B['go'])&(B['go']>0)&(~B['early'])&(~B['stim'])
tids=np.flatnonzero(valid)
neural,rates=neural_arrays(B,units,tids)
neural_valid=np.any(neural!=0,axis=(1,2))
invalid_zero_trials=tids[~neural_valid].tolist()
tids=tids[neural_valid]; neural=neural[neural_valid]
```

iii. The notes justify this as enforcing valid ephys mapping and fixing 61 trailing JEB24 trials that produced all-zero neural activity.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from each selected cluster's `quality`, `trial`, and `trialtm` fields, together with the per-trial go-cue times `bp.ev.goCue`.

ii.
```python
q=hchars(h,g['quality'][i,0]).strip().lower()
tr=np.asarray(h[g['trial'][i,0]],int).ravel()-1
tm=np.asarray(h[g['trialtm'][i,0]],float).ravel()
```

```python
al=tm[ok]-B['go'][tr]
```

iii. The notes explicitly describe this as the `alignSpikes`-style raw source for the neural stream.

## 2-b. How is the `neural` data processed?

i. For each retained unit, the script aligns spikes by subtracting the go cue, counts spikes into 5 ms bins on `[-2.5, 2.5)`, converts counts to Hz by dividing by `DT`, and then smooths each trial with a 15-point causal Gaussian-window kernel.

ii.
```python
al=tm[ok]-B['go'][tr]
...
mat[oi]=np.histogram(vals,EDGES)[0].astype(np.float32)/DT
...
N=15; alpha=2.5
nn=np.arange(N,dtype=float)-(N-1)/2
kern=np.exp(-0.5*(alpha*nn/((N-1)/2))**2)
kern[:N//2]=0; kern/=kern.sum()
mat=np.stack([np.convolve(row,kern,mode='same') for row in mat]).astype(np.float32)
```

iii. In the notes and code comments, the AI justifies this as an "Exact reference mySmooth" implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps all units except those labeled `garbage` or `gabrga`, then applies a mean firing-rate threshold of `> 0.5` Hz over the aligned window.

ii.
```python
BADQ={'garbage','gabrga'}
...
if q in BADQ: continue
...
rate=np.count_nonzero((al>=TMIN)&(al<TMAX))/(n*(TMAX-TMIN))
if rate<=0.5: continue
```

iii. The notes justify this by following the executable default loader behavior and by matching the aggregate unit count they expected from the supplied data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural spikes are aligned by a direct per-spike subtraction of the go cue from the spike's trial-relative time.

ii.
```python
al=tm[ok]-B['go'][tr]
```

iii. The notes explicitly compare this to the reference `alignSpikes` logic.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data are binned at 5 ms resolution over a `[-2.5, 2.5)` second window, yielding 1000 bins. No additional neural rebinning is applied after this; only smoothing is performed.

ii.
```python
DT=0.005; TMIN=-2.5; TMAX=2.5
EDGES=np.arange(TMIN,TMAX+DT/2,DT,dtype=np.float64)
TIME=((EDGES[:-1]+EDGES[1:])/2).astype(np.float32)
```

iii. The notes justify this as matching the go-cue-aligned 5 ms grid from the reference defaults.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a raw array directly; it is constructed from the globally defined time grid implied by the chosen go-cue-aligned window and bin size.

ii.
```python
DT=0.005; TMIN=-2.5; TMAX=2.5
EDGES=np.arange(TMIN,TMAX+DT/2,DT,dtype=np.float64)
TIME=((EDGES[:-1]+EDGES[1:])/2).astype(np.float32)
```

iii. No separate justification was recorded beyond matching the common reference time axis.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The input is simply the vector of bin centers. For every trial, the script copies `TIME` into a `(1, n_timepoints)` array.

ii.
```python
TIME=((EDGES[:-1]+EDGES[1:])/2).astype(np.float32)
...
trialsI.append(TIME[None,:].copy())
```

iii. No additional justification was recorded.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same `EDGES` array is used for neural spike histograms and the same `TIME` array is stored as the decoder input, so both streams share the same bin grid.

ii.
```python
mat[oi]=np.histogram(vals,EDGES)[0].astype(np.float32)/DT
...
trialsI.append(TIME[None,:].copy())
```

iii. The notes describe the decoder input as the common bin-center time series for the aligned neural grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the trial-level behavior flags `L`, `R`, `hit`, and `miss`. The `no` flag is loaded too, but the mapping itself falls back to a third class when neither hit nor miss applies.

ii.
```python
B={k:scalar('obj/bp/'+k).astype(bool) for k in ('R','L','hit','miss','no','early','autowater')}
...
lick=0 if (B['L'][tid] and B['hit'][tid]) or (B['R'][tid] and B['miss'][tid]) else 1 if (B['R'][tid] and B['hit'][tid]) or (B['L'][tid] and B['miss'][tid]) else 2
```

iii. The notes justify retaining a third no-lick/ignore class because the decoder target explicitly requires it.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The script maps hit/miss and instructed side into categorical labels: left for `L & hit` or `R & miss`, right for `R & hit` or `L & miss`, and none otherwise. The chosen label is then repeated over all time bins for that trial.

ii.
```python
lick=0 if (B['L'][tid] and B['hit'][tid]) or (B['R'][tid] and B['miss'][tid]) else 1 if (B['R'][tid] and B['hit'][tid]) or (B['L'][tid] and B['miss'][tid]) else 2
...
np.full(len(TIME),lick,np.int8)
```

iii. The notes say the first three outputs are trial-level labels repeated over time so all outputs share one array shape.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived directly from `autowater`.

ii.
```python
B={k:scalar('obj/bp/'+k).astype(bool) for k in ('R','L','hit','miss','no','early','autowater')}
...
np.full(len(TIME),int(B['autowater'][tid]),np.int8)
```

iii. The notes describe this as a direct use of the autowater flag.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code writes `int(B['autowater'][tid])` into the context row for all time bins, so the boolean encoding becomes `0 = DR` and `1 = WC` in the saved dataset.

ii.
```python
trialsO.append(np.vstack([
    np.full(len(TIME),lick,np.int8),
    np.full(len(TIME),int(B['autowater'][tid]),np.int8),
    np.full(len(TIME),outcome,np.int8),
    td[j],pd[j],md[j]]))
```

```python
'output_values':[['left','right','none'],['DR','WC'],['incorrect','correct','ignore'], ...]
```

iii. No explicit justification for this code order was recorded beyond using the raw boolean flag directly.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `hit` and `miss` behavior flags.

ii.
```python
outcome=1 if B['hit'][tid] else 0 if B['miss'][tid] else 2
```

iii. The notes justify this as keeping ignore/no-response trials as the residual third class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps `hit -> correct (1)`, `miss -> incorrect (0)`, and anything else to `ignore (2)`, then repeats that label across all time bins.

ii.
```python
outcome=1 if B['hit'][tid] else 0 if B['miss'][tid] else 2
...
np.full(len(TIME),outcome,np.int8)
```

iii. In the notes, the AI says ignore trials are retained because the target explicitly requires an ignore class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-camera trajectory entry `traj[0]`, specifically the feature named `'tongue'`, together with that trial's `frameTimes`, tracked coordinates `ts`, the `haveVid` flag, and the go cue.

ii.
```python
tongue=feature_speed(traj,tids,0,'tongue',B)
```

```python
ft,ts,names=z
...
fi=names.index(name)
...
xy=a[:m,:2,fi]
rel=ft-B['go'][tid]
```

iii. The notes say the AI intended to use a consistent named tongue feature and speed magnitude while preserving not-visible bins.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The script normalizes trajectory array layout, uses finite `x/y` coordinates as the visibility mask, computes speed from `np.gradient` of `x` and `y` at visible frames, linearly interpolates that speed onto the 5 ms decoder grid, masks bins judged not visible after interpolating the raw-frame visibility mask, and later median-splits the resulting values.

ii.
```python
a=np.asarray(ts,float)
if a.shape[1]==3: pass
elif a.shape[0]==3: a=np.transpose(a,(2,0,1))
elif a.shape[2]==3: a=np.transpose(a,(0,2,1))
...
xy=a[:m,:2,fi]
vis=np.isfinite(xy).all(1)
vel=np.full(m,np.nan); dx=np.gradient(xy[:,0]); dy=np.gradient(xy[:,1]); vel[vis]=np.hypot(dx[vis],dy[vis])
rel=ft-B['go'][tid]
speed=interp_visible(rel,vel,TIME)
...
vi=np.interp(TIME,rr,vv,left=0,right=0)>=0.5; speed[~vi]=np.nan
```

iii. The notes justify this generally as using speed magnitude, interpolating to the common time axis, and preserving explicit missingness for non-visible bins.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single session-wide median (`np.nanmedian`) is computed across all finite tongue-velocity samples. Finite samples below the median are coded `0`, samples at or above the median are coded `1`, and `NaN` samples become class `2`.

ii.
```python
def discretize(x,missing_class=True):
    med=float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
    y=np.full(x.shape,2 if missing_class else 0,np.int8); ok=np.isfinite(x)
    if np.isfinite(med): y[ok]=(x[ok]>=med).astype(np.int8)
    return y,med
```

iii. The notes explicitly say the median split was chosen because the decoder task required a per-session 50th-percentile threshold.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue values are aligned by subtracting the trial's go cue from the raw frame times and then interpolating onto the shared neural `TIME` grid. The final code does not apply a separate video/behavior clock offset.

ii.
```python
rel=ft-B['go'][tid]
speed=interp_visible(rel,vel,TIME)
...
vi=np.interp(TIME,rr,vv,left=0,right=0)>=0.5
```

iii. The notes justify alignment in terms of putting video on the same go-cue axis as neural data, but they do not record a separate rationale for omitting the bitcode-based video offset in the final code.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-camera trajectory entry `traj[1]`, specifically the feature named `'top_paw'`, along with that trial's frame times, coordinates, `haveVid`, and go cue.

ii.
```python
paw=feature_speed(traj,tids,1,'top_paw',B)
```

iii. The notes say the AI wanted to use a fixed bottom-view paw feature consistently across sessions.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw processing reuses `feature_speed`: the script normalizes trajectory layout, takes finite `x/y` as visibility, computes speed from coordinate gradients, interpolates to the 5 ms grid, masks bins deemed not visible, and then discretizes by the session median.

ii.
```python
def feature_speed(traj,trial_ids,view,name,B):
    ...
    vis=np.isfinite(xy).all(1)
    vel=np.full(m,np.nan); dx=np.gradient(xy[:,0]); dy=np.gradient(xy[:,1]); vel[vis]=np.hypot(dx[vis],dy[vis])
    rel=ft-B['go'][tid]
    speed=interp_visible(rel,vel,TIME)
    ...
    vi=np.interp(TIME,rr,vv,left=0,right=0)>=0.5; speed[~vi]=np.nan
```

iii. The notes justify this at a high level as computing speed magnitude while preserving explicit missingness.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The script uses the same `discretize()` helper as for the tongue: below the session median is class `0`, at or above the median is class `1`, and missing samples are class `2`.

ii.
```python
pd,pmed=discretize(paw)
```

```python
def discretize(x,missing_class=True):
    med=float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
    ...
```

iii. The notes explicitly say median thresholds were chosen session-wise to satisfy the task.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity is aligned by subtracting the go cue from `frameTimes` and interpolating onto the shared `TIME` grid. As with the tongue, the final code does not apply a video-offset correction.

ii.
```python
rel=ft-B['go'][tid]
speed=interp_visible(rel,vel,TIME)
```

iii. The notes only justify the common-axis alignment at a high level.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the companion `motionEnergy_<session>.mat` file. The script unwraps nested `me.data` wrappers, takes one per-trial trace from that file, and uses side-camera frame times from `traj[0]` plus the go cue to place the values on the aligned time axis.

ii.
```python
mf=f.with_name(f.name.replace('data_structure_','motionEnergy_'))
...
me=loadmat(mf,squeeze_me=True,struct_as_record=False)['me']
raw=me.data
while hasattr(raw,'data'):
    raw=raw.data
vals=np.asarray(raw).flat
```

```python
ft=traj[0][tid][0]
rel=np.asarray(ft[:m])-B['go'][tid]
```

iii. The notes justify this by matching the companion-file schema used in the source data and by explicitly handling legacy nested `me.data.data` cases.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each kept trial, the per-frame motion-energy trace is linearly interpolated onto the shared `TIME` grid and then nearest-filled across remaining `NaN` bins before discretization.

ii.
```python
z=interp_visible(rel,y[:m],TIME)
z=nearest_fill(z).astype(np.float32)
out.append(z)
```

iii. The notes justify the nested-schema handling, but do not give a separate detailed rationale for the interpolation-plus-nearest-fill choice beyond producing a common aligned time series.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy uses the same session-wise median split as the velocity outputs: values below the median are class `0`, values at or above the median are class `1`, and missing values are class `2`.

ii.
```python
md,mmed=discretize(motion)
```

```python
def discretize(x,missing_class=True):
    med=float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
    ...
```

iii. The notes explicitly say the median threshold was mandated by the decoder task.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting the go cue from side-camera frame times and interpolating onto the same `TIME` grid used for neural data. No separate video offset is applied in the final code.

ii.
```python
ft=traj[0][tid][0]
...
rel=np.asarray(ft[:m])-B['go'][tid]
z=interp_visible(rel,y[:m],TIME)
```

iii. The notes justify this only in terms of putting motion energy on the common go-cue-centered axis.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable video/tracking entries produce all-`NaN` streams that later become category `2`. Within valid support, tongue and paw speed are linearly interpolated to the neural grid and then masked back to invisible where the interpolated visibility mask is false; motion energy is additionally nearest-filled across aligned gaps. Trials with all-zero neural activity after unit selection are dropped entirely.

ii.
```python
if z is None or not B['haveV'][tid]:
    out.append(np.full(len(TIME),np.nan,np.float32)); continue
...
speed=interp_visible(rel,vel,TIME)
...
speed[~vi]=np.nan
```

```python
z=interp_visible(rel,y[:m],TIME); z=nearest_fill(z).astype(np.float32)
```

```python
neural_valid=np.any(neural!=0,axis=(1,2))
tids=tids[neural_valid]; neural=neural[neural_valid]
```

iii. The notes say the AI wanted to preserve categorical missingness where possible and explicitly remove trailing invalid neural periods.

## 11-a. What are the most time-consuming steps of the code?

i. The AI's notes emphasize per-trial trajectory handling and single-trial spike histogram construction as the expensive parts, especially because large MATLAB trajectory objects must be iterated trial-by-trial and unit-by-unit.

ii.
```python
for q,tr,tm in units:
    ...
    for oi,tid in enumerate(trial_ids):
        vals=al[tr==tid]; mat[oi]=np.histogram(vals,EDGES)[0].astype(np.float32)/DT
```

```python
for tid in trial_ids:
    z=trials[tid] if tid<len(trials) else None
    ...
```

iii. In `CONVERSION_NOTES.md` Step 6, the AI explicitly calls out large raw trajectory data and single-trial spike histograms as the main inefficiency risks.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining vectorization targets are the per-neuron/per-trial loop in `neural_arrays`, the per-trial loop in `feature_speed`, and the per-trial loop in `motion_arrays`.

ii.
```python
for q,tr,tm in units:
    ...
    for oi,tid in enumerate(trial_ids):
        vals=al[tr==tid]
        mat[oi]=np.histogram(vals,EDGES)[0].astype(np.float32)/DT
```

```python
for tid in trial_ids:
    ...
```

iii. The notes say full vectorization is limited by heterogeneous MATLAB trajectory structures and variable-length trial/frame content.

## 11-c. What processing does the code repeat multiple times?

i. The script rebuilds the same 15-point smoothing kernel inside every unit loop, fills constant per-trial label vectors anew for every trial, and makes separate full passes over trials for tongue, paw, and motion processing.

ii.
```python
for q,tr,tm in units:
    ...
    N=15; alpha=2.5
    nn=np.arange(N,dtype=float)-(N-1)/2
    kern=np.exp(-0.5*(alpha*nn/((N-1)/2))**2)
    kern[:N//2]=0; kern/=kern.sum()
```

```python
tongue=feature_speed(traj,tids,0,'tongue',B)
paw=feature_speed(traj,tids,1,'top_paw',B)
motion,hasme=motion_arrays(f,traj,B,tids)
```

iii. No explicit justification for these repeated computations was recorded beyond keeping the implementation simple and session-local.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The loader materializes all trajectory features for both cameras and several raw behavior flags, even though only a small subset is used downstream. It also computes continuous tongue, paw, and motion time series and then discards those continuous values after discretizing them into classes.

ii.
```python
for ref in h['obj/traj'][:,0]:
    g=h[ref]; trials=[]
    ...
    names=[hchars(h,r) for r in np.asarray(fn).flat]
    trials.append((ft,ts,names))
```

```python
tongue=feature_speed(traj,tids,0,'tongue',B)
paw=feature_speed(traj,tids,1,'top_paw',B)
motion,hasme=motion_arrays(f,traj,B,tids)
td,tmed=discretize(tongue); pd,pmed=discretize(paw); md,mmed=discretize(motion)
```

iii. The notes emphasize simplicity, compact dtypes, and session-wise processing, but they do not give a separate justification for retaining and then discarding these continuous intermediates.
