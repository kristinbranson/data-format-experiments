# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads sessions only from the `Ephys_Behavior` folder, not from `RandomizedDelay_Ephys_Behavior`. It globs for `data_structure_*.mat` files in `Ephys_Behavior`, then filters to only include sessions belonging to 6 "Figure 8" mice (JEB6, JEB7, EKH3, JGR2, JGR3, JEB19). Each file is loaded using `mat73.loadmat()`. Motion energy files are loaded separately via `scipy.io.loadmat()`. The probe assignments are hard-coded in a `PROBES` dictionary derived from the paper's `load*_ALMVideo.m` scripts.

ii.
```python
ROOT='/app/data/Ephys_Behavior'
MICE={'JEB6','JEB7','EKH3','JGR2','JGR3','JEB19'}
PROBES={
 ('EKH3','2021-08-11'):[2], ('JEB6','2021-04-18'):[2],
 ('JEB7','2021-04-29'):[1], ('JEB7','2021-04-30'):[1],
 ...
}

for f in glob.glob(ROOT+'/data_structure_*.mat'):
    base=os.path.basename(f)[15:-4]; mouse,day=base.split('_',1)
    if (mouse,day) in PROBES: files.append((mouse,day,f))
```

iii. The AI reasoned that "Figure 8 explicitly loads JEB6, JEB7, EKH3, JGR2, JGR3, and JEB19 -- the paper's six two-context mice." It excluded the RandomizedDelay folder because it "lacks the required WC context" and excluded optogenetic folders as "primarily behavioral."

## 1-b. How are the data split into subjects?

i. The mouse name is extracted from the filename by splitting `base` at the first underscore. The `subjects` list is the sorted set of the 6 Figure 8 mice, and `subject_idx` maps each session to its index in that list.

ii.
```python
base=os.path.basename(f)[15:-4]; mouse,day=base.split('_',1)
subjects=sorted(MICE); subject_idx=[]
...
subject_idx.append(subjects.index(mouse))
```

iii. The AI identified the 6 mice from the Figure 8 analysis scripts in the paper's code.

## 1-c. How are the data split into sessions?

i. Each `.mat` file in `Ephys_Behavior` whose (mouse, day) pair is in the `PROBES` dictionary becomes one session. This yields 11 sessions from the fixed-delay task only; the 19 randomized-delay sessions are excluded.

ii.
```python
for f in glob.glob(ROOT+'/data_structure_*.mat'):
    base=os.path.basename(f)[15:-4]; mouse,day=base.split('_',1)
    if (mouse,day) in PROBES: files.append((mouse,day,f))
```

iii. The AI chose to use only Figure 8 sessions, which it identified as the paper's core two-context analysis subset.

## 1-d. How are the data split into trials?

i. Trials are identified by indexing into the Bpod fields. `Ntrials` gives the total number of trials. The AI creates `keep = np.flatnonzero(~early)` to get trial indices after filtering early-lick trials.

ii.
```python
n=int(b['Ntrials'])
early=np.asarray(b['early']).astype(bool); keep=np.flatnonzero(~early)
```

iii. The AI used the standard Bpod trial structure from the data files.

## 1-e. How are trials filtered based on quality controls?

i. Only early-lick trials are excluded. The AI does NOT filter photostimulation trials (`bp.stim.enable`). It also does NOT filter trials that run past the end of the recording. Ignore (no-response) trials are retained because the instructions require an "ignore" outcome class.

ii.
```python
early=np.asarray(b['early']).astype(bool); keep=np.flatnonzero(~early)
```

iii. The AI reasoned: "Early-lick trials should be excluded consistent with the paper, while ignore trials must remain because explicitly requested as an output class." The AI noted `bp.stim` fields during data exploration but did not act on them for filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu[probe-1]` spike-sorted clusters, specifically the `trial` and `trialtm` (spike time relative to trial start) fields. The go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
c=o['clu'][p-1]
for tr,tm in zip(c['trial'],c['trialtm']): units.append((np.asarray(tr,int).ravel()-1,np.asarray(tm,float).ravel()))
```

iii. The AI read the data structure and identified `clu.trial` and `clu.trialtm` as the spike data source, consistent with the repository's `getSeq.m`.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 5 ms bins from -2.5 to +2.5 s around the go cue, converted to firing rates (Hz) by dividing by `DT`, then smoothed with a **causal** Gaussian kernel: a `gaussian(15, std=2.8)` window with the first 7 samples zeroed and then normalized. This matches the paper's `mySmooth.m` implementation of a causal filter.

ii.
```python
def smooth_rates(x):
    k=gaussian(15, std=(15-1)/(2*2.5), sym=True); k[:7]=0; k/=k.sum()
    return np.apply_along_axis(lambda z:np.convolve(z,k,mode='same'),1,x)
...
h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT
rates[ui,j]=smooth_rates(h[None,:])[0]
```

iii. The AI traced through `mySmooth.m` and noted: "The smoothing kernel is explicitly causal: a Gaussian window of length 15 has its first half zeroed, is normalized, and is convolved with each binned spike train."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI includes ALL cluster quality labels -- no quality-based filtering is applied. Only a mean firing rate threshold of >1 Hz is used.

ii.
```python
for tr,tm in zip(c['trial'],c['trialtm']): units.append(...)  # all clusters added
use=rates.mean(axis=(1,2))>1.0; rates=rates[use]
```

iii. The AI noted that `params.quality = {'all'}` in the repository defaults accepts clusters of any quality, and cited the paper: "All units with firing rates exceeding 1 Hz were included in all other analyses."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are aligned to the go cue by subtracting each trial's `bp.ev.goCue` time before histogramming. This is consistent with the reference approach.

ii.
```python
go=np.asarray(b['ev']['goCue'],float).ravel()
h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT
```

iii. The AI confirmed from the repository that `getSeq` aligns using `alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins from -2.5 to +2.5 s, yielding 1000 time points. This matches `params.dt = 1/200` and `params.tmin/tmax` from the reference. No rebinning is applied.

ii.
```python
DT=.005; T0=-2.5; T1=2.5
EDGES=np.arange(T0,T1+DT/2,DT); TIME=EDGES[:-1]+DT/2
```

iii. The AI read these parameters directly from `getDefaultParams.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is defined by the time bin centers, not derived from raw data. It is the center of each 5 ms bin in the [-2.5, 2.5] s window.

ii.
```python
TIME=EDGES[:-1]+DT/2
ins.append(TIME[None,:].astype(np.float32))
```

iii. The time axis is constructed from the binning parameters.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing -- it is the bin center array directly.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the bin center array itself, which is the same grid the neural data is binned onto. Both share the same time axis by construction.

ii.
```python
TIME=EDGES[:-1]+DT/2  # same EDGES used for spike histogramming
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `bp.L` (left-instructed trials), `bp.R` (right-instructed trials), `bp.hit` (correct outcome), `bp.miss` (incorrect outcome), and `bp.no` (no-response).

ii.
```python
L=np.asarray(b['L']).astype(bool); R=np.asarray(b['R']).astype(bool)
hit=np.asarray(b['hit']).astype(bool); miss=np.asarray(b['miss']).astype(bool)
no=np.asarray(b['no']).astype(bool)
```

iii. The AI reasoned: "actual lick direction is instructed side on hits, opposite side on misses, and none on ignores."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. On hit trials, lick direction matches the instructed side (L->left=0, R->right=1). On miss trials, lick direction is the opposite of the instructed side. On no-response/ignore trials, lick direction is "none" (2).

ii.
```python
lick=2 if no[t] or (not hit[t] and not miss[t]) else (0 if (hit[t] and L[t]) or (miss[t] and R[t]) else 1)
```

iii. The AI correctly inferred lick direction from the combination of instructed side and outcome.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater` -- a boolean per-trial field indicating water-cued (WC) trials.

ii.
```python
aw=np.asarray(b['autowater']).astype(bool)
```

iii. The AI confirmed from the repository: "bp.autowater is directly used as the WC-versus-DR trial label."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `autowater=True` maps to WC (0), `autowater=False` maps to DR (1). This is implemented as `int(not aw[t])`.

ii.
```python
const=np.vstack([..., np.full(TIME.size,int(not aw[t])), ...])
```

iii. The AI initially encoded autowater directly but then corrected to `int(not aw[t])` to match the output_values ordering ['WC','DR'].

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit`, `bp.miss`, and `bp.no` per-trial flags.

ii.
```python
hit=np.asarray(b['hit']).astype(bool); miss=np.asarray(b['miss']).astype(bool)
no=np.asarray(b['no']).astype(bool)
```

iii. Same flags used for lick direction.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit trials -> correct (1), miss trials -> incorrect (0), no-response/ignore trials -> ignore (2).

ii.
```python
outcome=2 if no[t] or (not hit[t] and not miss[t]) else (1 if hit[t] else 0)
```

iii. The mapping follows the instruction's specification: incorrect=0, correct=1, ignore=2.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. `obj.traj[1]` (side camera), specifically the first 4 tracked features (tongue landmarks: indices 0-3). The `ts` array contains x, y, and likelihood per feature per frame, and `frameTimes` provides the timing.

ii.
```python
cam=traj[1]; ts=np.asarray(cam['ts'][trial],float); ft=np.asarray(cam['frameTimes'][trial],float).ravel()-go
def speed(ids):
    ids=[i for i in ids if i<ts.shape[2]]
    xy=ts[:,0:2,ids]; lk=ts[:,2,ids]
```

iii. The AI identified tongue features from the side camera based on the `findVelocity.m` script's feature indexing.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI computes the centroid of visible tongue landmarks (likelihood >= 0.9), takes the gradient of x and y positions per frame (not per second -- `np.gradient` without explicit spacing), computes speed as `np.hypot(vx, vy)`, sets speed to NaN where no landmark is visible, then interpolates to the 5 ms time grid using nearest-neighbor interpolation. Finally, discretized at the session 50th percentile of visible values.

ii.
```python
vis=np.any(lk>=.9,axis=1)
xy=np.where((lk>=.9)[:,None,:],xy,np.nan)
pos=np.nanmean(xy,axis=2)
vx=np.gradient(pos[:,0]); vy=np.gradient(pos[:,1]); sp=np.hypot(vx,vy)
sp[~vis]=np.nan
return interp_nearest(ft,sp,TIME), interp_nearest(ft,vis.astype(float),TIME,0)>=.5
```

iii. The AI noted: "MATLAB findVelocity uses gradient per frame (pixel/frame), not /dt." It used a centroid of multiple tongue landmarks rather than a single feature.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Split at the 50th percentile (median) of visible values across the session. 0 = below median, 1 = at or above median, 2 = not visible.

ii.
```python
def classes(x,vis):
    th=np.nanpercentile(x[vis],50) if np.any(vis) else np.nan
    z=np.full(x.shape,2,np.int8); z[vis]=(x[vis]>=th).astype(np.int8); return z,float(th)
tc,tth=classes(tongue,tvis)
```

iii. Follows the instruction's specification for discretization.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are shifted by subtracting the go cue time: `ft = frameTimes - go`. No video offset correction is applied. The resulting per-frame velocities are then interpolated to the 5 ms time grid using nearest-neighbor interpolation (`np.interp`).

ii.
```python
ft=np.asarray(cam['frameTimes'][trial],float).ravel()-go
...
return interp_nearest(ft,sp,TIME)
```

iii. The AI did not compute a video offset correction (as `findVideoOffset.m` does in the reference code). It directly subtracted the go cue from frame times.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. `obj.traj[1]` (side camera), features at indices 4-5 (top_paw and bottom_paw from the side camera).

ii.
```python
paw,pvis=speed([4,5])
```

iii. The AI identified paw features from the side camera based on `findVelocity.m`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue velocity: centroid of visible paw landmarks, gradient per frame, speed as hypot, nearest-neighbor interpolation to 5 ms grid, discretized at session median.

ii.
```python
def speed(ids):
    ...
    pos=np.nanmean(xy,axis=2)
    vx=np.gradient(pos[:,0]); vy=np.gradient(pos[:,1]); sp=np.hypot(vx,vy)
```

iii. Same processing pipeline as tongue.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: 50th percentile of visible values, 0/1/2 classes.

ii.
```python
pc,pth=classes(paw,pvis)
```

iii. Follows instruction specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times minus go cue (no video offset), nearest-neighbor interpolation to the 5 ms grid.

ii.
```python
ft=np.asarray(cam['frameTimes'][trial],float).ravel()-go
return interp_nearest(ft,sp,TIME)
```

iii. No video offset correction applied.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_*.mat` files loaded via `scipy.io.loadmat`. The `me.data` field contains per-trial motion energy traces.

ii.
```python
mef=os.path.join(ROOT,f'motionEnergy_{mouse}_{day}.mat')
me=loadmat(mef,squeeze_me=True,struct_as_record=False)['me'].data
```

iii. The AI found motion energy files alongside the data structure files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-trial motion energy trace is interpolated to the 5 ms time grid using nearest-neighbor interpolation (via `np.interp`), then discretized at the session 50th percentile.

ii.
```python
mv=np.asarray(me[t],float).ravel(); m=min(len(mv),len(ft))
z=interp_nearest(ft[:m],mv[:m],TIME); mes.append(z)
```

iii. Motion energy is already a single value per frame, so only interpolation and discretization are needed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same scheme: 50th percentile of finite values, 0/1/2 classes.

ii.
```python
mc,mth=classes(mes,mvis)
```

iii. Follows instruction specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the side camera's frame times (shifted by go cue, no video offset), then nearest-neighbor interpolated to the 5 ms grid.

ii.
```python
ft=np.asarray(cam['frameTimes'][trial],float).ravel()-go
...
z=interp_nearest(ft[:m],mv[:m],TIME)
```

iii. Same alignment as tongue and paw.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. When trajectory data is missing or malformed (wrong dimensions, trial index out of range), the AI returns NaN-filled arrays for tongue/paw velocity and marks visibility as False. Motion energy is filled with NaN when the file doesn't exist or the trial is out of range. NaN values are discretized as class 2 ("not visible" or "no video").

ii.
```python
if len(traj)<2 or trial>=len(traj[1]['ts']):
    return np.full(TIME.size,np.nan),np.zeros(TIME.size,bool),...
if ts.ndim!=3 or ts.shape[0]!=ft.size:
    return np.full(TIME.size,np.nan),...
```

iii. The AI handles missing data by filling with NaN and using the "not visible" class, preserving trials rather than dropping them.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the .mat files with `mat73.loadmat()` is the dominant cost. The AI noted it was "too slow because mat73 eagerly loads all spike and video arrays for every session."

ii.
```python
o=mat73.loadmat(f)['obj']
```

iii. The AI initially tried scanning all 25 sessions but switched to only loading the 11 needed sessions.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike counting loop iterates per-unit and per-trial: for each unit, it loops over unique trial numbers and histograms spikes individually. This could be vectorized using `np.histogram2d` as the reference does. The velocity computation also loops per trial.

ii.
```python
for ui,(tr,tm) in enumerate(units):
    for t in np.unique(tr):
        j=keepmap.get(int(t))
        if j is None or not np.isfinite(go[t]): continue
        h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT
        rates[ui,j]=smooth_rates(h[None,:])[0]
```

iii. The nested loop over units and trials is significantly less efficient than a vectorized histogram2d approach.

## 11-c. What processing does the code repeat multiple times?

i. The smoothing function `smooth_rates` is called inside the inner loop for every unit-trial combination, recreating the Gaussian kernel each time. The kernel could be created once outside the loop.

ii.
```python
def smooth_rates(x):
    k=gaussian(15, std=(15-1)/(2*2.5), sym=True); k[:7]=0; k/=k.sum()
    return np.apply_along_axis(lambda z:np.convolve(z,k,mode='same'),1,x)
```

iii. Each call to `smooth_rates` reconstructs the kernel, though this is a minor cost compared to file I/O.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads the entire `obj` structure via `mat73.loadmat()`, materializing fields never used (spike waveforms, unused tracking features, etc.). Also, the `bottom_paw` feature is loaded implicitly through the trajectory structure but never explicitly used. The AI also processes both `top_paw` and `bottom_paw` together via `speed([4,5])` but this is intentional (centroid of two paw landmarks).

ii.
```python
o=mat73.loadmat(f)['obj']  # loads everything
```

iii. Full materialization of all fields is required by the mat73 loading approach.
