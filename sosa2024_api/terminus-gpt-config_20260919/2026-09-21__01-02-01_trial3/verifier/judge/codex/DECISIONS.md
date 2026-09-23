# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every NWB file under `/app/data` by recursively globbing `*.nwb`. Each file is opened with `pynwb.NWBHDF5IO`, then the converter pulls behavior from `processing['behavior']['BehavioralTimeSeries']` and neural data from `processing['ophys']['Deconvolved']`.

ii.
```python
with NWBHDF5IO(str(fn),'r',load_namespaces=True) as io:
    nwb=io.read(); beh=nwb.processing['behavior']['BehavioralTimeSeries'].time_series
...
files=sorted(Path('/app/data').rglob('*.nwb')); files=files[:2] if a.sample else files
for i,fn in enumerate(files):
    N,I,O,sub,info=process_file(fn,a.show_processing)
```

iii. `CONVERSION_NOTES.md` says the released dataset contains 152 `*_behavior+ophys.nwb` files, one per recording session, and that all files were opened with `pynwb.NWBHDF5IO(..., load_namespaces=True)`. It frames this as retaining every nonduplicate NWB session.

## 1-b. How are the data split into subjects?

i. Subjects are not taken from directory names; instead, the AI uses `nwb.subject.subject_id` from each NWB file and builds a unique subject list in first-seen order.

ii.
```python
N,I,O,sub,info=process_file(fn,a.show_processing)
...
return neural_trials,inputs,outputs,nwb.subject.subject_id,info
...
if sub not in subjects: subjects.append(sub)
sidx.append(subjects.index(sub))
```

iii. The notes say subject/session metadata are available in the NWB files and emphasize using native session identity rather than relying on the reference code’s pickle layout.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session ordering is the sorted file order from the recursive glob.

ii.
```python
files=sorted(Path('/app/data').rglob('*.nwb')); files=files[:2] if a.sample else files
...
for i,fn in enumerate(files):
    N,I,O,sub,info=process_file(fn,a.show_processing); neural.append(N);inputs.append(I);outputs.append(O);infos.append(info)
```

iii. The notes repeatedly describe the dataset as “one file per recording session” and state that all 152 NWB files are retained as sessions.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from the frame-sampled behavior stream. The AI takes all indices where `trial_start > 0`, all indices where `teleport > 0`, then pairs each start with the first teleport index after that start and before the next start. Trial timing comes from the `trial_start` timestamps array.

ii.
```python
bt=np.asarray(beh['trial_start'].timestamps[:],float)
B={k:np.asarray(v.data[:]) for k,v in beh.items() if v.data.shape[0]==len(bt)}
starts=np.flatnonzero(B['trial_start']>0); tele=np.flatnonzero(B['teleport']>0)
pairs=[]
for j,s in enumerate(starts):
    e0=tele[tele>=s]
    if len(e0) and (j+1==len(starts) or e0[0]<starts[j+1]): pairs.append((int(s),int(e0[0])))
```

iii. The notes justify this as using the explicit timestamped `trial_start` and `teleport` pulse streams because the NWB `trials` table is absent and trial alignment should be to trial start.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a minimum-trial-length filter. Instead, it excludes any trial whose start or end falls outside the time range covered by every neural plane, and it raises an error if fewer than two such trials remain.

ii.
```python
neural_start=max(x[1][0] for x in plane_data)
neural_end=min(x[1][-1] for x in plane_data)
n_pairs_raw=len(pairs)
pairs=[(s,e) for s,e in pairs if bt[s]>=neural_start and bt[e]<=neural_end]
excluded_no_neural=n_pairs_raw-len(pairs)
if len(pairs)<2: raise ValueError(f'{fn}: fewer than 2 fully imaging-covered trials')
```

iii. `CONVERSION_NOTES.md` says this filter was added after validator warnings showed some behavioral trials continued after imaging ended, producing all-zero neural data if retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the NWB `processing['ophys']['Deconvolved']` `RoiResponseSeries`, plus the `iscell` column in the referenced segmentation table for curation. It does not use raw `Fluorescence` or `Neuropil`.

ii.
```python
dec=nwb.processing['ophys']['Deconvolved'].roi_response_series
...
region=np.asarray(rr.rois.data[:],dtype=int)
table=rr.rois.table
iscell=np.asarray(table['iscell'].data[:])[:,0]>0
valid=iscell[region]
...
neural_all=np.asarray(rr.data[:, :],np.float32)[:,valid]
```

iii. The notes explicitly say the released NWBs already contain the author-processed deconvolved event signal and therefore “no new delta-F/F calculation is indicated”; the converter should use `Deconvolved` directly.

## 2-b. How is the `neural` data processed?

i. For each imaging plane, the AI applies `iscell` curation, determines timestamps or reconstructs them from `starting_time` and an effective sampling rate, and bins the native deconvolved signal into 100 ms bins by averaging samples whose timestamps fall inside each bin. If a bin has no samples, it copies the nearest sample. It then concatenates planes into one neuron-by-time matrix per trial.

ii.
```python
if rr.timestamps is not None:
    neural_t=np.asarray(rr.timestamps[:],float)
    effective_rate=1.0/np.median(np.diff(neural_t))
else:
    effective_rate=nominal_rate/n_planes
    neural_t=float(rr.starting_time)+np.arange(nt)/effective_rate
...
centers=t_start+DT/2+np.arange(max(1,int(np.floor((t_end-t_start)/DT))))*DT
...
ni=np.searchsorted(neural_t,centers-DT/2); nj=np.searchsorted(neural_t,centers+DT/2)
xp=np.empty((neural_all.shape[1],len(centers)),np.float32)
for k,(a,b) in enumerate(zip(ni,nj)):
    if b>a: xp[:,k]=neural_all[a:b].mean(0)
    else: xp[:,k]=neural_all[nearest_idx(neural_t,np.array([centers[k]]))[0]]
...
X=np.concatenate(binned_planes,axis=0)
```

iii. The notes justify this by saying the decoder requires a common time bin across sessions, so the AI chose 100 ms bins and mean-binned the deconvolved events instead of reproducing the paper’s original raw-to-dF/F-to-deconvolution pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural filtering consists of keeping only ROIs marked as `iscell` in the segmentation table. The AI does not apply any putative interneuron filter. Trial-level neural filtering is handled separately by excluding trials lacking complete neural coverage.

ii.
```python
iscell=np.asarray(table['iscell'].data[:])[:,0]>0
valid=iscell[region]
...
neural_all=np.asarray(rr.data[:, :],np.float32)[:,valid]
...
pairs=[(s,e) for s,e in pairs if bt[s]>=neural_start and bt[e]<=neural_end]
```

iii. The notes argue that `iscell` is the only general curation that should be applied for the decoder and that place/reward-cell selection or speed-related filtering would be analysis-specific leakage.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the explicit `trial_start` pulse. For each trial, the bin centers are defined relative to `t_start`, and neural bins are constructed on those trial-centered timestamps.

ii.
```python
t_start,t_end=bt[s],bt[e]
centers=t_start+DT/2+np.arange(max(1,int(np.floor((t_end-t_start)/DT))))*DT
centers=centers[centers<=t_end]
...
inp=np.vstack([centers-t_start, ...]).astype(np.float32)
```

iii. The notes say trial alignment should be to the explicit `trial_start` timestamp, and that inter-trial periods after teleport are excluded.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 100 ms bin size (`DT = 0.1`). Yes: the AI rebins both neural and behavioral variables onto these 100 ms bins.

ii.
```python
DT=0.1
...
centers=t_start+DT/2+np.arange(max(1,int(np.floor((t_end-t_start)/DT))))*DT
...
'metadata':{...,'time_bin_size':DT*1000,...}
```

iii. The notes explicitly call out “100 ms for every session” as a key decision so all sessions share one common temporal resolution.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This input is derived from the `trial_start` timestamps array, not from a per-sample behavior timestamps stream. The AI uses each trial’s start time `t_start` and synthetic 100 ms bin centers built from that start and end time.

ii.
```python
bt=np.asarray(beh['trial_start'].timestamps[:],float)
...
t_start,t_end=bt[s],bt[e]
centers=t_start+DT/2+np.arange(max(1,int(np.floor((t_end-t_start)/DT))))*DT
...
inp=np.vstack([centers-t_start, ...]).astype(np.float32)
```

iii. The notes justify this as part of the global 100 ms rebinned representation aligned to trial start.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The AI constructs evenly spaced 100 ms bin centers spanning the retained part of the trial, then subtracts the trial start time so the first row is seconds from trial start.

ii.
```python
centers=t_start+DT/2+np.arange(max(1,int(np.floor((t_end-t_start)/DT))))*DT
centers=centers[centers<=t_end]
...
inp=np.vstack([centers-t_start, ...]).astype(np.float32)
```

iii. The notes say the decoder needs a common temporal grid, so trial-relative time is represented on those rebinned centers rather than raw behavior timestamps.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: the time-from-start row uses the same 100 ms bin centers that are used to bin neural activity for that trial.

ii.
```python
ni=np.searchsorted(neural_t,centers-DT/2); nj=np.searchsorted(neural_t,centers+DT/2)
...
inp=np.vstack([centers-t_start, ...]).astype(np.float32)
assert X.shape[1]==inp.shape[1]==out.shape[1] and np.isfinite(X).all()
```

iii. The notes describe the converted representation as one common timestamp grid per trial shared across neural, input, and output arrays.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type comes from the `environment` behavior time series.

ii.
```python
env=int(round(float(B['environment'][s]))); env=max(0,min(1,env))
...
inp=np.vstack([centers-t_start,np.full(len(centers),env), ...]).astype(np.float32)
```

iii. The notes list `environment` as the source variable and describe it as a native 0/1 field mapped to ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI reads the environment value at the trial start index, rounds it to an integer, clamps it into `[0, 1]`, and repeats it across all time bins in the trial.

ii.
```python
env=int(round(float(B['environment'][s]))); env=max(0,min(1,env))
...
inp=np.vstack([centers-t_start,np.full(len(centers),env), ...]).astype(np.float32)
```

iii. The notes say environment is a per-trial binary context variable that should be constant across time within a trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is taken directly from the `trial number` behavior time series at the trial start index.

ii.
```python
trialnum=float(B['trial number'][s])
...
inp=np.vstack([centers-t_start,np.full(len(centers),env),np.full(len(centers),trialnum), ...]).astype(np.float32)
```

iii. The notes say `trial number` is a native session behavior variable and map it directly to the decoder input.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The AI converts the trial-start sample to `float` and repeats that constant value across all time bins in the trial.

ii.
```python
trialnum=float(B['trial number'][s])
...
inp=np.vstack([centers-t_start,np.full(len(centers),env),np.full(len(centers),trialnum),np.full(len(centers),prev)]).astype(np.float32)
```

iii. The notes treat trial number as a per-trial input that should be repeated over time for the validator format.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from the sparse `Reward` event timestamps. For each retained trial, the AI first computes whether the current trial was rewarded based on whether any reward timestamp falls within the trial window, then uses the previous retained trial’s outcome as the next trial’s `previous trial outcome`.

ii.
```python
rew_t=np.asarray(beh['Reward'].timestamps[:],float)
...
trialnum=float(B['trial number'][s]); rewarded=int(np.any((rew_t>=t_start)&(rew_t<=t_end)))
prev=outcomes[-1] if outcomes else 0; outcomes.append(rewarded)
...
inp=np.vstack([centers-t_start, ..., np.full(len(centers),prev)]).astype(np.float32)
```

iii. The notes say previous outcome should come from sparse Reward events and be represented as rewarded=1 versus omitted=0.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward outcome for each retained trial is computed as `any(Reward timestamp in [t_start, t_end])`. The first retained trial gets `0` for previous outcome; later retained trials inherit the previous retained trial’s `rewarded` label and repeat it across all bins.

ii.
```python
rewarded=int(np.any((rew_t>=t_start)&(rew_t<=t_end)))
prev=outcomes[-1] if outcomes else 0; outcomes.append(rewarded)
...
inp=np.vstack([centers-t_start,np.full(len(centers),env),np.full(len(centers),trialnum),np.full(len(centers),prev)]).astype(np.float32)
```

iii. The notes frame this as a per-trial context variable and explicitly say the first trial is set to 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from interpolated position and an inferred reward-zone identity for the trial. The zone identity itself is inferred from the `reward_zone` time series together with the animal’s position when `reward_zone > 0`.

ii.
```python
hit=np.flatnonzero(B['reward_zone'][s:e+1]>0)
if len(hit): raw_zone.append(zone_for_trial(float(np.median(B['position'][s+hit]))))
else: raw_zone.append(-1)
...
pos=np.interp(centers,bt,B['position']).astype(np.float32)
...
z=int(zones[ti]); lo,hi=ZONES[z]
dist=np.where(pos<lo,pos-lo,np.where(pos>hi,pos-hi,0.)).astype(np.float32)
```

iii. The notes say the native `reward_zone` stream is not itself an A/B/C label, so the active fixed zone must be inferred from where that stream activates in position space.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI first infers a zone per trial using the median position of samples where `reward_zone > 0`, mapped to the nearest fixed zone center among `[80,100]`, `[200,220]`, and `[320,340]`. Trials with no `reward_zone` activation are filled from the nearest labeled trial. It then computes signed distance to the nearest edge of that zone, with zero inside the zone.

ii.
```python
ZONES=np.array([[80.,100.],[200.,220.],[320.,340.]])
...
def zone_for_trial(pos):
    return int(np.argmin(np.abs(ZONES.mean(1)-pos)))
...
good=np.flatnonzero(np.asarray(raw_zone)>=0)
...
zones[bad]=zones[good[np.argmin(abs(good[:,None]-bad),axis=0)]] if len(bad) else zones[bad]
...
z=int(zones[ti]); lo,hi=ZONES[z]
dist=np.where(pos<lo,pos-lo,np.where(pos>hi,pos-hi,0.)).astype(np.float32)
```

iii. The notes justify the omission handling by saying reward-location blocks are contiguous, so unlabeled omission trials can be filled from nearby labeled trials.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is mapped into seven integer classes using explicit threshold comparisons: `< -50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, and `> 50`.

ii.
```python
def discretize_distance(d):
    y=np.empty(d.shape,np.int64)
    y[d < -50]=0; y[(d>=-50)&(d<-10)]=1; y[(d>=-10)&(d<0)]=2
    y[d==0]=3; y[(d>0)&(d<=10)]=4; y[(d>10)&(d<=50)]=5; y[d>50]=6
    return y
...
out=np.vstack([discretize_distance(dist), ...]).astype(np.int64)
```

iii. The notes say these bins were chosen to match the decoder task specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is first interpolated onto the same 100 ms bin centers used for neural activity, and distance is then computed on that shared time base.

ii.
```python
pos=np.interp(centers,bt,B['position']).astype(np.float32)
...
ni=np.searchsorted(neural_t,centers-DT/2); nj=np.searchsorted(neural_t,centers+DT/2)
...
dist=np.where(pos<lo,pos-lo,np.where(pos>hi,pos-hi,0.)).astype(np.float32)
```

iii. The notes explicitly describe position and speed as linearly interpolated to the common bin centers so outputs and neural data are time-aligned.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived from the `position` behavior time series.

ii.
```python
pos=np.interp(centers,bt,B['position']).astype(np.float32)
```

iii. The notes identify `position` as the native track-position variable in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI linearly interpolates raw position onto the 100 ms trial-centered bins and then discretizes those interpolated values into 5 track bins.

ii.
```python
pos=np.interp(centers,bt,B['position']).astype(np.float32)
poscls=np.digitize(pos,[90,180,270,360],right=False).astype(np.int64)
...
out=np.vstack([discretize_distance(dist),poscls, ...]).astype(np.int64)
```

iii. The notes say continuous behavior variables should be interpolated to the common temporal grid before discretization.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The AI uses `np.digitize` with thresholds at 90, 180, 270, and 360 cm, producing five categories.

ii.
```python
poscls=np.digitize(pos,[90,180,270,360],right=False).astype(np.int64)
```

iii. The notes say this implements five equal-width bins across the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Absolute position is aligned by interpolation to the same 100 ms bin centers used for neural binning.

ii.
```python
centers=t_start+DT/2+np.arange(max(1,int(np.floor((t_end-t_start)/DT))))*DT
...
pos=np.interp(centers,bt,B['position']).astype(np.float32)
```

iii. The notes say all continuous behavior outputs are put on the common neural bin grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from the `lick` behavior time series.

ii.
```python
jj=nearest_idx(bt,centers)
lick=(B['lick'][jj]>0).astype(np.int64)
```

iii. The notes describe lick as a frame-sampled behavior variable that is binarized for decoding.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI aligns lick by taking the nearest behavior sample to each 100 ms bin center and then binarizes it with `> 0`.

ii.
```python
jj=nearest_idx(bt,centers)
lick=(B['lick'][jj]>0).astype(np.int64)
...
out=np.vstack([...,lick,...]).astype(np.int64)
```

iii. The notes justify binarizing lick by citing the paper’s use of binary licks per frame before downstream smoothing in the GLM analyses.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned to neural data by nearest-neighbor resampling from behavior timestamps onto the shared 100 ms neural bin centers.

ii.
```python
jj=nearest_idx(bt,centers)
lick=(B['lick'][jj]>0).astype(np.int64)
```

iii. The notes distinguish discrete streams from continuous ones: continuous variables are interpolated, while discrete streams such as lick use nearest-sample alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the `reward_zone` behavior time series together with `position`, exactly as in the zone-inference step used for distance to reward zone.

ii.
```python
hit=np.flatnonzero(B['reward_zone'][s:e+1]>0)
if len(hit): raw_zone.append(zone_for_trial(float(np.median(B['position'][s+hit]))))
else: raw_zone.append(-1)
...
z=int(zones[ti])
```

iii. The notes state that the raw `reward_zone` stream is transient and not itself the A/B/C identity, so position-localized activation is used to infer the active fixed zone.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI infers a trial-level zone index `z` using the same procedure as in 7-b, then repeats that integer (`0=A`, `1=B`, `2=C`) across all time bins in the trial.

ii.
```python
good=np.flatnonzero(np.asarray(raw_zone)>=0)
...
zones[bad]=zones[good[np.argmin(abs(good[:,None]-bad),axis=0)]] if len(bad) else zones[bad]
...
z=int(zones[ti])
...
out=np.vstack([...,np.full(len(centers),z), ...]).astype(np.int64)
```

iii. The notes say reward-location blocks are contiguous, so nearest labeled trials can supply omission trials with no local `reward_zone` activation.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the sparse `Reward` event timestamps.

ii.
```python
rew_t=np.asarray(beh['Reward'].timestamps[:],float)
...
rewarded=int(np.any((rew_t>=t_start)&(rew_t<=t_end)))
```

iii. The notes identify `Reward` as an event `TimeSeries` whose timestamps determine rewarded versus omission trials.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI checks whether any reward timestamp lies between that trial’s start and end timestamps, converts that Boolean to `0/1`, and repeats it across all bins in the trial.

ii.
```python
rewarded=int(np.any((rew_t>=t_start)&(rew_t<=t_end)))
...
out=np.vstack([...,np.full(len(centers),rewarded)]).astype(np.int64)
```

iii. The notes say rewarded and omission trials should both be preserved, with reward outcome represented as a trial-level binary variable.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases directly in code: multi-plane ROI references are resolved plane-by-plane through each `DynamicTableRegion`; missing neural timestamps are reconstructed from `starting_time` and `rate`; empty 100 ms bins fall back to the nearest neural sample; trials outside complete neural coverage are discarded; omission trials with no `reward_zone` activation are filled from the nearest labeled trial; and the code raises hard errors if no zone can be inferred or fewer than two valid trials remain.

ii.
```python
if rr.timestamps is not None:
    neural_t=np.asarray(rr.timestamps[:],float)
    effective_rate=1.0/np.median(np.diff(neural_t))
else:
    effective_rate=nominal_rate/n_planes
    neural_t=float(rr.starting_time)+np.arange(nt)/effective_rate
...
if b>a: xp[:,k]=neural_all[a:b].mean(0)
else: xp[:,k]=neural_all[nearest_idx(neural_t,np.array([centers[k]]))[0]]
...
pairs=[(s,e) for s,e in pairs if bt[s]>=neural_start and bt[e]<=neural_end]
...
if not len(good): raise ValueError(f'{fn}: cannot infer reward zone')
```

iii. The notes explain that these checks were added after debugging multi-plane ROI mismatches and validator warnings from trials extending beyond imaging coverage.

## 13-a. What are the most time-consuming steps of the code?

i. The code’s main expensive operations are loading full NWB sessions, materializing full deconvolved arrays for every plane, looping over trials and 100 ms bins to average neural samples, and serializing the final pickle. Plot generation can add extra cost when `--show-processing` is enabled.

ii.
```python
with NWBHDF5IO(str(fn),'r',load_namespaces=True) as io:
    nwb=io.read()
...
neural_all=np.asarray(rr.data[:, :],np.float32)[:,valid]
...
for ti,(s,e) in enumerate(pairs):
    ...
    for plane_name,neural_t,neural_all in plane_data:
        ...
        for k,(a,b) in enumerate(zip(ni,nj)):
            if b>a: xp[:,k]=neural_all[a:b].mean(0)
```

iii. The notes say the converter reads each full deconvolved matrix once per session and identify the “small per-bin mean loop” as the remaining bounded hotspot.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunity is the inner loop over 100 ms bins inside each plane, which currently averages one bin at a time. The outer per-trial loops for pairing starts, inferring zones, and building constant label arrays are also serial.

ii.
```python
for j,s in enumerate(starts):
    e0=tele[tele>=s]
    if len(e0) and (j+1==len(starts) or e0[0]<starts[j+1]): pairs.append((int(s),int(e0[0])))
...
for s,e in pairs:
    hit=np.flatnonzero(B['reward_zone'][s:e+1]>0)
...
for ti,(s,e) in enumerate(pairs):
    ...
    for plane_name,neural_t,neural_all in plane_data:
        ...
        for k,(a,b) in enumerate(zip(ni,nj)):
            if b>a: xp[:,k]=neural_all[a:b].mean(0)
```

iii. The notes claim large repeated I/O was already removed, so the remaining optimization target is mostly within the per-trial binning loops.

## 13-c. What processing does the code repeat multiple times?

i. The code largely avoids a separate survey/conversion pass, but it still repeats a few computations: `searchsorted` is rerun for every plane and trial, `nearest_idx` can be called again for empty bins, and constant trial-level labels are rebuilt with `np.full` for every trial. Diagnostic plots also re-read already computed arrays when plotting.

ii.
```python
for ti,(s,e) in enumerate(pairs):
    ...
    for plane_name,neural_t,neural_all in plane_data:
        ni=np.searchsorted(neural_t,centers-DT/2); nj=np.searchsorted(neural_t,centers+DT/2)
        ...
        for k,(a,b) in enumerate(zip(ni,nj)):
            if b>a: xp[:,k]=neural_all[a:b].mean(0)
            else: xp[:,k]=neural_all[nearest_idx(neural_t,np.array([centers[k]]))[0]]
    ...
    inp=np.vstack([centers-t_start,np.full(len(centers),env),np.full(len(centers),trialnum),np.full(len(centers),prev)]).astype(np.float32)
```

iii. The notes explicitly contrast this with the earlier survey-heavy design and say the final script reads each full neural matrix only once per session.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter computes and stores session-level bookkeeping that the decoder itself does not use, including `rates`, per-session `info`, trial-zone histograms, reward counts, and optional plotting support. It also carries an unused `show` argument in `process_file`. These are mostly diagnostics/metadata rather than data used downstream by the decoder.

ii.
```python
def process_file(fn, show=False):
...
plane_data=[]; rates=[]
...
info=dict(session_id=sid,file=str(fn),n_trials=len(pairs),n_neurons=int(n_valid),rate=float(rates[0]),plane_rates=rates,n_planes=len(plane_data),zones=np.bincount(zones,minlength=3).tolist(),rewarded=int(sum(outcomes)),excluded_no_neural=excluded_no_neural)
...
if a.show_processing and i<2: plot_session(info,N,I,O)
```

iii. The notes emphasize extensive sanity checks and metadata auditing, so this extra bookkeeping was intentional for validation rather than for the decoder model itself.
