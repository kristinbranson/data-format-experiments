# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as a master index, iterates over experiment groups and their entries, and builds a mapping from base session IDs to behavioral data. Each behavior file (`Beh_<group>.npy`) is loaded, and sessions are matched by constructing full and base session IDs. Neural data files are discovered via glob from `spk/` and matched by session ID. Retinotopy files are loaded from the `retinotopy/` directory.

ii.
```python
info=np.load(os.path.join(ROOT,'beh','Imaging_Exp_info.npy'),allow_pickle=True).item()
beh={}; source_group={}; rowmap={}
for group,rows in info.items():
    bf=os.path.join(ROOT,'beh','Beh_'+group+'.npy')
    if not os.path.exists(bf): continue
    bd=np.load(bf,allow_pickle=True).item()
    for r in rows:
        full=sid_of(r); base=base_sid(r)
        key=full if full in bd else base
        if key in bd and base not in beh:
            beh[base]=bd[key]; source_group[base]=group; rowmap[base]=r
spkfiles={os.path.basename(f).replace('_neural_data.npy',''):f for f in glob.glob(os.path.join(ROOT,'spk','*_neural_data.npy'))}
sessions=sorted(set(beh)&set(spkfiles))
```

iii. The AI explored the data structure in multiple steps, discovering that behavioral files are keyed by session IDs and neural files are stored per session. It built a mapping keeping only the first occurrence of each base session ID.

## 1-b. How are the data split into subjects?

i. The mouse name is extracted as the first component of the session ID string (splitting by `_`). Subjects are collected from all processed sessions as a sorted unique set.

ii.
```python
subj.append(s.split('_')[0])
subjects=sorted(set(subj)); subject_idx=np.array([subjects.index(x) for x in subj],dtype=np.int16)
```

iii. The AI used the session ID naming convention where the first underscore-delimited field is the mouse name.

## 1-c. How are the data split into sessions?

i. A session is identified by the base session ID (`mname_datexp_blk`). The AI intersects available behavioral data with available spike files to determine usable sessions. Duplicate sessions appearing under multiple experiment types are deduplicated by keeping only the first occurrence of each base session ID.

ii.
```python
sessions=sorted(set(beh)&set(spkfiles))
```

iii. The AI identified that the same recording can appear under multiple experiment types in `Imaging_Exp_info` and kept only the first occurrence per base session ID.

## 1-d. How are the data split into trials?

i. Trials are identified using `ft_trInd` (frame-to-trial index) and `ft_CorrSpc` (corridor space flag). The AI filters to frames where the mouse is in the corridor space and where position is within `[0, Texture_Length)`. Trials with fewer than 2 valid frames are skipped.

ii.
```python
for tr in range(min(ntr,len(trialstim))):
    ix=np.where((tri==tr)&np.asarray(b['ft_CorrSpc'][:nfr],dtype=bool)&np.isfinite(pos)&(pos>=0)&(pos<float(b['Texture_Length'])))[0]
    if ix.size<2: continue
```

iii. The AI used the corridor space indicator combined with position bounds to identify valid frames within each trial, consistent with the paper's focus on visual corridor traversals.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 valid corridor frames are dropped. No outlier trial length filtering is applied. Sessions with fewer than 2 valid trials are skipped entirely.

ii.
```python
if ix.size<2: continue
...
if len(ns)<2: print('skip',s,'too few trials'); continue
```

iii. The AI applied a minimal filter requiring at least 2 frames per trial and 2 trials per session. It did not implement any trial length percentile-based filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files (`spk/<session_id>_neural_data.npy`), which contains a list of arrays (one per imaging plane). The arrays are concatenated along the neuron axis.

ii.
```python
raw=np.load(spkfiles[s],allow_pickle=True).item()['spks']; spk=np.concatenate(raw,axis=0); del raw
```

iii. The AI confirmed from the methods that all analyses use Suite2p deconvolved fluorescence traces and that `spks` contains these.

## 2-b. How is the `neural` data processed?

i. The neural data is temporally rebinned into non-overlapping 1-second bins. Within each bin, the neural activity is averaged across frames falling in that bin. The data is stored as float32.

ii.
```python
nb=max(1,int(np.floor(rel[-1]/BIN_S))+1); bins=np.minimum((rel/BIN_S).astype(int),nb-1)
N=np.empty((spk.shape[0],nb),np.float32)
for j in range(nb):
    jj=ix[bins==j]
    if jj.size: N[:,j]=spk[:,jj].mean(1,dtype=np.float32)
    else: N[:,j]=N[:,j-1] if j else 0
```

iii. The AI decided to average neural activity in 1-second bins to make the dataset more manageable while retaining temporal structure. Empty bins are forward-filled.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are filtered. All neurons from all imaging planes are kept. Brain region indices are assigned via retinotopy, but neurons outside the four visual areas are retained (labeled as "unassigned") rather than dropped.

ii.
```python
def region_indices(sid,n):
    ...
    out=np.zeros(len(a),dtype=np.int16)
    out[a==1]=1; out[np.isin(a,[2,3])]=2; out[np.isin(a,[4,5])]=3; out[np.isin(a,[6,7])]=4
    ...
    return out
```

iii. The AI treated region assignment as metadata rather than a filter, keeping all neurons regardless of their retinotopic area.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry). The time axis starts at the first valid corridor frame of each trial. Frames are rebinned into 1-second bins relative to this start time.

ii.
```python
t0=tsec[ix[0]]; rel=tsec[ix]-t0; nb=max(1,int(np.floor(rel[-1]/BIN_S))+1); bins=np.minimum((rel/BIN_S).astype(int),nb-1)
```

iii. The AI aligned to corridor entry by setting `t0` to the timestamp of the first valid corridor frame.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins the data into 1-second non-overlapping bins (`BIN_S=1.0`), averaging neural activity within each bin. The time_bin_size in metadata is set to 1000.0 ms.

ii.
```python
ROOT='/app/data'; BIN_S=1.0
...
'time_bin_size':1000.0
```

iii. The AI chose 1-second bins to reduce data size while retaining temporal structure relevant to the decoder task.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr`, the frame at which the sound cue was played in each trial, and frame timestamps `ft` converted to seconds.

ii.
```python
sound=np.asarray(b['SoundFr'])
cue_rel=(np.interp(sound[tr],np.arange(nfr),tsec)-t0)
```

iii. The AI identified `SoundFr` as the fractional frame index of the sound cue.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue frame is interpolated onto the time axis, converted to seconds relative to trial start, then subtracted from each bin's elapsed time. The result is `cue_time - bin_time`, so it is positive before the cue and negative after.

ii.
```python
cue_rel=(np.interp(sound[tr],np.arange(nfr),tsec)-t0) if tr<len(sound) and np.isfinite(sound[tr]) else np.nan
elapsed=(np.arange(nb,dtype=np.float32)+.5)*BIN_S
tcue=(cue_rel-elapsed).astype(np.float32) if np.isfinite(cue_rel) else np.full(nb,np.nan,np.float32)
```

iii. The AI computes time to cue as `cue_rel - elapsed`, making it positive before the sound cue and negative after.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The elapsed time array uses the same bin centers as the neural data bins, so alignment is through the shared 1-second binning grid.

ii.
```python
elapsed=(np.arange(nb,dtype=np.float32)+.5)*BIN_S
tcue=(cue_rel-elapsed).astype(np.float32)
```

iii. Both neural and input data share the same temporal binning, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date string in each session's experiment info entry (`datexp`), converted to a date ordinal. The first session date per subject is subtracted to get relative days.

ii.
```python
def date_ordinal(date):
    return float(np.datetime64(date.replace('_','-'),'D').astype(int))
subject_day0={}
for s in sessions:
    mouse=s.split('_')[0]; d=date_ordinal(rowmap[s]['datexp'])
    subject_day0[mouse]=min(subject_day0.get(mouse,d),d)
```

iii. The AI computed training day as the number of calendar days since the subject's first session.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The date ordinal of each session is computed, the earliest session date per subject is found, and the difference plus 1 gives the day of training. This is broadcast across all bins in each trial.

ii.
```python
day=np.full(nb,date_ordinal(rowmap[s]['datexp'])-subject_day0[s.split('_')[0]]+1.0,np.float32)
```

iii. The AI adds 1 so training starts at day 1 rather than day 0. This is a per-trial constant broadcast across time bins.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from the elapsed time within each trial, computed from the 1-second bin structure.

ii.
```python
elapsed=(np.arange(nb,dtype=np.float32)+.5)*BIN_S
```

iii. The AI constructs elapsed time from the bin indices rather than from raw frame timestamps.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The elapsed time is computed as `(bin_index + 0.5) * BIN_S`, placing each value at the center of its 1-second bin.

ii.
```python
elapsed=(np.arange(nb,dtype=np.float32)+.5)*BIN_S
```

iii. The AI centers the time at the midpoint of each bin.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. The elapsed time array has the same number of bins as the neural data, so alignment is through the shared binning grid.

ii.
```python
elapsed=(np.arange(nb,dtype=np.float32)+.5)*BIN_S
```

iii. Same bin structure as neural data ensures alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `TrialStim` and `RewardFr`. The AI infers which stimulus category is rewarded by counting which stimulus has the most finite reward frames. A trial is marked as rewarded if its stimulus matches the inferred rewarded stimulus.

ii.
```python
reward_stim={}
for s in sessions:
    b=beh[s]; st=np.asarray(b['TrialStim']); rw=np.asarray(b['RewardFr'])
    counts={x:int(np.isfinite(rw[st==x]).sum()) for x in np.unique(st)}
    reward_stim[s]=max(counts,key=counts.get) if counts and max(counts.values())>0 else None
...
rewarded=np.full(nb,int(reward_stim[s] is not None and str(trialstim[tr])==reward_stim[s]),np.float32)
```

iii. The AI inferred reward availability from which stimulus type received the most rewards in each session, rather than using `isRew` directly.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. For each session, the AI counts how many finite `RewardFr` values correspond to each unique `TrialStim`. The stimulus with the most rewards is deemed the "rewarded" stimulus. Each trial is then marked 1 if its stimulus matches the rewarded stimulus, 0 otherwise. Unsupervised/naive sessions with no rewards get all zeros.

ii.
```python
counts={x:int(np.isfinite(rw[st==x]).sum()) for x in np.unique(st)}
reward_stim[s]=max(counts,key=counts.get) if counts and max(counts.values())>0 else None
rewarded=np.full(nb,int(reward_stim[s] is not None and str(trialstim[tr])==reward_stim[s]),np.float32)
```

iii. The AI inferred reward availability indirectly rather than using the direct `isRew` field.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `TrialStim`, which names the stimulus of each trial.

ii.
```python
stim_names=sorted({str(x) for s in sessions for x in np.unique(beh[s]['TrialStim'])})
stim_id={x:i for i,x in enumerate(stim_names)}
cat=np.full(nb,stim_id[str(trialstim[tr])],np.int16)
```

iii. The AI used `TrialStim` to identify stimulus categories.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique `TrialStim` values across sessions are sorted and assigned integer indices. Each trial's stimulus is mapped to its index. The value is per-trial, broadcast across all time bins.

ii.
```python
stim_names=sorted({str(x) for s in sessions for x in np.unique(beh[s]['TrialStim'])})
stim_id={x:i for i,x in enumerate(stim_names)}
cat=np.full(nb,stim_id[str(trialstim[tr])],np.int16)
```

iii. The AI uses the raw `TrialStim` values directly rather than mapping them to broader texture categories (circle, leaf, rock, wood).

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame number of each lick in the session.

ii.
```python
lick=np.asarray(b['LickFr'])
lf=lick[np.isfinite(lick)] if lick.size else lick
```

iii. The AI identified `LickFr` as containing lick event frame indices.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frames are interpolated onto the time axis and assigned to 1-second bins. If any lick falls within a bin, that bin is marked as 1 (licking); otherwise 0.

ii.
```python
lf=lick[np.isfinite(lick)] if lick.size else lick
if lf.size:
    lt=np.interp(lf,np.arange(nfr),tsec)-t0; lb=(lt/BIN_S).astype(int); lb=lb[(lb>=0)&(lb<nb)]; L[np.unique(lb)]=1
```

iii. The AI converts lick frames to times and bins them into the 1-second bins.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is binned into the same 1-second bins as neural data, ensuring alignment through the shared temporal grid.

ii.
```python
L=np.zeros(nb,np.int16)
lt=np.interp(lf,np.arange(nfr),tsec)-t0; lb=(lt/BIN_S).astype(int)
```

iii. Same binning as neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position of the mouse at each imaging frame, and `Texture_Length`, the length of the textured corridor.

ii.
```python
pos=np.asarray(b['ft_Pos'][:nfr])
pcat=np.clip((P/float(b['Texture_Length'])*4).astype(int),0,3).astype(np.int16)
```

iii. The AI used `ft_Pos` for position and normalized by `Texture_Length`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position values within each 1-second bin are averaged. The average position is normalized by `Texture_Length` and multiplied by 4 to get 4 equal bins, then clipped to [0, 3].

ii.
```python
P=np.empty(nb,np.float32)
for j in range(nb):
    jj=ix[bins==j]
    if jj.size: P[j]=np.nanmean(pos[jj])
    else: P[j]=P[j-1] if j else 0
pcat=np.clip((P/float(b['Texture_Length'])*4).astype(int),0,3).astype(np.int16)
```

iii. The AI averages position within each temporal bin before discretizing.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is normalized by `Texture_Length` (which is 40 for most sessions, representing decimeters), multiplied by 4, and integer-truncated. This gives 4 equal bins mapping to 0-1m, 1-2m, 2-3m, 3-4m.

ii.
```python
pcat=np.clip((P/float(b['Texture_Length'])*4).astype(int),0,3).astype(np.int16)
```

iii. The AI normalizes by `Texture_Length` to handle the position units.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is averaged within the same 1-second bins as neural data.

ii.
```python
P[j]=np.nanmean(pos[jj])
```

iii. Same temporal binning as neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed=np.asarray(b['ft_RunSpeed'][:nfr])
V=np.empty(nb,np.float32)
```

iii. The AI used `ft_RunSpeed` directly.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed values within each 1-second bin are averaged. The averaged speeds are then discretized into quartiles using global speed thresholds computed across all sessions' valid corridor frames.

ii.
```python
# Global speed quartiles
speeds=[]
for s in sessions:
    ...
    speeds.append(v[ok].astype(np.float32))
q=np.quantile(np.concatenate(speeds),[.25,.5,.75]).astype(np.float32)
...
V[j]=np.nanmean(speed[jj])
vcat=np.digitize(V,q,right=False).astype(np.int16)
```

iii. The AI computed global quartile thresholds from all corridor frames across all sessions to discretize speed.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is discretized into 4 bins using global quartile thresholds. `np.digitize` with `right=False` assigns each averaged speed to one of 4 bins (0-3).

ii.
```python
q=np.quantile(np.concatenate(speeds),[.25,.5,.75]).astype(np.float32)
vcat=np.digitize(V,q,right=False).astype(np.int16)
```

iii. The AI used threshold-based quartiles rather than rank-based quartiles.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is averaged within the same 1-second bins as neural data.

ii.
```python
V[j]=np.nanmean(speed[jj])
```

iii. Same temporal binning as neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: frames beyond the minimum of neural and behavioral data are excluded (`nfr=min(spk.shape[1],len(b['ft_trInd']))`). Non-finite position and speed values are filtered with `np.isfinite`. Empty temporal bins are forward-filled from the previous bin. Trials with `SoundFr` that is non-finite produce NaN for time-to-cue.

ii.
```python
nfr=min(spk.shape[1],len(b['ft_trInd']))
if jj.size: N[:,j]=spk[:,jj].mean(1,dtype=np.float32)
else: N[:,j]=N[:,j-1] if j else 0
cue_rel=... if tr<len(sound) and np.isfinite(sound[tr]) else np.nan
```

iii. The AI implemented forward-filling for empty bins and NaN handling for missing cue information.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural data files, which total hundreds of GB. Each file must be fully loaded with `allow_pickle=True` since they are object-dtype arrays.

ii.
```python
raw=np.load(spkfiles[s],allow_pickle=True).item()['spks']; spk=np.concatenate(raw,axis=0)
```

iii. I/O dominates the runtime, loading ~400 GB of neural data files.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner loop over temporal bins for computing bin-averaged neural activity, position, and speed is done with a Python for-loop rather than vectorized operations. This iterates over each bin `j` and indexes frames belonging to that bin.

ii.
```python
for j in range(nb):
    jj=ix[bins==j]
    if jj.size: N[:,j]=spk[:,jj].mean(1,dtype=np.float32); P[j]=np.nanmean(pos[jj]); V[j]=np.nanmean(speed[jj])
    else: N[:,j]=N[:,j-1] if j else 0; P[j]=P[j-1] if j else 0; V[j]=V[j-1] if j else 0
```

iii. This loop could be replaced with `np.add.at` or similar vectorized binning operations.

## 12-c. What processing does the code repeat multiple times?

i. The global speed quartile computation iterates over all sessions and loads behavioral data, which is similar to the main conversion loop. The reward stimulus inference also iterates over all sessions separately.

ii.
```python
# Speed quartiles - first pass
for s in sessions:
    b=beh[s]; ...
    speeds.append(v[ok].astype(np.float32))
# Reward inference - second pass
for s in sessions:
    b=beh[s]; ...
# Main conversion - third pass
for si,s in enumerate(sessions):
    ...
```

iii. The behavioral data is accessed in three separate loops over sessions.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores `experiment_group`, `rewarded_stimulus`, and other metadata in `session_info` that are not used by the decoder. The reward stimulus inference is an elaborate computation that could have been replaced by simply reading the `isRew` field.

ii.
```python
session_info.append({'session_id':s,'experiment_group':source_group[s],'date':rowmap[s]['datexp'],'n_source_trials':ntr,'trial_indices':kept,'rewarded_stimulus':reward_stim[s]})
```

iii. Extra metadata is computed but not used in downstream decoder training.
