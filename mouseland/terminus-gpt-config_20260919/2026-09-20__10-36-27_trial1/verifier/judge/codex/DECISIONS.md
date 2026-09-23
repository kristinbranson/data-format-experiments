# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every `Beh_*.npy` file into a single in-memory `records` dictionary, loads `Imaging_Exp_info.npy` into `db`, lists all neural session IDs from `spk/*_neural_data.npy`, and then matches behavior to each neural session by stripping `_swap1`/`_swap2` suffixes.

ii. 
```python
def load_sources():
    records={}; record_files={}
    for f in sorted((ROOT/'beh').glob('Beh_*.npy')):
        o=np.load(f,allow_pickle=True).item()
        for k,v in o.items():
            if isinstance(v,dict) and 'ntrials' in v:
                records[k]=v; record_files[k]=f.name
    info=np.load(ROOT/'beh'/'Imaging_Exp_info.npy',allow_pickle=True).item()
    db={}
    for group,arr in info.items():
        for d in arr:
            sid=f"{d['mname']}_{d['datexp']}_{d['blk']}"
            db.setdefault(sid,[]).append((group,d))
    neural=sorted(re.sub(r'_neural_data$','',f.stem) for f in (ROOT/'spk').glob('*_neural_data.npy'))
    aliases={sid:sorted([k for k in records if base_id(k)==sid],key=lambda k:('_swap' in k,k)) for sid in neural}
```

iii. In `CONVERSION_NOTES.md`, the agent justified this as a way to recover all 89 physical recordings, including sessions whose behavior appears only under swap-suffixed aliases.

## 1-b. How are the data split into subjects?

i. Subjects are derived from the mouse-name prefix of each session ID, and `subject_idx` is built from those prefixes after sorting unique subject names.

ii.
```python
subjects=sorted({s.split('_')[0] for s in sids}); subjmap={x:i for i,x in enumerate(subjects)}
data.update(subjects=subjects,subject_idx=np.asarray([subjmap[s.split('_')[0]] for s in sids],dtype=np.int32),
```

iii. The notes treat one session ID prefix as one mouse and report 19 unique mice after this split.

## 1-c. How are the data split into sessions?

i. Each neural file stem is treated as one physical session. Behavior aliases for the same recording are merged onto that session instead of creating additional sessions.

ii.
```python
neural=sorted(re.sub(r'_neural_data$','',f.stem) for f in (ROOT/'spk').glob('*_neural_data.npy'))
aliases={sid:sorted([k for k in records if base_id(k)==sid],key=lambda k:('_swap' in k,k)) for sid in neural}
```

iii. In the notes, the agent explicitly says there should be “one target session per 89 neural recordings” so swap records do not duplicate neurons or trials.

## 1-d. How are the data split into trials?

i. Trials are reconstructed per session by taking frame indices where `ft_trInd == t` and the retained-frame mask is true. The retained-frame mask requires both `ft_CorrSpc` and `ft_move > 0`, so each converted trial contains only running corridor frames from that trial.

ii.
```python
def retained_mask(b,nfr=None):
    n=len(b['ft_trInd']) if nfr is None else min(nfr,len(b['ft_trInd']))
    return np.asarray(b['ft_CorrSpc'][:n],bool)&(np.asarray(b['ft_move'][:n])>0)
```
```python
for t in range(int(b['ntrials'])):
    idx=np.flatnonzero(mask&(tri==t))
    if len(idx): trial_indices.append(idx); trial_ids.append(t)
```

iii. The notes justify this by citing the paper/reference use of running-only corridor frames and by treating the decoder trials as temporal versions of those retained samples.

## 1-e. How are trials filtered based on quality controls?

i. The code drops trials only if they have no retained frames after masking. It does not apply any explicit long-trial or outlier-length exclusion.

ii.
```python
for t in range(int(b['ntrials'])):
    idx=np.flatnonzero(mask&(tri==t))
    if len(idx): trial_indices.append(idx); trial_ids.append(t)
```

iii. The notes say every investigated trial had at least one valid frame and that any empty trial would simply be skipped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from the `spks` object in each session’s `*_neural_data.npy`. Brain-region labels come from `iarea` in the matching retinotopy file.

ii.
```python
def load_neural(sid):
    o=np.load(ROOT/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
    return np.concatenate([np.asarray(x,dtype=np.float32) for x in o],axis=0)
```
```python
with np.load(ROOT/'retinotopy'/f'{datebase}_trans.npz') as z: a=z['iarea'].astype(int)
```

iii. The notes describe the session neural file as three imaging planes that should be concatenated, with retinotopy labels matched neuron-for-neuron.

## 2-b. How is the `neural` data processed?

i. The three planes are concatenated along neurons, truncated to the shared neural/behavior frame count, then one contiguous matrix of all retained trial frames is built. Per-trial neural arrays are views into that selected matrix and remain `float32`.

ii.
```python
neural=load_neural(sid); nfr=min(neural.shape[1],len(b['ft_trInd']))
neural=neural[:,:nfr]
```
```python
ordered=np.concatenate(trial_indices) if trial_indices else np.empty(0,dtype=int)
selected=np.ascontiguousarray(neural[:,ordered],dtype=np.float32)
```
```python
T=len(idx); ntrial=selected[:,offset:offset+T]
```

iii. The notes justify this as preserving the provided deconvolved traces while reducing conversion-time allocation overhead.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code does not drop any neurons. It maps every neuron into one of six categories: four grouped visual areas plus `other visual area` and `unassigned`.

ii.
```python
REGIONS=['V1','medial higher visual','lateral higher visual','anterior higher visual','other visual area','unassigned']
```
```python
out=np.full(len(a),5,dtype=np.int16)
out[a==8]=0; out[np.isin(a,[0,1,2,9])]=1; out[np.isin(a,[5,6])]=2
out[np.isin(a,[3,4])]=3; out[a==7]=4; out[a==-1]=5
return out
```

iii. In the notes, the agent argues that the reference loader keeps all Suite2p-classified cells and that selectivity/area masks are figure-specific rather than global quality-control filters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to the first retained running corridor frame of each trial. The arrays start at that point and continue for the remaining retained frames of that trial, with variable trial length and no padding.

ii.
```python
for t,idx in zip(trial_ids,trial_indices):
    T=len(idx); ntrial=selected[:,offset:offset+T]; offset+=T
```
```python
'temporal_alignment_event':'entry into 4-m visual corridor (first retained running corridor frame)',
'off_start':0.0,'off_end':None,
```

iii. The notes explicitly describe trial time zero as the “first retained corridor-running frame.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code keeps one sample per acquired imaging frame. It estimates a per-session `dt` from the median positive `diff(ft)` and reports a global nominal bin size `DT_GLOBAL * 1000` ms in metadata. No temporal rebinning is applied.

ii.
```python
DT_GLOBAL=0.31480416655540466
```
```python
ft=np.asarray(b['ft'][:nfr],float); d=np.diff(ft)*86400; d=d[np.isfinite(d)&(d>0)]
dt=float(np.median(d)) if len(d) else DT_GLOBAL
```
```python
'time_bin_size':DT_GLOBAL*1000,
```

iii. The notes justify this as preserving the native synchronized imaging clock shared by neural and behavioral streams.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and the retained frame indices of the trial, together with the estimated session frame duration `dt`.

ii.
```python
rew=np.asarray(b['isRew']).astype(int); sound=np.asarray(b['SoundFr'],float)
```
```python
cue=(sound[t]-idx)*dt
```

iii. The notes describe the cue input as coming from `SoundFr` converted to seconds on the imaging-frame clock.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame in a trial, the code subtracts the retained frame index from the trial’s fractional `SoundFr` and multiplies by the session median `dt`, yielding signed seconds until the cue.

ii.
```python
cue=(sound[t]-idx)*dt
inp=np.vstack([cue,np.full(T,day),since,np.full(T,rew[t])]).astype(np.float32)
```

iii. The notes justify the sign convention as positive before the cue and negative after it.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same retained frame indices `idx` that define each trial’s neural slice, so the cue-time vector has the same timepoints as the neural array.

ii.
```python
T=len(idx); ntrial=selected[:,offset:offset+T]
cue=(sound[t]-idx)*dt
```

iii. The trajectory repeatedly states that all converted variables are built on the same retained frame grid as the neural slices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the experiment-database entries in `Imaging_Exp_info.npy`, specifically the per-entry `sess#` field when present, with experiment-group names used as fallback context.

ii.
```python
for group,d in db.get(sid,[]):
    groups.append(group); v=d.get('sess#')
    if v is not None:
        try: vals.append(float(v))
        except Exception: pass
```

iii. In the notes, the agent calls `sess#` the only source-provided training-session/day-like field and says missing values are imputed from condition/date context.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The code takes the maximum `sess#` value found across database entries for the session. If none exists, it imputes 0 for “before”, 2 for `train2_after`, and 1 otherwise, then broadcasts that scalar over all time bins in the trial.

ii.
```python
if vals: return float(max(vals)),False,groups
text=' '.join(groups)
if 'before' in text: return 0.0,True,groups
if 'train2_after' in text: return 2.0,True,groups
return 1.0,True,groups
```
```python
inp=np.vstack([cue,np.full(T,day),since,np.full(T,rew[t])]).astype(np.float32)
```

iii. The notes justify this as preserving the source-provided training-stage annotation, while recording when imputation was needed.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the retained frame indices of the trial, not from `StartFr`. The start point is the first retained frame index `idx[0]`.

ii.
```python
since=(idx-idx[0])*dt
```

iii. The notes define trial time zero as the first retained corridor-running frame after masking.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The code subtracts the first retained frame index of the trial from every retained frame index, then multiplies by the session median `dt` to get elapsed seconds from that retained start.

ii.
```python
since=(idx-idx[0])*dt
inp=np.vstack([cue,np.full(T,day),since,np.full(T,rew[t])]).astype(np.float32)
```

iii. The notes justify this as making trial start coincide with the retained running corridor segment rather than the full raw-trial start.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is built from the same `idx` array used to slice the neural data, so it has identical timepoints to the per-trial neural matrix.

ii.
```python
T=len(idx); ntrial=selected[:,offset:offset+T]
since=(idx-idx[0])*dt
```

iii. The agent’s notes describe all per-trial arrays as living on one retained frame grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the trial-level `isRew` variable.

ii.
```python
rew=np.asarray(b['isRew']).astype(int)
```

iii. No extra justification is given beyond using the provided rewarded/unrewarded trial flag.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The code simply converts `isRew` to integer and repeats the per-trial value over all retained time bins of that trial.

ii.
```python
inp=np.vstack([cue,np.full(T,day),since,np.full(T,rew[t])]).astype(np.float32)
```

iii. The notes treat this as a direct readout of reward availability, with no additional transformation.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, after reconciling swap aliases by checking that all alias records have identical `WallName` arrays.

ii.
```python
def merged_stimuli(alias_names,records):
    out=np.asarray(records[alias_names[0]]['WallName']).astype(str).copy()
    for k in alias_names[1:]:
        other=np.asarray(records[k]['WallName']).astype(str)
        if not np.array_equal(out,other):
            raise ValueError(f'Behavior aliases disagree on WallName: {alias_names}')
    return out
```

iii. The notes say an earlier attempt using `TrialStim` was wrong and that `WallName` is the authoritative physical visual identity.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The code uses the sorted set of all observed `WallName` strings as the category list, builds a string-to-index map, and repeats each trial’s category index across all retained bins of that trial. It keeps the 15 physical wall labels rather than collapsing them to 4 broad texture families.

ii.
```python
speed_q,stim_values,scan_stats=scan(records,db,sids,aliases); stim_to_idx={x:i for i,x in enumerate(stim_values)}
```
```python
stim=np.full(T,stim_to_idx[str(stimuli[t])],dtype=np.int16)
```
```python
output_values=[stim_values,['not licking','licking'],['0-1 m','1-2 m','2-3 m','3-4 m'],['Q1','Q2','Q3','Q4']]
```

iii. The notes justify this by preserving the physical `WallName` labels, including rock/wood and swap variants, instead of paper-analysis-normalized stimulus names.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` together with `LickTrind`, so each lick can be placed into its supplied trial before assignment to retained frames.

ii.
```python
lick_by_trial={t:[] for t in range(int(b['ntrials']))}
lf=np.asarray(b['LickFr'],float); lt=np.asarray(b['LickTrind'],float)
for f,t in zip(lf,lt):
    if np.isfinite(f) and np.isfinite(t): lick_by_trial.setdefault(int(t),[]).append(float(f))
```

iii. The notes justify using both fields because `LickFr` is fractional and trial boundaries can otherwise create assignment errors.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the code starts with an all-zero vector, then assigns a 1 at the nearest retained frame to each lick in that trial, but only if the lick is within 0.5 frame of a retained sample.

ii.
```python
lick=np.zeros(T,dtype=np.int16)
for f in lick_by_trial.get(t,[]):
    j=int(np.argmin(np.abs(idx-f)))
    if abs(float(idx[j])-f)<=0.5001: lick[j]=1
```

iii. The notes say this was tightened specifically to avoid relocating licks from excluded gray/stationary periods onto kept frames.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are aligned to the same retained frame grid as the neural data by choosing the nearest retained frame within the same trial, so the lick vector length matches the neural trial length.

ii.
```python
T=len(idx); ntrial=selected[:,offset:offset+T]
lick=np.zeros(T,dtype=np.int16)
for f in lick_by_trial.get(t,[]):
    j=int(np.argmin(np.abs(idx-f)))
```

iii. The notes explicitly describe this as a retained-frame alignment decision made after excluding gray/stationary frames.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the frame-wise `ft_Pos` variable.

ii.
```python
pos=np.asarray(b['ft_Pos'][:nfr],np.float32)
```

iii. The notes interpret the source units as decimeters over the 4 m corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code converts source units to 1 m bins by dividing position by 10, casting to integer, and clipping into the range 0 to 3.

ii.
```python
position=np.clip((pos[idx]/10).astype(np.int16),0,3)
```

iii. The notes justify this by mapping 0–40 source units onto the requested four 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded into four equal-length spatial categories: `[0,1)`, `[1,2)`, `[2,3)`, and `[3,4]` meters, encoded as integer classes 0–3.

ii.
```python
position=np.clip((pos[idx]/10).astype(np.int16),0,3)
```
```python
output_values=[stim_values,['not licking','licking'],['0-1 m','1-2 m','2-3 m','3-4 m'],['Q1','Q2','Q3','Q4']]
```

iii. The notes state that source units are decimeters and that four requested 1 m bins should be used.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled on the same retained frame indices `idx` used for each neural trial slice, so it is aligned bin-for-bin with the neural data.

ii.
```python
T=len(idx); ntrial=selected[:,offset:offset+T]
position=np.clip((pos[idx]/10).astype(np.int16),0,3)
```

iii. The trajectory and notes both describe all time-varying outputs as living on the retained neural frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the frame-wise `ft_RunSpeed` variable.

ii.
```python
speed=np.asarray(b['ft_RunSpeed'][:nfr],np.float32)
```

iii. The notes treat this as the direct source stream for the speed output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The code first gathers all running-speed values from retained frames across the selected sessions, computes global 25th/50th/75th percentile thresholds, and then uses `np.digitize` to assign each retained frame to a quartile bin.

ii.
```python
speed=np.concatenate(speeds); qs=np.quantile(speed,[.25,.5,.75]).astype(np.float32)
```
```python
speedbin=np.digitize(speed[idx],speed_q,right=False).astype(np.int16)
```

iii. The notes justify this as producing globally balanced quartile classes over the exact retained-frame population.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the global 25th, 50th, and 75th percentiles of retained-frame `ft_RunSpeed`, stored in metadata as `speed_quartile_thresholds`; categories are `Q1` through `Q4`.

ii.
```python
'position_source_units_per_meter':10.0,'speed_quartile_thresholds':speed_q.tolist(),
```
```python
output_values=[stim_values,['not licking','licking'],['0-1 m','1-2 m','2-3 m','3-4 m'],['Q1','Q2','Q3','Q4']]
```

iii. The notes explicitly report these thresholds as a global pre-scan product.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled at the same retained frame indices `idx` used to define the neural trial arrays, so it is aligned timepoint-by-timepoint with the neural data.

ii.
```python
T=len(idx); ntrial=selected[:,offset:offset+T]
speedbin=np.digitize(speed[idx],speed_q,right=False).astype(np.int16)
```

iii. The notes consistently describe outputs as being generated on the same retained frame grid as neural activity.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code truncates all streams to the shorter of neural frames and behavior frames, remaps nonfinite `ft_trInd` values to `-1`, raises an error if a neural session has no behavior alias, imputes missing `sess#` values from experiment-group labels, and drops licks that are not within 0.5 frame of any retained frame in the same trial.

ii.
```python
neural=load_neural(sid); nfr=min(neural.shape[1],len(b['ft_trInd']))
neural=neural[:,:nfr]
tri_raw=np.asarray(b['ft_trInd'][:nfr],float); tri=np.where(np.isfinite(tri_raw),tri_raw,-1).astype(int)
```
```python
if any(not x for x in aliases.values()): raise RuntimeError('Missing behavior: '+str([k for k,v in aliases.items() if not v]))
```
```python
if 'before' in text: return 0.0,True,groups
if 'train2_after' in text: return 2.0,True,groups
return 1.0,True,groups
```

iii. The notes explicitly mention fixing NaN trial IDs, reconciling swap aliases, tightening lick assignment, and recording training-day imputation in metadata.

## 12-a. What are the most time-consuming steps of the code?

i. The main costs are loading the large object-NPY neural sessions, selecting/copying retained frames into contiguous arrays, and serializing the very large output pickle.

ii.
```python
o=np.load(ROOT/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
return np.concatenate([np.asarray(x,dtype=np.float32) for x in o],axis=0)
```
```python
selected=np.ascontiguousarray(neural[:,ordered],dtype=np.float32)
```
```python
with open(out,'wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes say full conversion time was dominated by large object-NPY reads and 161.6 GiB serialization.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still loops explicitly over trials during the pre-scan, over trials again during session conversion, and over lick events inside each trial. Those are the obvious remaining vectorization opportunities.

ii.
```python
for t in range(int(b['ntrials'])):
    z=int(np.sum(m&(tri==t))); stats['frames']+=z
```
```python
for t in range(int(b['ntrials'])):
    idx=np.flatnonzero(mask&(tri==t))
```
```python
for f in lick_by_trial.get(t,[]):
    j=int(np.argmin(np.abs(idx-f)))
```

iii. The trajectory says the agent already optimized away a worse per-trial neural-copy pattern, but these Python loops remained.

## 12-c. What processing does the code repeat multiple times?

i. It performs a full behavior pre-scan to compute global speed quartiles, stimulus values, and trial counts, then recomputes retained masks and trial indices during actual session conversion. It also calls `choose_behavior` and `merged_stimuli` in both passes.

ii.
```python
speed_q,stim_values,scan_stats=scan(records,db,sids,aliases)
```
```python
m=retained_mask(b); tri=np.where(np.isfinite(np.asarray(b['ft_trInd'][:len(m)],float)),np.asarray(b['ft_trInd'][:len(m)],float),-1).astype(int)
```
```python
tri_raw=np.asarray(b['ft_trInd'][:nfr],float); tri=np.where(np.isfinite(tri_raw),tri_raw,-1).astype(int); mask=retained_mask(b,nfr)
```

iii. The notes describe this repeated pre-scan as an intentional tradeoff to get global quartile thresholds and scan statistics before conversion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script keeps some bookkeeping and optional-analysis work that the downstream decoder does not use: `record_files` is built and returned but never consumed, the converter stores detailed `session_info` and `scan_statistics` metadata, and it contains plotting code plus a `plot` parameter that is irrelevant to normal conversion.

ii.
```python
records={}; record_files={}
...
return records,record_files,db,neural,aliases
```
```python
def convert_session(sid,records,db,aliases,stim_to_idx,speed_q,plot=False):
```
```python
'session_info':sinfo,'scan_statistics':scan_stats
```
```python
def make_plot(sid,neural,inp,out):
    ...
```

iii. The notes emphasize extensive documentation, plots, and metadata for validation, even though the decoder itself only needs the converted arrays and label metadata.
