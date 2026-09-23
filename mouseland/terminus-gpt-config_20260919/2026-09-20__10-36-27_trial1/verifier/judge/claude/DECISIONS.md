# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three subdirectories: `beh/` (behavior), `spk/` (neural), and `retinotopy/` (area labels). It first reads `Imaging_Exp_info.npy` as a master index, then iterates over all `Beh_*.npy` files to find session records. Neural files are loaded per session from `spk/<sid>_neural_data.npy`, and area labels from `retinotopy/<datebase>_trans.npz`. The AI discovers all sessions by cross-referencing neural file names with behavior keys, handling swap aliases by stripping `_swap1`/`_swap2` suffixes.

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
    ...
    neural=sorted(re.sub(r'_neural_data$','',f.stem) for f in (ROOT/'spk').glob('*_neural_data.npy'))
    aliases={sid:sorted([k for k in records if base_id(k)==sid],...) for sid in neural}
```

iii. The AI documented in CONVERSION_NOTES.md Step 2 that all 89 neural files have matching retinotopy and behavior when swap aliases are recognized. The approach loads all behavior files first, then matches them to neural sessions.

## 1-b. How are the data split into subjects?

i. The mouse name is extracted from the session ID by splitting on `_` and taking the first element. Unique sorted mouse names form the subjects list, and `subject_idx` maps each session to its subject.

ii.
```python
subjects=sorted({s.split('_')[0] for s in sids})
subjmap={x:i for i,x in enumerate(subjects)}
data.update(subjects=subjects,subject_idx=np.asarray([subjmap[s.split('_')[0]] for s in sids],dtype=np.int32))
```

iii. The session ID format is `<mouse>_<YYYY>_<MM>_<DD>_<block>`, so the first `_`-separated token is the mouse name. 19 subjects are identified, matching the paper.

## 1-c. How are the data split into sessions?

i. A session is identified by the neural file name, which encodes mouse, date, and block. The AI discovers all 89 neural files and treats each as a session. Swap behavior aliases (same physical recording with different stimulus annotations) are merged into a single session.

ii.
```python
neural=sorted(re.sub(r'_neural_data$','',f.stem) for f in (ROOT/'spk').glob('*_neural_data.npy'))
aliases={sid:sorted([k for k in records if base_id(k)==sid],key=lambda k:('_swap' in k,k)) for sid in neural}
```

iii. The AI noted that 13 neural sessions initially lacked direct behavior matches, but all 89 were matched when swap aliases were recognized. This is documented in CONVERSION_NOTES.md Step 4.

## 1-d. How are the data split into trials?

i. Trials are identified by `ft_trInd` (trial index per frame). For each trial, the AI retains only frames that satisfy both `ft_CorrSpc` (inside corridor) AND `ft_move > 0` (animal is moving). This differs from the reference, which uses only `ft_CorrSpc`.

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

iii. The AI justified the `ft_move > 0` filter by citing the paper: "The paper states that only running timepoints were analyzed, removing pauses for reward collection." and "Reference code implements this with `ft_move > 0` before spatial interpolation."

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials that have zero retained frames (after applying the corridor + running mask). No trial length-based filtering is applied. The reference solution additionally drops trials longer than the 99th percentile.

ii.
```python
for t in range(int(b['ntrials'])):
    idx=np.flatnonzero(mask&(tri==t))
    if len(idx): trial_indices.append(idx); trial_ids.append(t)
```

iii. The AI's CONVERSION_NOTES.md does not mention length-based trial filtering. The `ft_move > 0` mask may partially address the issue of very long stationary trials, since stationary frames are excluded, but it does not remove them entirely.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which contains a list of three neuron-by-frame matrices (one per imaging plane), concatenated along the neuron axis. Area labels come from `iarea` in `retinotopy/<datebase>_trans.npz`.

ii.
```python
def load_neural(sid):
    o=np.load(ROOT/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
    return np.concatenate([np.asarray(x,dtype=np.float32) for x in o],axis=0)
```

iii. The AI documented this in CONVERSION_NOTES.md Step 1, noting the reference `load_spk` function does the same concatenation.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied. The deconvolved traces are taken as-is and stored as float32. The reference uses float16.

ii.
```python
neural=load_neural(sid)  # float32
...
ntrial=selected[:,offset:offset+T]  # slice for each trial
ns.append(ntrial)
```

iii. The AI noted in CONVERSION_NOTES.md Step 3: "Paper analyses use nonnegative deconvolved fluorescence traces generated by Suite2p; no additional dF/F operation should be applied."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps ALL neurons. It maps area labels to 6 regions: V1, medial HV, lateral HV, anterior HV, other visual area, and unassigned. No neurons are dropped. The reference keeps only neurons in the 4 main visual areas (V1, mHV, lHV, aHV), dropping ~585k neurons.

ii.
```python
REGIONS=['V1','medial higher visual','lateral higher visual','anterior higher visual','other visual area','unassigned']

def area_indices(sid):
    ...
    out=np.full(len(a),5,dtype=np.int16)
    out[a==8]=0; out[np.isin(a,[0,1,2,9])]=1; out[np.isin(a,[5,6])]=2
    out[np.isin(a,[3,4])]=3; out[a==7]=4; out[a==-1]=5
    return out
```

iii. The AI justified this in CONVERSION_NOTES.md Step 4: "Keep all neurons. Selectivity masks are figure-specific, not quality curation." and "reference `load_spk` concatenates all planes and the paper's reported population range includes all Suite2p-classified cells."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry. The first retained running-corridor frame of each trial is the start. Trials are variable length. No padding or fixed window is applied.

ii.
```python
for t in range(int(b['ntrials'])):
    idx=np.flatnonzero(mask&(tri==t))
    if len(idx): trial_indices.append(idx); trial_ids.append(t)
...
ntrial=selected[:,offset:offset+T]
```

iii. The AI set `temporal_alignment_event` to "entry into 4-m visual corridor (first retained running corridor frame)" and `off_start` to 0.0, `off_end` to None.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native imaging frame rate (~3.17 Hz, ~314.8 ms per frame) is preserved. The AI computes a per-session dt from `median(diff(ft)*86400)` and reports a global constant `DT_GLOBAL=0.31480416655540466` s.

ii.
```python
DT_GLOBAL=0.31480416655540466
...
ft=np.asarray(b['ft'][:nfr],float); d=np.diff(ft)*86400; d=d[np.isfinite(d)&(d>0)]
dt=float(np.median(d)) if len(d) else DT_GLOBAL
```
```python
'time_bin_size':DT_GLOBAL*1000
```

iii. The AI noted: "Preserve one sample per acquired volume/frame; report 314.804 ms nominal bin size. No temporal resampling is needed because all streams already share this clock."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the sound cue frame for each trial) and the retained frame indices.

ii.
```python
sound=np.asarray(b['SoundFr'],float)
...
cue=(sound[t]-idx)*dt
```

iii. The AI uses frame-index differences multiplied by a fixed dt, rather than interpolating onto actual frame timestamps.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue time is computed as `(SoundFr[trial] - frame_index) * dt`, giving seconds. The sign convention is positive before the cue, negative after (consistent with "time TO sound cue"). The reference instead interpolates SoundFr onto actual frame timestamps from `ft`.

ii.
```python
cue=(sound[t]-idx)*dt
```

iii. The AI uses a per-session median dt rather than actual frame timestamps, which introduces small timing errors due to frame-to-frame jitter.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same retained frame indices used for the neural data of each trial.

ii.
```python
idx=np.flatnonzero(mask&(tri==t))
...
cue=(sound[t]-idx)*dt
```

iii. Alignment is inherent since all data streams use the same frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `sess#` in the experiment database (`Imaging_Exp_info.npy`), with fallback imputation from experiment group names when `sess#` is missing.

ii.
```python
def session_day(sid,db):
    vals=[]; groups=[]
    for group,d in db.get(sid,[]):
        groups.append(group); v=d.get('sess#')
        if v is not None:
            try: vals.append(float(v))
            except Exception: pass
    if vals: return float(max(vals)),False,groups
    text=' '.join(groups)
    if 'before' in text: return 0.0,True,groups
    if 'train2_after' in text: return 2.0,True,groups
    return 1.0,True,groups
```

iii. The AI justified this in CONVERSION_NOTES.md Step 4: "Use numeric `sess#` as source-provided training-session/day index."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The `sess#` value is taken directly (max if multiple entries exist). When missing, it is imputed: 0 for "before" sessions, 2 for "train2_after", 1 for others. Values range from 0 to 12. The reference instead counts sequential recording days per mouse (0,1,2,...,max 7).

ii.
```python
if vals: return float(max(vals)),False,groups
text=' '.join(groups)
if 'before' in text: return 0.0,True,groups
if 'train2_after' in text: return 2.0,True,groups
return 1.0,True,groups
```

iii. The AI noted the `sess#` field is the only numeric training-session field in the data. However, `sess#` values are non-sequential and represent something different from the count of recording days.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the retained frame indices of each trial, with time computed as `(frame_index - first_frame_index) * dt`.

ii.
```python
since=(idx-idx[0])*dt
```

iii. The first retained running-corridor frame serves as trial start (time = 0).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Time since trial start is `(idx - idx[0]) * dt`, where idx[0] is the first retained frame and dt is the per-session median frame interval. The reference instead uses actual frame timestamps and interpolates `StartFr` onto the time axis.

ii.
```python
since=(idx-idx[0])*dt
```

iii. This starts at exactly 0 for each trial and increases monotonically. The use of fixed dt introduces small timing approximations.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed from the same retained frame indices used for neural data.

ii.
```python
idx=np.flatnonzero(mask&(tri==t))
since=(idx-idx[0])*dt
```

iii. Alignment is inherent since all streams use the same frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial is in the rewarded corridor.

ii.
```python
rew=np.asarray(b['isRew']).astype(int)
...
np.full(T,rew[t])
```

iii. Directly from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to int (0 or 1) and broadcast across all frames of the trial.

ii.
```python
rew=np.asarray(b['isRew']).astype(int)
...
inp=np.vstack([cue,np.full(T,day),since,np.full(T,rew[t])]).astype(np.float32)
```

iii. No additional processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the wall texture for each trial.

ii.
```python
def merged_stimuli(alias_names,records):
    out=np.asarray(records[alias_names[0]]['WallName']).astype(str).copy()
    ...
    return out
```

iii. The AI initially tried `TrialStim` but switched to `WallName` after finding that `TrialStim` contained placeholders in some sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses the raw `WallName` values (15 distinct names like circle1, circle2, leaf1, leaf1_swap1, etc.) as separate categories, giving 15 output classes. The reference maps these to 4 base texture categories (circle, leaf, rock, wood).

ii.
```python
stim_to_idx={x:i for i,x in enumerate(stim_values)}
...
stim=np.full(T,stim_to_idx[str(stimuli[t])],dtype=np.int16)
```

iii. The AI justified using raw wall names. However, the instructions say "Visual stimulus category. e.g. circle, leaf, etc." which strongly implies 4 categories, not 15.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (lick frame numbers) and `LickTrind` (lick trial indices).

ii.
```python
lick_by_trial={t:[] for t in range(int(b['ntrials']))}
lf=np.asarray(b['LickFr'],float); lt=np.asarray(b['LickTrind'],float)
for f,t in zip(lf,lt):
    if np.isfinite(f) and np.isfinite(t): lick_by_trial.setdefault(int(t),[]).append(float(f))
```

iii. Both `LickFr` and `LickTrind` are used to correctly assign licks to trials.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each lick in the trial, the AI finds the nearest retained frame within 0.5 frame distance and sets that frame to 1. The reference simply truncates `LickFr` to int and directly indexes the licking array.

ii.
```python
lick=np.zeros(T,dtype=np.int16)
for f in lick_by_trial.get(t,[]):
    j=int(np.argmin(np.abs(idx-f)))
    if abs(float(idx[j])-f)<=0.5001: lick[j]=1
```

iii. The AI tightened lick assignment to prevent licks from excluded periods being relocated. This is a more careful approach but involves a loop over individual licks.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are assigned to the nearest retained frame of their trial, ensuring alignment with the neural data.

ii.
```python
j=int(np.argmin(np.abs(idx-f)))
if abs(float(idx[j])-f)<=0.5001: lick[j]=1
```

iii. Because the AI excludes non-moving frames (`ft_move > 0`), lick frames that fall during stationary periods are excluded by the 0.5-frame threshold.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in decimeters at each imaging frame.

ii.
```python
pos=np.asarray(b['ft_Pos'][:nfr],np.float32)
...
position=np.clip((pos[idx]/10).astype(np.int16),0,3)
```

iii. Directly from the frame-level position variable.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by 10 to convert to meters, then truncated to int and clipped to [0, 3], giving four 1-meter bins.

ii.
```python
position=np.clip((pos[idx]/10).astype(np.int16),0,3)
```

iii. Same approach as the reference.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal-length 1-meter bins: [0,1), [1,2), [2,3), [3,4] m. Integer truncation of meters gives the bin index.

ii.
```python
position=np.clip((pos[idx]/10).astype(np.int16),0,3)
```
Output values: `['0-1 m','1-2 m','2-3 m','3-4 m']`

iii. Consistent with the instruction's "4 equal-length, 1-m-long spatial bins".

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed at the same retained frame indices as neural data.

ii.
```python
position=np.clip((pos[idx]/10).astype(np.int16),0,3)
```

iii. Alignment is inherent since all streams use the same frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed=np.asarray(b['ft_RunSpeed'][:nfr],np.float32)
```

iii. Directly from frame-level speed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes GLOBAL quartile thresholds (25th, 50th, 75th percentiles) across ALL retained running-corridor frames from all sessions being converted, using `np.quantile`. Then `np.digitize` assigns each frame to a bin. The reference computes PER-SESSION rank-based quartiles ensuring exactly 25% of frames per bin per session.

ii.
```python
# In scan():
speed=np.concatenate(speeds); qs=np.quantile(speed,[.25,.5,.75]).astype(np.float32)

# In convert_session():
speedbin=np.digitize(speed[idx],speed_q,right=False).astype(np.int16)
```

iii. The AI documented global thresholds: [12.39, 25.33, 40.83]. Using global rather than per-session quartiles means individual sessions may have unequal bin distributions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins using global quantile thresholds via `np.digitize`. Bins represent the lowest 25%, low 25%, high 25%, and highest 25% of speeds globally.

ii.
```python
speedbin=np.digitize(speed[idx],speed_q,right=False).astype(np.int16)
```
Output values: `['Q1','Q2','Q3','Q4']`

iii. The instruction says "4 bins, each corresponding to 25% of the data" - the AI interprets this globally while the reference interprets it per-session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is indexed at the same retained frame indices as neural data.

ii.
```python
speedbin=np.digitize(speed[idx],speed_q,right=False).astype(np.int16)
```

iii. Alignment is inherent since all streams use the same frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) behavior arrays are truncated to neural frame count, (2) NaN values in `ft_trInd` are mapped to -1 before masking, (3) lick frames that fall outside retained frames are dropped via the 0.5-frame threshold, (4) missing `sess#` values are imputed from experiment group names, (5) trials with zero retained frames are skipped.

ii.
```python
neural=neural[:,:nfr]  # truncate to neural frame count
tri=np.where(np.isfinite(tri_raw),tri_raw,-1).astype(int)  # handle NaN trial IDs
if abs(float(idx[j])-f)<=0.5001: lick[j]=1  # lick distance threshold
```

iii. The AI documented these in CONVERSION_NOTES.md Step 10, noting that NaN-to-integer warnings and lick relocation risks were discovered and fixed during development.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural data files (each session is multi-GB), and serializing the final 161.6 GiB pickle file. Total conversion took ~64 minutes.

ii.
```python
o=np.load(ROOT/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
return np.concatenate([np.asarray(x,dtype=np.float32) for x in o],axis=0)
```

iii. The AI noted that object-NPY files cannot be memory-mapped and that the final pickle was 161.6 GiB due to retaining all neurons.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The lick assignment loop iterates over individual lick events per trial using `argmin`, which could be vectorized. The trial frame discovery also loops per trial.

ii.
```python
for f in lick_by_trial.get(t,[]):
    j=int(np.argmin(np.abs(idx-f)))
    if abs(float(idx[j])-f)<=0.5001: lick[j]=1
```

iii. The AI noted allocation fragmentation as a bottleneck and fixed it with contiguous selection, but the per-lick loop remains.

## 12-c. What processing does the code repeat multiple times?

i. The behavior data is loaded twice: once in the `scan()` function to compute global statistics (speed quartiles, stimulus list), and again during `convert_session()` for each session.

ii.
```python
# In scan():
b=choose_behavior(sid,aliases,records)
# In convert_session():
b=choose_behavior(sid,aliases,records)
```

iii. The scan is a lightweight pre-pass to determine global values before conversion, but it does re-read the same behavior dictionaries.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI retains neurons in "other visual area" and "unassigned" regions that are not part of the paper's analysis areas. These ~585k extra neurons add substantial data volume but may not contribute meaningfully to decoding. The 161.6 GiB output file (vs the reference's ~10 GiB) is largely due to this.

ii.
```python
REGIONS=['V1','medial higher visual','lateral higher visual','anterior higher visual','other visual area','unassigned']
```

iii. The AI justified keeping all neurons to match the paper's reported population range, but the reference code and paper analyses focus on the four main visual areas.
