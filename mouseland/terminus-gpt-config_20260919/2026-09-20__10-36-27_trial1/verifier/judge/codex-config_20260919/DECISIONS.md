# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans every `Beh_*.npy` into a behavior-record dictionary, reads `Imaging_Exp_info.npy` as metadata, enumerates every neural file as the authoritative set of 89 physical recordings, matches `_swap1`/`_swap2` behavior aliases to each base session, and later loads each session's three neural planes and retinotopy file. Behavior is held in memory; neural data is loaded one session at a time.

ii.
```python
for f in sorted((ROOT/'beh').glob('Beh_*.npy')):
    o=np.load(f,allow_pickle=True).item()
    for k,v in o.items():
        if isinstance(v,dict) and 'ntrials' in v:
            records[k]=v; record_files[k]=f.name
info=np.load(ROOT/'beh'/'Imaging_Exp_info.npy',allow_pickle=True).item()
neural=sorted(re.sub(r'_neural_data$','',f.stem) for f in (ROOT/'spk').glob('*_neural_data.npy'))
aliases={sid:sorted([k for k in records if base_id(k)==sid],key=lambda k:('_swap' in k,k)) for sid in neural}
```

iii. The notes say direct IDs initially matched only 76 recordings; recognizing swap suffixes recovered all 89 paper recordings without duplicating the same physical neural data. Loading one neural session at a time controls memory.

## 1-b. How are the data split into subjects?

i. The mouse ID is parsed as the session-ID prefix. Sorted unique IDs form `subjects`, and every session receives the corresponding integer `subject_idx`.

ii.
```python
subjects=sorted({s.split('_')[0] for s in sids}); subjmap={x:i for i,x in enumerate(subjects)}
data.update(subjects=subjects,
  subject_idx=np.asarray([subjmap[s.split('_')[0]] for s in sids],dtype=np.int32),
```

iii. Session names have the form `<mouse>_<date>_<block>`; the notes validate 19 unique mice.

## 1-c. How are the data split into sessions?

i. Each `*_neural_data.npy` base name defines one physical session. Behavior keys differing only by `_swap1` or `_swap2` are aliases of that session, not additional sessions.

ii.
```python
def base_id(k): return re.sub(r'_swap[12]$','',k)
neural=sorted(re.sub(r'_neural_data$','',f.stem) for f in (ROOT/'spk').glob('*_neural_data.npy'))
aliases={sid:sorted([k for k in records if base_id(k)==sid],key=lambda k:('_swap' in k,k)) for sid in neural}
```

iii. The aliases have identical frame streams and represent statistical annotations of the same recording. The AI therefore avoids duplicated neurons and trials and obtains 89 sessions.

## 1-d. How are the data split into trials?

i. Trials are the integer values `0..ntrials-1` in `ft_trInd`. For each trial the AI retains frame indices that are both inside the textured corridor (`ft_CorrSpc`) and running (`ft_move > 0`), after truncating behavior to the neural frame count. This can remove frames from the middle of a trial.

ii.
```python
def retained_mask(b,nfr=None):
    n=len(b['ft_trInd']) if nfr is None else min(nfr,len(b['ft_trInd']))
    return np.asarray(b['ft_CorrSpc'][:n],bool)&(np.asarray(b['ft_move'][:n])>0)

for t in range(int(b['ntrials'])):
    idx=np.flatnonzero(mask&(tri==t))
    if len(idx): trial_indices.append(idx); trial_ids.append(t)
```

iii. The AI interpreted the paper's “only running timepoints were analyzed” rule as a general curation rule and retained temporal samples rather than applying the paper's spatial interpolation, to preserve lick and speed timing.

## 1-e. How are trials filtered based on quality controls?

i. A trial is skipped only if no corridor-and-running frame remains. No long-trial or stopped-animal trial filter is applied; in the full conversion all 38,110 trials retained at least one frame. Stationary frames are instead removed within trials.

ii.
```python
for t in range(int(b['ntrials'])):
    idx=np.flatnonzero(mask&(tri==t))
    if len(idx): trial_indices.append(idx); trial_ids.append(t)
```

iii. The notes state every investigated trial had a valid frame and treat `ft_move > 0` as reference-consistent. They do not justify or implement the human reference's whole-trial 99th-percentile duration exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the `spks` list in each session's `spk/<session>_neural_data.npy`; the three imaging-plane arrays are concatenated along the neuron axis. `iarea` from retinotopy supplies region labels.

ii.
```python
o=np.load(ROOT/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
return np.concatenate([np.asarray(x,dtype=np.float32) for x in o],axis=0)
```

iii. The files already contain Suite2p nonnegative deconvolved activity, so the AI correctly concluded that no dF/F calculation or additional deconvolution was needed.

## 2-b. How is the `neural` data processed?

i. The three planes are concatenated, converted to float32, truncated to the common frame count, subset to all retained trial frames once as a contiguous matrix, then exposed as per-trial views. There is no normalization, temporal resampling, padding, or spatial interpolation.

ii.
```python
neural=load_neural(sid); nfr=min(neural.shape[1],len(b['ft_trInd']))
neural=neural[:,:nfr]
ordered=np.concatenate(trial_indices) if trial_indices else np.empty(0,dtype=int)
selected=np.ascontiguousarray(neural[:,ordered],dtype=np.float32)
ntrial=selected[:,offset:offset+T]
```

iii. The AI chose native temporal frames because the requested cue, licking, and speed variables are time-varying. A single selected matrix plus views was introduced after per-trial advanced indexing caused severe allocation slowdown.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed. Every Suite2p trace is retained, including neurons labeled “other visual area” and “unassigned”; `iarea` is only mapped to six region categories.

ii.
```python
out=np.full(len(a),5,dtype=np.int16)
out[a==8]=0; out[np.isin(a,[0,1,2,9])]=1; out[np.isin(a,[5,6])]=2
out[np.isin(a,[3,4])]=3; out[a==7]=4; out[a==-1]=5
return out
```

iii. The AI reasoned that Suite2p had already performed cell classification and that selectivity masks were analysis-specific. It interpreted the reported 20,547–89,577 traces as support for retaining all 4,691,034 neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial starts at its first retained running frame in the corridor, not necessarily the actual `StartFr`. Only retained running/corridor columns are stored; stopped frames may create elapsed-time gaps even though adjacent stored columns remain adjacent in the matrix.

ii.
```python
idx=np.flatnonzero(mask&(tri==t))
T=len(idx); ntrial=selected[:,offset:offset+T]
since=(idx-idx[0])*dt
```

iii. The AI reconciled corridor-entry alignment with the paper's running-only analysis by defining time zero as the first retained corridor-running frame.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging frames are retained without temporal rebinning. Session event times use that session's median positive `diff(ft)`; metadata reports the global nominal interval 314.804 ms.

ii.
```python
d=np.diff(ft)*86400; d=d[np.isfinite(d)&(d>0)]
dt=float(np.median(d)) if len(d) else DT_GLOBAL
...
'time_bin_size':DT_GLOBAL*1000,
```

iii. Neural and behavior streams already share the imaging clock, so the AI found no need to resample.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr`, each retained frame's original index, and the session frame interval estimated from `ft`.

ii.
```python
sound=np.asarray(b['SoundFr'],float)
ft=np.asarray(b['ft'][:nfr],float); d=np.diff(ft)*86400
cue=(sound[t]-idx)*dt
```

iii. The notes identify `SoundFr` as the sound-cue frame and the common imaging clock as the synchronization basis.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI subtracts each retained integer frame index from the fractional cue frame and multiplies by one session-wide median frame duration. Values are positive before and negative after the cue.

ii.
```python
cue=(sound[t]-idx)*dt
```

iii. This preserves fractional cue timing while producing seconds on the native clock. The AI validated cue zero-crossings against raw data.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Cue time is calculated for exactly the same `idx` frame indices used to select neural columns, so it has one value per stored neural timepoint.

ii.
```python
idx=np.flatnonzero(mask&(tri==t))
ntrial=selected[:,offset:offset+T]
cue=(sound[t]-idx)*dt
```

iii. The AI states that all temporal streams use the shared imaging-frame clock; independent checks found matching shapes and raw indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is primarily taken from `sess#` entries in `Imaging_Exp_info.npy`, using the maximum numeric value among duplicate group records. If absent, experiment-group names provide a stage-based fallback.

ii.
```python
for group,d in db.get(sid,[]):
    groups.append(group); v=d.get('sess#')
    if v is not None:
        try: vals.append(float(v))
        except Exception: pass
if vals: return float(max(vals)),False,groups
```

iii. The AI considered `sess#` the only explicit source-provided training-session/day index and recorded whether a value was imputed.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Numeric `sess#` is copied as a float and broadcast over every retained timepoint. Missing values become 0 for “before,” 2 for `train2_after`, and 1 otherwise.

ii.
```python
if 'before' in text: return 0.0,True,groups
if 'train2_after' in text: return 2.0,True,groups
return 1.0,True,groups
...
np.full(T,day)
```

iii. The fallback was justified as an explicit experimental-stage ordering, with imputation exposed in session metadata.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from retained original frame indices and the session median frame interval. `StartFr` and raw timestamps at each retained frame are not used directly.

ii.
```python
since=(idx-idx[0])*dt
```

iii. The AI defines trial start as the first retained running corridor frame.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The first retained index is subtracted from every retained index and the difference is multiplied by `dt`, producing zero at the first stored bin and retaining elapsed gaps caused by removed stationary frames.

ii.
```python
since=(idx-idx[0])*dt
```

iii. This was chosen to implement the AI's running-only corridor alignment and was checked to be monotonic and start at zero.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses exactly the same retained `idx` array and has length `T`, matching the neural trial view.

ii.
```python
T=len(idx); ntrial=selected[:,offset:offset+T]
since=(idx-idx[0])*dt
inp=np.vstack([cue,np.full(T,day),since,np.full(T,rew[t])])
```

iii. The AI's global structural checks confirmed identical neural/input/output time dimensions.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from trial-level `isRew`.

ii.
```python
rew=np.asarray(b['isRew']).astype(int)
```

iii. The notes identify this as the source flag for rewarded-corridor availability.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The value is cast to integer and broadcast unchanged across all retained frames in its trial.

ii.
```python
np.full(T,rew[t])
```

iii. No further transformation is needed because `isRew` already represents the requested binary variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from trial-level `WallName`. Alias records are required to agree on `WallName`; the first alias supplies the array.

ii.
```python
out=np.asarray(records[alias_names[0]]['WallName']).astype(str).copy()
for k in alias_names[1:]:
    other=np.asarray(records[k]['WallName']).astype(str)
    if not np.array_equal(out,other): raise ValueError(...)
```

iii. The AI initially explored `TrialStim`, then corrected the implementation to physical wall identity because swap annotations could otherwise produce a literal placeholder and omit rock/wood labels.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Every distinct raw `WallName` string is globally sorted, assigned its own class index, and broadcast across the trial. Variants such as `circle1`, `circle2`, and `circle3` are not collapsed to four base textures.

ii.
```python
speed_q,stim_values,scan_stats=scan(...)
stim_to_idx={x:i for i,x in enumerate(stim_values)}
stim=np.full(T,stim_to_idx[str(stimuli[t])],dtype=np.int16)
```

iii. The AI describes this as using “physical visual wall identity” and preserving global categorical values. Full decoder chance (0.0667) confirms 15 categories rather than the reference's four.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses lick frame `LickFr` together with lick trial identity `LickTrind`.

ii.
```python
lf=np.asarray(b['LickFr'],float); lt=np.asarray(b['LickTrind'],float)
for f,t in zip(lf,lt):
    if np.isfinite(f) and np.isfinite(t): lick_by_trial.setdefault(int(t),[]).append(float(f))
```

iii. Trial identity was included to prevent boundary assignment errors.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary vector is initialized per retained trial. Each finite lick is assigned to the nearest retained frame in its stated trial only if it lies within 0.5001 frame; multiple licks in a bin remain 1. Licks during excluded stationary/gray periods are not relocated.

ii.
```python
lick=np.zeros(T,dtype=np.int16)
for f in lick_by_trial.get(t,[]):
    j=int(np.argmin(np.abs(idx-f)))
    if abs(float(idx[j])-f)<=0.5001: lick[j]=1
```

iii. The half-frame guard was added after review to avoid moving an excluded-period lick to an unrelated retained frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are mapped onto positions within the same retained `idx` array that selects neural columns, yielding exactly `T` binary samples.

ii.
```python
j=int(np.argmin(np.abs(idx-f)))
...
out=np.vstack([stim,lick,position,speedbin])
```

iii. The AI used the common imaging-frame coordinate and verified time lengths for every trial.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from frame-level `ft_Pos` at retained indices.

ii.
```python
pos=np.asarray(b['ft_Pos'][:nfr],np.float32)
position=np.clip((pos[idx]/10).astype(np.int16),0,3)
```

iii. Cross-checks established that source position is in decimeters and that `ft_CorrSpc` corresponds to the 0–4 m texture.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI divides source values by 10, truncates to an integer, and clips to class 0–3.

ii.
```python
position=np.clip((pos[idx]/10).astype(np.int16),0,3)
```

iii. This converts decimeters to the requested four one-meter categories.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Classes correspond to `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40]` source units, labeled 0–1 m through 3–4 m; clipping handles boundaries.

ii.
```python
output_values=[...,['0-1 m','1-2 m','2-3 m','3-4 m'],...]
position=np.clip((pos[idx]/10).astype(np.int16),0,3)
```

iii. The corridor geometry and exact mask/position relationship were checked against data and paper.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed by the exact retained neural frame indices `idx`.

ii.
```python
ntrial=selected[:,offset:offset+T]
position=np.clip((pos[idx]/10).astype(np.int16),0,3)
```

iii. Raw-trial checks found exact agreement and median position progression was fully nondecreasing.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from frame-level `ft_RunSpeed` at retained corridor/running indices.

ii.
```python
speed=np.asarray(b['ft_RunSpeed'][:nfr],np.float32)
speedbin=np.digitize(speed[idx],speed_q,right=False).astype(np.int16)
```

iii. `ft_RunSpeed` is the complete synchronized source stream; an absent alternative `run_pos` field was unnecessary.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A behavior pre-scan concatenates speeds from all retained frames across the selected dataset, computes global 25th/50th/75th percentile thresholds, and applies them using `np.digitize`.

ii.
```python
speeds.append(speed[m])
speed=np.concatenate(speeds); qs=np.quantile(speed,[.25,.5,.75]).astype(np.float32)
...
speedbin=np.digitize(speed[idx],speed_q,right=False).astype(np.int16)
```

iii. The AI chose global thresholds so the four labels have consistent physical meaning across sessions and approximately 25% of all retained samples each.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Categories are `<Q25`, `Q25–Q50`, `Q50–Q75`, and `>=Q75` using dataset-wide numeric quantiles and `right=False`. Ties are not forcibly split, though global proportions were nearly equal.

ii.
```python
qs=np.quantile(speed,[.25,.5,.75]).astype(np.float32)
speedbin=np.digitize(speed[idx],speed_q,right=False).astype(np.int16)
```

iii. The notes report thresholds around 12.42, 25.35, and 40.85 and validate near-25% global class fractions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed values are selected using exactly the same `idx` used for the neural trial.

ii.
```python
ntrial=selected[:,offset:offset+T]
speedbin=np.digitize(speed[idx],speed_q,right=False).astype(np.int16)
```

iii. Independent raw checks found exact speed-bin and temporal agreement for sampled trials.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior is truncated to the neural frame count; nonfinite `ft_trInd` values become sentinel -1; nonfinite lick fields are ignored; missing behavior aliases, alias disagreement, and neuron/area length mismatches raise errors. Missing `sess#` is imputed from experiment stage and flagged in metadata. Empty trials are skipped.

ii.
```python
nfr=min(neural.shape[1],len(b['ft_trInd']))
tri=np.where(np.isfinite(tri_raw),tri_raw,-1).astype(int)
if np.isfinite(f) and np.isfinite(t): ...
if len(areas)!=neural.shape[0]: raise ValueError(...)
```

iii. The notes document fixes for NaN trial IDs and lick relocation, and expose training-day imputation rather than silently hiding it.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating the enormous object-NPY neural files, selecting/copying retained neural frames, and writing the roughly 161.6 GiB pickle dominate conversion. The later full decoder's per-session SVD and CPU training are costly but are outside `convert_data.py`.

ii.
```python
o=np.load(ROOT/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
return np.concatenate([np.asarray(x,dtype=np.float32) for x in o],axis=0)
selected=np.ascontiguousarray(neural[:,ordered],dtype=np.float32)
pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes identify 412 GB of source data and show that allocation fragmentation originally increased session time dramatically; one contiguous selection fixed that slowdown.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial construction of `idx`, trial input/output arrays, and the per-lick nearest-frame search could potentially be grouped/vectorized. Session iteration and per-session variable-shaped trial output remain structurally necessary. The AI already vectorizes masks, frame selection, position, and speed.

ii.
```python
for t in range(int(b['ntrials'])):
    idx=np.flatnonzero(mask&(tri==t))
...
for t,idx in zip(trial_ids,trial_indices):
...
    for f in lick_by_trial.get(t,[]):
        j=int(np.argmin(np.abs(idx-f)))
```

iii. The notes emphasize vectorized masks and a single contiguous neural selection; they do not claim the remaining small loops dominate compared with neural I/O and copying.

## 12-c. What processing does the code repeat multiple times?

i. The pre-scan and conversion both choose behavior aliases, construct the retained mask, normalize trial IDs, and traverse trials. `merged_stimuli` is also called during both phases. This repetition avoids loading neural data during the global stimulus/speed scan but repeats behavior-side work.

ii.
```python
# scan
b=choose_behavior(sid,aliases,records); st=merged_stimuli(aliases[sid],records)
m=retained_mask(b)
...
# convert_session
b=choose_behavior(sid,aliases,records); stimuli=merged_stimuli(aliases[sid],records)
...
mask=retained_mask(b,nfr)
```

iii. The notes justify the behavior-only pre-scan as a way to compute global categories and quartiles without loading neural files; the repeated lightweight work is intentional.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `record_files` is populated and returned but never used; `scan` accepts unused `db`; `convert_session` accepts unused `plot`; and `kept_ids`/extensive session metadata are not used by decoder training, though useful for audit. Float32 neural storage also retains more precision than the human reference's float16. The core selected activity and target streams are used downstream.

ii.
```python
records={}; record_files={}
records[k]=v; record_files[k]=f.name
...
def scan(records,db,sids,aliases):
...
def convert_session(...,plot=False):
```

iii. The agent did not explicitly discuss these minor unused values. It did justify metadata/source trial IDs for traceability and float32 as preserving exact source activity.
