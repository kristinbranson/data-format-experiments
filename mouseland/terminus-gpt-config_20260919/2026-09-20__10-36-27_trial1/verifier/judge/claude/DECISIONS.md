# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads three subdirectories of `/app/data`: `beh/` (behavior), `spk/` (deconvolved traces) and `retinotopy/` (visual-area label per neuron). In `load_sources()` it globs **every** `Beh_*.npy` file and flattens all of them into one dict `records` keyed by behavior-session key (any entry that is a dict containing `ntrials`). It separately loads `beh/Imaging_Exp_info.npy` into `db`, keyed by `mname_datexp_blk`, keeping a *list* of (experiment-group, entry) pairs per session because a recording is listed under several experiment groups. The authoritative session list is taken from the **spike directory** (`spk/*_neural_data.npy` → 89 ids), and each session id is linked to its behavior key(s) by stripping a `_swap1`/`_swap2` suffix (`base_id`). Spikes and retinotopy are then loaded once per session inside `convert_session`. Result: 89 sessions, 19 mice, 38,110 trials, 4,691,034 neurons.

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
    if any(not x for x in aliases.values()): raise RuntimeError('Missing behavior: '+...)
    return records,record_files,db,neural,aliases
```
```python
def load_neural(sid):
    o=np.load(ROOT/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
    return np.concatenate([np.asarray(x,dtype=np.float32) for x in o],axis=0)

def area_indices(sid):
    datebase='_'.join(sid.split('_')[:-1])
    with np.load(ROOT/'retinotopy'/f'{datebase}_trans.npz') as z: a=z['iarea'].astype(int)
```

iii. From CONVERSION_NOTES Step 4: "All 89 physical recordings have neural, retinotopy, and behavior data when swap suffix aliases are recognized." The AI found that direct id matching only covered 76 of 89 recordings and diagnosed the missing 13 as behavior keys carrying a `_swap1`/`_swap2` suffix; stripping only that suffix recovers all 89. It anchors the session list on the spike files so that no neural recording can be silently dropped, and mirrors the reference `load_spk` by concatenating the three imaging planes on the neuron axis.

## 1-b. How are the data split into subjects (mice)?

i. The mouse is the underscore-prefix of the session id (`sid.split('_')[0]`). `subjects` is the sorted unique set of prefixes (19 mice) and `subject_idx` is each session's index into that list, in the same order as `neural`/`input`/`output`.

ii.
```python
subjects=sorted({s.split('_')[0] for s in sids}); subjmap={x:i for i,x in enumerate(subjects)}
data.update(subjects=subjects,
            subject_idx=np.asarray([subjmap[s.split('_')[0]] for s in sids],dtype=np.int32), ...)
```

iii. The session id is built as `mname_datexp_blk` from the experiment database, so the mouse name is already carried in the id and no separate lookup is needed. The AI verified the result against the paper ("We performed 89 recordings in 19 mice"): 19 subjects, with 1–8 sessions each.

## 1-c. How are the data split into sessions?

i. A session is one neural recording file, i.e. one mouse / date / block triple. The 89 `spk/*_neural_data.npy` files define the session list. Behavior records that are swap aliases of the same recording (`..._swap1`, `..._swap2`) are **merged into one session**, not treated as separate sessions; the first (non-swap-preferred) alias is used for the frame-wise streams after checking that all aliases carry identical `WallName`.

ii.
```python
def base_id(k): return re.sub(r'_swap[12]$','',k)
...
neural=sorted(re.sub(r'_neural_data$','',f.stem) for f in (ROOT/'spk').glob('*_neural_data.npy'))
aliases={sid:sorted([k for k in records if base_id(k)==sid],key=lambda k:('_swap' in k,k)) for sid in neural}

def choose_behavior(sid,aliases,records): return records[aliases[sid][0]]

def merged_stimuli(alias_names,records):
    """Return physical visual wall identity; aliases share identical WallName."""
    out=np.asarray(records[alias_names[0]]['WallName']).astype(str).copy()
    for k in alias_names[1:]:
        other=np.asarray(records[k]['WallName']).astype(str)
        if not np.array_equal(out,other):
            raise ValueError(f'Behavior aliases disagree on WallName: {alias_names}')
    return out
```

iii. CONVERSION_NOTES Step 4/Key decision 1: "Physical sessions, not statistical aliases: one target session per 89 neural recordings. Swap behavior aliases share identical frames and are merged only to recover both swap stimulus labels." The AI verified that swap1/swap2 records have identical frame streams, trial indices, reward, cue and lick arrays and only differ in the `TrialStim`/`stim_id` annotation, so duplicating them would duplicate neurons and frames. Critical Review 1 includes an explicit "SWAP ALIAS EDGE CASE PASS" check.

## 1-d. How are the data split into trials?

i. Trials are the ones the behavior declares (`ntrials`), and each frame's trial membership comes from `ft_trInd`. Within a trial the AI keeps only frames that are **inside the textured corridor and during running**: `ft_CorrSpc & (ft_move > 0)`. All streams are first truncated to `nfr = min(n_neural_frames, len(ft_trInd))`. Non-finite `ft_trInd` values are mapped to −1 so they can never match a trial. Trials therefore have variable length (mean 22.25, min 11, max 178 bins) and, because stationary frames are removed from the middle of a traversal, a trial's frames need not be temporally contiguous. 821,579 frames are retained over 38,110 trials.

ii.
```python
def retained_mask(b,nfr=None):
    n=len(b['ft_trInd']) if nfr is None else min(nfr,len(b['ft_trInd']))
    return np.asarray(b['ft_CorrSpc'][:n],bool)&(np.asarray(b['ft_move'][:n])>0)
```
```python
neural=load_neural(sid); nfr=min(neural.shape[1],len(b['ft_trInd']))
tri_raw=np.asarray(b['ft_trInd'][:nfr],float)
tri=np.where(np.isfinite(tri_raw),tri_raw,-1).astype(int); mask=retained_mask(b,nfr)
...
for t in range(int(b['ntrials'])):
    idx=np.flatnonzero(mask&(tri==t))
    if len(idx): trial_indices.append(idx); trial_ids.append(t)
```

iii. From CONVERSION_NOTES Step 3/4: "The paper states that only running timepoints were analyzed, removing pauses for reward collection. Reference code implements this with `ft_move > 0` before spatial interpolation," and "Reference-consistent trial samples use deconvolved traces during running inside the 4 m visual corridor. Gray-space and stationary frames are excluded." This mirrors `/app/code/utils.py` (`VRmove = beh['ft_move'][:nfr]>0; corr_fr = beh['ft_CorrSpc'][:nfr] & VRmove`) and the methods text ("We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards").

## 1-e. How are trials filtered based on quality controls?

i. Essentially no trial-level quality filter is applied. The only rule is that a trial with **zero** retained frames after masking/truncation is skipped; the pre-scan reported `empty: 0`, so all 38,110 source trials were kept. There is no trial-duration outlier rule, no minimum-length rule, and no per-session minimum-trial rule in the code (the AI verified post hoc that every session has ≥2 trials). The running mask implicitly removes stationary frames, so a "parked mouse" trial shrinks rather than being dropped — but its frame *indices* still span the whole pause, so `time since trial start` reaches 1,763.7 s and `time to sound cue` reaches −1,762.0 s for such trials.

ii.
```python
for t in range(int(b['ntrials'])):
    idx=np.flatnonzero(mask&(tri==t))
    if len(idx): trial_indices.append(idx); trial_ids.append(t)
```
```python
stats={'trials_source':0,'trials_kept':0,'frames':0,'empty':0}
...
z=int(np.sum(m&(tri==t))); stats['frames']+=z
if z: stats['trials_kept']+=1
else: stats['empty']+=1
```

iii. CONVERSION_NOTES Key decision 4: "Trial filtering: `ft_trInd == trial`, `ft_CorrSpc`, and `ft_move > 0`, after truncation to neural frame count. Every investigated trial has at least one valid frame; any empty trial is skipped and reported." Step 3 argues that the figure-specific trial splits in the reference code "are not a general reason to discard trials for the requested decoder". In the trajectory (step 39) the AI explicitly saw the long traversals and dismissed them: "Cue/time-since-start ranges are long in a few slow traversals, but remain consistent with source frame indices and the paper's running-only criterion."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy` — a list of one (neurons × frames) array per imaging plane — concatenated on the neuron axis. The area label of each neuron comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`. The code asserts that the retinotopy length equals the concatenated neuron count.

ii.
```python
def load_neural(sid):
    o=np.load(ROOT/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
    return np.concatenate([np.asarray(x,dtype=np.float32) for x in o],axis=0)
...
neural=load_neural(sid); nfr=min(neural.shape[1],len(b['ft_trInd']))
neural=neural[:,:nfr]; areas=area_indices(sid)
if len(areas)!=neural.shape[0]: raise ValueError(f'{sid}: area/neuron mismatch {len(areas)} {neural.shape[0]}')
```

iii. Step 4: "`load_spk` concatenates list entries on axis 0 … Each file contains three same-frame-count matrices; retinotopy length equals summed neuron count … Concatenate all three imaging planes along neurons exactly as reference code."

## 2-b. How is the `neural` data processed?

i. No transformation at all: the deconvolved traces are used as-is. The columns belonging to a trial are selected and stored as **float32**. Trials keep their native, variable length; nothing is padded, smoothed, normalized, or spatially interpolated. For efficiency the whole session is first re-indexed once into one contiguous array in trial order, and each trial array is a slice of it.

ii.
```python
ordered=np.concatenate(trial_indices) if trial_indices else np.empty(0,dtype=int)
selected=np.ascontiguousarray(neural[:,ordered],dtype=np.float32)
del neural; gc.collect()
offset=0
for t,idx in zip(trial_ids,trial_indices):
    T=len(idx); ntrial=selected[:,offset:offset+T]; offset+=T
```

iii. Step 3: "Paper analyses use nonnegative deconvolved fluorescence traces generated by Suite2p; no additional dF/F operation should be applied." Step 5 key decision 10: "Data types: neural/input float32; categorical outputs int16." Step 9/10 records that the per-trial advanced indexing was replaced by one contiguous per-session selection plus trial views because allocation fragmentation was slowing conversion from ~5 s to ~66 s per session.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neurons are removed.** All 4,691,034 Suite2p cells are kept. `iarea` is mapped to six region labels — `V1` (8), `medial higher visual` (0,1,2,9), `lateral higher visual` (5,6), `anterior higher visual` (3,4), plus two extra labels the reference code does not define: `other visual area` (7, 188,331 neurons) and `unassigned` (−1, 397,310 neurons) — and stored in `brain_region_idx` so downstream code can filter by area. Per-session neuron counts are 20,547–89,577, exactly the paper's stated range.

ii.
```python
REGIONS=['V1','medial higher visual','lateral higher visual','anterior higher visual','other visual area','unassigned']

def area_indices(sid):
    datebase='_'.join(sid.split('_')[:-1])
    with np.load(ROOT/'retinotopy'/f'{datebase}_trans.npz') as z: a=z['iarea'].astype(int)
    out=np.full(len(a),5,dtype=np.int16)
    out[a==8]=0; out[np.isin(a,[0,1,2,9])]=1; out[np.isin(a,[5,6])]=2
    out[np.isin(a,[3,4])]=3; out[a==7]=4; out[a==-1]=5
    return out
```

iii. Step 4: "Suite2p already performed cell classification; paper reports all 20,547–89,577 traces … Keep all neurons. Selectivity masks are figure-specific, not quality curation. Map all numeric area labels, including −1 as unassigned/outside mapped areas." Key decision 2: "All neurons retained: reference `load_spk` concatenates all planes and the paper's reported population range includes all Suite2p-classified cells. No arbitrary subsampling is applied despite the large (~162 GiB estimated) output."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry, operationalised as the **first retained (in-corridor, running) frame of the trial**. Every trial array starts at that frame and runs to the last retained frame of its own traversal; lengths differ between trials and nothing is padded or truncated to a common window. Metadata records `off_start = 0.0`, `off_end = None`, `temporal_alignment_event = 'entry into 4-m visual corridor (first retained running corridor frame)'`.

ii.
```python
since=(idx-idx[0])*dt          # zero at the first retained corridor frame
...
'temporal_alignment_event':'entry into 4-m visual corridor (first retained running corridor frame)',
'off_start':0.0,'off_end':None,
```

iii. Step 4: "Define each trial from `ft_trInd`, retain corridor frames (`ft_CorrSpc`) that also satisfy reference running criterion (`ft_move>0`), and set time zero at the first retained corridor frame." The AI rejected `StartFr` as the zero point because "Some cue-minus-StartFr offsets are extreme during pauses", preferring a zero that is guaranteed to be a real retained sample.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame is the bin — one column of `spks` per frame — so **no rebinning or resampling is done**. The bin duration is measured empirically per session as the median positive `diff(ft) * 86400` s (≈0.3145–0.3153 s across sessions), with a hard-coded global fallback `DT_GLOBAL = 0.31480416655540466` s. `metadata['time_bin_size']` is reported as `DT_GLOBAL*1000` ≈ 314.804 ms for all sessions. The per-session `dt` is stored in `session_info['session_dt_seconds']` and is what is actually used to convert frame offsets to seconds.

ii.
```python
DT_GLOBAL=0.31480416655540466
...
ft=np.asarray(b['ft'][:nfr],float); d=np.diff(ft)*86400; d=d[np.isfinite(d)&(d>0)]
dt=float(np.median(d)) if len(d) else DT_GLOBAL
...
'time_bin_size':DT_GLOBAL*1000,
```

iii. Step 4: "The paper does not state neural/behavior frame duration. This must be estimated from synchronized frame timestamps in source data … Median synchronized frame interval is 0.314804 s (1st–99th percentile 0.301–0.330 s) … Preserve one sample per acquired volume/frame; report 314.804 ms nominal bin size. No temporal resampling is needed because all streams already share this clock." Step 5 decision 3 adds that the paper's 60-bin *spatial* interpolation is deliberately not used because it would destroy lick timing and running speed.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the fractional imaging-frame index of the cue on each trial), the retained frame indices of the trial, and the session's frame duration `dt` derived from `ft`.

ii.
```python
rew=np.asarray(b['isRew']).astype(int); sound=np.asarray(b['SoundFr'],float)
...
cue=(sound[t]-idx)*dt
```

iii. Step 5 mapping table: "`SoundFr`, retained frame indices, frame interval → `input[0]`; Signed seconds until cue: `(SoundFr[t] - frame_index) * dt`; reference function `spk_2_cue`; Continuous, time-varying; positive before cue, negative after." The AI notes that the reference code aligns sound analyses on the same `SoundFr` frame index.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue frame index is subtracted from each retained frame index and the difference is scaled by the session frame duration, giving seconds. The sign convention is *time to* the cue: positive before the cue, zero at the cue, negative after. No interpolation of `ft` is used; the uniform per-session `dt` is assumed. Values are stored as float32 in row 0 of the trial input matrix. Global range is [−1762.0, 722.7] s (the extremes come from trials in which the mouse stood still for many minutes mid-corridor).

ii.
```python
cue=(sound[t]-idx)*dt
since=(idx-idx[0])*dt
inp=np.vstack([cue,np.full(T,day),since,np.full(T,rew[t])]).astype(np.float32)
```

iii. Step 5, decision 5: "Nominal bin duration: use each session's median positive `diff(ft)*86400` for event/time variables." The AI validated the sign crossing as a planned sanity check ("Check … cue-time sign crossing") and plots `inp[j][0]` against a zero line in `--show-processing` mode.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from exactly the same `idx` array of retained frame indices that selects the neural columns for that trial, so it is sample-for-sample aligned and has the same length `T`.

ii.
```python
for t,idx in zip(trial_ids,trial_indices):
    T=len(idx); ntrial=selected[:,offset:offset+T]; offset+=T
    cue=(sound[t]-idx)*dt
```

iii. Step 4/10: "Temporal alignment — Common imaging frame indices (`SoundFr`, `LickFr`) … Same frame clock; corridor entry is trial time zero." All streams live on the single imaging-frame clock, so alignment is by construction. Critical Review 1 re-derived the cue times from raw behavior for nine trials in three sessions and compared with `np.allclose` (PASS).

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `sess#` field of the entries in `beh/Imaging_Exp_info.npy` for that recording. A recording can be listed under several experiment groups, and the AI takes the **maximum** `sess#` over all its listings. If no listing has `sess#` (8 of 142 entries — these are exactly the entries that instead carry a `days` field, which the code never reads), a hard-coded stage heuristic is used: 0 if any group name contains `before`, 2 if any contains `train2_after`, else 1. Whether the value was imputed is recorded per session in metadata.

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
    # Explicit stage fallback: before=0, first after/test=1, later train2 after=2.
    text=' '.join(groups)
    if 'before' in text: return 0.0,True,groups
    if 'train2_after' in text: return 2.0,True,groups
    return 1.0,True,groups
```

iii. Step 4: "Reference metadata includes `sess#` … `sess#` is 0 before learning, 1 after/test in many groups, and later values (6, 10, 12) for later sessions; a few are missing … Paper discusses before/after and days of training but does not provide another per-recording numeric day field. Use numeric `sess#` as source-provided training-session/day index. Resolve missing values from ordered date/condition context in mapping step and record imputation." Key decision 9: "Training day: use source `sess#`, the only numeric training-session field."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The scalar returned by `session_day` is broadcast as a constant row across every bin of every trial of the session and stored as float32. No per-mouse ordering, date arithmetic, or normalisation is applied. The resulting values across the 89 sessions are mostly 0/1/2 with a handful of 4, 6, 10 and 12, i.e. the same variable encodes a before/after-learning stage for most sessions and an actual day count for a few.

ii.
```python
day,imputed,groups=session_day(sid,db)
...
inp=np.vstack([cue,np.full(T,day),since,np.full(T,rew[t])]).astype(np.float32)
...
meta={'session_id':sid,...,'training_day':day,'training_day_imputed':imputed,
      'experiment_groups':groups,...}
```

iii. As above: the AI treated `sess#` as "the only numeric training-session field" available in the source metadata and deliberately preferred a source-provided field over a derived one, documenting each session's value and imputation flag in `metadata['session_info']` so the mapping can be audited.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the retained frame indices of the trial (derived from `ft_trInd`, `ft_CorrSpc`, `ft_move`) and the session frame duration `dt` (from `ft`). `StartFr` is deliberately not used.

ii.
```python
tri=np.where(np.isfinite(tri_raw),tri_raw,-1).astype(int); mask=retained_mask(b,nfr)
...
idx=np.flatnonzero(mask&(tri==t))
...
since=(idx-idx[0])*dt
```

iii. Step 5 mapping table: "retained frame order → `input[2]`; Seconds since first retained corridor-running frame; Continuous, time-varying; trial start is corridor entry." Step 4 gives the reason for not using `StartFr`: "Behavior has fractional `StartFr` … Some cue-minus-StartFr offsets are extreme during pauses."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The first retained frame index of the trial is subtracted from every retained frame index and scaled by `dt`, giving seconds. It therefore starts at exactly 0.0 for every trial and increases monotonically. Because stationary frames are excluded from the trial but the frame *indices* still advance through the pause, the series jumps over pauses; its global maximum is 1,763.7 s even though a normal 4 m traversal is ~7 s. Stored as float32 in row 2.

ii.
```python
since=(idx-idx[0])*dt
inp=np.vstack([cue,np.full(T,day),since,np.full(T,rew[t])]).astype(np.float32)
```

iii. The AI's planned sanity checks include "Check finite values, monotonic time-since-start …", and Step 12 check 5 reports "every input/output pair has identical time length; raw retained indices match trial/corridor/running masks". The choice of the first retained frame as zero is justified in Step 4 as making time zero a real sample rather than a fractional event index that may fall inside an excluded pause.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is built from the same `idx` array used to slice the neural columns, so it is sample-for-sample aligned and of the same length `T`.

ii.
```python
T=len(idx); ntrial=selected[:,offset:offset+T]; offset+=T
since=(idx-idx[0])*dt
```

iii. Same as 3-c: all streams share the imaging-frame clock, and the AI re-derived time-since-entry from raw behavior for nine spot-check trials and compared with `np.allclose` in Critical Review 1.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, the per-trial flag marking trials run in the rewarded corridor.

ii.
```python
rew=np.asarray(b['isRew']).astype(int)
```

iii. Step 5 mapping table: "`isRew[t]` → `input[3]`; Boolean converted to 0/1 and repeated over time; `get_cat_id` uses same reward field; Reward availability in rewarded corridor."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean is cast to int and broadcast as a constant row across all bins of the trial, stored as float32 in row 3. Nothing else. It is 0 for every trial of the unsupervised/naive sessions and 0/1 per trial in task sessions, as the per-session input ranges in the verification log show.

ii.
```python
inp=np.vstack([cue,np.full(T,day),since,np.full(T,rew[t])]).astype(np.float32)
```

iii. No further justification is given beyond the mapping table; the AI's sanity checks include "binary licking/reward ranges", and the sample-session selection deliberately picked sessions containing both reward classes.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the per-trial name of the corridor wall texture. The AI first tried `TrialStim`, found it unsuitable, and switched to `WallName`. Where a session has swap aliases, all aliases are checked to carry identical `WallName` and the first one is used.

ii.
```python
def merged_stimuli(alias_names,records):
    """Return physical visual wall identity; aliases share identical WallName."""
    out=np.asarray(records[alias_names[0]]['WallName']).astype(str).copy()
    for k in alias_names[1:]:
        other=np.asarray(records[k]['WallName']).astype(str)
        if not np.array_equal(out,other):
            raise ValueError(f'Behavior aliases disagree on WallName: {alias_names}')
    return out
```

iii. Step 9 iteration log: "First full attempt was stopped after 9 sessions when the pre-scan revealed a literal `stimulus_of_trial` output class and missing rock/wood labels. Cause: `TrialStim` is a paper-analysis normalization field, not physical wall identity; it maps rock/wood families onto circle/leaf and contains placeholders in some tests. Fix: use `WallName`, which directly provides all physical visual categories including swap variants." (Note: the Step 5 mapping table still says "merged `TrialStim` aliases / `WallName`"; the Step 9/10 text supersedes it and the code uses `WallName` only.)

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The set of distinct `WallName` strings over the whole dataset is collected in the behavior-only pre-scan, sorted, and used directly as the category list — **15 classes**: `circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5`. No mapping onto the four base textures (circle/leaf/rock/wood) is performed: different crops of the same photograph and spatially shuffled versions are separate classes. The per-trial index is broadcast across all bins as int16 (row 0 of the output matrix). Global class fractions are very uneven (0.008–0.264) and most sessions contain only 2–4 of the 15 classes.

ii.
```python
speeds=[]; stimset=set(); ...
    stimset.update(map(str,st))
...
return qs,sorted(stimset),stats
...
speed_q,stim_values,scan_stats=scan(...); stim_to_idx={x:i for i,x in enumerate(stim_values)}
...
stim=np.full(T,stim_to_idx[str(stimuli[t])],dtype=np.int16)
out=np.vstack([stim,lick,position,speedbin]).astype(np.int16)
...
output_values=[stim_values, ['not licking','licking'], ['0-1 m','1-2 m','2-3 m','3-4 m'], ['Q1','Q2','Q3','Q4']]
```

iii. Step 5 decision 8: "Stimulus labels: use merged `TrialStim` because it distinguishes leaf/circle/rock/wood and swap variants … Output values are global sorted labels" — i.e. the AI's explicit goal was to preserve the *fine-grained physical stimulus identity*, and after the `TrialStim` failure it achieved that with `WallName`. Step 9 records "15 physical visual categories". Step 11/12 then evaluate the decoder against uniform chance 1/15 = 0.0667 and report 0.5921 validation balanced accuracy, "8.88× chance".

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (the fractional imaging-frame index of every lick in the session) together with `LickTrind` (the trial each lick belongs to). Only licks with finite values in both arrays are used.

ii.
```python
lick_by_trial={t:[] for t in range(int(b['ntrials']))}
lf=np.asarray(b['LickFr'],float); lt=np.asarray(b['LickTrind'],float)
for f,t in zip(lf,lt):
    if np.isfinite(f) and np.isfinite(t): lick_by_trial.setdefault(int(t),[]).append(float(f))
```

iii. Step 5 mapping table: "`LickFr`, `LickTrind` → `output[1]`; Binary vector; assign each lick to nearest retained frame in its supplied trial; reference `spk_2_cue`, `get_lick_raster`; Multiple licks in one bin remain class 1."

## 8-b. What processing is involved in computing `output` *Licking*?

i. Per trial, a zero vector of length `T` is created; for each lick of that trial the nearest retained frame index is found and set to 1, but only if it is within 0.5 frames of the fractional lick frame (i.e. the lick is effectively rounded to its nearest imaging frame and discarded if that frame was not retained). Multiple licks in one bin collapse to 1. Stored as int16 in row 1. Globally 3.47% of bins are licks.

ii.
```python
lick=np.zeros(T,dtype=np.int16)
for f in lick_by_trial.get(t,[]):
    j=int(np.argmin(np.abs(idx-f)))
    if abs(float(idx[j])-f)<=0.5001: lick[j]=1
```

iii. Step 5 decision 6: "Lick assignment: `LickFr` is fractional. For each event, use `LickTrind` and choose the nearest retained frame belonging to that trial, preventing boundary assignment errors." Step 10 issues: "Lick relocation risk: nearest retained frame could move excluded-period licks. Fixed by requiring distance ≤0.5 imaging frame and matching supplied `LickTrind`." Step 12 check 6 confirms "licking is sparse but not degenerate (3.47% positive globally)".

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes imaging frames, the same clock as `spks`, and the lick vector is indexed by position within the trial's retained-frame array `idx`, so it is exactly aligned with the neural columns and has length `T`.

ii.
```python
T=len(idx); ntrial=selected[:,offset:offset+T]
...
j=int(np.argmin(np.abs(idx-f)))
if abs(float(idx[j])-f)<=0.5001: lick[j]=1
```

iii. Step 10 comparison table: "Temporal alignment — Common imaging frame indices (`SoundFr`, `LickFr`) — Same frame clock". Critical Review 1 re-derived "nearest-frame licking with `LickTrind`" from raw behavior for nine trials and compared with `np.allclose` (PASS).

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the in-corridor position at each imaging frame, in decimeters (0–40 across the 4 m texture, 40–60 through the grey space).

ii.
```python
pos=np.asarray(b['ft_Pos'][:nfr],np.float32)
...
position=np.clip((pos[idx]/10).astype(np.int16),0,3)
```

iii. Step 4: "Source position units — Code uses `Corridor_Length=60`, `Texture_Length=40`, and corridor mask; `ft_CorrSpc == (ft_Pos < 40)` exactly; gray is 40–60 … Source units are decimeters. Within corridor, physical meters = `ft_Pos / 10`."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by 10 and truncated to an integer, giving the metre index. Stored as int16 in row 2. Only in-corridor frames are in a trial, so the value is always 0–3; the `np.clip(...,0,3)` is a guard. Resulting global class fractions are 0.250 / 0.249 / 0.250 / 0.252.

ii.
```python
position=np.clip((pos[idx]/10).astype(np.int16),0,3)
```

iii. Step 5 mapping table: "`ft_Pos` → `output[2]`; Convert decimeters to meters and classify by `floor(pos/10)` into four bins; Classes: 0–1, 1–2, 2–3, 3–4 m." Planned sanity check: "Verify four position classes are approximately equally represented because corridors use fixed 1 m extents" — confirmed in Step 12.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four fixed, equal-length 1 m bins with boundaries at 10, 20 and 30 decimeters, labelled `['0-1 m','1-2 m','2-3 m','3-4 m']`, exactly as the Decoder Task specifies. The thresholds are geometric, not data-driven.

ii.
```python
position=np.clip((pos[idx]/10).astype(np.int16),0,3)
...
output_values=[stim_values,['not licking','licking'],['0-1 m','1-2 m','2-3 m','3-4 m'],['Q1','Q2','Q3','Q4']]
...
'position_source_units_per_meter':10.0,
```

iii. Directly from the instructions ("Position in corridor discretized into 4 equal-length, 1-m-long spatial bins") combined with the AI's Step 4 finding that source units are decimeters and the corridor mask is `ft_Pos < 40`.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` gives one value per imaging frame, and it is indexed with the same `idx` that selects the neural columns, so it is sample-for-sample aligned and of length `T`.

ii.
```python
T=len(idx); ntrial=selected[:,offset:offset+T]; offset+=T
position=np.clip((pos[idx]/10).astype(np.int16),0,3)
```

iii. Same frame-clock argument as above. Step 12 check 5 additionally reports "median fraction nondecreasing position transitions 1.0", i.e. position increases monotonically within converted trials, which is the expected behaviour for a forward corridor traversal and is evidence against misalignment or frame scrambling. `--show-processing` plots position class against time since trial start.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed=np.asarray(b['ft_RunSpeed'][:nfr],np.float32)
...
speedbin=np.digitize(speed[idx],speed_q,right=False).astype(np.int16)
```

iii. Step 5 mapping table: "`ft_RunSpeed` → `output[3]`; Global quartile thresholds from all retained frames; Classes use 25th/50th/75th percentiles; ties handled by `np.digitize(..., right=False)`."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A behavior-only pre-scan pass over **all** sessions concatenates `ft_RunSpeed` over the retained (in-corridor and running) frames and computes the global 25th/50th/75th percentiles once (12.3884, 25.3285, 40.8291 source units). Every frame in every session is then assigned to a bin with `np.digitize` against those three global thresholds. Stored as int16 in row 3. Global fractions are 0.2493 / 0.2502 / 0.2501 / 0.2504; per-session fractions are far from uniform (e.g. one session has 95.6% of its frames in Q1 and 0.0% in Q4). Note that the pre-scan's `retained_mask(b)` is computed on the full behavior length rather than being truncated to the neural frame count, so the thresholds are measured on very slightly more frames than are actually written.

ii.
```python
def scan(records,db,sids,aliases):
    speeds=[]; ...
    for sid in sids:
        b=choose_behavior(sid,aliases,records); ...
        m=retained_mask(b); ...
        speed=np.asarray(b['ft_RunSpeed'][:len(m)],np.float32)
        speeds.append(speed[m]); ...
    speed=np.concatenate(speeds); qs=np.quantile(speed,[.25,.5,.75]).astype(np.float32)
    return qs,sorted(stimset),stats
```
```python
speedbin=np.digitize(speed[idx],speed_q,right=False).astype(np.int16)
```

iii. Step 5 decision 7: "Speed bins: compute global quartiles over the exact retained frames before conversion, yielding preliminary thresholds 12.4224, 25.3526, 40.8546 source speed units. Recompute in script for reproducibility." Planned sanity check: "Verify speed class fractions are approximately 25% each globally" — confirmed in Step 12 (0.24932, 0.25022, 0.25011, 0.25036). Because the running mask already removes stationary frames, the zero-speed tie that would otherwise break a threshold-based split is largely avoided.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three dataset-wide numeric thresholds (the global 25/50/75 percentiles of retained-frame speed) applied with `np.digitize(..., right=False)`, giving classes labelled `['Q1','Q2','Q3','Q4']`. The thresholds are stored in metadata as `speed_quartile_thresholds`. They are global, not per-session and not per-trial.

ii.
```python
qs=np.quantile(speed,[.25,.5,.75]).astype(np.float32)
...
speedbin=np.digitize(speed[idx],speed_q,right=False).astype(np.int16)
...
'speed_quartile_thresholds':speed_q.tolist(),
```

iii. Directly implements "Running speed discretized into 4 bins, each corresponding to 25% of the data", interpreted as 25% of the whole converted dataset. The AI computes the thresholds before conversion in a cheap behavior-only pass so that a single consistent set of thresholds applies to every session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` has one value per imaging frame and is indexed with the same `idx` used for the neural columns, so it is sample-for-sample aligned and of length `T`.

ii.
```python
T=len(idx); ntrial=selected[:,offset:offset+T]; offset+=T
speedbin=np.digitize(speed[idx],speed_q,right=False).astype(np.int16)
```

iii. Same frame-clock argument. Critical Review 2 spot-checked raw speed and quartile classes for three trials in three sessions against the converted arrays with `np.allclose` (PASS), and `--show-processing` plots the speed class against time since trial start.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several guards are present:
- Behavior can run past imaging, so every stream is truncated to `nfr = min(n_neural_frames, len(ft_trInd))`.
- Non-finite `ft_trInd` values (frames outside a valid task period) are mapped to −1, so they never match any trial and are dropped.
- Licks with non-finite `LickFr`/`LickTrind` are skipped, and a lick whose nearest retained frame is more than 0.5 frames away is dropped rather than relocated.
- If the frame-time differences are unusable (empty after filtering to finite, positive values) the session falls back to the global `DT_GLOBAL`.
- A mismatch between the retinotopy length and the neuron count raises an error for that session.
- Missing behavior for any neural recording raises at load time; the swap-alias suffix is stripped so that the 13 recordings whose behavior keys are suffixed are not treated as missing.
- Trials with zero retained frames are skipped (none occurred).
- Missing `sess#` is imputed from the experiment-group names, and the imputation is flagged per session in metadata (`training_day_imputed`).

ii.
```python
neural=load_neural(sid); nfr=min(neural.shape[1],len(b['ft_trInd']))
neural=neural[:,:nfr]; areas=area_indices(sid)
if len(areas)!=neural.shape[0]: raise ValueError(f'{sid}: area/neuron mismatch {len(areas)} {neural.shape[0]}')
tri_raw=np.asarray(b['ft_trInd'][:nfr],float); tri=np.where(np.isfinite(tri_raw),tri_raw,-1).astype(int)
ft=np.asarray(b['ft'][:nfr],float); d=np.diff(ft)*86400; d=d[np.isfinite(d)&(d>0)]
dt=float(np.median(d)) if len(d) else DT_GLOBAL
...
for f,t in zip(lf,lt):
    if np.isfinite(f) and np.isfinite(t): lick_by_trial.setdefault(int(t),[]).append(float(f))
...
if any(not x for x in aliases.values()): raise RuntimeError('Missing behavior: '+...)
```

iii. Step 10 issues found and resolved: "NaN trial IDs: invalid non-task frames produced integer-cast warnings. Fixed by mapping nonfinite raw trial IDs to −1 before valid-frame masking." Step 4 records the truncation rule as matching the reference code's `beh[...][:nfr]` convention, and Step 10 check 7 reports "Global finite/range checks: every stored value is finite; licking is binary; position and speed outputs are in 0–3; trial time is monotonic and starts at zero."

## 12-a. What are the most time-consuming steps of the code?

i. Per the AI's own timing output and notes, the dominant costs are (1) reading the object-dtype `spk/*_neural_data.npy` files, which cannot be memory-mapped and are multi-GB each, and (2) pickling the 161.6 GiB output. Per-session times in `conversion_full_out.txt` range from 2 s to 217 s and total conversion was 3,846 s (~64 min), well over the 15-minute target in the instructions. A secondary cost that the AI identified and fixed mid-run was allocation fragmentation from hundreds of per-trial advanced-indexing copies.

ii.
```python
def load_neural(sid):
    o=np.load(ROOT/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
    return np.concatenate([np.asarray(x,dtype=np.float32) for x in o],axis=0)
```
```python
ordered=np.concatenate(trial_indices) if trial_indices else np.empty(0,dtype=int)
selected=np.ascontiguousarray(neural[:,ordered],dtype=np.float32)
del neural; gc.collect()
```
```python
print(f'{sid}: neurons={len(areas)} trials={len(ns)} frames={meta["n_retained_frames"]} dt={dt:.6f}s time={time.time()-t0:.1f}s',flush=True)
```

iii. Step 6: "Native object-NPY neural files cannot be memory mapped and each session is multi-GB." Step 9: "Full conversion runtime: 3,849 s (~64 min). This exceeded the 15-minute estimate because large object-NPY reads and 161.6 GiB serialization dominated despite allocation optimization." Step 9 iteration log documents the contiguous-selection fix after per-session time rose from ~4–10 s to ~66 s.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python-level loops remain, none of which the AI flags as remaining inefficiencies (it claims "frame selection is vectorized"):
- The per-trial frame search `np.flatnonzero(mask & (tri==t))` re-scans the entire session frame index once per trial (O(n_trials × n_frames)); grouping every frame by its trial in one pass (e.g. `np.argsort` / `np.split`) would do it in one scan. The same pattern appears a second time inside `scan()` as `np.sum(m & (tri==t))`.
- The lick assignment loops over licks in Python and does an O(T) `np.argmin(np.abs(idx-f))` per lick; `np.searchsorted` on the sorted `idx` would vectorize this.
- `load_sources` loops over every key of every behavior file, and `merged_stimuli` re-reads and compares the full `WallName` arrays of aliases.

All of these are negligible next to the neural file I/O and pickling.

ii.
```python
for t in range(int(b['ntrials'])):
    idx=np.flatnonzero(mask&(tri==t))
    if len(idx): trial_indices.append(idx); trial_ids.append(t)
```
```python
for t in range(int(b['ntrials'])):
    z=int(np.sum(m&(tri==t))); stats['frames']+=z
```
```python
for f in lick_by_trial.get(t,[]):
    j=int(np.argmin(np.abs(idx-f)))
    if abs(float(idx[j])-f)<=0.5001: lick[j]=1
```

iii. Step 6 "Code speedups added": "Behavior-only pre-scan computes global categorical values/quartiles without loading neural files. Neural data is loaded exactly once per session, planes are concatenated once, and frame selection is vectorized. Sessions are processed sequentially and raw session arrays are released with garbage collection." The AI's optimization effort went into memory/allocation behaviour rather than into removing these loops.

## 12-c. What processing does the code repeat multiple times?

i. The behavior-only `scan()` pass recomputes, for every session, exactly the work that `convert_session` later redoes: the retained-frame mask (`ft_CorrSpc & ft_move>0`), the sanitised trial index array, and the per-trial frame membership — plus per-trial frame counts that are only used for a printed statistic. All behavior files are also held in memory for the whole run (`records`), and `merged_stimuli` is called once in `scan` and again in `convert_session` for each session. The repeated pass is unavoidable in principle because the global speed quartiles and the global stimulus label set must be known before any trial is written; it is cheap because it never touches the neural files.

ii.
```python
def scan(records,db,sids,aliases):
    for sid in sids:
        b=choose_behavior(sid,aliases,records); st=merged_stimuli(aliases[sid],records)
        m=retained_mask(b); tri=...
        for t in range(int(b['ntrials'])):
            z=int(np.sum(m&(tri==t))); stats['frames']+=z
```
```python
def convert_session(sid,records,db,aliases,stim_to_idx,speed_q,plot=False):
    t0=time.time(); b=choose_behavior(sid,aliases,records); stimuli=merged_stimuli(aliases[sid],records)
    ...
    mask=retained_mask(b,nfr)
```

iii. Step 6: "Behavior-only pre-scan computes global categorical values/quartiles without loading neural files" — i.e. the duplication is a deliberate trade to keep the expensive neural read to exactly one pass. The trial-count/empty-trial statistics gathered in the pre-scan are used only for the printed `scan` line and the `scan_statistics` metadata entry.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest amounts:
- The pre-scan's per-trial frame counting (`trials_kept`, `frames`, `empty`) only feeds a console line and a metadata field; the conversion does not use it.
- `record_files` is built in `load_sources`, returned, and never used.
- `merged_stimuli` re-validates alias `WallName` equality for every session in both passes.
- 585,641 neurons whose retinotopy label is 7 ("other visual area") or −1 ("unassigned") are read, stored and pickled even though the reference analyses never use them; combined with storing the traces as **float32** rather than float16, this is the main driver of the 161.6 GiB output and the 64-minute runtime. The AI treats keeping them as a correctness requirement rather than as overhead.
- `plot` is accepted as a parameter of `convert_session` and never used (plotting is done by the caller).

ii.
```python
stats={'trials_source':0,'trials_kept':0,'frames':0,'empty':0}
...
'session_info':sinfo,'scan_statistics':scan_stats})
```
```python
records,record_files,db,all_sids,aliases=load_sources()   # record_files unused thereafter
```
```python
out[a==7]=4; out[a==-1]=5      # 585,641 neurons outside the four reference areas, all retained
selected=np.ascontiguousarray(neural[:,ordered],dtype=np.float32)
```

iii. Step 6: "Target format requires separate trial arrays, while decoder later concatenates them per session; this creates unavoidable array/object overhead. Full all-neuron target is estimated at ~162 GiB, but arbitrary subsampling would violate the reference loader and reported population statistics." Key decision 10 fixes the neural dtype at float32. The AI does not identify any of its own processing as discardable.
