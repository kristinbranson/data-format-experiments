# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI ignores the master index `beh/Imaging_Exp_info.npy` entirely. It globs every `beh/Beh_*.npy` file in sorted order, unpickles each into a dict of `{session_key: behavior_dict}`, and keeps the first occurrence of each key that looks like behavior (`'ft' in b`). Separately it globs `spk/*_neural_data.npy` and builds `{session_id: path}`. The dataset is then the **set intersection** of behavior keys and spike-file stems, sorted by `(mouse, date, session_id)`. Per session it loads the spike file (list of per-plane arrays, concatenated along the neuron axis) and the retinotopy file `retinotopy/<mouse>_<Y>_<M>_<D>_trans.npz` for `iarea`. All 99 behavior session dicts are held in RAM for the whole run; spike arrays are loaded one session at a time and `del`'d + `gc.collect()`'d.

This intersection yields **76 of the 89 recordings**. The 13 missing recordings are all "swap" test sessions, whose behavior is stored under the key `<session_id>_<stimtype>` (e.g. `TX88_2022_07_22_1_swap1`), so the exact-key match never finds them. The AI records them in metadata as `excluded_neural_sessions_without_behavior`, i.e. it concluded their behavior is absent, which is not the case.

ii.
```python
def load_behavior_sessions():
    out={}; source={}
    # Stable precedence. Duplicate entries represent alternate paper comparisons;
    # frame-aligned behavior itself is the same recording.
    for p in sorted(glob.glob(ROOT+'/beh/Beh_*.npy')):
        try: d=np.load(p,allow_pickle=True).item()
        except Exception: continue
        for sid,b in d.items():
            if isinstance(b,dict) and 'ft' in b and sid not in out:
                out[sid]=b; source[sid]=os.path.basename(p)
    return out,source
```
```python
beh,bsrc=load_behavior_sessions()
spk={os.path.basename(p).replace('_neural_data.npy',''):p for p in glob.glob(ROOT+'/spk/*_neural_data.npy')}
sids=sorted(set(beh)&set(spk),key=lambda s:(s.split('_')[0],sid_date(s),s))
```
```python
obj=np.load(spk[sid],allow_pickle=True).item()['spks']
n=min(len(b['ft']),min(x.shape[1] for x in obj))
X=np.concatenate([x[:,:n] for x in obj],axis=0)
rp=glob.glob(ROOT+'/retinotopy/'+('_'.join(sid.split('_')[:4]))+'_trans.npz')
```
```python
'excluded_neural_sessions_without_behavior':sorted(set(spk)-set(beh)),
```

iii. From the trajectory (steps 8–10): "We have 76 neural sessions with aligned behavioral data and 13 neural recordings without behavior, which cannot support the requested decoder variables and should be excluded with documentation." Earlier (step 6) it had observed that `Imaging_Exp_info.npy` has 142 entries for 89 unique recordings "because the same recording can contribute multiple stimulus comparisons", and decided to key off the behavior files directly with "stable precedence" rather than off the index. It never checked whether the 13 unmatched ids appear as a *prefix* of a behavior key.

## 1-b. How are the data split into subjects (mice)?

i. The mouse name is the first underscore-separated token of the session id. `subjects` is the sorted set of mouse names over the sessions actually written (19 mice), and `subject_idx` indexes into it per session. Sessions are sorted by mouse first so all sessions of a mouse are contiguous.

ii.
```python
subjects=sorted({s.split('_')[0] for s in sids}); subjmap={s:i for i,s in enumerate(subjects)}
```
```python
used_subjects=sorted({x['session_id'].split('_')[0] for x in session_info}); um={s:i for i,s in enumerate(used_subjects)}
data={... 'subjects':used_subjects,
      'subject_idx':np.asarray([um[x['session_id'].split('_')[0]] for x in session_info],dtype=np.int32), ...}
```

iii. Not discussed explicitly; the session id is self-describing (`mouse_YYYY_MM_DD_blk`), so the mouse is read straight off it. Note `subjects`/`subject_idx` are rebuilt from `session_info` after the loop, so a mouse that lost all its sessions would not appear (all 19 survived).

## 1-c. How are the data split into sessions?

i. A session is one `mouse_date_block` recording, i.e. one spike file. Where the same recording appears in several `Beh_*.npy` files (multiplicity up to 5), the first file in sorted filename order wins and the rest are ignored; the chosen file is recorded in `session_info['behavior_source']`. Sessions are ordered by `(mouse, date, session_id)`. A session is dropped if it ends up with fewer than 2 usable trials (none were dropped for this reason in the run). Because of the key-matching issue in 1-a, the 13 swap recordings are not present at all, so 76 sessions are written.

ii.
```python
if isinstance(b,dict) and 'ft' in b and sid not in out:
    out[sid]=b; source[sid]=os.path.basename(p)
```
```python
sids=sorted(set(beh)&set(spk),key=lambda s:(s.split('_')[0],sid_date(s),s))
```
```python
if len(sn)<2:
    print('skip',sid,'fewer than 2 usable trials',flush=True); continue
```

iii. Step 6: "There are 89 unique imaging sessions but 142 experiment-group entries because the same recording can contribute multiple stimulus comparisons." The code comment states the rationale for the de-duplication: "Duplicate entries represent alternate paper comparisons; frame-aligned behavior itself is the same recording." The `<2 trials` guard follows the instruction that "There needs to be at least two trials within each session in order to evaluate the decoder performance."

## 1-d. How are the data split into trials?

i. Trials are the `ntrials` trials the behavior declares, and a frame belongs to trial `tr` when `ft_trInd == tr`. Frames within a trial are further restricted to those that are **inside the textured corridor** (`ft_CorrSpc`), **moving** (`ft_isMoving`), and with a finite position in `[0, 40)` decimetres. All streams are first truncated to `n = min(len(ft), min plane frame count)`.

This means frames are removed *from the middle of trials*: in the session I checked (`TX108_2023_01_05_2`) the `ft_isMoving` filter removes 28.5% of the in-corridor frames and leaves 143 of 453 trials with at least one internal gap in frame index. The dataset-wide effect is visible in the verifier statistics: mean trial length 22.6 bins for the AI versus 32.6 for the expert reference.

ii.
```python
def trial_mask(b,tr,n):
    tri=np.asarray(b['ft_trInd'])[:n]
    pos=np.asarray(b['ft_Pos'])[:n]
    moving=np.asarray(b['ft_isMoving'],bool)[:n]
    corr=np.asarray(b['ft_CorrSpc'],bool)[:n]
    # Corridor positions are represented as 0..40 (source position unit = 0.1 m).
    return (tri==tr) & moving & corr & np.isfinite(pos) & (pos>=0) & (pos<40)
```
```python
for tr in range(int(b['ntrials'])):
    mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
    if ix.size<2: continue
```

iii. The module docstring and metadata say this "matches paper restriction to running in the textured 4-m corridor": `'frame_filter':'ft_isMoving AND ft_CorrSpc AND 0 <= ft_Pos < 40; matches paper restriction to running in the textured 4-m corridor'`. Step 7: "The paper explicitly uses Suite2p deconvolved fluorescence traces and restricts analyses to running timepoints." Step 13 notes the tension it spotted but did not resolve: "running speed includes many near-zero values even during VR movement; this is plausible because `ft_isMoving` refers to VR progression while `ft_RunSpeed` is measured ball speed."

## 1-e. How are trials filtered based on quality controls?

i. The only trial-level quality control is a minimum length: a trial is kept if at least 2 frames survive the mask of 1-d. There is no upper bound on trial duration and no outlier removal. The `ft_isMoving` filter caps the *number of bins* a stalled trial contributes (max trial length 178 bins versus 238 for the reference), but it does not cap the wall-clock span of a trial: the converted `time_since_trial_start` input reaches **1765 s** and `time_to_sound_cue` reaches **−1763 s** (reference: 74.8 s and ±73 s), i.e. trials where the mouse stood still for ~29 minutes are still present, stitched together from their scattered moving frames. 31,443 trials are written, against 37,728 for the reference.

ii.
```python
mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
if ix.size<2: continue
```
```python
session_info.append({... 'source_trials':int(b['ntrials']),'retained_trials':len(sn),
                     'retained_trial_indices':kept, ...})
```

iii. No explicit justification is given for the `>=2` threshold beyond the format requirement that a trial be a usable time series; the number of source and retained trials per session is recorded in metadata for transparency. The AI never discusses stalled/outlier trials in the trajectory — it appears to have assumed the `ft_isMoving` restriction handled them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy` — a list of one (neurons x frames) float32 array per imaging plane — concatenated along the neuron axis. The per-neuron visual area comes from `iarea` in `retinotopy/<mouse>_<Y>_<M>_<D>_trans.npz`, which indexes the concatenated neuron axis.

ii.
```python
obj=np.load(spk[sid],allow_pickle=True).item()['spks']
n=min(len(b['ft']),min(x.shape[1] for x in obj))
X=np.concatenate([x[:,:n] for x in obj],axis=0)
```
```python
rp=glob.glob(ROOT+'/retinotopy/'+('_'.join(sid.split('_')[:4]))+'_trans.npz')
if rp:
    ia=np.load(rp[0])['iarea']; ri=area_index(ia)
else: ri=np.full(X.shape[0],4,dtype=np.int8)
```

iii. Docstring: "Decisions follow the released analysis code where applicable: Suite2p deconvolved traces are concatenated over planes; retinotopic IDs are grouped with `utils.neu_area_ID`." Step 9: "The reference loader concatenates all imaging planes without neuron filtering, and maps retinotopy into V1, medial higher visual, lateral higher visual, and anterior higher visual areas."

## 2-b. How is the `neural` data processed?

i. No transformation of the values: the deconvolved traces are used as-is. Time columns are truncated to the shared minimum length `n`; per trial, the columns at the kept frame indices are copied out and cast to **float32**. Trials keep their own (variable) length; nothing is padded, smoothed, normalized, or z-scored. Metadata records `'neural_signal':'Suite2p non-negative deconvolved fluorescence trace'`.

ii.
```python
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```
```python
'time_bin_size':1000.0/FS, 'native_imaging_rate_hz':FS,
'neural_signal':'Suite2p non-negative deconvolved fluorescence trace',
```

iii. Step 7: "The paper explicitly uses Suite2p deconvolved fluorescence traces", so the stored traces need no further processing. The docstring explains why the paper's own 60-bin spatial interpolation is *not* applied: "Native imaging frames (3.17 Hz) are retained because this task requires temporal lick, cue-time, speed and elapsed-time variables (the paper's 60-bin spatial interpolation would discard their natural timing)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two things happen, only the first of which is a quality control:

1. Every neuron is assigned a region via the reference's `iarea` grouping (V1 = 8; mHV = 0,1,2,9; lHV = 5,6; aHV = 3,4). Neurons that fall outside those codes (`iarea` −1 or 7) are **not dropped** — they are put in a fifth region called `unassigned` and kept (17,685 of the 152,000 written neurons, ~11.6%).
2. Every session is then **randomly subsampled to 2,000 neurons** (`--max-neurons`, default 2000), with a region-proportional, seed-0 deterministic draw. Sessions average ~46,000 neurons in the source, so roughly **96% of the recorded neurons are discarded**. The written dataset has exactly 2,000 neurons in every session (reference: mean 46,128, min 17,363, max 78,815; 4.1M neurons total).

If `len(iarea) != n_neurons` the code prints a warning and `np.resize`s the region vector, which would silently misassign every neuron's region (it does not trigger on this dataset — `iarea` length equals the concatenated neuron count).

ii.
```python
def area_index(ia):
    ia=np.asarray(ia); z=np.full(ia.shape,4,dtype=np.int8)
    z[ia==8]=0
    z[np.isin(ia,[0,1,2,9])]=1
    z[np.isin(ia,[5,6])]=2
    z[np.isin(ia,[3,4])]=3
    return z
```
```python
# Deterministic proportional stratified sample, preserving every region.
if a.max_neurons and X.shape[0]>a.max_neurons:
    rng=np.random.default_rng(0)
    chosen=[]
    for r in range(len(REGIONS)):
        ids=np.flatnonzero(ri==r)
        k=int(round(a.max_neurons*len(ids)/len(ri)))
        if len(ids) and k==0: k=1
        chosen.extend(rng.choice(ids,min(k,len(ids)),replace=False).tolist())
    if len(chosen)>a.max_neurons: chosen=chosen[:a.max_neurons]
    ...
    chosen=np.sort(np.asarray(chosen)); X=X[chosen]; ri=ri[chosen]
```
```python
'neuron_selection':f'deterministic region-stratified sample capped at {a.max_neurons} neurons/session; reference plane concatenation retained before sampling',
```

iii. Step 12: "The decoder processes one session at a time but creates a trainable projection for every neuron, so preserving 20,000–90,000 neurons across 76 sessions would make both the pickle and model impractically large. A deterministic cap of 2,000 neurons per session matches the decoder's own SVD projection threshold and should be stratified across visual regions." Step 10 had initially said the opposite — "The conversion should nevertheless follow the reference loader and preserve all neurons unless execution proves infeasible" — but infeasibility was never demonstrated: the AI's own diagnostic reported 3,649 GB free on `/app`, and the `svd_max_neurons=2000` value it cites is only the size of a random projection used to *initialize* the SVD, not a cap on the decoder's inputs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to corridor entry (trial start): a trial's first column is its first kept in-corridor frame and the trial runs to its last kept in-corridor frame. Trials are variable length, nothing is padded or truncated to a common window. Metadata declares `temporal_alignment_event = 'trial start (entry into textured corridor)'`, `off_start = 0.0`, `off_end = None`.

ii.
```python
mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
...
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```
```python
'temporal_alignment_event':'trial start (entry into textured corridor)','off_start':0.0,'off_end':None,
```

iii. Metadata: "Behavior arrays are supplied at neural-frame timestamps; neural and behavior streams truncated to shared minimum length (typically neural is one frame shorter)." All streams are indexed with the same `ix`, so alignment across neural/input/output is exact by construction. Note the caveat from 1-d: because non-moving frames are removed, successive columns of a trial are not necessarily successive imaging frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. The native imaging frame is the bin; `FS = 3.17` Hz is hard-coded, giving `time_bin_size = 1000/3.17 = 315.46 ms` — identical to the expert reference. The paper's spatial (60-bin) interpolation is explicitly rejected. Caveat: because non-moving frames are dropped inside trials, the *actual* spacing between consecutive columns is 315.46 ms only for the ~68% of frame pairs with no intervening dropped frame; the declared bin size is nominal rather than exact for gapped trials.

ii.
```python
ROOT='/app/data'; FS=3.17
```
```python
'time_bin_size':1000.0/FS, 'native_imaging_rate_hz':FS,
```

iii. Docstring: "Native imaging frames (3.17 Hz) are retained because this task requires temporal lick, cue-time, speed and elapsed-time variables (the paper's 60-bin spatial interpolation would discard their natural timing)." Step 5 established the 3.17 Hz rate from the reference notebook.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the MATLAB datenum timestamp of the cue on each trial) and `ft` (the datenum timestamp of every imaging frame). It does not use the frame-number form `SoundFr` that the expert reference interpolates; `SoundTime` and `SoundFr` are the same event expressed in different units.

ii.
```python
ft=np.asarray(b['ft'],float)[:n]
...
cue=float(np.asarray(b['SoundTime'])[tr])
tocue=(cue-ft[ix])*86400.0
```

iii. Not discussed beyond the field survey in steps 6 and 10, which listed `SoundTime`/`Trial_start_time` as float datenum arrays of length `ntrials` and `ft` as the matching per-frame datenum axis. Using the time fields directly avoids the interpolation step the reference needs.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. `(SoundTime[trial] - ft[frame]) * 86400`, i.e. seconds, **positive before the cue and negative after**, matching the "time *to* cue" sense used by the expert. It is stored as a time-varying float32 row. No clipping, no binary-event encoding. Because trials with long stalls survive (see 1-e), the range is −1763 s to +403 s.

ii.
```python
elapsed=(ft[ix]-t0)*86400.0; tocue=(cue-ft[ix])*86400.0
inp=np.vstack([tocue,np.full(ix.size,day),elapsed,np.full(ix.size,int(np.asarray(b['isRew'])[tr]))]).astype(np.float32)
```
```python
'input_names':['time_to_sound_cue_s','training_day_elapsed','time_since_trial_start_s','reward_available'],
```

iii. The AI treats the timestamps as MATLAB datenums (days) and multiplies by 86400 to get seconds; the input name carries the `_s` unit suffix. The docstring justifies keeping cue time as a continuous time-varying variable rather than a spatial or per-trial value: the task "requires temporal lick, cue-time, speed and elapsed-time variables".

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at `ft[ix]`, the timestamps of exactly the frames whose neural columns were taken, so it is aligned by construction, with the same per-trial length.

ii.
```python
mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
...
tocue=(cue-ft[ix])*86400.0
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. Metadata: "Behavior arrays are supplied at neural-frame timestamps; neural and behavior streams truncated to shared minimum length." Every stream of a trial is indexed with the same `ix`.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date embedded in the session id (`mouse_YYYY_MM_DD_blk`), parsed to a `datetime.date`. It deliberately does **not** use the `sess#` field of `Imaging_Exp_info.npy`.

ii.
```python
def sid_date(s): return datetime.datetime.strptime('_'.join(s.split('_')[1:4]),'%Y_%m_%d').date()
```
```python
firstdate={m:min(sid_date(s) for s in sids if s.split('_')[0]==m) for m in subjects}
```

iii. Step 11: "The experiment table's `sess#` is a stage index rather than an unambiguous chronological day because the same recording is reused for multiple test comparisons. A robust continuous training-day variable should therefore be derived from recording date relative to each subject's first included recording." Its own diagnostic (step 10) had found 12 recordings with conflicting `sess#` values across experiment groups, which motivated this.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Elapsed **calendar days** between the session's date and the mouse's earliest *included* session date, as a float, broadcast constant across all bins of every trial of that session. Range across the dataset: 0 to 89 days. (The expert reference instead uses the ordinal index of the recording day, 0 to 7.) Because the origin is the first *included* session, dropping the 13 swap sessions could in principle shift a mouse's origin.

ii.
```python
day=float((sid_date(sid)-firstdate[sid.split('_')[0]]).days)
inp=np.vstack([tocue,np.full(ix.size,day),elapsed,...]).astype(np.float32)
```
```python
'training_day_definition':'elapsed calendar days from each subject first included imaging recording',
```

iii. Step 11, as quoted in 4-a: a date-relative continuous variable is described as "robust" against the ambiguity of `sess#`, and the definition is recorded verbatim in metadata. It is broadcast per timepoint (rather than stored as a per-trial scalar) so every input row has the same shape.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (per-trial datenum of corridor entry) and `ft`. I verified this is the same event the expert's `StartFr` encodes: for `TX108_2023_01_05_2`, `(Trial_start_time[0] - ft[0]) * 86400 / 0.3155 = 9.35` frames against `StartFr[0] = 9.338`.

ii.
```python
t0=float(np.asarray(b['Trial_start_time'])[tr])
```

iii. Not separately justified; `Trial_start_time` is the natural per-trial timestamp counterpart to the frame-number field `StartFr`, and it is already in the same datenum units as `ft`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `(ft[frame] - Trial_start_time[trial]) * 86400`, i.e. seconds since corridor entry, positive after start, stored as a time-varying float32 row. Minimum over the dataset is 6.0e-5 s (the first kept frame of a trial is essentially at entry) and maximum is 1765 s for the stalled trials discussed in 1-e.

ii.
```python
elapsed=(ft[ix]-t0)*86400.0
inp=np.vstack([tocue,np.full(ix.size,day),elapsed,np.full(ix.size,int(np.asarray(b['isRew'])[tr]))]).astype(np.float32)
```

iii. Same rationale as 3-b: datenum-to-seconds conversion and a difference against the trial's own start; the `_s` suffix in `input_names` documents the unit.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Evaluated at `ft[ix]` — the same frame indices as the neural columns — so it is aligned by construction and has the same length as the trial's neural matrix.

ii.
```python
elapsed=(ft[ix]-t0)*86400.0
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. All streams share the single index vector `ix`; metadata states that behavior arrays are supplied at neural-frame timestamps.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From the per-trial boolean `isRew`.

ii.
```python
np.full(ix.size,int(np.asarray(b['isRew'])[tr]))
```

iii. Not discussed; `isRew` directly marks trials run in the rewarded corridor, which is exactly what the instruction asks for.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to `int` (0/1) and broadcast constant across all bins of the trial, then stored as a float32 row named `reward_available`. No further processing. Observed range over the dataset is 0 to 1.

ii.
```python
inp=np.vstack([tocue,np.full(ix.size,day),elapsed,np.full(ix.size,int(np.asarray(b['isRew'])[tr]))]).astype(np.float32)
```

iii. No justification given — none needed, the flag is already the requested variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the per-trial string naming the wall texture. (`TrialStim` is not used; it is masked in swap sessions.)

ii.
```python
categories=set()
for sid in sids:
    b=beh[sid]; categories.update(map(str,np.asarray(b['WallName']).reshape(-1)))
categories=sorted(categories); catmap={x:i for i,x in enumerate(categories)}
```
```python
name=str(np.asarray(b['WallName'])[tr])
```

iii. Not discussed explicitly; `WallName` was identified in the step-10 field survey, which also counted the 15 distinct names present across all behavior files.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. A **global vocabulary** of the raw wall-name strings is built in a first pass over the included sessions, sorted alphabetically, and each trial's name is mapped to its index; the index is broadcast across the trial's bins as int8. The names are used verbatim, so crops and spatial shuffles of the same texture become **separate categories**: the vocabulary has 13 entries — `circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood2, wood5` — where the expert collapses the same names onto 4 base textures (`circle, leaf, rock, wood`). The resulting class distribution is very uneven: two classes hold 0.23% and 0.25% of the bins.

ii.
```python
out=np.vstack([np.full(ix.size,catmap[name]),lick[ix],np.clip((pos//10).astype(int),0,3),np.digitize(speed,speed_edges)]).astype(np.int8)
```
```python
'output_values':[categories,['not_licking','licking'],['0-1 m','1-2 m','2-3 m','3-4 m'],['Q1 (slowest)','Q2','Q3','Q4 (fastest)']],
```

iii. The code comment on the first pass gives the reason for making it global: "First pass: global category vocabulary and running-speed quartiles over all retained moving corridor frames (not per-session, ensuring common classes)." There is no statement in the trajectory about whether variants should be collapsed to base textures; step 11 simply notes "Visual categories span circle, leaf, rock, and wood variants" and the code then keeps the variants.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickTime`, the datenum timestamp of each lick, together with `ft`. (The equivalent frame-number field `LickFr`, used by the expert, is not used.) Sessions with no licking have `LickTime` as an empty `uint8` array, which the guard handles.

ii.
```python
lt=np.asarray(b['LickTime']).reshape(-1)
if lt.size and np.issubdtype(lt.dtype,np.number):
    lt=lt[np.isfinite(lt.astype(float))].astype(float)
```

iii. Not discussed in the trajectory; the field survey in step 10 showed `LickTime` alongside the other datenum event fields, and the AI used the time-domain fields uniformly for cue, trial start and licks.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A per-frame binary flag over the whole session: each lick is assigned to its **nearest** imaging frame (searchsorted, then compare the two neighbouring frame times), duplicates collapsed with `np.unique`, and the flag set to 1. The expert instead truncates the fractional `LickFr` to the frame it lands in, which differs from the AI's rounding by at most half a bin (158 ms). Frame indices are clipped into `[1, n-1]`, so a lick recorded after the last imaged frame is attributed to the last frame rather than dropped. Resulting lick fraction: 3.55% of bins (reference: 4.14%).

ii.
```python
# Assign each lick to its nearest imaging frame.
lick=np.zeros(n,dtype=np.int8)
lt=np.asarray(b['LickTime']).reshape(-1)
if lt.size and np.issubdtype(lt.dtype,np.number):
    lt=lt[np.isfinite(lt.astype(float))].astype(float)
    q=np.searchsorted(ft,lt); q=np.clip(q,1,n-1)
    q-=((lt-ft[q-1]) <= (ft[q]-lt)).astype(int)
    lick[np.unique(q)]=1
```
```python
'lick_binning':'each lick assigned to nearest native imaging frame',
```

iii. The code comment and the metadata key `lick_binning` state the rule; the AI's stated reason for keeping licking as a native-rate binary time series is in the docstring ("this task requires temporal lick ... variables"), matching the instruction that a behavioural event be represented as a binary time series.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The flag is built on the session's imaging-frame grid (truncated to `n`) and then indexed with the trial's `ix`, the same indices used for the neural columns.

ii.
```python
out=np.vstack([np.full(ix.size,catmap[name]),lick[ix],...]).astype(np.int8)
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. The lick times are mapped onto `ft`, which is the neural frame time axis, so the flag is on the neural grid before any trial indexing happens.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the per-imaging-frame corridor position in decimetres (0–40 across the texture, up to ~60 through the grey space).

ii.
```python
pos=np.asarray(b['ft_Pos'],float)[:n][ix]
```

iii. Code comment in `trial_mask`: "Corridor positions are represented as 0..40 (source position unit = 0.1 m)." Metadata: `'position_conversion':'Source ft_Pos units are 0.1 m; bins [0,10), [10,20), [20,30), [30,40).'`

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Integer division by 10 (decimetres → metres), clipped to `[0, 3]`, stored as int8 and broadcast nowhere (it is already per-frame). Frames outside `[0, 40)` never reach this point because `trial_mask` excludes them.

ii.
```python
np.clip((pos//10).astype(int),0,3)
```

iii. Metadata `position_conversion`, as quoted above. The resulting distribution is near-uniform (0.2505 / 0.2484 / 0.2492 / 0.2519), matching the reference closely.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four fixed, equal-length 1 m bins with edges at 0, 1, 2, 3, 4 m, labelled `['0-1 m','1-2 m','2-3 m','3-4 m']` — exactly the discretization the instruction specifies, and identical to the expert's.

ii.
```python
'output_values':[categories,['not_licking','licking'],['0-1 m','1-2 m','2-3 m','3-4 m'],['Q1 (slowest)','Q2','Q3','Q4 (fastest)']],
```
```python
np.clip((pos//10).astype(int),0,3)
```

iii. Directly from the Decoder Task spec: "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins".

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is already one value per imaging frame; it is truncated to `n` and indexed with the trial's `ix`, the same indices as the neural columns.

ii.
```python
pos=np.asarray(b['ft_Pos'],float)[:n][ix]
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. Metadata: "Behavior arrays are supplied at neural-frame timestamps"; all `ft_*` streams share the neural frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the per-imaging-frame running speed (ball speed).

ii.
```python
speed=np.asarray(b['ft_RunSpeed'],float)[:n][ix]
```
```python
v=np.asarray(b['ft_RunSpeed'],float)[:n][ok]; speeds.append(v[np.isfinite(v)])
```

iii. Not discussed beyond the field survey. Step 13 records the AI's observation that `ft_RunSpeed` can be near zero even when `ft_isMoving` is true, "because `ft_isMoving` refers to VR progression while `ft_RunSpeed` is measured ball speed".

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A **first pass** over all included sessions collects `ft_RunSpeed` at every frame that passes the retention mask, concatenates them, and computes three **global** quantile edges (0.25/0.50/0.75) with `np.quantile`: 12.19, 25.21, 40.60. In the main pass each trial's speeds are assigned with `np.digitize` against those global edges. This is a single global rule, unlike the expert's per-session rank-based split.

ii.
```python
categories=set(); speeds=[]
for sid in sids:
    ...
    ok=np.isfinite(tri)&np.asarray(b['ft_isMoving'],bool)[:n]&np.asarray(b['ft_CorrSpc'],bool)[:n]&np.isfinite(pos)&(pos>=0)&(pos<40)
    v=np.asarray(b['ft_RunSpeed'],float)[:n][ok]; speeds.append(v[np.isfinite(v)])
speed_edges=np.quantile(np.concatenate(speeds),[.25,.5,.75]); del speeds
```
```python
np.digitize(speed,speed_edges)
```
```python
'speed_quartile_edges':speed_edges.tolist(),
'speed_quartile_scope':'all retained timepoints across included sessions',
```

iii. Code comment: "First pass: global category vocabulary and running-speed quartiles over all retained moving corridor frames (not per-session, ensuring common classes)." Step 13: "the global quartile edges should be checked on the full cohort"; step 29 confirms "Global running-speed quartiles are sensible in the full cohort (12.19, 25.21, 40.60)." The consequence, visible in the verifier statistics, is that individual sessions are far from 25%/25%/25%/25% — the lowest-quartile share ranges from 0.038 to 0.955 across sessions, where the expert's per-session split gives 0.2500 in every bin.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins defined by the three global quartile edges, `np.digitize` giving 0–3, labelled `['Q1 (slowest)','Q2','Q3','Q4 (fastest)']`. Globally each bin holds ~25% of the retained timepoints; per session it does not. Because the non-moving frames were already removed at the trial-masking stage, the large mass of exactly-zero speeds that would otherwise make a value threshold degenerate is mostly absent, so `digitize` ties are not a practical problem here.

ii.
```python
speed_edges=np.quantile(np.concatenate(speeds),[.25,.5,.75])
...
np.digitize(speed,speed_edges)
```

iii. Follows the Decoder Task spec "Running speed discretized into 4 bins, each corresponding to 25% of the data", with "the data" read as the pooled dataset rather than each session; the scope is documented in metadata as `'speed_quartile_scope':'all retained timepoints across included sessions'`.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is one value per imaging frame; it is truncated to `n` and indexed with the trial's `ix`, the same indices as the neural columns.

ii.
```python
speed=np.asarray(b['ft_RunSpeed'],float)[:n][ix]
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. Same as the other `ft_*` streams: they are supplied on the neural frame grid, so a shared index vector aligns them.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several guards:
- **Behaviour longer than imaging**: all streams are truncated to `n = min(len(ft), min over planes of frames)`. The AI verified the neural axis is typically one frame shorter than `ft`.
- **Unlabelled frames**: `ft_trInd` is NaN outside trials; `tri==tr` is False for NaN, and the first pass uses `np.isfinite(tri)` explicitly. Non-finite or out-of-range `ft_Pos` is excluded by the mask.
- **Empty lick arrays**: `LickTime` is an empty `uint8` array in sessions without licking; guarded by `lt.size and np.issubdtype(lt.dtype,np.number)`, and non-finite lick times are filtered out.
- **Licks outside the imaging window**: clipped into `[1, n-1]` rather than dropped, so a post-imaging lick is attributed to the last frame (the expert drops these).
- **Missing retinotopy file**: all neurons fall back to region `unassigned`.
- **Region/neuron count mismatch**: a warning is printed and `np.resize` is used, which would silently produce wrong region labels for every neuron. It does not fire on this dataset (`iarea` length equals the concatenated neuron count).
- **Degenerate sessions**: a session with fewer than 2 usable trials is skipped with a message. A behaviour file that fails to unpickle is skipped silently (`except: continue`).
- **Spike files with no matching behaviour key**: excluded and listed in metadata — but this guard is what silently loses the 13 swap sessions (see 1-a).

ii.
```python
n=min(len(b['ft']),min(x.shape[1] for x in obj))
X=np.concatenate([x[:,:n] for x in obj],axis=0)
```
```python
if len(ri)!=X.shape[0]:
    print('WARNING region/neuron mismatch',sid,len(ri),X.shape[0],flush=True)
    ri=np.resize(ri,X.shape[0]).astype(np.int8)
```
```python
q=np.searchsorted(ft,lt); q=np.clip(q,1,n-1)
```
```python
if len(sn)<2:
    print('skip',sid,'fewer than 2 usable trials',flush=True); continue
```

iii. Step 10: "Neural time axes are one sample shorter than behavior frame arrays, so all aligned streams must be truncated to the shared minimum length." Metadata records the alignment rule and the excluded session list. No justification is given for the `np.resize` fallback or the lick clipping.

## 12-a. What are the most time-consuming steps of the code?

i. By far the dominant cost is reading and unpickling the ~434 GB of `spk/*_neural_data.npy` files — one `np.load(..., allow_pickle=True)` per session, whose result is then `np.concatenate`d into a single (n_neurons x n_frames) array (a second full copy in RAM). The trajectory shows the full run occupying steps 14–28, i.e. roughly 7–14 minutes of pure waiting, with no per-session progress until the log was read. Note the cost is paid in full even though 96% of the loaded neurons are immediately thrown away. Secondary costs: the first pass over all behaviour files (`load_behavior_sessions` unpickles every `Beh_*.npy` and keeps all 99 session dicts resident), and pickling the 5.5 GB output.

ii.
```python
obj=np.load(spk[sid],allow_pickle=True).item()['spks']
n=min(len(b['ft']),min(x.shape[1] for x in obj))
X=np.concatenate([x[:,:n] for x in obj],axis=0)
del obj
```
```python
for p in sorted(glob.glob(ROOT+'/beh/Beh_*.npy')):
    try: d=np.load(p,allow_pickle=True).item()
```

iii. Step 8: "Full-neuron conversion across 89 sessions would be prohibitively large"; step 14: "reading 434 GB of source neural arrays may take several minutes"; step 24: "consistent with reading and deserializing the full 434 GB neural archive". The AI mitigated memory (not time) with `del obj`, `del X` and `gc.collect()` per session.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop calls `trial_mask(b, tr, n)`, which rebuilds **five full-session arrays** (`ft_trInd[:n]`, `ft_Pos[:n]`, `ft_isMoving[:n]`, `ft_CorrSpc[:n]`, plus the finite/range comparisons) and does a full-length boolean AND and `flatnonzero` for **every trial**. With ~450 trials and ~23,000 frames that is ~10M element operations per array per session, where a single pass grouping frames by `ft_trInd` (e.g. `np.argsort`/`np.searchsorted` on the trial index, with the static part of the mask computed once outside the loop) would do the same work in O(n_frames). The same pattern repeats for `ft_Pos` and `ft_RunSpeed`, which are re-sliced with `np.asarray(...)[:n]` inside the trial loop. The neuron-selection loop over 5 regions and the per-trial `np.vstack` calls are negligible by comparison. All of this is still small next to the I/O cost of 12-a.

ii.
```python
for tr in range(int(b['ntrials'])):
    mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
    if ix.size<2: continue
    name=str(np.asarray(b['WallName'])[tr]); pos=np.asarray(b['ft_Pos'],float)[:n][ix]
    speed=np.asarray(b['ft_RunSpeed'],float)[:n][ix]
```

iii. No justification is offered — the AI never discusses the cost of the trial loop, presumably because it is dwarfed by the neural file reads.

## 12-c. What processing does the code repeat multiple times?

i. Three clear repetitions:
1. The static part of the frame mask (`ft_isMoving & ft_CorrSpc & isfinite(pos) & 0<=pos<40`) is recomputed once per trial inside `trial_mask` instead of once per session (see 12-b).
2. That same static mask is computed a **second time**, verbatim but inline, in the first pass that collects running speeds — the same expression written twice in two places, which is also a maintenance hazard (the two copies differ: the first pass uses `n = len(b['ft'])`, the main pass uses the shorter neural-truncated `n`, so the speed quartiles are computed over a slightly different frame set than the one written out).
3. `np.asarray(b['ft_Pos'], float)[:n]` and `np.asarray(b['ft_RunSpeed'], float)[:n]` are re-created from the source arrays on every trial iteration.

ii.
```python
ok=np.isfinite(tri)&np.asarray(b['ft_isMoving'],bool)[:n]&np.asarray(b['ft_CorrSpc'],bool)[:n]&np.isfinite(pos)&(pos>=0)&(pos<40)
```
```python
return (tri==tr) & moving & corr & np.isfinite(pos) & (pos>=0) & (pos<40)
```

iii. Not discussed. The duplication is a side effect of needing a global speed distribution before the main conversion pass; the AI justified the two-pass design (step 12: "a fully reproducible converter with a two-pass design: collect matched sessions and global speed quartiles, then stream neural files one at a time") but not the duplicated mask code.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **The largest by far**: every session's full neuron set (~20,000–90,000 neurons) is loaded, truncated and `np.concatenate`d into one array, and then ~96% of it is discarded by the 2,000-neuron subsample. The subsample indices are known before the concatenation and could be applied per plane, avoiding the full copy — and, more importantly, the discarded neurons are data the downstream decoder could have used.
- **Redundant mask terms**: `np.isfinite(pos) & (pos>=0) & (pos<40)` is already implied by `ft_CorrSpc` — in the session I checked, zero `ft_CorrSpc` frames have `pos >= 40` — so those three comparisons over the full frame axis, per trial, never change the result.
- **All 99 behaviour session dicts are retained in memory** for the whole run, including the 23 that are never converted.
- Minor: `np.unique(q)` before the lick assignment (assigning 1 twice is harmless), the `area_index` computation over all neurons before subsampling, and the `retained_trial_indices` list per session, which is metadata only.

ii.
```python
X=np.concatenate([x[:,:n] for x in obj],axis=0)
del obj
...
if a.max_neurons and X.shape[0]>a.max_neurons:
    ...
    chosen=np.sort(np.asarray(chosen)); X=X[chosen]; ri=ri[chosen]
```
```python
return (tri==tr) & moving & corr & np.isfinite(pos) & (pos>=0) & (pos<40)
```

iii. The neuron cap is justified in step 12 on size grounds ("preserving 20,000–90,000 neurons across 76 sessions would make both the pickle and model impractically large"), and metadata notes "reference plane concatenation retained before sampling" — i.e. the AI deliberately concatenated first to stay faithful to the reference loader's neuron ordering before drawing its sample. The redundant position terms are presented in the code comment as a unit clarification ("Corridor positions are represented as 0..40"), i.e. defensive rather than necessary.
