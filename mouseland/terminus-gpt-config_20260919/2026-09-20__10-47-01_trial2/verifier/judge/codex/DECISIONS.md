# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master experiment index from `beh/Imaging_Exp_info.npy`, then eagerly loads every `Beh_<exp>.npy` file and deduplicates physical sessions by `mouse_date_block`. It checks that the resulting behavior session ids exactly match the spike filenames. During per-session conversion it loads `spks` from the session neural file and later loads retinotopy for that session.

ii. 
```python
info = np.load(DATA/'beh/Imaging_Exp_info.npy', allow_pickle=True).item()
for exp, rows in info.items():
    d = np.load(DATA/f'beh/Beh_{exp}.npy', allow_pickle=True).item()
```
```python
neural_ids = {p.name.removesuffix('_neural_data.npy') for p in (DATA/'spk').glob('*_neural_data.npy')}
if neural_ids != set(beh):
    raise ValueError(...)
```
```python
raw=np.load(DATA/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
```

iii. In `CONVERSION_NOTES.md`, the AI says duplicate experiment memberships point to identical behavior and that conversion should deduplicate by physical session while ensuring exact neural/behavior id agreement.

## 1-b. How are the data split into subjects?

i. Subjects are split by the mouse prefix of the session id. The AI builds `subjects` from `sid.split('_')[0]` and `subject_idx` from the same rule.

ii. 
```python
subjects=sorted({x.split('_')[0] for x in ids})
subidx={x:i for i,x in enumerate(subjects)}
...
'subject_idx':np.array([subidx[x.split('_')[0]] for x in ids],dtype=np.int16),
```

iii. The notes say the released metadata already encode the mouse name in the session id, so no further inference is needed.

## 1-c. How are the data split into sessions?

i. A session is one physical recording identified as `<mouse>_<date>_<block>`. The AI deduplicates sessions on that id even if they appear under multiple experiment types.

ii. 
```python
sid = f"{r['mname']}_{r['datexp']}_{r['blk']}"
...
if sid in beh:
    if int(beh[sid]['ntrials']) != int(d[hit]['ntrials']):
        raise ValueError(f'Conflicting duplicate behavior for {sid}')
else:
    beh[sid] = d[hit]
```

iii. The notes repeatedly justify this as physical-session deduplication: experiment dictionaries are treated as alternative labels on the same recording, not separate sessions.

## 1-d. How are the data split into trials?

i. Trials are split by the framewise trial index `ft_trInd`. For each trial, the AI keeps frames whose trial id equals that trial and whose position is finite and inside the 0–4 m corridor (`0 <= ft_Pos < 40`).

ii. 
```python
tri=np.asarray(b['ft_trInd'])[:nfr]
pos=np.asarray(b['ft_Pos'],float)[:nfr]
...
for tr in range(int(b['ntrials'])):
    ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
```

iii. The notes justify this as aligning to corridor entry and restricting to the texture corridor only, using trial-assigned frames rather than `GrayFr` or a spatial interpolation workflow.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply the reference long-trial filter. It effectively requires only that each trial have at least two retained corridor frames; otherwise it raises an error. In practice its notes say all 38,110 trials survive.

ii. 
```python
ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
if len(ix)<2: raise ValueError(f'{sid} trial {tr}: only {len(ix)} corridor frames')
```

iii. The notes explicitly argue against dropping long stopped trials and state that no source quality flag justifies further trial filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `spks` in each session neural file. Brain-region indexing is derived from retinotopy `iarea`.

ii. 
```python
raw=np.load(DATA/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
spk=np.concatenate(raw,axis=0)
```
```python
with np.load(rp) as z: ia = np.asarray(z['iarea'])
```

iii. The notes say the provided `spks` are already the deconvolved neural signal used by the paper, so no alternate raw fluorescence stream is introduced.

## 2-b. How is the `neural` data processed?

i. The AI concatenates the three `spks` arrays across neurons, truncates to the common imaged frame count, slices per-trial corridor frames, and stores each trial in `float16`. It does not compute dF/F or resample in time.

ii. 
```python
nfr=min(a.shape[1] for a in raw)
spk=np.concatenate(raw,axis=0)
nfr=min(nfr,spk.shape[1],len(b['ft_trInd']))
spk=spk[:,:nfr]
...
neural.append(np.asarray(spk[:,ix],dtype=np.float16,order='C'))
```

iii. The notes justify using the native deconvolved traces and casting to `float16` as a storage optimization for a very large dataset.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not drop neurons outside the four named visual areas. Instead it preserves them as a fifth `"unmapped"` region class.

ii. 
```python
REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'unmapped']
...
out=np.full(nneurons,4,dtype=np.int16)
out[ia==8]=0; out[np.isin(ia,[0,1,2,9])]=1; out[np.isin(ia,[5,6])]=2; out[np.isin(ia,[3,4])]=3
return out
```

iii. The notes explicitly argue that retinotopy label 7 should be retained rather than silently discarded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns each trial to corridor entry, operationalized as the first retained imaging frame assigned to that trial in the 0–4 m corridor. Trials remain variable-length.

ii. 
```python
for tr in range(int(b['ntrials'])):
    ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
    ...
    neural.append(np.asarray(spk[:,ix],dtype=np.float16,order='C'))
```
```python
'temporal_alignment_event':'corridor entry (first imaging frame assigned to trial; ceil StartFr)',
'off_start':0.0,'off_end':None,
```

iii. The notes say `GrayFr` is corridor exit, not entry, and defend using first trial-assigned corridor frame / `ceil(StartFr)` as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps native imaging frames, does not temporally rebin, and reports a representative bin size from the median per-session frame interval.

ii. 
```python
ft=np.asarray(b['ft'],float)[:nfr]*86400.0
dt=float(np.median(np.diff(ft)))
```
```python
'time_bin_size':float(np.median(dts)*1000)
```

iii. The notes justify this by saying the source already lives on the imaging-frame grid and has small timestamp jitter, so native frames should be preserved.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives it from `SoundFr`, the trial-local retained frame indices `ix`, and a session-level median frame interval `dt` estimated from `ft`.

ii. 
```python
ft=np.asarray(b['ft'],float)[:nfr]*86400.0
dt=float(np.median(np.diff(ft)))
sound=np.asarray(b['SoundFr'],float)
...
cue=((sound[tr]-ix.astype(float))*dt).astype(np.float32)
```

iii. The notes describe this as a signed seconds-to-cue variable computed from cue frame coordinates on the native frame grid.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, the AI subtracts the frame index from `SoundFr` and multiplies by the session median frame duration, producing positive values before the cue and negative values after it.

ii. 
```python
cue=((sound[tr]-ix.astype(float))*dt).astype(np.float32)
...
inp=np.vstack([cue, np.full(len(ix),day,np.float32), elapsed,
               np.full(len(ix),float(rew[tr]),np.float32)]).astype(np.float32)
```

iii. The notes justify using frame differences times median `dt` because the inputs remain on the imaging-frame grid and preserve the cue sign convention.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the exact retained frame indices `ix` used to slice the neural array for that trial.

ii. 
```python
ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
cue=((sound[tr]-ix.astype(float))*dt).astype(np.float32)
...
neural.append(np.asarray(spk[:,ix],dtype=np.float16,order='C'))
```

iii. The notes say all streams are kept on the same frame grid after behavior-to-neural truncation.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session id date string, parsed from `<mouse>_<YYYY>_<MM>_<DD>_<block>`.

ii. 
```python
dates = {s: datetime.strptime('_'.join(s.split('_')[1:4]), '%Y_%m_%d') for s in session_ids}
```

iii. The notes say the date embedded in the session id is the cleanest source for a continuous training-day proxy.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes elapsed calendar days from each subject’s first recorded session, then repeats that scalar across all timepoints in the trial.

ii. 
```python
for s,d in dates.items():
    mouse=s.split('_')[0]; first[mouse]=min(first.get(mouse,d),d)
return {s: float((d-first[s.split('_')[0]]).days) for s,d in dates.items()}
```
```python
inp=np.vstack([cue, np.full(len(ix),day,np.float32), elapsed,
               np.full(len(ix),float(rew[tr]),np.float32)]).astype(np.float32)
```

iii. The notes explicitly justify elapsed calendar days rather than ordinal session count, arguing that it is continuous and preserves gaps between sessions.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from frame timestamps `ft` and the first retained corridor frame of the trial, not directly from `StartFr`.

ii. 
```python
ft=np.asarray(b['ft'],float)[:nfr]*86400.0
...
elapsed=(ft[ix]-ft[ix[0]]).astype(np.float32)
```

iii. The notes say corridor entry should be the first retained frame assigned to the trial.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI re-zeros the actual frame timestamps at the first retained corridor frame, so the first retained bin is exactly 0 s and later bins are elapsed native imaging time.

ii. 
```python
# Actual timestamps can have small jitter; re-zero at corridor entry.
elapsed=(ft[ix]-ft[ix[0]]).astype(np.float32)
```

iii. The notes justify this as preserving small timestamp jitter while aligning time zero to corridor entry.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same retained frame indices `ix` that are used for the trial’s neural slice.

ii. 
```python
elapsed=(ft[ix]-ft[ix[0]]).astype(np.float32)
...
neural.append(np.asarray(spk[:,ix],dtype=np.float16,order='C'))
```

iii. The notes state that all per-trial variables are built on the same truncated neural frame grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial behavior field `isRew`.

ii. 
```python
rew=np.asarray(b['isRew'],bool)
...
np.full(len(ix),float(rew[tr]),np.float32)
```

iii. The notes describe this as indicating whether the corridor was rewarded, independent of actual licking or reward delivery.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` value is cast to float and repeated across all timepoints of the trial.

ii. 
```python
inp=np.vstack([cue, np.full(len(ix),day,np.float32), elapsed,
               np.full(len(ix),float(rew[tr]),np.float32)]).astype(np.float32)
```

iii. The notes treat this as a per-trial decoder input that should be broadcast over time.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`.

ii. 
```python
wall=np.asarray(b['WallName']).astype(str)
stim_to_idx={x:i for i,x in enumerate(STIMULI)}
```

iii. The notes justify this by saying `WallName` is complete for all trials whereas `TrialStim` contains placeholders.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI maps each concrete wall identity to one of 15 global class ids and repeats that id across the whole trial. It does not collapse to four broad texture categories.

ii. 
```python
STIMULI = ['circle1','circle2','circle3','leaf1','leaf1_swap1','leaf1_swap2',
           'leaf2','leaf3','rock1','rock2','wood1','wood1_swap1','wood1_swap2','wood2','wood5']
...
out=np.vstack([np.full(len(ix),stim_to_idx[wall[tr]],np.int16),lick,pbin,sbin])
```

iii. The notes explicitly defend preserving all 15 concrete wall identities as the presented visual stimulus categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` and `LickTrind`.

ii. 
```python
lickfr=np.asarray(b['LickFr'],float)
licktr=np.asarray(b['LickTrind'],int)
```

iii. The notes say `LickFr` gives fractional lick-frame coordinates and `LickTrind` is used to keep each event in its declared trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI builds a binary vector over retained corridor frames, assigns each lick in that trial to the nearest retained frame, and keeps it only if the nearest retained frame is within one native frame.

ii. 
```python
lick=np.zeros(len(ix),dtype=np.int16)
for ev in lickfr[licktr==tr]:
    k=int(np.argmin(np.abs(ix.astype(float)-ev)))
    if abs(float(ix[k])-float(ev)) <= 1.0: lick[k]=1
```

iii. The notes justify nearest-frame assignment as more robust than floor/ceil for fractional lick frames near trial boundaries.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick vector is defined directly on the retained frame indices `ix` of each trial, so it matches the neural trial length exactly.

ii. 
```python
for tr in range(int(b['ntrials'])):
    ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
    ...
    lick=np.zeros(len(ix),dtype=np.int16)
    ...
    neural.append(np.asarray(spk[:,ix],dtype=np.float16,order='C'))
```

iii. The notes state that licks outside the retained corridor window are intentionally absent from the converted trial.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`.

ii. 
```python
pos=np.asarray(b['ft_Pos'],float)[:nfr]
...
pbin=np.clip(np.floor(pos[ix]/10.0),0,3).astype(np.int16)
```

iii. The notes describe the native position units as decimeters and the kept corridor range as 0–4 m.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI divides `ft_Pos` by 10, floors to a 1 m bin, clips to `[0, 3]`, and stores the categorical bin id at each retained frame.

ii. 
```python
pbin=np.clip(np.floor(pos[ix]/10.0),0,3).astype(np.int16)
```

iii. The notes justify this as the direct decoder-task discretization of the 4 m texture corridor into four equal 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Categories are `[0,1)`, `[1,2)`, `[2,3)`, and `[3,4]` meters, implemented as floor of decimeters divided by 10 and clipped to 0–3.

ii. 
```python
'output_values':[STIMULI,['not licking','licking'],['0-1 m','1-2 m','2-3 m','3-4 m'],
                 ['speed Q1','speed Q2','speed Q3','speed Q4']],
```
```python
pbin=np.clip(np.floor(pos[ix]/10.0),0,3).astype(np.int16)
```

iii. The notes say this follows the requested 1 m corridor bins exactly.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is computed at the same retained frame indices `ix` used for the neural slice, so it is time-aligned and trial-length matched.

ii. 
```python
pbin=np.clip(np.floor(pos[ix]/10.0),0,3).astype(np.int16)
...
neural.append(np.asarray(spk[:,ix],dtype=np.float16,order='C'))
```

iii. The notes say position is already defined on the frame grid and is simply restricted to the retained corridor frames.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii. 
```python
speed=np.asarray(b['ft_RunSpeed'],float)[:nfr]
```

iii. The notes describe speed as already synchronized to the imaging-frame grid.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI applies fixed global thresholds to the retained framewise speed values using `np.searchsorted(..., side='right')`. It does not compute per-session rank quartiles.

ii. 
```python
SPEED_EDGES = np.array([0.0, 8.32701545, 30.15676260], dtype=np.float32)
...
sbin=np.searchsorted(SPEED_EDGES,speed[ix],side='right').astype(np.int16)
```

iii. The notes justify this with a one-time global quantile computation and the claim that exact equal-count bins are impossible because many frames have exactly zero speed.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four categories are created by thresholds at 0, 8.32701545, and 30.15676260 cm/s using right-sided binning.

ii. 
```python
SPEED_EDGES = np.array([0.0, 8.32701545, 30.15676260], dtype=np.float32)
...
sbin=np.searchsorted(SPEED_EDGES,speed[ix],side='right').astype(np.int16)
```

iii. The notes say these thresholds came from global corridor-frame speed quantiles across the dataset.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed bins are computed on the same retained frame indices `ix` used for neural slicing.

ii. 
```python
sbin=np.searchsorted(SPEED_EDGES,speed[ix],side='right').astype(np.int16)
...
neural.append(np.asarray(spk[:,ix],dtype=np.float16,order='C'))
```

iii. The notes say speed remains on the imaging-frame grid and is simply categorized after truncation/restriction.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates behavior arrays to the imaged frame count, checks for neural/behavior id mismatches and retinotopy length mismatches, ignores non-finite/out-of-corridor positions, and only keeps lick events whose nearest retained frame is within one frame. Trials with too few retained frames trigger errors rather than being silently dropped.

ii. 
```python
if neural_ids != set(beh):
    raise ValueError(...)
...
if len(ia) != nneurons: raise ValueError(...)
```
```python
nfr=min(nfr,spk.shape[1],len(b['ft_trInd']))
spk=spk[:,:nfr]
```
```python
ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
if len(ix)<2: raise ValueError(...)
```

iii. The notes frame this dataset as very clean and prefer hard assertions when expected invariants fail.

## 12-a. What are the most time-consuming steps of the code?

i. The AI identifies session neural I/O, concatenation/slicing/casting of large `spks` arrays, and final pickle serialization as the dominant costs.

ii. 
```python
raw=np.load(DATA/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
spk=np.concatenate(raw,axis=0)
```
```python
with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes explicitly say the source object format forces full-session materialization and that trial-list pickle serialization is also expensive.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining explicit nontrivial loop is per-lick nearest-frame assignment inside each trial. The AI’s notes say this loop is sparse and negligible relative to neural I/O.

ii. 
```python
for ev in lickfr[licktr==tr]:
    k=int(np.argmin(np.abs(ix.astype(float)-ev)))
    if abs(float(ix[k])-float(ev)) <= 1.0: lick[k]=1
```

iii. The notes describe this as an accepted small loop rather than a major optimization target.

## 12-c. What processing does the code repeat multiple times?

i. The AI repeats per-trial mask construction and repeated `ix.astype(float)` conversions inside the lick loop for every trial. It also repeatedly loads behavior files in `load_catalog` once per experiment type rather than grouping later.

ii. 
```python
for tr in range(int(b['ntrials'])):
    ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
```
```python
for ev in lickfr[licktr==tr]:
    k=int(np.argmin(np.abs(ix.astype(float)-ev)))
```

iii. The notes do not highlight major harmful duplication beyond the unavoidable per-session work and trial-level slicing.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Optional plot generation is purely diagnostic. The code also collects experiment labels and detailed `session_info` metadata that are not used by the downstream decoder itself.

ii. 
```python
if show: make_plot(sid,neural,inputs,outputs)
```
```python
sinfo.append({'session_id':sid,'experiment_types':sorted(labels[sid]),'training_day_elapsed':days[sid],
              'n_trials':len(n),'n_neurons':len(r),'median_frame_interval_ms':dt*1000})
```

iii. The notes present these as validation/documentation aids rather than as necessary parts of the decoder training representation.
