# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Loads `Imaging_Exp_info.npy`, then every `Beh_<experiment>.npy`; builds one canonical behavior record per physical session and verifies behavior/neural ID equality. Spike files and retinotopy are loaded per session.

ii. ```python
info = np.load(DATA/'beh/Imaging_Exp_info.npy', allow_pickle=True).item()
d = np.load(DATA/f'beh/Beh_{exp}.npy', allow_pickle=True).item()
raw=np.load(DATA/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
```

iii. The notes say 142 experiment memberships reduce to 89 unique physical recordings and that all 89 neural filenames match metadata-linked behavior.

## 1-b. How are the data split into subjects?

i. Derives mouse IDs from the prefix of each session ID, sorts unique IDs, and maps every session to `subject_idx`.

ii. ```python
subjects=sorted({x.split('_')[0] for x in ids}); subidx={x:i for i,x in enumerate(subjects)}
'subject_idx':np.array([subidx[x.split('_')[0]] for x in ids],dtype=np.int16)
```

iii. The notes report 19 mice and identify `mname` as the subject field.

## 1-c. How are the data split into sessions?

i. Defines a physical session by mouse, date, and block; duplicate experiment memberships are merged under that session ID, while experiment labels are retained as metadata.

ii. ```python
sid = f"{r['mname']}_{r['datexp']}_{r['blk']}"
if sid in beh: ...
else: beh[sid] = d[hit]
```

iii. Exploration found 142 memberships but 89 unique neural recordings; duplicates were checked for matching trial counts.

## 1-d. How are the data split into trials?

i. For each raw trial number, selects frames with matching `ft_trInd`, finite position, and position in the 0–40 dm textured corridor. Trials remain variable-length.

ii. ```python
for tr in range(int(b['ntrials'])):
    ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
```

iii. The agent found that trial assignment begins at corridor entry and that `pos<40` ends at the gray-space boundary.

## 1-e. How are trials filtered based on quality controls?

i. Keeps every trial with at least two corridor frames and raises an error otherwise; it does not reject abnormally long/stopped trials.

ii. ```python
if len(ix)<2: raise ValueError(f'{sid} trial {tr}: only {len(ix)} corridor frames')
```

iii. The notes explicitly argue that long stopped/slow trials are scientifically meaningful and report retaining all 38,110 trials, including a 5,607-frame trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Uses the per-plane `spks` arrays as neural activity and `iarea` only to construct neuron region indices.

ii. ```python
raw=np.load(DATA/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
spk=np.concatenate(raw,axis=0)
with np.load(rp) as z: ia = np.asarray(z['iarea'])
```

iii. The supplied `spks` are already Suite2p non-negative deconvolved fluorescence; no dF/F is needed.

## 2-b. How is the `neural` data processed?

i. Concatenates planes, truncates to synchronized frames, slices trial columns, and stores the values as contiguous float16 without normalization, smoothing, deconvolution, padding, or rebinning.

ii. ```python
spk=np.concatenate(raw,axis=0)
spk=spk[:,:nfr]
neural.append(np.asarray(spk[:,ix],dtype=np.float16,order='C'))
```

iii. Float16 was chosen to halve a very large output; sampled quantization error was small and the decoder converts to float32.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Keeps every supplied neuron. It maps the four reference areas and assigns `iarea==7` or other unmatched labels to an additional `unmapped` region rather than filtering them.

ii. ```python
REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'unmapped']
out=np.full(nneurons,4,dtype=np.int16)
```

iii. The agent reasoned that no generic cell-quality filter exists and that label 7 should not be silently discarded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligns each variable-length trial at its first retained corridor frame; neural columns are those same native frame indices, with no padding or fixed end.

ii. ```python
ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
neural.append(np.asarray(spk[:,ix],dtype=np.float16,order='C'))
```

iii. The notes identify first trial-assigned frame/ceil `StartFr` as corridor entry and prefer native temporal sampling.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Uses native imaging frames with no temporal rebinning. It computes each session’s median timestamp interval and reports the median across sessions, about 314.8 ms.

ii. ```python
ft=np.asarray(b['ft'],float)[:nfr]*86400.0
dt=float(np.median(np.diff(ft)))
'time_bin_size':float(np.median(dts)*1000)
```

iii. The agent considered native frames the finest available resolution and used timestamps to account for small jitter.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Uses `SoundFr`, current native frame indices, and the session median frame interval `dt`.

ii. ```python
sound=np.asarray(b['SoundFr'],float)
cue=((sound[tr]-ix.astype(float))*dt).astype(np.float32)
```

iii. The notes describe signed seconds until cue and cite frame synchronization.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Computes `(SoundFr - current_frame) * median_dt`, positive before and negative after the cue.

ii. ```python
cue=((sound[tr]-ix.astype(float))*dt).astype(np.float32)
```

iii. The agent treats fractional cue-frame coordinates as offsets on a constant native-frame grid.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Creates one cue-time value for each frame in the same `ix` used for neural data.

ii. ```python
inp=np.vstack([cue, ...])
neural.append(np.asarray(spk[:,ix],dtype=np.float16,order='C'))
```

iii. The notes claim all streams share imaging-frame coordinates.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derives training day from the date encoded in each session ID and each mouse’s earliest supplied imaging date.

ii. ```python
dates = {s: datetime.strptime('_'.join(s.split('_')[1:4]), '%Y_%m_%d') for s in session_ids}
```

iii. The agent calls elapsed calendar days objective and says it preserves gaps between recordings.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Computes calendar days elapsed from the mouse’s first imaging date and broadcasts that float across every trial frame.

ii. ```python
return {s: float((d-first[s.split('_')[0]]).days) for s,d in dates.items()}
np.full(len(ix),day,np.float32)
```

iii. The notes prefer elapsed calendar days over session ordinal to preserve gaps.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Uses actual frame timestamps `ft` and the first retained corridor frame, rather than `StartFr` directly.

ii. ```python
ft=np.asarray(b['ft'],float)[:nfr]*86400.0
elapsed=(ft[ix]-ft[ix[0]]).astype(np.float32)
```

iii. The agent found the first assigned frame equals ceil `StartFr` and chose to re-zero there.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Subtracts the actual timestamp of the first retained frame, making the first bin exactly zero.

ii. ```python
elapsed=(ft[ix]-ft[ix[0]]).astype(np.float32)
```

iii. The notes explicitly require first time-since-start to be zero and cite small timestamp jitter.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Indexes elapsed time on exactly the same retained frames and stacks it as input row 2.

ii. ```python
elapsed=(ft[ix]-ft[ix[0]]).astype(np.float32)
inp=np.vstack([cue, ..., elapsed, ...])
```

iii. The agent verified equal neural/input/output time lengths.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Uses the trial-level boolean `isRew` field.

ii. ```python
rew=np.asarray(b['isRew'],bool)
```

iii. The notes interpret it as rewarded-corridor availability, not actual reward delivery.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Casts reward availability to 0/1 float and repeats it over all frames of a trial.

ii. ```python
np.full(len(ix),float(rew[tr]),np.float32)
```

iii. It is a per-trial decoder input and thus broadcast to satisfy the common time dimension.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Uses `WallName` and a global vocabulary of all 15 concrete wall/exemplar names.

ii. ```python
wall=np.asarray(b['WallName']).astype(str)
STIMULI = ['circle1', ... ,'wood5']
```

iii. The agent rejected placeholder-containing `TrialStim` and chose the complete concrete presentation identity.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Maps each exact wall name to 0–14 and broadcasts it over the trial.

ii. ```python
stim_to_idx={x:i for i,x in enumerate(STIMULI)}
np.full(len(ix),stim_to_idx[wall[tr]],np.int16)
```

iii. The agent argues that concrete identity is the presented visual category and preserves more information.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Uses fractional lick frame coordinates `LickFr` and trial assignments `LickTrind`.

ii. ```python
lickfr=np.asarray(b['LickFr'],float); licktr=np.asarray(b['LickTrind'],int)
```

iii. The notes use trial identity to prevent boundary assignment errors.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For every lick in a trial, finds the nearest retained corridor frame and sets that bin to 1 only if it is within one frame; multiple licks collapse to binary 1.

ii. ```python
for ev in lickfr[licktr==tr]:
    k=int(np.argmin(np.abs(ix.astype(float)-ev)))
    if abs(float(ix[k])-float(ev)) <= 1.0: lick[k]=1
```

iii. The agent says nearest-frame assignment avoids fractional-boundary off-by-one errors and excludes licks outside the corridor.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Produces a binary vector with one element per retained neural frame.

ii. ```python
lick=np.zeros(len(ix),dtype=np.int16)
out=np.vstack([... ,lick,...])
```

iii. The same trial frame window is used for all streams.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Uses framewise `ft_Pos`, interpreted in decimeters.

ii. ```python
pos=np.asarray(b['ft_Pos'],float)[:nfr]
```

iii. The notes confirm native 0–40 corresponds to the 4 m textured corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Divides position by 10, floors it, and clips the result to class indices 0–3.

ii. ```python
pbin=np.clip(np.floor(pos[ix]/10.0),0,3).astype(np.int16)
```

iii. This implements the requested four equal 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Uses [0,10), [10,20), [20,30), and [30,40) dm, labeled 0–1, 1–2, 2–3, and 3–4 m.

ii. ```python
['0-1 m','1-2 m','2-3 m','3-4 m']
```

iii. The corridor mask guarantees retained positions are below 40 dm.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Subsets position with the identical `ix` used for each neural trial.

ii. ```python
pbin=np.clip(np.floor(pos[ix]/10.0),0,3)
neural.append(np.asarray(spk[:,ix],...))
```

iii. Behavior and neural streams are synchronized by native frame number.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Uses synchronized framewise `ft_RunSpeed`.

ii. ```python
speed=np.asarray(b['ft_RunSpeed'],float)[:nfr]
```

iii. The notes preserve negative speeds and stops as valid behavior.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Applies three hard-coded global edges, previously computed across all finite valid corridor frames, using `searchsorted`.

ii. ```python
SPEED_EDGES = np.array([0.0, 8.32701545, 30.15676260], dtype=np.float32)
sbin=np.searchsorted(SPEED_EDGES,speed[ix],side='right').astype(np.int16)
```

iii. The agent sought reproducible global value thresholds and accepted unequal class sizes caused by zero-speed ties.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Defines classes `<0`, `[0, 8.327)`, `[8.327, 30.157)`, and `>=30.157` cm/s.

ii. ```python
np.searchsorted(SPEED_EDGES,speed[ix],side='right')
```

iii. The notes explain that ties at exactly zero prevent equal value-threshold bins.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Computes speed classes from `speed[ix]`, the same frames as neural data.

ii. ```python
sbin=np.searchsorted(SPEED_EDGES,speed[ix],side='right')
neural.append(np.asarray(spk[:,ix],...))
```

iii. The notes treat behavior as already synchronized to imaging frames.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Truncates all behavior streams to available neural frames, checks plane dtypes and retinotopy lengths, errors on missing keys/unknown stimuli/short trials, and uses assertions before writing. It does not silently impute missing values.

ii. ```python
nfr=min(nfr,spk.shape[1],len(b['ft_trInd']))
if len(ia) != nneurons: raise ValueError(...)
if wall[tr] not in stim_to_idx: raise ValueError(...)
```

iii. The notes describe 1–3 trailing behavior frames as the only routine discrepancy and report a clean dataset.

## 12-a. What are the most time-consuming steps of the code?

i. Identifies loading/concatenating multi-GB spike files, copying corridor slices, float16 conversion, and serializing the 137.9 GiB pickle as dominant costs.

ii. ```python
raw=np.load(...).item()['spks']
spk=np.concatenate(raw,axis=0)
pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes report 668 s total, including 135 s serialization, and call neural I/O/copying dominant.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop repeatedly scans `tri` to build each mask, and the sparse lick loop repeatedly computes nearest distances; both could be grouped/vectorized, though neural I/O dominates.

ii. ```python
for tr in range(int(b['ntrials'])):
    ix=np.flatnonzero((tri==tr)...)
for ev in lickfr[licktr==tr]:
    k=int(np.argmin(np.abs(ix.astype(float)-ev)))
```

iii. The notes mention vectorized frame masks but acknowledge per-lick loops are negligible (74,483 events).

## 12-c. What processing does the code repeat multiple times?

i. Repeated work includes scanning all session frames once per trial, repeatedly converting `ix` to float for every lick, and loading behavior dictionaries separately for each experiment type even where sessions overlap.

ii. ```python
ix=np.flatnonzero((tri==tr)...)
np.abs(ix.astype(float)-ev)
d = np.load(DATA/f'beh/Beh_{exp}.npy', ...)
```

iii. The agent largely considered this negligible relative to neural I/O and used bounded threads for the dominant session work.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No major intermediate transform is computed and then discarded. Optional plotting reads only already-converted arrays. Some retained data—unmapped neurons and extreme stopped trials—would be removed by the human reference, but the agent intentionally treats them as final data rather than disposable processing.

ii. ```python
if show: make_plot(sid,neural,inputs,outputs)
return neural,inputs,outputs,ridx,dt
```

iii. The notes say source object materialization and copying are required and document no disposable analysis-specific processing.
