# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI glob-loads every `Beh_*.npy`, keeps the first top-level behavior dictionary entry for each exact key, glob-indexes all spike files, and processes only the exact intersection of behavior keys and spike session IDs. It loads each retained spike file and matching retinotopy file per session. This yielded 76 sessions, not all 89 recordings.

ii.
```python
for p in sorted(glob.glob(ROOT+'/beh/Beh_*.npy')):
    d=np.load(p,allow_pickle=True).item()
    for sid,b in d.items():
        if isinstance(b,dict) and 'ft' in b and sid not in out:
            out[sid]=b
spk={os.path.basename(p).replace('_neural_data.npy',''):p
     for p in glob.glob(ROOT+'/spk/*_neural_data.npy')}
sids=sorted(set(beh)&set(spk), key=...)
```

iii. The trajectory says duplicate behavior entries are alternate paper comparisons and exact matches give frame-aligned behavior. It explicitly accepted exclusion of 13 spike sessions lacking an exact behavior key because their requested decoder variables could not be matched. It also chose streaming because the neural archive is about 434 GB.

## 1-b. How are the data split into subjects?

i. The subject is parsed as the first underscore-separated component of each session ID. Unique used subjects are sorted, and every retained session gets the corresponding integer index.

ii.
```python
subjects=sorted({s.split('_')[0] for s in sids})
used_subjects=sorted({x['session_id'].split('_')[0] for x in session_info})
'subject_idx': np.asarray([um[x['session_id'].split('_')[0]] for x in session_info])
```

iii. The trajectory treated session IDs as reliably encoding mouse, date, and block. The final data contained all 19 mice.

## 1-c. How are the data split into sessions?

i. Each spike filename stem is a session ID (`mouse_YYYY_MM_DD_block`). Exact matching behavior keys define the retained sessions; duplicate behavior keys use the first file in sorted filename order. Sessions with fewer than two usable trials are skipped.

ii.
```python
sids=sorted(set(beh)&set(spk),key=lambda s:(s.split('_')[0],sid_date(s),s))
if len(sn)<2:
    continue
```

iii. The AI reasoned that repeated behavior entries represent comparisons of the same recording and that stable filename precedence avoids duplication. It did not reconstruct suffixed behavior keys from the master experiment index.

## 1-d. How are the data split into trials?

i. Trials are enumerated from `0` to `ntrials-1`. A trial consists only of frames with its `ft_trInd`, `ft_isMoving`, `ft_CorrSpc`, finite position, and position in `[0,40)`. Thus stationary frames inside a trial are removed and a retained trial need not be temporally contiguous.

ii.
```python
return (tri==tr) & moving & corr & np.isfinite(pos) & (pos>=0) & (pos<40)
for tr in range(int(b['ntrials'])):
    ix=np.flatnonzero(trial_mask(b,tr,n))
```

iii. The trajectory cites the paper's restriction to running timepoints and the 4-m textured corridor. It chose native frame trials because temporal outputs would be lost by the paper's spatial interpolation.

## 1-e. How are trials filtered based on quality controls?

i. Frames failing movement/corridor/finite-position checks are removed; trials with fewer than two remaining frames are dropped. Entire sessions with fewer than two retained trials are dropped. There is no long-trial/outlier filter.

ii.
```python
if ix.size<2: continue
...
if len(sn)<2:
    print('skip',sid,'fewer than 2 usable trials',flush=True); continue
```

iii. The AI justified the movement filter from the methods and the two-trial minimum from decoder validation. It did not discuss the reference's 99th-percentile long-trial exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from each plane's `spks` in `<sid>_neural_data.npy`; neuron region labels come from retinotopy `iarea`.

ii.
```python
obj=np.load(spk[sid],allow_pickle=True).item()['spks']
X=np.concatenate([x[:,:n] for x in obj],axis=0)
ia=np.load(rp[0])['iarea']; ri=area_index(ia)
```

iii. The trajectory identified `spks` as the Suite2p deconvolved traces used by the paper and `iarea` as the released retinotopic assignment.

## 2-b. How is the `neural` data processed?

i. Plane arrays are truncated to a shared frame count and concatenated. If there are more than 2,000 neurons, a deterministic region-stratified random subset is selected. Per-trial matrices are sliced at retained frames and stored as float32.

ii.
```python
n=min(len(b['ft']),min(x.shape[1] for x in obj))
X=np.concatenate([x[:,:n] for x in obj],axis=0)
if a.max_neurons and X.shape[0]>a.max_neurons:
    ...
    X=X[chosen]; ri=ri[chosen]
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. The AI considered full-neuron conversion impractical given 434 GB of sources and decoder cost, so it capped neurons while preserving regional proportions. It retained native frames instead of spatial interpolation to support temporal variables.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron is removed based on released cell/area quality. Unassigned areas are kept as a fifth region; mismatched region vectors are resized. The main filtering is the 2,000-neuron stratified subsample.

ii.
```python
REGIONS=['V1','mHV','lHV','aHV','unassigned']
z=np.full(ia.shape,4,dtype=np.int8)
...
ri=np.resize(ri,X.shape[0]).astype(np.int8)
```

iii. The trajectory notes that the reference loader concatenates planes without an extra cell-quality filter. The cap was justified by storage and training feasibility, not biological quality.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural columns use frame indices labeled with that trial and passing the moving textured-corridor mask. Time-since-start is referenced to `Trial_start_time`, but the neural array starts at the first retained moving frame rather than necessarily the entry frame; stationary gaps are omitted.

ii.
```python
ix=np.flatnonzero(mask)
t0=float(np.asarray(b['Trial_start_time'])[tr])
sn.append(np.asarray(X[:,ix],dtype=np.float32))
elapsed=(ft[ix]-t0)*86400.0
```

iii. The AI described this as alignment to trial start/corridor entry and relied on the frame-aligned behavior arrays. Its methods-based movement restriction drove the omitted frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native 3.17-Hz imaging frames are retained, giving approximately 315.46 ms per bin. No temporal rebinning is applied.

ii.
```python
ROOT='/app/data'; FS=3.17
'time_bin_size':1000.0/FS
```

iii. The AI reasoned that native frames are the natural common grid for lick, cue, speed, and elapsed-time variables; spatial interpolation would discard their timing.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived directly from per-trial `SoundTime` and per-frame `ft` timestamps.

ii.
```python
ft=np.asarray(b['ft'],float)[:n]
cue=float(np.asarray(b['SoundTime'])[tr])
```

iii. The AI found timestamps already aligned to neural frames and used them instead of converting `SoundFr`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Frame time is subtracted from cue time and MATLAB-day units are converted to seconds, producing positive values before and negative values after the cue.

ii.
```python
tocue=(cue-ft[ix])*86400.0
```

iii. The trajectory identifies behavior as richly pre-aligned, so only a time difference and units conversion were considered necessary.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses `ft[ix]`, with exactly the same `ix` columns as the trial's neural matrix.

ii.
```python
tocue=(cue-ft[ix])*86400.0
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. The AI relied on behavior streams supplied at neural-frame timestamps and truncation to the shared minimum length.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the date encoded in the session ID and the earliest included recording date for that subject.

ii.
```python
firstdate={m:min(sid_date(s) for s in sids if s.split('_')[0]==m) for m in subjects}
day=float((sid_date(sid)-firstdate[sid.split('_')[0]]).days)
```

iii. The AI interpreted day of training as elapsed calendar days from a mouse's first included imaging recording.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar dates are parsed, subtracted, converted to a float day count, and broadcast to every retained frame of a trial.

ii.
```python
np.full(ix.size,day)
```

iii. This definition is explicitly recorded in metadata; the trajectory did not justify why elapsed calendar days should replace ordinal recorded training sessions.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from per-trial `Trial_start_time` and per-frame `ft`.

ii.
```python
t0=float(np.asarray(b['Trial_start_time'])[tr])
elapsed=(ft[ix]-t0)*86400.0
```

iii. The AI treated the timestamps as directly aligned frame-level behavior.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Trial-start time is subtracted from each retained frame timestamp and converted from days to seconds.

ii.
```python
elapsed=(ft[ix]-t0)*86400.0
```

iii. Only the difference and units conversion were considered necessary.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed for the exact `ix` used to slice neural columns, although stationary frames are absent.

ii.
```python
elapsed=(ft[ix]-t0)*86400.0
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. The AI relied on the shared frame grid and shared truncation.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes from the per-trial `isRew` flag.

ii.
```python
int(np.asarray(b['isRew'])[tr])
```

iii. No separate justification was given; it is the direct released reward field.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The flag is cast to an integer and broadcast across every retained frame in the trial.

ii.
```python
np.full(ix.size,int(np.asarray(b['isRew'])[tr]))
```

iii. The AI treated reward availability as a per-trial context variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from each trial's exact `WallName` string.

ii.
```python
categories.update(map(str,np.asarray(b['WallName']).reshape(-1)))
name=str(np.asarray(b['WallName'])[tr])
```

iii. The trajectory calls this a global category vocabulary. It did not collapse crop/swap variants to their four base textures.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All exact wall names are sorted, mapped to integer indices, and the trial's index is broadcast over its frames. This produces 15 categories rather than four broad texture categories.

ii.
```python
categories=sorted(categories); catmap={x:i for i,x in enumerate(categories)}
np.full(ix.size,catmap[name])
```

iii. The AI favored a common vocabulary across sessions but provided no paper-based justification for treating variants as distinct categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from event timestamps in `LickTime` and frame timestamps `ft`.

ii.
```python
lt=np.asarray(b['LickTime']).reshape(-1)
q=np.searchsorted(ft,lt)
```

iii. The AI chose timestamps because they allow each lick to be assigned to the closest imaging frame.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Finite lick times are mapped to the nearest imaging frame; duplicate licks in one frame collapse to a single binary 1.

ii.
```python
q=np.clip(q,1,n-1)
q-=((lt-ft[q-1]) <= (ft[q]-lt)).astype(int)
lick[np.unique(q)]=1
```

iii. Metadata explicitly states “each lick assigned to nearest native imaging frame.”

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The frame-level lick vector is indexed by the same retained `ix` as neural data.

ii.
```python
out=np.vstack([... ,lick[ix], ...])
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. The AI relied on the common frame-time axis.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from frame-level `ft_Pos`.

ii.
```python
pos=np.asarray(b['ft_Pos'],float)[:n][ix]
```

iii. The AI identified the source units as 0.1 m and restricted values to the textured 4-m corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is floor-divided by 10 source units per meter, converted to integers, and clipped to category indices 0–3.

ii.
```python
np.clip((pos//10).astype(int),0,3)
```

iii. Metadata documents bins `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40)`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It uses four fixed equal-length 1-m bins: 0–1, 1–2, 2–3, and 3–4 m.

ii.
```python
['0-1 m','1-2 m','2-3 m','3-4 m']
```

iii. This directly follows the decoder specification.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos[ix]` uses the same retained frame indices as neural columns.

ii.
```python
pos=np.asarray(b['ft_Pos'],float)[:n][ix]
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. The AI used the released frame-aligned behavior grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from frame-level `ft_RunSpeed` over moving, in-corridor, finite-position frames.

ii.
```python
v=np.asarray(b['ft_RunSpeed'],float)[:n][ok]
speed=np.asarray(b['ft_RunSpeed'],float)[:n][ix]
```

iii. The trajectory identifies this as the released frame-level running speed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Three global quantile edges are computed in a first behavior-only pass across included sessions, then each trial speed is digitized against them.

ii.
```python
speed_edges=np.quantile(np.concatenate(speeds),[.25,.5,.75])
np.digitize(speed,speed_edges)
```

iii. The AI wanted common class boundaries across sessions. It noted full-cohort edges of approximately 12.19, 25.21, and 40.60.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Values below/above the three global 25th, 50th, and 75th percentile thresholds receive categories 0–3.

ii.
```python
'speed_quartile_edges':speed_edges.tolist(),
'output_values': [..., ['Q1 (slowest)','Q2','Q3','Q4 (fastest)']]
```

iii. The AI interpreted “each corresponding to 25% of the data” globally and preferred common thresholds, rather than the reference's per-session rank split.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is read at exactly `ix`, the same frame indices as neural columns.

ii.
```python
speed=np.asarray(b['ft_RunSpeed'],float)[:n][ix]
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. The AI used the common frame-aligned behavior grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and behavior streams are truncated to their shared minimum length. Nonfinite positions/speeds are excluded. Sessions without exact behavior matches are omitted. Missing retinotopy becomes “unassigned”; neuron/region length mismatches trigger a warning and `np.resize`. Lick times are filtered for finiteness. Sessions with too few trials are skipped.

ii.
```python
n=min(len(b['ft']),min(x.shape[1] for x in obj))
lt=lt[np.isfinite(lt.astype(float))].astype(float)
if rp: ...
else: ri=np.full(X.shape[0],4,dtype=np.int8)
if len(ri)!=X.shape[0]: ri=np.resize(ri,X.shape[0])
```

iii. The trajectory observed behavior commonly has one extra frame and explicitly chose shared-length truncation. It considered sessions without matched behavior unusable for the required variables.

## 12-a. What are the most time-consuming steps of the code?

i. Loading/deserializing and concatenating every large neural session dominates runtime; the conversion reads roughly 434 GB even though most neurons are then discarded.

ii.
```python
obj=np.load(spk[sid],allow_pickle=True).item()['spks']
X=np.concatenate([x[:,:n] for x in obj],axis=0)
```

iii. The trajectory repeatedly says runtime is dominated by streaming the 434-GB neural archive and waited several minutes for it to finish.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop repeatedly builds full-length boolean masks and extracts arrays. Trials could be grouped by `ft_trInd` once, and several per-trial broadcasts/slices could be batched. The per-region sampling loop is small and not important.

ii.
```python
for tr in range(int(b['ntrials'])):
    mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
```

iii. The trajectory does not explicitly discuss loop vectorization; this characterization follows from the written code. Its optimization focus was I/O and memory.

## 12-c. What processing does the code repeat multiple times?

i. Frame masks and behavior arrays (`ft_Pos`, `ft_RunSpeed`, `isRew`, `WallName`) are repeatedly converted/sliced inside the trial loop. Behavior is also scanned once globally for categories/speed thresholds and again during conversion.

ii.
```python
pos=np.asarray(b['ft_Pos'],float)[:n][ix]
speed=np.asarray(b['ft_RunSpeed'],float)[:n][ix]
name=str(np.asarray(b['WallName'])[tr])
```

iii. No explicit justification was given. The first pass was intentional to establish a global vocabulary and speed thresholds before session conversion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and concatenates every neuron before selecting at most 2,000, so the overwhelming majority of neural values are read/copied and discarded. It computes `tri` in the speed first pass but does not use it, and creates `subjmap` without using it.

ii.
```python
X=np.concatenate([x[:,:n] for x in obj],axis=0)
...
X=X[chosen]
tri=np.asarray(b['ft_trInd'])[:n]
subjmap={s:i for i,s in enumerate(subjects)}
```

iii. The trajectory recognized the I/O cost but considered streaming necessary because the serialized source format must be deserialized before neuron selection. It gave no justification for the unused local variables.
