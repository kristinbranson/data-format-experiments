# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads behavior by globbing all `Beh_*.npy` files and iterating over their dictionary entries, keeping the first occurrence of each session ID. Neural data is loaded per-session from `spk/<session_id>_neural_data.npy`. Retinotopy is loaded from `retinotopy/` files matched by the first four underscore-delimited parts of the session ID. The AI does NOT use `Imaging_Exp_info.npy` as a master index; instead it discovers sessions by intersecting behavior keys with available neural files.

ii.
```python
def load_behavior_sessions():
    out={}; source={}
    for p in sorted(glob.glob(ROOT+'/beh/Beh_*.npy')):
        try: d=np.load(p,allow_pickle=True).item()
        except Exception: continue
        for sid,b in d.items():
            if isinstance(b,dict) and 'ft' in b and sid not in out:
                out[sid]=b; source[sid]=os.path.basename(p)
    return out,source
```
```python
spk={os.path.basename(p).replace('_neural_data.npy',''):p for p in glob.glob(ROOT+'/spk/*_neural_data.npy')}
sids=sorted(set(beh)&set(spk),key=lambda s:(s.split('_')[0],sid_date(s),s))
```

iii. From trajectory step 10: "There are 99 behavior sessions but only 76 overlap neural recordings; those 76 are the usable sessions." The AI chose to discover sessions by intersection rather than using the experiment index file, reasoning that sessions without both behavior and neural data cannot support decoding.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the session ID by taking the first underscore-delimited token (e.g., `TX108` from `TX108_2023_01_05_2`). The unique sorted subjects from included sessions form the subjects list.

ii.
```python
subjects=sorted({s.split('_')[0] for s in sids}); subjmap={s:i for i,s in enumerate(subjects)}
```
```python
used_subjects=sorted({x['session_id'].split('_')[0] for x in session_info}); um={s:i for i,s in enumerate(used_subjects)}
```

iii. The AI derived subjects from the session IDs, which encode the mouse name as the first field. This is functionally equivalent to using `mname` from the experiment info.

## 1-c. How are the data split into sessions?

i. A session corresponds to one neural recording file, matched with its behavior data via the session ID. The AI identifies 76 sessions (those with both behavior and neural data), excluding 13 neural-only recordings.

ii.
```python
sids=sorted(set(beh)&set(spk),key=lambda s:(s.split('_')[0],sid_date(s),s))
```

iii. From trajectory step 10: "We have 76 neural sessions with aligned behavioral data and 13 neural recordings without behavior, which cannot support the requested decoder variables and should be excluded." The reference solution also arrives at 76 usable sessions (89 total minus 13 without behavior, though some are duplicated across experiment types).

## 1-d. How are the data split into trials?

i. Trials are iterated from 0 to `ntrials`. For each trial, a mask selects frames where `ft_trInd == trial` AND `ft_isMoving` AND `ft_CorrSpc` AND `0 <= ft_Pos < 40`. Trials with fewer than 2 frames after masking are dropped.

ii.
```python
def trial_mask(b,tr,n):
    tri=np.asarray(b['ft_trInd'])[:n]
    pos=np.asarray(b['ft_Pos'])[:n]
    moving=np.asarray(b['ft_isMoving'],bool)[:n]
    corr=np.asarray(b['ft_CorrSpc'],bool)[:n]
    return (tri==tr) & moving & corr & np.isfinite(pos) & (pos>=0) & (pos<40)
```

iii. From trajectory step 7: "The paper explicitly uses Suite2p deconvolved fluorescence traces and restricts analyses to running timepoints." The AI applies the paper's running-only restriction (`ft_isMoving`) as an additional filter beyond the corridor space flag, and also adds explicit position bounds.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 valid frames (after applying the moving+corridor+position mask) are dropped. No trial length outlier filtering is applied.

ii.
```python
if ix.size<2: continue
```
```python
if len(sn)<2:
    print('skip',sid,'fewer than 2 usable trials',flush=True); continue
```

iii. The AI applies only a minimum frame count filter. There is no equivalent of the reference's 99th percentile trial length filter that removes excessively long trials (where mice stopped for extended periods).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files, which contain deconvolved calcium traces per imaging plane, concatenated across planes. Visual area assignments come from `iarea` in the retinotopy files.

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
```

iii. From trajectory step 10: "The conversion should follow the reference loader and preserve all neurons unless execution proves infeasible." The AI concatenates planes identically to the reference code's `load_spk` function.

## 2-b. How is the `neural` data processed?

i. Neural data is stored as float32. Neurons are subsampled to a maximum of 2000 per session via a deterministic region-stratified sampling procedure. Only frames matching the trial mask (moving, corridor, valid position) are included.

ii.
```python
if a.max_neurons and X.shape[0]>a.max_neurons:
    rng=np.random.default_rng(0)
    chosen=[]
    for r in range(len(REGIONS)):
        ids=np.flatnonzero(ri==r)
        k=int(round(a.max_neurons*len(ids)/len(ri)))
        if len(ids) and k==0: k=1
        chosen.extend(rng.choice(ids,min(k,len(ids)),replace=False).tolist())
    ...
    chosen=np.sort(np.asarray(chosen)); X=X[chosen]; ri=ri[chosen]
```
```python
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. From trajectory step 12: "The decoder processes one session at a time but creates a trainable projection for every neuron, so preserving 20,000-90,000 neurons across 76 sessions would make both the pickle and model impractically large." The AI caps neurons at 2000 per session, matching the decoder's SVD projection threshold.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI assigns brain regions using the same area mapping as the reference (V1=8, mHV=0/1/2/9, lHV=5/6, aHV=3/4), but additionally includes an "unassigned" category (index 4) for neurons not in these four areas. Unlike the reference, neurons in the "unassigned" region are NOT dropped -- they are retained and can be included in the stratified sample.

ii.
```python
REGIONS=['V1','mHV','lHV','aHV','unassigned']

def area_index(ia):
    ia=np.asarray(ia); z=np.full(ia.shape,4,dtype=np.int8)
    z[ia==8]=0
    z[np.isin(ia,[0,1,2,9])]=1
    z[np.isin(ia,[5,6])]=2
    z[np.isin(ia,[3,4])]=3
    return z
```

iii. The AI preserves all neurons initially and only subsamples later. The reference drops neurons outside the four visual areas entirely.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is trial start (corridor entry). Only frames matching the trial mask (moving + corridor + position bounds) are included. Frames where the mouse is not moving are excluded, which can create temporal gaps within a trial.

ii.
```python
mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. From trajectory step 11: "Trial streams will retain native 3.17 Hz samples and paper-required moving corridor frames." The AI filters to running-only frames, consistent with the paper's analysis restriction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging frames at 3.17 Hz are retained (315.46 ms bins). No rebinning is applied.

ii.
```python
FS=3.17
'time_bin_size':1000.0/FS
```

iii. From trajectory step 11: "Native imaging frames (3.17 Hz) are retained because this task requires temporal lick, cue-time, speed and elapsed-time variables."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the timestamp of the sound cue for each trial) and `ft` (the timestamp of every imaging frame).

ii.
```python
cue=float(np.asarray(b['SoundTime'])[tr])
tocue=(cue-ft[ix])*86400.0
```

iii. The AI uses `SoundTime` (absolute timestamps) rather than `SoundFr` (frame numbers) used by the reference. Both encode the same event but via different representations.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to the sound cue is computed as `(SoundTime - ft[frame]) * 86400`, converting from MATLAB datenum (days) to seconds. This gives positive values before the cue and negative after.

ii.
```python
tocue=(cue-ft[ix])*86400.0
inp=np.vstack([tocue, ...]).astype(np.float32)
```

iii. The computation is a simple time difference converted from days to seconds. The sign convention (positive = before cue) matches the reference's `cue[trial] - time`.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the frame timestamps (`ft`) of the same masked frames (`ix`) used for neural data, so alignment is inherent.

ii.
```python
tocue=(cue-ft[ix])*86400.0
```

iii. All streams use the same frame indices, ensuring temporal alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the recording date extracted from the session ID and the first recording date for each subject.

ii.
```python
def sid_date(s): return datetime.datetime.strptime('_'.join(s.split('_')[1:4]),'%Y_%m_%d').date()
firstdate={m:min(sid_date(s) for s in sids if s.split('_')[0]==m) for m in subjects}
day=float((sid_date(sid)-firstdate[sid.split('_')[0]]).days)
```

iii. From trajectory step 11: "A robust continuous training-day variable should therefore be derived from recording date relative to each subject's first included recording."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day is computed as the number of elapsed calendar days between the session date and the subject's first included recording date. This means gaps between recording days are preserved (e.g., if recordings are on day 0, day 7, day 14, the values are 0, 7, 14). The value is broadcast across all frames of a trial.

ii.
```python
day=float((sid_date(sid)-firstdate[sid.split('_')[0]]).days)
inp=np.vstack([..., np.full(ix.size,day), ...]).astype(np.float32)
```

iii. The reference counts ordinal recording sessions (0, 1, 2, ...), while the AI uses elapsed calendar days. The AI's max value is 89, while the reference's max is 7.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (the absolute timestamp of trial start) and `ft` (frame timestamps).

ii.
```python
t0=float(np.asarray(b['Trial_start_time'])[tr])
elapsed=(ft[ix]-t0)*86400.0
```

iii. The AI uses `Trial_start_time` rather than the reference's `StartFr`. Both represent the trial start but in different units (timestamps vs frame numbers).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The elapsed time is computed as `(ft[frame] - Trial_start_time) * 86400`, converting from MATLAB datenum days to seconds. This gives time since trial start in seconds.

ii.
```python
elapsed=(ft[ix]-t0)*86400.0
```

iii. The computation is a direct time difference. Note that because the AI filters to moving frames only, the first frame of a trial may not be exactly at corridor entry, potentially introducing a small positive offset.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. It is computed from the same masked frame indices used for neural data.

ii.
```python
elapsed=(ft[ix]-t0)*86400.0
```

iii. Same alignment mechanism as other variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks rewarded trials.

ii.
```python
np.full(ix.size,int(np.asarray(b['isRew'])[tr]))
```

iii. Directly used from the behavior data, same as reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` value for each trial is cast to int (0 or 1) and broadcast across all frames of the trial.

ii.
```python
np.full(ix.size,int(np.asarray(b['isRew'])[tr]))
```

iii. No additional processing needed; same approach as reference.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the texture on the corridor walls for each trial.

ii.
```python
categories.update(map(str,np.asarray(b['WallName']).reshape(-1)))
categories=sorted(categories); catmap={x:i for i,x in enumerate(categories)}
name=str(np.asarray(b['WallName'])[tr])
np.full(ix.size,catmap[name])
```

iii. From trajectory step 10: The AI collects all unique wall names across the dataset and sorts them alphabetically to form the category vocabulary.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each individual wall name (e.g., `circle1`, `circle2`, `leaf1`, `leaf1_swap1`) is treated as a separate category. This yields 13 categories rather than the 4 broad texture categories (circle, leaf, rock, wood) used by the reference.

ii.
```python
categories=sorted(categories)  # ['circle1', 'circle2', 'circle3', 'leaf1', 'leaf1_swap1', ...]
catmap={x:i for i,x in enumerate(categories)}
'output_values':[categories, ...]  # 13 categories
```

iii. The AI did not group wall names into broader texture categories. The instructions say "Visual stimulus category. e.g. circle, leaf, etc." which implies broad categories. The reference maps 15 wall names to 4 categories via a lookup table.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickTime`, which contains the absolute timestamps of each lick event in the session.

ii.
```python
lt=np.asarray(b['LickTime']).reshape(-1)
```

iii. The AI uses `LickTime` (absolute timestamps) while the reference uses `LickFr` (frame numbers). Both represent the same lick events.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Each lick timestamp is assigned to its nearest imaging frame using `searchsorted` on the frame timestamps. A binary flag (0/1) per frame is created.

ii.
```python
lick=np.zeros(n,dtype=np.int8)
lt=np.asarray(b['LickTime']).reshape(-1)
if lt.size and np.issubdtype(lt.dtype,np.number):
    lt=lt[np.isfinite(lt.astype(float))].astype(float)
    q=np.searchsorted(ft,lt); q=np.clip(q,1,n-1)
    q-=((lt-ft[q-1]) <= (ft[q]-lt)).astype(int)
    lick[np.unique(q)]=1
```

iii. The AI rounds to the nearest frame, while the reference truncates fractional frame numbers via `astype(int)`. Both produce binary lick indicators per frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick binary flag is indexed by the same masked frame indices used for neural data.

ii.
```python
out=np.vstack([..., lick[ix], ...])
```

iii. Same frame-based alignment as all other variables.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in the corridor at each imaging frame, in units of 0.1 m (decimeters).

ii.
```python
pos=np.asarray(b['ft_Pos'],float)[:n][ix]
np.clip((pos//10).astype(int),0,3)
```

iii. Same source variable as reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 10 (converting from decimeters to meters), floor-divided, and clipped to the range [0, 3], yielding four 1-meter bins.

ii.
```python
np.clip((pos//10).astype(int),0,3)
```

iii. Same binning logic as reference: `np.clip(beh['ft_Pos'][:n_frames] // 10, 0, 3)`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal 1-meter bins: [0,10), [10,20), [20,30), [30,40) in decimeters, mapped to categories 0-3.

ii.
```python
'output_values':[..., ['0-1 m','1-2 m','2-3 m','3-4 m'], ...]
```

iii. Matches the reference's position binning exactly.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is taken from the same masked frame indices used for neural data.

ii.
```python
pos=np.asarray(b['ft_Pos'],float)[:n][ix]
```

iii. Same frame-based alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed=np.asarray(b['ft_RunSpeed'],float)[:n][ix]
```

iii. Same source variable as reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 bins using global quantile thresholds computed across all retained moving corridor frames from all sessions. The thresholds are at the 25th, 50th, and 75th percentiles. `np.digitize` is used to assign bins.

ii.
```python
# First pass: global speed quartiles
speeds=[]
for sid in sids:
    ...
    ok=np.isfinite(tri)&np.asarray(b['ft_isMoving'],bool)[:n]&np.asarray(b['ft_CorrSpc'],bool)[:n]&np.isfinite(pos)&(pos>=0)&(pos<40)
    v=np.asarray(b['ft_RunSpeed'],float)[:n][ok]; speeds.append(v[np.isfinite(v)])
speed_edges=np.quantile(np.concatenate(speeds),[.25,.5,.75])
```
```python
np.digitize(speed,speed_edges)
```

iii. The reference uses per-session rank-based quartiles (each session's frames are independently ranked and divided into 4 equal bins). The AI uses global threshold-based quartiles, meaning the same speed value maps to the same bin regardless of session. This is a significant difference in approach.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins defined by global quantile edges: [0, 12.19), [12.19, 25.21), [25.21, 40.60), [40.60, inf). Bin assignment uses `np.digitize`, yielding values 0-3.

ii.
```python
speed_edges=np.quantile(np.concatenate(speeds),[.25,.5,.75])
np.digitize(speed,speed_edges)
```
```python
'output_values':[..., ['Q1 (slowest)','Q2','Q3','Q4 (fastest)']]
```

iii. The reference ensures each session's bins contain exactly 25% of that session's data. The AI's global thresholds may produce uneven bin distributions within individual sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is taken from the same masked frame indices used for neural data.

ii.
```python
speed=np.asarray(b['ft_RunSpeed'],float)[:n][ix]
```

iii. Same frame-based alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and behavior streams are truncated to their shared minimum length. Lick timestamps that are non-finite are filtered out. If `LickTime` is empty or non-numeric, licking is set to all zeros. Trials with fewer than 2 valid frames are skipped. Sessions with fewer than 2 valid trials are skipped. Retinotopy/neuron count mismatches are handled by resizing the region array.

ii.
```python
n=min(len(b['ft']),min(x.shape[1] for x in obj))
```
```python
lt=lt[np.isfinite(lt.astype(float))].astype(float)
```
```python
if len(ri)!=X.shape[0]:
    print('WARNING region/neuron mismatch',sid,len(ri),X.shape[0],flush=True)
    ri=np.resize(ri,X.shape[0]).astype(np.int8)
```

iii. The AI handles edge cases like non-finite values and length mismatches that the reference assumes won't occur.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the neural spike files, which total ~434 GB of NumPy data. The full conversion took approximately 8 minutes.

ii.
```python
obj=np.load(spk[sid],allow_pickle=True).item()['spks']
X=np.concatenate([x[:,:n] for x in obj],axis=0)
```

iii. Same bottleneck as reference.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop processes each trial individually, creating arrays and appending to lists. The global speed quartile first pass iterates all sessions to collect speeds. Both could theoretically be more vectorized but are not the bottleneck.

ii.
```python
for tr in range(int(b['ntrials'])):
    mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
    ...
```

iii. The trial-level loop is similar to the reference's approach and is not a significant performance concern relative to I/O.

## 12-c. What processing does the code repeat multiple times?

i. The first pass over all sessions to compute global speed quartiles and category vocabulary requires reading and processing behavior data that is then read again in the main conversion loop. This doubles the behavior processing work (though behavior files are small relative to neural files).

ii.
```python
# First pass
for sid in sids:
    b=beh[sid]; categories.update(...)
    ...speeds.append(...)

# Second pass (main loop)
for si,sid in enumerate(sids):
    b=beh[sid]
    ...
```

iii. The reference avoids this by computing speed quartiles per-session, eliminating the need for a global first pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and includes "unassigned" brain region neurons which the reference code would have discarded. These contribute to the neuron count but may not be relevant for visual cortex decoding. Additionally, the detailed metadata fields (excluded sessions list, speed quartile edges, etc.) are thorough but not used by the decoder.

ii.
```python
REGIONS=['V1','mHV','lHV','aHV','unassigned']
```

iii. Including unassigned neurons dilutes the visual cortex signal that the decoder is trying to learn from.
