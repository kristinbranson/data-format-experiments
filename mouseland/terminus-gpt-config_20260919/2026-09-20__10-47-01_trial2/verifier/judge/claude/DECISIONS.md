# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data from `beh/`, `spk/`, and `retinotopy/` subdirectories. It reads `Imaging_Exp_info.npy` as the master index, then loads ALL behavior files upfront in `load_catalog()`, building a single deduplicated dictionary keyed by session ID. Neural data files are loaded per-session in `convert_session()`. Retinotopy files are loaded per-session in `region_indices()`.

ii.
```python
info = np.load(DATA/'beh/Imaging_Exp_info.npy', allow_pickle=True).item()
# ...
d = np.load(DATA/f'beh/Beh_{exp}.npy', allow_pickle=True).item()
# ...
raw=np.load(DATA/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
```

iii. The AI's CONVERSION_NOTES document the deduplication of 142 experiment-membership entries into 89 unique physical recordings. The approach loads all behavior data upfront, validates consistency of duplicates, and checks that neural and behavior session IDs match exactly.

## 1-b. How are the data split into subjects?

i. The mouse name is extracted from the first component of the session ID (splitting on `_`). Subjects are the sorted unique mouse names across all sessions.

ii.
```python
subjects=sorted({x.split('_')[0] for x in ids})
subidx={x:i for i,x in enumerate(subjects)}
# ...
'subject_idx':np.array([subidx[x.split('_')[0]] for x in ids],dtype=np.int16),
```

iii. 19 subjects are identified, consistent with the reference data.

## 1-c. How are the data split into sessions?

i. A session is a unique combination of mouse name, date, and block number, forming the session ID. Deduplication is done in `load_catalog()` by checking if the session ID already exists in the behavior dictionary. 89 unique sessions are identified.

ii.
```python
sid = f"{r['mname']}_{r['datexp']}_{r['blk']}"
# ...
if sid in beh:
    if int(beh[sid]['ntrials']) != int(d[hit]['ntrials']):
        raise ValueError(f'Conflicting duplicate behavior for {sid}')
```

iii. The AI validates that duplicate entries have identical trial counts, providing a consistency check.

## 1-d. How are the data split into trials?

i. Trials are iterated from 0 to `ntrials`. For each trial, frames are selected where `ft_trInd == trial` AND position is finite, non-negative, and less than 40 (the 4m corridor texture region). This selects corridor-only frames.

ii.
```python
for tr in range(int(b['ntrials'])):
    ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
    if len(ix)<2: raise ValueError(f'{sid} trial {tr}: only {len(ix)} corridor frames')
```

iii. The AI restricts to position 0-40 decimeters (0-4m corridor texture region) consistent with the paper's analysis of the texture corridor. The AI requires at least 2 frames per trial and raises an error if fewer are found.

## 1-e. How are trials filtered based on quality controls?

i. The AI does NOT filter trials based on length. All 38,110 trials are kept, including extremely long trials (max 5,607 frames / ~29 minutes). The only check is that each trial has at least 2 corridor frames.

ii.
```python
if len(ix)<2: raise ValueError(f'{sid} trial {tr}: only {len(ix)} corridor frames')
```

iii. The AI's CONVERSION_NOTES state: "No global bad-trial exclusion is described. Some figure analyses restrict to moving periods... Such figure-specific restrictions do not justify dropping valid decoder trials." The AI intentionally keeps all trials including extremely long stationary ones.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files (`spk/<session_id>_neural_data.npy`), which contains a list of arrays from multiple imaging planes, concatenated along the neuron axis. The visual area of each neuron comes from `iarea` in the retinotopy files.

ii.
```python
raw=np.load(DATA/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
spk=np.concatenate(raw,axis=0)
```

iii. Documented in CONVERSION_NOTES: neural data is Suite2p deconvolved fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The traces are not further processed. The columns belonging to a trial's corridor frames are extracted and stored as float16. Trials are variable length.

ii.
```python
neural.append(np.asarray(spk[:,ix],dtype=np.float16,order='C'))
```

iii. The AI notes the data is already deconvolved by Suite2p and no further dF/F computation is needed. Float16 storage is used to reduce memory.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps ALL neurons, including those outside the four visual areas (V1, mHV, lHV, aHV). Neurons with `iarea=7` or other unmapped values are assigned to a 5th "unmapped" brain region rather than being dropped.

ii.
```python
REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'unmapped']
# ...
out=np.full(nneurons,4,dtype=np.int16)  # default to 'unmapped' (index 4)
out[ia==8]=0; out[np.isin(ia,[0,1,2,9])]=1; out[np.isin(ia,[5,6])]=2; out[np.isin(ia,[3,4])]=3
```

iii. The AI's CONVERSION_NOTES state: "Label 7 is not assigned by that function and must be represented as an unmapped/outside-reference-area class rather than silently dropped." This results in 585,641 extra "unmapped" neurons being included (4,691,034 total vs the reference's 4,105,393).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry. The first frame assigned to the trial within the corridor (position 0-40) serves as the start. Trials are variable length, ending wherever the corridor traversal ends. `off_start` is 0.0 and `off_end` is None.

ii.
```python
ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
elapsed=(ft[ix]-ft[ix[0]]).astype(np.float32)
```

iii. The AI metadata records `'temporal_alignment_event':'corridor entry (first imaging frame assigned to trial; ceil StartFr)'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The native imaging frame rate (~3.17 Hz, ~315 ms/frame) is used. The bin size is computed from the median frame interval across all sessions.

ii.
```python
dt=float(np.median(np.diff(ft)))
# ...
'time_bin_size':float(np.median(dts)*1000)
```

iii. The AI notes the imaging frame is the finest temporal resolution available and all behavior streams are already synchronized to it.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame number of the sound cue for each trial) and the frame indices of the current trial frames.

ii.
```python
sound=np.asarray(b['SoundFr'],float)
cue=((sound[tr]-ix.astype(float))*dt).astype(np.float32)
```

iii. The AI uses `SoundFr` and frame indices with a constant dt approximation.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue is computed as `(SoundFr - current_frame_index) * median_frame_dt`. This uses a constant dt approximation rather than actual timestamps. The result is positive before the cue and negative after.

ii.
```python
cue=((sound[tr]-ix.astype(float))*dt).astype(np.float32)
```

iii. The AI uses a simplified constant-dt approach. The reference interpolates `SoundFr` onto actual frame timestamps, which accounts for small timing jitter.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same frame indices (`ix`) used for the neural data extraction, ensuring alignment.

ii.
```python
ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
cue=((sound[tr]-ix.astype(float))*dt).astype(np.float32)
neural.append(np.asarray(spk[:,ix],dtype=np.float16,order='C'))
```

iii. All data streams use the same frame indices for alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session ID, which encodes the date as `YYYY_MM_DD`. The date is parsed and compared to the earliest session date for each mouse.

ii.
```python
def day_offsets(session_ids):
    dates = {s: datetime.strptime('_'.join(s.split('_')[1:4]), '%Y_%m_%d') for s in session_ids}
    first = {}
    for s,d in dates.items():
        mouse=s.split('_')[0]; first[mouse]=min(first.get(mouse,d),d)
    return {s: float((d-first[s.split('_')[0]]).days) for s,d in dates.items()}
```

iii. The AI uses calendar days elapsed from the first session date.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day of training is the number of calendar days elapsed from the subject's earliest imaging session date. This gives values ranging from 0 to 92 (actual calendar days between sessions).

ii.
```python
return {s: float((d-first[s.split('_')[0]]).days) for s,d in dates.items()}
```

iii. The AI's metadata documents: `'training_day_definition':'calendar days elapsed from subject first supplied imaging session'`. The reference solution counts session order (0, 1, 2, ...) giving max value ~7. The AI uses actual calendar day differences, giving much larger values (up to 92).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `ft` (the timestamps of each imaging frame), using the first corridor frame of each trial as the reference.

ii.
```python
ft=np.asarray(b['ft'],float)[:nfr]*86400.0
elapsed=(ft[ix]-ft[ix[0]]).astype(np.float32)
```

iii. The timestamps are converted from MATLAB datenums (days) to seconds by multiplying by 86400.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The elapsed time is computed as the difference between each frame's timestamp and the first corridor frame's timestamp, in seconds. This starts at 0 and increases.

ii.
```python
elapsed=(ft[ix]-ft[ix[0]]).astype(np.float32)
```

iii. Simple subtraction from the first retained corridor frame timestamp.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same frame indices (`ix`) as the neural data.

ii.
```python
ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
elapsed=(ft[ix]-ft[ix[0]]).astype(np.float32)
```

iii. All data streams share the same frame selection.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, a per-trial flag indicating whether the corridor is rewarded.

ii.
```python
rew=np.asarray(b['isRew'],bool)
np.full(len(ix),float(rew[tr]),np.float32)
```

iii. Direct use of the `isRew` field.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` is converted to float (0.0 or 1.0) and broadcast across all time bins of the trial.

ii.
```python
np.full(len(ix),float(rew[tr]),np.float32)
```

iii. No additional processing. It is 0 for unrewarded/unsupervised/naive sessions and 1 for rewarded corridors.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the texture on the corridor walls for each trial.

ii.
```python
wall=np.asarray(b['WallName']).astype(str)
stim_to_idx={x:i for i,x in enumerate(STIMULI)}
# ...
np.full(len(ix),stim_to_idx[wall[tr]],np.int16)
```

iii. The AI uses `WallName` rather than `TrialStim` because `TrialStim` has placeholders in some sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses all 15 individual wall names as separate categories (`circle1`, `circle2`, `circle3`, `leaf1`, `leaf1_swap1`, `leaf1_swap2`, `leaf2`, `leaf3`, `rock1`, `rock2`, `wood1`, `wood1_swap1`, `wood1_swap2`, `wood2`, `wood5`). It does NOT group them into the 4 broad texture categories (circle, leaf, rock, wood) as the instructions and reference do.

ii.
```python
STIMULI = ['circle1','circle2','circle3','leaf1','leaf1_swap1','leaf1_swap2',
           'leaf2','leaf3','rock1','rock2','wood1','wood1_swap1','wood1_swap2','wood2','wood5']
```

iii. The AI's CONVERSION_NOTES describe this as: "Global categorical index over concrete visual identities." The output_values list has 15 entries.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame numbers of licks) and `LickTrind` (trial index of each lick).

ii.
```python
lickfr=np.asarray(b['LickFr'],float); licktr=np.asarray(b['LickTrind'],int)
```

iii. The AI uses both `LickFr` and `LickTrind` to assign licks to trials.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, licks belonging to that trial (via `LickTrind`) are identified. Each lick is mapped to the nearest retained corridor frame. A lick is included only if the nearest frame is within 1.0 frame distance.

ii.
```python
lick=np.zeros(len(ix),dtype=np.int16)
for ev in lickfr[licktr==tr]:
    k=int(np.argmin(np.abs(ix.astype(float)-ev)))
    if abs(float(ix[k])-float(ev)) <= 1.0: lick[k]=1
```

iii. This is more conservative than the reference, which simply marks any frame where a lick's truncated frame number falls. The AI's approach uses `LickTrind` to filter by trial and checks proximity, while the reference directly indexes into a session-wide licking array.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are assigned to the same frame indices used for neural data via the `argmin` proximity search within the trial's retained frames.

ii. Same `ix` array is used for both neural and licking data.

iii. All streams share the same corridor frame selection.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position at each imaging frame in decimeters.

ii.
```python
pos=np.asarray(b['ft_Pos'],float)[:nfr]
pbin=np.clip(np.floor(pos[ix]/10.0),0,3).astype(np.int16)
```

iii. Direct use of `ft_Pos`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by 10 to convert to meters, then floored to get the bin index. Clipped to [0, 3] for the four 1-meter bins.

ii.
```python
pbin=np.clip(np.floor(pos[ix]/10.0),0,3).astype(np.int16)
```

iii. Simple conversion from decimeters to 1-meter bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal 1-meter bins: [0-1m), [1-2m), [2-3m), [3-4m]. The floor operation creates the bin index, with clipping to keep values in [0, 3].

ii.
```python
POSITION = ['0-1 m', '1-2 m', '2-3 m', '3-4 m']  # output_values
pbin=np.clip(np.floor(pos[ix]/10.0),0,3).astype(np.int16)
```

iii. Consistent with the instruction to discretize into 4 equal-length, 1-m-long spatial bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed by the same frame indices (`ix`) as the neural data.

ii.
```python
pbin=np.clip(np.floor(pos[ix]/10.0),0,3).astype(np.int16)
```

iii. All streams share the same frame selection.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed=np.asarray(b['ft_RunSpeed'],float)[:nfr]
sbin=np.searchsorted(SPEED_EDGES,speed[ix],side='right').astype(np.int16)
```

iii. Direct use of `ft_RunSpeed`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI uses pre-computed GLOBAL fixed speed threshold edges `[0.0, 8.327, 30.157]` cm/s applied via `searchsorted`. This creates 4 bins: `<0`, `[0, 8.327)`, `[8.327, 30.157)`, `>=30.157`.

ii.
```python
SPEED_EDGES = np.array([0.0, 8.32701545, 30.15676260], dtype=np.float32)
sbin=np.searchsorted(SPEED_EDGES,speed[ix],side='right').astype(np.int16)
```

iii. The AI pre-computed these edges from the global data quantiles. However, the resulting bin distribution is very uneven: Q1=9.8%, Q2=40.2%, Q3=25%, Q4=25%. The instructions say "4 bins, each corresponding to 25% of the data." The reference uses per-session rank-based quartiles ensuring exact 25% splits.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Fixed global speed edges are used with `searchsorted`. Due to many exact-zero speed values, the first bin captures very few frames (9.8%) while the second bin is oversized (40.2%).

ii.
```python
SPEED_EDGES = np.array([0.0, 8.32701545, 30.15676260], dtype=np.float32)
sbin=np.searchsorted(SPEED_EDGES,speed[ix],side='right').astype(np.int16)
```

iii. The AI notes in CONVERSION_NOTES: "Exact 25% counts are impossible due to 20.38% exact-zero ties; deterministic quantile convention documented."

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is indexed by the same frame indices (`ix`) as the neural data.

ii.
```python
sbin=np.searchsorted(SPEED_EDGES,speed[ix],side='right').astype(np.int16)
```

iii. All streams share the same frame selection.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior streams that extend beyond neural data are truncated to the neural frame count. Position values are checked for finiteness. Lick events outside the corridor window (distance > 1.0 frame) are excluded. The AI raises an error if any trial has fewer than 2 corridor frames.

ii.
```python
nfr=min(nfr,spk.shape[1],len(b['ft_trInd']))
spk=spk[:,:nfr]
# ...
ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
if len(ix)<2: raise ValueError(...)
```

iii. The AI checks for finite positions, which the reference does not explicitly do. The reference handles behavior/neural length mismatch the same way.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the neural spike files (total ~411 GB of data). Each session's neural data file is loaded sequentially. The full conversion takes ~668 seconds (~11 minutes).

ii.
```python
raw=np.load(DATA/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
```

iii. I/O-bound on the large neural data files.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The lick assignment loop iterates over each lick event in a trial, computing `argmin` for each one. This could be vectorized.

ii.
```python
for ev in lickfr[licktr==tr]:
    k=int(np.argmin(np.abs(ix.astype(float)-ev)))
    if abs(float(ix[k])-float(ev)) <= 1.0: lick[k]=1
```

iii. The per-lick loop with `argmin` is inefficient compared to the reference's vectorized approach of building a session-wide licking array and indexing into it.

## 12-c. What processing does the code repeat multiple times?

i. The AI uses `ThreadPoolExecutor` with 3 workers for parallel processing when not in sample mode. The `load_catalog()` function loads all behavior files upfront, so they don't need to be re-read per session. No significant repeated processing.

ii.
```python
from concurrent.futures import ThreadPoolExecutor
executor=ThreadPoolExecutor(max_workers=3)
results=executor.map(job,items)
```

iii. The parallel processing approach avoids redundant I/O.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes 585,641 "unmapped" neurons (brain region index 4) that are outside the four visual areas analyzed in the paper. These neurons consume substantial memory and computation but may not contribute meaningful signal for the decoder.

ii.
```python
REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'unmapped']
out=np.full(nneurons,4,dtype=np.int16)  # default to 'unmapped'
```

iii. The AI explicitly decided to keep unmapped neurons rather than drop them, arguing they should not be "silently dropped."
