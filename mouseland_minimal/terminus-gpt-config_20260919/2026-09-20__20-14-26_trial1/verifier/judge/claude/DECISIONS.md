# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over all `Beh_*.npy` files in the `beh/` directory, loads each as a dictionary keyed by session ID, and collects unique sessions. It also collects all `*_neural_data.npy` files from `spk/`. Sessions are defined as the intersection of behavior keys and spike files. Test3 sessions with `_swap1`/`_swap2` suffixes are canonicalized by stripping the suffix. Retinotopy files are loaded per-session from the `retinotopy/` directory.

ii.
```python
for fn in sorted(glob.glob(os.path.join(ROOT,'beh','Beh_*.npy'))):
    B=np.load(fn,allow_pickle=True).item()
    for raw_sid,b in B.items():
        sid=re.sub(r'_swap[12]$', '', raw_sid)
        if sid not in beh_by_session:
            beh_by_session[sid]=b
spk_files={os.path.basename(f).replace('_neural_data.npy',''):f
           for f in glob.glob(os.path.join(ROOT,'spk','*_neural_data.npy'))}
sessions=sorted(set(beh_by_session)&set(spk_files), key=parse_sid)
```

iii. The AI explored the data directory structure and discovered the behavior files, spike files, and retinotopy files. It identified that test3 behavior keys have `_swap1`/`_swap2` suffixes that needed to be stripped to match spike files. The AI chose to iterate directly over behavior files rather than using `Imaging_Exp_info.npy`.

## 1-b. How are the data split into subjects?

i. The mouse name is extracted as the first underscore-delimited component of the session ID via `parse_sid()`. Subjects are the sorted unique mouse names across all sessions.

ii.
```python
def parse_sid(s):
    p=s.split('_')
    return p[0], '_'.join(p[1:4]), p[4]
subjects=sorted({parse_sid(s)[0] for s in sessions})
subject_id={s:i for i,s in enumerate(subjects)}
```

iii. The AI recognized that session IDs follow the format `mouse_YYYY_MM_DD_block` and built a parser accordingly.

## 1-c. How are the data split into sessions?

i. A session is one unique `*_neural_data.npy` spike file. The AI deduplicates by using the intersection of spike file names and behavior dictionary keys (after stripping swap suffixes). Each physical recording appears once.

ii.
```python
sessions=sorted(set(beh_by_session)&set(spk_files), key=parse_sid)
```

iii. The AI noted that behavior files contain analysis aliases (train/test, before/after learning) that map to the same physical recording, and that test3 entries are duplicated with swap suffixes. The deduplication retains the first behavior record encountered for each physical session.

## 1-d. How are the data split into trials?

i. Trials are defined by iterating over `StartFr` and `GrayFr` arrays. Each trial spans from `ceil(StartFr)` to just before `ceil(GrayFr)`, i.e., from corridor entry to the start of the gray space. Trials where `j <= i` (no valid frames) are skipped.

ii.
```python
for tr,(st,en,sf,wall,rw) in enumerate(zip(starts,ends,sounds,walls,rewarded)):
    i=max(0,int(np.ceil(st))); j=min(nframes,int(np.ceil(en)))
    if j<=i: continue
    inds=np.arange(i,j); T=j-i
```

iii. The AI used `StartFr` as corridor entry and `GrayFr` as the boundary to the gray space, using `ceil` for frame index rounding. This defines the trial as the textured corridor segment.

## 1-e. How are trials filtered based on quality controls?

i. The only filter is `j <= i`, which removes trials with no valid frames (e.g., when `GrayFr` rounds to the same or earlier frame than `StartFr`). Sessions with fewer than 2 surviving trials are skipped. No trial length outlier filtering is applied.

ii.
```python
if j<=i: continue
...
if len(sn)<2: continue
```

iii. The AI did not implement any trial length outlier filtering. There is no discussion of removing excessively long trials in the trajectory.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the per-session `*_neural_data.npy` files, which contain a list of arrays (one per imaging plane) concatenated along the neuron axis. Retinotopy data comes from `iarea` in `*_trans.npz` files.

ii.
```python
obj=np.load(spk_files[sid],allow_pickle=True).item()
planes=obj['spks']
spk=np.concatenate([np.asarray(x) for x in planes],axis=0)
```

iii. The AI confirmed this matches the paper repository's `utils.load_spk` function.

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are used directly with no additional smoothing, normalization, or filtering. Data is stored as float32. The columns for each trial are sliced based on the trial frame range.

ii.
```python
nn=np.asarray(spk[:,i:j],dtype=np.float32)
```

iii. The AI stated: "Neural data are published deconvolved activity, no added smoothing/normalization."

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neurons are filtered.** All neurons are kept, including those outside the four visual areas. The retinotopy mapping assigns region indices but uses an "unassigned" category (index 4) for neurons that don't match iarea values 1-4. However, the mapping itself is incorrect: the AI maps `iarea` values 1=V1, 2=medial, 3=anterior, 4=lateral, while the actual data uses different codes (e.g., `iarea=8` is V1).

ii.
```python
brain_regions=['V1','medial','anterior','lateral','unassigned']
...
rid=np.full(ar.shape,4,dtype=np.int64)
for raw,outid in ((1,0),(2,1),(3,2),(4,3)): rid[ar==raw]=outid
reps=int(np.ceil(ns/len(rid))); rid=np.tile(rid,reps)[:ns]
```

iii. The AI stated "Retinotopy iarea uses 1=V1, 2=medial, 3=anterior, 4=lateral" based on incorrect assumptions. The actual `iarea` values range from -1 to 9 with a different mapping.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial starts at `ceil(StartFr)`, which is corridor entry (trial start). The trial ends at just before `ceil(GrayFr)`. Trials are variable length.

ii.
```python
i=max(0,int(np.ceil(st))); j=min(nframes,int(np.ceil(en)))
nn=np.asarray(spk[:,i:j],dtype=np.float32)
```

iii. The AI metadata states: "temporal_alignment_event: trial start (entry into the 4-m visual corridor; StartFr)".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The imaging frames are the time bins. The time bin size is computed as the median frame interval across sessions, approximately 315 ms (~3.17 Hz).

ii.
```python
dt=float(np.nanmedian(np.diff(ft))*86400.0)
...
'time_bin_size':float(np.median([x['median_frame_interval_s'] for x in session_info]))*1000.0,
```

iii. The AI recognized the imaging frame as the native resolution and did not apply rebinning.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr`, the frame number of the sound cue for each trial, and the frame indices of the trial.

ii.
```python
sounds=np.asarray(b['SoundFr'])
...
t_sound=(float(sf)-inds)*dt
```

iii. The AI used the sound frame number and frame indices to compute the time difference.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue is computed as `(SoundFr - frame_index) * dt`, where `dt` is the median frame interval in seconds. This gives a positive value before the cue and negative after.

ii.
```python
t_sound=(float(sf)-inds)*dt
```

iii. The AI initially had the sign reversed (`(inds - sf)*dt`) but corrected it in a patch, noting "the requested 'time to sound cue' should conventionally be cue time minus current time."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the same frame indices `inds = np.arange(i, j)` as the neural data slice.

ii.
```python
inds=np.arange(i,j)
t_sound=(float(sf)-inds)*dt
```

iii. The same frame range is used for all data streams within a trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date component of each session ID, parsed as `YYYY_MM_DD`.

ii.
```python
subject_dates=defaultdict(list)
for sid in sessions:
    m,d,_=parse_sid(sid); subject_dates[m].append(d)
first_date={m:min(v) for m,v in subject_dates.items()}
```

iii. The AI stated: "Subject-specific chronological session rank is a reproducible continuous day covariate (calendar days from that animal's first included recording)."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day of training is computed as the number of calendar days between the session date and the subject's first session date. This differs from a simple ordinal count of sessions.

ii.
```python
def training_day(sid):
    m,d,_=parse_sid(sid)
    return float((datetime.strptime(d,'%Y_%m_%d')-datetime.strptime(first_date[m],'%Y_%m_%d')).days)
```

iii. The AI used actual calendar day differences rather than session ordinal counts.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr`, the fractional frame number of corridor entry, and the frame indices of the trial.

ii.
```python
starts=np.asarray(b['StartFr'])
...
t_since=(inds-float(st))*dt
```

iii. N/A

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Time since trial start is computed as `(frame_index - StartFr) * dt`, where `dt` is the median frame interval. This gives a near-zero value at trial start, increasing over the trial.

ii.
```python
t_since=(inds-float(st))*dt
```

iii. The AI initially attempted to use absolute MATLAB datenums but switched to frame-interval-based computation "to avoid cancellation."

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. It uses the same frame indices `inds = np.arange(i, j)` as the neural data slice.

ii.
```python
inds=np.arange(i,j)
t_since=(inds-float(st))*dt
```

iii. Same frame range for all streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, a boolean array indicating whether each trial is in a rewarded corridor.

ii.
```python
rewarded=np.asarray(b['isRew'],bool)
...
np.full(T,float(rw))
```

iii. N/A

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` value is cast to float (0.0 or 1.0) and broadcast across all time bins of the trial.

ii.
```python
np.full(T,float(rw))
```

iii. No additional processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the wall texture for each trial.

ii.
```python
walls=np.asarray(b['WallName']).astype(str)
...
visual_values=sorted({str(x) for sid in sessions for x in np.asarray(beh_by_session[sid]['WallName'])})
visual_id={x:i for i,x in enumerate(visual_values)}
```

iii. The AI collected all unique WallName values across the dataset.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses the **literal WallName** as the category, resulting in 15 distinct categories (circle1, circle2, circle3, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2, rock1, rock2, wood1, wood2, wood5, wood1_swap1, wood1_swap2). The value is broadcast per-trial across all time bins.

ii.
```python
visual_values=sorted({str(x) for sid in sessions for x in np.asarray(beh_by_session[sid]['WallName'])})
visual_id={x:i for i,x in enumerate(visual_values)}
...
np.full(T,visual_id[wall],dtype=np.int64)
```

iii. The AI described this as "Global visual labels are the literal published WallName categories."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame numbers of each lick in the session.

ii.
```python
lickfr=np.rint(np.asarray(b['LickFr'],float)).astype(int)
lickset=set(lickfr[(lickfr>=0)&(lickfr<nframes)].tolist())
```

iii. The AI identified `LickFr` as the source for lick events.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame numbers are rounded to the nearest integer (`np.rint`), filtered to valid frame range, and converted to a set. For each trial, a frame is marked 1 if it appears in the lick set, 0 otherwise.

ii.
```python
lickfr=np.rint(np.asarray(b['LickFr'],float)).astype(int)
lickset=set(lickfr[(lickfr>=0)&(lickfr<nframes)].tolist())
...
lick=np.fromiter((1 if q in lickset else 0 for q in inds),dtype=np.int64,count=T)
```

iii. The AI used nearest-frame rounding (`np.rint`) rather than truncation (`.astype(int)`).

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The same frame indices `inds = np.arange(i, j)` are used for lick lookup and neural slicing.

ii.
```python
inds=np.arange(i,j)
lick=np.fromiter((1 if q in lickset else 0 for q in inds),dtype=np.int64,count=T)
```

iii. Same frame range for all streams.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position of the mouse at each imaging frame, in decimeters (0-60).

ii.
```python
pos=np.asarray(b['ft_Pos'],float)
```

iii. N/A

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 10 (converting decimeters to meters) and floored to get integer bin indices, then clipped to range [0, 3].

ii.
```python
pbin=np.clip(np.floor(pos[i:j]/10.0),0,3).astype(np.int64)
```

iii. The AI described: "Published ft_Pos in 10 units/m, floor(ft_Pos/10), clipped to bins 0..3."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Floor division by 10 creates 4 bins of 1 meter each: 0-1m (bin 0), 1-2m (bin 1), 2-3m (bin 2), 3-4m (bin 3). Values beyond 4m are clipped to bin 3.

ii.
```python
pbin=np.clip(np.floor(pos[i:j]/10.0),0,3).astype(np.int64)
```

iii. The instructions specified 4 equal-length 1-m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Uses the same frame slice `pos[i:j]` as the neural data.

ii.
```python
pbin=np.clip(np.floor(pos[i:j]/10.0),0,3).astype(np.int64)
```

iii. Same frame range.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed=np.asarray(b['ft_RunSpeed'],float)
```

iii. N/A

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 bins using **global** quantile edges computed across all valid corridor frames from all sessions. The edges are at the 25th, 50th, and 75th percentiles. `np.digitize` assigns each frame to a bin.

ii.
```python
all_speed=[]
for sid in sessions:
    b=beh_by_session[sid]; n=len(b['ft'])
    for st,en in zip(np.asarray(b['StartFr']),np.asarray(b['GrayFr'])):
        i=max(0,int(np.ceil(st))); j=min(n,int(np.ceil(en)))
        if j>i: all_speed.append(np.asarray(b['ft_RunSpeed'])[i:j])
speed_edges=np.nanquantile(np.concatenate(all_speed),[.25,.5,.75]).astype(float)
...
sbin=np.digitize(speed[i:j],speed_edges,right=False).astype(np.int64)
```

iii. The AI stated: "Global quartiles over valid corridor frames, as requested (each bin is 25% of data)."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The speed values are binned using `np.digitize` with edges at global 25th, 50th, and 75th percentile values, producing 4 bins (0-25%, 25-50%, 50-75%, 75-100%).

ii.
```python
sbin=np.digitize(speed[i:j],speed_edges,right=False).astype(np.int64)
```

iii. N/A

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Uses the same frame slice `speed[i:j]` as the neural data.

ii.
```python
sbin=np.digitize(speed[i:j],speed_edges,right=False).astype(np.int64)
```

iii. Same frame range.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The number of frames is clipped to the minimum of neural and behavioral frame counts. Lick frames outside the valid range are filtered out. Trials with no valid frames (`j <= i`) are skipped. Sessions with fewer than 2 trials are skipped.

ii.
```python
nframes=min(spk.shape[1],len(b['ft']))
lickfr=np.rint(np.asarray(b['LickFr'],float)).astype(int)
lickset=set(lickfr[(lickfr>=0)&(lickfr<nframes)].tolist())
if j<=i: continue
if len(sn)<2: continue
```

iii. The AI handled the mismatch between neural and behavioral frame counts, and filtered invalid lick frames.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the spike files, which total approximately 405 GB. Each session's neural data file is loaded entirely into memory.

ii.
```python
obj=np.load(spk_files[sid],allow_pickle=True).item()
spk=np.concatenate([np.asarray(x) for x in planes],axis=0)
```

iii. The conversion took roughly 50 minutes to process all 89 sessions, dominated by I/O.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The lick lookup uses a Python generator with a set membership test per frame, which could be vectorized with numpy array operations.

ii.
```python
lick=np.fromiter((1 if q in lickset else 0 for q in inds),dtype=np.int64,count=T)
```

iii. N/A

## 12-c. What processing does the code repeat multiple times?

i. The global speed quantile computation iterates over all sessions and trials once to collect speeds, then the main conversion loop iterates over all sessions and trials again. These two passes could potentially be merged.

ii.
```python
# First pass: compute speed edges
for sid in sessions:
    ...
# Second pass: process sessions
for si,sid in enumerate(sessions):
    ...
```

iii. N/A

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI keeps all neurons including those in non-visual areas and "unassigned" regions. The brain_region_idx includes an "unassigned" category with 3,736,488 neurons, which are not typically used in the analyses described in the paper. This results in a 277 GB pickle file instead of a much smaller one.

ii.
```python
brain_regions=['V1','medial','anterior','lateral','unassigned']
# No neuron filtering - all neurons kept
```

iii. The verification output showed: "V1: 147071 neurons, medial: 139295 neurons, anterior: 568741 neurons, lateral: 99439 neurons, unassigned: 3736488 neurons" - with 80% of neurons unassigned due to incorrect iarea mapping.
