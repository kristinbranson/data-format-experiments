# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent glob-loads every `Beh_*.npy` dictionary, canonicalizes `_swap1`/`_swap2` behavior keys, intersects these keys with all spike filenames, then loads each matched spike dictionary and retinotopy file session by session. It does not use `Imaging_Exp_info.npy` as the master index.

ii.
```python
for fn in sorted(glob.glob(os.path.join(ROOT,'beh','Beh_*.npy'))):
    B=np.load(fn,allow_pickle=True).item()
spk_files={os.path.basename(f).replace('_neural_data.npy',''):f
           for f in glob.glob(os.path.join(ROOT,'spk','*_neural_data.npy'))}
sessions=sorted(set(beh_by_session)&set(spk_files), key=parse_sid)
```

iii. The trajectory says behavior files contain synchronized frame-level data and that aliases represent duplicate analysis views. The agent chose one physical session per spike file and reported 89 sessions and 38,110 trials.

## 1-b. How are the data split into subjects?

i. Subject is the first component of the canonical session ID. Unique subjects are sorted, and each retained session receives its subject's integer index.

ii.
```python
def parse_sid(s):
    p=s.split('_')
    return p[0], '_'.join(p[1:4]), p[4]
subjects=sorted({parse_sid(s)[0] for s in sessions})
subject_idx=np.asarray([subject_id[x['subject']] for x in session_info])
```

iii. The trajectory identified IDs as `subject_YYYY_MM_DD_block` and corrected an earlier parsing error to make subject, date, sorting, and metadata consistent.

## 1-c. How are the data split into sessions?

i. A session is one canonical `mouse_date_block` ID backed by one spike file. `_swap1` and `_swap2` suffixes are stripped, and only the first matching behavior view is retained.

ii.
```python
sid=re.sub(r'_swap[12]$', '', raw_sid)
if sid not in beh_by_session:
    beh_by_session[sid]=b
sessions=sorted(set(beh_by_session)&set(spk_files), key=parse_sid)
```

iii. The agent inspected test3 records and concluded the suffixed records have identical synchronized ranges and are alternate views of one physical recording, not distinct sessions.

## 1-d. How are the data split into trials?

i. Trials are paired entries of `StartFr` and `GrayFr`. The retained frame interval is `ceil(StartFr)` through the frame before `ceil(GrayFr)`, clipped to available imaging frames; trial identity fields such as `ft_trInd` and `ft_CorrSpc` are not used.

ii.
```python
for tr,(st,en,sf,wall,rw) in enumerate(zip(starts,ends,sounds,walls,rewarded)):
    i=max(0,int(np.ceil(st))); j=min(nframes,int(np.ceil(en)))
    if j<=i: continue
    nn=np.asarray(spk[:,i:j],dtype=np.float32)
```

iii. The agent treated corridor entry through the end of the textured corridor as the requested trial window and recorded this convention explicitly in metadata.

## 1-e. How are trials filtered based on quality controls?

i. Only empty or wholly out-of-range windows are removed. Sessions with fewer than two resulting trials are removed. There is no long-trial/outlier filter.

ii.
```python
if j<=i: continue
...
if len(sn)<2: continue
```

iii. The trajectory gives no data-quality justification for retaining extreme-duration stopped trials; it focused on keeping all synchronized corridor trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from every array in the spike dictionary's `spks` list, concatenated along neurons. Brain-region labels come from retinotopy `iarea` when the file exists.

ii.
```python
planes=obj['spks']
spk=np.concatenate([np.asarray(x) for x in planes],axis=0)
ar=np.asarray(np.load(rf)['iarea']).astype(int)
```

iii. The agent explicitly followed repository `utils.load_spk`, which concatenates the published `spks` arrays, and regarded retinotopy as auxiliary neuron metadata.

## 2-b. How is the `neural` data processed?

i. Published deconvolved activity is concatenated, sliced to each trial, and cast to float32. It is not smoothed, normalized, rebinned, or padded.

ii.
```python
spk=np.concatenate([np.asarray(x) for x in planes],axis=0)
nn=np.asarray(spk[:,i:j],dtype=np.float32)
sn.append(nn)
```

iii. The trajectory states that the supplied `spks` are already processed neural activity and that repository loading applies no further normalization.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are filtered. Region codes 1–4 are mapped to four named regions; everything else is labeled `unassigned`. If region labels are shorter than the concatenated neuron count, labels are tiled; missing retinotopy makes all neurons unassigned.

ii.
```python
rid=np.full(ar.shape,4,dtype=np.int64)
for raw,outid in ((1,0),(2,1),(3,2),(4,3)): rid[ar==raw]=outid
reps=int(np.ceil(ns/len(rid))); rid=np.tile(rid,reps)[:ns]
```

iii. The agent reasoned that all already-curated published activity should be retained rather than applying paper analyses' stimulus-selective-neuron filtering. It did not justify the assumed retinotopy code map or label tiling.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each variable-length trial starts at `ceil(StartFr)`, the first retained frame at or after corridor entry, and ends just before `ceil(GrayFr)`. There is no padding or common endpoint.

ii.
```python
i=max(0,int(np.ceil(st))); j=min(nframes,int(np.ceil(en)))
nn=np.asarray(spk[:,i:j],dtype=np.float32)
```

iii. The agent identified `StartFr` as corridor entry and the requested alignment event, and used variable-length traversals.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One imaging frame is one bin; no temporal rebinning is applied. Metadata uses the median session frame interval in milliseconds (about 315 ms).

ii.
```python
dt=float(np.nanmedian(np.diff(ft))*86400.0)
'time_bin_size':float(np.median([x['median_frame_interval_s'] for x in session_info]))*1000.0
```

iii. The agent determined behavior was already synchronized to imaging frames, so it retained the native sampling grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr`, each retained imaging-frame index, and the session's median interval computed from `ft`.

ii.
```python
sounds=np.asarray(b['SoundFr'])
dt=float(np.nanmedian(np.diff(ft))*86400.0)
t_sound=(float(sf)-inds)*dt
```

iii. The trajectory corrected the sign after noting that “time to” conventionally means cue time minus current time.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The difference between the possibly fractional sound frame and each integer frame is multiplied by the median seconds per frame. Values are positive before and negative after the cue.

ii.
```python
t_sound=(float(sf)-inds)*dt
```

iii. The agent chose frame intervals instead of subtracting large absolute MATLAB datenums to avoid floating-point cancellation.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated for exactly `inds = arange(i, j)`, the columns used in the trial's neural slice.

ii.
```python
inds=np.arange(i,j); T=j-i
nn=np.asarray(spk[:,i:j],dtype=np.float32)
t_sound=(float(sf)-inds)*dt
```

iii. The agent relied on the behavior's imaging-frame synchronization so neural and covariate arrays share columns.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the `YYYY_MM_DD` portion of each session ID and the earliest included date for that subject.

ii.
```python
first_date={m:min(v) for m,v in subject_dates.items()}
return float((datetime.strptime(d,'%Y_%m_%d')-
              datetime.strptime(first_date[m],'%Y_%m_%d')).days)
```

iii. The agent described calendar days from the animal's first included recording as a reproducible continuous day covariate.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar-date subtraction produces a zero-based elapsed-day value, which is repeated across every time bin of each trial in the session.

ii.
```python
np.full(T,training_day(sid))
```

iii. The agent preferred elapsed calendar days; it did not justify deviating from ordinal recorded training-session count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from fractional `StartFr`, retained imaging-frame indices, and the median interval from `ft`.

ii.
```python
starts=np.asarray(b['StartFr'])
t_since=(inds-float(st))*dt
```

iii. The agent used frame-relative time to avoid cancellation in absolute MATLAB timestamps.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The fractional start frame is subtracted from every retained integer frame and multiplied by median seconds per frame. A preceding dead assignment based on `ft` is immediately overwritten.

ii.
```python
t_since=(ft[i:j]-float(st if st<nframes else i)*0)*86400.0
t_since=(inds-float(st))*dt
```

iii. The second expression is the intentional computation; the first is leftover, discarded work.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same `inds` corresponding to neural columns `i:j`, so the first retained bin is at or shortly after zero.

ii.
```python
nn=np.asarray(spk[:,i:j],dtype=np.float32)
t_since=(inds-float(st))*dt
```

iii. The agent treated frame index as the common synchronization grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from trial-level `isRew`.

ii.
```python
rewarded=np.asarray(b['isRew'],bool)
```

iii. The trajectory identified `isRew` as the published rewarded-corridor indicator.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The Boolean trial value is converted to float and broadcast across the trial's time bins.

ii.
```python
np.full(T,float(rw))
```

iii. No further processing was considered necessary because availability is constant within a trial.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived directly from each trial's literal `WallName` string.

ii.
```python
walls=np.asarray(b['WallName']).astype(str)
visual_values=sorted({str(x) for sid in sessions
                      for x in np.asarray(beh_by_session[sid]['WallName'])})
```

iii. The agent called the literal published wall names the global visual labels.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Every distinct wall-name variant receives a sorted global integer ID, and that ID is broadcast across the trial. Crops and swap variants are not collapsed to circle, leaf, rock, and wood.

ii.
```python
visual_id={x:i for i,x in enumerate(visual_values)}
np.full(T,visual_id[wall],dtype=np.int64)
```

iii. The agent chose literal categories without explaining why the requested broad visual categories should include multiple variants.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from session-level `LickFr` imaging-frame coordinates.

ii.
```python
lickfr=np.rint(np.asarray(b['LickFr'],float)).astype(int)
```

iii. The agent recognized that the published behavior is already synchronized to imaging frames.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frames are rounded to the nearest integer, clipped to valid imaging bounds, stored in a set, and each trial frame is labeled 1 if present and 0 otherwise.

ii.
```python
lickset=set(lickfr[(lickfr>=0)&(lickfr<nframes)].tolist())
lick=np.fromiter((1 if q in lickset else 0 for q in inds),
                 dtype=np.int64,count=T)
```

iii. The agent documented nearest-frame rounding but did not justify it against the repository/reference truncation convention.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Binary lick values are generated for the same integer `inds` corresponding to neural columns `i:j`.

ii.
```python
inds=np.arange(i,j)
nn=np.asarray(spk[:,i:j],dtype=np.float32)
lick=np.fromiter((1 if q in lickset else 0 for q in inds), ...)
```

iii. Frame indices are used as the common alignment grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-level `ft_Pos`.

ii.
```python
pos=np.asarray(b['ft_Pos'],float)
pbin=np.clip(np.floor(pos[i:j]/10.0),0,3).astype(np.int64)
```

iii. The agent inferred the raw position units are 10 per meter.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 10, floored, clipped to 0–3, and stored as integer categorical labels.

ii.
```python
pbin=np.clip(np.floor(pos[i:j]/10.0),0,3).astype(np.int64)
```

iii. This implements the requested four equal 1 m bins over the 4 m corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 0, 10, 20, 30, and 40 raw units, corresponding to 0–1, 1–2, 2–3, and 3–4 m; out-of-range values are clipped to an endpoint bin.

ii.
```python
['0-1 m','1-2 m','2-3 m','3-4 m']
np.clip(np.floor(pos[i:j]/10.0),0,3)
```

iii. The thresholds follow directly from the requested equal-length bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos[i:j]` is sliced over exactly the same frames as `spk[:, i:j]`.

ii.
```python
nn=np.asarray(spk[:,i:j],dtype=np.float32)
pbin=np.clip(np.floor(pos[i:j]/10.0),0,3).astype(np.int64)
```

iii. Behavior is already interpolated onto imaging frames.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-level `ft_RunSpeed` over trial intervals defined by `StartFr` and `GrayFr`.

ii.
```python
speed=np.asarray(b['ft_RunSpeed'],float)
all_speed.append(np.asarray(b['ft_RunSpeed'])[i:j])
```

iii. The agent used the synchronized published speed stream directly.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speeds from all valid corridor frames in all unique sessions are concatenated; global 25th, 50th, and 75th percentile value thresholds are computed once, then `digitize` assigns categories.

ii.
```python
speed_edges=np.nanquantile(np.concatenate(all_speed),[.25,.5,.75]).astype(float)
sbin=np.digitize(speed[i:j],speed_edges,right=False).astype(np.int64)
```

iii. The agent interpreted “each corresponding to 25% of the data” as global quartiles over the complete converted dataset.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global numeric quantiles delimit four bins. Ties remain together, so bins need not contain exactly equal frame counts.

ii.
```python
speed_edges=np.nanquantile(...,[.25,.5,.75])
np.digitize(speed[i:j],speed_edges,right=False)
```

iii. The trajectory gives no special handling for the large tie at zero speed.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The already frame-aligned speed array is sliced with `i:j`, the same interval as neural data.

ii.
```python
nn=np.asarray(spk[:,i:j],dtype=np.float32)
sbin=np.digitize(speed[i:j],speed_edges,right=False).astype(np.int64)
```

iii. The common imaging-frame grid provides alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural/behavior length is capped at their minimum; invalid trial windows and out-of-bounds licks are dropped; sessions with fewer than two trials are skipped; missing retinotopy becomes `unassigned`; spike sessions lacking behavior raise an error. Nonfinite speed values are ignored while finding quantiles, but later digitization would place NaNs in a category.

ii.
```python
nframes=min(spk.shape[1],len(b['ft']))
if j<=i: continue
lickfr[(lickfr>=0)&(lickfr<nframes)]
if len(sn)<2: continue
else: rid=np.full(ns,4,dtype=np.int64)
```

iii. The agent aimed to keep conversion running for incomplete metadata while enforcing synchronized behavior for every spike session. It did not investigate whether tiling region labels is scientifically valid.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating 405 GB of spike files, copying every trial slice, and serializing the resulting 277 GB pickle dominate. The global speed scan is comparatively small.

ii.
```python
obj=np.load(spk_files[sid],allow_pickle=True).item()
spk=np.concatenate([np.asarray(x) for x in planes],axis=0)
nn=np.asarray(spk[:,i:j],dtype=np.float32)
pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory repeatedly identified conversion as I/O-heavy; processing 89 sessions and later loading the 277 GB pickle each took many minutes.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-frame lick membership, per-trial slicing/stacking, region-code assignment, and the preliminary per-trial global-speed collection could be more vectorized. The lick generator is the clearest avoidable Python loop.

ii.
```python
lick=np.fromiter((1 if q in lickset else 0 for q in inds),
                 dtype=np.int64,count=T)
for raw,outid in ((1,0),(2,1),(3,2),(4,3)): rid[ar==raw]=outid
```

iii. The trajectory did not discuss these optimization opportunities; it regarded disk I/O as the dominant cost.

## 12-c. What processing does the code repeat multiple times?

i. Every behavior file is loaded once during session discovery and its arrays are revisited during global visual-label construction, the global speed pass, and session conversion. `training_day(sid)` is also recomputed for every session in both input construction and metadata.

ii.
```python
for sid in sessions: ... all_speed.append(...)
for si,sid in enumerate(sessions): ...
np.full(T,training_day(sid))
'training_day':training_day(sid)
```

iii. No explicit justification was given; these repetitions simplify the implementation and are small relative to neural I/O.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes an absolute-`ft` version of `t_since` and immediately overwrites it. It also loads/retains unassigned neurons and creates tiled region metadata that the downstream decoder does not use; `gc.collect()` on every session adds overhead.

ii.
```python
t_since=(ft[i:j]-float(st if st<nframes else i)*0)*86400.0
t_since=(inds-float(st))*dt
gc.collect()
```

iii. The first expression appears to be an abandoned implementation. The trajectory emphasized preserving all neural activity, explaining the large unused region population but not the dead computation.
