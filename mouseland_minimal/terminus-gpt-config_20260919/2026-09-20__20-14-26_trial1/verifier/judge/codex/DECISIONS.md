# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script scans every `Beh_*.npy` file under `/app/data/beh`, loads each entire behavior dictionary, canonicalizes `_swap1`/`_swap2` test3 keys to a physical session id, and keeps the first behavior record it sees for each physical session. It separately scans all spike files under `/app/data/spk`, then defines the dataset as the intersection of canonicalized behavior session ids and spike-file session ids.

ii. ```python
for fn in sorted(glob.glob(os.path.join(ROOT,'beh','Beh_*.npy'))):
    B=np.load(fn,allow_pickle=True).item()
    for raw_sid,b in B.items():
        sid=re.sub(r'_swap[12]$', '', raw_sid)
        if sid not in beh_by_session:
            beh_by_session[sid]=b
            source_by_session[sid]=os.path.basename(fn)+(':'+raw_sid if raw_sid != sid else '')
spk_files={os.path.basename(f).replace('_neural_data.npy',''):f
           for f in glob.glob(os.path.join(ROOT,'spk','*_neural_data.npy'))}
sessions=sorted(set(beh_by_session)&set(spk_files), key=parse_sid)
```

iii. In the trajectory, the agent justified this as a way to avoid duplicate analysis aliases in the behavior files while preserving one physical recording per spike file. After inspecting the test3 files, it argued that `_swap1` and `_swap2` were alternate analysis views of the same recording and should be canonicalized to one physical session.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are inferred from the first underscore-delimited field of the session id, and `subject_idx` is built from that parsed subject name.

ii. ```python
def parse_sid(s):
    p=s.split('_')
    return p[0], '_'.join(p[1:4]), p[4]

subjects=sorted({parse_sid(s)[0] for s in sessions})
subject_id={s:i for i,s in enumerate(subjects)}
...
'subject_idx':np.asarray([subject_id[x['subject']] for x in session_info],dtype=np.int64),
```

iii. The trajectory repeatedly treated session ids as `mouse_YYYY_MM_DD_block`, and after fixing an earlier parsing bug, explicitly said subject/date/block metadata should be derived from a single robust session-id parser.

## 1-c. How are the data split into sessions?

i. A session is one canonicalized physical session id matching one spike file, i.e. one `mouse_date_block` recording. Duplicate behavior aliases are removed, especially `_swap1`/`_swap2`.

ii. ```python
sid=re.sub(r'_swap[12]$', '', raw_sid)
...
sessions=sorted(set(beh_by_session)&set(spk_files), key=parse_sid)
if set(spk_files)-set(beh_by_session):
    raise RuntimeError('Spike sessions without synchronized behavior: '+repr(sorted(set(spk_files)-set(beh_by_session))))
```

iii. In the trajectory, the agent said the 89 spike files correspond to unique physical sessions and that behavior aliases should be deduplicated against those spike sessions.

## 1-d. How are the data split into trials?

i. Trials are defined as contiguous frame windows from `ceil(StartFr)` to `ceil(GrayFr)` for each trial, clipped to the number of imaged frames. The script uses those start/end frame intervals directly rather than `ft_trInd` plus `ft_CorrSpc`.

ii. ```python
starts=np.asarray(b['StartFr']); ends=np.asarray(b['GrayFr'])
...
for tr,(st,en,sf,wall,rw) in enumerate(zip(starts,ends,sounds,walls,rewarded)):
    i=max(0,int(np.ceil(st))); j=min(nframes,int(np.ceil(en)))
    if j<=i: continue
    inds=np.arange(i,j); T=j-i
```

iii. The trajectory says `StartFr` marks corridor entry and `GrayFr` marks transition out of the 4 m visual corridor, so slicing `[ceil(StartFr), ceil(GrayFr))` was taken to give corridor-aligned trials.

## 1-e. How are trials filtered based on quality controls?

i. The only per-trial filter is dropping trials with no surviving frames after clipping to the imaged interval (`j <= i`). Sessions with fewer than two surviving trials are later skipped entirely. There is no long-trial or stationary-trial filtering.

ii. ```python
i=max(0,int(np.ceil(st))); j=min(nframes,int(np.ceil(en)))
if j<=i: continue
...
if len(sn)<2: continue
```

iii. The trajectory did not present a separate QC argument here beyond saying corridor trials should be sliced directly from synchronized frame streams and that the converter should preserve all published activity/session data unless a trial had no valid window.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the `spks` list in each session’s `*_neural_data.npy` file. Region labels are derived from retinotopy `iarea`.

ii. ```python
obj=np.load(spk_files[sid],allow_pickle=True).item()
planes=obj['spks']
spk=np.concatenate([np.asarray(x) for x in planes],axis=0)
...
ar=np.asarray(np.load(rf)['iarea']).astype(int)
```

iii. The trajectory explicitly cites the repository’s `utils.load_spk` and says the conversion should preserve that loading behavior by concatenating the three published `spks` arrays.

## 2-b. How is the `neural` data processed?

i. The three `spks` arrays are concatenated on the neuron axis, then each trial takes a contiguous frame slice and stores it as `float32`. No additional smoothing or normalization is applied.

ii. ```python
spk=np.concatenate([np.asarray(x) for x in planes],axis=0)
...
nn=np.asarray(spk[:,i:j],dtype=np.float32)
...
sn.append(nn)
```

iii. The trajectory says the paper loader concatenates all three `spks` arrays, that the data are already synchronized and published as deconvolved activity, and that no extra smoothing or normalization should be added.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script does not drop neurons by visual-area quality criteria. Instead, it keeps all neurons and assigns each one to one of five region categories: `V1`, `medial`, `anterior`, `lateral`, or `unassigned`. If retinotopy exists, the retinotopy labels are remapped and tiled to match the concatenated neuron count; otherwise all neurons become `unassigned`.

ii. ```python
brain_regions=['V1','medial','anterior','lateral','unassigned']
...
if os.path.exists(rf):
    ar=np.asarray(np.load(rf)['iarea']).astype(int)
    rid=np.full(ar.shape,4,dtype=np.int64)
    for raw,outid in ((1,0),(2,1),(3,2),(4,3)): rid[ar==raw]=outid
    reps=int(np.ceil(ns/len(rid))); rid=np.tile(rid,reps)[:ns]
else: rid=np.full(ns,4,dtype=np.int64)
```

iii. The trajectory says the converter should “map retinotopy regions” but “retain all supplied neurons,” rather than filtering cells down to the four reference visual areas.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to `StartFr` as trial start / corridor entry and extend until `GrayFr`, with variable trial lengths and no padding.

ii. ```python
'temporal_alignment_event':'trial start (entry into the 4-m visual corridor; StartFr)',
...
i=max(0,int(np.ceil(st))); j=min(nframes,int(np.ceil(en)))
...
nn=np.asarray(spk[:,i:j],dtype=np.float32)
```

iii. The trajectory explicitly states that `StartFr` is corridor entry and `GrayFr` is the exit from the textured corridor, so trial windows should be corridor-aligned and variable length.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Each imaging frame is one time bin, and the reported bin size is the median frame interval, in milliseconds, computed from `ft`.

ii. ```python
dt=float(np.nanmedian(np.diff(ft))*86400.0)
...
'time_bin_size':float(np.median([x['median_frame_interval_s'] for x in session_info]))*1000.0,
```

iii. The trajectory noted the imaging interval is about 0.315 s and that the synchronized frame grid should be used directly without extra resampling.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr`, together with frame indices and the frame interval estimated from `ft`.

ii. ```python
dt=float(np.nanmedian(np.diff(ft))*86400.0)
...
sounds=np.asarray(b['SoundFr'])
...
t_sound=(float(sf)-inds)*dt
```

iii. The trajectory discusses `SoundFr` as the event frame and says the behavior is already synchronized at imaging-frame resolution, so the cue timing can be computed directly on that grid.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each kept frame index `inds`, the script computes `(SoundFr - inds) * dt`, giving positive values before the sound cue and negative values after it. It does not interpolate onto the `ft` time axis.

ii. ```python
t_sound=(float(sf)-inds)*dt
xx=np.vstack((t_sound,
              np.full(T,training_day(sid)),
              t_since,
              np.full(T,float(rw)))).astype(np.float32)
```

iii. In the trajectory, the agent explicitly corrected the sign so that “time to sound cue” would be cue time minus current time, and chose frame-interval arithmetic rather than absolute MATLAB datenums.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the exact same per-trial frame indices `inds` used to slice the neural data.

ii. ```python
nn=np.asarray(spk[:,i:j],dtype=np.float32)
t_sound=(float(sf)-inds)*dt
```

iii. The trajectory repeatedly emphasizes that the supplied behavior is already synchronized to imaging frames, so inputs and neural data should share the same frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session id date for each subject, parsed from the session name.

ii. ```python
def parse_sid(s):
    p=s.split('_')
    return p[0], '_'.join(p[1:4]), p[4]
...
for sid in sessions:
    m,d,_=parse_sid(sid); subject_dates[m].append(d)
```

iii. The trajectory says subject/date/block metadata should come from a robust session-id parser.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The script defines day of training as the calendar-day difference between a session’s date and that subject’s first included recording date, then broadcasts that scalar across all frames of the trial.

ii. ```python
first_date={m:min(v) for m,v in subject_dates.items()}
from datetime import datetime
def training_day(sid):
    m,d,_=parse_sid(sid)
    return float((datetime.strptime(d,'%Y_%m_%d')-datetime.strptime(first_date[m],'%Y_%m_%d')).days)
...
np.full(T,training_day(sid))
```

iii. The trajectory explicitly describes this as a “calendar days from that animal’s first included recording” covariate.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `StartFr`, together with frame indices and the frame interval estimated from `ft`.

ii. ```python
dt=float(np.nanmedian(np.diff(ft))*86400.0)
...
t_since=(inds-float(st))*dt
```

iii. The trajectory treats `StartFr` as the trial-start event and uses frame-grid arithmetic rather than interpolated timestamps.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The script computes `(frame_index - StartFr) * dt`, where `StartFr` can be fractional, so bins near trial start can be slightly offset from zero. It does not interpolate `StartFr` onto the actual `ft` timestamps.

ii. ```python
# Use frame interval rather than absolute MATLAB datenums to avoid cancellation.
t_since=(inds-float(st))*dt
```

iii. The trajectory explicitly says it chose frame-interval arithmetic “to avoid cancellation” from large MATLAB datenum values.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed from the same frame indices `inds` used for the trial’s neural slice.

ii. ```python
nn=np.asarray(spk[:,i:j],dtype=np.float32)
t_since=(inds-float(st))*dt
```

iii. The trajectory’s general justification is that all streams are already synchronized on imaging frames and should stay on that shared grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `isRew`.

ii. ```python
rewarded=np.asarray(b['isRew'],bool)
...
np.full(T,float(rw))
```

iii. The trajectory does not add a separate argument here; it simply treats `isRew` as the per-trial reward-availability flag.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No real transformation is applied beyond converting to boolean / float and broadcasting the per-trial value across all frames of the trial.

ii. ```python
rewarded=np.asarray(b['isRew'],bool)
...
np.full(T,float(rw))
```

iii. The trajectory gives no separate processing rationale beyond using the synchronized behavior fields directly.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`.

ii. ```python
visual_values=sorted({str(x) for sid in sessions for x in np.asarray(beh_by_session[sid]['WallName'])})
...
walls=np.asarray(b['WallName']).astype(str)
...
np.full(T,visual_id[wall],dtype=np.int64)
```

iii. The trajectory describes these as the “literal published WallName categories.”

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script does not collapse wall textures to four base categories. Instead it collects all unique literal `WallName` strings across the dataset, sorts them, assigns integer ids, and broadcasts the trial’s wall label across all frames.

ii. ```python
visual_values=sorted({str(x) for sid in sessions for x in np.asarray(beh_by_session[sid]['WallName'])})
visual_id={x:i for i,x in enumerate(visual_values)}
...
yy=np.vstack((np.full(T,visual_id[wall],dtype=np.int64),lick,pbin,sbin))
```

iii. In the trajectory, the agent explicitly says the “global visual labels are the literal published WallName categories.”

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`.

ii. ```python
lickfr=np.rint(np.asarray(b['LickFr'],float)).astype(int)
lickset=set(lickfr[(lickfr>=0)&(lickfr<nframes)].tolist())
```

iii. The trajectory uses `LickFr` as the frame-indexed licking source already synchronized to imaging frames.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The script rounds `LickFr` to the nearest integer frame with `np.rint`, drops out-of-range lick frames, converts the result to a set, and marks each trial frame as 1 if its frame index is in that set and 0 otherwise.

ii. ```python
lickfr=np.rint(np.asarray(b['LickFr'],float)).astype(int)
lickset=set(lickfr[(lickfr>=0)&(lickfr<nframes)].tolist())
...
lick=np.fromiter((1 if q in lickset else 0 for q in inds),dtype=np.int64,count=T)
```

iii. The trajectory says trial bounds use `ceil` and “event frame for licking uses nearest frame.”

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is evaluated on the same `inds` frame indices used for the neural trial slice.

ii. ```python
nn=np.asarray(spk[:,i:j],dtype=np.float32)
...
lick=np.fromiter((1 if q in lickset else 0 for q in inds),dtype=np.int64,count=T)
```

iii. The trajectory’s general alignment argument is that behavior and neural data are already synchronized frame-by-frame.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`.

ii. ```python
pos=np.asarray(b['ft_Pos'],float)
...
pbin=np.clip(np.floor(pos[i:j]/10.0),0,3).astype(np.int64)
```

iii. The trajectory interprets `ft_Pos` as the synchronized corridor-position signal and says the source units correspond to 10 units per meter.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The script divides position by 10, floors it, and clips to bins `0..3`, giving four 1 m bins over the 4 m textured corridor.

ii. ```python
pbin=np.clip(np.floor(pos[i:j]/10.0),0,3).astype(np.int64)
```

iii. In the trajectory, the agent says these are the requested four 1 m spatial bins from the 40-unit textured segment.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded by `floor(ft_Pos / 10)` with clipping to the four category ids `0, 1, 2, 3`.

ii. ```python
'output_values':[visual_values,['not licking','licking'],
                 ['0-1 m','1-2 m','2-3 m','3-4 m'],
                 ['0-25%','25-50%','50-75%','75-100%']],
...
pbin=np.clip(np.floor(pos[i:j]/10.0),0,3).astype(np.int64)
```

iii. The trajectory gives the same justification as 9-b: source position units are treated as decimeters / 10 units per meter.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is taken on the same kept frame window `i:j` used for the neural data.

ii. ```python
nn=np.asarray(spk[:,i:j],dtype=np.float32)
pbin=np.clip(np.floor(pos[i:j]/10.0),0,3).astype(np.int64)
```

iii. The trajectory’s alignment rationale is that all behavior is already synchronized to imaging frames.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii. ```python
speed=np.asarray(b['ft_RunSpeed'],float)
...
all_speed.append(np.asarray(b['ft_RunSpeed'])[i:j])
...
sbin=np.digitize(speed[i:j],speed_edges,right=False).astype(np.int64)
```

iii. The trajectory says global speed quartiles should be computed from corridor-valid synchronized frames.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Before per-session processing, the script collects all speed values from every kept corridor frame across all sessions, computes global 25th/50th/75th percentile edges with `np.nanquantile`, and later digitizes each trial frame against those global edges.

ii. ```python
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

iii. In the trajectory, the agent explicitly says it will compute “global running-speed quartiles” and that each bin should contain 25% of the data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded by global percentile edges at 25%, 50%, and 75%, then binned with `np.digitize`.

ii. ```python
speed_edges=np.nanquantile(np.concatenate(all_speed),[.25,.5,.75]).astype(float)
...
sbin=np.digitize(speed[i:j],speed_edges,right=False).astype(np.int64)
```

iii. The trajectory justifies this as matching the task request for four bins each containing 25% of the data.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is taken over the same trial frame window `i:j` used for the neural data.

ii. ```python
nn=np.asarray(spk[:,i:j],dtype=np.float32)
sbin=np.digitize(speed[i:j],speed_edges,right=False).astype(np.int64)
```

iii. The trajectory again relies on the fact that the behavioral frame streams are already synchronized to imaging frames.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script clips each trial window to `nframes = min(spk.shape[1], len(b['ft']))`, drops trials whose resulting interval is empty, drops lick events outside the imaged frame range, canonicalizes `_swap1`/`_swap2` aliases to physical sessions, and assigns `unassigned` if retinotopy is missing.

ii. ```python
nframes=min(spk.shape[1],len(b['ft']))
...
i=max(0,int(np.ceil(st))); j=min(nframes,int(np.ceil(en)))
if j<=i: continue
...
lickfr=np.rint(np.asarray(b['LickFr'],float)).astype(int)
lickset=set(lickfr[(lickfr>=0)&(lickfr<nframes)].tolist())
...
if os.path.exists(rf):
    ...
else: rid=np.full(ns,4,dtype=np.int64)
```

iii. The trajectory explicitly discusses two such handling rules: test3 aliases are alternate analysis views that should be canonicalized, and frame windows / lick frames should be clipped to the imaged interval because behavior is already synchronized but may extend beyond imaged data.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant costs are loading each giant spike file, concatenating the `spks` arrays, processing every session/trial, and then writing a very large pickle. The script also performs an additional full-dataset pass to compute global speed quartiles.

ii. ```python
obj=np.load(spk_files[sid],allow_pickle=True).item()
planes=obj['spks']
spk=np.concatenate([np.asarray(x) for x in planes],axis=0)
...
speed_edges=np.nanquantile(np.concatenate(all_speed),[.25,.5,.75]).astype(float)
...
with open(OUT,'wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory repeatedly notes that the spike files total about 405 GB, that conversion will be I/O-heavy and slow, and later that the output pickle becomes extremely large.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial Python loop inside each session, the loop that builds licking by membership testing frame-by-frame, and the first full-dataset loop that accumulates `all_speed` could all have been vectorized further.

ii. ```python
for sid in sessions:
    ...
    for st,en in zip(np.asarray(b['StartFr']),np.asarray(b['GrayFr'])):
        ...
        all_speed.append(np.asarray(b['ft_RunSpeed'])[i:j])
...
for tr,(st,en,sf,wall,rw) in enumerate(zip(starts,ends,sounds,walls,rewarded)):
    ...
    lick=np.fromiter((1 if q in lickset else 0 for q in inds),dtype=np.int64,count=T)
```

iii. The trajectory does not give an explicit vectorization discussion, but it does acknowledge the conversion is I/O-heavy and built around straightforward per-session/per-trial loops.

## 12-c. What processing does the code repeat multiple times?

i. It recomputes `training_day(sid)` inside every trial, repeatedly converts behavior fields to arrays inside the session loop, and performs two passes over the dataset: one for global speed quantiles and one for session conversion.

ii. ```python
all_speed=[]
for sid in sessions:
    ...
speed_edges=np.nanquantile(np.concatenate(all_speed),[.25,.5,.75]).astype(float)
...
for si,sid in enumerate(sessions):
    ...
    xx=np.vstack((t_sound,
                  np.full(T,training_day(sid)),
                  t_since,
                  np.full(T,float(rw)))).astype(np.float32)
```

iii. The trajectory explicitly says it will compute global speed quartiles first and then run the full conversion, but it does not otherwise call out the repeated `training_day` or per-trial repeated work.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There is one obviously discarded computation: `t_since` is first computed from `ft` and immediately overwritten by a second formula, so the first value is unused. More broadly, the script also stores `neural` as `float32` instead of a smaller dtype, which inflates a pickle the trajectory later reports as 277 GB.

ii. ```python
nn=np.asarray(spk[:,i:j],dtype=np.float32)
t_since=(ft[i:j]-float(st if st<nframes else i)*0)*86400.0
# Use frame interval rather than absolute MATLAB datenums to avoid cancellation.
t_since=(inds-float(st))*dt
```

iii. The trajectory does not explicitly discuss this overwritten `t_since` line. It does, however, acknowledge after conversion that the retained dataset is exceptionally large.
