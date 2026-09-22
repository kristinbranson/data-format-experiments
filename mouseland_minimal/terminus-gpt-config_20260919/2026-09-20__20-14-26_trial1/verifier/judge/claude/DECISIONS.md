# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the master index `beh/Imaging_Exp_info.npy`. Instead it discovers sessions from the filesystem: it globs every `beh/Beh_*.npy` file (in sorted filename order), loads each one, and builds a dictionary `beh_by_session` keyed by the session id, keeping the first record seen for each id. In parallel it globs `spk/*_neural_data.npy` to build `spk_files`. The session list is the intersection of the two key sets, sorted by (mouse, date, block); if any spike file has no matching behavior record the script raises. Per session it then loads (1) the behavior dict, (2) the spike file (dict with a `spks` list of three per-plane arrays, concatenated on the neuron axis exactly as `code/utils.py:load_spk` does), and (3) the retinotopy file `retinotopy/<mouse>_<date>_trans.npz` for `iarea`. Behavior for all sessions is loaded once up front and held in memory for the whole run; spike files are loaded one at a time and freed (`del spk,obj,planes; gc.collect()`). This yields 89 sessions / 19 mice / 38,110 trials / 4,691,034 neurons — the same session and trial inventory as the human reference.

ii.
```python
beh_by_session={}
source_by_session={}
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
if set(spk_files)-set(beh_by_session):
    raise RuntimeError('Spike sessions without synchronized behavior: '+repr(sorted(set(spk_files)-set(beh_by_session))))
```
```python
obj=np.load(spk_files[sid],allow_pickle=True).item()
planes=obj['spks']
# Same operation as code/utils.py:load_spk.
spk=np.concatenate([np.asarray(x) for x in planes],axis=0)
```

iii. From the trajectory: "Behavior files are dictionaries keyed by session ID, and each synchronized session already includes frame-level trial indices, position, running speed, wall ID, and frame indices for start/end/sound/lick/reward. This is ideal for conversion and avoids reimplementing raw timestamp synchronization" (step 11). And: "The experiment metadata contains intentional aliases and duplicates, but the 89 spike files correspond to unique physical sessions. The paper's loader concatenates all three supplied `spks` arrays, so the conversion should preserve that behavior" (step 16). The spike files were therefore treated as the authoritative session list, with `Imaging_Exp_info.npy` used only to understand the grouping.

## 1-b. How are the data split into subjects (mice)?

i. The mouse name is the first underscore-separated field of the session id (`parse_sid`), which is the same string as `mname` in the experiment index. `subjects` is the sorted set of those names (19 mice) and `subject_idx` is the index of each session's mouse into that list.

ii.
```python
def parse_sid(s):
    p=s.split('_')
    return p[0], '_'.join(p[1:4]), p[4]

subjects=sorted({parse_sid(s)[0] for s in sessions})
subject_id={s:i for i,s in enumerate(subjects)}
...
'subject_idx':np.asarray([subject_id[x['subject']] for x in session_info],dtype=np.int64),
```

iii. No explicit justification beyond the session-id parsing fix in step 19: "session IDs were parsed incorrectly: their structure is `subject_YYYY_MM_DD_block` ... This also affects sorting, subject/date metadata, and retinotopy filename construction", after which a single robust parser was introduced and used everywhere.

## 1-c. How are the data split into sessions?

i. A session is one spike file, i.e. one mouse / date / block. Behavior records that are aliases of the same physical recording are collapsed: the same session id appears in several `Beh_*.npy` files (e.g. `Beh_test1_before_grating` and `Beh_test2_before_grating`), and the `*_test3` files key a recording as `<sid>_swap1` / `<sid>_swap2`. The AI strips the `_swap1/_swap2` suffix with a regex and keeps only the first record encountered for each canonical id (sorted filename order), so exactly one behavior record is used per spike file. Sessions that survive the whole pipeline with fewer than 2 trials would be skipped (`if len(sn)<2: continue`); no session was actually dropped — all 89 were written.

ii.
```python
# Test3 stores two analysis views (_swap1/_swap2) of one physical
# recording. Both have the same synchronized frames and include the
# literal swapped WallName labels; match either view to the one spike file.
sid=re.sub(r'_swap[12]$', '', raw_sid)
if sid not in beh_by_session:
    beh_by_session[sid]=b
```
```python
if len(sn)<2: continue
```

iii. Step 18: "Test3 behavior keys append `_swap1` or `_swap2` to the physical session ID. Paired records have identical frame/trial ranges and contain both swapped stimulus labels; they are alternate analysis views of the same recording, not separate neural sessions. The correct deduplication is therefore to canonicalize these suffixes and retain one record per physical spike session." Step 13 had already noted that "behavioral experiment files contain overlapping session sets ... so blindly processing every file would duplicate sessions".

## 1-d. How are the data split into trials?

i. A trial is the span of imaging frames from corridor entry to the start of the grey space: `i = ceil(StartFr[t])` to `j = ceil(GrayFr[t])` (half-open), clipped to the number of usable frames `nframes = min(spk.shape[1], len(ft))`. Trials with `j <= i` are skipped. All `ntrials` trials of every session are otherwise used (38,110 trials total). Trials are variable length (median 22–23 frames). I verified this window is numerically equivalent to the reference's `(ft_trInd == t) & ft_CorrSpc` mask: identical trial counts, identical median length, and lengths differing by at most one frame on 33/453 trials in the session I checked, with identical position ranges (0–40 source units = the 4 m texture).

ii.
```python
starts=np.asarray(b['StartFr']); ends=np.asarray(b['GrayFr'])
...
for tr,(st,en,sf,wall,rw) in enumerate(zip(starts,ends,sounds,walls,rewarded)):
    i=max(0,int(np.ceil(st))); j=min(nframes,int(np.ceil(en)))
    if j<=i: continue
    inds=np.arange(i,j); T=j-i
```

iii. Step 16: "`StartFr` marks corridor entry; `GrayFr` marks transition out of the 4-m visual corridor into the 2-m gray interval, so slicing `[ceil(StartFr), ceil(GrayFr))` naturally gives corridor-aligned trials and four 10-source-unit (1-m) position bins." Step 14 established that "behavioral event frame indices are fractional because they were interpolated from timestamps", motivating the `ceil` rounding.

## 1-e. How are trials filtered based on quality controls?

i. Essentially no quality filtering is applied. The only rejections are (a) trials whose frame window is empty after truncation to the imaged frames (`j <= i`), and (b) sessions left with fewer than two trials (never triggered). All 38,110 trials are kept, including 382 extreme-duration trials (> 238.9 frames, the 99th percentile) in which the animal stopped inside the corridor. Those 382 trials (1.0% of trials) contribute 216,868 of the 1,375,142 retained time bins (15.8% of all data); 74.9% of their frames are at zero or negative running speed (versus 21.8% in ordinary trials) and 45.1% of their frames sit in the first 1-m position bin. The AI never discusses trial-quality filtering anywhere in the trajectory.

ii.
```python
i=max(0,int(np.ceil(st))); j=min(nframes,int(np.ceil(en)))
if j<=i: continue
```
```python
if len(sn)<2: continue
```

iii. No justification is given; the possibility of outlier/stationary trials is never raised. The only documented curation reasoning is about not over-filtering neurons: "paper-specific selective-neuron analyses ... likely should not be applied here because the decoder needs all curated neural activity rather than only stimulus-selective neurons" (step 8).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy` — a list of three arrays of shape (n_neurons_plane, n_frames), float32, concatenated along the neuron axis, exactly as `utils.load_spk` does. Per-neuron area comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz` (length equals the concatenated neuron count in all 89 sessions).

ii.
```python
obj=np.load(spk_files[sid],allow_pickle=True).item()
planes=obj['spks']
# Same operation as code/utils.py:load_spk.
spk=np.concatenate([np.asarray(x) for x in planes],axis=0)
nframes=min(spk.shape[1],len(b['ft']))
```

iii. Step 12: "The repository performs no extra neural normalization or quality filtering in `load_spk`; the supplied `spks` are already processed neural activity." Step 13: "Each stores three float32 `spks` matrices with identical neuron/time dimensions, and the paper code concatenates these along the neuron axis."

## 2-b. How is the `neural` data processed?

i. Not processed at all: the per-trial matrix is the raw column slice `spk[:, i:j]` of the published deconvolved traces, copied as **float32**. No smoothing, no normalization, no deconvolution, no neuron subselection, no padding. Because 4.69 M neurons × all frames are stored in float32, the resulting pickle is 296 GB (277 GiB); the provided verifier needed tens of minutes just to `pickle.load` it.

ii.
```python
# Neural data are published deconvolved activity, no added smoothing/normalization.
nn=np.asarray(spk[:,i:j],dtype=np.float32)
...
'neural_processing':'Published spks; concatenated exactly as repository utils.load_spk; no additional smoothing, normalization, or neuron filtering.',
```

iii. Step 12/16: the repository loader applies no normalization, so the conversion "preserves the paper neural loading (concatenate all `spks`)". Step 40 acknowledges the size consequence: "The pickle is exceptionally large because the paper loader concatenates three full activity arrays and the conversion preserves all unique sessions/trials."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed — every neuron in `spks` is kept. The retinotopy `iarea` is used only to label neurons, with the mapping `1→V1, 2→medial, 3→anterior, 4→lateral`, everything else → a fifth `unassigned` category. This mapping contradicts the repository's own `utils.neu_area_ID`, where `V1 = (iarea==8)`, `mHV = iarea in {0,1,2,9}`, `lHV = iarea in {5,6}`, `aHV = iarea in {3,4}`. Consequently 3,736,488 of 4,691,034 neurons (79.7%) are labelled `unassigned` — including all 1,833,035 true V1 neurons (39.1%) — and the 147,071 neurons labelled `V1` are actually medial higher visual area neurons. The verifier printed this breakdown ("V1: 147071 neurons, medial: 139295, anterior: 568741, lateral: 99439, unassigned: 3736488") and the AI did not react. There is also a `np.tile` fallback that silently repeats the area vector if it is shorter than the neuron count (a no-op here, since `len(iarea) == n_neurons` in all 89 sessions).

ii.
```python
# Retinotopy iarea uses 1=V1, 2=medial, 3=anterior, 4=lateral;
# one map applies to each of the three concatenated arrays.
rf=os.path.join(ROOT,'retinotopy',mname+'_'+date+'_trans.npz')
if os.path.exists(rf):
    ar=np.asarray(np.load(rf)['iarea']).astype(int)
    rid=np.full(ar.shape,4,dtype=np.int64)
    for raw,outid in ((1,0),(2,1),(3,2),(4,3)): rid[ar==raw]=outid
    reps=int(np.ceil(ns/len(rid))); rid=np.tile(rid,reps)[:ns]
else: rid=np.full(ns,4,dtype=np.int64)
brain_regions=['V1','medial','anterior','lateral','unassigned']
```

iii. The "no filtering" part is justified in step 8: "the decoder needs all curated neural activity rather than only stimulus-selective neurons" (the data were already curated by Suite2p's cell classifier). The area codes were never verified: the AI read `load_retino` (which shows `areasN = ['All','V1','medial','anterior','lateral']` and calls `neu_area_ID`) but never read the body of `neu_area_ID`, and inferred the integer codes from the position of the names in `areasN`. Step 16 only planned to "Inspect the short region-mapping function"; the trajectory shows it never did.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial starts at its own corridor entry, `ceil(StartFr)`, and runs to the grey-space transition, so bin 0 of every trial is the first imaged frame at/after corridor entry. Trials keep their natural, variable length; nothing is padded or truncated to a common window. Metadata records `temporal_alignment_event = 'trial start (entry into the 4-m visual corridor; StartFr)'`, `off_start = 0.0`, `off_end = None`. Inputs and outputs are sliced with exactly the same `[i:j)` frame window, so all streams are aligned by construction.

ii.
```python
i=max(0,int(np.ceil(st))); j=min(nframes,int(np.ceil(en)))
inds=np.arange(i,j); T=j-i
nn=np.asarray(spk[:,i:j],dtype=np.float32)
...
'temporal_alignment_event':'trial start (entry into the 4-m visual corridor; StartFr)',
'off_start':0.0,'off_end':None,
'trial_window':'ceil(StartFr) through frame before ceil(GrayFr), excluding the following gray space.',
```

iii. Step 16: slicing `[ceil(StartFr), ceil(GrayFr))` "naturally gives corridor-aligned trials". Step 11 notes the behavior is already synchronized to imaging frames, so alignment across streams needs nothing beyond using the same frame indices.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame is the bin; no rebinning, resampling or interpolation is done. The bin size is computed per session as the median of `diff(ft)` converted from MATLAB datenum days to seconds, and the metadata value is the median of those across sessions × 1000 ms ≈ 315 ms (≈3.17 Hz). The per-session `dt` is also what the two time-valued inputs are built from.

ii.
```python
dt=float(np.nanmedian(np.diff(ft))*86400.0)
...
'time_bin_size':float(np.median([x['median_frame_interval_s'] for x in session_info]))*1000.0,
'median_frame_interval_s':dt   # stored per session in session_info
```

iii. Step 14: "Frame timestamps show an imaging interval of about 0.315 s (~3.18 Hz), and behavioral event frame indices are fractional because they were interpolated from timestamps." Since the behavior is already supplied on the imaging-frame grid, no resampling was considered necessary.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (fractional frame index of the sound cue in each trial) and the frame-time grid, represented by the per-session median frame interval `dt` derived from `ft`.

ii.
```python
sounds=np.asarray(b['SoundFr'])
dt=float(np.nanmedian(np.diff(ft))*86400.0)
...
t_sound=(float(sf)-inds)*dt
```

iii. Step 11 identifies `SoundFr` as one of the already frame-synchronized event fields; no separate justification is given for preferring `SoundFr` over `SoundTime`/`SoundPos` beyond that synchronization.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame index in the trial window, the signed distance in frames to the cue is multiplied by the session's median frame interval, giving seconds: positive before the cue, negative after. It is a continuous time-varying input of length T. (Note: the value is computed from frame-index differences × median `dt` rather than from the actual `ft` timestamps of each frame; with a regular ~3.17 Hz frame clock these agree to within frame jitter.)

ii.
```python
t_sound=(float(sf)-inds)*dt
xx=np.vstack((t_sound,
              np.full(T,training_day(sid)),
              t_since,
              np.full(T,float(rw)))).astype(np.float32)
```

iii. Step 19: "the requested 'time to sound cue' should conventionally be cue time minus current time; the script currently has the opposite sign" — the sign was deliberately flipped so that the value counts down to the cue. The code comment explains the use of `dt`: "Use frame interval rather than absolute MATLAB datenums to avoid cancellation" (i.e. avoiding float precision loss from large datenum values).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on `inds = arange(i, j)`, the exact frame indices used to slice the neural matrix, so it has length T and is sample-for-sample aligned with the neural bins.

ii.
```python
inds=np.arange(i,j); T=j-i
nn=np.asarray(spk[:,i:j],dtype=np.float32)
t_sound=(float(sf)-inds)*dt
```

iii. Step 11: the behavioral streams are already interpolated to imaging frames, so using the same frame window is sufficient for alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date embedded in the session id (`subject_YYYY_MM_DD_block`), i.e. the same information as `datexp` in the experiment index. No explicit training-day field (such as `days` in a few `Imaging_Exp_info` entries) is used.

ii.
```python
subject_dates=defaultdict(list)
for sid in sessions:
    m,d,_=parse_sid(sid); subject_dates[m].append(d)
first_date={m:min(v) for m,v in subject_dates.items()}
```

iii. Step 16 planned to "assign chronological training-day rank per subject"; the final implementation uses calendar-day offsets instead. The code comment states the rationale: "Subject-specific chronological session rank is a reproducible continuous day covariate (calendar days from that animal's first included recording)."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each session, the number of calendar days between its date and the earliest included recording date of the same mouse, as a float (range 0–92 across the dataset, mean ≈ 20.7). The value is constant within a session and broadcast across all bins of every trial of that session. (The human reference instead uses the ordinal rank of the recording within the mouse, range 0–7.)

ii.
```python
from datetime import datetime
def training_day(sid):
    m,d,_=parse_sid(sid)
    return float((datetime.strptime(d,'%Y_%m_%d')-datetime.strptime(first_date[m],'%Y_%m_%d')).days)
...
np.full(T,training_day(sid)),
...
'day_definition':'Calendar days since each subject first included imaging session.',
```

iii. Code comment: a calendar-day offset from the animal's first included recording is a "reproducible continuous day covariate". Note `training_day(sid)` is re-evaluated (including two `strptime` calls) inside the per-trial loop, i.e. ~38,000 times rather than 89.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr` (fractional frame index of corridor entry) and the per-session median frame interval `dt` derived from `ft`.

ii.
```python
starts=np.asarray(b['StartFr'])
dt=float(np.nanmedian(np.diff(ft))*86400.0)
t_since=(inds-float(st))*dt
```

iii. Step 16: "`StartFr` marks corridor entry"; step 14 notes these frame indices are fractional because they were interpolated from timestamps.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The frame offset from the fractional entry frame is multiplied by `dt`, giving seconds elapsed since corridor entry: it starts between 0 and one frame interval and increases monotonically through the trial. It is continuous and time-varying. There is one line of dead code: an earlier absolute-datenum formulation is computed and then immediately overwritten by the frame-interval formulation.

ii.
```python
t_since=(ft[i:j]-float(st if st<nframes else i)*0)*86400.0   # computed, then discarded
# Use frame interval rather than absolute MATLAB datenums to avoid cancellation.
t_since=(inds-float(st))*dt
```

iii. The code comment gives the reason for the second form: absolute MATLAB datenums are large numbers whose differences lose precision, so the frame index × frame interval is used instead. The leftover first line is an artifact of that edit (step 19's fixes) that was never removed.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same as the other time-varying streams: evaluated on `inds = arange(i, j)`, the frames used for the neural slice, so it is length T and bin-for-bin aligned; bin 0 corresponds to the first imaged frame at/after corridor entry (`off_start = 0.0`).

ii.
```python
inds=np.arange(i,j)
nn=np.asarray(spk[:,i:j],dtype=np.float32)
t_since=(inds-float(st))*dt
```

iii. Alignment follows from the already-synchronized frame grid (step 11) and the trial-start alignment decision (step 16).

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From the per-trial boolean `isRew`, which marks trials run in the rewarded corridor.

ii.
```python
rewarded=np.asarray(b['isRew'],bool)
...
for tr,(st,en,sf,wall,rw) in enumerate(zip(starts,ends,sounds,walls,rewarded)):
```

iii. Step 11: `isRew` is one of the trial-level reward fields available in the synchronized behavior dictionaries. No further justification is given.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean is cast to float (0.0/1.0) and broadcast across all T bins of the trial, so it is a constant time series per trial, as requested by the instructions (1 if in rewarded corridor, else 0).

ii.
```python
np.full(T,float(rw))
```

iii. None given; the field is used as-is.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the per-trial name of the wall texture. (`TrialStim`/`stim_id` are not used.)

ii.
```python
walls=np.asarray(b['WallName']).astype(str)
visual_values=sorted({str(x) for sid in sessions for x in np.asarray(beh_by_session[sid]['WallName'])})
visual_id={x:i for i,x in enumerate(visual_values)}
```

iii. Step 5: the behavioral files contain "trial-level sound, reward, corridor (`WallType`), and stimulus (`WallName`) fields". The code comment states the choice: "Global visual labels are the literal published WallName categories."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. No grouping is applied: the 15 distinct `WallName` strings that occur anywhere in the dataset (`circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5`) are sorted and used directly as the 15 category values, shared globally across sessions. The per-trial index is broadcast across all T bins. The human reference instead maps these 15 names onto the 4 base textures (circle/leaf/rock/wood), which is also how the paper pools them (e.g. Methods: the two leaf1-swap stimuli "are pooled together for statistics analysis"). Within any one session only ~2–8 of the 15 labels occur, so most classes are absent from any given decoder fit.

ii.
```python
visual_values=sorted({str(x) for sid in sessions for x in np.asarray(beh_by_session[sid]['WallName'])})
...
yy=np.vstack((np.full(T,visual_id[wall],dtype=np.int64),lick,pbin,sbin))
...
'output_values':[visual_values, ...]
```

iii. Code comment only: the labels are "the literal published WallName categories". The trajectory contains no discussion of collapsing crops/swaps into texture categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the (fractional) imaging-frame index of every detected lick in the session.

ii.
```python
lickfr=np.rint(np.asarray(b['LickFr'],float)).astype(int)
lickset=set(lickfr[(lickfr>=0)&(lickfr<nframes)].tolist())
```

iii. Step 9/11: `LickFr` (with `LickTrind`) is listed among the "frame-aligned behavioral fields", indicating the repository already interpolated licks onto imaging frames.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame indices are rounded to the **nearest** frame (`np.rint`), out-of-range values are dropped, and the survivors are put into a Python `set`. For each trial, a binary vector is built by testing membership of each frame index in that set: 1 if at least one lick was assigned to that frame, else 0. Output values are `['not licking','licking']`. (The human reference truncates instead of rounding, i.e. assigns a lick to the frame interval it falls in; the difference is at most one frame.)

ii.
```python
lick=np.fromiter((1 if q in lickset else 0 for q in inds),dtype=np.int64,count=T)
...
'frame_index_rounding':'Trial bounds use ceil; event frame for licking uses nearest frame.',
'output_values':[visual_values,['not licking','licking'], ...]
```

iii. Metadata comment: event frames are fractional, so the lick is attributed to the nearest frame; this is documented as an explicit rounding convention alongside the `ceil` used for trial bounds.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary vector is built over `inds = arange(i, j)` — the same frames used for the neural slice — so it is length T and bin-aligned with the neural data.

ii.
```python
inds=np.arange(i,j)
nn=np.asarray(spk[:,i:j],dtype=np.float32)
lick=np.fromiter((1 if q in lickset else 0 for q in inds),dtype=np.int64,count=T)
```

iii. As for the other streams: `LickFr` is already expressed in imaging frames, so using the trial's frame window aligns it.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the per-imaging-frame position in the corridor in source units (10 units = 1 m; 0–40 across the textured corridor, continuing to 60 through the grey space).

ii.
```python
pos=np.asarray(b['ft_Pos'],float)
pbin=np.clip(np.floor(pos[i:j]/10.0),0,3).astype(np.int64)
```

iii. Step 14: "corridor position is 0–60 in source units, with 20 units of gray space and 40 units of textured corridor. Thus the requested four 1-m spatial bins correspond to the 40-unit textured segment after converting source units (10 units/m)."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 10 (units → metres) and floored to an integer bin, then clipped to 0–3. Because trials end at `GrayFr`, `ft_Pos` within the window stays in [0, 40), so the clip is effectively a safety net. The result is a time-varying categorical of length T.

ii.
```python
pbin=np.clip(np.floor(pos[i:j]/10.0),0,3).astype(np.int64)
...
'position_processing':'Published ft_Pos in 10 units/m, floor(ft_Pos/10), clipped to bins 0..3.',
```

iii. Step 14/16, as above: restricting the trial to `[StartFr, GrayFr)` means the corridor position naturally spans the 4 m texture, so `floor(ft_Pos/10)` gives the four 1-m bins the instructions ask for.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four fixed, equal-length 1-m bins with edges at 0/10/20/30/40 source units, labelled `['0-1 m','1-2 m','2-3 m','3-4 m']` — exactly the discretization the instructions prescribe. The realized marginal distribution is [0.285, 0.233, 0.235, 0.247] rather than approximately uniform, because the retained stationary outlier trials (see 1-e) pile 45% of their bins into the first metre.

ii.
```python
pbin=np.clip(np.floor(pos[i:j]/10.0),0,3).astype(np.int64)
'output_values':[..., ['0-1 m','1-2 m','2-3 m','3-4 m'], ...]
```

iii. Directly from the Decoder Task specification ("discretized into 4 equal-length, 1-m-long spatial bins").

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is sliced with the same `[i:j)` frame window as the neural matrix, so it is length T and bin-aligned.

ii.
```python
nn=np.asarray(spk[:,i:j],dtype=np.float32)
pbin=np.clip(np.floor(pos[i:j]/10.0),0,3).astype(np.int64)
```

iii. `ft_Pos` is supplied on the imaging-frame grid (step 11), so slicing the same frames aligns it.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the per-imaging-frame running speed.

ii.
```python
speed=np.asarray(b['ft_RunSpeed'],float)
```

iii. Step 5/11: running speed is one of the frame-level behavioral streams already synchronized to imaging frames.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A pre-pass concatenates `ft_RunSpeed` over the corridor window of every trial of every one of the 89 sessions, and takes the global 25/50/75% quantiles as fixed bin edges. Each trial's speeds are then assigned with `np.digitize(..., right=False)`. The edges are stored in metadata. The scope is global (all sessions pooled), whereas the human reference computes the split within each session.

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
...
'speed_quartile_scope':'All valid corridor frames across all unique sessions.',
```

iii. Code comment: "Global quartiles over valid corridor frames, as requested (each bin is 25% of data)."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins defined by the three global quantile edges, labelled `['0-25%','25-50%','50-75%','75-100%']`. The realized edges are `[0.0, 8.381, 30.189]`, and because 30.2% of the retained frames have speed ≤ 0 (the animal is stationary, largely in the unfiltered outlier trials of 1-e) and `digitize(..., right=False)` sends every value equal to the first edge into bin 1, the actual bin occupancies are **[9.8%, 40.2%, 25.0%, 25.0%]**, not the 25% each the instructions require and the labels claim. The lowest bin ends up containing only frames with *negative* speed. (Had the stationary outlier trials been excluded, the same quantile procedure would have produced exactly [25%, 25%, 25%, 25%]; the human reference instead sidesteps the tie entirely with a rank-based split.)

ii.
```python
speed_edges=np.nanquantile(np.concatenate(all_speed),[.25,.5,.75]).astype(float)
sbin=np.digitize(speed[i:j],speed_edges,right=False).astype(np.int64)
'output_values':[..., ['0-25%','25-50%','50-75%','75-100%']],
'speed_quartile_edges':speed_edges.tolist(),
```

iii. Code comment: "Global quartiles over valid corridor frames, as requested (each bin is 25% of data)." The tie mass at zero speed is never examined, and the resulting bin occupancies are never checked.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Sliced with the same `[i:j)` frame window as the neural matrix; length T, bin-aligned.

ii.
```python
nn=np.asarray(spk[:,i:j],dtype=np.float32)
sbin=np.digitize(speed[i:j],speed_edges,right=False).astype(np.int64)
```

iii. `ft_RunSpeed` is on the imaging-frame grid, so the same frame window aligns it.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive measures: (a) behavior can run past the imaging, so everything is truncated to `nframes = min(spk.shape[1], len(ft))` and trial ends are clipped to it; (b) licks outside `[0, nframes)` are discarded; (c) trials whose window is empty are skipped; (d) sessions with fewer than two usable trials are skipped; (e) a missing retinotopy file falls back to labelling every neuron `unassigned`; (f) a spike file with no matching behavior record raises rather than being silently dropped; (g) `np.nanmedian`/`np.nanquantile` are used for the frame interval and speed edges. In this dataset none of (c)–(e) actually triggered, and the relevant event fields contain no NaNs. Two silent-failure modes remain: the retinotopy fallback and the `np.tile` repetition of the area vector would both fabricate region labels instead of erroring.

ii.
```python
nframes=min(spk.shape[1],len(b['ft']))
...
lickset=set(lickfr[(lickfr>=0)&(lickfr<nframes)].tolist())
...
if j<=i: continue
if len(sn)<2: continue
...
if os.path.exists(rf): ...
else: rid=np.full(ns,4,dtype=np.int64)
if set(spk_files)-set(beh_by_session):
    raise RuntimeError('Spike sessions without synchronized behavior: '+...)
```

iii. Step 17: 13 spike sessions initially failed to match a behavior key and the AI explicitly refused to drop them — "We must inspect those keys and map them correctly rather than silently discard sessions" — leading to the `_swap` canonicalization. The `RuntimeError` guard enforces that no session is lost silently.

## 12-a. What are the most time-consuming steps of the code?

i. (1) Reading the 89 spike files (405 GB) and concatenating the three plane arrays per session — this dominates and is largely unavoidable; the run took roughly 30–40 minutes for 89 sessions (~3 sessions/min). (2) Serializing the 296 GB pickle, and correspondingly `pickle.load`ing it in the verifier, which the trajectory shows taking tens of minutes (steps 36–42 are almost entirely spent waiting for `pickle.load` to return). This second cost is largely self-inflicted: storing float32 instead of float16 doubles it, and keeping the 79.7% of neurons the reference drops adds a further ~14%, so the file is ~2.3× the expert's. (3) The behavior pre-pass, which loads and retains all 6.6 GB of `Beh_*.npy` objects in memory for the entire run.

ii.
```python
obj=np.load(spk_files[sid],allow_pickle=True).item()
spk=np.concatenate([np.asarray(x) for x in planes],axis=0)
...
nn=np.asarray(spk[:,i:j],dtype=np.float32)
...
with open(OUT,'wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Steps 20–35 document the I/O-bound nature of the run: "The process is I/O-heavy and expected to require multiple minutes"; step 40: "The pickle is exceptionally large because the paper loader concatenates three full activity arrays and the conversion preserves all unique sessions/trials." The AI judged waiting to be safer than interrupting, and never revisited the dtype/size choice.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. (1) The per-trial licking vector, built with a Python generator doing set membership per frame (`np.fromiter((1 if q in lickset else 0 for q in inds), ...)`) — 1.37 M Python-level iterations overall, where a single `licking = np.zeros(nframes); licking[lick_frames] = 1` followed by slicing (the reference's approach) would be one vectorized operation. (2) The speed pre-pass loops over all 38,110 trials of all 89 sessions in Python to build frame windows, recomputing bounds that the main loop computes again; the windows could be derived from `ft_trInd`/`ft_CorrSpc` in one vectorized pass per session. (3) The main per-trial loop rebuilds `arange`, `full` and `vstack` arrays trial by trial. All of these are negligible next to the spike-file I/O, so the practical gain is small.

ii.
```python
lick=np.fromiter((1 if q in lickset else 0 for q in inds),dtype=np.int64,count=T)
```
```python
for sid in sessions:
    for st,en in zip(np.asarray(b['StartFr']),np.asarray(b['GrayFr'])):
        i=max(0,int(np.ceil(st))); j=min(n,int(np.ceil(en)))
        if j>i: all_speed.append(np.asarray(b['ft_RunSpeed'])[i:j])
```

iii. Not discussed anywhere in the trajectory; the AI made no efficiency analysis of its own loops.

## 12-c. What processing does the code repeat multiple times?

i. (1) `training_day(sid)` is called inside the per-trial loop, so two `datetime.strptime` parses run for every one of the 38,110 trials instead of once per session. (2) Trial frame bounds (`ceil(StartFr)`, `min(nframes, ceil(GrayFr))`) are computed once in the speed pre-pass and again in the main loop. (3) Every `Beh_*.npy` file is loaded even though only the first occurrence of each session id is kept, so duplicate records (25 session ids appear in 2–5 files) are deserialized and discarded. (4) `source_by_session`, `session_info` and metadata strings duplicate information already derivable from the session id. None of this is expensive relative to the spike I/O.

ii.
```python
xx=np.vstack((t_sound,
              np.full(T,training_day(sid)),   # re-parsed per trial
              t_since,
              np.full(T,float(rw)))).astype(np.float32)
```

iii. Not discussed in the trajectory.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) A genuinely dead statement: `t_since` is computed once from absolute MATLAB datenums and then overwritten on the next line by the frame-interval version — the first computation is pure waste, left over from the sign/parsing fixes of step 19. (2) The `np.tile` of the retinotopy area vector is a no-op in all 89 sessions (`len(iarea) == n_neurons` everywhere) and exists only because of a mistaken belief that one map covers each of the three plane arrays. (3) Every trial's neural slice is copied to a fresh float32 array (`np.asarray(spk[:, i:j], dtype=np.float32)`) although `spks` is already float32 — the copy is required to release the session array, but storing at this precision rather than float16 doubles the 296 GB output for no analysable gain. (4) 3.7 M neurons carrying no usable area label and 216,868 bins of stationary outlier trials are written out, both of which the expert solution excludes. (5) Bookkeeping fields such as `kept_trial_indices` and `behavior_source` are stored per session but unused downstream (harmless, and arguably useful provenance).

ii.
```python
t_since=(ft[i:j]-float(st if st<nframes else i)*0)*86400.0   # discarded immediately
t_since=(inds-float(st))*dt
```
```python
reps=int(np.ceil(ns/len(rid))); rid=np.tile(rid,reps)[:ns]   # always reps==1
```
```python
nn=np.asarray(spk[:,i:j],dtype=np.float32)
```

iii. Not discussed in the trajectory; the AI never audited the script for redundant work after its two rounds of patches (steps 18–19).
