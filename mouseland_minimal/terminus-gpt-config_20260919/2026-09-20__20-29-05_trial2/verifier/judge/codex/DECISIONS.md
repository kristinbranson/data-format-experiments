# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads behavior by globbing every `Beh_*.npy` file under `/app/data/beh`, taking dictionary entries that contain `ft`, and keeping the first occurrence of each session key. It separately builds a spike-file index from `/app/data/spk/*_neural_data.npy`, intersects behavior keys with spike basenames to define usable sessions, and then loads retinotopy per session from `/app/data/retinotopy/*_trans.npz`.

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
```python
rp=glob.glob(ROOT+'/retinotopy/'+('_'.join(sid.split('_')[:4]))+'_trans.npz')
if rp:
    ia=np.load(rp[0])['iarea']; ri=area_index(ia)
```

iii. In the trajectory, the AI said there were 76 usable sessions where behavior and neural data overlapped, and 13 neural recordings without matched behavior that “cannot support the requested decoder variables and should be excluded with documentation.” It also emphasized streaming one neural file at a time because the spike files total roughly 434 GB.

## 1-b. How are the data split into subjects?

i. Subjects are defined from the session id prefix before the first underscore, both when building the initial subject list and when assembling the final `subjects` and `subject_idx`.

ii. 
```python
subjects=sorted({s.split('_')[0] for s in sids}); subjmap={s:i for i,s in enumerate(subjects)}
```
```python
used_subjects=sorted({x['session_id'].split('_')[0] for x in session_info}); um={s:i for i,s in enumerate(used_subjects)}
data={'subjects':used_subjects,
      'subject_idx':np.asarray([um[x['session_id'].split('_')[0]] for x in session_info],dtype=np.int32),
```

iii. The trajectory says the matched dataset contains 76 sessions from 19 subjects, and treats the mouse prefix embedded in each session id as sufficient to identify the subject.

## 1-c. How are the data split into sessions?

i. Sessions are defined as the intersection of behavior-session keys and spike-file basenames, then sorted by mouse prefix, parsed date, and full session id. The AI does not use `Imaging_Exp_info.npy` as a master index or de-duplicate repeated experiment-type listings through that table.

ii. 
```python
beh,bsrc=load_behavior_sessions()
spk={os.path.basename(p).replace('_neural_data.npy',''):p for p in glob.glob(ROOT+'/spk/*_neural_data.npy')}
sids=sorted(set(beh)&set(spk),key=lambda s:(s.split('_')[0],sid_date(s),s))
if a.max_sessions: sids=sids[:a.max_sessions]
```

iii. In trajectory step 10, the AI concluded that there were 99 behavior sessions but only 76 that overlapped neural recordings, so those 76 were the usable sessions. It justified excluding the rest because the requested decoder variables require both neural and behavioral streams.

## 1-d. How are the data split into trials?

i. Trials are defined by `ft_trInd == trial`, but the retained bins within each trial are further restricted to frames that are moving, inside the textured corridor, finite in position, and between 0 and 40 position units. The result is not the full contiguous traversal; it is the subset of moving corridor samples for that trial.

ii. 
```python
def trial_mask(b,tr,n):
    tri=np.asarray(b['ft_trInd'])[:n]
    pos=np.asarray(b['ft_Pos'])[:n]
    moving=np.asarray(b['ft_isMoving'],bool)[:n]
    corr=np.asarray(b['ft_CorrSpc'],bool)[:n]
    return (tri==tr) & moving & corr & np.isfinite(pos) & (pos>=0) & (pos<40)
```
```python
for tr in range(int(b['ntrials'])):
    mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
    if ix.size<2: continue
```

iii. The trajectory repeatedly states that the paper analyzed only running periods and that the conversion should “retain only moving corridor frames” while keeping native temporal resolution for time-varying decoder targets.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped if fewer than 2 bins survive the moving-corridor mask. A whole session is dropped if fewer than 2 such trials remain. The AI does not apply the reference solution’s global long-trial percentile filter.

ii. 
```python
for tr in range(int(b['ntrials'])):
    mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
    if ix.size<2: continue
```
```python
if len(sn)<2:
    print('skip',sid,'fewer than 2 usable trials',flush=True); continue
```

iii. The trajectory cites the decoder requirement that each session must have at least two trials and treats the moving-corridor mask as the main quality-control rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the per-plane `spks` arrays in each session’s neural `.npy` file. Brain-region labels come from retinotopy `iarea` values in the corresponding `_trans.npz` file when present.

ii. 
```python
obj=np.load(spk[sid],allow_pickle=True).item()['spks']
X=np.concatenate([x[:,:n] for x in obj],axis=0)
```
```python
rp=glob.glob(ROOT+'/retinotopy/'+('_'.join(sid.split('_')[:4]))+'_trans.npz')
if rp:
    ia=np.load(rp[0])['iarea']; ri=area_index(ia)
```

iii. In trajectory steps 5 to 7, the AI noted that neural files contain one spike array per imaging plane and that retinotopy maps raw area ids into grouped visual regions.

## 2-b. How is the `neural` data processed?

i. The AI truncates the session to the minimum shared frame count between behavior and all spike planes, concatenates planes across neurons, optionally downsamples neurons to a deterministic cap of 2,000 per session, and stores each retained trial as a `float32` neuron-by-time array at native imaging resolution.

ii. 
```python
obj=np.load(spk[sid],allow_pickle=True).item()['spks']
n=min(len(b['ft']),min(x.shape[1] for x in obj))
X=np.concatenate([x[:,:n] for x in obj],axis=0)
```
```python
if a.max_neurons and X.shape[0]>a.max_neurons:
    rng=np.random.default_rng(0)
    ...
    chosen=np.sort(np.asarray(chosen)); X=X[chosen]; ri=ri[chosen]
```
```python
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. In trajectory steps 10 to 12, the AI said it wanted to keep native 3.17 Hz bins for time-varying variables but cap neurons at 2,000 because retaining 20,000 to 90,000 neurons per session would make the pickle and decoder training impractical.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are assigned to five region labels: `V1`, `mHV`, `lHV`, `aHV`, and `unassigned`. Unlike the reference, neurons with unassigned retinotopy are kept. If the retinotopy length does not match the neuron count, the region array is resized. Then, if necessary, neurons are subsampled to a deterministic region-stratified cap of 2,000 per session.

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
```python
if rp:
    ia=np.load(rp[0])['iarea']; ri=area_index(ia)
else: ri=np.full(X.shape[0],4,dtype=np.int8)
if len(ri)!=X.shape[0]:
    print('WARNING region/neuron mismatch',sid,len(ri),X.shape[0],flush=True)
    ri=np.resize(ri,X.shape[0]).astype(np.int8)
```
```python
if a.max_neurons and X.shape[0]>a.max_neurons:
    ...
```

iii. The trajectory says the AI wanted a tractable dataset size while “preserving every region” and handling missing or inconsistent retinotopy defensively rather than failing the conversion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to trial start only indirectly: the metadata says “trial start (entry into textured corridor),” but the actual per-trial neural matrix uses `X[:, ix]`, where `ix` are only the moving textured-corridor frames belonging to that trial. So the first retained bin is the first kept running frame after corridor entry, not necessarily the literal first corridor-entry frame.

ii. 
```python
for tr in range(int(b['ntrials'])):
    mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
    if ix.size<2: continue
    ...
    sn.append(np.asarray(X[:,ix],dtype=np.float32))
```
```python
'temporal_alignment_event':'trial start (entry into textured corridor)',
'off_start':0.0,
'off_end':None,
```

iii. The trajectory justifies this by saying the paper restricted analysis to running timepoints and that native frame-aligned time-varying variables would be lost by the paper’s spatial interpolation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native imaging frame rate of 3.17 Hz, with one time bin per imaging frame. No temporal rebinning is applied.

ii. 
```python
ROOT='/app/data'; FS=3.17
```
```python
'time_bin_size':1000.0/FS,
'native_imaging_rate_hz':FS,
```

iii. The module docstring and trajectory steps 5 and 12 explicitly say the AI retained native imaging frames because the requested decoder variables are time-varying.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from the session-wide frame timestamps `ft` and the per-trial sound timestamps `SoundTime`.

ii. 
```python
ft=np.asarray(b['ft'],float)[:n]
...
t0=float(np.asarray(b['Trial_start_time'])[tr]); cue=float(np.asarray(b['SoundTime'])[tr])
elapsed=(ft[ix]-t0)*86400.0; tocue=(cue-ft[ix])*86400.0
```

iii. The trajectory emphasizes using native time-resolved behavioral variables instead of the paper’s spatial interpolation, and this choice follows that pattern.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI converts MATLAB-date-like timestamps to seconds by subtracting each retained frame time from the trial’s `SoundTime` and multiplying by `86400.0`.

ii. 
```python
cue=float(np.asarray(b['SoundTime'])[tr])
tocue=(cue-ft[ix])*86400.0
```

iii. The trajectory says the conversion should retain native 3.17 Hz temporal structure for cue-time variables rather than spatially resampling them.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on the same frame indices `ix` that are used to slice the neural matrix for that trial, so it has exactly the same length and time base as the retained neural bins.

ii. 
```python
mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
...
tocue=(cue-ft[ix])*86400.0
...
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. The trajectory says the behavior arrays are already aligned to neural-frame timestamps and that all streams are truncated to a shared minimum length before per-trial slicing.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session id’s encoded recording date, parsed by `sid_date`, and from the earliest included recording date for that mouse.

ii. 
```python
def sid_date(s): return datetime.datetime.strptime('_'.join(s.split('_')[1:4]),'%Y_%m_%d').date()
```
```python
subjects=sorted({s.split('_')[0] for s in sids}); subjmap={s:i for i,s in enumerate(subjects)}
firstdate={m:min(sid_date(s) for s in sids if s.split('_')[0]==m) for m in subjects}
```

iii. In trajectory step 11, the AI argued that the experiment table’s session number was not a reliable chronological training-day measure, so it used recording date instead.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes `day` as the elapsed calendar days from each subject’s first included imaging recording, converts it to `float`, and broadcasts that value across all time bins of the trial.

ii. 
```python
day=float((sid_date(sid)-firstdate[sid.split('_')[0]]).days)
```
```python
inp=np.vstack([tocue,np.full(ix.size,day),elapsed,np.full(ix.size,int(np.asarray(b['isRew'])[tr]))]).astype(np.float32)
```

iii. The trajectory explicitly says this was chosen as a “robust continuous training-day variable” derived from recording date relative to each subject’s first included recording.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the frame timestamps `ft` and the per-trial trial-start timestamps `Trial_start_time`.

ii. 
```python
ft=np.asarray(b['ft'],float)[:n]
...
t0=float(np.asarray(b['Trial_start_time'])[tr])
elapsed=(ft[ix]-t0)*86400.0
```

iii. The trajectory consistently prefers using native timestamps for time-varying variables.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained bin, the AI subtracts the trial’s `Trial_start_time` from the frame timestamp and converts the difference from days to seconds with `86400.0`.

ii. 
```python
t0=float(np.asarray(b['Trial_start_time'])[tr])
elapsed=(ft[ix]-t0)*86400.0
```

iii. The trajectory rationale is the same as for cue time: retain native timing on the frame grid instead of spatially interpolating the trial.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same `ix` frame indices that define the neural slice for the trial, so the two arrays are exactly aligned in length and sample positions.

ii. 
```python
mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
elapsed=(ft[ix]-t0)*86400.0
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. The trajectory states that the behavior arrays are already aligned to neural-frame timestamps and are jointly truncated before trial extraction.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial `isRew` flag.

ii. 
```python
np.full(ix.size,int(np.asarray(b['isRew'])[tr]))
```

iii. The trajectory did not provide a separate justification beyond following the task definition of reward availability as a per-trial decoder input.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. There is no transformation beyond casting the trial’s `isRew` value to `int` and broadcasting it across the retained bins of that trial.

ii. 
```python
inp=np.vstack([tocue,np.full(ix.size,day),elapsed,np.full(ix.size,int(np.asarray(b['isRew'])[tr]))]).astype(np.float32)
```

iii. The trajectory treats reward availability as a trial-level variable that should be repeated across time bins so the per-trial input shape matches the neural data.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`.

ii. 
```python
categories=set(); speeds=[]
for sid in sids:
    b=beh[sid]; categories.update(map(str,np.asarray(b['WallName']).reshape(-1)))
```
```python
name=str(np.asarray(b['WallName'])[tr])
```

iii. In trajectory step 11, the AI noted that the visual categories present in the data include circle, leaf, rock, and wood variants and chose to build categories from the observed `WallName` values.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI builds a global sorted vocabulary of all distinct `WallName` strings across included sessions and maps each trial’s exact `WallName` to its index in that vocabulary. It does not collapse variants such as `circle1`, `circle2`, or shuffled variants into a shared base texture category.

ii. 
```python
categories=set(); speeds=[]
for sid in sids:
    b=beh[sid]; categories.update(map(str,np.asarray(b['WallName']).reshape(-1)))
categories=sorted(categories); catmap={x:i for i,x in enumerate(categories)}
```
```python
out=np.vstack([np.full(ix.size,catmap[name]),lick[ix],np.clip((pos//10).astype(int),0,3),np.digitize(speed,speed_edges)]).astype(np.int8)
```

iii. The trajectory says the observed visual categories span multiple named variants and uses a global category vocabulary so all sessions share one discrete label set.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from lick timestamps `LickTime` and the session frame timestamps `ft`.

ii. 
```python
ft=np.asarray(b['ft'],float)[:n]
lick=np.zeros(n,dtype=np.int8)
lt=np.asarray(b['LickTime']).reshape(-1)
```

iii. In trajectory step 11, the AI said it would “bin licks into frame intervals” while preserving native time resolution.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Each lick time is assigned to the nearest imaging frame by binary search over `ft`; duplicate assignments are collapsed with `np.unique`, and the result is a binary per-frame lick vector.

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

iii. The trajectory justifies this by saying the decoder task needs a time-resolved lick output on the native imaging frame grid.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. After session-wide lick binning, the AI slices `lick[ix]` using the same per-trial frame indices as the neural data, so the output has the same length and sample positions as the neural slice.

ii. 
```python
mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
...
out=np.vstack([np.full(ix.size,catmap[name]),lick[ix],np.clip((pos//10).astype(int),0,3),np.digitize(speed,speed_edges)]).astype(np.int8)
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. The trajectory says all streams are aligned on neural-frame timestamps before trial extraction.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from framewise corridor position `ft_Pos`.

ii. 
```python
pos=np.asarray(b['ft_Pos'],float)[:n][ix]
```

iii. The code comments and metadata state that the source position unit is 0.1 m and that corridor positions are represented from 0 to 40 over the textured corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI uses integer floor division by 10 and clips the result to `[0, 3]`, yielding four 1 m bins over the 4 m corridor.

ii. 
```python
np.clip((pos//10).astype(int),0,3)
```
```python
'position_conversion':'Source ft_Pos units are 0.1 m; bins [0,10), [10,20), [20,30), [30,40).',
```

iii. The trajectory did not separately justify this beyond following the decoder task’s requirement of four equal-length 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are fixed spatial bins: `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40)` in the source’s 0.1 m units, corresponding to 0-1 m, 1-2 m, 2-3 m, and 3-4 m.

ii. 
```python
np.clip((pos//10).astype(int),0,3)
```
```python
'output_values':[categories,['not_licking','licking'],['0-1 m','1-2 m','2-3 m','3-4 m'],['Q1 (slowest)','Q2','Q3','Q4 (fastest)']],
```

iii. The trajectory rationale is implicit: the bins are chosen to satisfy the task specification exactly.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is read on the same `ix` indices used for the neural slice, so it is aligned sample-for-sample to the retained neural bins.

ii. 
```python
mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
pos=np.asarray(b['ft_Pos'],float)[:n][ix]
...
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. The trajectory says all behavioral streams are already on the neural-frame grid and are sliced with the same per-trial indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from framewise running speed `ft_RunSpeed`.

ii. 
```python
v=np.asarray(b['ft_RunSpeed'],float)[:n][ok]; speeds.append(v[np.isfinite(v)])
```
```python
speed=np.asarray(b['ft_RunSpeed'],float)[:n][ix]
```

iii. The trajectory explicitly planned a first pass to compute running-speed quartiles and a second pass to assign speed categories during session conversion.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first collects all finite running-speed values from retained moving textured-corridor frames across all included sessions, computes global 25th/50th/75th percentile thresholds with `np.quantile`, and then discretizes each retained trial bin with `np.digitize`.

ii. 
```python
ok=np.isfinite(tri)&np.asarray(b['ft_isMoving'],bool)[:n]&np.asarray(b['ft_CorrSpc'],bool)[:n]&np.isfinite(pos)&(pos>=0)&(pos<40)
v=np.asarray(b['ft_RunSpeed'],float)[:n][ok]; speeds.append(v[np.isfinite(v)])
...
speed_edges=np.quantile(np.concatenate(speeds),[.25,.5,.75]); del speeds
```
```python
out=np.vstack([np.full(ix.size,catmap[name]),lick[ix],np.clip((pos//10).astype(int),0,3),np.digitize(speed,speed_edges)]).astype(np.int8)
```

iii. In trajectory steps 10, 12, and 29, the AI says it wanted global speed quartiles shared across sessions and checked that the resulting full-cohort quartiles were sensible.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded by three global percentile edges stored in `speed_edges`; `np.digitize` turns those edges into four quartile categories labeled Q1 to Q4.

ii. 
```python
speed_edges=np.quantile(np.concatenate(speeds),[.25,.5,.75]); del speeds
```
```python
np.digitize(speed,speed_edges)
```

iii. The trajectory says the first pass exists specifically to establish common quartile boundaries over all retained timepoints.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is read on the same per-trial `ix` frame indices used for the neural data and then discretized, so alignment is sample-for-sample with the retained neural bins.

ii. 
```python
mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
speed=np.asarray(b['ft_RunSpeed'],float)[:n][ix]
...
sn.append(np.asarray(X[:,ix],dtype=np.float32))
```

iii. The trajectory says all retained outputs are defined on the same native frame grid as the neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles data problems defensively. Behavior files that fail to load are skipped. Sessions are limited to the intersection of available behavior and spike data. Within a session, all streams are truncated to a shared minimum frame count. Trial masks reject non-finite positions and out-of-range corridor positions. Non-finite lick times are dropped. Missing retinotopy yields an `unassigned` region for every neuron, and retinotopy/neuron count mismatches are handled by resizing the region array. Trials with fewer than 2 retained bins and sessions with fewer than 2 usable trials are skipped.

ii. 
```python
for p in sorted(glob.glob(ROOT+'/beh/Beh_*.npy')):
    try: d=np.load(p,allow_pickle=True).item()
    except Exception: continue
```
```python
n=min(len(b['ft']),min(x.shape[1] for x in obj))
```
```python
return (tri==tr) & moving & corr & np.isfinite(pos) & (pos>=0) & (pos<40)
```
```python
if lt.size and np.issubdtype(lt.dtype,np.number):
    lt=lt[np.isfinite(lt.astype(float))].astype(float)
```
```python
if rp:
    ia=np.load(rp[0])['iarea']; ri=area_index(ia)
else: ri=np.full(X.shape[0],4,dtype=np.int8)
if len(ri)!=X.shape[0]:
    ri=np.resize(ri,X.shape[0]).astype(np.int8)
```

iii. In trajectory steps 10 to 12, the AI justified excluding unmatched sessions, truncating streams to their shared frame length, and adding defensive handling so the full conversion would finish on the large dataset rather than failing on irregular files.

## 12-a. What are the most time-consuming steps of the code?

i. The slowest part is loading and deserializing the spike files, then concatenating per-plane arrays session by session. The first pass over all sessions to collect speed statistics is much lighter than neural I/O.

ii. 
```python
for si,sid in enumerate(sids):
    b=beh[sid]
    obj=np.load(spk[sid],allow_pickle=True).item()['spks']
    n=min(len(b['ft']),min(x.shape[1] for x in obj))
    X=np.concatenate([x[:,:n] for x in obj],axis=0)
```

iii. The trajectory repeatedly says runtime is dominated by streaming roughly 434 GB of neural arrays and warns that the full conversion will take several minutes for that reason.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the per-trial loop inside each session, the first-pass loop that scans every session again to collect category and speed statistics, and repeated per-trial `np.asarray(...)` conversions of the same behavioral arrays. The neuron-subsampling loop over regions could also be simplified.

ii. 
```python
for sid in sids:
    b=beh[sid]
    ...
```
```python
for tr in range(int(b['ntrials'])):
    mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
    ...
    name=str(np.asarray(b['WallName'])[tr]); pos=np.asarray(b['ft_Pos'],float)[:n][ix]
    speed=np.asarray(b['ft_RunSpeed'],float)[:n][ix]
```
```python
for r in range(len(REGIONS)):
    ids=np.flatnonzero(ri==r)
    ...
```

iii. The trajectory does not discuss vectorization explicitly. Its only efficiency rationale is that neural-file I/O dominates runtime, so the code focuses on streaming one session at a time and passing the verifier.

## 12-c. What processing does the code repeat multiple times?

i. The code does a full first pass over all sessions to collect `WallName` categories and global speed thresholds, then a second pass to build the converted dataset. Inside the per-trial loop it repeatedly reconverts `WallName`, `ft_Pos`, `ft_RunSpeed`, and `isRew` with `np.asarray(...)` instead of caching session-level arrays once.

ii. 
```python
categories=set(); speeds=[]
for sid in sids:
    b=beh[sid]; categories.update(map(str,np.asarray(b['WallName']).reshape(-1)))
    ...
    v=np.asarray(b['ft_RunSpeed'],float)[:n][ok]; speeds.append(v[np.isfinite(v)])
```
```python
for si,sid in enumerate(sids):
    ...
    for tr in range(int(b['ntrials'])):
        ...
        name=str(np.asarray(b['WallName'])[tr]); pos=np.asarray(b['ft_Pos'],float)[:n][ix]
        speed=np.asarray(b['ft_RunSpeed'],float)[:n][ix]
        ...
        np.full(ix.size,int(np.asarray(b['isRew'])[tr]))
```

iii. The trajectory explicitly justifies the first pass as needed to compute global speed quartiles and a shared category vocabulary. It does not separately justify the repeated `np.asarray(...)` calls inside the trial loop.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code builds `subjmap` but never uses it. It imports `collections` but never uses it. It also stores detailed `session_info`, `behavior_source`, `retained_trial_indices`, and `excluded_neural_sessions_without_behavior` metadata that the downstream decoder does not need. More substantively, it computes fine-grained `WallName` categories rather than collapsing them, which increases label complexity even though downstream analyses only need the category output.

ii. 
```python
import argparse, collections, datetime, gc, glob, os, pickle
```
```python
subjects=sorted({s.split('_')[0] for s in sids}); subjmap={s:i for i,s in enumerate(subjects)}
```
```python
session_info.append({'session_id':sid,'behavior_source':bsrc[sid],'recording_date':str(sid_date(sid)),'training_day_elapsed_from_first_recording':day,'source_trials':int(b['ntrials']),'retained_trials':len(sn),'retained_trial_indices':kept,'neurons_retained':int(len(ri))})
```
```python
'excluded_neural_sessions_without_behavior':sorted(set(spk)-set(beh)),'session_info':session_info
```

iii. The trajectory justifies extra metadata mainly as documentation of conversion choices. It does not claim these fields are needed by the decoder itself.
