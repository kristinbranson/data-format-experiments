# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are discovered by recursively globbing `/app/data` for `*.nwb` files, sorted alphabetically. Each file is opened with `pynwb.NWBHDF5IO`. Behavior and neural data are extracted from the NWB processing modules within a single pass per file.

ii.
```python
files=sorted(Path('/app/data').rglob('*.nwb')); files=files[:2] if a.sample else files
# ...
for i,fn in enumerate(files):
    N,I,O,sub,info=process_file(fn,a.show_processing)
```
Inside `process_file`:
```python
with NWBHDF5IO(str(fn),'r',load_namespaces=True) as io:
    nwb=io.read(); beh=nwb.processing['behavior']['BehavioralTimeSeries'].time_series
```

iii. The AI's CONVERSION_NOTES document that all 152 NWB files were found and opened successfully. The recursive glob approach ensures all files are found regardless of directory structure. Using `pynwb` satisfies the instruction constraint to not use `h5py`.

## 1-b. How are the data split into subjects?

i. Subjects are identified from `nwb.subject.subject_id` in each NWB file. A running list of unique subjects is maintained; each session is mapped to its subject via index lookup.

ii.
```python
neural_trials,inputs,outputs,nwb.subject.subject_id,info
# ...
if sub not in subjects: subjects.append(sub)
sidx.append(subjects.index(sub))
```

iii. Extracting subject ID from the NWB metadata is a direct and reliable approach. The CONVERSION_NOTES confirm 11 subjects were identified, matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed in sorted filename order.

ii.
```python
files=sorted(Path('/app/data').rglob('*.nwb'))
for i,fn in enumerate(files):
    N,I,O,sub,info=process_file(fn,a.show_processing)
    # ...
    neural.append(N); inputs.append(I); outputs.append(O)
```

iii. The one-file-per-session mapping is documented in the CONVERSION_NOTES Step 2 and matches the data organization.

## 1-d. How are the data split into trials?

i. Trials are identified by pairing each `trial_start > 0` pulse with the first subsequent `teleport > 0` pulse that occurs before the next trial start. Data between start and teleport timestamps form each trial.

ii.
```python
starts=np.flatnonzero(B['trial_start']>0); tele=np.flatnonzero(B['teleport']>0)
pairs=[]
for j,s in enumerate(starts):
    e0=tele[tele>=s]
    if len(e0) and (j+1==len(starts) or e0[0]<starts[j+1]): pairs.append((int(s),int(e0[0])))
```

iii. The AI's CONVERSION_NOTES Step 2 states: "Trials are reconstructed from each `trial_start` pulse to the following `teleport` pulse; all scanned starts had a valid following teleport before the next start." This matches the paper's trial structure.

## 1-e. How are trials filtered based on quality controls?

i. Two filters are applied: (1) trials where behavior extends beyond neural imaging coverage are excluded (the trial's start and end behavior timestamps must fall within the neural recording time range for all planes), and (2) sessions must have at least 2 complete trials.

ii.
```python
neural_start=max(x[1][0] for x in plane_data)
neural_end=min(x[1][-1] for x in plane_data)
pairs=[(s,e) for s,e in pairs if bt[s]>=neural_start and bt[e]<=neural_end]
if len(pairs)<2: raise ValueError(f'{fn}: fewer than 2 fully imaging-covered trials')
```

iii. The CONVERSION_NOTES Step 9 documents that one trial (terminal trial of m14_12) was excluded due to incomplete neural coverage. This filter ensures all retained trials have valid neural data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the NWB `Deconvolved` RoiResponseSeries under `processing/ophys`. This is suite2p's own deconvolution of raw fluorescence, stored in the NWB file.

ii.
```python
dec=nwb.processing['ophys']['Deconvolved'].roi_response_series
n_planes=len(dec)
for plane_name,rr in sorted(dec.items()):
    region=np.asarray(rr.rois.data[:],dtype=int)
    table=rr.rois.table
    iscell=np.asarray(table['iscell'].data[:])[:,0]>0
    valid=iscell[region]
    neural_all=np.asarray(rr.data[:, :],np.float32)[:,valid]
```

iii. The AI's CONVERSION_NOTES Step 1 states: "The main neural activity key used for place-cell detection is `events` (deconvolved calcium-event activity), not raw fluorescence. Thus no new delta-F/F calculation is indicated for this conversion when the NWB already provides processed event/deconvolved traces." Step 4 maps: "Neural signal | `timeseries['events']` | `processing/ophys/Deconvolved/plane0` | Deconvolved calcium events | Use NWB Deconvolved series."

## 2-b. How is the `neural` data processed?

i. The pre-computed Deconvolved data from the NWB is read directly. For each trial, the neural data is averaged into 100 ms temporal bins. Multi-plane sessions iterate over each plane's RoiResponseSeries, bin independently, and concatenate curated cells.

ii.
```python
DT=0.1
# ...
centers=t_start+DT/2+np.arange(max(1,int(np.floor((t_end-t_start)/DT))))*DT
centers=centers[centers<=t_end]
for plane_name,neural_t,neural_all in plane_data:
    ni=np.searchsorted(neural_t,centers-DT/2); nj=np.searchsorted(neural_t,centers+DT/2)
    xp=np.empty((neural_all.shape[1],len(centers)),np.float32)
    for k,(a,b) in enumerate(zip(ni,nj)):
        if b>a: xp[:,k]=neural_all[a:b].mean(0)
        else: xp[:,k]=neural_all[nearest_idx(neural_t,np.array([centers[k]]))[0]]
    binned_planes.append(xp)
X=np.concatenate(binned_planes,axis=0)
```

iii. The AI chose to use the NWB's pre-computed deconvolved data rather than recomputing from raw fluorescence. The 100 ms binning was chosen to "ensure a common bin size" across sessions with different native frame rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only suite2p's `iscell` classification is applied. ROIs where `iscell[:,0] > 0` are retained. No additional cell-type filtering (e.g., interneuron exclusion) is applied.

ii.
```python
iscell=np.asarray(table['iscell'].data[:])[:,0]>0
valid=iscell[region]
neural_all=np.asarray(rr.data[:, :],np.float32)[:,valid]
```

iii. The AI's CONVERSION_NOTES Step 3 states: "Use Suite2p `iscell` classification. Place-cell significance and reward-relative classification are downstream analysis labels and must not restrict a general neural decoder." The AI chose not to apply the paper's interneuron exclusion filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data are aligned to trial start. The 100 ms time bins begin at the trial start timestamp (`t_start`), so the first bin center is at `t_start + DT/2`.

ii.
```python
t_start,t_end=bt[s],bt[e]
centers=t_start+DT/2+np.arange(max(1,int(np.floor((t_end-t_start)/DT))))*DT
```

iii. The instructions specify "Temporally align based on start of the trial." The bin grid starts at trial start, so alignment is automatic.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned into 100 ms (DT=0.1 s) temporal bins. The native imaging frame rate (~15.5 Hz for single-plane, ~31 Hz aggregate for two-plane sessions) is averaged into these bins.

ii.
```python
DT=0.1
# ...
'time_bin_size':DT*1000  # 100.0 ms
```

iii. The AI's CONVERSION_NOTES Step 5 states: "100 ms for every session. This is fine enough relative to ~15.5/30 Hz imaging, ensures a common bin size, and avoids pretending unequal native frame durations are identical."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the behavior timestamps (`bt`), which come from `beh['trial_start'].timestamps[:]`.

ii.
```python
bt=np.asarray(beh['trial_start'].timestamps[:],float)
# ...
t_start,t_end=bt[s],bt[e]
centers=t_start+DT/2+np.arange(max(1,int(np.floor((t_end-t_start)/DT))))*DT
inp=np.vstack([centers-t_start, ...])
```

iii. The behavior timestamps provide the time reference for each sample. Trial start time is subtracted to get time-from-start.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The 100 ms bin centers are computed relative to trial start, then `t_start` is subtracted to give time from trial start in seconds.

ii.
```python
centers=t_start+DT/2+np.arange(max(1,int(np.floor((t_end-t_start)/DT))))*DT
centers=centers[centers<=t_end]
inp=np.vstack([centers-t_start, ...])
```

iii. Straightforward computation. The first time value is DT/2 = 0.05 s (center of first bin), not 0.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both neural and behavioral data use the same 100 ms time grid (bin centers). The neural data is binned to these same centers, ensuring alignment.

ii.
```python
# Same `centers` array used for both neural binning and input computation
ni=np.searchsorted(neural_t,centers-DT/2); nj=np.searchsorted(neural_t,centers+DT/2)
# ...
inp=np.vstack([centers-t_start, ...])
```

iii. The common time grid ensures all data streams are temporally aligned.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
B={k:np.asarray(v.data[:]) for k,v in beh.items() if v.data.shape[0]==len(bt)}
# ...
env=int(round(float(B['environment'][s]))); env=max(0,min(1,env))
```

iii. The environment variable is constant within a trial and takes values 0 or 1, matching ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The environment value at the trial start index is read, rounded to int, and clamped to [0, 1]. The value is repeated across all timepoints in the trial.

ii.
```python
env=int(round(float(B['environment'][s]))); env=max(0,min(1,env))
inp=np.vstack([..., np.full(len(centers),env), ...])
```

iii. Minimal processing. The clamping is a safety measure.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the native `trial number` behavior time series in the NWB file.

ii.
```python
trialnum=float(B['trial number'][s])
inp=np.vstack([..., np.full(len(centers),trialnum), ...])
```

iii. The AI chose to use the native trial number stored in the NWB rather than a sequential loop index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number at the trial start index is read as a float and repeated across all timepoints. No transformation is applied.

ii.
```python
trialnum=float(B['trial number'][s])
inp=np.vstack([..., np.full(len(centers),trialnum), ...])
```

iii. The native trial number is used directly.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the sparse `Reward` timestamps in the NWB file.

ii.
```python
rew_t=np.asarray(beh['Reward'].timestamps[:],float)
# ...
rewarded=int(np.any((rew_t>=t_start)&(rew_t<=t_end)))
prev=outcomes[-1] if outcomes else 0; outcomes.append(rewarded)
```

iii. Reward is an event-based time series with its own timestamps, separate from the behavior frame rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check if any Reward event timestamp falls within the trial's time window [t_start, t_end]. The previous trial's outcome is used as the input for the current trial. For the first trial, the value is 0. The value is constant across all timepoints in the trial.

ii.
```python
rewarded=int(np.any((rew_t>=t_start)&(rew_t<=t_end)))
prev=outcomes[-1] if outcomes else 0; outcomes.append(rewarded)
inp=np.vstack([..., np.full(len(centers),prev)])
```

iii. The `outcomes` list tracks the running sequence of retained trial outcomes, so `outcomes[-1]` gives the previous retained trial's outcome.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` behavior time series and fixed reward zone definitions. Zone identity is determined from the `reward_zone` behavior variable: when `reward_zone > 0`, the median position is computed and the nearest of three fixed 20 cm zones is assigned. For omission trials (no activation), the nearest labeled trial's zone is used.

ii.
```python
ZONES=np.array([[80.,100.],[200.,220.],[320.,340.]])

def zone_for_trial(pos):
    return int(np.argmin(np.abs(ZONES.mean(1)-pos)))

# In process_file:
raw_zone=[]
for s,e in pairs:
    hit=np.flatnonzero(B['reward_zone'][s:e+1]>0)
    if len(hit): raw_zone.append(zone_for_trial(float(np.median(B['position'][s+hit]))))
    else: raw_zone.append(-1)
# Fill missing with nearest labeled trial
good=np.flatnonzero(np.asarray(raw_zone)>=0)
zones=np.asarray(raw_zone)
bad=np.flatnonzero(zones<0)
zones[bad]=zones[good[np.argmin(abs(good[:,None]-bad),axis=0)]]
```

iii. The AI identified that the `reward_zone` stream is not a direct A/B/C label but a transient activation signal. The zone is inferred from the position at activation time and matched to fixed zone definitions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the signed distance from the animal's position to the nearest edge of the assigned reward zone is computed. Distance is 0 inside the zone, negative before it, and positive after it. The zone boundaries are 20 cm wide: A=[80,100], B=[200,220], C=[320,340].

ii.
```python
z=int(zones[ti]); lo,hi=ZONES[z]
dist=np.where(pos<lo,pos-lo,np.where(pos>hi,pos-hi,0.)).astype(np.float32)
```

iii. Standard signed-distance computation to an interval.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 categories using explicit conditional logic matching the instruction bins.

ii.
```python
def discretize_distance(d):
    y=np.empty(d.shape,np.int64)
    y[d < -50]=0; y[(d>=-50)&(d<-10)]=1; y[(d>=-10)&(d<0)]=2
    y[d==0]=3; y[(d>0)&(d<=10)]=4; y[(d>10)&(d<=50)]=5; y[d>50]=6
    return y
```

iii. The bins match the instruction specification: 0: < -50 cm, 1: -50 to -10 cm, 2: -10 cm to < 0 cm, 3: 0 cm, 4: >0 cm to +10 cm, 5: +10 to +50 cm, 6: > +50 cm.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is interpolated to the same 100 ms bin centers used for neural data, ensuring alignment.

ii.
```python
pos=np.interp(centers,bt,B['position']).astype(np.float32)
dist=np.where(pos<lo,pos-lo,np.where(pos>hi,pos-hi,0.))
```

iii. The common time grid ensures all data streams share the same temporal alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos=np.interp(centers,bt,B['position']).astype(np.float32)
```

iii. Position records the animal's location in the virtual corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is linearly interpolated to 100 ms bin centers, then discretized into 5 equal-width bins.

ii.
```python
pos=np.interp(centers,bt,B['position']).astype(np.float32)
poscls=np.digitize(pos,[90,180,270,360],right=False).astype(np.int64)
```

iii. Linear interpolation is appropriate for the continuous position signal. The 5 bins span 0-450 cm in 90 cm increments.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized using `np.digitize` with edges [90, 180, 270, 360], producing 5 bins: 0: <90 cm, 1: 90-180 cm, 2: 180-270 cm, 3: 270-360 cm, 4: >360 cm.

ii.
```python
poscls=np.digitize(pos,[90,180,270,360],right=False).astype(np.int64)
```

iii. This matches the instruction specification of 5 equal-sized bins spanning the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is interpolated to the same 100 ms bin centers as the neural data.

ii.
```python
pos=np.interp(centers,bt,B['position']).astype(np.float32)
```

iii. Same common time grid as all other variables.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
jj=nearest_idx(bt,centers)
lick=(B['lick'][jj]>0).astype(np.int64)
```

iii. Lick records lick events at each behavior timestamp.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick values are looked up using nearest-sample indexing to the 100 ms bin centers, then binarized (any positive value maps to 1).

ii.
```python
jj=nearest_idx(bt,centers)
lick=(B['lick'][jj]>0).astype(np.int64)
```

iii. Binarization matches the instruction specification (0 = no, 1 = yes). Nearest-sample lookup is appropriate for a discrete event signal.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Nearest-sample lookup to the same 100 ms bin centers used for neural data.

ii.
```python
jj=nearest_idx(bt,centers)
lick=(B['lick'][jj]>0).astype(np.int64)
```

iii. Same time grid as neural and all other variables.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series and `position` behavior time series. The `reward_zone` activation signal (> 0) identifies when the animal is in the reward zone; the median position during activation is used to assign the nearest fixed zone (A, B, or C).

ii. See 7-a code snippets.

iii. See 7-a justification.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For trials with reward_zone activation, the median position during activation is compared to the centers of three fixed 20 cm zones [80-100, 200-220, 320-340], and the nearest is assigned. For omission trials (no activation), the nearest labeled trial's zone is propagated. The zone index (A=0, B=1, C=2) is constant across all timepoints in the trial.

ii.
```python
z=int(zones[ti])
out=np.vstack([..., np.full(len(centers),z), ...])
```

iii. The nearest-neighbor fill for omission trials ensures every trial has a zone label, consistent with the blockwise zone structure.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the sparse `Reward` timestamps.

ii.
```python
rew_t=np.asarray(beh['Reward'].timestamps[:],float)
rewarded=int(np.any((rew_t>=t_start)&(rew_t<=t_end)))
```

iii. Reward delivery events have their own timestamp series, separate from the behavior frame rate.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any Reward timestamp falls within [t_start, t_end]. Output is binary (0=no, 1=yes) and constant across all timepoints in the trial.

ii.
```python
rewarded=int(np.any((rew_t>=t_start)&(rew_t<=t_end)))
out=np.vstack([..., np.full(len(centers),rewarded)])
```

iii. Directly checks event timestamps against trial boundaries.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Trials beyond neural coverage**: Trials where behavior timestamps extend past neural recording are excluded (1 trial excluded across all data).
- **Omission trials without reward_zone activation**: Zone label is propagated from the nearest trial with activation.
- **Multi-plane sessions**: Each plane's RoiResponseSeries is processed independently with its own timestamps and DynamicTableRegion for `iscell` subsetting.
- **Sessions with fewer than 2 trials**: Raise an error (none encountered in practice).
- **Neural bin with no samples**: Falls back to nearest-neighbor lookup.

ii.
```python
pairs=[(s,e) for s,e in pairs if bt[s]>=neural_start and bt[e]<=neural_end]
# ...
if b>a: xp[:,k]=neural_all[a:b].mean(0)
else: xp[:,k]=neural_all[nearest_idx(neural_t,np.array([centers[k]]))[0]]
```

iii. The CONVERSION_NOTES document the multi-plane ROI mismatch bug found and fixed in Step 9, and the incomplete neural coverage issue found in Step 10.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files and reading large neural data arrays. The full conversion takes ~125 seconds for 152 sessions. Per-session processing is fast (~0.8 s average including I/O).

ii. N/A (timing is printed during execution)

iii. The CONVERSION_NOTES Step 7 estimates <2 minutes for full conversion; actual was ~125 seconds.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-time-bin neural averaging loop iterates over each 100 ms bin:
```python
for k,(a,b) in enumerate(zip(ni,nj)):
    if b>a: xp[:,k]=neural_all[a:b].mean(0)
    else: xp[:,k]=neural_all[nearest_idx(neural_t,np.array([centers[k]]))[0]]
```
This could potentially be vectorized with `np.add.reduceat` or similar, though variable bin occupancy makes it awkward.

ii. See above code snippet.

iii. The loop is bounded by the number of time bins per trial (typically 100-300), so the overhead is modest.

## 13-c. What processing does the code repeat multiple times?

i. The code makes a single pass through all NWB files, so there is minimal repeated processing. Each file is opened once and all data extracted in one pass.

ii. N/A

iii. This is more efficient than the two-pass approach (survey + conversion) that would require loading files twice.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes session-level info dictionaries with metadata (zone distributions, rates, etc.) that are stored in `metadata['session_info']` but not used by the decoder. This is minor overhead.

ii.
```python
info=dict(session_id=sid,file=str(fn),n_trials=len(pairs),n_neurons=int(n_valid),
          rate=float(rates[0]),plane_rates=rates,n_planes=len(plane_data),
          zones=np.bincount(zones,minlength=3).tolist(),rewarded=int(sum(outcomes)),
          excluded_no_neural=excluded_no_neural)
```

iii. This metadata is useful for documentation but not consumed by the decoder training pipeline.
