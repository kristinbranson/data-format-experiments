# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not use `one.search()` or `SessionLoader` to enumerate sessions. Instead, it reads `/app/code/code_zhang2025/data/bwm_release.csv` as the release manifest, takes unique `eid`s from that table, optionally restricts them with `/app/data/DATALIMIT_SUBSET.csv`, and then loads each session's trial, wheel, camera, and spike datasets directly through `ONE`/`SpikeSortingLoader`.

ii. 
```python
RELEASE=Path('/app/code/code_zhang2025/data/bwm_release.csv')
...
CACHE.mkdir(exist_ok=True); rel=pd.read_csv(RELEASE)
eids=list(dict.fromkeys(rel.eid));
subset=Path('/app/data/DATALIMIT_SUBSET.csv')
if subset.exists():
    ss=pd.read_csv(subset); allowed=set(ss['eid'] if 'eid' in ss else ss.iloc[:,0]); eids=[e for e in eids if e in allowed]
```

```python
tr=one.load_object(eid,'trials',collection='alf')
w=one.load_object(eid,'wheel',collection='alf')
t=np.asarray(one.load_dataset(eid,f'_ibl_{cam}Camera.times.npy',collection='alf'),float)
sl=SpikeSortingLoader(one=one,pid=row.pid,eid=row.eid,pname=row.probe_name)
```

iii. In trajectory step 20, the agent said the converter should use the release CSV for "all 459 release sessions" and load wheel/camera streams directly because `SessionLoader` was trying to write derived files into the read-only cache. Step 16 also framed the direct camera loading as a workaround for loader behavior.

## 1-b. How are the data split into subjects?

i. The agent uses the `subject` column already present in `bwm_release.csv`. After per-session conversion, it builds `subjects` as the sorted unique subject names and `subject_idx` as an index into that list for each session.

ii. 
```python
return {'eid':eid,'subject':str(rows.subject.iloc[0]),'neural':neural,'input':inputs,
        'output':outputs,'regions':regions,'wheel':np.concatenate(wheel_raw),
        'whisk':np.concatenate(whisk_raw),'camera':cam,'n_source_trials':len(good)}
```

```python
subjects=sorted(set(s['subject'] for s in sessions)); smap={s:i for i,s in enumerate(subjects)}
...
'subjects':subjects,'subject_idx':np.asarray([smap[s['subject']] for s in sessions],np.int64),
```

iii. Step 20 says the converter should process all release sessions and then assemble the final pickle. The subject identity comes from the release table rows, so no extra derivation was described in the trajectory.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid`s from the release CSV. The converter processes one `eid` at a time, caches one pickle per session, and then assembles those cached session objects into the final dataset.

ii. 
```python
eids=list(dict.fromkeys(rel.eid));
...
for k,eid in enumerate(eids):
    f=CACHE/f'{eid}.pkl'
    ...
    s=convert_session(one,eid,rel[rel.eid==eid])
```

```python
sessions=[]
for eid in eids:
    f=CACHE/f'{eid}.pkl'
    if f.exists(): sessions.append(pickle.load(open(f,'rb')))
```

iii. In steps 20, 34, and 39, the agent explicitly described the work as "all 459 release sessions," processed independently and checkpointed one session at a time.

## 1-d. How are the data split into trials?

i. The converter loads the session's `trials` object, builds a boolean mask `good`, and then treats each surviving row index as one trial. The per-trial conversion loop iterates over `idx = np.flatnonzero(good)`.

ii. 
```python
tr=one.load_object(eid,'trials',collection='alf')
...
n=len(d['choice']); good=np.ones(n,bool)
```

```python
tr,good=trial_table(one,eid)
idx=np.flatnonzero(good)
...
for j in idx:
    onset=tr['stimOn_times'][j]
```

iii. The trajectory consistently treats the trials table as the source of trial structure. In step 21, the agent compared its trial mask against the reference and reasoned about how many rows survived.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only trials with finite values for `goCue_times`, `choice`, `feedback_times`, `stimOn_times`, `response_times`, `firstMovement_times`, `probabilityLeft`, and `feedbackType`; reaction time `firstMovement_times - stimOn_times` between 0.08 s and 2.0 s; `feedback_times - goCue_times <= 10.0`; and `choice != 0`. It does not apply a wheel/camera coverage check per trial.

ii. 
```python
keys=['goCue_times','choice','feedback_times','stimOn_times','response_times',
      'firstMovement_times','probabilityLeft','feedbackType']
d={k:np.asarray(tr[k]) for k in keys}
n=len(d['choice']); good=np.ones(n,bool)
for k in keys: good &= np.isfinite(d[k])
rt=d['firstMovement_times']-d['stimOn_times']
good &= (rt >= 0.08) & (rt <= 2.0)
good &= (d['feedback_times']-d['goCue_times'] <= 10.0)
good &= d['choice'] != 0
```

iii. Steps 21, 23, 24, and 26 show the agent debugging this mask against the reference. Step 26 says the "mismatch is resolved" by adding the 0.08-2.0 s reaction-time filter, and step 27 summarizes the filter as trial validity, no-choice exclusion, and the reaction-time rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrices are derived from `spikes.times` and `spikes.clusters` loaded by `SpikeSortingLoader`. The merged cluster table is also read so cluster count and region labels can be attached, but the binned neural matrix itself comes from spike times and cluster assignments.

ii. 
```python
sl=SpikeSortingLoader(one=one,pid=row.pid,eid=row.eid,pname=row.probe_name)
s,c,ch=sl.load_spike_sorting()
merged=sl.merge_clusters(s,c,ch).to_df()
st=np.asarray(s['times']); sc=np.asarray(s['clusters'],np.int64)
```

```python
tt=st[lo:hi]; cc=sc[lo:hi]
tb=np.floor((tt-edges[0])/BIN).astype(np.int64)
flat=cc*NBIN+tb
counts=np.bincount(flat,minlength=ncl*NBIN).reshape(ncl,NBIN)
```

iii. In steps 16-20, the agent focused on whether to keep all clusters or only QC-passing ones, but the raw neural source remained spike times and cluster IDs from the release probes.

## 2-b. How is the `neural` data processed?

i. The agent merges probes within a session, sorts all spikes by time, slices the spikes that fall inside each trial window, bins them into 100 bins of 20 ms each, and stores the result as `uint8` spike counts clipped at 255. It does not divide by bin width to convert counts to firing rates.

ii. 
```python
t=np.concatenate(times); c=np.concatenate(clus)
order=np.argsort(t,kind='mergesort')
return t[order],c[order],regions
```

```python
tb=np.floor((tt-edges[0])/BIN).astype(np.int64)
flat=cc*NBIN+tb
counts=np.bincount(flat,minlength=ncl*NBIN).reshape(ncl,NBIN)
counts=np.minimum(counts,255).astype(np.uint8)
```

iii. Step 16 says the agent planned to "bin spikes into 100 x 20 ms bins," and step 20 confirmed the final settings as "20 ms bins in [-0.5, 1.5] s around stimulus onset." Step 16 also notes that compact integer spike counts were important for the full conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps all clusters in the release probes as long as the spike's cluster index is in bounds for the merged cluster table. It does not filter clusters by QC label and does not remove `void` regions.

ii. 
```python
# Cluster IDs index the merged table. Keep all clusters as in prepare_data(qc=None).
n=len(merged); valid=(sc>=0)&(sc<n)
times.append(st[valid]); clus.append(sc[valid]+offset)
reg=np.asarray(merged['acronym'].fillna('void').astype(str)) if 'acronym' in merged else np.repeat('void',n)
regions.extend(reg.tolist()); offset += n
```

iii. Step 20 is explicit: "the reference `prepare_data` deliberately loads all clusters (`qc=None`), not only label-1 units; therefore the converter should retain all clusters from the BWM release probes." Step 17 shows the agent considering strict `label >= 1` filtering and then rejecting it after checking the methods code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. For each surviving trial, the agent builds bin edges from `stimOn_times[j] - 0.5` to `stimOn_times[j] + 1.5` and bins spikes within that window relative to stimulus onset.

ii. 
```python
OFF0=-.5; OFF1=1.5; NBIN=100
CENTERS=np.arange(NBIN,dtype=np.float64)*BIN + OFF0 + BIN/2
```

```python
onset=tr['stimOn_times'][j]; edges=onset+OFF0+np.arange(NBIN+1)*BIN
lo=np.searchsorted(st,edges[0],'left'); hi=np.searchsorted(st,edges[-1],'left')
tt=st[lo:hi]; cc=sc[lo:hi]
tb=np.floor((tt-edges[0])/BIN).astype(np.int64)
```

iii. Steps 16 and 20 both describe the alignment window as 100 20-ms bins spanning `[-0.5, 1.5)` relative to stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20 ms. Each trial has 100 bins covering a 2 s window from -0.5 s to 1.5 s relative to stimulus onset. No additional temporal rebinning is applied after this binning.

ii. 
```python
BIN=.02; OFF0=-.5; OFF1=1.5; NBIN=100
CENTERS=np.arange(NBIN,dtype=np.float64)*BIN + OFF0 + BIN/2
```

```python
edges=onset+OFF0+np.arange(NBIN+1)*BIN
...
'time_bin_size':20.0,
'off_start':-0.5,'off_end':1.5,
```

iii. The agent described the same 20 ms, 100-bin configuration in steps 16, 20, 27, and 52.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is defined from the alignment event `stimOn_times` plus the fixed 20 ms bin centers. The actual per-trial input values come from the precomputed `CENTERS` array repeated for every trial.

ii. 
```python
CENTERS=np.arange(NBIN,dtype=np.float64)*BIN + OFF0 + BIN/2
```

```python
onset=tr['stimOn_times'][j]
...
inp=np.vstack((CENTERS.astype(np.float32),np.full(NBIN,blockno[j],np.float32)))
```

iii. Step 20 says the data should be aligned in 20 ms bins around stimulus onset; the trajectory does not describe any more elaborate derivation for this input.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No signal processing is applied. The agent simply defines the 100 bin centers from -0.49 s to 1.49 s in 20 ms steps and copies that same vector into every trial's first input row.

ii. 
```python
BIN=.02; OFF0=-.5; OFF1=1.5; NBIN=100
CENTERS=np.arange(NBIN,dtype=np.float64)*BIN + OFF0 + BIN/2
```

```python
inp=np.vstack((CENTERS.astype(np.float32),np.full(NBIN,blockno[j],np.float32)))
```

iii. The trajectory only justifies the time input by the chosen decoding window and bin size, especially in steps 16 and 20.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the same bin centers as the neural trial windows. The neural spikes are binned using edges built from `onset + OFF0 + np.arange(NBIN+1) * BIN`, and the time input uses the corresponding centers `CENTERS`.

ii. 
```python
edges=onset+OFF0+np.arange(NBIN+1)*BIN
...
tb=np.floor((tt-edges[0])/BIN).astype(np.int64)
```

```python
CENTERS=np.arange(NBIN,dtype=np.float64)*BIN + OFF0 + BIN/2
inp=np.vstack((CENTERS.astype(np.float32),np.full(NBIN,blockno[j],np.float32)))
```

iii. Steps 16 and 20 present the time axis and spike binning as one shared 20 ms grid around stimulus onset.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw `probabilityLeft` trial column. The agent treats a change in `probabilityLeft` as the start of a new block.

ii. 
```python
def block_trial_numbers(prior):
    out=np.zeros(len(prior),np.float32); k=0
    for i in range(1,len(prior)):
        k = 0 if prior[i] != prior[i-1] else k+1
        out[i]=k
    return out
```

```python
ncl=len(regions); prior_all=tr['probabilityLeft']; blockno=block_trial_numbers(prior_all)
```

iii. The trajectory does not separately justify this choice, but step 20 says the script should "encode choice and prior," and the implementation infers block structure directly from the prior sequence.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent counts upward within runs of constant `probabilityLeft`, resetting to zero whenever the prior changes. This is computed on the full trial table before applying the `good` mask, and the resulting value is then looked up for each kept trial.

ii. 
```python
def block_trial_numbers(prior):
    out=np.zeros(len(prior),np.float32); k=0
    for i in range(1,len(prior)):
        k = 0 if prior[i] != prior[i-1] else k+1
        out[i]=k
    return out
```

```python
idx=np.flatnonzero(good)
...
ncl=len(regions); prior_all=tr['probabilityLeft']; blockno=block_trial_numbers(prior_all)
...
inp=np.vstack((CENTERS.astype(np.float32),np.full(NBIN,blockno[j],np.float32)))
```

iii. The trajectory did not call out this variable specifically, so the justification is implicit in the code: block count is recovered from `probabilityLeft` before trial filtering.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the raw `choice` trial column. The agent filters out `choice == 0` trials and then encodes `choice == -1` as 0 and everything else that survives as 1, which means right choices become 0 and left choices become 1.

ii. 
```python
good &= d['choice'] != 0
```

```python
choice=np.uint8(0 if tr['choice'][j] == -1 else 1)
```

iii. Step 20 only says the converter should "encode choice and prior"; the trajectory does not show any explicit reasoning about the sign convention, so the code itself is the only evidence of the decision.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Other than dropping no-response trials, the only processing is the binary recoding above. The output is then broadcast across all 100 time bins of the trial.

ii. 
```python
choice=np.uint8(0 if tr['choice'][j] == -1 else 1)
...
out=np.empty((4,NBIN),np.float32); out[0]=choice; out[1]=prior; out[2]=ws; out[3]=me
```

iii. The trajectory contains no detailed justification beyond the general plan in step 20 to encode the task outputs categorically.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the raw `probabilityLeft` trial column.

ii. 
```python
pv=float(prior_all[j]); prior=np.uint8({.2:0,.5:1,.8:2}[round(pv,1)])
```

iii. The trajectory repeatedly refers to encoding the block prior, especially in step 20, but does not provide extra reasoning beyond using the task's three prior values.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent rounds the raw prior to one decimal place and maps 0.2 to 0, 0.5 to 1, and 0.8 to 2. It then broadcasts that categorical value across all 100 time bins of the trial.

ii. 
```python
pv=float(prior_all[j]); prior=np.uint8({.2:0,.5:1,.8:2}[round(pv,1)])
...
out=np.empty((4,NBIN),np.float32); out[0]=choice; out[1]=prior; out[2]=ws; out[3]=me
```

iii. The trajectory does not discuss this mapping separately; it is implied by the plan in step 20 to encode the required categorical outputs.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from the wheel object's raw `timestamps` and `position` arrays.

ii. 
```python
w=one.load_object(eid,'wheel',collection='alf')
t=np.asarray(w['timestamps'],float); p=np.asarray(w['position'],float)
```

```python
speed=np.abs(np.gradient(p,t))
return t,speed
```

iii. Step 18 says "Wheel velocity must be derived from position/timestamps," which is the justification the agent followed in the final code.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent drops nonfinite wheel samples, removes non-increasing timestamps, computes the absolute central-difference gradient `np.abs(np.gradient(p, t))` as speed, and then linearly interpolates that speed onto the 100 trial bin centers. It does not use `SessionLoader`'s interpolated/filtered velocity.

ii. 
```python
ok=np.isfinite(t)&np.isfinite(p); t,p=t[ok],p[ok]
u=np.r_[True,np.diff(t)>0]; t,p=t[u],p[u]
# Central-difference angular velocity, matching SessionLoader's velocity meaning.
speed=np.abs(np.gradient(p,t))
```

```python
x=onset+CENTERS
ws=np.interp(x,wt,wv).astype(np.float32)
```

iii. In step 18, the agent concluded that wheel velocity "must be derived from position/timestamps." The direct computation is also motivated by the loader limitations described in steps 16 and 20.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The agent first stores continuous wheel speed traces for all sessions, concatenates them across the full converted dataset, computes global tertile cutoffs at 1/3 and 2/3 quantiles, and then discretizes each trial's wheel speed into three categories with those global thresholds.

ii. 
```python
return {'eid':eid,'subject':str(rows.subject.iloc[0]),'neural':neural,'input':inputs,
        'output':outputs,'regions':regions,'wheel':np.concatenate(wheel_raw),
        'whisk':np.concatenate(whisk_raw),'camera':cam,'n_source_trials':len(good)}
```

```python
qw=np.quantile(np.concatenate([s['wheel'] for s in sessions]),[1/3,2/3])
...
for s in sessions:
    for o in s['output']:
        o[2]=np.digitize(o[2],qw).astype(np.float32)
```

iii. Step 20 says the converter would "calculate global behavior tertiles," and steps 27 and 49 repeat that global-tertile choice as part of the final design.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. For each trial, the wheel speed is evaluated by linear interpolation at `onset + CENTERS`, so it shares the same 100-bin stimulus-aligned time axis as the neural data.

ii. 
```python
x=onset+CENTERS
ws=np.interp(x,wt,wv).astype(np.float32)
```

iii. Step 20 describes "direct wheel ... interpolation" onto the 20 ms stimulus-aligned bins, and step 18 explicitly notes that the stream can be aligned by its timestamps.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from one camera's raw frame timestamps and motion-energy array: `_ibl_leftCamera.times.npy` with `leftCamera.ROIMotionEnergy.npy`, or the right-camera equivalents if left is unavailable. Left is preferred, right is the fallback.

ii. 
```python
for cam in ('left','right'):
    try:
        t=np.asarray(one.load_dataset(eid,f'_ibl_{cam}Camera.times.npy',collection='alf'),float)
        v=np.asarray(one.load_dataset(eid,f'{cam}Camera.ROIMotionEnergy.npy',collection='alf'),float).squeeze()
```

```python
        if ok.sum()>2: return t[ok],v[ok],cam
```

iii. Step 20 explicitly says whisker motion energy should use the left camera when available and fall back to the right, rather than combining cameras.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The agent loads the released motion-energy trace directly, truncates time and value arrays to the same minimum length, drops nonfinite samples, and linearly interpolates the resulting trace onto the 100 bin centers of each trial. It does not filter or normalize the trace.

ii. 
```python
n=min(len(t),len(v)); t,v=t[:n],v[:n]
ok=np.isfinite(t)&np.isfinite(v)
if ok.sum()>2: return t[ok],v[ok],cam
```

```python
x=onset+CENTERS
me=np.interp(x,mt,mv).astype(np.float32)
```

iii. Steps 16 and 20 justify the direct array loading as a workaround for `SessionLoader`, and step 18 says camera motion energy can be loaded directly and aligned by timestamps.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, the agent concatenates the continuous whisker traces across all retained sessions, computes global tertile cutoffs, and discretizes each trial's whisker trace with those dataset-wide thresholds.

ii. 
```python
return {'eid':eid,'subject':str(rows.subject.iloc[0]),'neural':neural,'input':inputs,
        'output':outputs,'regions':regions,'wheel':np.concatenate(wheel_raw),
        'whisk':np.concatenate(whisk_raw),'camera':cam,'n_source_trials':len(good)}
```

```python
qm=np.quantile(np.concatenate([s['whisk'] for s in sessions]),[1/3,2/3])
...
for s in sessions:
    for o in s['output']:
        o[3]=np.digitize(o[3],qm).astype(np.float32)
```

iii. Step 20 says the converter would "calculate global behavior tertiles," and step 49 confirms that "Global tertiles were computed over all retained samples."

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. For each trial, the whisker trace is evaluated by linear interpolation at `onset + CENTERS`, so it shares the same 100-bin stimulus-aligned time axis as the neural data.

ii. 
```python
x=onset+CENTERS
me=np.interp(x,mt,mv).astype(np.float32)
```

iii. Step 18 says camera motion energy can be "aligned by camera timestamps," and step 20 says the direct wheel/camera interpolation should be done on the common stimulus-aligned bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several data issues defensively: it filters trials by finiteness of essential trial fields; drops no-choice and out-of-range reaction-time trials; truncates camera time/value arrays to the same length; removes nonfinite wheel/camera samples; removes non-increasing wheel timestamps; skips sessions with fewer than two valid trials, no release probes, or no whisker stream; and records failed sessions in metadata. It does not drop individual trials for incomplete wheel/camera coverage within the trial window.

ii. 
```python
for k in keys: good &= np.isfinite(d[k])
...
good &= d['choice'] != 0
```

```python
n=min(len(t),len(v)); t,v=t[:n],v[:n]
ok=np.isfinite(t)&np.isfinite(v)
```

```python
u=np.r_[True,np.diff(t)>0]; t,p=t[u],p[u]
...
if len(idx)<2: raise RuntimeError('fewer than two valid trials')
...
raise RuntimeError('no whisker motion-energy stream')
```

iii. Steps 16, 20, 48, 49, and 52 describe this as excluding sessions that lack required streams, especially missing whisker motion-energy data, and treating checkpointed failures as acceptable exclusions.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive part is session-by-session conversion over the full release, especially loading spike sorting for every probe and writing large per-session caches. The agent considered the sequential pass too slow and parallelized across shards.

ii. 
```python
for row in rows.itertuples():
    sl=SpikeSortingLoader(one=one,pid=row.pid,eid=row.eid,pname=row.probe_name)
    s,c,ch=sl.load_spike_sorting()
```

```python
for k,eid in enumerate(eids):
    ...
    s=convert_session(one,eid,rel[rel.eid==eid])
    tmp=f.with_suffix('.tmp'); pickle.dump(s,open(tmp,'wb'),protocol=5); os.replace(tmp,f)
```

iii. Steps 28-34 explicitly discuss runtime, with step 34 estimating another 45-55 minutes for sequential conversion and motivating sharded background workers.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main non-vectorized work is the per-trial loop in `convert_session`, which repeatedly slices spikes, bins them, interpolates wheel and whisker traces, and builds input/output arrays one trial at a time. The later nested loop over sessions and trial outputs also applies discretization trial by trial.

ii. 
```python
for j in idx:
    onset=tr['stimOn_times'][j]; edges=onset+OFF0+np.arange(NBIN+1)*BIN
    lo=np.searchsorted(st,edges[0],'left'); hi=np.searchsorted(st,edges[-1],'left')
    ...
    ws=np.interp(x,wt,wv).astype(np.float32)
    me=np.interp(x,mt,mv).astype(np.float32)
```

```python
for s in sessions:
    for o in s['output']:
        o[2]=np.digitize(o[2],qw).astype(np.float32); o[3]=np.digitize(o[3],qm).astype(np.float32)
        o[:]=o.astype(np.uint8)
```

iii. The trajectory does not discuss vectorization directly. Instead, steps 34 and 39 justify parallelizing across session shards rather than rewriting the per-trial inner loop.

## 10-c. What processing does the code repeat multiple times?

i. The code intentionally performs the conversion in multiple passes: it converts each session to an intermediate cache pickle, later reloads all caches to assemble the final dataset, and stores raw continuous wheel/whisk traces so it can revisit every output again after computing global quantile thresholds. The final unsharded pass also retries previously failed sessions.

ii. 
```python
tmp=f.with_suffix('.tmp'); pickle.dump(s,open(tmp,'wb'),protocol=5); os.replace(tmp,f)
```

```python
sessions=[]
for eid in eids:
    f=CACHE/f'{eid}.pkl'
    if f.exists(): sessions.append(pickle.load(open(f,'rb')))
```

```python
qw=np.quantile(np.concatenate([s['wheel'] for s in sessions]),[1/3,2/3])
qm=np.quantile(np.concatenate([s['whisk'] for s in sessions]),[1/3,2/3])
for s in sessions:
    for o in s['output']:
        o[2]=np.digitize(o[2],qw).astype(np.float32); o[3]=np.digitize(o[3],qm).astype(np.float32)
```

iii. Steps 20, 34, 39, and 48-49 explicitly justify this repeated processing as a resumability strategy: cache sessions first, then assemble later, and do a final unsharded retry/assembly pass after the shards finish.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The agent keeps continuous wheel and whisker traces in the intermediate session dictionaries only to compute global thresholds later, but those raw traces are not written into the final dataset. It also first stores rows 2 and 3 of each output as continuous `float32` traces and then overwrites them with discretized `uint8` categories before final assembly.

ii. 
```python
return {'eid':eid,'subject':str(rows.subject.iloc[0]),'neural':neural,'input':inputs,
        'output':outputs,'regions':regions,'wheel':np.concatenate(wheel_raw),
        'whisk':np.concatenate(whisk_raw),'camera':cam,'n_source_trials':len(good)}
```

```python
# Raw behavior occupies rows 2/3 temporarily; global tertiles are applied later.
out=np.empty((4,NBIN),np.float32); out[0]=choice; out[1]=prior; out[2]=ws; out[3]=me
```

```python
for s in sessions:
    for o in s['output']:
        o[2]=np.digitize(o[2],qw).astype(np.float32); o[3]=np.digitize(o[3],qm).astype(np.float32)
        o[:]=o.astype(np.uint8)
```

iii. The trajectory justifies this in step 20 as part of the plan to "calculate global behavior tertiles" after session conversion. It is deliberate, but the raw continuous traces themselves are not part of the final deliverable.
