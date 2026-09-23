# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the BWM release CSV (`bwm_release.csv`) from the reference code repository to define the session set, rather than searching the ONE cache with a release tag. It initializes a ONE client with credentials and uses `one.load_object()` for trials and wheel data, `one.load_dataset()` for camera motion energy, and `SpikeSortingLoader` for spike sorting. Sessions are processed sequentially (later parallelized via sharding) and cached individually as pickle files, then assembled at the end. A `DATALIMIT_SUBSET.csv` is checked to restrict sessions if present.

ii.
```python
RELEASE=Path('/app/code/code_zhang2025/data/bwm_release.csv')
rel=pd.read_csv(RELEASE)
eids=list(dict.fromkeys(rel.eid))
subset=Path('/app/data/DATALIMIT_SUBSET.csv')
if subset.exists():
    ss=pd.read_csv(subset); allowed=set(ss['eid'] if 'eid' in ss else ss.iloc[:,0]); eids=[e for e in eids if e in allowed]
one=ONE()
```

```python
tr=one.load_object(eid,'trials',collection='alf')
sl=SpikeSortingLoader(one=one,pid=row.pid,eid=row.eid,pname=row.probe_name)
```

iii. The agent noted that "Remote Alyx access is blocked, so direct ONE downloading is unavailable." It chose to use the BWM release CSV to define the session set because "The release/session scope is all locally staged BWM sessions when no datalimit CSV exists." The agent avoided using `SessionLoader` for camera data because "current SessionLoader tries to write derived features into the intentionally read-only source cache."

## 1-b. How are the data split into subjects?

i. Subjects are derived from the `subject` column of the BWM release CSV (`bwm_release.csv`). Each session's subject is taken from the release table row. At assembly, subjects are sorted unique names and `subject_idx` maps each session to its index.

ii.
```python
'subject':str(rows.subject.iloc[0])
```
```python
subjects=sorted(set(s['subject'] for s in sessions)); smap={s:i for i,s in enumerate(subjects)}
```

iii. The agent used the release CSV's subject column directly, which already provides unique subject identifiers per session.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values from the release CSV. Each unique eid is one session.

ii.
```python
eids=list(dict.fromkeys(rel.eid))
```

iii. The release CSV lists one row per probe insertion, so unique eids define sessions. The agent iterated over unique eids and processed each as a session.

## 1-d. How are the data split into trials?

i. Trials are loaded from the ONE trials object, which provides one row per trial. The agent iterates over valid trial indices.

ii.
```python
tr=one.load_object(eid,'trials',collection='alf')
keys=['goCue_times','choice','feedback_times','stimOn_times','response_times',
      'firstMovement_times','probabilityLeft','feedbackType']
d={k:np.asarray(tr[k]) for k in keys}
```

iii. The trials table is already organized as one row per trial, so no splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. Five criteria are applied together: (1) all required timestamp fields must be finite (goCue_times, choice, feedback_times, stimOn_times, response_times, firstMovement_times, probabilityLeft, feedbackType), (2) reaction time (firstMovement_times - stimOn_times) between 0.08 and 2.0 seconds, (3) feedback_times - goCue_times <= 10.0 seconds, (4) choice != 0 (exclude no-go trials), (5) at least 2 valid trials per session. There is no explicit check that `probabilityLeft` is in {0.2, 0.5, 0.8} or that wheel/camera data spans the trial window.

ii.
```python
n=len(d['choice']); good=np.ones(n,bool)
for k in keys: good &= np.isfinite(d[k])
rt=d['firstMovement_times']-d['stimOn_times']
good &= (rt >= 0.08) & (rt <= 2.0)
good &= (d['feedback_times']-d['goCue_times'] <= 10.0)
good &= d['choice'] != 0
```

iii. The agent initially had a more permissive mask, then refined it: "The mismatch is resolved: the reference applies a default reaction-time filter of 0.08-2.0 seconds using firstMovement_times - stimOn_times, in addition to finite-event, duration, and no-choice filters." The feedback-goCue <= 10s filter comes from the reference code's `load_trials_and_mask` defaults.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times` and `spikes.clusters`, loaded via `SpikeSortingLoader`. The cluster table from `merge_clusters` provides anatomical labels (acronyms).

ii.
```python
sl=SpikeSortingLoader(one=one,pid=row.pid,eid=row.eid,pname=row.probe_name)
s,c,ch=sl.load_spike_sorting()
merged=sl.merge_clusters(s,c,ch).to_df()
st=np.asarray(s['times']); sc=np.asarray(s['clusters'],np.int64)
```

iii. The agent stated that spike times and cluster assignments are directly available through the SpikeSortingLoader.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the trial window [-0.5, 1.5] s around stimulus onset, giving 100 bins. Spike counts are stored as uint8 (capped at 255), NOT converted to firing rates. When a session has multiple probes their units are pooled with continuous indexing.

ii.
```python
edges=onset+OFF0+np.arange(NBIN+1)*BIN
lo=np.searchsorted(st,edges[0],'left'); hi=np.searchsorted(st,edges[-1],'left')
tt=st[lo:hi]; cc=sc[lo:hi]
tb=np.floor((tt-edges[0])/BIN).astype(np.int64)
flat=cc*NBIN+tb
counts=np.bincount(flat,minlength=ncl*NBIN).reshape(ncl,NBIN)
counts=np.minimum(counts,255).astype(np.uint8)
```

iii. The agent noted: "The full release has hundreds of sessions and very large spike arrays, so the converter must process sequentially, use compact dtypes, and avoid holding raw sessions in memory." The agent chose uint8 spike counts for memory efficiency.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ALL clusters are kept (no QC filtering). The agent explicitly chose qc=None, keeping all sorted clusters from the BWM release probes, including low-quality ones. The only filtering is that cluster indices must be valid (non-negative and within the merged table range). Brain regions are taken from the `acronym` column without Beryl mapping, and `void` regions are NOT filtered out.

ii.
```python
# Cluster IDs index the merged table. Keep all clusters as in prepare_data(qc=None).
n=len(merged); valid=(sc>=0)&(sc<n)
times.append(st[valid]); clus.append(sc[valid]+offset)
reg=np.asarray(merged['acronym'].fillna('void').astype(str)) if 'acronym' in merged else np.repeat('void',n)
```

iii. The agent justified this: "The reference prepare_data deliberately loads all clusters (qc=None), not only label-1 units; therefore the converter should retain all clusters from the BWM release probes."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spike times are aligned to stimulus onset by using `stimOn_times` as the reference. The bin edges are computed as `onset + OFF0 + np.arange(NBIN+1)*BIN`, where onset is the stimulus onset time, so spikes are binned relative to stimulus onset.

ii.
```python
onset=tr['stimOn_times'][j]; edges=onset+OFF0+np.arange(NBIN+1)*BIN
lo=np.searchsorted(st,edges[0],'left'); hi=np.searchsorted(st,edges[-1],'left')
tt=st[lo:hi]
tb=np.floor((tt-edges[0])/BIN).astype(np.int64)
```

iii. The agent recognized that all timestamps are on the same session clock, so alignment is achieved by computing bin edges relative to each trial's stimulus onset time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, producing 100 bins over the 2 s window [-0.5, 1.5] s. No rebinning or interpolation is applied; spikes are directly counted into 20 ms bins.

ii.
```python
BIN=.02; OFF0=-.5; OFF1=1.5; NBIN=100
```

iii. The agent confirmed: "The reference code explicitly uses 20 ms bins and a stimulus-onset window from -0.5 to +1.5 seconds."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table, which defines the alignment event. The input is the bin centers of the 100 bins spanning [-0.5, 1.5] s around stimulus onset.

ii.
```python
CENTERS=np.arange(NBIN,dtype=np.float64)*BIN + OFF0 + BIN/2
```

iii. The bin centers are defined by the decoding window parameters and are the same for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing beyond computing the bin centers. The values are `OFF0 + BIN/2 + i*BIN` for i in 0..99, giving centers from -0.49 to 1.49 s.

ii.
```python
CENTERS=np.arange(NBIN,dtype=np.float64)*BIN + OFF0 + BIN/2
inp=np.vstack((CENTERS.astype(np.float32),np.full(NBIN,blockno[j],np.float32)))
```

iii. N/A

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It is the same bin grid used for the neural data. The bin centers correspond exactly to the neural activity time bins.

ii.
```python
CENTERS=np.arange(NBIN,dtype=np.float64)*BIN + OFF0 + BIN/2
```

iii. The agent used the same window and bin parameters for both neural data and input time series.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A block boundary is detected when `probabilityLeft` changes between consecutive trials, and the trial number resets to 0 at each boundary.

ii.
```python
def block_trial_numbers(prior):
    out=np.zeros(len(prior),np.float32); k=0
    for i in range(1,len(prior)):
        k = 0 if prior[i] != prior[i-1] else k+1
        out[i]=k
    return out
```

iii. The trials table carries no explicit block identifier, so blocks are recovered from transitions in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A counter starts at 0 for the first trial and increments by 1 for each subsequent trial with the same `probabilityLeft`. When `probabilityLeft` changes, the counter resets to 0. This is computed on ALL trials before filtering, so filtered-out trials still advance the counter.

ii.
```python
prior_all=tr['probabilityLeft']; blockno=block_trial_numbers(prior_all)
```

iii. The computation is done before trial filtering so the block count reflects the animal's real position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which takes values -1 (right), 0 (no-go), or +1 (left). No-go trials (choice==0) are excluded by the trial filter.

ii.
```python
choice=np.uint8(0 if tr['choice'][j] == -1 else 1)
```

iii. The IBL convention is +1 for left and -1 for right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The value is recoded: -1 (right in IBL) maps to 1, and +1 (left in IBL) maps to 0. This matches the instructions: left = 0, right = 1. Note the mapping is inverted from the reference: the AI maps choice==-1 to 0 and choice==1 to 1, while the reference maps choice==1 (left) to 0 and choice==-1 (right) to 1. Wait - let me re-examine: The AI code says `0 if tr['choice'][j] == -1 else 1`. In IBL, choice==-1 is rightward, so this maps right->0, left->1. The reference maps +1 (left)->0, -1 (right)->1. The instructions say left=0, right=1. So the reference is correct and the AI has the mapping inverted.

ii.
```python
choice=np.uint8(0 if tr['choice'][j] == -1 else 1)
```

iii. No explicit justification was given for the mapping direction.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, or 0.8.

ii.
```python
pv=float(prior_all[j]); prior=np.uint8({.2:0,.5:1,.8:2}[round(pv,1)])
```

iii. The three values are the block prior, and the instructions specify the mapping 0.2->0, 0.5->1, 0.8->2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A simple mapping: 0.2->0, 0.5->1, 0.8->2. The value is rounded to one decimal place before lookup.

ii.
```python
prior=np.uint8({.2:0,.5:1,.8:2}[round(pv,1)])
```

iii. N/A

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The wheel position (`_ibl_wheel.position.npy`) and timestamps (`_ibl_wheel.timestamps.npy`), loaded directly via `one.load_object`.

ii.
```python
w=one.load_object(eid,'wheel',collection='alf')
t=np.asarray(w['timestamps'],float); p=np.asarray(w['position'],float)
```

iii. The wheel data is directly available through the ONE API.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) Filter for finite values and strictly increasing timestamps, (2) compute speed as `np.abs(np.gradient(position, timestamps))` (central difference), (3) interpolate onto bin centers with `np.interp`. This differs from the reference which uses SessionLoader's Butterworth-filtered velocity on a 1000 Hz interpolated grid.

ii.
```python
ok=np.isfinite(t)&np.isfinite(p); t,p=t[ok],p[ok]
u=np.r_[True,np.diff(t)>0]; t,p=t[u],p[u]
speed=np.abs(np.gradient(p,t))
ws=np.interp(x,wt,wv).astype(np.float32)
```

iii. The code comment says "Central-difference angular velocity, matching SessionLoader's velocity meaning." However, this does not actually match SessionLoader, which first interpolates to 1000 Hz, then applies a 20 Hz Butterworth low-pass filter.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Global tertiles: the 1/3 and 2/3 quantiles are computed across ALL wheel speed values from ALL sessions and ALL time bins, then `np.digitize` assigns each value to category 0, 1, or 2.

ii.
```python
qw=np.quantile(np.concatenate([s['wheel'] for s in sessions]),[1/3,2/3])
for s in sessions:
    for o in s['output']:
        o[2]=np.digitize(o[2],qw).astype(np.float32)
```

iii. The agent stated: "Global finite tertiles give reproducible, balanced three-class behavior outputs." This differs from the reference which uses per-session percentiles at 33.3% and 66.7%.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto the same bin centers (CENTERS) as the neural data, relative to stimulus onset.

ii.
```python
x=onset+CENTERS
ws=np.interp(x,wt,wv).astype(np.float32)
```

iii. The wheel timestamps are on the same session clock as spike times, so interpolation at the bin centers aligns them.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The ROI motion energy from the side camera (`leftCamera.ROIMotionEnergy.npy` preferred, `rightCamera.ROIMotionEnergy.npy` fallback) and its timestamps (`_ibl_<side>Camera.times.npy`).

ii.
```python
for cam in ('left','right'):
    try:
        t=np.asarray(one.load_dataset(eid,f'_ibl_{cam}Camera.times.npy',collection='alf'),float)
        v=np.asarray(one.load_dataset(eid,f'{cam}Camera.ROIMotionEnergy.npy',collection='alf'),float).squeeze()
```

iii. The agent noted: "Whisker motion energy uses the left camera when available and falls back to right, rather than averaging cameras."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is, with no additional filtering or normalization. It is interpolated onto the bin centers with `np.interp`, then discretized into 3 categories using global tertiles.

ii.
```python
me=np.interp(x,mt,mv).astype(np.float32)
```

iii. No additional processing beyond interpolation and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Global tertiles: the 1/3 and 2/3 quantiles are computed across ALL whisker motion energy values from ALL sessions, then `np.digitize` assigns each value to category 0, 1, or 2. Same approach as wheel speed.

ii.
```python
qm=np.quantile(np.concatenate([s['whisk'] for s in sessions]),[1/3,2/3])
for s in sessions:
    for o in s['output']:
        o[3]=np.digitize(o[3],qm).astype(np.float32)
```

iii. The agent stated: "Global finite tertiles give reproducible, balanced three-class behavior outputs."

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker motion energy is interpolated onto the same bin centers (CENTERS) as the neural data, relative to stimulus onset.

ii.
```python
x=onset+CENTERS
me=np.interp(x,mt,mv).astype(np.float32)
```

iii. The camera frame times are on the same session clock, so interpolation at the bin centers aligns them.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) Trials with any NaN in required timestamp fields are excluded via a finite check. (2) Non-finite and duplicate-timestamp wheel data is filtered. (3) Non-finite camera data is filtered. (4) Sessions where neither camera has usable motion energy raise an exception and are skipped. (5) Sessions with fewer than 2 valid trials are skipped. (6) Failed sessions are logged and excluded.

ii.
```python
for k in keys: good &= np.isfinite(d[k])
```
```python
ok=np.isfinite(t)&np.isfinite(v)
if ok.sum()>2: return t[ok],v[ok],cam
```
```python
if len(idx)<2: raise RuntimeError('fewer than two valid trials')
```
```python
except Exception as ex:
    warnings.warn(f'{eid}: {ex}'); failures.append((eid,repr(ex)))
```

iii. The agent noted: "Fourteen sessions were excluded because required streams (notably whisker motion energy) were unavailable; this is appropriate because every requested output must be present and aligned."

## 10-a. What are the most time-consuming steps of the code?

i. Loading the spike sorting data from disk via `SpikeSortingLoader.load_spike_sorting()`, which reads large spike time and cluster arrays (hundreds of MB per probe). The agent noted that sequential processing of 459 sessions was very slow and parallelized via sharding.

ii.
```python
sl=SpikeSortingLoader(one=one,pid=row.pid,eid=row.eid,pname=row.probe_name)
s,c,ch=sl.load_spike_sorting()
```

iii. The agent observed: "The full release has hundreds of sessions and very large spike arrays." It parallelized using 8 shards after observing slow sequential throughput.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` that iterates over valid trial indices to bin spikes, interpolate wheel/whisker, and build input/output arrays. This could potentially be vectorized by computing all trial bin edges at once and using batch operations.

ii.
```python
for j in idx:
    onset=tr['stimOn_times'][j]; edges=onset+OFF0+np.arange(NBIN+1)*BIN
    lo=np.searchsorted(st,edges[0],'left'); hi=np.searchsorted(st,edges[-1],'left')
    ...
```

iii. No explicit justification for keeping the loop structure.

## 10-c. What processing does the code repeat multiple times?

i. The code recomputes `np.arange(NBIN+1)*BIN` for every trial (the bin edge offsets), though this is a trivial computation. More notably, the code reads and converts each session pickle twice during assembly: once to compute global tertiles and once implicitly during the final pickle assembly (though in practice the data is in memory after loading).

ii.
```python
for j in idx:
    edges=onset+OFF0+np.arange(NBIN+1)*BIN
```

iii. No explicit justification.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and checks finiteness of several trial fields (`goCue_times`, `response_times`, `feedbackType`, `feedback_times`) that are not used in the final output beyond trial filtering. The `feedback_times - goCue_times <= 10.0` filter is an additional criterion from the reference code that may filter trials unnecessarily compared to the simpler RT-based filter. Additionally, storing raw wheel and whisker arrays per session for global tertile computation requires extra memory and processing.

ii.
```python
keys=['goCue_times','choice','feedback_times','stimOn_times','response_times',
      'firstMovement_times','probabilityLeft','feedbackType']
good &= (d['feedback_times']-d['goCue_times'] <= 10.0)
```

iii. No explicit justification for including fields beyond what's needed.
