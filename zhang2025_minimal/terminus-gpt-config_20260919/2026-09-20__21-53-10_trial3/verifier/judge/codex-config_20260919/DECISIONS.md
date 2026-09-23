# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `/app/code/code_zhang2025/data/bwm_release.csv`, takes its unique `eid` values (optionally restricted by `DATALIMIT_SUBSET.csv`, `--limit`, or `--shard`), and uses ONE to load trials, wheel, camera, and spike-sorting data. Probe rows in the release CSV determine which probes are loaded. Converted sessions are cached as individual pickle files and later assembled; sessions that raise an exception are omitted.

ii.
```python
rel=pd.read_csv(RELEASE)
eids=list(dict.fromkeys(rel.eid))
...
tr=one.load_object(eid,'trials',collection='alf')
w=one.load_object(eid,'wheel',collection='alf')
...
s=convert_session(one,eid,rel[rel.eid==eid])
```

iii. The trajectory says the agent identified the Zhang 2025 release CSV as the intended release scope, confirmed that the payloads were available in the local ONE cache, and chose sequential session processing plus resumable per-session checkpoints because the full release and spike arrays were large. It later added sharding for speed.

## 1-b. How are the data split into subjects?

i. The subject is taken from the `subject` column of the release CSV for each session. At assembly, unique subject strings are sorted and each session receives an integer index into that list.

ii.
```python
'subject':str(rows.subject.iloc[0])
...
subjects=sorted(set(s['subject'] for s in sessions))
smap={s:i for i,s in enumerate(subjects)}
'subject_idx':np.asarray([smap[s['subject']] for s in sessions],np.int64)
```

iii. The trajectory treated the release metadata as authoritative for session and subject identity; no subject ID was inferred from paths or file names.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the release CSV is treated as one session. All probe rows sharing that `eid` are passed together to `convert_session`, and each successful session becomes one element of the top-level session lists.

ii.
```python
eids=list(dict.fromkeys(rel.eid))
...
s=convert_session(one,eid,rel[rel.eid==eid])
...
data={'neural':[s['neural'] for s in sessions],
      'input':[s['input'] for s in sessions], ...}
```

iii. The agent reasoned that the BWM release is already session-organized and that the `eid` is the unique session handle.

## 1-d. How are the data split into trials?

i. The trials object supplies one entry per trial. The code constructs a Boolean mask, obtains retained row indices, and iterates over them, producing one neural, input, and output array per retained trial.

ii.
```python
tr,good=trial_table(one,eid)
idx=np.flatnonzero(good)
...
for j in idx:
    onset=tr['stimOn_times'][j]
    ...
    neural.append(counts); inputs.append(inp); outputs.append(out)
```

iii. The trajectory recognized the trials table as the trial boundary definition and compared its retained count against the reference utility on a representative session.

## 1-e. How are trials filtered based on quality controls?

i. A trial is kept only if all eight loaded fields are finite, reaction time (`firstMovement_times - stimOn_times`) is 0.08–2 s inclusive, go-cue-to-feedback duration is at most 10 s, and `choice != 0`. Sessions with fewer than two retained trials are rejected. The code does not require `probabilityLeft` to be one of the three expected values in the mask and does not test that wheel/camera streams span each retained window.

ii.
```python
keys=['goCue_times','choice','feedback_times','stimOn_times','response_times',
      'firstMovement_times','probabilityLeft','feedbackType']
good=np.ones(n,bool)
for k in keys: good &= np.isfinite(d[k])
rt=d['firstMovement_times']-d['stimOn_times']
good &= (rt >= 0.08) & (rt <= 2.0)
good &= (d['feedback_times']-d['goCue_times'] <= 10.0)
good &= d['choice'] != 0
```

iii. The trajectory initially produced too many trials, then inspected the reference mask and added the 0.08–2 s reaction-time filter, obtaining the expected count for its test session. It described finite required fields, the 10 s duration bound, and no-choice exclusion as reference behavior, but did not discuss per-trial stream coverage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays are derived from spike timestamps (`spikes['times']`) and spike cluster assignments (`spikes['clusters']`) for every release probe. The merged cluster table determines the number/order of units and supplies region acronyms.

ii.
```python
s,c,ch=sl.load_spike_sorting()
merged=sl.merge_clusters(s,c,ch).to_df()
st=np.asarray(s['times']); sc=np.asarray(s['clusters'],np.int64)
```

iii. The trajectory inspected the reference spike loader and concluded that spike time and cluster assignment were the required raw neural variables.

## 2-b. How is the `neural` data processed?

i. Probes are merged by offsetting their cluster indices. For each trial, spikes in the two-second window are assigned to 20 ms bins and counted with `bincount`. Counts are clipped at 255 and stored as `uint8`; they are not divided by bin width into firing rates and are not smoothed.

ii.
```python
clus.append(sc[valid]+offset)
...
tb=np.floor((tt-edges[0])/BIN).astype(np.int64)
flat=cc*NBIN+tb
counts=np.bincount(flat,minlength=ncl*NBIN).reshape(ncl,NBIN)
counts=np.minimum(counts,255).astype(np.uint8)
```

iii. The agent said the reference used 20 ms bins without smoothing and deliberately chose compact integer spike counts to control the size of the full dataset. It did not justify omitting the reference conversion from counts to Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No cluster-quality or anatomical filter is applied. Every valid cluster-table row is retained, including low-QC and `void` clusters; only spike assignments outside the table range are discarded.

ii.
```python
# Cluster IDs index the merged table. Keep all clusters as in prepare_data(qc=None).
n=len(merged); valid=(sc>=0)&(sc<n)
times.append(st[valid]); clus.append(sc[valid]+offset)
reg=np.asarray(merged['acronym'].fillna('void').astype(str))
```

iii. The trajectory explicitly investigated this choice. It found that the repository's `prepare_data` called its loader with `qc=None` and therefore chose all clusters, despite also observing that strict `label == 1` left far fewer units. This differs from the human solution's decision to follow the data-paper curation and remove `void` units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial uses a window from -0.5 to +1.5 s relative to `stimOn_times`. Absolute spike times are searched between the corresponding absolute edges and then binned relative to the first edge.

ii.
```python
onset=tr['stimOn_times'][j]
edges=onset+OFF0+np.arange(NBIN+1)*BIN
lo=np.searchsorted(st,edges[0],'left')
hi=np.searchsorted(st,edges[-1],'left')
tb=np.floor((tt-edges[0])/BIN).astype(np.int64)
```

iii. The trajectory repeatedly confirmed from the paper/code that the target alignment was stimulus onset with a [-0.5, 1.5) s window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms, with 100 bins over two seconds. Raw spike times are binned once; no later temporal rebinning or interpolation of neural data is performed.

ii.
```python
BIN=.02; OFF0=-.5; OFF1=1.5; NBIN=100
```

iii. The agent cites the reference configuration as 100 20-ms bins spanning -0.5 to +1.5 s.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is defined from the stimulus-aligned bin grid: `stimOn_times` sets the zero point, and the stored values are the centers of the 100 bins from -0.49 to 1.49 s.

ii.
```python
CENTERS=np.arange(NBIN,dtype=np.float64)*BIN + OFF0 + BIN/2
onset=tr['stimOn_times'][j]
x=onset+CENTERS
```

iii. The agent justified the values from the confirmed reference alignment window and bin size.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code adds half a bin to the left edges to form bin centers, casts the vector to `float32`, and copies the same relative-time vector into every trial.

ii.
```python
CENTERS=np.arange(NBIN,dtype=np.float64)*BIN + OFF0 + BIN/2
inp=np.vstack((CENTERS.astype(np.float32),
               np.full(NBIN,blockno[j],np.float32)))
```

iii. The trajectory did not provide a separate rationale beyond matching the reference 20 ms grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input contains the centers of the exact edges used for neural binning, so each input sample corresponds to the same-index neural bin.

ii.
```python
edges=onset+OFF0+np.arange(NBIN+1)*BIN
x=onset+CENTERS
```

iii. The agent viewed the common stimulus-aligned grid as the alignment mechanism for all streams.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived solely from the full trials-table `probabilityLeft` sequence; a change in probability starts a new block.

ii.
```python
prior_all=tr['probabilityLeft']
blockno=block_trial_numbers(prior_all)
```

iii. The trajectory gives no special rationale, but the code follows the task structure in which probability left is constant within a block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter is reset to zero when the prior changes and incremented otherwise. It is computed before filtering, preserving the source trial's true position, and then broadcast over all 100 bins.

ii.
```python
for i in range(1,len(prior)):
    k = 0 if prior[i] != prior[i-1] else k+1
    out[i]=k
...
np.full(NBIN,blockno[j],np.float32)
```

iii. The agent did not state a specific justification in the trajectory; its implementation implicitly preserves excluded trials in the block count.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the trials-table `choice` field. Zero/no-response trials are removed. The remaining values are mapped by the code as `-1 -> 0` and all other retained values (`+1`) to `1`, which labels the IBL directions opposite to the requested/reference left=0, right=1 convention.

ii.
```python
good &= d['choice'] != 0
...
choice=np.uint8(0 if tr['choice'][j] == -1 else 1)
```

iii. The trajectory mentions excluding no-choice trials and categorical encoding but contains no reasoning that checks the IBL sign convention. The code's mapping conflicts with the human reference's `+1 -> left -> 0`, `-1 -> right -> 1` mapping.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After removing zero choices, the scalar binary code is broadcast across all 100 time bins and ultimately stored as `uint8`.

ii.
```python
out=np.empty((4,NBIN),np.float32)
out[0]=choice
...
[o.astype(np.uint8) for o in s['output']]
```

iii. The trajectory justified categorical outputs as required by the decoder format, but did not justify the reversed recoding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trials-table `probabilityLeft` value for the trial.

ii.
```python
prior_all=tr['probabilityLeft']
pv=float(prior_all[j])
```

iii. The agent treated `probabilityLeft` as the explicit block-prior field required by the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The value is rounded to one decimal place and mapped as 0.2→0, 0.5→1, and 0.8→2, then broadcast over 100 bins and stored as `uint8`.

ii.
```python
prior=np.uint8({.2:0,.5:1,.8:2}[round(pv,1)])
out[1]=prior
```

iii. This is the mapping specified directly by the task; the trajectory did not discuss an alternative.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from raw wheel timestamps and positions. Non-finite samples and non-increasing duplicate times are removed, position is differentiated numerically, and the absolute angular velocity is used as speed.

ii.
```python
t=np.asarray(w['timestamps'],float); p=np.asarray(w['position'],float)
ok=np.isfinite(t)&np.isfinite(p); t,p=t[ok],p[ok]
u=np.r_[True,np.diff(t)>0]; t,p=t[u],p[u]
speed=np.abs(np.gradient(p,t))
```

iii. The trajectory concluded that wheel velocity had to be derived from position/timestamps and described the central difference as matching `SessionLoader`'s velocity meaning. It did not account for the loader's regular-grid interpolation and low-pass filtering.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code cleans timestamps, uses `np.gradient(position, time)`, takes the absolute value, linearly interpolates that trace at neural-bin centers, and later discretizes it. It does not interpolate position to 1 kHz or apply the reference 20 Hz low-pass filter.

ii.
```python
speed=np.abs(np.gradient(p,t))
...
ws=np.interp(x,wt,wv).astype(np.float32)
```

iii. The agent used direct derivation because the installed `SessionLoader` behavior path attempted writes into a read-only cache. It believed the gradient preserved the loader's velocity semantics, but did not reproduce its filtering pipeline.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Two thresholds are computed at the 1/3 and 2/3 quantiles of every retained wheel sample pooled across all converted sessions. `np.digitize` assigns low/medium/high codes 0/1/2.

ii.
```python
qw=np.quantile(np.concatenate([s['wheel'] for s in sessions]),[1/3,2/3])
...
o[2]=np.digitize(o[2],qw).astype(np.float32)
```

iii. The trajectory explicitly chose “global finite tertiles” for reproducible, balanced behavior outputs. The human reference instead uses each session's own 33rd/67th percentiles.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Absolute query times are stimulus onset plus the neural-bin centers, and speed is linearly interpolated at those times. Because the code does not enforce window coverage, `np.interp` can use a boundary value outside the sampled range.

ii.
```python
x=onset+CENTERS
ws=np.interp(x,wt,wv).astype(np.float32)
```

iii. The agent reasoned that all IBL streams share the session clock, so evaluating wheel speed at the neural-bin centers aligns them.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `_ibl_<side>Camera.times.npy` and `<side>Camera.ROIMotionEnergy.npy`. The left camera is tried first and the right camera is used as fallback. Arrays are truncated to their common length and non-finite pairs are removed.

ii.
```python
for cam in ('left','right'):
    t=np.asarray(one.load_dataset(eid,f'_ibl_{cam}Camera.times.npy',collection='alf'),float)
    v=np.asarray(one.load_dataset(eid,f'{cam}Camera.ROIMotionEnergy.npy',collection='alf'),float).squeeze()
```

iii. The trajectory inspected the reference and specifically concluded that it uses left-camera motion energy when available and otherwise right, rather than averaging cameras.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Apart from length/finite-value cleanup, the released motion energy is not filtered or normalized. It is linearly interpolated at neural-bin centers, then discretized using thresholds pooled across all sessions.

ii.
```python
n=min(len(t),len(v)); t,v=t[:n],v[:n]
ok=np.isfinite(t)&np.isfinite(v)
...
me=np.interp(x,mt,mv).astype(np.float32)
```

iii. The agent said direct loading avoided a current `SessionLoader` attempt to create derived files in the read-only cache, and that the released motion-energy stream itself should be used.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Thresholds are the 1/3 and 2/3 quantiles of all retained motion-energy samples pooled over the full converted dataset. Values are digitized to 0/1/2.

ii.
```python
qm=np.quantile(np.concatenate([s['whisk'] for s in sessions]),[1/3,2/3])
...
o[3]=np.digitize(o[3],qm).astype(np.float32)
```

iii. As for wheel speed, the agent explicitly selected global tertiles for reproducibility and balance; the human reference uses per-session percentiles.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Motion energy is interpolated at `stimOn_times + CENTERS`, the same absolute times represented by the neural bins. No trial-coverage check is made, so out-of-range queries receive endpoint values from `np.interp`.

ii.
```python
x=onset+CENTERS
me=np.interp(x,mt,mv).astype(np.float32)
```

iii. The agent relied on synchronized IBL timestamps and common bin-center sampling as the alignment procedure.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Non-finite trial fields cause trial removal. Non-finite wheel/camera samples are removed; wheel timestamps are deduplicated and mismatched camera arrays are truncated. A missing camera, missing probes, or fewer than two retained trials raises an exception, causing the whole session to be logged and omitted. Existing session checkpoints are reused. The code does not drop individual trials lacking full wheel/camera coverage, and a missing probe can cause the session to fail rather than merely skipping that probe.

ii.
```python
ok=np.isfinite(t)&np.isfinite(v)
n=min(len(t),len(v)); t,v=t[:n],v[:n]
...
if len(idx)<2: raise RuntimeError('fewer than two valid trials')
...
except Exception as ex:
    warnings.warn(f'{eid}: {ex}'); failures.append((eid,repr(ex)))
```

iii. The trajectory reports that 14 sessions without required whisker streams were excluded so every requested output would be present. It viewed session omission as appropriate, but did not address the reference's finer-grained stream-coverage and missing-probe handling.

## 10-a. What are the most time-consuming steps of the code?

i. Loading and merging large per-probe spike-sorting arrays is the dominant per-session work; full conversion also spends substantial time reading/writing session checkpoints and assembling the 25 GB pickle. The trajectory's runtime reports show NFS I/O as the practical bottleneck.

ii.
```python
s,c,ch=sl.load_spike_sorting()
merged=sl.merge_clusters(s,c,ch).to_df()
...
pickle.dump(s,open(tmp,'wb'),protocol=5)
```

iii. The agent repeatedly described spike arrays as hundreds of megabytes, observed workers in I/O wait, and introduced resumable shards to improve throughput.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main per-trial loop independently bins spikes, interpolates two behavior streams, and builds arrays; it could be partly vectorized using trial-index offsets and concatenated interpolation queries. `block_trial_numbers` could be vectorized with change points and cumulative indices. The probe and session loops reflect variable-sized independent records and are less naturally vectorized.

ii.
```python
for j in idx:
    ...
    counts=np.bincount(flat,minlength=ncl*NBIN).reshape(ncl,NBIN)
    ws=np.interp(x,wt,wv).astype(np.float32)
    me=np.interp(x,mt,mv).astype(np.float32)
```

iii. The trajectory focused on parallelizing independent sessions rather than vectorizing within-session loops; it did not explicitly analyze these loops.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly casts the constant `CENTERS` vector for every trial, separately searches spike bounds and interpolates behavior for every trial, and converts outputs more than once. Session data are serialized to checkpoints and then read back for assembly; raw wheel/whisker traces are also duplicated alongside their copies in `output` until global thresholds are computed.

ii.
```python
inp=np.vstack((CENTERS.astype(np.float32), ...))
...
o[2]=np.digitize(o[2],qw).astype(np.float32)
o[:]=o.astype(np.uint8)
...
'output':[[o.astype(np.uint8) for o in s['output']] for s in sessions]
```

iii. The trajectory justified checkpoint rereads as necessary for resumability and global threshold assembly, and compact casts as a memory optimization; it did not identify the smaller repeated operations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `feedbackType`, `response_times`, and several other trial fields are loaded only to participate in filtering and are not saved. Continuous wheel and whisker samples are stored both in each trial's temporary output and in concatenated session vectors, then discarded after global discretization. Camera-side labels and source-trial counts survive only as metadata. The assignment `o[:] = o.astype(np.uint8)` converts to integer and immediately writes back into a float array, after which the array is cast to `uint8` again.

ii.
```python
keys=['goCue_times','choice','feedback_times','stimOn_times','response_times',
      'firstMovement_times','probabilityLeft','feedbackType']
...
'wheel':np.concatenate(wheel_raw), 'whisk':np.concatenate(whisk_raw)
...
o[:]=o.astype(np.uint8)
...
[o.astype(np.uint8) for o in s['output']]
```

iii. The agent considered temporary continuous behavior necessary to calculate full-dataset tertiles and used compact final types to limit a very large output. It did not discuss the redundant cast or fields used solely by its extra trial filters.
