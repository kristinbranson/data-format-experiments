# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is loaded through the ONE API against the local IBL cache, but the list of what to load comes from the reference repository's own release table, `/app/code/code_zhang2025/data/bwm_release.csv` (699 probe insertions / 459 sessions / 139 subjects), rather than from `one.search`. The unique `eid`s of that CSV define the session list; the rows of the CSV for a given `eid` define the probe insertions (`pid`, `probe_name`) of that session. If `/app/data/DATALIMIT_SUBSET.csv` exists the session list is intersected with it (it did not exist for this run, so all 459 were attempted). Per session the AI loads four streams with plain ONE calls instead of `SessionLoader`: the trials object (`one.load_object(eid,'trials',collection='alf')`), the wheel object, the side-camera motion energy and camera timestamps (`one.load_dataset`), and the spike sorting per probe through `SpikeSortingLoader` + `merge_clusters`. Sessions are processed one at a time in a loop, each cached to its own pickle under `/app/session_cache/<eid>.pkl`, and a final pass re-reads all the per-session pickles and assembles `/app/converted_data.pkl`. Any session that raises (e.g. no whisker motion energy) is warned about, recorded in `failures`, and skipped. 445 of 459 sessions converted; 14 were dropped for a missing whisker stream.

ii.
```python
RELEASE=Path('/app/code/code_zhang2025/data/bwm_release.csv')
...
rel=pd.read_csv(RELEASE)
eids=list(dict.fromkeys(rel.eid))
subset=Path('/app/data/DATALIMIT_SUBSET.csv')
if subset.exists():
    ss=pd.read_csv(subset); allowed=set(ss['eid'] if 'eid' in ss else ss.iloc[:,0]); eids=[e for e in eids if e in allowed]
...
one=ONE(); failures=[]
for k,eid in enumerate(eids):
    f=CACHE/f'{eid}.pkl'
    if f.exists() and not args.rebuild: ...continue
    try:
        s=convert_session(one,eid,rel[rel.eid==eid])
        tmp=f.with_suffix('.tmp'); pickle.dump(s,open(tmp,'wb'),protocol=5); os.replace(tmp,f)
    except Exception as ex:
        warnings.warn(f'{eid}: {ex}'); failures.append((eid,repr(ex)))
```
```python
def trial_table(one,eid):
    tr=one.load_object(eid,'trials',collection='alf')
...
def load_camera(one,eid):
    t=np.asarray(one.load_dataset(eid,f'_ibl_{cam}Camera.times.npy',collection='alf'),float)
    v=np.asarray(one.load_dataset(eid,f'{cam}Camera.ROIMotionEnergy.npy',collection='alf'),float).squeeze()
...
def load_wheel(one,eid):
    w=one.load_object(eid,'wheel',collection='alf')
...
def load_spikes(one,rows):
    for row in rows.itertuples():
        sl=SpikeSortingLoader(one=one,pid=row.pid,eid=row.eid,pname=row.probe_name)
        s,c,ch=sl.load_spike_sorting()
        merged=sl.merge_clusters(s,c,ch).to_df()
```

iii. From the trajectory: the AI first tried to run the reference `prepare_data` directly, found it broke on the installed IBL API (`SessionLoader` now requires keyword arguments) and that `SessionLoader.load_motion_energy` tries to write derived camera features into the intentionally read-only NFS cache. It therefore decided to "implement the same logic directly rather than patching the repository", reading the already-released ALF arrays with ONE and taking the session/probe scope from the repository's shipped `bwm_release.csv` ("use release CSV sessions/probes"). It kept a patched copy of the reference `prepare_data` as ground truth to validate its own trial and cluster counts on one example session. Per-session checkpointing was chosen so the long run could be resumed and later sharded across 8 background workers.

## 1-b. How are the data split into subjects?

i. Not derived at all: the subject name is read from the `subject` column of `bwm_release.csv` for the session's rows. At assembly the unique subject strings are sorted and `subject_idx` holds each session's index into that list. Result: 136 subjects for 445 sessions.

ii.
```python
return {'eid':eid,'subject':str(rows.subject.iloc[0]), ...}
...
subjects=sorted(set(s['subject'] for s in sessions)); smap={s:i for i,s in enumerate(subjects)}
...
'subject_idx':np.asarray([smap[s['subject']] for s in sessions],np.int64),
```

iii. No explicit reasoning in the trajectory beyond the general decision to drive everything off the reference repository's release table, which already carries subject and lab identity per insertion, so no path parsing or extra database query is needed.

## 1-c. How are the data split into sessions?

i. A session is the `eid`, which is already the unit the release table is organised by; `dict.fromkeys(rel.eid)` gives the unique sessions in release order and each is processed independently. No splitting logic is needed.

ii.
```python
rel=pd.read_csv(RELEASE)
eids=list(dict.fromkeys(rel.eid))
...
s=convert_session(one,eid,rel[rel.eid==eid])
```

iii. Implicit: the AI states the scope is "all locally staged BWM sessions" and that "sessions are independent", which is what let it shard the run across 8 workers.

## 1-d. How are the data split into trials?

i. The trials object has one entry per trial, so the split is given by the data. The AI pulls eight per-trial columns into a dict of arrays, builds a boolean `good` mask over them, and then loops over `idx=np.flatnonzero(good)`; each surviving trial index `j` becomes one element of the session's `neural`/`input`/`output` lists.

ii.
```python
tr=one.load_object(eid,'trials',collection='alf')
keys=['goCue_times','choice','feedback_times','stimOn_times','response_times',
      'firstMovement_times','probabilityLeft','feedbackType']
d={k:np.asarray(tr[k]) for k in keys}
...
idx=np.flatnonzero(good)
for j in idx:
    onset=tr['stimOn_times'][j]
```

iii. No decision was flagged; the trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI reproduced the reference repository's `load_trials_and_mask` exactly, as called by `prepare_data` (`max_trial_len=10.0`, defaults otherwise). Four conditions, all ANDed: (1) every one of the eight trial fields must be finite (a superset of the reference's `nan_exclude` list, which the AI extended with `goCue_times` and `response_times`); (2) reaction time `firstMovement_times - stimOn_times` in [0.08, 2.0] s; (3) trial duration `feedback_times - goCue_times <= 10 s`; (4) `choice != 0`, i.e. no-response trials dropped. Unbiased (`probabilityLeft == 0.5`) blocks are kept, matching the reference default `exclude_unbiased=False`. A session with fewer than two surviving trials is dropped. No filtering is applied for whether the wheel or camera streams actually span the trial window.

ii.
```python
n=len(d['choice']); good=np.ones(n,bool)
for k in keys: good &= np.isfinite(d[k])
rt=d['firstMovement_times']-d['stimOn_times']
good &= (rt >= 0.08) & (rt <= 2.0)
good &= (d['feedback_times']-d['goCue_times'] <= 10.0)
good &= d['choice'] != 0
```
```python
idx=np.flatnonzero(good)
if len(idx)<2: raise RuntimeError('fewer than two valid trials')
```

iii. This was the one thing the AI iterated on empirically. Its first version kept 538 of 565 trials on the example session, while a patched copy of the reference `load_trials_and_mask` kept 407. The AI treated the reference count as authoritative ("the patched reference run establishes the authoritative trial filter: 407 of 565 trials"), introspected the reference function's signature to find the defaults it had missed, found "the reference applies a default reaction-time filter of 0.08–2.0 seconds ... in addition to finite-event, duration, and no-choice filters", added it and confirmed the count matched before launching the full run.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` from `SpikeSortingLoader.load_spike_sorting()`, for every probe insertion of the session listed in `bwm_release.csv`. The merged cluster table (`merge_clusters(...).to_df()`) is used only for the cluster count and for the per-neuron `acronym` written into `brain_region_idx`; it does not gate which spikes are used.

ii.
```python
s,c,ch=sl.load_spike_sorting()
merged=sl.merge_clusters(s,c,ch).to_df()
st=np.asarray(s['times']); sc=np.asarray(s['clusters'],np.int64)
n=len(merged); valid=(sc>=0)&(sc<n)
times.append(st[valid]); clus.append(sc[valid]+offset)
reg=np.asarray(merged['acronym'].fillna('void').astype(str)) if 'acronym' in merged else np.repeat('void',n)
regions.extend(reg.tolist()); offset += n
```

iii. The AI followed the reference `prepare_data`, which loads spikes/clusters per pid and then merges probes; it noted "raw clusters do not directly include anatomical labels; the reference loader uses `SpikeSortingLoader.merge_clusters`, which adds QC and atlas fields".

## 2-b. How is the `neural` data processed?

i. Probes of a session are concatenated into one population: cluster ids of the second probe are offset by the cluster count of the first, and the pooled spike times are re-sorted (stable mergesort), reproducing the reference `merge_probes`. For each kept trial, the spikes falling in `[onset-0.5, onset+1.5)` are sliced with `searchsorted`, assigned a bin index by `floor((t-edges[0])/0.02)`, and counted with a single flat `bincount` into an (n_clusters, 100) matrix. The values stored are raw **spike counts per 20 ms bin** (not rates), clipped at 255 and cast to `uint8`. No smoothing, no z-scoring, no firing-rate conversion.

ii.
```python
t=np.concatenate(times); c=np.concatenate(clus)
order=np.argsort(t,kind='mergesort')
return t[order],c[order],regions
```
```python
onset=tr['stimOn_times'][j]; edges=onset+OFF0+np.arange(NBIN+1)*BIN
lo=np.searchsorted(st,edges[0],'left'); hi=np.searchsorted(st,edges[-1],'left')
tt=st[lo:hi]; cc=sc[lo:hi]
tb=np.floor((tt-edges[0])/BIN).astype(np.int64)
flat=cc*NBIN+tb
counts=np.bincount(flat,minlength=ncl*NBIN).reshape(ncl,NBIN)
counts=np.minimum(counts,255).astype(np.uint8)
```

iii. "The reference configuration is confirmed as 100 time bins at 20 ms over [-0.5, 1.5] s around `stimOn_times`", and probes are merged because the reference does so (its `merge_probes` docstring: probes in one session are not statistically independent). `uint8` counts were chosen explicitly for size: "Large sessions are handled with compact uint8 spike-count matrices". The validator warned 189,315 times that neural dtype is `uint8` rather than float32, but converts at training time.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No quality-control filter is applied.** Every sorted cluster of every released probe is kept, including MUA/noise-labelled clusters and clusters whose histology places them outside the brain (`void`/`root`); only clusters with an out-of-range id are dropped (`0 <= id < n_clusters`, which never removes anything in practice). The result is 600,174 neurons over 445 sessions (mean 1,349, max 3,140 per session), versus 72,422 (mean 164) for the expert reference, and a 26.7 GB pickle.

ii.
```python
# Cluster IDs index the merged table. Keep all clusters as in prepare_data(qc=None).
n=len(merged); valid=(sc>=0)&(sc<n)
```
```python
'neural_measure':'spike counts per 20 ms bin; all release clusters (reference qc=None)',
```

iii. This was an explicit, investigated decision. The AI first planned strict QC ("load/merge release probes with `SpikeSortingLoader` QC label >= 1"), then checked the numbers on the example probe ("898 total clusters but only 76 with the strict IBL QC label of 1") and said "the task explicitly requires matching reference curation; we must verify whether the methods code uses that strict label". It then grepped the actual call site and concluded: "The reference `prepare_data` deliberately loads all clusters (`qc=None`), not only label-1 units; therefore the converter should retain all clusters from the BWM release probes." This is consistent with `/app/methods.txt`, which says the method paper bins "spike counts using all neurons, sorted by Kilosort 2.5, from each session".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams are already on one synchronised session clock, so alignment is just building the bin edges from each trial's `stimOn_times`: `edges = stimOn_times + (-0.5) + k*0.02`, k = 0..100. Spikes are sliced by absolute time between the first and last edge and binned relative to `edges[0]`, which puts bin 0 at 500 ms before stimulus onset and bin 99 ending 1.5 s after it. No resampling or cross-stream clock correction is done.

ii.
```python
onset=tr['stimOn_times'][j]; edges=onset+OFF0+np.arange(NBIN+1)*BIN
lo=np.searchsorted(st,edges[0],'left'); hi=np.searchsorted(st,edges[-1],'left')
tb=np.floor((tt-edges[0])/BIN).astype(np.int64)
```
```python
'temporal_alignment_event':'visual stimulus onset (stimOn_times)','off_start':-0.5,'off_end':1.5,
```

iii. "The reference configuration is confirmed as 100 time bins at 20 ms over [-0.5, 1.5] s around `stimOn_times`" — taken from the reference decoding config and from `/app/methods.txt` ("For choice, we align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset"), which is also what the task instructions ask for.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial, identical for every trial and session; `metadata['time_bin_size'] = 20.0` ms. Spikes are binned once directly at 20 ms, so there is no rebinning or interpolation of the neural data. (The method paper uses 50 ms for choice/prior and 20 ms for the dynamic behaviours; the AI used 20 ms throughout, as the target format requires a single bin size.)

ii.
```python
BIN=.02; OFF0=-.5; OFF1=1.5; NBIN=100
CENTERS=np.arange(NBIN,dtype=np.float64)*BIN + OFF0 + BIN/2
...
'time_bin_size':20.0,
```

iii. Same as 2-d: the 20 ms / 100-bin configuration was read off the reference code and methods text and confirmed against the validator.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from raw data at all — it is the fixed bin-centre grid defined by the analysis window (`-0.5 + 0.02k + 0.01`, giving -0.49 … 1.49 s). It is implicitly tied to `stimOn_times`, the event the window is anchored on, and is the same vector for every trial of every session.

ii.
```python
CENTERS=np.arange(NBIN,dtype=np.float64)*BIN + OFF0 + BIN/2
...
inp=np.vstack((CENTERS.astype(np.float32),np.full(NBIN,blockno[j],np.float32)))
```

iii. No separate justification; it falls out of the alignment window chosen in 2-d.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None. The constant `CENTERS` vector is cast to float32 and stacked as row 0 of the (2, 100) input array for every trial.

ii.
```python
inp=np.vstack((CENTERS.astype(np.float32),np.full(NBIN,blockno[j],np.float32)))
```
```python
'input_names':['time since stimulus onset','trial number in block'],
```

iii. N/A.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the centre of exactly the same bins the spikes are counted into: the neural bin edges are `onset + OFF0 + k*BIN` and `CENTERS[k] = OFF0 + k*BIN + BIN/2`, so input column k and neural column k describe the same 20 ms interval. The AI verified the reported range (-0.49, 1.49) with the validator.

ii.
```python
edges=onset+OFF0+np.arange(NBIN+1)*BIN      # neural
x=onset+CENTERS                             # behaviour sampling points
CENTERS=np.arange(NBIN,dtype=np.float64)*BIN + OFF0 + BIN/2
```

iii. "Its time input prints as [-0.5, 1.5] due to rounding of the actual bin centers" — the AI checked the validator summary and confirmed the grid was the intended bin-centre grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table only. Because the block prior is constant within a block, a change in its value marks a block boundary; the trials table carries no explicit block id.

ii.
```python
prior_all=tr['probabilityLeft']; blockno=block_trial_numbers(prior_all)
```

iii. No explicit statement in the trajectory; the AI noted early that the trials table contains "the expected three prior levels" and used them to recover blocks.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A running counter over **all** trials of the session (computed before the quality mask is applied, so dropped trials still advance the animal's true position in the block), reset to 0 whenever `probabilityLeft` differs from the previous trial, and 0 for the first trial. The scalar is broadcast across all 100 bins as row 1 of the input array. Observed range 0–98, matching the reference.

ii.
```python
def block_trial_numbers(prior):
    out=np.zeros(len(prior),np.float32); k=0
    for i in range(1,len(prior)):
        k = 0 if prior[i] != prior[i-1] else k+1
        out[i]=k
    return out
...
inp=np.vstack((CENTERS.astype(np.float32),np.full(NBIN,blockno[j],np.float32)))
```

iii. Not discussed explicitly; the AI listed it among the "categorical encodings required by the task" and confirmed the resulting value range with the validator.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 / -1 / 0. Trials with `choice == 0` (no response) are already removed by the mask. The AI maps `-1 → 0` and `+1 → 1`, and labels the classes `['left','right']`, i.e. it treats `choice == -1` as "left". In the IBL convention (`ibllib/brainbox/behavior/training.py`: `rightward = trials.choice == -1`, and `# choice == -1 means contrast on right hand side`) `choice == +1` is a leftward choice and `-1` is rightward, so this mapping is inverted relative to the instructions' `left = 0, right = 1`. The resulting class fractions (0: 0.4921, 1: 0.5079) are the mirror image of the reference's (0: 0.5075, 1: 0.4925).

ii.
```python
choice=np.uint8(0 if tr['choice'][j] == -1 else 1)
...
out=np.empty((4,NBIN),np.float32); out[0]=choice; ...
'output_values':[['left','right'], ...]
```

iii. No justification is given anywhere — neither in the code (the line has no comment) nor in the trajectory, which only mentions "encode choice and prior" and excluding `choice == 0`. The sign convention was never checked against the IBL documentation or against `contrastLeft`/`feedbackType`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Nothing beyond the recode described above: the per-trial scalar is broadcast across all 100 bins (row 0 of the (4, 100) output array) and finally cast to `uint8`, so choice is stored as a constant time series rather than a per-trial scalar.

ii.
```python
out=np.empty((4,NBIN),np.float32); out[0]=choice; out[1]=prior; out[2]=ws; out[3]=me
...
'output':[[o.astype(np.uint8) for o in s['output']] for s in sessions],
```

iii. The format instructions ask for time-varying outputs "if at all possible", and the AI kept all four outputs on the same (4, 100) grid so that static and dynamic variables share one array.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, rounded to one decimal and mapped through `{0.2: 0, 0.5: 1, 0.8: 2}`, exactly the mapping given in the instructions. Trials with a non-finite `probabilityLeft` are already excluded; a value outside the three expected ones would raise a `KeyError` and drop the whole session.

ii.
```python
pv=float(prior_all[j]); prior=np.uint8({.2:0,.5:1,.8:2}[round(pv,1)])
```

iii. The AI confirmed on the example session that the trials table contains "the expected three prior levels" before writing the mapping.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the recode; the value is broadcast across the 100 bins as row 1 of the output array. Unbiased (0.5) blocks are retained, giving fractions 0.417 / 0.141 / 0.442, essentially identical to the reference (0.418 / 0.141 / 0.442).

ii.
```python
out[1]=prior
```

iii. N/A.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The raw ALF wheel object, `wheel.timestamps` and `wheel.position` (radians), loaded directly with `one.load_object` rather than through `SessionLoader`.

ii.
```python
def load_wheel(one,eid):
    w=one.load_object(eid,'wheel',collection='alf')
    t=np.asarray(w['timestamps'],float); p=np.asarray(w['position'],float)
```

iii. "Wheel velocity must be derived from position/timestamps" — the AI chose not to use `SessionLoader` for any behavioural stream after `load_motion_energy` failed by trying to write into the read-only cache.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Non-finite samples are dropped, non-monotonic/duplicate timestamps are dropped, and the speed is the absolute value of a **central difference of the raw encoder position against its irregular timestamps** (`np.abs(np.gradient(p, t))`). The trace is then sampled at the 100 bin centres of each trial with `np.interp`. This is *not* what the reference does: the reference's `load_target_behavior('wheel-speed')` takes `np.abs(SessionLoader.wheel['velocity'])`, and `SessionLoader.load_wheel` first interpolates position onto a uniform 1000 Hz grid and then differentiates it with a 20 Hz low-pass Butterworth filter. So the AI's speed is an unsmoothed, non-uniformly-sampled estimate of the same physical quantity; the code comment claims it matches "SessionLoader's velocity meaning", which is true of the units (rad/s) but not of the filtering.

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

iii. The stated reason for bypassing `SessionLoader` is in the module docstring: "Existing ALF camera arrays are read directly because current SessionLoader tries to write derived features into the intentionally read-only source cache." That argument concerns the camera; the AI extended it to the wheel without separately testing whether `SessionLoader.load_wheel` (which computes in memory) would have worked, and without noting the loss of the 20 Hz filter.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Three classes split at the **global** 1/3 and 2/3 quantiles computed over the concatenation of every trial of every session (18.9 M samples), not per session. The thresholds (0.0626 and ~0.4 rad/s) are stored in `metadata['behavior_bin_edges']`. Globally the three classes are exactly equal in size (0.3333 each), matching the reference's marginal fractions, but within an individual session the classes are not balanced.

ii.
```python
qw=np.quantile(np.concatenate([s['wheel'] for s in sessions]),[1/3,2/3])
...
o[2]=np.digitize(o[2],qw).astype(np.float32)
...
'behavior_bin_edges':{'wheel_speed':qw.tolist(),'whisker_motion_energy':qm.tolist()},
```

iii. "Global finite tertiles give reproducible, balanced three-class behavior outputs" (code comment); earlier the AI described this as "quantile-discretize the two continuous behaviors into three global/session-consistent bins", i.e. it explicitly wanted one common threshold so that the class label means the same physical speed in every session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Linear interpolation of the wheel-speed trace at `stimOn_times + CENTERS`, i.e. the centres of exactly the 100 bins the spikes are counted into, on the same session clock. Bin k of the output therefore corresponds to bin k of the neural matrix. Values outside the available wheel samples are not dropped — `np.interp` clamps to the first/last sample.

ii.
```python
x=onset+CENTERS
ws=np.interp(x,wt,wv).astype(np.float32)
```

iii. Follows directly from the shared bin grid; the AI stated its plan as "interpolate wheel speed and left/right whisker motion energy onto bin centers".

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The released side-camera ROI motion energy, `<cam>Camera.ROIMotionEnergy.npy`, with frame times `_ibl_<cam>Camera.times.npy`. The left camera is used when it loads and has >2 finite samples, otherwise the right camera; the camera actually used is recorded per session in `metadata['session_info']`. 14 sessions had neither and were dropped.

ii.
```python
def load_camera(one,eid):
    # Reference uses left whisker ME and falls back to right if unavailable.
    for cam in ('left','right'):
        try:
            t=np.asarray(one.load_dataset(eid,f'_ibl_{cam}Camera.times.npy',collection='alf'),float)
            v=np.asarray(one.load_dataset(eid,f'{cam}Camera.ROIMotionEnergy.npy',collection='alf'),float).squeeze()
            n=min(len(t),len(v)); t,v=t[:n],v[:n]
            ok=np.isfinite(t)&np.isfinite(v)
            if ok.sum()>2: return t[ok],v[ok],cam
        except Exception: pass
    raise RuntimeError('no whisker motion-energy stream')
```

iii. The AI grepped the reference `bin_behaviors` and concluded: "Whisker motion energy uses the left camera when available and falls back to right, rather than averaging cameras" — an explicit correction of its own earlier plan to average the two cameras.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is: length-mismatch between times and values is truncated to the shorter, non-finite samples are dropped, and the trace is linearly interpolated onto the 100 bin centres of each trial. No filtering, normalisation or per-session standardisation before discretisation.

ii.
```python
me=np.interp(x,mt,mv).astype(np.float32)
```

iii. "Direct stream inspection confirms camera motion energy can be loaded without SessionLoader and aligned by camera timestamps." The reference also uses the released `whiskerMotionEnergy` column unmodified.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same rule as the wheel: three classes split at the global 1/3 and 2/3 quantiles pooled over all sessions, applied after the whole dataset has been converted. Note that this pools left-camera (60 Hz) and right-camera (150 Hz) motion energy, whose absolute values depend on the camera, the ROI size and the video resolution, into one set of thresholds.

ii.
```python
qm=np.quantile(np.concatenate([s['whisk'] for s in sessions]),[1/3,2/3])
...
o[3]=np.digitize(o[3],qm).astype(np.float32)
```

iii. Same justification as 7-c: "Global finite tertiles give reproducible, balanced three-class behavior outputs."

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Linear interpolation of the camera trace at `stimOn_times + CENTERS`, the same bin centres as the neural data and on the same session clock, so it shares the neural time axis bin for bin. As with the wheel, no check is made that the camera actually covers the trial window; outside its range `np.interp` clamps.

ii.
```python
x=onset+CENTERS
me=np.interp(x,mt,mv).astype(np.float32)
```

iii. As 8-b: the camera timestamps are already on the session clock, so evaluating at the bin centres is all that is needed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms, all "drop and record":
- Trials with a non-finite value in any of the eight required trial fields are dropped (superset of the reference's `nan_exclude`).
- Non-finite wheel or camera samples are dropped before interpolation; wheel samples with non-increasing timestamps are dropped; camera times/values of unequal length are truncated to the shorter.
- A missing whisker stream falls back left → right, and if neither loads the whole session raises and is skipped.
- Sessions with fewer than two valid trials, or with no release probes, raise and are skipped.
- Every per-session exception is caught, emitted as a warning and appended to `failures`, which is written into `metadata['failed_sessions']` (14 sessions, all for a missing whisker stream).

What is *not* handled: there is no check that the wheel/camera streams span the trial window, so a trial at the edge of a recording gets a silently flat, edge-clamped behavioural trace instead of being dropped (the reference code does apply such a check — "target data starts too late" / "ends too early" — and the expert reference implements it as `covered()`). Also `counts=np.minimum(counts,255)` silently clips, and a spike landing exactly on the last bin edge could in principle produce `tb == 100` and mis-attribute a count (or raise on reshape) rather than being clipped into the last bin.

ii.
```python
n=len(d['choice']); good=np.ones(n,bool)
for k in keys: good &= np.isfinite(d[k])
```
```python
n=min(len(t),len(v)); t,v=t[:n],v[:n]
ok=np.isfinite(t)&np.isfinite(v)
if ok.sum()>2: return t[ok],v[ok],cam
```
```python
except Exception as ex:
    warnings.warn(f'{eid}: {ex}'); failures.append((eid,repr(ex)))
...
'failed_sessions':failures
```

iii. "Fourteen sessions were excluded because required streams (notably whisker motion energy) were unavailable; this is appropriate because every requested output must be present and aligned." The AI also deliberately reviewed the failure list before assembly to check the exclusions were legitimate missing-data cases rather than loader bugs.

## 10-a. What are the most time-consuming steps of the code?

i. (1) Reading the spike sorting: `sl.load_spike_sorting()` is called per probe with the default `SPIKES_ATTRIBUTES = ['clusters','times','amps','depths']`, so twice as many large arrays are read off the NFS cache as the conversion needs, and `merge_clusters` occasionally recomputes cluster metrics (the AI observed "metric-computation progress bars"). (2) The per-trial spike-binning loop, whose `bincount` is over `n_clusters * 100` entries — with all clusters kept this is up to 314,000 bins per trial for ~500 trials. (3) Serialisation: 445 session pickles are written and then read back, and a 26.7 GB final pickle is written — the dominant I/O cost, and a direct consequence of keeping all 600 k clusters. The AI measured ~8–9 sessions/min single-threaded (≈55 min total) and sharded the run across 8 background workers to cut it to a few minutes.

ii.
```python
sl=SpikeSortingLoader(one=one,pid=row.pid,eid=row.eid,pname=row.probe_name)
s,c,ch=sl.load_spike_sorting()
merged=sl.merge_clusters(s,c,ch).to_df()
```
```python
counts=np.bincount(flat,minlength=ncl*NBIN).reshape(ncl,NBIN)
```
```python
tmp=Path(args.output+'.tmp'); pickle.dump(data,open(tmp,'wb'),protocol=5); os.replace(tmp,args.output)
```

iii. "The sequential run has reached session 59. At the current rate, completing 459 sessions would take roughly another 45–55 minutes. Since sessions are independent and the host has ample CPU/RAM, conversion can be safely parallelized while retaining per-session atomic checkpoints." The AI then added `--shard`/`--cache-only` and ran 8 disjoint workers; it also chose `uint8` neural arrays specifically to keep the pickle manageable.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three.
- The main per-trial loop in `convert_session`: the spike binning could be done for all trials at once with a single `bincount` whose flat index also encodes the trial (`trial*ncl*NBIN + cluster*NBIN + bin`), and the two `np.interp` calls could be replaced by one `np.interp` over the concatenated query vector `(stim_on[:,None] + CENTERS).ravel()` since both traces are monotone in time.
- `block_trial_numbers` is a Python loop over every trial; it is a group-wise cumulative count and can be written vectorised (`(prior != shift(prior)).cumsum()` + `groupby(...).cumcount()`, as the expert reference does).
- The final relabelling loop `for s in sessions: for o in s['output']` calls `np.digitize` once per trial (189,315 times) on a 2×100 slice; it could digitize a whole session's stacked behaviour at once.

ii.
```python
for j in idx:
    ...
    counts=np.bincount(flat,minlength=ncl*NBIN).reshape(ncl,NBIN)
    ws=np.interp(x,wt,wv).astype(np.float32)
    me=np.interp(x,mt,mv).astype(np.float32)
```
```python
def block_trial_numbers(prior):
    out=np.zeros(len(prior),np.float32); k=0
    for i in range(1,len(prior)):
        k = 0 if prior[i] != prior[i-1] else k+1
```
```python
for s in sessions:
    for o in s['output']:
        o[2]=np.digitize(o[2],qw).astype(np.float32); o[3]=np.digitize(o[3],qm).astype(np.float32)
```

iii. Not discussed. The AI addressed throughput by process-level parallelism (8 shards) rather than vectorisation.

## 10-c. What processing does the code repeat multiple times?

i.
- Every session is pickled to `/app/session_cache/<eid>.pkl` and then, in the assembly pass, read back and re-pickled into the 26.7 GB output — the full dataset is serialised and deserialised twice, and both copies stay on disk.
- The behavioural traces are stored twice per session: once inside `output` rows 2/3 and once again in the flat `s['wheel']` / `s['whisk']` arrays kept only to compute the global quantiles.
- The output arrays are cast to `uint8` twice, once in-place into the float32 buffer and once when the dict is built.
- `rel[rel.eid==eid]` rescans the 699-row release table once per session (cheap, but a `groupby` would do it once).
- The per-trial `np.searchsorted` on the full spike array is repeated per trial rather than computed once for all trial boundaries.

ii.
```python
tmp=f.with_suffix('.tmp'); pickle.dump(s,open(tmp,'wb'),protocol=5); os.replace(tmp,f)
...
for eid in eids:
    f=CACHE/f'{eid}.pkl'
    if f.exists(): sessions.append(pickle.load(open(f,'rb')))
```
```python
return {... 'output':outputs, ..., 'wheel':np.concatenate(wheel_raw),'whisk':np.concatenate(whisk_raw), ...}
```
```python
o[:]=o.astype(np.uint8)
...
'output':[[o.astype(np.uint8) for o in s['output']] for s in sessions],
```

iii. The duplication was a deliberate trade: "It will cache each converted session separately for resumability, calculate global behavior tertiles, then assemble the final pickle" — the checkpoint files are what made the interrupted single-process run resumable and the 8-way sharding possible.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Retaining all 600,174 clusters.** The decoder projects each session onto 100 PCs, so the ~8× extra noise/MUA units (and the units whose histology places them outside the brain, kept under the `void`/`root` acronyms) mostly add I/O, 26.7 GB of storage and PCA dilution; measured decoding fell below the reference for choice (0.561 vs 0.614), prior (0.546 vs 0.666) and wheel speed (0.560 vs 0.606).
- Loading `amps` and `depths` for every spike (default `SPIKES_ATTRIBUTES`) when only `times` and `clusters` are used — roughly doubles the spike-file I/O.
- Building and pickling the per-session `wheel`/`whisk` raw concatenations, which are used once for the global quantiles and never appear in the output.
- `np.minimum(counts,255)` — 255 spikes in a 20 ms bin is 12.75 kHz, so the clip never binds.
- Building `out` as float32 and then casting it to `uint8` twice.
- 576 fine-grained Allen acronyms are stored as `brain_regions` (the reference code maps to the coarser Beryl atlas before using regions); the decoder run here uses all neurons and never indexes by region.
- The whole per-session cache directory (a second full copy of the dataset) is left on disk after assembly.

ii.
```python
s,c,ch=sl.load_spike_sorting()          # loads amps and depths too
```
```python
counts=np.minimum(counts,255).astype(np.uint8)
```
```python
'wheel':np.concatenate(wheel_raw),'whisk':np.concatenate(whisk_raw)
```
```python
reg=np.asarray(merged['acronym'].fillna('void').astype(str)) if 'acronym' in merged else np.repeat('void',n)
```

iii. The AI never revisited the cost of `qc=None` after choosing it; it justified the choice purely on fidelity to `prepare_data` and addressed the resulting size with compact dtypes ("compact integer spike counts are important", "The final pickle is 25 GB"). The redundant caches were accepted for resumability, and the extra spike attributes and the `255` clip were never discussed.
