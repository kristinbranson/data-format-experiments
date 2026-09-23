# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the BWM release parquet tables (`sessions.pqt` and `datasets.pqt`) from the ONE cache to identify available sessions and datasets. It then selects exactly 10 sessions (one per subject) using a seeded random selection (`np.random.seed(42)`) from `bwm_release.csv`. For each selected session, individual data files (spikes, clusters, trials, wheel, camera) are downloaded directly from the IBL public S3 bucket using constructed URLs with dataset UUIDs, rather than using the ONE API. Downloaded files are cached locally by UUID.

ii.
```python
QPATH=ROOT/'data/one_cache/Brainwidemap/datasets.pqt'; SPATH=ROOT/'data/one_cache/Brainwidemap/sessions.pqt'
RELEASE=ROOT/'code/code_zhang2025/data/bwm_release.csv'

def selected_eids(n):
 x=pd.read_csv(RELEASE,index_col=0); np.random.seed(42)
 subs=np.random.choice(np.unique(x.subject),10,replace=False); by=x.groupby('subject').indices
 eids=[str(x.iloc[by[s][0]].eid) for s in subs]
 return list(zip(subs[:n],eids[:n]))
```

```python
def load_record(eid,g,srow,needle,unrevisioned=False,allow_pickle=False):
 did,r=get_record(g,needle,unrevisioned); f=fetch(eid,did,r,srow)
 return pd.read_parquet(f) if f.suffix=='.pqt' else np.load(f,allow_pickle=allow_pickle)
```

iii. The AI justified using 10 sessions based on the reference code README demonstrating `--n_sessions 10` and the methods paper reporting IBL comparisons across 10 sessions. The AI also argued that processing all ~445 sessions would require ~346 GB of downloads and exceed the 15-minute workflow target. The AI used direct S3 downloads rather than the ONE API because the local cache contained only metadata, not numerical payloads.

## 1-b. How are the data split into subjects?

i. The AI selects 10 unique subjects via `np.random.seed(42)` and `np.random.choice` from the BWM release CSV. Each subject contributes exactly one session. Subject names are taken from the release CSV and preserved in session order.

ii.
```python
subs=np.random.choice(np.unique(x.subject),10,replace=False)
by=x.groupby('subject').indices
eids=[str(x.iloc[by[s][0]].eid) for s in subs]
```

```python
subjects=[]
for x in sessions:
 if x['subject'] not in subjects: subjects.append(x['subject'])
```

iii. The AI reasoned that the reference code's example uses 10 sessions across 10 subjects, so it followed this pattern. Subjects are identified directly from the release CSV metadata.

## 1-c. How are the data split into sessions?

i. Each session is identified by an EID from the release CSV. The AI selects exactly 10 sessions (one per selected subject, always the first session listed for that subject). The `--full` flag processes all 10; `--sample` processes 2.

ii.
```python
n=2 if args.sample else 10; ...chosen=selected_eids(n)
```

iii. The AI treated sessions as the natural unit from the release. The choice of one session per subject comes from the `selected_eids` function which picks the first listed EID for each randomly selected subject.

## 1-d. How are the data split into trials?

i. Trials are rows of the session's trials table, loaded as a parquet file. Each row corresponds to one trial with associated timestamps and behavioral variables.

ii.
```python
trials=load_record(eid,g,srow,'trials.table')
```

iii. The trials table is already organized with one row per trial; no further splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) finite stimulus onset time, (2) valid choice in {-1, 1} (excluding no-go trials), (3) valid prior probability in {0.2, 0.5, 0.8}. Additionally, the trial window must be covered by both wheel and camera timestamps. Trials with non-finite interpolated wheel or whisker values are also dropped. No reaction time filter is applied.

ii.
```python
stim=np.asarray(trials.stimOn_times,float); choice=np.asarray(trials.choice,float); prior=np.asarray(trials.probabilityLeft,float)
valid=np.isfinite(stim)&np.isin(choice,[-1,1])&np.isin(np.round(prior,1),[.2,.5,.8])
valid &= (stim+OFF0>=wtime[0])&(stim+OFF1<=wtime[-1])&(stim+OFF0>=ct[0])&(stim+OFF1<=ct[-1])
```

```python
if not (np.isfinite(ww).all() and np.isfinite(mm).all()): continue
```

iii. The AI's CONVERSION_NOTES describe excluding trials with invalid labels, timing, or behavior coverage. No mention is made of reaction time filtering. The data paper's reaction time bounds (80 ms to 2 s) are noted in Step 3 of CONVERSION_NOTES but not applied in the code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times` (spike timestamps) and `spikes.clusters` (cluster IDs). Cluster metrics (`clusters.metrics`) provide quality labels, and channel brain location IDs provide anatomical information.

ii.
```python
st=np.asarray(load_record(eid,g,srow,base+'spikes.times',True),float)
sc=np.asarray(load_record(eid,g,srow,base+'spikes.clusters',True),int)
met=load_record(eid,g,srow,base+'clusters.metrics',True)
```

iii. The AI identifies spike times and cluster IDs as the primary neural data, consistent with the reference approach.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over a 2.1 s window (-0.6 to 1.5 s around stimulus onset), giving 105 bins. Raw spike counts are stored (not converted to firing rates). When a session has multiple probes, units from all probes are concatenated. The binning is done with a per-spike Python loop using a lookup table for good cluster indices.

ii.
```python
OFF0,OFF1,DT=-.6,1.5,.02
EDGES=np.arange(OFF0,OFF1+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

```python
M=np.zeros((len(good),len(CENTERS)),np.float32); lut={int(c):j for j,c in enumerate(good)}
bins=np.floor((rel-OFF0)/DT).astype(int)
for c,b in zip(cid,bins):
 j=lut.get(int(c));
 if j is not None and 0<=b<M.shape[1]: M[j,b]+=1
```

```python
neural.append(np.concatenate(mats))
```

iii. The AI justified storing raw spike counts (not rates) by stating "both reference papers/code define neural regressors by spike counting." The AI chose -0.6 to 1.5 s as the union of the methods paper's choice window (-0.5 to 1.5 s) and prior window (-0.6 to -0.1 s).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` from `clusters.metrics` are retained. Anatomical regions are mapped from channel atlas IDs using BrainRegions' raw Allen acronyms (not Beryl mapping). Clusters mapped to `void` (unknown atlas IDs) are NOT filtered out -- they are retained with 'void' as their region label.

ii.
```python
labels=np.asarray(met['label'],float); good=np.flatnonzero(labels>=1)
```

```python
BR=BrainRegions(); ID2AC={int(i):str(a) for i,a in zip(BR.id,BR.acronym)}
regs=[ID2AC.get(int(atlas[c]),'void') if 0<=c<len(atlas) else 'void' for c in ch[good]]
```

iii. The AI applies the `label >= 1` filter matching the reference code's `good_clusters` criterion. The CONVERSION_NOTES state "Filter to label>=1 good clusters." However, the code does not filter out void regions and uses Allen atlas acronyms rather than the coarser Beryl mapping.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are aligned to stimulus onset (`stimOn_times`). Spike times within the window [stimOn + OFF0, stimOn + OFF1] are selected, and the bin index is computed relative to OFF0 (-0.6 s).

ii.
```python
lo=np.searchsorted(st,stim[ti]+OFF0); hi=np.searchsorted(st,stim[ti]+OFF1)
rel=st[lo:hi]-stim[ti]
bins=np.floor((rel-OFF0)/DT).astype(int)
```

iii. The AI uses `stimOn_times` for alignment, consistent with the task instructions and reference paper.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms. The window spans -0.6 to 1.5 s around stimulus onset, producing 105 time bins. No rebinning or smoothing is applied.

ii.
```python
OFF0,OFF1,DT=-.6,1.5,.02
EDGES=np.arange(OFF0,OFF1+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

iii. The AI justifies 20 ms bins as matching the reference code and papers. The -0.6 s start was chosen to encompass both the choice window (-0.5 to 1.5 s) and prior window (-0.6 to -0.1 s) from the methods paper.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from the bin center times of the common temporal grid, defined by the window parameters and bin size. The bin centers are computed as `(EDGES[:-1]+EDGES[1:])/2`, spanning -0.59 to 1.49 s.

ii.
```python
CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

```python
ins.append(np.vstack([CENTERS,np.full(len(CENTERS),b)]).astype(np.float32))
```

iii. The time input is the bin center grid itself, which is identical for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing beyond computing the bin centers from the edge array. The same vector is broadcast to every trial.

ii.
```python
EDGES=np.arange(OFF0,OFF1+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

iii. The bin centers define the time axis; no additional transformation is needed.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input IS the neural binning grid -- the spike counts and the time input share the same bin centers, so they are aligned by construction.

ii.
```python
times=stim[ti]+CENTERS  # used for interpolating behavior
bins=np.floor((rel-OFF0)/DT).astype(int)  # used for spike counting
```

iii. Both neural and input data use the same `CENTERS` grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `trials.probabilityLeft`. A change in the prior value signals a new block, and the trial's position within its block is computed.

ii.
```python
pr=np.round(prior,1); blocknum=np.zeros(len(trials),np.float32)
for i in range(1,len(trials)): blocknum[i]=0 if pr[i]!=pr[i-1] else blocknum[i-1]+1
```

iii. The trials table carries no explicit block identifier, so blocks are recovered by detecting changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A sequential scan through all trials (before filtering) resets a counter to 0 whenever `probabilityLeft` changes and increments it otherwise. The counter gives the 0-based position of each trial in its block. After filtering, only the block numbers of retained trials are used.

ii.
```python
blocknum=np.zeros(len(trials),np.float32)
for i in range(1,len(trials)): blocknum[i]=0 if pr[i]!=pr[i-1] else blocknum[i-1]+1
```

```python
ins.append(np.vstack([CENTERS,np.full(len(CENTERS),b)]).astype(np.float32))
```

iii. Computing block numbers before filtering ensures that dropped trials still advance the count, preserving the animal's true position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From `trials.choice`, which takes values -1, 0, or 1 in the raw IBL data.

ii.
```python
choice=np.asarray(trials.choice,float)
valid=...&np.isin(choice,[-1,1])...
```

iii. The choice column directly encodes the animal's response.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps choice=-1 to 0 and choice=+1 to 1. No-go trials (choice=0) are excluded. The mapped value is broadcast across all time bins.

ii.
```python
cc=0 if c==-1 else 1
outs.append(np.vstack([np.full(len(CENTERS),cc),...]).astype(np.int64))
```

iii. The AI's CONVERSION_NOTES state the IBL convention as "Raw IBL choice is -1 (left), +1 (right), 0 (no-go)" and maps "-1->0 and +1->1". This differs from the reference code which uses `{1.0: 0, -1.0: 1}` with the comment "+1 is a leftward choice, -1 rightward."

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `trials.probabilityLeft`, which takes values 0.2, 0.5, or 0.8.

ii.
```python
prior=np.asarray(trials.probabilityLeft,float)
valid=...&np.isin(np.round(prior,1),[.2,.5,.8])
```

iii. The prior probability is directly available in the trials table.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are mapped: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2, matching the instructions exactly. The mapped value is broadcast across all time bins.

ii.
```python
pp={.2:0,.5:1,.8:2}[float(p)]
outs.append(np.vstack([...,np.full(len(CENTERS),pp),...]).astype(np.int64))
```

iii. The mapping follows the task instructions directly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.timestamps` and `_ibl_wheel.position`, loaded as raw arrays. These are processed into speed using the IBL reference wheel functions.

ii.
```python
wt=load_record(eid,g,srow,'wheel.timestamps'); wp=load_record(eid,g,srow,'wheel.position')
wpos,wtime=interpolate_position(wt,wp,freq=1000); wvel,_=velocity_filtered(wpos,fs=1000)
wspeed=np.abs(wvel)
```

iii. The AI uses the same IBL wheel processing functions (`interpolate_position`, `velocity_filtered`) as the reference code.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) The wheel position is interpolated to a regular 1000 Hz grid using `interpolate_position`. (2) A filtered velocity is computed using `velocity_filtered` (20 Hz Butterworth low-pass). (3) Speed is the absolute value of velocity. The continuous speed trace is then interpolated onto the trial's bin centers using `np.interp`, and discretized into 3 bins using pooled tertile thresholds computed across ALL sessions.

ii.
```python
wpos,wtime=interpolate_position(wt,wp,freq=1000); wvel,_=velocity_filtered(wpos,fs=1000)
wspeed=np.abs(wvel)
```

```python
ww=np.interp(times,wtime,wspeed)
```

```python
wheel=np.concatenate([np.concatenate(x['wheel']) for x in sessions])
wq=np.quantile(wheel,[1/3,2/3])
np.digitize(w,wq)
```

iii. The AI justified using pooled tertile thresholds (computed across all sessions) rather than per-session percentiles, arguing this "creates comparable dataset-wide low/medium/high categories."

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The 1/3 and 2/3 quantiles are computed from ALL wheel speed values across all sessions (pooled). These two thresholds are then applied to every trial using `np.digitize`, producing categories 0 (low), 1 (medium), 2 (high).

ii.
```python
wheel=np.concatenate([np.concatenate(x['wheel']) for x in sessions])
wq=np.quantile(wheel,[1/3,2/3])
np.digitize(w,wq)
```

iii. The pooled approach ensures global balance (~33/33/33% overall) but causes highly unbalanced distributions within individual sessions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed trace is interpolated to the same bin centers (`CENTERS`) as the neural data, using `np.interp` with stimulus-onset-aligned times.

ii.
```python
times=stim[ti]+CENTERS
ww=np.interp(times,wtime,wspeed)
```

iii. Both wheel and neural data use the same temporal grid, ensuring alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy` and `leftCamera.times` (with fallback to right camera if left is unavailable or has too few valid samples).

ii.
```python
for v in ('left','right'):
 try:
  ct=load_record(eid,g,srow,f'{v}Camera.times'); cm=load_record(eid,g,srow,f'{v}Camera.ROIMotionEnergy')
  n=min(len(ct),len(cm)); ct=np.asarray(ct[:n],float); cm=np.asarray(cm[:n],float)
  if n>10 and np.isfinite(cm).mean()>.95: side=v; break
```

iii. The AI prefers left camera and falls back to right, consistent with the reference code's approach.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is (no filtering or normalization). It is interpolated onto the trial bin centers using `np.interp`, then discretized into 3 bins using pooled tertile thresholds computed across all sessions.

ii.
```python
mm=np.interp(times,ct,cm)
```

```python
whisk=np.concatenate([np.concatenate(x['whisk']) for x in sessions])
mq=np.quantile(whisk,[1/3,2/3])
np.digitize(m,mq)
```

iii. Same pooled tertile approach as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The 1/3 and 2/3 quantiles are computed from ALL whisker motion energy values across all sessions (pooled). These thresholds are applied using `np.digitize`.

ii.
```python
whisk=np.concatenate([np.concatenate(x['whisk']) for x in sessions])
mq=np.quantile(whisk,[1/3,2/3])
np.digitize(m,mq)
```

iii. The verification output shows this causes highly unbalanced per-session distributions (e.g., one session has 97.4% low, 0.2% high).

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker motion energy is interpolated to the same bin centers as the neural data, ensuring temporal alignment.

ii.
```python
times=stim[ti]+CENTERS
mm=np.interp(times,ct,cm)
```

iii. Same alignment approach as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Trials with non-finite stimulus onset, invalid choice, or invalid prior are excluded. (2) Trials where wheel or camera timestamps don't span the full window are excluded. (3) Trials with non-finite interpolated values are skipped. (4) Sessions with fewer than 2 valid trials or no good neurons raise errors and are skipped. (5) Camera timestamps/motion energy arrays are truncated to the shorter length if mismatched. (6) Probes that fail to load are skipped.

ii.
```python
valid=np.isfinite(stim)&np.isin(choice,[-1,1])&np.isin(np.round(prior,1),[.2,.5,.8])
valid &= (stim+OFF0>=wtime[0])&(stim+OFF1<=wtime[-1])&(stim+OFF0>=ct[0])&(stim+OFF1<=ct[-1])
```

```python
if not (np.isfinite(ww).all() and np.isfinite(mm).all()): continue
```

```python
if len(kept)<2: raise ValueError('fewer than two valid trials')
```

iii. The AI's approach is to exclude problematic data rather than impute it, which is consistent with the reference approach.

## 10-a. What are the most time-consuming steps of the code?

i. The AI's most time-consuming steps are: (1) Downloading data from S3 (network I/O), which dominates for fresh runs. (2) The per-spike Python loop for binning spikes, which iterates over every spike individually. (3) Loading spike sorting data from disk. The AI notes cached processing takes 0.6-3.5 s per session.

ii.
```python
for c,b in zip(cid,bins):
 j=lut.get(int(c));
 if j is not None and 0<=b<M.shape[1]: M[j,b]+=1
```

iii. The AI's CONVERSION_NOTES note that cached runs complete quickly but initial downloads dominate runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The primary candidate is the per-spike Python loop used for spike binning. This loop iterates over every spike individually using Python's `for c,b in zip(cid,bins)` and a dictionary lookup, which is extremely slow compared to a vectorized approach. The reference code uses `np.bincount` with a flat index, which processes all spikes in one NumPy call.

ii.
```python
# AI's per-spike Python loop (slow):
lut={int(c):j for j,c in enumerate(good)}
bins=np.floor((rel-OFF0)/DT).astype(int)
for c,b in zip(cid,bins):
 j=lut.get(int(c));
 if j is not None and 0<=b<M.shape[1]: M[j,b]+=1
```

```python
# Reference's vectorized approach (fast):
flat_index = spike_unit * N_BINS + bin_index
counts[trial] = np.bincount(flat_index, minlength=n_units * N_BINS).reshape(n_units, N_BINS)
```

iii. The AI's CONVERSION_NOTES mention "Binary-search spike slicing avoids scanning full recordings for each trial" as a speedup, but don't acknowledge that the inner spike loop itself is a bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. No significant repeated processing is apparent. Each session is processed once. The pooled tertile computation reads all sessions' wheel/whisker data once after all sessions are loaded.

ii. N/A

iii. N/A

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI downloads and loads `clusters.channels` and `channels.brainLocationIds_ccf_2017` for anatomical mapping, which involves multiple fallback attempts. The fine-grained Allen atlas region labels are computed but the downstream decoder likely only uses the brain_region_idx for grouping, not the specific nomenclature level. Additionally, the cluster labels for all clusters are loaded even though only label>=1 clusters are retained.

ii.
```python
try: atlas=np.asarray(load_record(eid,g,srow,f'{probe}/channels.brainLocationIds_ccf_2017'))
except Exception:
 try: atlas=np.asarray(load_record(eid,g,srow,f'{probe}/pykilosort/channels.brainLocationIds_ccf_2017',True))
 except Exception: atlas=np.zeros(max(ch.max()+1,1),int)
```

iii. The multiple fallback attempts for atlas data are a form of defensive coding rather than necessary processing.
