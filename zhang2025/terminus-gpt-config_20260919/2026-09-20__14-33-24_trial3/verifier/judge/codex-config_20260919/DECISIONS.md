# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the authoritative 459-session release table `bwm_release.csv`, derives each session's local ALF path, and resolves trial, behavior, spike, cluster, and anatomy files directly from `/app/data/one_cache`. It chooses complete/current revisions explicitly rather than using ONE/SessionLoader for most streams.

ii. `rel=pd.read_csv(COHORT,index_col=0); sessions=rel.drop_duplicates('eid',keep='first')`

```python
def session_path(r):
    return ROOT / str(r.lab) / 'Subjects' / str(r.subject) / str(r.date) / f'{int(r.session_number):03d}'
```

iii. The agent said the cohort CSV exactly reproduced the paper's 459 sessions, 139 subjects, 699 probes, and 621,733 clusters. Explicit revision selection avoided partial legacy ALF objects in the mixed cache.

## 1-b. How are the data split into subjects?

i. Subject names come from the cohort's `subject` column. Only subjects with retained sessions are included, in first-retained-session order, and each session gets an integer index.

ii. `subjects=list(dict.fromkeys(subj)); subject_idx=np.array([subjects.index(x) for x in subj],dtype=np.int32)`

iii. The notes treat the cohort's subject identifier as authoritative and report 136 retained subjects after behavior/session filtering.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in the cohort; all probe rows sharing an EID are merged into that session. Sessions lacking required streams or two jointly valid trials are excluded.

ii. `sessions=rel.drop_duplicates('eid',keep='first')`

```python
probe_rows=rel[rel.eid.astype(str)==eid]
```

iii. The agent viewed EID as the native session unit and merging simultaneous probes as matching the methods pipeline.

## 1-d. How are the data split into trials?

i. A complete `_ibl_trials.table.pqt` is selected; each retained table row is one trial. Its stimulus onset defines a two-second window, and only trial indices jointly available from trial QC, wheel, and camera processing survive.

ii. `common=np.intersect1d(ci,wi,assume_unique=True)`

```python
np.savez(cache_file,trial_idx=common.astype(np.int32), ..., stim=stim[common])
```

iii. The notes state that trial rows are native trials and that intersecting masks fixes missing-stream handling while retaining at least two trials per session.

## 1-e. How are trials filtered based on quality controls?

i. The code requires finite stimulus, choice, feedback, prior, first movement, and feedback type; reaction time in [0.08, 2] s; nonzero choice; feedback minus go cue at most 10 s; and finite wheel/camera coverage across the trial window. Sessions with fewer than two joint trials are removed.

ii.
```python
rt=x.firstMovement_times.to_numpy()-x.stimOn_times.to_numpy()
m &= (rt >= .08) & (rt <= 2.)
m &= x.choice.to_numpy()!=0
m &= (x.feedback_times.to_numpy()-x.goCue_times.to_numpy() <= 10.)
```

iii. The agent attributed these rules to the methods-paper trial mask, adding explicit joint stream coverage to prevent undefined outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural matrices are derived from per-probe `spikes.times.npy` and `spikes.clusters.npy`. Cluster metrics identify the cluster universe; channel/anatomy arrays supply regions but not activity values.

ii.
```python
st=np.load(p['spikes.times.npy'],mmap_mode='r')
sc=np.load(p['spikes.clusters.npy'],mmap_mode='r')
```

iii. The agent identified spike timestamps and assignments as the activity sources, with cluster/channel files needed for indexing and anatomy.

## 2-b. How is the `neural` data processed?

i. All Kilosort clusters are retained, probes are concatenated within session, and spikes are counted in 100 half-open 20-ms bins from -0.5 to +1.5 s. Counts are stored losslessly as `uint8` (or `uint16` if needed), without division by bin width or smoothing.

ii.
```python
tb=np.floor((spike_t-beg)/BIN).astype(int)
flat=mapped[valid2]*NBIN+tb[valid2]
a=np.bincount(flat,minlength=n*NBIN).reshape(n,NBIN)
```

iii. The notes say this follows the methods executable's whole-session merged-probe spike-count representation and compact integers avoid roughly fourfold storage overhead.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not quality-filtered: every cluster represented by `clusters.metrics.pqt` is retained, regardless of `label`, `root`, or `void` status.

ii. `cluster_ids=metrics.cluster_id.to_numpy(dtype=int)`

iii. The agent deliberately prioritized the methods-paper code (`qc=None`, “all neurons”) over the data-paper's 75,708 label-1 well-isolated units, documenting the discrepancy.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute spike times are windowed relative to each retained `stimOn_times`; zero is visual stimulus onset and bins cover -0.5 through +1.5 s.

ii. `beg=s+OFF0; end=s+OFF1; ib=np.searchsorted(st,beg,'left'); ie=np.searchsorted(st,end,'left')`

iii. The agent said task-required stimulus alignment takes precedence over variable-specific prose and agrees with the executable reference parameters.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms. Spikes are binned directly at that resolution into 100 bins; no later temporal rebinning or smoothing is applied.

ii. `BIN = 0.02; OFF0, OFF1 = -0.5, 1.5; NBIN = 100`

iii. The agent linked 20-ms bins and the two-second window to the reference caching code and paper.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is defined from the fixed offsets around each trial's `stimOn_times`, using 100 bin-end offsets from -0.48 through +1.50 s.

ii. `REL_TIME = np.linspace(OFF0 + BIN, OFF1, NBIN, dtype=np.float32)`

iii. The agent said the executable behavior interpolation grid uses bin ends.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A fixed float32 linear grid is constructed once and copied as the first input row for every trial.

ii. `inp=[np.vstack((REL_TIME,np.full(NBIN,tib[j],np.float32))).astype(np.float32) for j in range(ntr)]`

iii. No raw transformation beyond constructing the chosen relative-time grid was claimed.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Both use the same stimulus onset and contain 100 samples, but the input labels bin ends (-0.48 to 1.50) whereas neural spike bins begin at -0.50. Thus each value represents the right edge of the corresponding neural bin.

ii. `REL_TIME = np.linspace(OFF0 + BIN, OFF1, NBIN, dtype=np.float32)`

iii. The notes describe this deliberate 20-ms convention as matching reference behavior interpolation.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from transitions in the full, unfiltered `trials.probabilityLeft` sequence.

ii. `tib=trial_in_block(probs)`

iii. The agent reasoned that the trials table lacks a block ID and prior transitions recover block boundaries.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A change in prior resets the counter, which is one-based. It is calculated before filtering and broadcast across all 100 timepoints.

ii.
```python
change=np.r_[True, prob[1:] != prob[:-1]]
starts=np.maximum.accumulate(np.where(change,np.arange(len(prob)),0))
return (np.arange(len(prob))-starts+1).astype(np.float32)
```

iii. Pre-filter calculation preserves the animal's actual position despite dropped trials; the agent explicitly chose first trial = 1.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from the trials table's `choice` field.

ii. `choices=x.choice.to_numpy(dtype=float)`

iii. The agent treated native choice as the authoritative source and excluded zero/no-response trials.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Native -1 is mapped to class 0 (“left”) and +1 to class 1 (“right”), then broadcast over 100 bins.

ii. `cmap={-1:0,1:1}`

```python
np.full(NBIN,c,np.uint8)
```

iii. The notes assert that native IBL choice -1 means left and +1 means right, and follow that interpretation plus the task's class labels.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft`.

ii. `probs=x.probabilityLeft.to_numpy(dtype=float)`

iii. The agent notes these are protocol-defined block priors.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are mapped to 0, 1, and 2 and broadcast over time. A nearest-key lookup only tolerates float representation error and rejects discrepancies above 1e-6.

ii. `pmap={0.2:0,0.5:1,0.8:2}`

iii. The mapping is prescribed directly by the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii. `t=np.asarray(np.load(tp),dtype=float); p=np.asarray(np.load(pp),dtype=float)`

iii. The agent followed Brainbox's standard position-to-velocity path.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated to 1 kHz, filtered/differentiated by `velocity_filtered`, converted to absolute velocity, and linearly interpolated/extrapolated to trial bin-end times.

ii.
```python
pos,ti=interpolate_position(t,p,freq=1000)
vel,_=velocity_filtered(pos,1000)
return np.asarray(ti),np.abs(np.asarray(vel))
```

iii. The notes say this exactly follows the Brainbox/reference wheel-speed definition.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Two global empirical tertiles are calculated over every retained session/trial/time sample. `searchsorted(..., side='right')` assigns low/medium/high (0/1/2).

ii. `q=np.quantile(x,[1/3,2/3])`

```python
wc=np.searchsorted(wthr,z['wheel'],side='right').astype(np.uint8)
```

iii. The agent preferred global thresholds so class meanings were common across sessions and described them as training-independent; collapsed tertiles cause an error.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. For each stimulus onset, wheel speed is interpolated at absolute times `stimOn + REL_TIME`, corresponding to neural-bin right edges.

ii. `grid=stim[i]+REL_TIME.astype(float)`

iii. The agent said this matches the executable reference's bin-end sampling convention.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses matched `<side>Camera.ROIMotionEnergy.npy` and `_ibl_<side>Camera.times.npy`, preferring left camera and falling back to right.

ii. `for sd in ('left','right'):`

```python
tp,vp=paired_stream(alf,sd)
```

iii. The agent followed the reference's side-camera preference and required matched lengths/revisions.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy values are sorted, duplicate timestamps removed, checked for finite/window coverage, and linearly interpolated/extrapolated at bin-end times; no filtering or normalization is applied.

ii. `y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)`

iii. The agent stated the released metric should otherwise be used unchanged.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, it uses global 1/3 and 2/3 quantiles and right-sided thresholding into 0/1/2.

ii. `mthr=thresholds(mz)`

```python
mc=np.searchsorted(mthr,z['whisker'],side='right').astype(np.uint8)
```

iii. The agent chose globally consistent physical classes rather than session-relative classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Motion energy is evaluated at `stimOn + REL_TIME`, so every categorical sample corresponds to the right edge of a neural bin.

ii. `grid=stim[i]+REL_TIME.astype(float)`

iii. The notes say stimulus alignment and the common 100-point bin-end grid match the reference executable.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code searches candidate revisions, validates paired arrays, removes duplicate timestamps, filters nonfinite/incompletely covered trials, catches per-session failures, and excludes sessions with fewer than two valid trials. Unexpected spike cluster IDs are discarded and structural assertions run before saving.

ii.
```python
try:
    info=preprocess_behavior(...)
except Exception as e: print(f'EXCLUDE {eid}: {type(e).__name__}: {e}',flush=True)
```

iii. The agent documented mixed-revision defects and an erroneous reference mask combination, choosing explicit validation and safe exclusion rather than fabricating missing values.

## 10-a. What are the most time-consuming steps of the code?

i. Loading/memory-mapping large spike arrays, binning every retained spike across trials, constructing the very large all-cluster neural payload, and serializing it dominate; wheel preprocessing is secondary.

ii. `m,rg,mx=bin_probe(pr,stim)`

```python
pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes estimate ~26.5 billion neural elements and identify spike processing plus large-pickle I/O as intrinsic costs.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial behavioral interpolation, per-trial spike binning, per-trial probe concatenation/input construction, output assembly, and repeated region-index lookup could be further vectorized. The expensive work inside each spike trial already uses `bincount`.

ii. `for i in idx:` and `for s in stim:`

iii. The agent retained Python trial loops because event windows vary, while using binary search and vectorized counting within each loop.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly opens each behavior cache, constructs interpolation objects per trial, creates identical time rows and constant output rows per trial, concatenates probe matrices per trial, instantiates `BrainRegions` per probe, and performs list-based subject/region indexing.

ii. `z=np.load(cf)` and `br=BrainRegions()`

iii. The notes emphasize that a two-pass cache avoids repeating the substantially costlier wheel filtering/interpolation between threshold estimation and neural conversion.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads cluster `label` values while retaining all clusters; saves `trial_idx` and `trial_path` in temporary caches but does not use them during final assembly; collects `logs` that are not consumed; and computes plot-only raw traces when plots are requested. Constant trial outputs and time rows are redundantly materialized at every timepoint, although required by the chosen rectangular format.

ii. `logs=[]` and `np.savez(cache_file,trial_idx=..., ..., trial_path=...)`

iii. The agent did not explicitly flag these as discarded; it justified the larger repeated arrays as compatibility with the decoder format and compact integer storage.
