# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading the Zhang cohort CSV (`bwm_release.csv`) to enumerate all 459 sessions and their probe insertions, then resolves file paths directly by constructing ALF directory paths from the lab, subject, date, and session number columns. Trial tables, wheel arrays, camera streams, and spike sorting files are loaded by globbing for the relevant filenames under each session's ALF directory. The AI does NOT use the ONE API; instead it manually navigates the file hierarchy.

ii.
```python
COHORT = Path('/app/code/code_zhang2025/data/bwm_release.csv')
rel=pd.read_csv(COHORT,index_col=0); sessions=rel.drop_duplicates('eid',keep='first')
```
```python
def session_path(r):
    return ROOT / str(r.lab) / 'Subjects' / str(r.subject) / str(r.date) / f'{int(r.session_number):03d}'
```

iii. The AI chose to resolve files directly because the mixed ONE cache had revision ambiguities (generic SessionLoader selected partial legacy objects). Using the cohort CSV and explicit path resolution avoids these issues while ensuring all 459 paper sessions are addressed.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column of `bwm_release.csv`. At assembly, a unique ordered list of subjects is created preserving first-appearance order, and `subject_idx` maps each session to its position in that list.

ii.
```python
subj.append(str(r.subject))
...
subjects=list(dict.fromkeys(subj)); subject_idx=np.array([subjects.index(x) for x in subj],dtype=np.int32)
```

iii. The subject identity comes directly from the cohort CSV. No parsing of paths or filenames is needed.

## 1-c. How are the data split into sessions?

i. Each row in `bwm_release.csv` (after deduplication by eid) corresponds to one session. Sessions are processed one at a time in a loop.

ii.
```python
sessions=rel.drop_duplicates('eid',keep='first')
for k,(_,r) in enumerate(sessions.iterrows(),1):
    eid=str(r.eid)
```

iii. The cohort CSV already lists sessions individually, so no splitting is required.

## 1-d. How are the data split into trials?

i. The trial table (`_ibl_trials.table.pqt`) has one row per trial. The AI loads the full trial table and applies a mask to select valid trials.

ii.
```python
def full_trial_table(alf: Path):
    for p in alf.glob('**/_ibl_trials.table.pqt'):
        x=pd.read_parquet(p)
        if REQ_TRIAL.issubset(x.columns): candidates.append((len(x),str(p),p,x))
```

iii. The trials table is already one row per trial. The AI resolves revision ambiguities by preferring the table with the most rows and latest revision.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters: (1) required event columns must be non-NaN (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), (2) reaction time (firstMovement_times - stimOn_times) must be 0.08-2.0 s, (3) choice must be non-zero, (4) feedback_times - goCue_times must be <= 10 s. Additionally, trials must have complete finite wheel and camera coverage over the [-0.5, +1.5] window as checked during behavior interpolation.

ii.
```python
def trial_mask(x):
    m=np.ones(len(x),bool)
    for c in ['stimOn_times','choice','feedback_times','probabilityLeft','firstMovement_times','feedbackType']:
        m &= x[c].notna().to_numpy()
    rt=x.firstMovement_times.to_numpy()-x.stimOn_times.to_numpy()
    m &= (rt >= .08) & (rt <= 2.)
    m &= x.choice.to_numpy()!=0
    m &= (x.feedback_times.to_numpy()-x.goCue_times.to_numpy() <= 10.)
    return m
```

iii. These checks match the Zhang reference code's `load_trials_and_mask` function, which includes the RT bounds, non-NaN field requirements, no-choice exclusion, and the 10-second feedback-goCue duration limit.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` per probe, plus `clusters.metrics.pqt` for cluster identification, `clusters.channels.npy` for channel assignments, and `channels.brainLocationIds_ccf_2017.npy` for anatomical locations.

ii.
```python
st=np.load(p['spikes.times.npy'],mmap_mode='r')
sc=np.load(p['spikes.clusters.npy'],mmap_mode='r')
```

iii. Spike times and cluster assignments are the fundamental data for constructing neural activity matrices.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 100 half-open 20 ms bins spanning the trial window [-0.5, +1.5] s around stimulus onset. Counts are stored as raw integer spike counts (uint8 or uint16 depending on maximum count), NOT converted to firing rates. When a session has multiple probes, their clusters are concatenated to form a merged population.

ii.
```python
tb=np.floor((spike_t-beg)/BIN).astype(int)
valid2=(tb>=0)&(tb<NBIN); flat=mapped[valid2]*NBIN+tb[valid2]
a=np.bincount(flat,minlength=n*NBIN).reshape(n,NBIN)
mats.append(a)
```
```python
if maxcount>255: dtype=np.uint16
else: dtype=np.uint8
neural=[np.concatenate([pmats[p][j] for p in range(len(pmats))],axis=0).astype(dtype,copy=False) for j in range(ntr)]
```

iii. The AI stores raw spike counts rather than firing rates to save space (~26.5 GB vs ~106 GB). The decoder converts to float32 during training. Probe merging follows the reference's `merge_probes` logic.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI retains ALL Kilosort clusters (no quality filtering). It uses `clusters.metrics.pqt` to enumerate cluster IDs but does not filter by the `label` column. All clusters with valid IDs are included, regardless of their sorting quality score.

ii.
```python
def resolve_probe(row):
    ...
    for p in base.glob('**/clusters.metrics.pqt'):
        x=pd.read_parquet(p)
        if 'cluster_id' in x and 'label' in x: candidates.append((len(x),str(p),p,x))
    ...
    cluster_ids=metrics.cluster_id.to_numpy(dtype=int)
    # No label filtering applied
```

iii. The AI justified this by noting that the Zhang reference code's `prepare_data` calls `load_spiking_data` with `qc=None`, retaining all clusters, and the methods paper states "all neurons sorted by Kilosort 2.5." The AI explicitly chose not to apply the data paper's QC label >= 1 filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are extracted relative to stimulus onset. The window spans from stimOn_times - 0.5 s to stimOn_times + 1.5 s. Binary search (`searchsorted`) identifies the spike indices within this absolute time window, then bins are computed relative to the window start.

ii.
```python
beg=s+OFF0; end=s+OFF1
ib=np.searchsorted(st,beg,'left'); ie=np.searchsorted(st,end,'left')
spike_t=np.asarray(st[ib:ie])
tb=np.floor((spike_t-beg)/BIN).astype(int)
```

iii. All streams share a common session clock, so alignment is achieved by subtracting the stimulus onset time. This matches the reference approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial, spanning 2 s from -0.5 to +1.5 s around stimulus onset. No rebinning is applied.

ii.
```python
BIN = 0.02
OFF0, OFF1 = -0.5, 1.5
NBIN = 100
```

iii. This matches the reference code's `binsize=0.02` and `time_window=(-0.5, 1.5)`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. This is a synthetic variable derived from the bin timing grid. It represents the time at each bin relative to stimulus onset.

ii.
```python
REL_TIME = np.linspace(OFF0 + BIN, OFF1, NBIN, dtype=np.float32)
```

iii. The time input is defined by the binning parameters, not from a raw data variable.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes bin-end times using `linspace(-0.48, 1.50, 100)`, which gives 100 evenly spaced values from -0.48 to 1.50 s. These are bin ends, not bin centers.

ii.
```python
REL_TIME = np.linspace(OFF0 + BIN, OFF1, NBIN, dtype=np.float32)
# = np.linspace(-0.48, 1.50, 100) -> [-0.48, -0.46, ..., 1.50]
```

iii. The AI followed the Zhang reference code's convention of using bin-end times (`linspace(interval_start + binsize, interval_end, n_bins)`), which differs from the human reference that uses bin centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same `REL_TIME` array is used both as the time input values and as the interpolation grid for behavioral variables. The spike bins cover the same overall window, so the time input is inherently aligned with the neural data (though the time values represent bin ends while spike bins are half-open intervals starting at the bin's left edge).

ii.
```python
inp=[np.vstack((REL_TIME,np.full(NBIN,tib[j],np.float32))).astype(np.float32) for j in range(ntr)]
```

iii. The behavior and time input share the same 100-element time grid, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. Block boundaries are detected where `probabilityLeft` changes value.

ii.
```python
def trial_in_block(prob):
    prob=np.asarray(prob)
    change=np.r_[True, prob[1:] != prob[:-1]]
    starts=np.maximum.accumulate(np.where(change,np.arange(len(prob)),0))
    return (np.arange(len(prob))-starts+1).astype(np.float32)
```

iii. The trials table has no explicit block identifier, so blocks are inferred from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Block boundaries are detected where `probabilityLeft` changes. The trial number within each block is computed starting from 1 (the first trial in a block is 1). This is computed on the full unfiltered trial sequence before any quality filtering, so excluded trials still count toward the position. The value is broadcast over all 100 time bins.

ii.
```python
change=np.r_[True, prob[1:] != prob[:-1]]
starts=np.maximum.accumulate(np.where(change,np.arange(len(prob)),0))
return (np.arange(len(prob))-starts+1).astype(np.float32)
```
```python
tib=trial_in_block(probs)
...
inp=[np.vstack((REL_TIME,np.full(NBIN,tib[j],np.float32))).astype(np.float32) for j in range(ntr)]
```

iii. Computing block position on the unfiltered sequence preserves the animal's actual experimental position. Starting from 1 (rather than 0) is a minor convention choice.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From `choice` in the trials table, which takes values -1 (rightward), 0 (no response), and +1 (leftward). Trials with choice 0 are excluded.

ii.
```python
choices=x.choice.to_numpy(dtype=float)
...
cmap={-1:0,1:1}
c=cmap[int(z['choice'][j])]
```

iii. The choice column directly encodes the animal's decision.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps choice -1 (rightward) to 0 and choice +1 (leftward) to 1. The value is broadcast over all 100 time bins. The output_values labels are `['left', 'right']`.

ii.
```python
cmap={-1:0,1:1}; pmap={0.2:0,0.5:1,0.8:2}
...
c=cmap[int(z['choice'][j])]
out.append(np.vstack((np.full(NBIN,c,np.uint8),...)))
```

iii. The AI's CONVERSION_NOTES (Step 4) state "Map -1 to 0 and +1 to 1; exclude 0." However, in IBL convention, +1 = left and -1 = right, so this mapping produces left=1, right=0, which contradicts the instructions specifying "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
pmap={0.2:0,0.5:1,0.8:2}
pv=float(z['prior'][j]); key=min(pmap,key=lambda q:abs(q-pv))
```

iii. The three values are the block prior probabilities used in the task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps 0.2 to 0, 0.5 to 1, and 0.8 to 2, matching the instructions. The mapping includes a tolerance check (`abs(key-pv)>1e-6`) for floating-point comparison. The value is broadcast over all 100 time bins.

ii.
```python
pmap={0.2:0,0.5:1,0.8:2}
...
key=min(pmap,key=lambda q:abs(q-pv))
if abs(key-pv)>1e-6: raise ValueError(f'unexpected prior {pv}')
out.append(np.vstack((...,np.full(NBIN,pmap[key],np.uint8),...)))
```

iii. The mapping exactly matches the task specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`. The AI loads these raw arrays directly and derives velocity using Brainbox functions.

ii.
```python
def derive_wheel(alf):
    tp=newest(alf.glob('**/_ibl_wheel.timestamps.npy'))
    pp=newest(alf.glob('**/_ibl_wheel.position.npy'))
    t=np.asarray(np.load(tp),dtype=float); p=np.asarray(np.load(pp),dtype=float)
    pos,ti=interpolate_position(t,p,freq=1000)
    vel,_=velocity_filtered(pos,1000)
    return np.asarray(ti),np.abs(np.asarray(vel))
```

iii. Uses the same Brainbox functions (`interpolate_position`, `velocity_filtered`) as `SessionLoader.load_wheel()` internally.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) The raw wheel position is interpolated to 1 kHz using `interpolate_position`, (2) velocity is computed with a 20 Hz Butterworth low-pass filter using `velocity_filtered`, (3) speed is the absolute value of velocity. The speed trace is then linearly interpolated onto the 100 bin-end time points for each trial using `interp1d` with extrapolation. Finally, the continuous speed is discretized into 3 classes using global tertile thresholds.

ii.
```python
pos,ti=interpolate_position(t,p,freq=1000)
vel,_=velocity_filtered(pos,1000)
return np.asarray(ti),np.abs(np.asarray(vel))
```
```python
y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)
```

iii. The wheel processing uses the same Brainbox pipeline as the reference. The interpolation to bin ends (rather than bin centers) follows the Zhang code convention.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI uses GLOBAL empirical tertiles computed across ALL retained sessions' wheel speed values. The 1/3 and 2/3 quantiles are computed once from the concatenated wheel speed data of all sessions, then `np.searchsorted` with `side='right'` maps each value to class 0, 1, or 2.

ii.
```python
def thresholds(values):
    x=np.concatenate([v.reshape(-1) for v in values])
    q=np.quantile(x,[1/3,2/3])
    return q.astype(float)
```
```python
wz=[np.load(cf)['wheel'] for _,_,cf in kept]
wthr=thresholds(wz)
wc=np.searchsorted(wthr,z['wheel'],side='right').astype(np.uint8)
```

iii. The AI chose global thresholds so that the physical meaning of each class is consistent across sessions. This contrasts with per-session percentiles which would ensure equal class sizes within each session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed trace is interpolated at the same 100 bin-end times (`REL_TIME`) as the time input, relative to stimulus onset. This ensures the wheel data shares the same temporal grid as the neural data.

ii.
```python
grid=stim[i]+REL_TIME.astype(float)
y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)
```

iii. The wheel trace and neural spikes share a common session clock, so interpolating at the bin-end times around stimulus onset aligns them.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `<side>Camera.ROIMotionEnergy.npy` and `_ibl_<side>Camera.times.npy`, preferring the left camera and falling back to right.

ii.
```python
for sd in ('left','right'):
    tp,vp=paired_stream(alf,sd)
    if tp is not None: side=sd; break
```
```python
def paired_stream(alf: Path, side: str):
    times=list(alf.glob(f'**/_ibl_{side}Camera.times.npy'))
    vals=list(alf.glob(f'**/{side}Camera.ROIMotionEnergy.npy'))
```

iii. The left camera preference matches the reference code. The AI additionally validates that timestamps and values have matching lengths.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is (no filtering or normalization). It is linearly interpolated onto the 100 bin-end time points for each trial. Then it is discretized into 3 classes using global tertile thresholds, same as wheel speed.

ii.
```python
cam,ci=interp_trials(ct,cv,stim,base)
...
mc=np.searchsorted(mthr,z['whisker'],side='right').astype(np.uint8)
```

iii. No additional processing beyond interpolation and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same method as wheel speed: global empirical tertiles across all retained sessions, applied with `np.searchsorted`.

ii.
```python
mz=[np.load(cf)['whisker'] for _,_,cf in kept]
mthr=thresholds(mz)
mc=np.searchsorted(mthr,z['whisker'],side='right').astype(np.uint8)
```

iii. Global thresholds ensure consistent physical meaning across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated at the same 100 bin-end times relative to stimulus onset.

ii.
```python
grid=stim[i]+REL_TIME.astype(float)
y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)
```

iii. The camera frame times are on the same session clock as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling: (1) The trial mask excludes trials with NaN in required fields. (2) The behavior interpolation function (`interp_trials`) checks for sufficient temporal coverage and finite values, dropping trials that fail. (3) Only the intersection of trials valid for both wheel and camera is kept. (4) Sessions with fewer than 2 jointly valid trials are excluded. (5) Sessions missing camera streams entirely are excluded.

ii.
```python
common=np.intersect1d(ci,wi,assume_unique=True)
if len(common)<2: raise ValueError(f'only {len(common)} jointly valid trials')
```
```python
if not np.isfinite(vy).all(): continue
if np.isfinite(y).all(): out.append(y.astype(np.float32)); kept.append(i)
```

iii. The AI explicitly intersects all stream validity masks, which it notes fixes a bug in the reference code where Python's `and` on lists doesn't properly combine masks.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies spike binning and behavior preprocessing (wheel filtering/interpolation) as the main costs. The two-pass design separates behavior caching from spike processing to avoid redundant computation.

ii.
```python
# Behavior pass
info=preprocess_behavior(r,eid,cf)
# Spike pass
m,rg,mx=bin_probe(pr,stim)
```

iii. From timing estimates in CONVERSION_NOTES: behavior alignment ~0.27 s/session, spike binning ~0.55 s/session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `interp_trials` iterates over each trial for behavior interpolation. The per-trial loop in `bin_probe` iterates over each trial for spike binning. Both could potentially be vectorized, though the AI uses vectorized `bincount` within each trial iteration.

ii.
```python
for i in idx:
    beg,end=stim[i]+OFF0,stim[i]+OFF1
    ...
```
```python
for s in stim:
    beg=s+OFF0; end=s+OFF1
    ...
```

iii. The AI noted these loops but kept them for clarity since each trial has a different time offset.

## 10-c. What processing does the code repeat multiple times?

i. The behavior data is processed in a first pass (caching wheel/whisker interpolation to .npz files) and then loaded again in a second pass for spike binning. The cached .npz files are loaded twice: once for computing global thresholds and once for final assembly. The wheel Brainbox processing (1 kHz interpolation + velocity filtering) is done once per session and cached.

ii.
```python
# First pass: cache behavior
info=preprocess_behavior(r,eid,cf)
# Load cached behavior for thresholds
wz=[np.load(cf)['wheel'] for _,_,cf in kept]
# Load cached behavior again for assembly
z=np.load(cf)
```

iii. The two-pass design is intentional to compute global thresholds before final discretization. The repeated loads of .npz files could be avoided by keeping data in memory.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI stores neural data as integer spike counts (uint8/uint16) rather than firing rates. The downstream decoder converts these to float32, so the compact representation is not wasted. The behavior caching to disk (.npz files) is intermediate and could be kept in memory instead. The AI also stores duplicate timestamp arrays in the cache that are not needed for the final output.

ii.
```python
np.savez(cache_file,trial_idx=common.astype(np.int32),wheel=wheel,whisker=cam,
         choice=choices[common].astype(np.int8),prior=probs[common].astype(np.float32),
         trial_in_block=tib[common],stim=stim[common],side=np.array(side),trial_path=np.array(str(trial_p)))
```

iii. The caching design trades disk I/O for the ability to compute global thresholds in a separate pass.
