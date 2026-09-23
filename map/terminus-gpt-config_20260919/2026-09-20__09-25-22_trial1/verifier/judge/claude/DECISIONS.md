# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by globbing for `*.nwb` files under `/app/data` using `pathlib.Path.rglob`, then iterates over each file, opening it with `h5py` (not `pynwb`) and processing it in `convert_session()`. Each NWB file contains spike times, trials, behavioral events, and tongue tracking for one session.

ii.
```python
DATA_ROOT = Path('/app/data')
files=sorted(DATA_ROOT.rglob('*.nwb'))
for i,p in enumerate(files):
    r=convert_session(p,args.show_processing and len(results)<2)
```

```python
def convert_session(path, make_plot=False):
    with h5py.File(path,'r') as f:
        good,valid=curate(f)
        ...
```

iii. The AI chose h5py over pynwb for direct HDF5 access, which is functionally equivalent for reading NWB data. The glob finds all 174 NWB files organized by subject directories.

## 1-b. How are the data split into subjects?

i. The AI extracts the subject ID from the filename using a regex `sub-([^_]+)`, capturing the numeric ID (e.g., `440956`). Unique subjects are collected and sorted to build the `subjects` list and `subject_idx` mapping.

ii.
```python
subject=re.search(r'sub-([^_]+)',path.name).group(1)
```

```python
subjects=sorted({r[4] for r in results}); smap={x:i for i,x in enumerate(subjects)}
```

iii. The subject ID is extracted from the filename rather than from `nwb.subject.subject_id` (as the reference does via pynwb). These are equivalent since the directory names encode the same subject IDs.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. The sorted file list determines session order. No splitting or grouping logic is needed.

ii.
```python
files=sorted(DATA_ROOT.rglob('*.nwb'))
```

iii. The NWB dataset stores one session per file, so file boundaries define session boundaries. This is the same approach as the reference.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. The AI reads go cue timestamps from `BehavioralEvents/go_start_times/timestamps` and uses the `curate()` function to determine which trials are valid.

ii.
```python
go_all=be['go_start_times/timestamps'][()]
```

iii. Each row in the trials table corresponds to one trial, with one go cue event per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI uses a multi-step filtering approach: (1) for every good unit, it maps `obs_intervals` to trial indices and intersects with `is_good_trials` (a per-unit boolean array), (2) takes the intersection across all retained units (a trial must be valid for ALL good units), (3) after computing neural data, removes trials where the entire neural population has zero spikes. There is no `free_water` filter. The reference uses `obs_intervals` from only the first good unit and filters `free_water == 0`.

ii.
```python
def curate(f):
    u=f['units']; good=np.where(decode(u['classification'][()]) == 'good')[0]
    ...
    valid=np.ones(len(go), bool)
    for ui in good:
        z=flat[a[ui]:b[ui]]; m=manual[ui]
        if len(z)!=len(m): raise ValueError('Validity/interval length mismatch')
        jj=map_observed_trials(trial_st,z)
        uv=np.zeros(len(go), bool)
        uv[jj]=m
        valid &= uv
    return good, valid
```

```python
has_neural=np.any(neural != 0, axis=(1,2))
inds=inds[has_neural]
```

iii. The AI's approach is more thorough than the reference: it checks validity across all good units rather than just one, and uses `is_good_trials` in addition to `obs_intervals`. The all-zero neural filter catches residual missing data. However, the AI does not filter `free_water` trials, instead relying on the all-zero filter to catch most of them (2,449 of 2,451 according to the reference's DECISIONS.md).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times` (the sorted spike times for each unit) accessed via h5py with the ragged array index `spike_times_index`.

ii.
```python
u=f['units']; flat=u['spike_times']; a,b=ragged_bounds(u['spike_times_index'][()])
for col,ui in enumerate(units):
    sp=np.asarray(flat[a[ui]:b[ui]])
```

iii. Same source variable as the reference.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50ms bins spanning -2.5s to +1.5s relative to the go cue. For each unit, the AI assigns each spike to a trial using `searchsorted` on trial start times, then computes the bin index within that trial. Spike counts are accumulated with `np.bincount` and divided by bin width (0.05s) to get firing rates in Hz. No smoothing or normalization is applied.

ii.
```python
starts=go_valid+OFF_START; ends=go_valid+OFF_END
for col,ui in enumerate(units):
    sp=np.asarray(flat[a[ui]:b[ui]])
    ti=np.searchsorted(starts, sp, side='right')-1
    ok=(ti>=0)
    ti=ti[ok]; ss=sp[ok]
    ok=ss < ends[ti]
    ti=ti[ok]; ss=ss[ok]
    bi=np.floor((ss-starts[ti])/BIN_S).astype(np.int64)
    ok=(bi>=0)&(bi<NT); code=ti[ok]*NT+bi[ok]
    if len(code): counts[:,col,:]=np.bincount(code,minlength=ntr*NT).reshape(ntr,NT)
return counts.astype(np.float32).transpose(0,1,2) / np.float32(BIN_S)
```

iii. The approach differs in implementation from the reference (which uses searchsorted on flattened edges) but produces equivalent spike count / bin width firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. No additional metric thresholds are applied. Sessions with no good units are dropped.

ii.
```python
good=np.where(decode(u['classification'][()]) == 'good')[0]
if len(good)==0: return good, np.zeros(len(f['intervals/trials/id']), bool)
```

iii. Same as reference. Uses the QC classifier verdict from `ChenLiuEtAl2023_SpikeSortingQC.pdf`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The bin edges are defined relative to the go cue. For each trial, the absolute window start is `go + OFF_START` (-2.5s) and end is `go + OFF_END` (+1.5s). Spikes are assigned to trial/bin combinations based on these absolute windows.

ii.
```python
starts=go_valid+OFF_START; ends=go_valid+OFF_END
```

iii. Same alignment event (go cue) as reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms bins, 80 bins total from -2.5s to +1.5s relative to go cue. No rebinning is applied since the data starts from raw spike times.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
EDGES_REL = np.arange(OFF_START, OFF_END + BIN_S/2, BIN_S, dtype=np.float64)
NT = len(CENTERS_REL)
assert NT == 80
```

iii. Matches the reference and task instructions exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `BehavioralEvents/sample_start_times/timestamps` (tone onset times) and `intervals/trials/start_time` (trial starts), together with go cue times. The AI finds the last sample onset before the go cue, bounded by trial start.

ii.
```python
sample=np.sort(f['acquisition/BehavioralEvents/sample_start_times/timestamps'][()])
for k,i in enumerate(trial_indices):
    lo=np.searchsorted(sample, starts[i]-1e-8, 'left')
    hi=np.searchsorted(sample, go[i]+1e-8, 'right')
    out[k]=sample[hi-1]
```

iii. Same source variable as reference (`sample_start_times`). The AI's approach is more explicit, bounding the search within each trial's start and go cue, while the reference takes the last sample before go without an explicit lower bound. Both handle the early-lick replay scenario.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset at each bin center is computed as `abs_centers - tone`, where `abs_centers = go + CENTERS_REL` (the absolute time of each bin center) and `tone` is the final sample onset.

ii.
```python
abs_centers=go[:,None]+CENTERS_REL[None,:]
tone=final_tone_onsets(f,inds,go_all)
inp=np.stack([abs_centers-tone[:,None], ...], axis=1)
```

iii. Equivalent to the reference's `CENTERS + (go - tone)`. Both compute time from tone onset at each bin center.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The bin centers used for the input are the same `CENTERS_REL` offsets from the go cue used for the neural bins, ensuring alignment.

ii.
```python
abs_centers=go[:,None]+CENTERS_REL[None,:]
```

iii. Same alignment approach as reference.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI uses `BehavioralEvents/photostim_start_times/timestamps` and `BehavioralEvents/photostim_stop_times/timestamps` (absolute event timestamps). The reference instead uses the trials table columns `photostim_onset` and `photostim_duration`.

ii.
```python
be=f['acquisition/BehavioralEvents']
starts=be['photostim_start_times/timestamps'][()]; stops=be['photostim_stop_times/timestamps'][()]
for a,b in zip(starts,stops): out |= ((abs_centers>=a)&(abs_centers<b))
```

iii. The AI uses the BehavioralEvents time series which stores absolute timestamps, while the reference uses the trials table which stores onset relative to trial start as strings. Both represent the same photostimulation periods but from different source fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series: for each bin center, the value is 1 if it falls within any `[photostim_start, photostim_stop)` interval, else 0. The AI iterates over all start/stop pairs globally (not per-trial).

ii.
```python
def stim_series(f, abs_centers):
    be=f['acquisition/BehavioralEvents']; out=np.zeros(abs_centers.shape, dtype=bool)
    starts=be['photostim_start_times/timestamps'][()]; stops=be['photostim_stop_times/timestamps'][()]
    for a,b in zip(starts,stops): out |= ((abs_centers>=a)&(abs_centers<b))
    return out.astype(np.float32)
```

iii. The approach is slightly different from the reference (which operates per-trial using onset+duration from the trials table), but produces the same binary result.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The `abs_centers` used for photostimulation comparison are the same absolute bin centers used for neural data, ensuring alignment.

ii.
```python
abs_centers=go[:,None]+CENTERS_REL[None,:]
```

iii. Same alignment as reference.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore) from the trials table, since actual choice is not stored directly.

ii.
```python
instruction=decode(trial['trial_instruction'][()])[inds]
outcome_s=decode(trial['outcome'][()])[inds]
for k,(ins,o) in enumerate(zip(instruction,outcome_s)):
    if o=='ignore': choice[k]=2
    elif o=='hit': choice[k]=0 if ins=='left' else 1
    elif o=='miss': choice[k]=1 if ins=='left' else 0
```

iii. Same derivation logic as reference.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as 0=left, 1=right, 2=no lick. Hit means licked the instructed side, miss means licked the opposite side, ignore means no lick. The per-trial value is repeated across all 80 bins.

ii.
```python
output=np.stack([np.repeat(choice[:,None],NT,1), ...], axis=1)
```

iii. Same coding and processing as reference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which stores `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome_s=decode(trial['outcome'][()])[inds]
omap={'ignore':0,'miss':1,'hit':2}; outcome=np.array([omap[x] for x in outcome_s],np.int64)
```

iii. Same as reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct categorical mapping: ignore=0, miss=1, hit=2. Repeated across all 80 bins.

ii.
```python
omap={'ignore':0,'miss':1,'hit':2}; outcome=np.array([omap[x] for x in outcome_s],np.int64)
output=np.stack([..., np.repeat(outcome[:,None],NT,1), ...], axis=1)
```

iii. Same as reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds `'no early'` and `'early'`.

ii.
```python
early_s=decode(trial['early_lick'][()])[inds]
early=np.array([1 if x=='early' else 0 for x in early_s],np.int64)
```

iii. Same as reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary mapping: `'early'` -> 1, everything else -> 0. Repeated across all 80 bins.

ii.
```python
early=np.array([1 if x=='early' else 0 for x in early_s],np.int64)
```

iii. Same as reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains `(tongue_x, tongue_y, tongue_likelihood)` with timestamps. Column 1 is y-position and column 2 is likelihood.

ii.
```python
ts=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
tt=ts['timestamps'][()]; d=ts['data'][()]; y=d[:,1]; likelihood=d[:,2]
```

iii. Same source as reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI uses linear interpolation (`np.interp`) to get tongue y and likelihood at each bin center. Frames with likelihood >= 0.9 are considered visible. Session-wide percentiles (40th/60th) are computed from raw visible frames (not bin means). Bin centers where interpolated likelihood >= 0.9 are discretized: 0 (< p40), 1 (p40 to p60 inclusive), 2 (> p60). Bins with interpolated likelihood < 0.9 are assigned class 3 (not visible).

The reference instead: (a) uses likelihood threshold 0.5, (b) computes bin means (not interpolation) of y-values within each 50ms bin, (c) takes percentiles of session-wide bin means (not raw frames), (d) assigns bins with no visible frames to class 3.

ii.
```python
visible=likelihood>=0.9
p40,p60=np.percentile(y[visible],[40,60])
yi=np.interp(flat,tt,y,left=np.nan,right=np.nan).reshape(abs_centers.shape)
li=np.interp(flat,tt,likelihood,left=0,right=0).reshape(abs_centers.shape)
cat=np.full(abs_centers.shape,3,dtype=np.int64)
v=(li>=0.9)&np.isfinite(yi)
cat[v & (yi<p40)]=0; cat[v & (yi>=p40) & (yi<=p60)]=1; cat[v & (yi>p60)]=2
```

iii. The AI uses interpolation at bin centers rather than binning/averaging frames within each bin. The likelihood threshold is 0.9 vs the reference's 0.5. The percentiles are computed from raw visible frames rather than from bin means.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-wide 40th and 60th percentiles of y-values from visible frames (likelihood >= 0.9) define the class boundaries. Class 0: y < p40, Class 1: p40 <= y <= p60, Class 2: y > p60, Class 3: not visible (interpolated likelihood < 0.9).

ii.
```python
p40,p60=np.percentile(y[visible],[40,60])
cat[v & (yi<p40)]=0; cat[v & (yi>=p40) & (yi<=p60)]=1; cat[v & (yi>p60)]=2
```

iii. The thresholds follow the instructions (40th/60th percentiles, per-session). The reference uses percentiles of bin means rather than raw frames, which is a subtle difference that affects the exact threshold values.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI interpolates tongue y and likelihood at the same absolute bin centers (`go + CENTERS_REL`) used for neural data, ensuring temporal alignment.

ii.
```python
abs_centers=go[:,None]+CENTERS_REL[None,:]
flat=abs_centers.ravel()
yi=np.interp(flat,tt,y,left=np.nan,right=np.nan).reshape(abs_centers.shape)
```

iii. Using the same bin centers for tongue interpolation and neural binning guarantees alignment. The reference instead bins camera frames into the same 50ms windows used for neural data, which is a different but also valid alignment approach.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Sessions where `classification` values are NaN (not strings) are handled by `decode()` converting non-bytes to strings, which won't match 'good', so the session is dropped. (2) Trials without valid neural data are filtered via the `curate()` function's obs_intervals/is_good_trials intersection, plus the all-zero neural population filter. (3) Tongue frames outside the recording period get NaN from `np.interp(..., left=np.nan, right=np.nan)` and are assigned class 3.

ii.
```python
def decode(a):
    return np.asarray([x.decode(errors='replace') if isinstance(x, (bytes, np.bytes_)) else str(x) for x in a])
```

```python
has_neural=np.any(neural != 0, axis=(1,2))
```

iii. The AI handles the same edge cases as the reference but uses different mechanisms (e.g., `is_good_trials` instead of `free_water` filter).

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file via h5py and loading the spike times buffer dominates. The full conversion takes ~247s for 174 sessions. Pickling the ~11GB result takes ~40s additional.

ii. N/A (timing info from CONVERSION_NOTES.md)

iii. I/O-bound performance, similar to the reference.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit spike binning loop iterates over each good unit. The per-trial tongue output could potentially be vectorized but is not a bottleneck.

ii.
```python
for col,ui in enumerate(units):
    sp=np.asarray(flat[a[ui]:b[ui]])
    ti=np.searchsorted(starts, sp, side='right')-1
    ...
```

iii. The per-unit loop is necessary due to ragged spike time arrays (different number of spikes per unit). Similar constraint exists in the reference.

## 10-c. What processing does the code repeat multiple times?

i. No significant repeated processing. Each NWB file is opened once and all quantities derived in a single pass.

ii. N/A

iii. Similar to reference - single pass over files.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `curate()` function iterates over ALL good units to compute the intersection of valid trials, which is more thorough than necessary. The reference only checks `obs_intervals` for the first good unit. The all-zero neural filter also requires computing neural data for trials that may be subsequently discarded.

ii.
```python
for ui in good:
    z=flat[a[ui]:b[ui]]; m=manual[ui]
    ...
    valid &= uv
```

iii. The per-unit intersection is more work than needed but ensures consistency. The neural computation for subsequently-removed trials is wasted work.
