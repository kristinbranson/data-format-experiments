# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files by recursively globbing `*.nwb` under `/app/data`, sorts them, and processes each with `pynwb.NWBHDF5IO`. Each file is opened once; trials, units, events, and behavioral time series are extracted within a single `with` block.

ii.
```python
files=sorted(Path('/app/data').rglob('*.nwb'))
...
for p in files:
    st=time.time(); res=convert_session(p,make_plot=args.show_processing and len(sessions)<2)
```

```python
with NWBHDF5IO(str(path),'r',load_namespaces=True) as io:
    nwb=io.read(); tr=nwb.trials
    ev=nwb.acquisition['BehavioralEvents'].time_series
    go=np.asarray(ev['go_start_times'].timestamps[:],dtype=np.float64)
    sample=np.asarray(ev['sample_start_times'].timestamps[:],dtype=np.float64)
```

iii. The AI's CONVERSION_NOTES.md documents 174 NWB files found, matching the dataset. Using `pynwb` is required by the instructions. The recursive glob (`rglob`) finds all NWB files regardless of directory depth, which is functionally equivalent to the reference's `glob.glob('sub-*/*.nwb')`.

## 1-b. How are the data split into subjects?

i. Each NWB file's `nwb.subject.subject_id` is read and stored. At assembly, unique sorted subject IDs form the `subjects` list and per-session indices are computed.

ii.
```python
return dict(..., subject=info['subject'], ...)
# where info['subject'] = str(nwb.subject.subject_id)
```

```python
subjects=sorted({x['subject'] for x in sessions})
smap={v:i for i,v in enumerate(subjects)}
'subject_idx':np.asarray([smap[x['subject']] for x in sessions],dtype=np.int64)
```

iii. Same approach as the reference. Subject IDs are numeric strings from the NWB files. The AI reports 28 subjects matching the dataset.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. The AI iterates over files in sorted order, each producing one session in the output. Sessions with insufficient valid units/trials are skipped.

ii.
```python
files=sorted(Path('/app/data').rglob('*.nwb'))
...
for p in files:
    res=convert_session(p, ...)
    if res is None: print(f'SKIP {p.name}: insufficient valid neurons/trials', ...); continue
    sessions.append(res)
```

iii. The one-file-per-session structure of the NWB dataset makes this straightforward. The AI identifies `nwb.identifier` for session identification and reports 173 sessions retained (1 dropped for zero good units).

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table. The AI takes the minimum of trial count and go-cue count, then filters to "structurally valid" trials (those with a preceding sample onset). Further filtering via `select_units_and_trials` uses `obs_intervals` and `is_good_trials`.

ii.
```python
go=np.asarray(ev['go_start_times'].timestamps[:],dtype=np.float64)
sample=np.asarray(ev['sample_start_times'].timestamps[:],dtype=np.float64)
n=min(len(tr),len(go))
ix=np.searchsorted(sample,go[:n],side='right')-1
structural_idx=np.flatnonzero(ix>=0)
if structural_idx.size<2: return None
units,trial_idx=select_units_and_trials(nwb,structural_idx)
```

iii. The AI documents that trials are paired with go cues and require a preceding sample onset. This is more conservative than the reference which asserts `len(go) == len(trials)` and doesn't filter based on preceding sample onsets.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three filters: (1) structural validity — trials must have a preceding sample onset; (2) observation validity — trials must be in the intersection of `is_good_trials` for all selected units, mapped via `obs_intervals`; (3) zero-neural removal — trials where all units have zero spikes in the window are dropped. Notably, the AI does NOT filter `free_water` trials.

ii.
```python
# Structural validity
ix=np.searchsorted(sample,go[:n],side='right')-1
structural_idx=np.flatnonzero(ix>=0)

# Observation validity via is_good_trials
units,trial_idx=select_units_and_trials(nwb,structural_idx)

# Zero-neural removal
neural_valid=np.any(rates>0,axis=(1,2))
dropped_zero_neural=int((~neural_valid).sum())
trial_idx=trial_idx[neural_valid]
```

iii. The AI's CONVERSION_NOTES document the `is_good_trials` mapping approach extensively. The AI chose to use `is_good_trials` as a per-unit validity mask and intersect valid trials across all units, rather than using `obs_intervals` start times matched to trial start times as the reference does. The AI also does not filter `free_water` trials, instead relying on the zero-neural filter to catch most of them. The AI reports 90,378 retained trials vs the reference's 90,860.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']` for each classifier-good unit, with go-cue times from `BehavioralEvents/go_start_times` used to define bin edges.

ii.
```python
spikes=[np.asarray(nwb.units['spike_times'][j],dtype=np.float64) for j in units]
edge_matrix=go[:,None]+EDGES_REL[None,:]
rates=np.empty((len(go),len(units),N_TIME),dtype=np.float32)
for ui,sp in enumerate(spikes):
    rates[:,ui,:]=np.diff(np.searchsorted(sp,edge_matrix,side='left'),axis=1).astype(np.float32)/BIN_S
```

iii. Same source variables as the reference. The AI reads individual unit spike times rather than the bulk buffer approach the reference uses.

## 2-b. How is the `neural` data processed?

i. Spike times are histogrammed into 50-ms non-overlapping bins spanning -2.5 to +1.5 s relative to go cue. Counts are divided by bin width (0.05 s) to get firing rates in Hz. No smoothing or normalization is applied.

ii.
```python
EDGES_REL=np.arange(OFF_START, OFF_END + BIN_S/2, BIN_S, dtype=np.float64)
...
edge_matrix=go[:,None]+EDGES_REL[None,:]
rates[:,ui,:]=np.diff(np.searchsorted(sp,edge_matrix,side='left'),axis=1).astype(np.float32)/BIN_S
```

iii. This matches the reference approach: searchsorted-based binning, count divided by bin width for Hz. The edge construction uses `np.arange` with a half-bin offset stop, which should produce 81 edges for 80 bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are retained. Additionally, the AI uses `is_good_trials` to further restrict which trials are valid for each unit, keeping only the intersection of valid trials across all selected units.

ii.
```python
cls=arrcol(nwb.units,'classification').astype(str)
units=np.flatnonzero(cls=='good')
...
# In select_units_and_trials:
obs=np.asarray(nwb.units['obs_intervals'][j],dtype=np.float64)
valid=np.asarray(nwb.units['is_good_trials'][j],dtype=bool)
...
common.intersection_update(map(int,mapped[valid & (mapped>=0)]))
```

iii. The classifier-good filter matches the reference. The additional `is_good_trials` filtering is more conservative — the reference only uses `obs_intervals` from the first good unit to determine which trials have coverage, without consulting `is_good_trials`. The AI reports retaining all 69,453 classifier-good units (same as reference) but fewer trials (90,378 vs 90,860).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are constructed relative to each trial's go-cue time by adding the fixed relative edge array to the absolute go-cue timestamp. Spikes are binned against these absolute edges using `searchsorted`.

ii.
```python
edge_matrix=go[:,None]+EDGES_REL[None,:]
rates[:,ui,:]=np.diff(np.searchsorted(sp,edge_matrix,side='left'),axis=1).astype(np.float32)/BIN_S
```

iii. Same alignment approach as the reference. All timestamps are on the same absolute clock, so adding go-cue times to relative offsets gives the correct absolute bin edges.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50-ms non-overlapping bins, 80 bins per trial spanning -2.5 to +1.5 s relative to go cue. No rebinning or smoothing.

ii.
```python
BIN_S=0.050
OFF_START=-2.5
OFF_END=1.5
EDGES_REL=np.arange(OFF_START, OFF_END + BIN_S/2, BIN_S, dtype=np.float64)
CENTERS_REL=(EDGES_REL[:-1]+EDGES_REL[1:])/2
N_TIME=len(CENTERS_REL)  # 80
```

iii. Matches the instructions and reference. 50 ms bins, 80 timepoints.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (the tone onsets) and `go_start_times` (the go cues). The tone for each trial is the last sample onset before the go cue.

ii.
```python
sample=np.asarray(ev['sample_start_times'].timestamps[:],dtype=np.float64)
...
ix=np.searchsorted(sample,go[:n],side='right')-1
...
tone=sample[ix[trial_idx]]
```

iii. Same approach as the reference. The AI correctly identifies that early licks replay the sample epoch, so the last tone before go is appropriate.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin center, the elapsed time from tone onset is computed as `center_time - tone_time`, where center times are absolute (go + relative center offset).

ii.
```python
centers=g+CENTERS_REL
inp[0]=(centers-to).astype(np.float32)
```

iii. Equivalent to the reference's `CENTERS[None, :] + (go - tone)[:, None]`. Both compute `bin_center_abs - tone_abs = (go + rel_center) - tone = rel_center + (go - tone)`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin-center grid (go + relative offsets) is used for both neural binning and the tone input, ensuring temporal alignment.

ii.
```python
centers=g+CENTERS_REL  # same grid as neural edges
inp[0]=(centers-to).astype(np.float32)
```

iii. Same alignment as reference — both use the go-cue-relative grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_start_times` and `photostim_stop_times` event timestamps in `BehavioralEvents`, rather than the trial-table `photostim_onset`/`photostim_duration` used by the reference.

ii.
```python
pstart=np.asarray(ev['photostim_start_times'].timestamps[:],dtype=np.float64)
pstop=np.asarray(ev['photostim_stop_times'].timestamps[:],dtype=np.float64)
```

iii. The AI chose to use event timestamps rather than trial-table fields for photostimulation. The CONVERSION_NOTES state: "Event timestamps, rather than string-formatted trial photostimulation fields, are authoritative for temporal inputs." This is a different source than the reference but should produce equivalent results since both encode the same stimulation periods.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each bin center, the AI finds the latest `photostim_start_times` before the center and checks if the center falls before the corresponding `photostim_stop_times`. This produces a binary 0/1 time-varying signal.

ii.
```python
if len(pstart):
    jj=np.searchsorted(pstart,centers,side='right')-1
    valid=jj>=0; stim=np.zeros(N_TIME,dtype=bool)
    stim[valid]=centers[valid] < pstop[jj[valid]]
    inp[1]=stim.astype(np.float32)
else: inp[1]=0
```

iii. This differs from the reference which uses trial-specific `photostim_onset`/`photostim_duration` relative to trial start. The AI's approach is session-global: it searches across all stimulation events for any trial's bin. This could potentially match stimulation from adjacent trials if they overlap.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The same bin centers (go + relative offsets) are used for both neural and photostim, ensuring temporal alignment.

ii.
```python
centers=g+CENTERS_REL
...
jj=np.searchsorted(pstart,centers,side='right')-1
```

iii. Same alignment approach as reference.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From `trial_instruction` (left/right) and `outcome` (hit/miss/ignore) in the trials table.

ii.
```python
instruction=arrcol(tr,'trial_instruction').astype(str)[trial_idx]
outcome_s=arrcol(tr,'outcome').astype(str)[trial_idx]
...
def trial_choice(instruction,outcome):
    if outcome=='ignore': return 2
    if outcome=='hit': return 0 if instruction=='left' else 1
    if outcome=='miss': return 1 if instruction=='left' else 0
```

iii. Same source variables and derivation logic as reference.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is derived as: ignore -> 2 (no lick), hit -> instructed side (left=0, right=1), miss -> opposite side. The value is repeated across all 80 time bins.

ii.
```python
out[0]=trial_choice(instruction[k],outcome_s[k])
```

iii. Same logic as reference's `np.where` approach. Both map to left=0, right=1, no lick=2.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table.

ii.
```python
outcome_s=arrcol(tr,'outcome').astype(str)[trial_idx]
...
out[1]={'ignore':0,'miss':1,'hit':2}[outcome_s[k]]
```

iii. Same source as reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct mapping: ignore=0, miss=1, hit=2. Repeated across all 80 bins.

ii.
```python
out[1]={'ignore':0,'miss':1,'hit':2}[outcome_s[k]]
```

iii. Same mapping as reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table.

ii.
```python
early_s=arrcol(tr,'early_lick').astype(str)[trial_idx]
...
out[2]={'no early':0,'early':1}[early_s[k]]
```

iii. Same source as reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Direct mapping: no early=0, early=1. Repeated across all 80 bins.

ii.
```python
out[2]={'no early':0,'early':1}[early_s[k]]
```

iii. Same mapping as reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `Camera0_side_TongueTracking` in `BehavioralTimeSeries`, specifically the y-position (column 1) and likelihood (column 2), plus the timestamps.

ii.
```python
ts=series[key]  # key='Camera0_side_TongueTracking'
times=np.asarray(ts.timestamps[:],dtype=np.float64)
data=np.asarray(ts.data[:],dtype=np.float32)
y=data[:,1]; likelihood=data[:,2]
```

iii. Same source as reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI uses a nearest-frame approach instead of bin-averaging. For each bin center, it finds the nearest video frame. If the frame's likelihood >= 0.9 and falls within video coverage, the raw y value is compared against session-wide percentile thresholds. Session percentiles (40th, 60th) are computed from all visible frames across the session (not bin-averaged).

ii.
```python
visible=np.isfinite(y)&np.isfinite(likelihood)&(likelihood>=LIKELIHOOD_THRESHOLD)
thresholds=tuple(np.percentile(y[visible],[40,60]).tolist())
...
q=nearest_indices(vt,centers)
covered=(centers>=vt[0])&(centers<=vt[-1])&(np.abs(vt[q]-centers)<=0.010)
vis=covered&np.isfinite(vy[q])&np.isfinite(vl[q])&(vl[q]>=LIKELIHOOD_THRESHOLD)&np.isfinite(p40)&np.isfinite(p60)
yy=vy[q]; tc[vis & (yy<p40)]=0; tc[vis & (yy>=p40)&(yy<=p60)]=1; tc[vis & (yy>p60)]=2
```

iii. Key differences from reference: (1) LIKELIHOOD_THRESHOLD=0.90 instead of 0.50; (2) nearest-frame interpolation instead of bin-mean averaging; (3) percentiles computed over raw visible frames rather than session-wide bin means; (4) middle class uses `>=p40 & <=p60` (inclusive on both ends) vs reference's `np.digitize` behavior. The reference averages multiple frames per bin and takes percentiles of the bin means, which is more statistically robust.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Categories are: 0 if y < p40, 1 if p40 <= y <= p60, 2 if y > p60, 3 if not visible. Thresholds are the 40th and 60th percentiles of visible (likelihood >= 0.9) raw y values across the session.

ii.
```python
tc[vis & (yy<p40)]=0
tc[vis & (yy>=p40)&(yy<=p60)]=1
tc[vis & (yy>p60)]=2
```

iii. The reference uses `np.digitize(m[ok], edges)` which gives: 0 if y < p40, 1 if p40 <= y < p60, 2 if y >= p60. The AI's inclusive-both-ends middle class means boundary values at exactly p60 go to class 1 instead of class 2. Additionally, the AI computes percentiles over raw frames (not bin means), and uses a higher confidence threshold (0.9 vs 0.5).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI uses the same bin-center grid (go + relative offsets) to query the nearest video frame for each bin, aligning tongue data with neural data.

ii.
```python
centers=g+CENTERS_REL
q=nearest_indices(vt,centers)
```

iii. The reference similarly uses the go-relative grid, but bins frames into the same 50-ms windows rather than taking the nearest frame. Both ensure temporal alignment with neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Session with no classifier-good units — dropped (returns None); (2) Trials outside observation validity — dropped via `is_good_trials` intersection; (3) Zero-neural trials — dropped after binning; (4) Tongue frames with low likelihood — assigned "not visible" class (3); (5) Blank anatomy labels — raises ValueError rather than silently proceeding.

ii.
```python
if units.size==0 or trial_idx.size<2: return None
...
neural_valid=np.any(rates>0,axis=(1,2))
...
if np.any(names==''): raise ValueError(f'Blank anatomy among selected units: {path.name}')
```

iii. The AI's approach is more aggressive in filtering — it drops trials with zero neural activity even if they are otherwise valid, and raises errors on blank anatomy labels. The reference handles the NaN classification case with a `_text` helper that converts non-strings to empty strings, and doesn't explicitly check for blank anatomy.

## 10-a. What are the most time-consuming steps of the code?

i. Reading NWB files dominates. The AI reports ~0.6-1.4 s per session, with total conversion taking 209 s for 173 sessions. The vectorized searchsorted approach was adopted after a 9.8x speedup from the initial per-unit/per-trial histogram approach.

ii. N/A (timing infrastructure throughout code)

iii. Similar profile to reference. The AI improved from an initial implementation that would have taken >15 min.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit spike binning loop remains (one `searchsorted` per unit), and the per-trial loop for constructing inputs/outputs. The AI reads each unit's spike times individually rather than the bulk buffer approach used by the reference.

ii.
```python
spikes=[np.asarray(nwb.units['spike_times'][j],dtype=np.float64) for j in units]
for ui,sp in enumerate(spikes):
    rates[:,ui,:]=np.diff(np.searchsorted(sp,edge_matrix,side='left'),axis=1)
...
for k,(g,to) in enumerate(zip(go,tone)):
    # construct inputs and outputs per trial
```

iii. The reference avoids the per-trial loop for inputs/outputs by using vectorized numpy operations. The AI's per-trial loop for constructing inputs and outputs is less efficient but functionally equivalent. The reference also reads the spike times buffer once in bulk rather than per-unit.

## 10-c. What processing does the code repeat multiple times?

i. No major repeated processing. Each NWB file is opened once. However, the per-trial loop reconstructs `centers = g + CENTERS_REL` for every trial, which could be vectorized as a single matrix operation.

ii.
```python
for k,(g,to) in enumerate(zip(go,tone)):
    centers=g+CENTERS_REL
```

iii. Minor inefficiency, not a significant concern.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores extensive per-session metadata (`info` dict) including raw trial counts, structural validity counts, timing, etc. This metadata is preserved in the output but not used by the decoder. The `nearest_indices` function computes both forward and backward distances to find the nearest frame, which is more work than needed.

ii.
```python
info=dict(file=path.name,identifier=nwb.identifier,subject=str(nwb.subject.subject_id),
          raw_trials=len(tr),structurally_valid_trials=len(structural_idx),
          observation_valid_trials=int(len(neural_valid)),
          dropped_zero_neural_trials=dropped_zero_neural, ...)
```

iii. The extra metadata is useful for debugging/auditing but not required by the decoder format. The reference stores less metadata per session.
