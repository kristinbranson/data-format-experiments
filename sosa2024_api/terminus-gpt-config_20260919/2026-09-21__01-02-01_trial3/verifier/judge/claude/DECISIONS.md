# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All `*.nwb` files under `/app/data` are discovered recursively with `Path.rglob` and sorted, giving 152 session files across the 11 `sub-*` directories. Each file is opened once with `pynwb.NWBHDF5IO(..., 'r', load_namespaces=True)` (no `h5py` anywhere), and every stream needed for the conversion is read inside that single context: the frame-sampled behavior streams in `processing['behavior']['BehavioralTimeSeries']`, the sparse `Reward` event series, and every `RoiResponseSeries` in `processing['ophys']['Deconvolved']`. Behavior streams are selected by matching their length to the `trial_start` stream length, so all frame-sampled streams are loaded in one dict. `--sample` processes the first 2 files; `--full` (default) processes all 152.

ii.
```python
files=sorted(Path('/app/data').rglob('*.nwb')); files=files[:2] if a.sample else files
...
with NWBHDF5IO(str(fn),'r',load_namespaces=True) as io:
    nwb=io.read(); beh=nwb.processing['behavior']['BehavioralTimeSeries'].time_series
    bt=np.asarray(beh['trial_start'].timestamps[:],float)
    B={k:np.asarray(v.data[:]) for k,v in beh.items() if v.data.shape[0]==len(bt)}
    ...
    dec=nwb.processing['ophys']['Deconvolved'].roi_response_series
```

iii. From CONVERSION_NOTES.md Step 2: "`/app/data` contains 152 `*_behavior+ophys.nwb` files, one file per recording session, organized in subject directories. All files were opened and inspected with `pynwb.NWBHDF5IO(..., mode="r", load_namespaces=True)`; no `h5py` was used." The AI independently scanned all 152 files and cross-checked per-session trial and neuron counts against the converted product (Step 9/Step 10 Check 6), confirming nothing was dropped: 152 sessions, 12,216 behavioral trials, 138,678 curated neuron-recordings.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the NWB metadata field `nwb.subject.subject_id` of each file (not from the directory name). Subjects are accumulated in order of first appearance into `data['subjects']`, and each session records `subject_idx = subjects.index(sub)`. This yields the 11 subjects m11–m19, m3, m4, m7.

ii.
```python
N,I,O,sub,info=process_file(fn,a.show_processing)
...
if sub not in subjects: subjects.append(sub)
sidx.append(subjects.index(sub))
...
'subjects':subjects,'subject_idx':np.asarray(sidx,np.int64),
```
and in `process_file`:
```python
sid=f'{nwb.subject.subject_id}_{nwb.session_id or fn.stem}'
...
return neural_trials,inputs,outputs,nwb.subject.subject_id,info
```

iii. CONVERSION_NOTES.md Step 2 records "Subjects | 11" and "Session/subject metadata: NWB subject ID, session ID, start time, and imaging-plane metadata". The AI used the in-file subject metadata rather than parsing the path, and verified in Step 9 that the subject count (11) and per-subject session counts match an independent pynwb scan of the raw files.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Files are processed in sorted path order, so sessions are ordered by subject then session number, and `neural`/`input`/`output`/`subject_idx`/`brain_region_idx` all use that same session ordering. No merging or splitting of files is done, and no cross-session neuron alignment is attempted.

ii.
```python
files=sorted(Path('/app/data').rglob('*.nwb'))
for i,fn in enumerate(files):
    N,I,O,sub,info=process_file(fn,a.show_processing)
    neural.append(N);inputs.append(I);outputs.append(O);infos.append(info)
...
sid=f'{nwb.subject.subject_id}_{nwb.session_id or fn.stem}'
```

iii. Step 2: "one file per recording session"; Step 4 resolution: "One NWB per released session ... Retain every nonduplicate NWB session." The AI explicitly rejected the reference repository's habit of selecting only the first session of a multi-session day ("For conversion, each NWB file/session will be treated according to its native session identity rather than silently dropping sessions").

## 1-d. How are the data split into trials?

i. A trial runs from each `trial_start` pulse to the next `teleport` pulse. Pulses are found as nonzero samples of the frame-sampled `trial_start` and `teleport` streams; for each start the first teleport at or after it is taken, and the pair is kept only if that teleport precedes the next trial start. The inter-trial interval (teleport → next start, which carries the −500 cm position sentinel) is excluded. Trial boundaries are then converted to absolute times `bt[s]`, `bt[e]` and all streams are re-sampled onto a 100 ms grid within that window.

ii.
```python
starts=np.flatnonzero(B['trial_start']>0); tele=np.flatnonzero(B['teleport']>0)
pairs=[]
for j,s in enumerate(starts):
    e0=tele[tele>=s]
    if len(e0) and (j+1==len(starts) or e0[0]<starts[j+1]): pairs.append((int(s),int(e0[0])))
if len(pairs)<2: raise ValueError(f'{fn}: fewer than 2 complete trials')
...
for ti,(s,e) in enumerate(pairs):
    t_start,t_end=bt[s],bt[e]
    centers=t_start+DT/2+np.arange(max(1,int(np.floor((t_end-t_start)/DT))))*DT
    centers=centers[centers<=t_end]
```

iii. Step 2: "NWB `trials`, `intervals`, and `units` tables are absent. Trials are reconstructed from each `trial_start` pulse to the following `teleport` pulse; all scanned starts had a valid following teleport before the next start. Native trials include inter-trial periods between teleport and the next trial start; these are excluded because temporal alignment is to explicit trial start." Step 10 edge-case check: "Explicit starts pair one-to-one with following teleports." This reproduces 12,216 trials, matching the number of `trial_start` pulses in the raw files.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, no minimum-duration filter:
1. **Complete imaging coverage** — a trial is kept only if both its start and its teleport timestamp lie inside the neural time range of *every* imaging plane. This removed exactly 1 trial (the terminal trial of m14_12), where behavior continued after imaging stopped and nearest-frame extrapolation would have produced fabricated (all-zero) neural data.
2. **Session-level** — a session must yield at least 2 complete, fully covered trials, otherwise the file raises (no session actually triggered this).
Final dataset: 12,215 of 12,216 trials. Shortest retained trial is 61 bins (6.1 s).

ii.
```python
# Retain only trials fully covered by every neural plane. Behavioral recording
# can continue after imaging ends; nearest-neighbor extrapolation is invalid.
neural_start=max(x[1][0] for x in plane_data)
neural_end=min(x[1][-1] for x in plane_data)
n_pairs_raw=len(pairs)
pairs=[(s,e) for s,e in pairs if bt[s]>=neural_start and bt[e]<=neural_end]
excluded_no_neural=n_pairs_raw-len(pairs)
if len(pairs)<2: raise ValueError(f'{fn}: fewer than 2 fully imaging-covered trials')
```

iii. Step 10 / Step 12: "Final audit exposed validator warning 'session 76, trial 42: all neural data is zero.' Raw pynwb inspection showed behavior continued after imaging ended in m17_09; nearest-frame extrapolation was invalid. A full-coverage rule was added, requiring trial start and teleport to lie within every plane's neural time range." For the one long (216.6 s) trial, the AI checked raw timestamps, `scanning`, and pulse ordering and concluded it was "a prolonged pause rather than an indexing error. It is valid and was retained rather than applying an unsupported duration filter."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is the NWB `processing['ophys']['Deconvolved']` RoiResponseSeries (suite2p's `spks` array, stored in raw fluorescence units), subset to ROIs with `iscell[:,0] > 0`. For multi-plane sessions every plane's series is used and the curated cells are concatenated. The raw `Fluorescence` (F) and `Neuropil` (Fneu) series present in the same files are **not** used, and no dF/F is computed.

ii.
```python
dec=nwb.processing['ophys']['Deconvolved'].roi_response_series
n_planes=len(dec)
for plane_name,rr in sorted(dec.items()):
    region=np.asarray(rr.rois.data[:],dtype=int)
    table=rr.rois.table
    iscell=np.asarray(table['iscell'].data[:])[:,0]>0
    valid=iscell[region]
    ...
    neural_all=np.asarray(rr.data[:, :],np.float32)[:,valid]
    plane_data.append((plane_name,neural_t,neural_all))
```

iii. Step 1: "The main neural activity key used for place-cell detection is `events` (deconvolved calcium-event activity), not raw fluorescence. Thus no new delta-F/F calculation is indicated for this conversion when the NWB already provides processed event/deconvolved traces." Step 4 discrepancy table: "Neural signal | `timeseries['events']` | `processing/ophys/Deconvolved/plane0` | Deconvolved calcium events | Use NWB Deconvolved series." The AI equated the paper's `sess.timeseries['events']` with the NWB `Deconvolved` array without testing that equivalence.

## 2-b. How is the `neural` data processed?

i. Minimal processing: read the stored deconvolved values for curated ROIs, then average the samples falling in each 100 ms bin of the trial window (`mean` over samples whose time lies in `[center−50 ms, center+50 ms)`); if a bin happens to contain no sample, the nearest sample is copied. Planes are concatenated along the neuron axis, and the result is transposed to (n_neurons, n_timepoints), float32. There is no neuropil subtraction, no baseline estimation, no dF/F, no Gaussian smoothing, no OASIS deconvolution, and no per-cell normalization.

ii.
```python
# Mean native deconvolved samples in each temporal bin.
binned_planes=[]
for plane_name,neural_t,neural_all in plane_data:
    ni=np.searchsorted(neural_t,centers-DT/2); nj=np.searchsorted(neural_t,centers+DT/2)
    xp=np.empty((neural_all.shape[1],len(centers)),np.float32)
    for k,(a,b) in enumerate(zip(ni,nj)):
        if b>a: xp[:,k]=neural_all[a:b].mean(0)
        else: xp[:,k]=neural_all[nearest_idx(neural_t,np.array([centers[k]]))[0]]
    binned_planes.append(xp)
X=np.concatenate(binned_planes,axis=0)
```

iii. Step 5 mapping: "`Deconvolved/plane0` + `iscell` → `neural`: retain `iscell[:,0]>0`; average event amplitudes in 100 ms bins; transpose to neuron×time ... Timestamp-aligned, no spatial smoothing." Step 10 Check 5: "Binning: paper spatial analyses use 10 cm bins, while required decoder alignment necessitates common 100 ms temporal bins; neural events are mean-binned without spatial smoothing." The metadata records `'neural_signal': 'Suite2p deconvolved calcium events; iscell curated; mean in 100 ms bins'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single filter: suite2p/manual curation via `iscell[:,0] > 0`, applied per plane through each `RoiResponseSeries`'s own `rois` `DynamicTableRegion` (so that in two-plane sessions each series indexes only its own rows of the shared PlaneSegmentation table). This keeps 138,678 of 312,110 segmented ROIs (155–2,341 per session). No other neural curation is applied — in particular the paper's exclusion of putative interneurons (dF/F–speed Pearson r > 0.5) is not implemented, nor are place-cell/SI or speed filters.

ii.
```python
region=np.asarray(rr.rois.data[:],dtype=int)
table=rr.rois.table
iscell=np.asarray(table['iscell'].data[:])[:,0]>0
valid=iscell[region]
...
neural_all=np.asarray(rr.data[:, :],np.float32)[:,valid]
```

iii. Step 3 curation rules: "Use Suite2p `iscell` classification. Place-cell significance and reward-relative classification are downstream analysis labels and must not restrict a general neural decoder." Step 12 Check 7: "Deconvolved events and Suite2p `iscell` match the reference ... Applying target-selective cell filtering would introduce leakage." The interneuron-exclusion step from the Methods is never mentioned anywhere in the notes or trajectory.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the explicit `trial_start` pulse. For each trial the 100 ms bin grid is built forward from the absolute trial-start timestamp `bt[s]` (first bin center at +0.05 s) and truncated at the teleport timestamp, so bin 0 of every trial begins exactly at trial start. Neural samples are assigned to those bins by their own absolute times (`neural_t`), not by index, so neural and behavior share one absolute-time grid. Metadata records `temporal_alignment_event = 'explicit trial_start pulse'`, `off_start = 0.0`, `off_end = None`. No pre-trial window is included.

ii.
```python
t_start,t_end=bt[s],bt[e]
centers=t_start+DT/2+np.arange(max(1,int(np.floor((t_end-t_start)/DT))))*DT
centers=centers[centers<=t_end]
...
if rr.timestamps is not None:
    neural_t=np.asarray(rr.timestamps[:],float)
    effective_rate=1.0/np.median(np.diff(neural_t))
else:
    effective_rate=nominal_rate/n_planes
    neural_t=float(rr.starting_time)+np.arange(nt)/effective_rate
```

iii. Step 2: "behavior and imaging have differing sample counts/rates and therefore must be aligned by time, not index." Step 10 Check 5: "Alignment: both use trial start and teleport; conversion uses their timestamps to avoid cross-stream index assumptions." Step 10/12 also document the discovery that the NWB `rate` on two-plane sessions is the aggregate ~31 Hz scanner rate and that the per-plane effective rate is `rate/n_planes` ≈ 15.5 Hz, which was required for correct timestamps on m17/m18.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 100 ms for every trial and session (`metadata['time_bin_size'] = 100.0`). This is a rebinning: native imaging is ~15.5078 Hz per plane (~64.5 ms), so each 100 ms bin averages 1 or 2 native samples (empty bins fall back to the nearest sample). Continuous behavior (position, speed) is linearly interpolated to bin centers, discrete behavior (lick) is nearest-sample sampled, and per-trial scalars are tiled across bins.

ii.
```python
DT=0.1
...
centers=t_start+DT/2+np.arange(max(1,int(np.floor((t_end-t_start)/DT))))*DT
...
ni=np.searchsorted(neural_t,centers-DT/2); nj=np.searchsorted(neural_t,centers+DT/2)
...
'metadata':{... 'time_bin_size':DT*1000, ...}
```

iii. Step 5 Key Decision 1: "**Temporal bins**: 100 ms for every session. This is fine enough relative to ~15.5/30 Hz imaging, ensures a common bin size, and avoids pretending unequal native frame durations are identical." Step 4: "Decoder requires time from trial start | Use 100 ms temporal bins; do not apply spatial smoothing."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From the timestamps of the behavioral `trial_start` series: `bt = beh['trial_start'].timestamps[:]`. The trial-start time is `bt[s]` at the pulse sample, and the time axis is the constructed 100 ms bin-center grid, so this input is `centers − t_start`, i.e. 0.05, 0.15, 0.25, … s.

ii.
```python
bt=np.asarray(beh['trial_start'].timestamps[:],float)
...
t_start,t_end=bt[s],bt[e]
centers=t_start+DT/2+np.arange(max(1,int(np.floor((t_end-t_start)/DT))))*DT
...
inp=np.vstack([centers-t_start, ...])
```

iii. Step 2: "Each stream has timestamps; behavior and imaging have differing sample counts/rates and therefore must be aligned by time, not index." All frame-sampled behavior streams share one timestamp vector, so using `trial_start`'s timestamps is equivalent to using any other stream's.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtraction of the trial-start timestamp from the bin centers. Because the grid is regular by construction, the value is exactly `0.05 + 0.1·k` seconds. Observed range across the dataset: [0.1, 216.6] s (reported to 1 decimal by the validator). The value is stored as float32 as row 0 of `input`.

ii.
```python
inp=np.vstack([centers-t_start,np.full(len(centers),env),np.full(len(centers),trialnum),np.full(len(centers),prev)]).astype(np.float32)
```

iii. Step 5 mapping: "trial-relative bin centers → `input[0]`: seconds from explicit trial start; continuous time-varying". Step 10 edge checks confirm "all 100 ms time vectors are strictly increasing."

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the same `centers` array defines both the neural bins and the time input, so row 0 of `input` is exactly the time axis of the neural matrix. Every trial asserts equal widths across neural, input, and output.

ii.
```python
assert X.shape[1]==inp.shape[1]==out.shape[1] and np.isfinite(X).all()
```

iii. Step 10 Check 3: independently reconstructed time, environment, trial number, and previous outcome from raw NWB loads and compared with `np.allclose`; all passed. Step 2: alignment is done by absolute timestamps rather than by index because "behavior and imaging have differing sample counts/rates".

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The frame-sampled `environment` behavior stream, read at the trial-start sample index `s`.

ii.
```python
env=int(round(float(B['environment'][s]))); env=max(0,min(1,env))
```

iii. Step 2/Step 5: "`environment` uses valid trial codes with `-1` outside trials"; mapping table: "`environment` → `input[1]`: map native 0/1 to ENV1/ENV2; constant per trial". The AI verified in Step 10 edge checks that "environment values at starts" are valid, so the `-1` inter-trial sentinel never leaks in (confirmed: env at every trial-start sample is 0 or 1).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The single value read at trial start is rounded, clipped to {0,1}, and tiled across all bins of the trial (constant per trial, delivered as a time-varying row). Values across the dataset span [0,1], with m17/m18 sessions being ENV2 (=1) as expected.

ii.
```python
env=int(round(float(B['environment'][s]))); env=max(0,min(1,env))
...
inp=np.vstack([centers-t_start,np.full(len(centers),env), ...])
```

iii. Step 5 Key Decision 6: "**Trial labels as matrices**: repeat per-trial inputs/outputs across time so every trial has `(n_variables, n_timepoints)` and validator semantics are unambiguous." The clip guards against the `-1` sentinel.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The native `trial number` behavior stream, sampled at the trial-start index. (I verified independently that in all 152 sessions `trial number` at the `trial_start` pulses equals `0,1,2,…,n−1` exactly, so this is identical to a sequential within-session index; range [0,99].)

ii.
```python
trialnum=float(B['trial number'][s])
...
inp=np.vstack([...,np.full(len(centers),trialnum),...])
```

iii. Step 5 mapping: "`trial number` → `input[2]`: native continuous trial number; constant per trial, repeated." The AI preferred the stored task variable over a derived counter; Step 10 edge checks confirmed per-trial constancy and in-range values.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond reading the value at trial start, casting to float, and tiling it across the trial's time bins. Note that it is the *native, session-local* trial number, so excluded trials do not renumber the surviving ones (only relevant for the one excluded terminal trial of m14_12).

ii.
```python
trialnum=float(B['trial number'][s]); ...
inp=np.vstack([centers-t_start,np.full(len(centers),env),np.full(len(centers),trialnum),np.full(len(centers),prev)]).astype(np.float32)
```

iii. Same as 5-a; README: "Session-local trial number".

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the sparse `Reward` event `TimeSeries` timestamps (`beh['Reward'].timestamps`), which are event times rather than frame-sampled values. A trial is "rewarded" if at least one reward timestamp lies in `[t_start, t_end]`; the *previous* trial's value becomes this trial's input.

ii.
```python
rew_t=np.asarray(beh['Reward'].timestamps[:],float)
...
rewarded=int(np.any((rew_t>=t_start)&(rew_t<=t_end)))
prev=outcomes[-1] if outcomes else 0; outcomes.append(rewarded)
```

iii. Step 2: "`Reward` is a sparse event `TimeSeries` (0.004 mL per event) with its own timestamps. Reward outcome is determined by whether a Reward timestamp lies from trial start through teleport." Step 10: "Sparse Reward series edge case: Reward is event-based and shorter than frame-sampled behavior. It is aligned directly by timestamps, never indexed as a behavioral frame array."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Running state: `outcomes` accumulates each retained trial's binary outcome; the current trial takes the last appended value, and the first trial of each session is set to 0. The value is tiled across the trial's bins. Because `outcomes` tracks *retained* trials, a dropped trial would shift the reference; the only dropped trial is terminal, so no trial is affected.

ii.
```python
prev=outcomes[-1] if outcomes else 0; outcomes.append(rewarded)
...
inp=np.vstack([...,np.full(len(centers),prev)]).astype(np.float32)
```

iii. Step 5 mapping: "previous Reward event → `input[3]`: previous retained trial rewarded=1 else 0; first trial=0; repeated over time." Step 10 Check 3 independently reconstructed this from raw NWB and matched with `np.allclose`.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behavior stream plus a per-trial active-zone label. The label is inferred from the native `reward_zone` stream: for each trial the samples where `reward_zone > 0` are found, the median position over those samples is taken, and the trial is assigned to whichever of three hard-coded zones has the nearest center. Trials with no `reward_zone` activation (omissions/no-lick trials) inherit the label of the nearest labeled trial in the session. The zone table is hard-coded as `A=[80,100]`, `B=[200,220]`, `C=[320,340]` cm.

ii.
```python
ZONES=np.array([[80.,100.],[200.,220.],[320.,340.]])

def zone_for_trial(pos):
    return int(np.argmin(np.abs(ZONES.mean(1)-pos)))
...
raw_zone=[]
for s,e in pairs:
    hit=np.flatnonzero(B['reward_zone'][s:e+1]>0)
    if len(hit): raw_zone.append(zone_for_trial(float(np.median(B['position'][s+hit]))))
    else: raw_zone.append(-1)
# Fill omission/no-activation labels from nearest labeled trial; blocks are contiguous.
good=np.flatnonzero(np.asarray(raw_zone)>=0)
if not len(good): raise ValueError(f'{fn}: cannot infer reward zone')
zones=np.asarray(raw_zone)
bad=np.flatnonzero(zones<0)
zones[bad]=zones[good[np.argmin(abs(good[:,None]-bad),axis=0)]] if len(bad) else zones[bad]
```

iii. Step 4/Step 10: "`reward_zone` stream ... 0–6 transient values near reward delivery ... Derive zone identity from task block/location; do not interpret this stream as A/B/C." Trajectory step 22: "The NWB `reward_zone` stream is not a zone identity; it behaves like a within-zone sample/event count and is therefore unsuitable as the A/B/C label. Trial inspection shows the first reward zone begins near 80 cm, consistent with the task's three fixed reward locations." The 20 cm zone extents were inferred from the span of positions over which `reward_zone` was non-zero in the inspected trials (typically ~80–90 cm), not from the paper or the reference code.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from the interpolated position to the nearest edge of the trial's active zone: negative before the zone (`pos − lo`), zero inside `[lo, hi]`, positive after (`pos − hi`), using `lo,hi = ZONES[z]` with the 20 cm spans above. Position itself is linearly interpolated from the raw `position` stream onto the 100 ms bin centers. The continuous distance is then discretized (7-c).

ii.
```python
z=int(zones[ti]); lo,hi=ZONES[z]
pos=np.interp(centers,bt,B['position']).astype(np.float32)
dist=np.where(pos<lo,pos-lo,np.where(pos>hi,pos-hi,0.)).astype(np.float32)
```

iii. Step 5 Key Decision 4: "**Reward zones**: fixed 20 cm zones A=80–100, B=200–220, C=320–340 cm. Determine active zone from the reward block/location evident in the trial behavior; distance is to the closest point in that zone, making all in-zone samples exactly zero." Mapping table: "position + active fixed zone → `output[0]`: signed distance to nearest point in zone ... zero throughout zone", matching the instruction that class 3 is exactly 0 cm.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes via explicit boolean masks: `<−50` → 0; `[−50,−10)` → 1; `[−10,0)` → 2; exactly 0 → 3; `(0,10]` → 4; `(10,50]` → 5; `>50` → 6. This matches the instruction's bin list; the only convention difference from `np.digitize`-style binning is that exactly +10 and exactly +50 fall in the lower class. Resulting fractions: [0.252, 0.102, 0.074, 0.168, 0.027, 0.083, 0.295].

ii.
```python
def discretize_distance(d):
    y=np.empty(d.shape,np.int64)
    y[d < -50]=0; y[(d>=-50)&(d<-10)]=1; y[(d>=-10)&(d<0)]=2
    y[d==0]=3; y[(d>0)&(d<=10)]=4; y[(d>10)&(d<=50)]=5; y[d>50]=6
    return y
```

iii. Step 5 mapping: "bins `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`", copied directly from the Decoder Task specification. Step 10 checks verified all output values stay within the declared range and that `output_values` labels line up with the class indices.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is interpolated onto exactly the same 100 ms bin centers used to bin the neural data, so the distance row shares the neural time axis sample-for-sample; zone identity is a per-trial constant tiled over those bins. Equal widths are asserted per trial.

ii.
```python
pos=np.interp(centers,bt,B['position']).astype(np.float32)
...
out=np.vstack([discretize_distance(dist),poscls,speedcls,lick,np.full(len(centers),z),np.full(len(centers),rewarded)]).astype(np.int64)
assert X.shape[1]==inp.shape[1]==out.shape[1] and np.isfinite(X).all()
```

iii. Step 10 Check 4: "Raw timestamp-interpolated position/speed, nearest-sample lick, zone distance/classes, and sparse Reward outcome matched converted outputs exactly with `np.allclose(rtol=0, atol=0)`" for three independently loaded session/trial combinations, including a multi-plane session.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The frame-sampled `position` behavior stream (cm along the 450 cm corridor), read for the whole session and interpolated to the trial's bin centers. The −500 cm inter-trial sentinel is never included because trials stop at the teleport pulse.

ii.
```python
B={k:np.asarray(v.data[:]) for k,v in beh.items() if v.data.shape[0]==len(bt)}
...
pos=np.interp(centers,bt,B['position']).astype(np.float32)
```

iii. Step 2/Step 5: "position includes -500 cm inter-trial sentinel values that disappear when slicing explicit trials"; mapping table: "`position` → `output[1]`".

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Only linear interpolation of the raw position onto the 100 ms bin centers, then discretization. No smoothing, unwrapping, or clipping is applied; the open first/last bins absorb the few samples marginally outside [0,450].

ii.
```python
pos=np.interp(centers,bt,B['position']).astype(np.float32)
poscls=np.digitize(pos,[90,180,270,360],right=False).astype(np.int64)
```

iii. Step 5 mapping: "`position` → `output[1]`: bins `<90`, `[90,180)`, `[180,270)`, `[270,360]`, `>360`; clip only tiny interpolation excursions for binning". Step 5 Key Decision 5: "linear interpolation for position/speed; nearest prior/sample for discrete streams."

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize(pos, [90,180,270,360], right=False)` → 5 classes of 90 cm each: <90, 90–180, 180–270, 270–360, >360, with the outer bins open-ended. Resulting fractions: [0.209, 0.178, 0.232, 0.227, 0.153].

ii.
```python
poscls=np.digitize(pos,[90,180,270,360],right=False).astype(np.int64)
...
'output_values':[...,['<90 cm','90-180 cm','180-270 cm','270-360 cm','>360 cm'],...]
```

iii. Taken verbatim from the Decoder Task specification ("5 equal-sized bins spanning the 450 cm track"), recorded in the Step 5 mapping table.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same mechanism as 7-d: position is interpolated onto the neural bin centers, so the position class row is sample-aligned to the neural matrix, and the per-trial length assertion enforces equal widths.

ii.
```python
pos=np.interp(centers,bt,B['position']).astype(np.float32)
...
assert X.shape[1]==inp.shape[1]==out.shape[1]
```

iii. Step 10 Check 4 (independent raw-NWB `np.allclose` comparison of interpolated position and its classes) and the Step 7 processing plots, which show position class ramping monotonically within a trial with no boundary artifacts.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The frame-sampled `lick` behavior stream, which stores a per-sample lick count (observed values 0–6).

ii.
```python
jj=nearest_idx(bt,centers)
lick=(B['lick'][jj]>0).astype(np.int64)
```

iii. Step 2/Step 5: "`lick` is a per-sample count (0–6), so decoder lick should be binarized as count > 0"; Step 3: "Licks are binary per frame before smoothing in that model [the paper GLM]".

## 9-b. What processing is involved in computing `output` *Lick*?

i. For each 100 ms bin, the single nearest raw sample is selected (`nearest_idx`) and binarized as `> 0`. Because native samples are ~64.5 ms and bins are 100 ms, roughly a third of raw samples are not represented in any bin, so licks occurring only on a skipped sample are lost (an "any lick within the bin" rule was not used). Observed lick fraction is 0.231, which is essentially identical to the reference's 0.230.

ii.
```python
def nearest_idx(t,q):
    i=np.searchsorted(t,q); i=np.clip(i,1,len(t)-1)
    return np.where(q-t[i-1] <= t[i]-q,i-1,i)
...
jj=nearest_idx(bt,centers)
lick=(B['lick'][jj]>0).astype(np.int64)
```

iii. Step 5 Key Decision 5: "linear interpolation for position/speed; nearest prior/sample for discrete streams." Step 12 Check 8: "Native lick count is binarized, matching the paper's binary licks per frame before GLM-specific smoothing. The positive class occupies 23.1% of retained time points."

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The nearest-sample indices are computed against the same bin centers used for the neural binning, so the lick row is on the neural time axis by construction.

ii.
```python
jj=nearest_idx(bt,centers)
lick=(B['lick'][jj]>0).astype(np.int64)
out=np.vstack([discretize_distance(dist),poscls,speedcls,lick,...])
```

iii. Step 10 Check 4: nearest-sample lick was reconstructed independently from raw NWB for three trials and matched exactly with `np.allclose(rtol=0, atol=0)`.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Same derivation as 7-a: the native `reward_zone` activation samples plus `position`, mapped to the nearest of the three hard-coded zone centers, with omission/no-activation trials filled from the nearest labeled trial.

ii. See 7-a (`raw_zone` / `zone_for_trial` / nearest-labeled fill).

iii. See 7-a. Additionally, Step 10: "Transient `reward_zone` semantics: It is not an A/B/C label. Active fixed-zone identity is inferred from its position-localized activation and nearest trial in contiguous blocks for omissions; global A/B/C counts are balanced and all three classes are present."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The zone index (0=A, 1=B, 2=C, ordered by track position) is a per-trial integer tiled across the trial's time bins, with value labels `['A','B','C']`. Resulting trial counts A=4,183, B=4,008, C=4,024 (time-point fractions 0.328 / 0.337 / 0.335), i.e. an even three-way split as expected from the counterbalanced design.

ii.
```python
z=int(zones[ti]); lo,hi=ZONES[z]
...
out=np.vstack([...,np.full(len(centers),z),...]).astype(np.int64)
...
'output_values':[..., ['A','B','C'], ...]
```

iii. Step 9 consistency table: "Zone trial counts ... A=4,183; B=4,008; C=4,024 retained | Exact/inferred from native location activation". Step 5 Key Decision 6 covers the per-trial-tiled representation.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The sparse `Reward` event `TimeSeries` timestamps only (the `data` amounts are not used, being a constant 0.004 mL).

ii.
```python
rew_t=np.asarray(beh['Reward'].timestamps[:],float)
...
rewarded=int(np.any((rew_t>=t_start)&(rew_t<=t_end)))
```

iii. Step 2: "`Reward` is a sparse event `TimeSeries` (0.004 mL per event) with its own timestamps. Reward outcome is determined by whether a Reward timestamp lies from trial start through teleport."

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Per trial, a boolean test of whether any reward timestamp falls in the closed window `[t_start, t_end]` (trial start through teleport), with the result tiled across the trial's bins. No snapping of reward times to behavior samples is needed since the comparison is in absolute time. Result: 10,341 rewarded / 1,874 omission trials → time-point fractions [0.157, 0.843], consistent with the paper's "~15% omission".

ii.
```python
rewarded=int(np.any((rew_t>=t_start)&(rew_t<=t_end)))
...
out=np.vstack([...,np.full(len(centers),rewarded)]).astype(np.int64)
...
outcomes.append(rewarded)
```

iii. Step 12 Check 4: "Specific rewarded, omission, and multi-plane trials were independently loaded through pynwb. Sparse Reward timestamps within start-to-teleport windows exactly matched converted labels." Step 12 Check 5: "Retained data contain 1,874 omissions and 10,341 rewarded trials, so both classes have substantial support despite imbalance."

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Behavior extending past the end of imaging**: trials not fully inside every plane's neural time range are dropped (1 trial, m14_12), preventing fabricated/all-zero neural data.
- **Multi-plane ROI indexing**: each `RoiResponseSeries` is curated through its own `rois` `DynamicTableRegion`, so `iscell` rows are matched to the right plane's columns (this was a bug found and fixed in Step 9).
- **Aggregate vs per-plane rate**: the NWB `rate` on two-plane sessions is the ~31 Hz scanner rate; the effective per-plane rate `rate/n_planes` is used to build timestamps.
- **Unpaired pulses**: a trial start with no teleport before the next start is silently skipped; a file with fewer than 2 usable trials raises.
- **`environment = -1` sentinel** is clipped to {0,1}; the **`position = -500` sentinel** is excluded by construction because trials end at teleport.
- **Sparse `Reward`** is never index-aligned to frame-sampled behavior; it is compared in absolute time.
- **Empty neural bins** fall back to the nearest neural sample.
- A per-trial assertion checks finiteness and equal time widths of neural/input/output.
No minimum trial-duration filter and no explicit NaN imputation are applied.

ii.
```python
pairs=[(s,e) for s,e in pairs if bt[s]>=neural_start and bt[e]<=neural_end]
if len(pairs)<2: raise ValueError(f'{fn}: fewer than 2 fully imaging-covered trials')
...
env=int(round(float(B['environment'][s]))); env=max(0,min(1,env))
...
if b>a: xp[:,k]=neural_all[a:b].mean(0)
else: xp[:,k]=neural_all[nearest_idx(neural_t,np.array([centers[k]]))[0]]
...
assert X.shape[1]==inp.shape[1]==out.shape[1] and np.isfinite(X).all()
```

iii. Step 10 "Issues Found and Resolved" documents each of these as a defect discovered by the validator or by independent raw-NWB checks, with the fix and the re-verification ("no validator warnings remain and no retained trial has all-zero neural data").

## 13-a. What are the most time-consuming steps of the code?

i. The full conversion runs in 124.5 s for all 152 sessions. The dominant costs are (1) HDF5 reads of the deconvolved matrices — `rr.data[:, :]` pulls every ROI column before curation, i.e. ~312k ROI traces rather than the 139k retained; (2) the per-bin Python loop that averages neural samples into 100 ms bins, which runs once per time bin per plane per trial; (3) serializing the 5.89 GiB pickle. Per-session times printed in `conversion_full_out.txt` are 0.2–1.5 s, scaling with ROI count, so I/O dominates.

ii.
```python
neural_all=np.asarray(rr.data[:, :],np.float32)[:,valid]
...
for k,(a,b) in enumerate(zip(ni,nj)):
    if b>a: xp[:,k]=neural_all[a:b].mean(0)
    else: xp[:,k]=neural_all[nearest_idx(neural_t,np.array([centers[k]]))[0]]
...
with open(a.output,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
print(f'{sid}: {info["n_trials"]} trials, {info["n_neurons"]} cells, {time.time()-t0:.2f}s',flush=True)
```

iii. Step 6: "Full deconvolved matrices are read once per session; repeated per-trial file reads are avoided. The small per-bin mean loop is bounded by trial duration." Step 7 run-time table: "Conversion (without plots) ~0.25 s for sample sessions | <2 minutes allowing for larger files and serialization". Per-session timing is printed to let the user find bottlenecks, as the instructions require.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops remain: (1) the per-bin neural averaging loop, which is the only hot one and could be replaced by `np.add.reduceat`/cumulative-sum differencing over the sample axis; (2) the per-trial loop in `process_file`, which is hard to vectorize because trials have different lengths; (3) the trial-pairing loop over `starts`, which could be done with a single `searchsorted` of `tele` into `starts`. The AI acknowledged only the first and chose not to vectorize it.

ii.
```python
for j,s in enumerate(starts):
    e0=tele[tele>=s]
    if len(e0) and (j+1==len(starts) or e0[0]<starts[j+1]): pairs.append((int(s),int(e0[0])))
...
for k,(a,b) in enumerate(zip(ni,nj)):
    if b>a: xp[:,k]=neural_all[a:b].mean(0)
```

iii. Step 6: "Code speedups added: Vectorized timestamp searches/interpolation and one-time curated-column extraction." The bin edges are already computed vectorially with `np.searchsorted`; only the reduction inside each bin stayed in Python, which the AI judged acceptable ("bounded by trial duration") given the measured 2-minute total runtime.

## 13-c. What processing does the code repeat multiple times?

i. Very little. Each NWB file is opened exactly once and each array is read once; there is no separate survey/scan pass, and per-session results are appended straight into the output structure. The only repetition is within the per-trial loop, where `nearest_idx`/`np.interp` are recomputed per trial over the full-session behavior arrays (`np.interp(centers, bt, B['position'])` rescans the session-length `bt` for every trial), and `np.searchsorted(neural_t, ...)` likewise scans the full neural time vector each trial. These are O(log n) per query, so the cost is negligible.

ii.
```python
for ti,(s,e) in enumerate(pairs):
    ...
    pos=np.interp(centers,bt,B['position']).astype(np.float32)
    speed=np.interp(centers,bt,B['speed']).astype(np.float32)
    jj=nearest_idx(bt,centers)
```

iii. Step 6: "Full deconvolved matrices are read once per session; repeated per-trial file reads are avoided." The exploratory all-session scans the AI ran during Steps 2/10 were separate throwaway scripts (moved to `/app/cache/`), not part of `convert_data.py`.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Minor items only:
- All frame-sampled behavior streams are materialized into `B`, including `scanning` and `autoreward`, which are never used.
- The full deconvolved matrix is read for *all* ROIs and then subset to `iscell` (`rr.data[:, :]` then `[:, valid]`), reading roughly 2.2× more data than needed; subsetting through the HDF5 read would avoid this.
- `rates`/`plane_rates`/`zones`/`excluded_no_neural` bookkeeping is stored in `metadata['session_info']` and is not used by the decoder.
- Per-trial scalars (environment, trial number, previous outcome, zone, reward outcome) are tiled across time, multiplying storage — but this is required by the target format, not waste.
- `process_file` accepts a `show` argument that is never used inside the function.

ii.
```python
B={k:np.asarray(v.data[:]) for k,v in beh.items() if v.data.shape[0]==len(bt)}
...
neural_all=np.asarray(rr.data[:, :],np.float32)[:,valid]
...
info=dict(session_id=sid,file=str(fn),n_trials=len(pairs),n_neurons=int(n_valid),rate=float(rates[0]),plane_rates=rates,n_planes=len(plane_data),zones=np.bincount(zones,minlength=3).tolist(),rewarded=int(sum(outcomes)),excluded_no_neural=excluded_no_neural)
```

iii. The AI did not document any wasted computation; Step 6 only states that file I/O is done once per session and that the code was vectorized where practical. The retained per-session `info` is deliberate — it is exposed as `metadata['session_info']` and documented in README.md as "`metadata`, including session details and reward-zone coordinates".
