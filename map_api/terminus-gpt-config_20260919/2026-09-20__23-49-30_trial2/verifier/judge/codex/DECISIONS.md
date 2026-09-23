# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all sessions by recursively finding every `.nwb` file under `/app/data`, sorting the paths, and opening each file with `pynwb.NWBHDF5IO`. Within each file it reads the NWB trial table, the `BehavioralEvents` event streams, the `BehavioralTimeSeries` streams when needed, and the units table.

ii. 
```python
files=sorted(Path('/app/data').rglob('*.nwb'))
for p in files:
    res=convert_session(p,make_plot=args.show_processing and len(sessions)<2)
```

```python
with NWBHDF5IO(str(path),'r',load_namespaces=True) as io:
    nwb=io.read(); tr=nwb.trials
    ev=nwb.acquisition['BehavioralEvents'].time_series
```

iii. In `CONVERSION_NOTES.md`, the AI says the dataset contains 174 NWB files and that “all access and inspection used `pynwb.NWBHDF5IO` with loaded namespaces (never `h5py`)”. The trajectory also shows it treated “one NWB file per session” as the organizing principle.

## 1-b. How are the data split into subjects?

i. The AI uses `nwb.subject.subject_id` as the mouse identifier for each session, then builds `subjects` as the sorted unique set of subject IDs and `subject_idx` as the per-session index into that list.

ii. 
```python
return dict(
    neural=neural,input=inputs,output=outputs,
    subject=info['subject'],regions=names.tolist(),info=info,plot=plot_payload)
```

```python
info=dict(
    file=path.name,identifier=nwb.identifier,
    subject=str(nwb.subject.subject_id), ...
)
```

```python
subjects=sorted({x['subject'] for x in sessions})
smap={v:i for i,v in enumerate(subjects)}
'subjects':subjects,
'subject_idx':np.asarray([smap[x['subject']] for x in sessions],dtype=np.int64),
```

iii. The notes say each NWB file stores its subject in `nwbfile.subject.subject_id`, that there are 28 subjects total, and that these IDs should be used directly rather than inferred from filenames.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order follows the sorted file list. It stores per-session metadata from `nwb.identifier` in `metadata['session_info']`.

ii. 
```python
files=sorted(Path('/app/data').rglob('*.nwb'))
```

```python
info=dict(
    file=path.name,identifier=nwb.identifier,
    subject=str(nwb.subject.subject_id), ...
)
```

```python
'metadata':{
    ...,
    'session_info':[x['info'] for x in sessions]
}
```

iii. The notes explicitly state that “each NWB represents one electrophysiology/behavior session” and repeatedly compare totals in terms of “174 NWB files / sessions”.

## 1-d. How are the data split into trials?

i. The AI starts from the NWB trials table and the `go_start_times` event stream, but it does not simply trust a 1:1 trial mapping. It first limits to `n=min(len(tr), len(go))`, then keeps only trial indices whose go cue has a preceding sample/tone event. Those “structurally valid” indices are passed into later trial selection.

ii. 
```python
nwb=io.read(); tr=nwb.trials
ev=nwb.acquisition['BehavioralEvents'].time_series
go=np.asarray(ev['go_start_times'].timestamps[:],dtype=np.float64)
sample=np.asarray(ev['sample_start_times'].timestamps[:],dtype=np.float64)
n=min(len(tr),len(go))
ix=np.searchsorted(sample,go[:n],side='right')-1
structural_idx=np.flatnonzero(ix>=0)
if structural_idx.size<2: return None
```

iii. In the notes, the AI says “retain completed trial rows with a corresponding go cue and preceding sample onset” and justifies this as a structural validity check. The trajectory shows this came from concern that sample-event arrays can contain replayed epochs after early licks.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters. It keeps only structurally valid trials, then intersects the per-unit valid trial sets obtained by mapping each unit’s `obs_intervals` and `is_good_trials` onto behavioral trial rows. After neural binning, it drops any trial whose entire selected-neuron-by-time neural matrix is zero. It drops sessions with no good units or fewer than two surviving trials. It does not explicitly exclude `free_water` trials.

ii. 
```python
units,trial_idx=select_units_and_trials(nwb,structural_idx)
if units.size==0 or trial_idx.size<2: return None
```

```python
common=set(map(int,structurally_valid_trials))
for j in units:
    obs=np.asarray(nwb.units['obs_intervals'][j],dtype=np.float64)
    valid=np.asarray(nwb.units['is_good_trials'][j],dtype=bool)
    ...
    common.intersection_update(map(int,mapped[valid & (mapped>=0)]))
return units, np.asarray(sorted(common),dtype=int)
```

```python
neural_valid=np.any(rates>0,axis=(1,2))
dropped_zero_neural=int((~neural_valid).sum())
trial_idx=trial_idx[neural_valid]
...
if len(go)<2: return None
```

iii. The AI’s notes say it wanted to “retain their common valid intersection” so that all kept neurons are valid on every retained trial and no data must be imputed. Later notes say 2,423 “all-zero neural trials” were investigated in raw NWB and then removed as recording gaps. It also explicitly chose to retain early-lick, stimulation, auto-water, and free-water classes unless neural validity prevented representation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `units['spike_times']`, using only units whose `classification` is `'good'`. It uses `go_start_times` to position the per-trial bin edges.

ii. 
```python
cls=arrcol(nwb.units,'classification').astype(str)
units=np.flatnonzero(cls=='good')
```

```python
go=np.asarray(ev['go_start_times'].timestamps[:],dtype=np.float64)
spikes=[np.asarray(nwb.units['spike_times'][j],dtype=np.float64) for j in units]
edge_matrix=go[:,None]+EDGES_REL[None,:]
```

iii. The notes describe the intended mapping as “`units.spike_times` -> `neural`” and say classifier-good units are the reference-consistent neural curation rule from the QC paper.

## 2-b. How is the `neural` data processed?

i. For each selected unit, the AI bins absolute spike times into 50 ms go-aligned bins and divides by bin width to convert counts to Hz. It does not smooth, normalize, or baseline-subtract the rates.

ii. 
```python
edge_matrix=go[:,None]+EDGES_REL[None,:]
rates=np.empty((len(go),len(units),N_TIME),dtype=np.float32)
for ui,sp in enumerate(spikes):
    rates[:,ui,:]=np.diff(
        np.searchsorted(sp,edge_matrix,side='left'),
        axis=1
    ).astype(np.float32)/BIN_S
```

iii. The notes say the converter should “histogram selected-unit spikes into edges `go + arange(-2.5,1.5+0.05,0.05)`; divide counts by 0.05 for Hz” and that this is the task-mandated replacement for the authors’ 40 ms sliding windows.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI’s neuron-level QC rule is `classification == 'good'`. It then enforces trial validity by intersecting each good unit’s `obs_intervals`/`is_good_trials` mapping across all selected units, so it preserves all classifier-good units but restricts trials to the common valid subset. Sessions with zero good units are dropped.

ii. 
```python
cls=arrcol(nwb.units,'classification').astype(str)
units=np.flatnonzero(cls=='good')
if units.size==0:
    return units, np.asarray([],dtype=int)
```

```python
obs=np.asarray(nwb.units['obs_intervals'][j],dtype=np.float64)
valid=np.asarray(nwb.units['is_good_trials'][j],dtype=bool)
...
common.intersection_update(map(int,mapped[valid & (mapped>=0)]))
```

iii. In the notes, the AI says the classifier `good` label is the reference-consistent unit filter, and that the `is_good_trials` intersection “preserves all classifier-good units without fabricated values.” The trajectory shows this replaced an earlier idea to drop units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural activity to the go cue. It creates trial-specific absolute bin edges by adding a fixed relative window `[-2.5, 1.5]` s to each trial’s `go_start_times` timestamp, then bins spikes directly against those edges.

ii. 
```python
BIN_S=0.050
OFF_START=-2.5
OFF_END=1.5
EDGES_REL=np.arange(OFF_START, OFF_END + BIN_S/2, BIN_S, dtype=np.float64)
```

```python
go=np.asarray(ev['go_start_times'].timestamps[:],dtype=np.float64)
edge_matrix=go[:,None]+EDGES_REL[None,:]
```

iii. The notes explicitly state that NWB spikes/events are on an absolute session clock, so the converter should “subtract each trial’s `go_start_times` timestamp”, equivalently by histogramming absolute spikes with go-shifted edges.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 80 non-overlapping 50 ms bins covering -2.5 s to +1.5 s around the go cue. No extra temporal rebinning is applied after that.

ii. 
```python
BIN_S=0.050
OFF_START=-2.5
OFF_END=1.5
EDGES_REL=np.arange(OFF_START, OFF_END + BIN_S/2, BIN_S, dtype=np.float64)
CENTERS_REL=(EDGES_REL[:-1]+EDGES_REL[1:])/2
N_TIME=len(CENTERS_REL)
```

iii. The notes say the method-paper used 40 ms sliding windows, but the decoder task “overrides this with non-overlapping 50-ms bins from -2.5 to +1.5 s (80 bins).”

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `BehavioralEvents/sample_start_times` and `BehavioralEvents/go_start_times`. For each trial it uses the last sample/tone onset before the go cue.

ii. 
```python
sample=np.asarray(ev['sample_start_times'].timestamps[:],dtype=np.float64)
ix=np.searchsorted(sample,go[:n],side='right')-1
...
go=go[trial_idx]; tone=sample[ix[trial_idx]]
```

iii. The notes justify this by saying sample events can be replayed after early licks, so the correct tone for a trial is the “latest sample start before each go”.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes each bin center in absolute time and subtracts the chosen tone time, producing a continuous time-varying elapsed-time signal in seconds.

ii. 
```python
centers=g+CENTERS_REL
inp[0]=(centers-to).astype(np.float32)
```

iii. The notes say the decoder task explicitly wants a “continuous time-varying elapsed time” input, so each go-aligned bin center is converted into seconds since the tone.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same 80 go-aligned bin centers used for neural activity. Each trial’s `inp[0]` uses `centers = go + CENTERS_REL`, so the time input and neural activity share the same time axis.

ii. 
```python
CENTERS_REL=(EDGES_REL[:-1]+EDGES_REL[1:])/2
...
centers=g+CENTERS_REL
inp[0]=(centers-to).astype(np.float32)
```

iii. The notes describe “every retained trial uses a common go-aligned interval and 80 common 50-ms bins,” and list a sanity check comparing this input with independently reconstructed values.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the absolute `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times` event streams, not from the trial-table `photostim_onset`/`photostim_duration` fields.

ii. 
```python
pstart=np.asarray(ev['photostim_start_times'].timestamps[:],dtype=np.float64)
pstop=np.asarray(ev['photostim_stop_times'].timestamps[:],dtype=np.float64)
```

iii. The notes explicitly say “Event timestamps, rather than string-formatted trial photostimulation fields, are authoritative for temporal inputs” and justify this by noting that the input is time-varying and some stimulation falls outside the requested decoding window.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the AI marks a bin as 1 if the absolute bin center falls inside any half-open interval `[photostim_start, photostim_stop)`, otherwise 0. This yields a binary time series per trial.

ii. 
```python
if len(pstart):
    jj=np.searchsorted(pstart,centers,side='right')-1
    valid=jj>=0; stim=np.zeros(N_TIME,dtype=bool)
    stim[valid]=centers[valid] < pstop[jj[valid]]
    inp[1]=stim.astype(np.float32)
else:
    inp[1]=0
```

iii. The mapping table in the notes says photostimulation should be “Binary 1 when bin center is within any `[start, stop)` interval, else 0”, and a later sanity-check note says the independently reconstructed photostim input passed.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostimulation on the same absolute go-centered bin centers used for neural activity. It computes the absolute bin-center times for a trial and evaluates the event intervals directly at those times.

ii. 
```python
centers=g+CENTERS_REL
jj=np.searchsorted(pstart,centers,side='right')-1
stim[valid]=centers[valid] < pstop[jj[valid]]
```

iii. The notes say this was chosen because the event start/stop timestamps are already on the same absolute clock as go cues and spikes.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from two per-trial trial-table columns: `trial_instruction` and `outcome`. It does not use a stored choice label.

ii. 
```python
instruction=arrcol(tr,'trial_instruction').astype(str)[trial_idx]
outcome_s=arrcol(tr,'outcome').astype(str)[trial_idx]
```

```python
def trial_choice(instruction,outcome):
    if outcome=='ignore': return 2
    if outcome=='hit': return 0 if instruction=='left' else 1
    if outcome=='miss': return 1 if instruction=='left' else 0
```

iii. The notes say “Choice reconstructed from instruction/outcome agreed 100% with first post-go left/right lick” in spot checks, and therefore this deterministic mapping was adopted.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as `0=left`, `1=right`, `2=no lick`, using the mapping implemented in `trial_choice`. It then repeats the per-trial category across all 80 time bins.

ii. 
```python
out=np.empty((4,N_TIME),dtype=np.int64)
out[0]=trial_choice(instruction[k],outcome_s[k])
```

```python
'output_values':[
    ['left','right','no lick'],
    ['ignore','miss','hit'],
    ['no','yes'],
    ['<40th percentile','40th to 60th percentile','>60th percentile','not visible']
]
```

iii. The notes say static labels are repeated because the validator expects all output rows to share a common time axis.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI takes outcome directly from the NWB trials table `outcome` column.

ii. 
```python
outcome_s=arrcol(tr,'outcome').astype(str)[trial_idx]
```

iii. The notes describe outcome as a direct per-trial categorical label available in the NWB files.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, `hit -> 2`, and repeats the per-trial code across all 80 bins.

ii. 
```python
out[1]={'ignore':0,'miss':1,'hit':2}[outcome_s[k]]
```

iii. The notes list this as a direct label mapping and say the static outputs are repeated over time to satisfy the decoder/validator format.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The AI takes early lick directly from the NWB trials table `early_lick` column.

ii. 
```python
early_s=arrcol(tr,'early_lick').astype(str)[trial_idx]
```

iii. The notes describe early lick as one of the required decoder outputs that should be retained rather than used as a trial exclusion.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early -> 0`, `early -> 1`, and repeats the per-trial value across all 80 bins.

ii. 
```python
out[2]={'no early':0,'early':1}[early_s[k]]
```

iii. As with the other static outputs, the notes say the label is repeated across bins to satisfy the uniform `(n_output, n_timepoints)` format.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI uses a tongue-tracking time series from `BehavioralTimeSeries`, preferring `Camera0_side_TongueTracking` but falling back to the first available `*TongueTracking` stream. It takes column 1 as tongue y-position and column 2 as the tracking likelihood/confidence.

ii. 
```python
series=nwb.acquisition['BehavioralTimeSeries'].time_series
keys=sorted(k for k in series if 'TongueTracking' in k)
key='Camera0_side_TongueTracking' if 'Camera0_side_TongueTracking' in keys else keys[0]
ts=series[key]
times=np.asarray(ts.timestamps[:],dtype=np.float64)
data=np.asarray(ts.data[:],dtype=np.float32)
y=data[:,1]; likelihood=data[:,2]
```

iii. The notes say the intended source is `Camera0_side_TongueTracking` and describe it as the required y/likelihood stream for tongue output.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI marks frames visible when `likelihood >= 0.90`, computes session 40th and 60th percentiles from the visible raw-frame y values, then for each neural bin chooses the nearest camera frame to the bin center (only if within 10 ms and within video coverage). It does not average tongue y within 50 ms bins.

ii. 
```python
LIKELIHOOD_THRESHOLD=0.90
...
visible=np.isfinite(y)&np.isfinite(likelihood)&(likelihood>=LIKELIHOOD_THRESHOLD)
thresholds=tuple(np.percentile(y[visible],[40,60]).tolist()) if visible.any() else (np.nan,np.nan)
```

```python
q=nearest_indices(vt,centers)
covered=(centers>=vt[0])&(centers<=vt[-1])&(np.abs(vt[q]-centers)<=0.010)
vis=covered&np.isfinite(vy[q])&np.isfinite(vl[q])&(vl[q]>=LIKELIHOOD_THRESHOLD)&np.isfinite(p40)&np.isfinite(p60)
yy=vy[q]
tc[vis & (yy<p40)]=0
tc[vis & (yy>=p40)&(yy<=p60)]=1
tc[vis & (yy>p60)]=2
```

iii. The notes say the exact threshold was unclear in the supplied sources, that likelihood is strongly bimodal, and therefore “a conventional 0.9 threshold” was chosen. The mapping plan also explicitly says “Resample nearest video frame to each bin center”.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses four classes: `0` for y below the session 40th percentile, `1` for y from the 40th through 60th percentile, `2` for y above the 60th percentile, and `3` for low-confidence or uncovered bins (“not visible”).

ii. 
```python
tc=np.full(N_TIME,3,dtype=np.int64)
...
tc[vis & (yy<p40)]=0
tc[vis & (yy>=p40)&(yy<=p60)]=1
tc[vis & (yy>p60)]=2
out[3]=tc
```

```python
'output_values':[
    ...,
    ['<40th percentile','40th to 60th percentile','>60th percentile','not visible']
]
```

iii. The notes justify this as the task-required per-session percentile discretization, with class 3 used whenever the tongue is low-confidence or outside camera coverage.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output to the same go-centered bin centers as the neural data, but it does so by nearest-frame sampling rather than by averaging all frames inside each 50 ms neural bin.

ii. 
```python
centers=g+CENTERS_REL
...
q=nearest_indices(vt,centers)
covered=(centers>=vt[0])&(centers<=vt[-1])&(np.abs(vt[q]-centers)<=0.010)
```

iii. The notes explicitly say “Resample nearest video frame to each bin center” and treat missing coverage as the “not visible” class.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several kinds of imperfect data explicitly: it drops sessions with no classifier-good units, drops trials without a preceding sample or without membership in the common valid-trial intersection, removes all-zero neural trials after binning, and treats low-confidence or uncovered tongue bins as class 3. It raises an error if selected units have blank anatomy labels or if `obs_intervals` and `is_good_trials` have inconsistent shapes.

ii. 
```python
if units.size==0:
    return units, np.asarray([],dtype=int)
...
if structural_idx.size<2: return None
...
if obs.ndim!=2 or obs.shape[0]!=valid.size:
    raise ValueError(...)
...
if np.any(names==''): raise ValueError(...)
```

```python
neural_valid=np.any(rates>0,axis=(1,2))
...
tc=np.full(N_TIME,3,dtype=np.int64)
```

iii. The notes say missing or invalid neural data should be excluded rather than fabricated, while missing tongue visibility should be represented as the required explicit “not visible” class. The trajectory shows the all-zero neural trial filter was added only after the AI inspected validator warnings and raw NWB examples.

## 10-a. What are the most time-consuming steps of the code?

i. The AI’s code is dominated by NWB I/O, loading per-unit spike vectors, the per-unit `searchsorted` neural binning loop, loading tongue tracking arrays, and serializing the very large pickle output. Earlier versions also had a much slower per-unit/per-trial histogram loop.

ii. 
```python
spikes=[np.asarray(nwb.units['spike_times'][j],dtype=np.float64) for j in units]
for ui,sp in enumerate(spikes):
    rates[:,ui,:]=np.diff(np.searchsorted(sp,edge_matrix,side='left'),axis=1).astype(np.float32)/BIN_S
```

```python
data=np.asarray(ts.data[:],dtype=np.float32)
...
with open(args.outpicklefile,'wb') as f:
    pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
```

iii. The notes call the old per-unit/per-trial histograms a “runtime bottleneck” and say they were replaced with one vectorized `searchsorted` per unit, reducing sample runtime by about 9.8x.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain. `select_units_and_trials` still iterates over units; neural binning still iterates once per unit; and trial assembly still iterates once per retained trial to build inputs, outputs, and tongue categories. The tongue alignment also recomputes nearest-frame indices trial-by-trial.

ii. 
```python
for j in units:
    obs=np.asarray(nwb.units['obs_intervals'][j],dtype=np.float64)
    ...
```

```python
for ui,sp in enumerate(spikes):
    rates[:,ui,:]=...
```

```python
for k,(g,to) in enumerate(zip(go,tone)):
    ...
    jj=np.searchsorted(pstart,centers,side='right')-1
    ...
    q=nearest_indices(vt,centers)
```

iii. The notes mainly justify the neural binning optimization that was already done; they do not claim these remaining loops were fully vectorized. The trajectory shows the AI considered further vectorization where runtime mattered most.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats several operations per trial: it recomputes absolute bin centers, photostim interval membership, nearest tongue-frame lookup, and dictionary-based static output mappings. It also repeats the static choice/outcome/early-lick labels across all 80 bins for every trial.

ii. 
```python
for k,(g,to) in enumerate(zip(go,tone)):
    centers=g+CENTERS_REL
    ...
    jj=np.searchsorted(pstart,centers,side='right')-1
    ...
    out[1]={'ignore':0,'miss':1,'hit':2}[outcome_s[k]]
    out[2]={'no early':0,'early':1}[early_s[k]]
```

```python
out[0]=trial_choice(instruction[k],outcome_s[k])
out[1]=...
out[2]=...
```

iii. The notes explicitly justify one of these repetitions: static labels are intentionally repeated over time because the validator expects every output row to have the same time dimension.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is that the converter builds `tongue_visible` for every trial even when plotting is disabled, and only packages it into `plot_payload` for optional diagnostic plots. In normal conversion runs this visibility list is discarded. The code also carries plotting-specific payload assembly inside the main conversion path.

ii. 
```python
neural=[]; inputs=[]; outputs=[]
tongue_visible=[]
for k,(g,to) in enumerate(zip(go,tone)):
    ...
    vis=covered & ...
    ...
    neural.append(fr); inputs.append(inp); outputs.append(out); tongue_visible.append(vis)
```

```python
plot_payload=None
if make_plot:
    plot_payload=(go,tone,pstart,pstop,vt,vy,vl,p40,p60,neural,inputs,outputs,np.asarray(tongue_visible),info)
return dict(...,plot=plot_payload)
```

iii. The notes do not present this as a deliberate analysis decision; it appears to be incidental support for `--show-processing` diagnostics rather than part of the downstream dataset itself.
