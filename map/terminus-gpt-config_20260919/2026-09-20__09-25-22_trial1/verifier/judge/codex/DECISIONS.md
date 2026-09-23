# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all `.nwb` files recursively under `/app/data`, treats each file as one session, and reads raw HDF5 datasets directly with `h5py` instead of `pynwb`. Within each session it reads trials from `intervals/trials`, units from `units`, events from `acquisition/BehavioralEvents`, and video from `acquisition/BehavioralTimeSeries`.

ii.
```python
DATA_ROOT = Path('/app/data')
...
with h5py.File(path,'r') as f:
```

```python
files=sorted(DATA_ROOT.rglob('*.nwb')); files=files[:2] if args.sample else files
```

```python
u=f['units']
trial=f['intervals/trials']
be=f['acquisition/BehavioralEvents']
ts=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
```

iii. `CONVERSION_NOTES.md` says the source is 174 NWB/HDF5 files organized by subject/session and that the implementation uses “direct h5py access” and opens each NWB once per session.

## 1-b. How are the data split into subjects?

i. Subjects are split by parsing the subject id from each NWB filename with a regex like `sub-440956... -> 440956`. Unique parsed ids are sorted to form `subjects`, and each session gets a `subject_idx`.

ii.
```python
subject=re.search(r'sub-([^_]+)',path.name).group(1)
```

```python
subjects=sorted({r[4] for r in results}); smap={x:i for i,x in enumerate(subjects)}
...
'subjects':subjects,
'subject_idx':np.array([smap[r[4]] for r in results],np.int64),
```

iii. The notes describe 28 subject directories/files under `sub-<id>` and the final code follows that file naming convention. There is no separate justification beyond using the published directory layout.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The output session order is the sorted recursive file list.

ii.
```python
files=sorted(DATA_ROOT.rglob('*.nwb')); files=files[:2] if args.sample else files
```

```python
for i,p in enumerate(files):
    r=convert_session(p,args.show_processing and len(results)<2)
```

iii. The notes state the dataset contains 174 NWB files “organized in subject/session directories,” so the file boundary is taken as the session boundary.

## 1-d. How are the data split into trials?

i. Trials are indexed primarily from the go-cue event stream and a per-trial validity mask. The valid trial indices are `np.where(valid)[0]`, and those indices are then used consistently to subset go cues and the trials table.

ii.
```python
go=f['acquisition/BehavioralEvents/go_start_times/timestamps'][()]
...
valid=np.ones(len(go), bool)
```

```python
inds=np.where(valid)[0]
...
go_all=be['go_start_times/timestamps'][()]; go=go_all[inds]
...
instruction=decode(trial['trial_instruction'][()])[inds]
outcome_s=decode(trial['outcome'][()])[inds]
```

iii. The AI did not document a separate trial-boundary derivation. Its notes and trajectory show that it treated one go cue as one trial and used shared indexing between the go-cue timestamps and `intervals/trials`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if every retained good unit marks them as observed and manually good. The code maps each unit’s ragged `obs_intervals` rows onto behavioral trials, intersects the corresponding `is_good_trials` masks across all retained units, and then drops any trial whose entire retained neural population is all-zero after spike binning. Sessions with fewer than 2 remaining trials are dropped.

ii.
```python
flat=u['obs_intervals'][()]; a,b=ragged_bounds(u['obs_intervals_index'][()]); manual=u['is_good_trials'][()]
valid=np.ones(len(go), bool)
for ui in good:
    z=flat[a[ui]:b[ui]]; m=manual[ui]
    if len(z)!=len(m): raise ValueError('Validity/interval length mismatch')
    jj=map_observed_trials(trial_st,z)
    uv=np.zeros(len(go), bool)
    uv[jj]=m
    valid &= uv
```

```python
has_neural=np.any(neural != 0, axis=(1,2))
inds=inds[has_neural]
...
if len(inds)<2: return None
```

iii. The trajectory says the AI chose this because the target format requires a fixed neuron set per session, so it considered intersecting unit-valid trials “the only lossless option.” After an initial mistake using interval stop times, it revised the rule to “mapped observation-row presence and manual good flag for every retained unit,” and treated all-zero neural windows as missing neural segments.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times`, restricted to units whose `classification` is `'good'`, and aligned using go-cue timestamps from `acquisition/BehavioralEvents/go_start_times/timestamps`.

ii.
```python
u=f['units']; good=np.where(decode(u['classification'][()]) == 'good')[0]
```

```python
u=f['units']; flat=u['spike_times']; a,b=ragged_bounds(u['spike_times_index'][()])
```

```python
go_all=be['go_start_times/timestamps'][()]; go=go_all[inds]
```

iii. The notes say the dataset is extracellular electrophysiology, so raw spike times are the neural source, and the classifier-good units are the paper-like curated subset.

## 2-b. How is the `neural` data processed?

i. For each retained unit, spikes are assigned into 80 non-overlapping 50 ms bins from -2.5 s to +1.5 s around go cue. Counts are converted to firing rates in Hz by dividing by `0.05`.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
EDGES_REL = np.arange(OFF_START, OFF_END + BIN_S/2, BIN_S, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
NT = len(CENTERS_REL)
assert NT == 80
```

```python
counts=np.zeros((ntr, nn, NT), dtype=np.uint16)
starts=go_valid+OFF_START; ends=go_valid+OFF_END
...
bi=np.floor((ss-starts[ti])/BIN_S).astype(np.int64)
...
return counts.astype(np.float32).transpose(0,1,2) / np.float32(BIN_S)
```

iii. `CONVERSION_NOTES.md` says the AI intentionally used raw spikes and task-required 50 ms bins, with vectorized spike-to-trial/bin assignment per neuron.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level QC filter is `units/classification == 'good'`. If a session has zero such units, the session is dropped.

ii.
```python
u=f['units']; good=np.where(decode(u['classification'][()]) == 'good')[0]
if len(good)==0: return good, np.zeros(len(f['intervals/trials/id']), bool)
```

```python
good,valid=curate(f)
if not len(good): return None
```

iii. The notes explicitly justify this as the final region-specific classifier used in the papers, and reject `unit_quality` as an older, more permissive label.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue. Each trial’s absolute time grid is `go + CENTERS_REL` for bin centers and `go + [OFF_START, OFF_END]` for spike binning.

ii.
```python
go_all=be['go_start_times/timestamps'][()]; go=go_all[inds]
abs_centers=go[:,None]+CENTERS_REL[None,:]
```

```python
starts=go_valid+OFF_START; ends=go_valid+OFF_END
bi=np.floor((ss-starts[ti])/BIN_S).astype(np.int64)
```

iii. The notes repeatedly describe the conversion as “go-aligned” and treat all source timestamps as already lying on the same session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50 ms. Spike times are rebinned into 80 non-overlapping bins across the 4 s window; there is no additional smoothing or sliding-window stride.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
...
assert NT == 80
```

iii. The notes say the task specification overrides the paper’s 40 ms / 3.4 ms firing-rate representation, so the AI uses the requested 50 ms bins instead.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times` together with `intervals/trials/start_time` and `go_start_times`. For each retained trial, the AI selects the final sample/tone onset between that trial’s start and its go cue.

ii.
```python
starts=f['intervals/trials/start_time'][()]
sample=np.sort(f['acquisition/BehavioralEvents/sample_start_times/timestamps'][()])
...
lo=np.searchsorted(sample, starts[i]-1e-8, 'left'); hi=np.searchsorted(sample, go[i]+1e-8, 'right')
...
out[k]=sample[hi-1]
```

iii. The trajectory says `sample_start_times` can outnumber trials because early licks replay the sample/delay epoch, so the AI chose the last sample before go, bounded to the trial itself, to capture the final replay-aware tone onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After finding the relevant tone onset per trial, the code computes time from tone at each bin center as `absolute_bin_center - tone_onset`.

ii.
```python
abs_centers=go[:,None]+CENTERS_REL[None,:]
tone=final_tone_onsets(f,inds,go_all)
inp=np.stack([abs_centers-tone[:,None],stim_series(f,abs_centers)],axis=1).astype(np.float32)
```

iii. The notes describe this variable as “continuous seconds from the final tone/sample onset before go cue.”

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on exactly the same go-aligned bin centers used to represent neural activity.

ii.
```python
abs_centers=go[:,None]+CENTERS_REL[None,:]
inp=np.stack([abs_centers-tone[:,None],stim_series(f,abs_centers)],axis=1).astype(np.float32)
```

iii. The AI’s notes and plots describe the input and neural streams as sharing the same -2.5 s to +1.5 s go-aligned axis.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the event streams `BehavioralEvents/photostim_start_times/timestamps` and `BehavioralEvents/photostim_stop_times/timestamps`, not from the trials-table onset/duration strings.

ii.
```python
be=f['acquisition/BehavioralEvents']; out=np.zeros(abs_centers.shape, dtype=bool)
starts=be['photostim_start_times/timestamps'][()]; stops=be['photostim_stop_times/timestamps'][()]
```

iii. The notes say the AI chose “native BehavioralEvents” for the stimulation input and wanted a time-varying signal directly on the event clock.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The code creates a binary time series on the neural bin centers. A bin is 1 if its absolute center falls within any photostim start/stop interval and 0 otherwise.

ii.
```python
for a,b in zip(starts,stops): out |= ((abs_centers>=a)&(abs_centers<b))
return out.astype(np.float32)
```

iii. The trajectory and notes describe photostimulation as a required time-varying decoder input rather than a per-trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned by comparing the absolute photostim intervals against the same absolute go-aligned bin centers used for the neural data.

ii.
```python
abs_centers=go[:,None]+CENTERS_REL[None,:]
...
for a,b in zip(starts,stops): out |= ((abs_centers>=a)&(abs_centers<b))
```

iii. The AI’s justification was that all streams are timestamped on a shared session clock, so stimulation can be aligned directly on that clock without extra offsets.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `intervals/trials/trial_instruction` and `intervals/trials/outcome`.

ii.
```python
instruction=decode(trial['trial_instruction'][()])[inds]
outcome_s=decode(trial['outcome'][()])[inds]
```

iii. The notes state that there is no direct choice column, so actual lick direction is inferred from instructed side plus whether the trial was hit, miss, or ignore.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as 0 = left, 1 = right, 2 = no lick. Hits map to the instructed side, misses to the opposite side, and ignores to no lick. The per-trial label is repeated across all 80 bins.

ii.
```python
choice=np.empty(len(inds),np.int64)
for k,(ins,o) in enumerate(zip(instruction,outcome_s)):
    if o=='ignore': choice[k]=2
    elif o=='hit': choice[k]=0 if ins=='left' else 1
    elif o=='miss': choice[k]=1 if ins=='left' else 0
```

```python
output=np.stack([np.repeat(choice[:,None],NT,1), ...],axis=1)
```

iii. The AI’s notes say trial-level outputs are repeated through time so all outputs share one `4 x 80` array shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from `intervals/trials/outcome`.

ii.
```python
outcome_s=decode(trial['outcome'][()])[inds]
```

iii. The notes describe outcome as a direct categorical mapping from the trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped to `ignore=0`, `miss=1`, `hit=2` and then repeated across all 80 bins.

ii.
```python
omap={'ignore':0,'miss':1,'hit':2}; outcome=np.array([omap[x] for x in outcome_s],np.int64)
...
output=np.stack([..., np.repeat(outcome[:,None],NT,1), ...],axis=1)
```

iii. The notes say outcome is a per-trial output, so it is repeated through time for compatibility with the decoder format.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from `intervals/trials/early_lick`.

ii.
```python
early_s=decode(trial['early_lick'][()])[inds]
```

iii. The notes treat early lick as a required retained label, even though some reference analyses excluded those trials.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The strings are converted to `1` for `'early'` and `0` otherwise, then repeated across all 80 bins.

ii.
```python
early=np.array([1 if x=='early' else 0 for x in early_s],np.int64)
...
output=np.stack([..., np.repeat(early[:,None],NT,1), ...],axis=1)
```

iii. The AI used a direct binary mapping because the trials table already stores the categorical early-lick label.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using its timestamps, `data[:,1]` as y-position, and `data[:,2]` as tracking likelihood.

ii.
```python
ts=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
tt=ts['timestamps'][()]; d=ts['data'][()]; y=d[:,1]; likelihood=d[:,2]
```

iii. The notes identify side-camera tongue tracking as the source and say visibility must be inferred from the likelihood values because coordinates themselves remain finite.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI marks frames as visible when `likelihood >= 0.9`, computes session-wide 40th and 60th percentiles from all visible raw-frame y-values, linearly interpolates both y and likelihood to the neural bin centers, and classifies each bin-center sample into four classes.

ii.
```python
visible=likelihood>=0.9
if visible.sum()<2: raise ValueError('Insufficient visible tongue frames')
p40,p60=np.percentile(y[visible],[40,60])
```

```python
flat=abs_centers.ravel(); yi=np.interp(flat,tt,y,left=np.nan,right=np.nan).reshape(abs_centers.shape)
li=np.interp(flat,tt,likelihood,left=0,right=0).reshape(abs_centers.shape)
cat=np.full(abs_centers.shape,3,dtype=np.int64)
v=(li>=0.9)&np.isfinite(yi)
```

iii. The trajectory says the AI chose 0.9 because tongue likelihood is strongly bimodal and a “conventional DeepLabCut threshold” needed explicit justification. The notes say percentiles should be over the session and that interpolation follows the reference alignment style for marker trajectories.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Thresholding uses two session-wide cutoffs `p40` and `p60` computed from visible raw-frame y-values. Each interpolated bin-center sample is assigned 0 if `< p40`, 1 if `>= p40 and <= p60`, 2 if `> p60`, and 3 if not visible.

ii.
```python
p40,p60=np.percentile(y[visible],[40,60])
...
cat=np.full(abs_centers.shape,3,dtype=np.int64)
v=(li>=0.9)&np.isfinite(yi)
cat[v & (yi<p40)]=0; cat[v & (yi>=p40) & (yi<=p60)]=1; cat[v & (yi>p60)]=2
```

iii. The notes justify session-wide percentiles from the instruction text and justify the visibility threshold from the bimodal likelihood distribution.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y-position is aligned by evaluating the tongue signal at the same absolute go-aligned bin centers used for neural data, via linear interpolation of y and likelihood.

ii.
```python
abs_centers=go[:,None]+CENTERS_REL[None,:]
...
flat=abs_centers.ravel(); yi=np.interp(flat,tt,y,left=np.nan,right=np.nan).reshape(abs_centers.shape)
li=np.interp(flat,tt,likelihood,left=0,right=0).reshape(abs_centers.shape)
```

iii. The notes and trajectory state that camera timestamps share the session clock with spikes/events, so interpolation to the neural time basis was the AI’s chosen alignment strategy.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or malformed data by dropping sessions with no good units, raising errors when ragged validity arrays do not match, dropping trials with all-zero neural populations after binning, and encoding low-confidence tongue bins as class 3 (`not visible`) rather than imputing a y-position.

ii.
```python
u=f['units']; good=np.where(decode(u['classification'][()]) == 'good')[0]
if len(good)==0: return good, np.zeros(len(f['intervals/trials/id']), bool)
```

```python
if len(z)!=len(m): raise ValueError('Validity/interval length mismatch')
...
if hi<=lo: raise ValueError(f'No sample onset for trial {i}')
...
if visible.sum()<2: raise ValueError('Insufficient visible tongue frames')
```

```python
has_neural=np.any(neural != 0, axis=(1,2))
...
cat=np.full(abs_centers.shape,3,dtype=np.int64)
```

iii. The notes justify dropping all-zero neural windows as missing neural segments and treating low-confidence tongue as “not visible.” The trajectory also shows the AI explicitly corrected an earlier mistaken trial-coverage rule after verification exposed missing-neural artifacts.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work is iterating over all good units to intersect trial validity and to bin spikes, plus reading the large spike-time and video arrays from each NWB file.

ii.
```python
for ui in good:
    z=flat[a[ui]:b[ui]]; m=manual[ui]
    ...
    valid &= uv
```

```python
for col,ui in enumerate(units):
    sp=np.asarray(flat[a[ui]:b[ui]])
    ...
    if len(code): counts[:,col,:]=np.bincount(code,minlength=ntr*NT).reshape(ntr,NT)
```

iii. `CONVERSION_NOTES.md` says the AI optimized a naïve triple loop away, and specifically calls out “vectorized spike-to-trial/bin assignment per neuron,” single file opens, and preallocation as the key runtime choices.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain vectorizable: the per-unit `curate` loop, the per-unit spike loop in `spike_rates`, the per-trial loop in `final_tone_onsets`, the per-event photostim loop in `stim_series`, and the small Python loop used to derive `choice`.

ii.
```python
for ui in good:
    ...
```

```python
for col,ui in enumerate(units):
    ...
```

```python
for k,i in enumerate(trial_indices):
    ...
```

```python
for a,b in zip(starts,stops): out |= ((abs_centers>=a)&(abs_centers<b))
```

iii. The AI’s notes only explicitly justify vectorizing the spike assignment “within each neuron”; the remaining loops are left in straightforward Python form.

## 10-c. What processing does the code repeat multiple times?

i. The code makes multiple passes over the per-unit ragged structures: once to intersect trial validity from `obs_intervals/is_good_trials`, and again to read spike times for firing-rate computation. It also interpolates tongue y and tongue likelihood separately over the same flattened bin-center grid.

ii.
```python
flat=u['obs_intervals'][()]; a,b=ragged_bounds(u['obs_intervals_index'][()]); manual=u['is_good_trials'][()]
for ui in good:
    ...
```

```python
u=f['units']; flat=u['spike_times']; a,b=ragged_bounds(u['spike_times_index'][()])
for col,ui in enumerate(units):
    ...
```

```python
yi=np.interp(flat,tt,y,left=np.nan,right=np.nan).reshape(abs_centers.shape)
li=np.interp(flat,tt,likelihood,left=0,right=0).reshape(abs_centers.shape)
```

iii. The AI did not describe this as a problem in its notes; it mainly emphasized avoiding a slower neuron x trial x spike histogram implementation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some extra bookkeeping and temporary computation that is not needed by downstream decoder training: it keeps per-session diagnostics like `source_trial_indices`, `tongue_p40`, and `tongue_p60` in `session_info`, computes filtered `tone` after neural filtering even though that filtered `tone` is not later consumed, and optionally produces processing plots.

ii.
```python
tone=tone[has_neural]; inp=inp[has_neural]; neural=neural[has_neural]
```

```python
info={'file':path.name,'source_trial_indices':inds.tolist(),'n_source_trials':len(valid),'n_retained_trials':len(inds),'n_neurons':len(good),'tongue_p40':float(p40),'tongue_p60':float(p60),'seconds':time.time()-t0}
```

```python
if make_plot:
    import matplotlib.pyplot as plt
    ...
```

iii. The notes frame these mostly as debugging, documentation, and validation aids rather than core decoder inputs/outputs.
