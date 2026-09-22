# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter recursively finds every NWB file below `/app/data`, sorts the paths, and opens each session once with `h5py`. `--sample` truncates the list to two files; otherwise all files are processed.

ii.
```python
files=sorted(DATA_ROOT.rglob('*.nwb')); files=files[:2] if args.sample else files
for i,p in enumerate(files):
    r=convert_session(p,args.show_processing and len(results)<2)
```

iii. The notes identify 174 NWB files and one file per session. Direct HDF5 access was chosen to reconstruct the native NWB ragged arrays efficiently.

## 1-b. How are the data split into subjects?

i. The subject is parsed from each NWB filename, unique subject strings are sorted, and each session receives an index into that vocabulary.

ii.
```python
subject=re.search(r'sub-([^_]+)',path.name).group(1)
subjects=sorted({r[4] for r in results}); smap={x:i for i,x in enumerate(subjects)}
'subject_idx':np.array([smap[r[4]] for r in results],np.int64)
```

iii. Filenames consistently encode the NWB subject identifier; the resulting 28 subjects matched the source-data count.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session and contributes one element to each top-level session list. A file with no classifier-good units, or fewer than two retained trials, is skipped.

ii.
```python
with h5py.File(path,'r') as f:
    ...
    if not len(good): return None
...
'neural':[[x for x in r[0]] for r in results]
```

iii. The notes state that the release is organized as one NWB per session and that dropping the single uncurated session reproduces 173 analyzed sessions.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`. Go events are indexed by the retained trial indices, and the resulting trial-major arrays are converted to Python lists.

ii.
```python
inds=np.where(valid)[0]
go_all=be['go_start_times/timestamps'][()]; go=go_all[inds]
...
'neural':[[x for x in r[0]] for r in results]
```

iii. The trials table and one go cue per trial provide the native trial definition; observation intervals are mapped back to trial start times.

## 1-e. How are trials filtered based on quality controls?

i. For every classifier-good unit, the code maps that unit's ragged `obs_intervals` rows to trials, applies its `is_good_trials` flags, and intersects validity across all good units. It then removes trials whose entire four-second population window is zero and drops sessions with fewer than two trials. It does not explicitly remove `free_water` trials.

ii.
```python
valid=np.ones(len(go), bool)
for ui in good:
    ...
    uv[jj]=m
    valid &= uv
...
has_neural=np.any(neural != 0, axis=(1,2))
inds=inds[has_neural]
```

iii. The agent interpreted `is_good_trials` as manual probe-insertion validity and considered all-zero population windows missing neural segments. It retained early, ignore, stimulation, auto-water, and free-water trials when otherwise valid because required decoder categories should remain represented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values derive from the ragged `units/spike_times` arrays for units whose `classification` is `good`, aligned by go-cue timestamps.

ii.
```python
good=np.where(decode(u['classification'][()]) == 'good')[0]
flat=u['spike_times']; a,b=ragged_bounds(u['spike_times_index'][()])
```

iii. Raw spike times avoid applying an additional smoothing step to preprocessed firing rates and match the neural measurement available in NWB.

## 2-b. How is the `neural` data processed?

i. Spikes are assigned to trial and 50-ms bin, counted with `bincount`, converted to `float32`, and divided by 0.05 s to obtain Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
bi=np.floor((ss-starts[ti])/BIN_S).astype(np.int64)
code=ti[ok]*NT+bi[ok]
counts[:,col,:]=np.bincount(code,minlength=ntr*NT).reshape(ntr,NT)
return counts.astype(np.float32) / np.float32(BIN_S)
```

iii. The notes say this implements the requested raw-spike, non-overlapping 50-ms firing rates and independently matched `np.histogram` checks.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with the final `classification == 'good'` label are retained; a session with none is discarded. The code does not use `unit_quality` or individual metric thresholds.

ii.
```python
good=np.where(decode(u['classification'][()]) == 'good')[0]
if len(good)==0: return good, np.zeros(len(f['intervals/trials/id']), bool)
```

iii. The agent identified `classification` as the paper's final region-specific classifier verdict and reported 69,453 retained session-units, close to the paper's 69,943.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial window starts at `go - 2.5` and ends at `go + 1.5`; spikes are assigned using these absolute session-clock bounds.

ii.
```python
starts=go_valid+OFF_START; ends=go_valid+OFF_END
ti=np.searchsorted(starts, sp, side='right')-1
bi=np.floor((ss-starts[ti])/BIN_S).astype(np.int64)
```

iii. NWB spikes and events share a session-absolute clock, so adding relative offsets to the go cue is sufficient.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. There are 80 non-overlapping 50-ms bins spanning `[-2.5, 1.5)` seconds. Raw spikes are binned directly; no later rebinning occurs.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
EDGES_REL = np.arange(OFF_START, OFF_END + BIN_S/2, BIN_S)
NT = len(CENTERS_REL)
assert NT == 80
```

iii. This is the explicit decoder-task resolution, superseding the method paper's 40-ms sliding window.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `sample_start_times`, trial `start_time`, and each trial's go time. The final sample event between trial start and go is selected.

ii.
```python
starts=f['intervals/trials/start_time'][()]
sample=np.sort(f['acquisition/BehavioralEvents/sample_start_times/timestamps'][()])
out[k]=sample[hi-1]
```

iii. Early licking can replay the sample epoch, so the last eligible tone is the relevant onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Absolute neural-bin centers are formed and the chosen tone timestamp is subtracted from every center.

ii.
```python
abs_centers=go[:,None]+CENTERS_REL[None,:]
inp=np.stack([abs_centers-tone[:,None],stim_series(f,abs_centers)],axis=1)
```

iii. This directly produces the requested continuous seconds-from-tone variable without further transformation.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same go-relative bin centers as the neural firing rates.

ii.
```python
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
abs_centers=go[:,None]+CENTERS_REL[None,:]
```

iii. A shared 80-center grid makes input column `t` correspond to neural bin `t`.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses absolute `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, rather than the trials-table onset and duration fields.

ii.
```python
starts=be['photostim_start_times/timestamps'][()]
stops=be['photostim_stop_times/timestamps'][()]
```

iii. The notes treat these native timestamped events as the authoritative time-resolved stimulation stream and verified them against raw-data spot checks.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The code initializes zeros and ORs in every half-open stimulation interval, yielding 1 when a bin center is inside any interval and 0 otherwise.

ii.
```python
out=np.zeros(abs_centers.shape, dtype=bool)
for a,b in zip(starts,stops): out |= ((abs_centers>=a)&(abs_centers<b))
return out.astype(np.float32)
```

iii. A binary, time-varying series is required, and half-open intervals avoid double counting boundaries.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Stimulation membership is evaluated at the same absolute bin centers used for the go-aligned neural grid.

ii.
```python
abs_centers=go[:,None]+CENTERS_REL[None,:]
stim_series(f,abs_centers)
```

iii. All timestamps share the NWB session clock, so no clock correction is applied.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from trials-table `trial_instruction` and `outcome`.

ii.
```python
instruction=decode(trial['trial_instruction'][()])[inds]
outcome_s=decode(trial['outcome'][()])[inds]
```

iii. The actual side is implicit: hit means instructed side, miss means the opposite, and ignore means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Left, right, and no lick are coded 0, 1, and 2; the per-trial value is repeated across 80 bins.

ii.
```python
if o=='ignore': choice[k]=2
elif o=='hit': choice[k]=0 if ins=='left' else 1
elif o=='miss': choice[k]=1 if ins=='left' else 0
np.repeat(choice[:,None],NT,1)
```

iii. This reconstructs behavior from the available labels and provides a common time-shaped output representation.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from `intervals/trials/outcome`.

ii.
```python
outcome_s=decode(trial['outcome'][()])[inds]
```

iii. The native categories exactly match the requested output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to ignore=0, miss=1, hit=2 and repeated over all time bins.

ii.
```python
omap={'ignore':0,'miss':1,'hit':2}
outcome=np.array([omap[x] for x in outcome_s],np.int64)
np.repeat(outcome[:,None],NT,1)
```

iii. Fixed codes agree with `output_values`; repetition makes the trial-level label decoder-compatible.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `intervals/trials/early_lick`.

ii.
```python
early_s=decode(trial['early_lick'][()])[inds]
```

iii. The trials table explicitly stores the required binary label.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `early` is mapped to 1 and every other native value (`no early`) to 0, then repeated across 80 bins.

ii.
```python
early=np.array([1 if x=='early' else 0 for x in early_s],np.int64)
np.repeat(early[:,None],NT,1)
```

iii. The mapping supplies the requested no/yes categorical output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and the y and likelihood columns of `Camera0_side_TongueTracking`.

ii.
```python
tt=ts['timestamps'][()]; d=ts['data'][()]
y=d[:,1]; likelihood=d[:,2]
```

iii. This is the side-camera DeepLabCut tongue stream described in the source data.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Visible session frames are selected at likelihood at least 0.9; percentiles are computed from their raw y values. Both y and likelihood are linearly interpolated at each neural-bin center, after which low-likelihood centers become not visible.

ii.
```python
visible=likelihood>=0.9
p40,p60=np.percentile(y[visible],[40,60])
yi=np.interp(flat,tt,y,left=np.nan,right=np.nan)
li=np.interp(flat,tt,likelihood,left=0,right=0)
```

iii. The agent viewed interpolation as following the reference marker alignment and selected 0.9 because likelihood is strongly bimodal. It interpreted “over the session” as all visible session frames.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. At interpolated centers with likelihood at least 0.9: y below p40 is 0, p40 through p60 is 1, and above p60 is 2. Other centers are 3 (not visible).

ii.
```python
cat=np.full(abs_centers.shape,3,dtype=np.int64)
v=(li>=0.9)&np.isfinite(yi)
cat[v & (yi<p40)]=0
cat[v & (yi>=p40) & (yi<=p60)]=1
cat[v & (yi>p60)]=2
```

iii. The boundary convention follows the requested category wording, and class 3 explicitly represents missing/invisible tongue measurements.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Raw y and likelihood are linearly interpolated at the absolute center of each neural time bin.

ii.
```python
flat=abs_centers.ravel()
yi=np.interp(flat,tt,y,left=np.nan,right=np.nan).reshape(abs_centers.shape)
```

iii. Camera and electrophysiology timestamps share a clock; center interpolation was chosen to produce one tongue value per neural bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions without good units and sessions falling below two valid trials are skipped. Missing observation/manual validity removes trials; fully silent population windows are treated as missing neural segments and removed. Tongue values outside camera coverage or below confidence receive category 3. Unexpected interval mappings, labels, or insufficient visible tongue data raise errors.

ii.
```python
if not len(good): return None
if len(inds)<2: return None
has_neural=np.any(neural != 0, axis=(1,2))
cat=np.full(abs_centers.shape,3,dtype=np.int64)
```

iii. The notes distinguish unavailable neural data, which should be excluded rather than fabricated as zeros, from legitimately invisible tongue data, which has an explicit output class.

## 10-a. What are the most time-consuming steps of the code?

i. The agent identifies loading large spike/video arrays, assigning spikes for every retained neuron, accumulating the roughly 11-GiB result, and final pickle serialization as the main costs.

ii.
```python
for col,ui in enumerate(units):
    sp=np.asarray(flat[a[ui]:b[ui]])
    ...
with open(args.outpicklefile,'wb') as fh: pickle.dump(data,fh,...)
```

iii. Conversion timings scaled with unit count; the notes report generally 0.2–2 seconds per session before serialization and emphasize single-open I/O and preallocation.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit spike loop and per-stimulation-interval loop remain. The choice-label loop could also be replaced by vectorized boolean selection. Spike assignment is already vectorized over trials within each unit.

ii.
```python
for col,ui in enumerate(units): ...
for a,b in zip(starts,stops): ...
for k,(ins,o) in enumerate(zip(instruction,outcome_s)): ...
```

iii. Ragged spike trains make a unit loop natural; the expensive trial dimension was vectorized with `searchsorted`/`bincount`, avoiding a neuron-by-trial histogram loop.

## 10-c. What processing does the code repeat multiple times?

i. Core conversion quantities are computed once per session. It repeatedly maps observation intervals for every good unit and repeatedly ORs stimulation intervals across the full trial-by-time grid. Optional plots reread already computed arrays but do not redo conversion.

ii.
```python
for ui in good:
    ... jj=map_observed_trials(trial_st,z)
for a,b in zip(starts,stops): out |= ...
```

iii. Per-unit validity was intentional because the agent chose to intersect manual validity across all retained units; module-level bin grids and one file open per session avoid larger recomputation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Ordinary full conversion performs little discarded work. With `--show-processing`, it constructs and saves diagnostic figures that are not used by the dataset or decoder. It also computes `tone` again after silent-trial filtering, but that recomputed variable is not subsequently used.

ii.
```python
tone=tone[has_neural]
...
if make_plot:
    import matplotlib.pyplot as plt
    ... fig.savefig(...)
```

iii. The plots were deliberate sanity checks. The post-filter `tone` slice is harmless dead work; all major converted arrays and metadata are retained.
