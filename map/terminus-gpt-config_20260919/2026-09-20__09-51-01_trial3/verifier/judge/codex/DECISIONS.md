# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively finds every NWB file under `/app/data`, sorts the paths, first screens each file for a usable custom `units/classification` column, and processes every eligible file once with `h5py`. It reads subject metadata, the trials table, units, behavioral events, and tongue tracking directly from NWB HDF5 groups. One all-NaN/numeric-classification file is excluded, leaving 173 sessions.

ii.
```python
for p in sorted(DATA_ROOT.rglob('*.nwb')):
    with h5py.File(p, 'r') as f:
        d=f['units/classification']
        if d.dtype.kind in 'OSU': kept.append(p)
...
for i,p in enumerate(files):
    sess=process_session(p, ...)
```

iii. The notes justify NWB as the synchronized raw source and the custom classifier as the paper-compatible QC output. They report that 174 files exist but one lacks classifier output, consistent with the paper's 173 analyzed sessions.

## 1-b. How are the data split into subjects?

i. Each session's subject is read from `general/subject/subject_id`, with the `sub-*` directory name as a fallback. Unique IDs are sorted and each session receives an integer index into that list.

ii.
```python
subject=(f['general/subject/subject_id'].asstr()[()]
         if f['general/subject/subject_id'].dtype.kind in 'OSU'
         else path.parent.name.replace('sub-',''))
...
subjects=sorted(set(x['subject'] for x in sessions))
'subject_idx':np.asarray([smap[x['subject']] for x in sessions],dtype=np.int32)
```

iii. The agent treats the NWB subject field as authoritative and validates that this produces the expected 28 subjects.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Output session order is sorted path order, and file stems/source paths are retained in metadata.

ii.
```python
for p in sorted(DATA_ROOT.rglob('*.nwb')):
...
'session_id':path.stem, 'source_file':str(path)
```

iii. The notes state that the distributed layout already defines sessions and that excluding the sole file without classifier labels reconciles 174 source files with 173 paper-compatible sessions.

## 1-d. How are the data split into trials?

i. Rows of `intervals/trials` define behavioral trials, and `go_start_times` must contain one event per row. Because ephys covers only some behavioral trials in some files, the first unit's `obs_intervals` starts are matched to trial `start_time`; only mapped trials with video coverage are initially retained. After binning, wholly zero neural windows are also removed.

ii.
```python
trials=f['intervals/trials']; nt=len(trials['id'])
go=f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
if len(go)!=nt: raise ValueError(...)
ref_intervals=obs[obs_start[0]:obs_end[0]]
recorded_idx=np.searchsorted(starts,ref_intervals[:,0])
recorded_mask=np.zeros(nt,dtype=bool); recorded_mask[recorded_idx]=True
keep_trial=recorded_mask & video_mask
```

iii. The trajectory documents discovery that `is_good_trials` columns correspond to recorded observation intervals, not necessarily every behavioral row. Matching observation-interval starts to trial starts fixed initially all-zero, unrecorded trials.

## 1-e. How are trials filtered based on quality controls?

i. The agent retains trials represented in `obs_intervals` and having full requested-window coverage within the overall tongue timestamp range. It then removes any retained trial whose complete neural matrix is zero. It keeps early-lick, ignore, photostimulation, and free-water trial types when usable, and requires at least two trials per session. Although it computes `neural_window_mask`, that mask is only reported and is not applied.

ii.
```python
video_mask=(go+OFF_START>=video_t[0]) & (go+OFF_END<=video_t[-1])
keep_trial=recorded_mask & video_mask
...
neural_nonzero=np.any(neural_cube!=0,axis=(1,2))
kept=kept[neural_nonzero]
```

iii. The notes argue that required output classes prevent copying paper exclusions for early/ignore/stim trials; absent video cannot legitimately be called “not visible,” and completely zero neural windows provide no decoder input. The trajectory shows the agent rejected a stricter full-window `obs_intervals` filter because it disproportionately removed short ignore trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times` and `spike_times_index`, using go-cue timestamps to define trial bins. Unit selection also uses `units/classification` and `units/is_good_trials`.

ii.
```python
spikes=f['units/spike_times']; ends=f['units/spike_times_index'][:]
classification=decode_strings(f['units/classification'])
stable=good_trials.all(axis=1)
unit_ids=np.flatnonzero((classification=='good') & stable)
```

iii. The agent identifies raw sorted spikes as the appropriate neural source and the custom classifier as the QC field used by the papers.

## 2-b. How is the `neural` data processed?

i. For each selected unit, the code uses `searchsorted(..., side='left')` at every absolute bin edge, differences cumulative positions to obtain half-open-bin spike counts, divides by 0.05 s to obtain Hz, and stores float32 arrays. No smoothing, normalization, or baseline subtraction is used.

ii.
```python
cumulative=np.searchsorted(st, flat, side='left').reshape(edges.shape)
counts[j]=np.diff(cumulative, axis=1)/BIN_S
return counts.transpose(1,0,2)
```

iii. The notes say this matches the reference half-open spike-count/rate convention while using the task-required bin width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == 'good'` and must have `is_good_trials` true for every recorded observation interval in the session. A session must retain at least one such unit. No 2-Hz firing-rate filter is applied.

ii.
```python
stable=good_trials.all(axis=1)
unit_ids=np.flatnonzero((classification=='good') & stable)
if len(unit_ids)==0: raise ValueError(...)
```

iii. The agent calls this a conservative fixed-neuron-set rule that avoids treating trial-invalid unit periods as zero and preserves the target's constant neuron dimension. It rejects `unit_quality` and the method-paper's analysis-specific 2-Hz cutoff.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute go-cue time is added to the common relative edge grid from -2.5 to +1.5 s. Spikes and behavioral events are assumed to share the NWB session clock.

ii.
```python
edges=go_kept[:,None]+REL_EDGES[None,:]
flat=edges.ravel()
```

iii. The notes justify direct timestamp alignment because all NWB streams are synchronized in session-absolute seconds.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50-ms bins over four seconds. Raw spike times are newly binned at that resolution; there is no subsequent rebinning.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
N_BINS = int(round((OFF_END-OFF_START)/BIN_S))
REL_EDGES = OFF_START + np.arange(N_BINS+1)*BIN_S
```

iii. This directly follows the decoder instructions, superseding the different width/stride used in the method paper.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times`, `go_start_times`, trial start times, and the common bin centers. The last sample onset before go is selected.

ii.
```python
sample=f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
pos=np.searchsorted(sample, go, side='left')-1
tone=sample[pos]
```

iii. The agent notes that early licking can replay sample/delay epochs, so the most recent pre-go tone represents the epoch associated with the final go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Each go-aligned bin center is converted to absolute time and the selected tone onset is subtracted, yielding continuous elapsed seconds.

ii.
```python
centers=go_k[:,None]+REL_CENTERS[None,:]
time_from_tone=(centers-tone_all[kept,None]).astype(np.float32)
```

iii. The notes interpret the requested variable literally as continuous time since tone, rather than an onset pulse.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the same 80 bin centers whose edges define the neural firing-rate bins.

ii.
```python
REL_CENTERS = (REL_EDGES[:-1]+REL_EDGES[1:])/2
centers=go_k[:,None]+REL_CENTERS[None,:]
```

iii. The shared go-relative grid guarantees one input value per neural time bin.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from absolute `BehavioralEvents/photostim_start_times` and `photostim_stop_times` timestamps.

ii.
```python
on=ev['photostim_start_times/timestamps'][:]
off=ev['photostim_stop_times/timestamps'][:]
```

iii. The agent prefers direct event timestamps over parsing string-valued trial-table onset fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A Boolean series is initialized false and set true wherever an absolute bin center lies in any half-open `[start, stop)` laser interval, then cast to float32.

ii.
```python
result=np.zeros(absolute_centers.shape, dtype=bool)
for a,b in zip(on,off):
    result |= ((absolute_centers>=a)&(absolute_centers<b))
return result.astype(np.float32)
```

iii. This creates the required time-varying binary input and naturally leaves unstimulated trials at zero.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is sampled at the same absolute, go-aligned bin centers used for every other time series.

ii.
```python
centers=go_k[:,None]+REL_CENTERS[None,:]
stim=photostim_series(f,centers)
```

iii. Both laser events and go cues use the session-absolute NWB clock.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from the trial-table `outcome` and `trial_instruction` columns: ignore means no lick, hit means the instructed side, and miss means the opposite side.

ii.
```python
choice=np.where(outcome=='ignore',2,np.where(outcome=='hit',
       np.where(instruction=='left',0,1),
       np.where(instruction=='left',1,0))).astype(np.int8)
```

iii. The notes say curated trial labels were more reliable than reconstructing choice from raw lick-event timing, particularly for free-water edge cases.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choices are encoded as 0 left, 1 right, and 2 no lick, then repeated over all 80 time bins.

ii.
```python
output_cube[:,0,:]=choice[:,None]
'output_values':[['left','right','no lick'], ...]
```

iii. Repetition allows per-trial and time-varying outputs to share a single `(4, 80)` trial matrix.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from `intervals/trials/outcome`.

ii.
```python
outcome=decode_strings(trials['outcome'])[kept]
```

iii. The source already contains precisely the requested ignore/miss/hit categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to 0 ignore, 1 miss, and 2 hit, and the value is repeated across time.

ii.
```python
outcome_code=np.array([{'ignore':0,'miss':1,'hit':2}[x] for x in outcome],dtype=np.int8)
output_cube[:,1,:]=outcome_code[:,None]
```

iii. The mapping follows the specified output order; temporal repetition is a formatting choice.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `intervals/trials/early_lick`.

ii.
```python
early=decode_strings(trials['early_lick'])[kept]
```

iii. The trials table already explicitly flags early licking.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and `early` to 1, then the code repeats the trial label over all bins.

ii.
```python
early_code=np.array([{'no early':0,'early':1}[x] for x in early],dtype=np.int8)
output_cube[:,2,:]=early_code[:,None]
```

iii. This follows the requested no/yes category order.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking/data` columns 1 (y) and 2 (tracking likelihood), plus its timestamps and the go-aligned bin centers.

ii.
```python
tg=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
video_t=tg['timestamps'][:]
tongue=np.asarray(tg['data'][:],dtype=np.float64)
```

iii. The notes identify the side-camera DLC series as the dataset's tongue measurement.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The agent calls a frame visible when likelihood is at least 0.9, computes the session 40th/60th y percentiles over all visible raw frames, and selects the single nearest camera frame to every neural bin center. Invisible selected frames become class 3; visible y values are categorized using the percentiles. It does not average frames within 50-ms bins.

ii.
```python
visible_session=tongue[:,2]>=VISIBILITY_THRESHOLD
q40,q60=np.percentile(tongue[visible_session,1],[40,60])
vi=nearest_indices(video_t,centers.ravel()).reshape(centers.shape)
y=tongue[vi,1]; likelihood=tongue[vi,2]
```

iii. The notes justify 0.9 as a conservative DLC threshold on a strongly bimodal likelihood distribution and nearest-frame sampling because ~300-Hz video lies within about 1.7 ms of a bin center and avoids smoothing across visibility transitions.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. For visible frames: class 0 is `y < q40`, class 1 is `q40 <= y <= q60`, and class 2 is `y > q60`; likelihood below 0.9 is class 3, not visible.

ii.
```python
tongue_code=np.full(y.shape,3,dtype=np.int8)
vis=likelihood>=VISIBILITY_THRESHOLD
tongue_code[vis & (y<q40)]=0
tongue_code[vis & (y>=q40) & (y<=q60)]=1
tongue_code[vis & (y>q60)]=2
```

iii. The 40/60 split and per-session scope come from the instructions; the confidence cutoff is the agent's own documented choice.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For every absolute neural-bin center, the nearest camera timestamp is selected. Trials whose full requested window lies outside the global video timestamp range are excluded.

ii.
```python
video_mask=(go+OFF_START>=video_t[0]) & (go+OFF_END<=video_t[-1])
vi=nearest_indices(video_t,centers.ravel()).reshape(centers.shape)
```

iii. The agent relies on the common NWB session clock and argues nearest-frame error is negligible at the camera's sampling rate.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Files with missing classifier labels are skipped; malformed event counts or observation mappings raise errors; non-string labels are decoded robustly; unknown anatomy becomes `Unknown`; trials lacking overall video coverage, ephys observation mapping, or any spike in the entire retained-unit window are excluded; invisible tongue samples receive class 3 rather than imputation. Multiple assertions validate shapes, finiteness, and minimum trial/unit counts.

ii.
```python
if d.dtype.kind in 'OSU': kept.append(p)
else: skipped.append((p.name, 'missing classifier labels'))
...
anno=[x if x and x.lower() not in ('nan','none') else 'Unknown' for x in anno]
...
if len(kept)<2: raise ValueError(...)
```

iii. The notes distinguish absent source streams (exclude rather than fabricate) from meaningful nonvisibility (explicit class). The trajectory records iterative correction of initially misinterpreted partial ephys coverage.

## 10-a. What are the most time-consuming steps of the code?

i. The agent identifies reading large NWB spike/video arrays, per-unit `searchsorted` spike binning across all edges, serial processing of 173 files, and writing the roughly 11.6-GB pickle as the dominant work.

ii.
```python
for j,u in enumerate(unit_ids):
    st=np.asarray(unit_spike_slice(spikes, ends, int(u)), dtype=np.float64)
    cumulative=np.searchsorted(st, flat, side='left').reshape(edges.shape)
...
with out.open('wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Conversion notes report roughly 0.9-1.3 seconds/session in the sample and describe I/O and spike binning as the main scalable costs.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining per-unit spike loop is only partly vectorized (all trials and edges are handled together); the loop over photostimulation intervals could also be replaced by interval indexing or a sweep. The outer session loop is serial. List construction for string category mapping is minor. The tongue path is already vectorized across trials/bins.

ii.
```python
for j,u in enumerate(unit_ids): ...
for a,b in zip(on,off):
    result |= ((absolute_centers>=a)&(absolute_centers<b))
outcome_code=np.array([{'ignore':0,'miss':1,'hit':2}[x] for x in outcome])
```

iii. The agent explains that ragged spike trains make a unit loop natural and emphasizes that it eliminated the much worse trial-by-bin-by-spike loops.

## 10-c. What processing does the code repeat multiple times?

i. Every file is opened once during eligibility screening and again during conversion. Within conversion, trial filtering causes several arrays (`kept`, `go_k`, `centers`, neural data) to be subset again after all-zero detection. Per-session string decoding and construction of the same category dictionaries also recur.

ii.
```python
with h5py.File(p, 'r') as f:  # eligible_files
...
with h5py.File(path,'r') as f:  # process_session
...
kept=kept[neural_nonzero]
go_k=go_k[neural_nonzero]
centers=centers[neural_nonzero]
```

iii. The notes mainly claim that large arrays are loaded only once per conversion pass; they accept the lightweight pre-screen and prioritize deterministic validation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `next_starts` and `neural_window_mask` are computed for validation/metadata rather than used to filter converted arrays. Diagnostic plots are optional. Rich `session_info`, absolute source paths, bin centers, and detailed exclusion counts support provenance but are not decoder features. Eligibility screening opens every file before conversion.

ii.
```python
next_starts=np.r_[starts[1:], np.inf]
neural_window_mask=np.zeros(nt,dtype=bool)
...
'excluded_incomplete_neural_window_trials':int(np.sum(recorded_mask & ~neural_window_mask))
```

iii. The agent regards these checks and metadata as intentional sanity/provenance work. The notes do not identify any major computed data product that is subsequently thrown away.
