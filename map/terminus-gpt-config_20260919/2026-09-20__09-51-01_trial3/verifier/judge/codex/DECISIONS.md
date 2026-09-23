# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI recursively finds NWB files under `/app/data`, but prefilters them first: it only keeps files whose `units/classification` dataset has a string/object dtype. It then opens each retained NWB file with `h5py` and reads raw HDF5 groups directly rather than using `pynwb`.

ii.
```python
def eligible_files():
    kept=[]; skipped=[]
    for p in sorted(DATA_ROOT.rglob('*.nwb')):
        with h5py.File(p, 'r') as f:
            d=f['units/classification']
            if d.dtype.kind in 'OSU': kept.append(p)
            else: skipped.append((p.name, 'missing classifier labels'))
    return kept, skipped
```

```python
with h5py.File(path,'r') as f:
    trials=f['intervals/trials']; nt=len(trials['id'])
    go=f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. In `CONVERSION_NOTES.md`, the AI says the distributed NWBs are the native synchronized source, and that the paper-compatible session set is the 173 files with classifier labels. It treated direct `h5py` access as an NWB-native equivalent to the reference loaders.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from `general/subject/subject_id` when present as strings, otherwise the parent folder name `sub-<id>` is used as a fallback. Final `subjects` is the sorted unique set of these IDs, and `subject_idx` maps sessions into that list.

ii.
```python
subject=(f['general/subject/subject_id'].asstr()[()]
         if f['general/subject/subject_id'].dtype.kind in 'OSU'
         else path.parent.name.replace('sub-',''))
```

```python
subjects=sorted(set(x['subject'] for x in sessions)); smap={x:i for i,x in enumerate(subjects)}
'subject_idx':np.asarray([smap[x['subject']] for x in sessions],dtype=np.int32),
```

iii. The notes justify this as direct use of the NWB subject metadata, with a fallback only for robustness.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. Sessions are processed in sorted pathname order, and the session identifier stored in metadata is `path.stem`, not `nwb.identifier`.

ii.
```python
for p in sorted(DATA_ROOT.rglob('*.nwb')):
```

```python
session_info={
    'session_id':path.stem, 'source_file':str(path), 'subject':str(subject),
    ...
}
```

iii. The notes explicitly state that `/app/data` contains one NWB per recording session, so file boundaries are taken as session boundaries.

## 1-d. How are the data split into trials?

i. Behavioral trials come from `intervals/trials`, with a sanity check that the number of go cues equals the number of trial rows. The AI additionally maps electrophysiology observation intervals back onto behavioral trials by matching `obs_intervals` start times to `trials/start_time`.

ii.
```python
trials=f['intervals/trials']; nt=len(trials['id'])
go=f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
if len(go)!=nt: raise ValueError(f'{path.name}: {len(go)} go events != {nt} trials')
```

```python
ref_intervals=obs[obs_start[0]:obs_end[0]]
recorded_idx=np.searchsorted(starts,ref_intervals[:,0])
if np.any(recorded_idx>=nt) or not np.allclose(starts[recorded_idx],ref_intervals[:,0],atol=1e-6):
    raise ValueError(f'{path.name}: cannot map neural observation intervals to trials')
```

iii. The trajectory shows the AI initially misread the ephys coverage, then concluded that `obs_intervals` enumerates recorded behavioral trials and must be mapped to the trials table rather than inferred from go cues alone.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept if they are represented in `units/obs_intervals` and have full tongue-video coverage for the requested window. After neural binning, trials whose entire neural matrix is zero are removed. The AI does not explicitly exclude `free_water` trials.

ii.
```python
recorded_mask=np.zeros(nt,dtype=bool); recorded_mask[recorded_idx]=True
video_mask=(go+OFF_START>=video_t[0]) & (go+OFF_END<=video_t[-1])
keep_trial=recorded_mask & video_mask
kept=np.flatnonzero(keep_trial)
```

```python
neural_nonzero=np.any(neural_cube!=0,axis=(1,2))
excluded_all_zero=int(np.sum(~neural_nonzero))
if excluded_all_zero:
    kept=kept[neural_nonzero]
    go_k=go_k[neural_nonzero]
    centers=centers[neural_nonzero]
    neural_cube=neural_cube[neural_nonzero]
```

iii. The notes say the decoder task requires retaining photostimulation, early-lick, miss, and ignore trials, so the AI chose to exclude only trials with invalid required continuous streams or unusable all-zero neural input.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` plus `units/spike_times_index`, with `BehavioralEvents/go_start_times` providing the alignment event for bin edges.

ii.
```python
spikes=f['units/spike_times']; ends=f['units/spike_times_index'][:]
...
edges=go_kept[:,None]+REL_EDGES[None,:]
```

iii. The notes describe the NWB spikes as a ragged absolute-time representation, so firing rates are computed directly from those timestamps.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into half-open 50 ms windows around go cue, uses `np.searchsorted` on each unit’s sorted spike train across all trial edges at once, differences cumulative counts, and divides by `0.05` to convert counts to Hz.

ii.
```python
edges=go_kept[:,None]+REL_EDGES[None,:]
flat=edges.ravel()
...
for j,u in enumerate(unit_ids):
    st=np.asarray(unit_spike_slice(spikes, ends, int(u)), dtype=np.float64)
    cumulative=np.searchsorted(st, flat, side='left').reshape(edges.shape)
    counts[j]=np.diff(cumulative, axis=1)/BIN_S
return counts.transpose(1,0,2)
```

iii. `CONVERSION_NOTES.md` says this preserves the reference half-open counting/rate convention while switching to the task-required non-overlapping 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are retained only if `units/classification == 'good'` and `units/is_good_trials` is true for every source trial for that unit. Sessions with no such units are rejected.

ii.
```python
classification=decode_strings(f['units/classification'])
stable=good_trials.all(axis=1)
unit_ids=np.flatnonzero((classification=='good') & stable)
if len(unit_ids)==0: raise ValueError(f'{path.name}: no stable classifier-good units')
```

iii. The AI’s notes justify this as a fixed-neuron-set requirement of the target format: rather than represent unit-by-trial validity, it removes units that ever fail `is_good_trials`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go cue onset by adding fixed go-relative bin edges to each trial’s absolute go-cue timestamp.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
REL_EDGES = OFF_START + np.arange(N_BINS+1, dtype=np.float64)*BIN_S
...
edges=go_kept[:,None]+REL_EDGES[None,:]
```

iii. The notes say all streams are already on one session-absolute clock, so alignment is done directly in timestamp space.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins across a -2.5 s to +1.5 s window, giving 80 bins per trial. No temporal smoothing or secondary rebinning is applied.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
N_BINS = int(round((OFF_END-OFF_START)/BIN_S))
REL_EDGES = OFF_START + np.arange(N_BINS+1, dtype=np.float64)*BIN_S
REL_CENTERS = (REL_EDGES[:-1]+REL_EDGES[1:])/2
```

iii. The AI explicitly cites the task instructions as overriding the method paper’s 40 ms / 3.4 ms analysis bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times/timestamps`, using the last sample-start event before each trial’s go cue, together with the go-relative bin centers.

ii.
```python
sample=f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
pos=np.searchsorted(sample, go, side='left')-1
tone=sample[pos]
```

iii. The notes justify using the last pre-go sample event because early licks can replay the sample epoch.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each retained trial and each 50 ms bin center, the AI subtracts the trial’s selected tone onset from the absolute bin-center timestamp.

ii.
```python
go_k=go[kept]; centers=go_k[:,None]+REL_CENTERS[None,:]
...
time_from_tone=(centers-tone_all[kept,None]).astype(np.float32)
```

iii. The AI’s stated rationale is that the requested decoder input is continuous elapsed time, not a binary onset pulse.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on the exact same bin centers used for neural binning.

ii.
```python
centers=go_k[:,None]+REL_CENTERS[None,:]
time_from_tone=(centers-tone_all[kept,None]).astype(np.float32)
```

iii. The notes describe this as using the same go-aligned time grid for neural and input streams.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from `BehavioralEvents/photostim_start_times/timestamps` and `BehavioralEvents/photostim_stop_times/timestamps`, not from the trials-table `photostim_onset` and `photostim_duration` fields.

ii.
```python
ev=f['acquisition/BehavioralEvents']
on=ev['photostim_start_times/timestamps'][:]
off=ev['photostim_stop_times/timestamps'][:]
```

iii. In the notes, the AI says it chose direct event timestamps to avoid parsing `'N/A'` trial-table strings and to align stimulation on the native absolute clock.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. It builds a binary time series over neural bin centers: a bin is 1 if its absolute center time falls within any `[stim_on, stim_off)` interval, otherwise 0.

ii.
```python
result=np.zeros(absolute_centers.shape, dtype=bool)
for a,b in zip(on,off): result |= ((absolute_centers>=a)&(absolute_centers<b))
return result.astype(np.float32)
```

iii. The AI’s notes say this is the intended task-specific representation: a time-varying photostimulation-on signal rather than a trial-level flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Alignment is done by comparing absolute photostimulation event times to the absolute bin centers derived from the go-aligned neural grid.

ii.
```python
go_k=go[kept]; centers=go_k[:,None]+REL_CENTERS[None,:]
stim=photostim_series(f,centers)
```

iii. The notes treat this as equivalent to putting both streams on the same session-absolute timeline and then reading out the go-aligned bins.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from two trial-table columns: `outcome` and `trial_instruction`. There is no direct choice variable in the raw NWB.

ii.
```python
outcome=decode_strings(trials['outcome'])[kept]
instruction=decode_strings(trials['trial_instruction'])[kept]
choice=np.where(outcome=='ignore',2,np.where(outcome=='hit',
                 np.where(instruction=='left',0,1),
                 np.where(instruction=='left',1,0))).astype(np.int8)
```

iii. The trajectory notes that the AI compared this against lick-event-derived choice, but kept the curated trial labels because they were more authoritative on edge cases.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as `0=left`, `1=right`, `2=no lick`, then repeats the per-trial category across all 80 bins in output row 0.

ii.
```python
choice=np.where(outcome=='ignore',2,np.where(outcome=='hit',
                 np.where(instruction=='left',0,1),
                 np.where(instruction=='left',1,0))).astype(np.int8)
...
output_cube[:,0,:]=choice[:,None]
```

iii. The notes justify the no-lick third class because `ignore` trials are required decoder outputs and choice is a per-trial, not time-varying, label.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from `trials['outcome']`.

ii.
```python
outcome=decode_strings(trials['outcome'])[kept]
```

iii. No special justification beyond using the existing curated trial label appears in the notes.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped to `0=ignore`, `1=miss`, `2=hit`, then repeated across all bins in output row 1.

ii.
```python
outcome_code=np.array([{'ignore':0,'miss':1,'hit':2}[x] for x in outcome],dtype=np.int8)
...
output_cube[:,1,:]=outcome_code[:,None]
```

iii. The AI follows the task’s categorical output requirement and keeps the label constant over the trial.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from `trials['early_lick']`.

ii.
```python
early=decode_strings(trials['early_lick'])[kept]
```

iii. The notes treat this as an authoritative per-trial behavioral label from the NWB trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The strings are mapped to `0=no`, `1=yes`, then repeated across all bins in output row 2.

ii.
```python
early_code=np.array([{'no early':0,'early':1}[x] for x in early],dtype=np.int8)
...
output_cube[:,2,:]=early_code[:,None]
```

iii. The AI again uses a per-trial categorical label duplicated across the common time axis.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position comes from `BehavioralTimeSeries/Camera0_side_TongueTracking`, using `data[:,1]` as `y` and `data[:,2]` as the visibility/confidence signal, with `timestamps` for alignment.

ii.
```python
tg=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
video_t=tg['timestamps'][:]
...
tongue=np.asarray(tg['data'][:],dtype=np.float64)
...
y=tongue[vi,1]; likelihood=tongue[vi,2]
```

iii. The notes identify Camera0 side-view tongue tracking as the dataset-wide available tongue stream.

## 8-b. How is `output` *Tongue y-position* processed?

i. The AI thresholds frame visibility at likelihood `>= 0.9`, computes session-wide 40th and 60th percentiles from the raw `y` values of visible frames only, aligns each neural bin to the nearest video frame, and classifies that single frame’s `y` value. It does not average tongue `y` within 50 ms bins.

ii.
```python
VISIBILITY_THRESHOLD = 0.9
...
visible_session=tongue[:,2]>=VISIBILITY_THRESHOLD
q40,q60=np.percentile(tongue[visible_session,1],[40,60])
vi=nearest_indices(video_t,centers.ravel()).reshape(centers.shape)
y=tongue[vi,1]; likelihood=tongue[vi,2]
```

iii. The notes justify this by saying the likelihood is strongly bimodal, `0.9` is a conservative DLC threshold, and nearest-frame alignment is sufficient at roughly 300 Hz.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses four classes: `0` if visible and below the session 40th percentile, `1` if visible and between the 40th and 60th percentiles inclusive, `2` if visible and above the 60th percentile, and `3` if the aligned frame is not visible.

ii.
```python
tongue_code=np.full(y.shape,3,dtype=np.int8)
vis=likelihood>=VISIBILITY_THRESHOLD
tongue_code[vis & (y<q40)]=0
tongue_code[vis & (y>=q40) & (y<=q60)]=1
tongue_code[vis & (y>q60)]=2
```

iii. The notes say the percentiles are session-specific and computed only from visible frames so that occluded frames do not distort the thresholds.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Each neural bin center is matched to the nearest video frame timestamp, and the tongue class for that nearest frame is used for the corresponding neural bin.

ii.
```python
vi=nearest_indices(video_t,centers.ravel()).reshape(centers.shape)
y=tongue[vi,1]; likelihood=tongue[vi,2]
```

iii. The notes explicitly justify nearest-frame alignment as sufficiently accurate because video runs at roughly 300 Hz and the bin size is 50 ms.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI adds several robustness layers: it decodes mixed string-like HDF5 arrays with fallback stringification, skips sessions whose classifier labels are not string/object typed, falls back to folder names for missing subject strings, replaces missing anatomy labels with `'Unknown'`, rejects sessions with too few visible tongue frames, excludes trials lacking complete video coverage, and removes trials with all-zero neural windows.

ii.
```python
def decode_strings(dataset):
    a = dataset[:]
    out=[]
    for x in a:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode('utf-8', errors='replace').strip())
        elif isinstance(x, str):
            out.append(x.strip())
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)
```

```python
if d.dtype.kind in 'OSU': kept.append(p)
else: skipped.append((p.name, 'missing classifier labels'))
```

```python
anno=decode_strings(f['units/anno_name'])[unit_ids].tolist()
anno=[x if x and x.lower() not in ('nan','none') else 'Unknown' for x in anno]
```

iii. The notes frame these as defensive handling for the released NWBs: one session had all-NaN classifier labels, some labels are missing, and incomplete source streams should be excluded rather than fabricated.

## 10-a. What are the most time-consuming steps of the code?

i. The heavy steps are per-session loading of large spike and tongue-tracking arrays and the per-unit `searchsorted` neural binning loop.

ii.
```python
spikes=f['units/spike_times']; ends=f['units/spike_times_index'][:]
...
for j,u in enumerate(unit_ids):
    st=np.asarray(unit_spike_slice(spikes, ends, int(u)), dtype=np.float64)
    cumulative=np.searchsorted(st, flat, side='left').reshape(edges.shape)
```

```python
tongue=np.asarray(tg['data'][:],dtype=np.float64)
```

iii. The notes repeatedly describe spike-buffer I/O and vectorized per-unit binning as the main runtime cost on the ~100 GB dataset.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code already vectorizes trials and tongue alignment aggressively, but it still loops over units for spike binning, loops over stimulation intervals when building the photostim mask, and loops over elements while decoding string datasets.

ii.
```python
for j,u in enumerate(unit_ids):
    ...
```

```python
for a,b in zip(on,off): result |= ((absolute_centers>=a)&(absolute_centers<b))
```

```python
for x in a:
    ...
```

iii. The notes say the AI intentionally avoided trial-by-trial spike loops and vectorized tongue handling; the remaining ragged per-unit spike loop was accepted as the practical cost of the source storage format.

## 10-c. What processing does the code repeat multiple times?

i. There is little intentional recomputation. The bin grid is defined once at module scope and reused, each session is loaded once, and most per-session quantities are computed a single time.

ii.
```python
REL_EDGES = OFF_START + np.arange(N_BINS+1, dtype=np.float64)*BIN_S
REL_CENTERS = (REL_EDGES[:-1]+REL_EDGES[1:])/2
```

```python
for i,p in enumerate(files):
    sess=process_session(p,show_processing=args.show_processing and i<2)
```

iii. The notes describe the conversion as a single pass over sessions with precomputed shared time grids.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some extra diagnostic/bookkeeping work that is not used to build the decoder arrays: it computes `next_starts` only for an assertion, computes `neural_window_mask` and related exclusion counts without using that mask for filtering, and optionally produces processing plots unrelated to the serialized dataset.

ii.
```python
next_starts=np.r_[starts[1:], np.inf]
if np.any(go+OFF_END>next_starts): raise ValueError('requested window crosses next trial')
```

```python
neural_window_mask=np.zeros(nt,dtype=bool)
neural_window_mask[recorded_idx]=((go[recorded_idx]+OFF_START>=ref_intervals[:,0]) &
                                  (go[recorded_idx]+OFF_END<=ref_intervals[:,1]))
```

```python
if show_processing:
    plot_processing(path.stem, REL_CENTERS, neural_cube, input_cube, output_cube,
                    q40, q60, choice, outcome_code, early_code)
```

iii. There is no strong explicit justification in the notes beyond validation and documentation; these computations are mainly used for assertions, metadata, or optional plots rather than the final decoder tensors.
