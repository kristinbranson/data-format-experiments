# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats the dataset as one NWB/HDF5 file per session under `/app/data/sub-*/`. It discovers every session with a single recursive glob (`DATA_ROOT.rglob('*.nwb')`), sorted for determinism, and opens each file **directly with `h5py`** rather than with `pynwb`. Before conversion it makes one pass over all 174 files (`eligible_files()`) that opens each file solely to test whether `units/classification` is a string-typed dataset; files that fail the test are reported as `SKIP`. The remaining 173 files are then each opened a second time in `process_session()`, where subject, trials table, behavioral event timestamps, units/spike times, tongue video and unit annotations are all read from within the same handle. Reported totals: 173 sessions, 28 subjects, 90,094 trials, 68,888 neurons, 293 brain regions.

ii.
```python
DATA_ROOT = Path('/app/data')

def eligible_files():
    """Return files with the custom classifier output used by the papers."""
    kept=[]; skipped=[]
    for p in sorted(DATA_ROOT.rglob('*.nwb')):
        with h5py.File(p, 'r') as f:
            d=f['units/classification']
            if d.dtype.kind in 'OSU': kept.append(p)
            else: skipped.append((p.name, 'missing classifier labels'))
    return kept, skipped
```
```python
def process_session(path, show_processing=False):
    t0=time.perf_counter()
    with h5py.File(path,'r') as f:
        subject=(f['general/subject/subject_id'].asstr()[()] ...)
        trials=f['intervals/trials']; nt=len(trials['id'])
        go=f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
        ...
        spikes=f['units/spike_times']; ends=f['units/spike_times_index'][:]
        tg=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
```
```python
    files,skipped=eligible_files()
    if args.sample: files=files[:2]
    print(f'Eligible sessions: {len(files)}; skipped source files: {len(skipped)}',flush=True)
    for i,p in enumerate(files):
        sess=process_session(p,show_processing=args.show_processing and i<2)
```

iii. From CONVERSION_NOTES.md Step 2/Step 4: "`/app/data` contains 174 NWB 2.x/HDF5 files, one file per recording session, nested under `sub-<mouse>/`" and "The distributed NWBs are the raw synchronized source; session-absolute spikes, events, and tracking must be aligned directly by timestamps." The AI chose raw `h5py` access because the ragged `spike_times`/`spike_times_index` and `obs_intervals`/`obs_intervals_index` layouts are read directly, and because it wanted lazy per-unit slicing rather than materialising whole buffers ("Repeatedly loading full video arrays or scanning spikes per trial would duplicate I/O"). The pre-pass over files was justified as session-level QC: "`classification` is object/string in 173 sessions but float64 NaN in one session; robust code must not assume string dtype."

## 1-b. How are the data split into subjects (mice)?

i. Each session's animal is read from the NWB metadata field `general/subject/subject_id` (numeric strings such as `'440956'`), with the parent directory name (`sub-440956`) as a fallback if the field is not string-typed. At assembly, `subjects` is the sorted set of unique ids and `subject_idx` is an `int32` array giving each session's index into that list. This yields 28 subjects with 3–10 sessions each.

ii.
```python
subject=(f['general/subject/subject_id'].asstr()[()]
         if f['general/subject/subject_id'].dtype.kind in 'OSU'
         else path.parent.name.replace('sub-',''))
```
```python
subjects=sorted(set(x['subject'] for x in sessions)); smap={x:i for i,x in enumerate(subjects)}
data={
  ...
  'subjects':subjects,
  'subject_idx':np.asarray([smap[x['subject']] for x in sessions],dtype=np.int32),
```

iii. Step 5 mapping table: "`general/subject/subject_id` → `subjects`, `subject_idx`; Sorted unique IDs and per-session index; direct NWB metadata; 28 subjects expected." Step 9 records "Subjects | 28 | 28 | 28 | Exact", i.e. the count was checked against the 28 mice reported in the data paper. The AI treats the NWB subject field as canonical and adds the directory-name fallback only as robustness against a non-string field (the same defensive pattern it used for `classification`).

## 1-c. How are the data split into sessions?

i. One NWB file is one session, so no grouping is performed — the sorted file list *is* the session list and output ordering follows it. Session identity is recorded as `path.stem` (the filename, e.g. `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`) rather than the internal `nwb.identifier`, together with the full source path, in `metadata['session_info']`. One session-level filter is applied: the single file whose `units/classification` column is all-NaN (`sub-440958_ses-20190216T162508`) is skipped, giving 173 sessions. Two additional session-level drop conditions exist inside `process_session` (fewer than two complete trials, no stable classifier-good units) but neither fired on this dataset.

ii.
```python
for p in sorted(DATA_ROOT.rglob('*.nwb')):
    with h5py.File(p, 'r') as f:
        d=f['units/classification']
        if d.dtype.kind in 'OSU': kept.append(p)
        else: skipped.append((p.name, 'missing classifier labels'))
```
```python
session_info={
    'session_id':path.stem, 'source_file':str(path), 'subject':str(subject),
    'source_trials':int(nt), 'neural_recorded_trials':int(recorded_mask.sum()),
    'retained_trials':int(len(kept)), ...
```
```python
if len(kept)<2: raise ValueError(f'{path.name}: fewer than two complete neural+video trials')
if len(unit_ids)==0: raise ValueError(f'{path.name}: no stable classifier-good units')
```

iii. Step 4 discrepancy table: "174 NWBs; one session (`sub-440958_ses-20190216T162508`) has all-NaN `classification`; 173 have classifier labels | 173 analyzed sessions | Exclude the single session lacking classifier output; retain the 173 QC-compatible sessions." The AI explicitly rejected applying the data paper's behavioural session-selection criteria (>65% performance, ≥50 correct trials/side) because "Applying literal criteria to released labels excludes 22/174 files … Treat classifier availability as authoritative released-session curation. The behavioral criteria are analysis-selection criteria and cannot supersede task-required labels."

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` DynamicTable, one row per behavioural trial. The AI asserts a one-to-one correspondence between rows and go-cue events and additionally asserts that the requested extraction window never runs past the start of the next trial. Trial start times are then used as the join key against the ephys observation intervals (see 1-e).

ii.
```python
trials=f['intervals/trials']; nt=len(trials['id'])
go=f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
if len(go)!=nt: raise ValueError(f'{path.name}: {len(go)} go events != {nt} trials')
starts=trials['start_time'][:]
next_starts=np.r_[starts[1:], np.inf]
if np.any(go+OFF_END>next_starts): raise ValueError('requested window crosses next trial')
```

iii. Step 2 notes: "`acquisition/BehavioralEvents` contains absolute timestamp series for presample, sample, delay, go, trial end, left/right licks, and photostimulation start/stop. `go_start_times` has exactly one event per trial. Some sample/delay streams contain extra events (e.g. 405 samples for 368 trials), so trial-window/event matching rather than positional truncation is required." The AI therefore uses the trials table as the definition of a trial and uses `go_start_times` (the one stream with exactly one event per trial) as the per-trial anchor, while treating the multi-event sample/delay streams as needing explicit matching.

## 1-e. How are trials filtered based on quality controls?

i. Three filters, all data-availability rather than behavioural:

1. **Ephys observation**: `units/obs_intervals` (ragged, one `[start, stop]` per *recorded* trial per unit) is read for the first unit, its interval start times are matched to `trials.start_time` by `searchsorted`, and only matched trials are retained. Consistency is asserted (every unit must have the same number of intervals; every interval start must match a trial start to 1e-6).
2. **Video coverage**: a trial is kept only if its whole window `[go-2.5, go+1.5]` lies inside the session's tongue-camera timestamp range.
3. **All-zero neural**: after binning, any trial with no spikes at all from any retained unit anywhere in the window is dropped.

A session is dropped if fewer than two trials survive (this never fired). No behavioural exclusion is applied: photostimulation, free-water/auto-water, early-lick, miss and ignore trials are all deliberately kept. Across the dataset: 93,310 recorded trials → 802 removed by the video filter → 2,414 removed by the all-zero filter → **90,094 retained**. Notably, the AI has **no explicit `free_water` filter**; the all-zero-neural rule removes those trials empirically instead (there are 2,450 free-water trials among recorded trials).

ii.
```python
# NWB units are trialized: obs_intervals and is_good_trials columns identify
# behavioral trials that were actually recorded electrophysiologically.
good_trials=f['units/is_good_trials'][:]
obs=f['units/obs_intervals']; obs_end=f['units/obs_intervals_index'][:]
obs_start=np.r_[0,obs_end[:-1]]
interval_counts=obs_end-obs_start
if not np.all(interval_counts==good_trials.shape[1]):
    raise ValueError(f'{path.name}: inconsistent per-unit observation interval counts')
ref_intervals=obs[obs_start[0]:obs_end[0]]
recorded_idx=np.searchsorted(starts,ref_intervals[:,0])
if np.any(recorded_idx>=nt) or not np.allclose(starts[recorded_idx],ref_intervals[:,0],atol=1e-6):
    raise ValueError(f'{path.name}: cannot map neural observation intervals to trials')
recorded_mask=np.zeros(nt,dtype=bool); recorded_mask[recorded_idx]=True

# Complete video coverage is required for every retained output sample.
tg=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
video_t=tg['timestamps'][:]
video_mask=(go+OFF_START>=video_t[0]) & (go+OFF_END<=video_t[-1])
keep_trial=recorded_mask & video_mask
kept=np.flatnonzero(keep_trial)
if len(kept)<2: raise ValueError(f'{path.name}: fewer than two complete neural+video trials')
```
```python
# Exclude mapped trials having no spikes from any retained unit anywhere
# in the requested window; these have no usable neural decoder input.
neural_nonzero=np.any(neural_cube!=0,axis=(1,2))
excluded_all_zero=int(np.sum(~neural_nonzero))
if excluded_all_zero:
    kept=kept[neural_nonzero]; go_k=go_k[neural_nonzero]
    centers=centers[neural_nonzero]; neural_cube=neural_cube[neural_nonzero]
if len(kept)<2:
    raise ValueError(f'{path.name}: fewer than two usable neural trials')
```

iii. Step 4/Step 5: "Retain all trials because the task explicitly requires photostim input and early-lick/ignore/no-lick outputs" and "These trial exclusions [the papers' photoinhibition/free-water/early-lick/ignore exclusions] cannot be copied wholesale here: photostimulation is a required decoder input, and early lick, ignore outcome, and no-lick choice are required decoder outputs." Step 9 documents the iteration that produced this rule: "Initial full validation reported 3,470 wholly zero neural trials. Investigation showed `units/obs_intervals` and `is_good_trials` columns enumerate only electrophysiologically recorded trials … An intermediate overly strict rule requiring the entire requested window inside trial observation bounds disproportionately removed short ignore trials; it was rejected. Final logic preserves recorded short trials but excludes only wholly absent neural inputs." The video filter is justified as "Complete video coverage is required for every retained output sample" / "Exclude only genuine stream-coverage failures."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (a single concatenated float64 buffer of session-absolute spike times) together with `units/spike_times_index` (cumulative per-unit end offsets), restricted to the selected units. The go-cue timestamps (`acquisition/BehavioralEvents/go_start_times/timestamps`) supply the bin-edge anchors. Unit selection additionally consumes `units/classification` and `units/is_good_trials`; region labels come from `units/anno_name`.

ii.
```python
def unit_spike_slice(spikes, ends, unit_id):
    start=0 if unit_id==0 else int(ends[unit_id-1])
    return spikes[start:int(ends[unit_id])]

def bin_selected_units(f, unit_ids, go_kept):
    """Vectorized half-open spike counting, output trials x units x bins."""
    edges=go_kept[:,None]+REL_EDGES[None,:]
    flat=edges.ravel()
    spikes=f['units/spike_times']; ends=f['units/spike_times_index'][:]
```

iii. Step 2: "`units` is a ragged spike table. `spike_times` stores absolute session timestamps concatenated across units and `spike_times_index` gives cumulative unit endpoints." Step 3: "Data are extracellular electrophysiology spike times, not calcium imaging; delta-F/F is not applicable." Spike times are the only neural representation available, so firing rates are computed from them directly.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each selected unit, the 81 bin edges of every trial are flattened into a single sorted-query array, one `np.searchsorted(..., side='left')` gives the running spike count at every edge, and `np.diff` along the bin axis gives the spike count per half-open bin `[left, right)`. Counts are divided by the 0.05 s bin width to give Hz, stored `float32`, and transposed to `(trials, neurons, bins)`, then sliced per trial into `(n_neurons, 80)` arrays. No smoothing, normalisation, baseline subtraction or z-scoring is applied.

ii.
```python
def bin_selected_units(f, unit_ids, go_kept):
    """Vectorized half-open spike counting, output trials x units x bins."""
    edges=go_kept[:,None]+REL_EDGES[None,:]
    flat=edges.ravel()
    spikes=f['units/spike_times']; ends=f['units/spike_times_index'][:]
    counts=np.empty((len(unit_ids), len(go_kept), N_BINS), dtype=np.float32)
    for j,u in enumerate(unit_ids):
        st=np.asarray(unit_spike_slice(spikes, ends, int(u)), dtype=np.float64)
        cumulative=np.searchsorted(st, flat, side='left').reshape(edges.shape)
        counts[j]=np.diff(cumulative, axis=1)/BIN_S
    return counts.transpose(1,0,2)  # trials, neurons, time
```
```python
'spike_binning':'half-open [left,right) bins; counts divided by 0.05 s (Hz)',
```

iii. Step 1 identified the reference convention: "`sliding_histogram` … Count spikes in half-open windows `[center-width/2, center+width/2)` and divide by width for Hz". Step 4 resolution: "Use reference half-open counting but task-required 50-ms non-overlapping bins." Step 10 Check 3 records the comparison: "Binning | half-open bins, count/0.05 | `sliding_histogram`, half-open, count/bin width | Logic matched; 50 ms required instead of reference 40 ms." Step 10 Check 2 verified one bin by hand against raw timestamps with `np.allclose`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit-level criteria, both from the `units` table: (1) `classification == 'good'`, the region-specific QC classifier verdict of the Chen/Liu spike-sorting white paper; and (2) `is_good_trials` true for **every** recorded trial of the session ("stable"). Units failing either are dropped. No thresholds are applied to individual QC metrics (amplitude, presence ratio, ISI violation, drift), and the method paper's 2 Hz minimum-firing-rate filter is deliberately **not** applied. The older Kilosort `unit_quality` field is deliberately not used. Result: 68,888 units retained (from 69,453 classifier-good units in the 173 sessions), mean 398.2 per session, range 90–923.

ii.
```python
classification=decode_strings(f['units/classification'])
stable=good_trials.all(axis=1)
unit_ids=np.flatnonzero((classification=='good') & stable)
if len(unit_ids)==0: raise ValueError(f'{path.name}: no stable classifier-good units')
```
```python
'neuron_filter':"units/classification == 'good' and is_good_trials true for every source trial",
```

iii. Step 4 discrepancy table: "Quality field | `unit_quality` gives 154,948 `good` and is much broader; `classification` gives paper-scale 69,453 | Region-specific classifier units used | Use `classification=='good'`, not Kilosort `unit_quality`." On the extra stability filter: "`is_good_trials` has 64,612 invalid pairs among classifier-good units, affecting 565 units in four sessions … Because target format needs one fixed neuron matrix shape per session, retain only classifier-good units valid on every session trial (68,888 units expected). This avoids silently treating invalid periods as zero activity." On the rejected 2 Hz filter: "Do not apply: it is downstream method-paper curation, not data-paper QC, and would discard 41.5% of released good units."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go-cue onset**, taken from `acquisition/BehavioralEvents/go_start_times/timestamps`. Because spikes, events and video all live on one session-absolute clock, no resampling or offset correction is performed: the fixed relative edge grid `REL_EDGES` (−2.5 … +1.5 s) is broadcast onto each retained trial's go-cue time to produce absolute edges, and spikes are binned against those directly.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
REL_EDGES = OFF_START + np.arange(N_BINS+1, dtype=np.float64)*BIN_S
REL_CENTERS = (REL_EDGES[:-1]+REL_EDGES[1:])/2
```
```python
go=f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
go_k=go[kept]; centers=go_k[:,None]+REL_CENTERS[None,:]
...
edges=go_kept[:,None]+REL_EDGES[None,:]
```
```python
'temporal_alignment_event':'go cue onset (BehavioralEvents/go_start_times)',
'off_start':OFF_START, 'off_end':OFF_END, 'n_time_bins':N_BINS,
```

iii. Step 4: "The distributed NWBs are the raw synchronized source; session-absolute spikes, events, and tracking must be aligned directly by timestamps." Step 10 Check 3: "Alignment | absolute event timestamps, go=0 | subtract go cue from events/spikes | Equivalent." Step 10 Check 5 confirmed "bin centers are exactly `-2.475 + 0.05*k`, k=0..79, with metadata offsets -2.5/+1.5" and that "requested windows never cross the next trial start."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, non-overlapping, 80 bins per trial spanning −2.5 s to +1.5 s about the go cue. The grid is defined once at module level and reused for every trial and session, so `T` is exactly 80 everywhere (verified: `T: mean 80.00, min 80, max 80`). Spikes are binned once at this resolution — there is no intermediate finer binning followed by rebinning. The tongue video (~294 Hz) is *downsampled* onto the same grid by nearest-frame lookup rather than rebinned by averaging (see 8-b). The reference code's own 40 ms width / 3.4 ms stride sliding windows were deliberately not used.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
N_BINS = int(round((OFF_END-OFF_START)/BIN_S))
REL_EDGES = OFF_START + np.arange(N_BINS+1, dtype=np.float64)*BIN_S
REL_CENTERS = (REL_EDGES[:-1]+REL_EDGES[1:])/2
```
```python
'time_bin_size':50.0, 'time_bin_size_units':'ms',
'bin_centers_seconds':REL_CENTERS.astype(np.float32),
```

iii. Step 3/Step 4: "The method paper bins spikes into 40-ms firing-rate windows with 3.4-ms stride. Reference code counts half-open windows and divides by width. This conversion must preserve that counting convention but change width/stride to required non-overlapping 50 ms." The window and bin width are set by the Decoder Task section of the instructions, which the AI treats as overriding the reference analysis choice.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times/timestamps` (the instruction-tone onsets) together with the per-trial go-cue times. Because an early lick replays the sample epoch, a trial can carry several sample events; the AI takes the **last sample onset strictly before the go cue**, and asserts that this onset still falls after the trial's own start time.

ii.
```python
def session_tone_onsets(f, trial_starts, go):
    sample=f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
    pos=np.searchsorted(sample, go, side='left')-1
    if np.any(pos<0): raise ValueError('go cue without preceding sample onset')
    tone=sample[pos]
    if np.any(tone<trial_starts): raise ValueError('last pre-go sample is outside trial')
    return tone
```
```python
'tone_onset_definition':'last sample_start_times event after trial start and before go (handles replay)',
```

iii. Step 5 Key Decision 4: "**Tone event**: Use the last sample-start event between trial start and go. Early licking can replay sample/delay, so this represents the tone epoch causally associated with the final go cue." Step 2 flagged the underlying data fact: "Some sample/delay streams contain extra events (e.g. 405 samples for 368 trials), so trial-window/event matching rather than positional truncation is required."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying value: for every 50 ms bin, the absolute bin-centre time minus the trial's tone-onset time, in seconds, stored as `float32` in row 0 of the `(2, 80)` input array. No binarisation, clipping, or normalisation is applied. The observed range over the full dataset is `[-1.5, 11.9]` s (negative early in the window, since the window starts 2.5 s before the go cue and the tone can be later than that; large positive values occur on replay trials with long sample/delay repeats).

ii.
```python
go_k=go[kept]; centers=go_k[:,None]+REL_CENTERS[None,:]
...
tone_all=session_tone_onsets(f, starts, go)
time_from_tone=(centers-tone_all[kept,None]).astype(np.float32)
stim=photostim_series(f,centers)
input_cube=np.stack((time_from_tone,stim),axis=1).astype(np.float32)
```
```python
'input_names':['time from tone onset','photostimulation on'],
```

iii. Step 5 Key Decision 5: "**Tone representation**: Use continuous elapsed time in seconds, as explicitly requested by 'time from tone onset,' rather than a binary onset pulse." The mapping table records the transform as "At each bin center, absolute center time minus tone onset, in seconds … Continuous time-from-tone-onset as explicitly required; replay trials use most recent sample onset." Step 10 Check 2 verified "all 80 bin-center times minus the direct last pre-go sample timestamp" against the raw NWB with `np.allclose`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: the same `REL_CENTERS` grid that defines the spike-binning edges is used to build the absolute bin centres `centers = go_k[:,None] + REL_CENTERS[None,:]`, and the tone offset is subtracted from those centres. Bin *k* of the input therefore refers to exactly the midpoint of bin *k* of the firing-rate matrix. No interpolation or resampling is needed, since all timestamps share one session-absolute clock.

ii.
```python
REL_EDGES = OFF_START + np.arange(N_BINS+1, dtype=np.float64)*BIN_S
REL_CENTERS = (REL_EDGES[:-1]+REL_EDGES[1:])/2
```
```python
go_k=go[kept]; centers=go_k[:,None]+REL_CENTERS[None,:]
time_from_tone=(centers-tone_all[kept,None]).astype(np.float32)
```
```python
edges=go_kept[:,None]+REL_EDGES[None,:]   # same grid, edges instead of centres
```

iii. Implicit in the design rather than separately argued; the AI's Step 10 Check 5 verification ("Confirmed bin centers are exactly `-2.475 + 0.05*k`") and the `--show-processing` plots ("neural heatmaps/population means centered on go=0, tone/stimulation inputs … Neural and inputs have the expected temporal ranges") are the stated evidence that inputs and neural data share one time axis.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The event streams `acquisition/BehavioralEvents/photostim_start_times/timestamps` and `photostim_stop_times/timestamps`, which hold absolute laser on/off times. The AI deliberately does **not** use the trials-table columns `photostim_onset` / `photostim_duration` (which store strings measured from trial start, with `'N/A'` on unstimulated trials). Both sources are numerically equivalent — I confirmed on `sub-440956_ses-20190207T120657` that the 78 event onsets equal `start_time + float(photostim_onset)` and the event durations equal `photostim_duration` exactly.

ii.
```python
def photostim_series(f, absolute_centers):
    ev=f['acquisition/BehavioralEvents']
    on=ev['photostim_start_times/timestamps'][:]
    off=ev['photostim_stop_times/timestamps'][:]
    if len(on)!=len(off): raise ValueError('photostim on/off count mismatch')
```

iii. Step 5 mapping table: "`photostim_start_times` / `photostim_stop_times` → `input[1]` … Time-varying `(80,)`; direct event timestamps avoid string `N/A` parsing." Step 2 recorded the reason: "Photostimulation trial-table fields use string `N/A` on unstimulated trials, while BehavioralEvents provide numeric timestamps only for actual stim events."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary, time-varying series: a bin is 1 when its centre falls in the half-open interval `[on, off)` of any laser event in the session, else 0. The mask is accumulated as a boolean OR over all stimulation events of the session and then cast to `float32` for storage in row 1 of the input array. Because the OR runs over all session events rather than only the trial's own event, stimulation belonging to a neighbouring trial that physically overlaps the −2.5/+1.5 s window is also marked on.

ii.
```python
result=np.zeros(absolute_centers.shape, dtype=bool)
for a,b in zip(on,off): result |= ((absolute_centers>=a)&(absolute_centers<b))
return result.astype(np.float32)
```
```python
stim=photostim_series(f,centers)
input_cube=np.stack((time_from_tone,stim),axis=1).astype(np.float32)
```

iii. Step 5 Key Decision 6: "**Photostimulation representation**: Mark bin centers inside `[stim_on, stim_off)`; the paper states stimulation ends before go, providing a temporal sanity check." Step 7 fixed an implementation bug here: "Initial sample run failed because a float32 photostimulation array was used with Boolean in-place OR. Changed the accumulator to Boolean and cast to float32 after interval construction." Step 10 Check 2 independently reconstructed "all 80 values … from raw stimulation on/off event intervals" with `np.allclose`.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Same mechanism as 3-c: the comparison is done in absolute time against `centers = go_k[:,None] + REL_CENTERS[None,:]`, the go-cue-relative bin-centre grid shared with the firing rates. No per-stream offset is applied because the laser event timestamps are on the same session clock as the spikes.

ii.
```python
go_k=go[kept]; centers=go_k[:,None]+REL_CENTERS[None,:]
...
stim=photostim_series(f,centers)          # compares centers against absolute [on, off)
```
```python
for a,b in zip(on,off): result |= ((absolute_centers>=a)&(absolute_centers<b))
```

iii. Same rationale as the other streams: "session-absolute spikes, events, and tracking must be aligned directly by timestamps" (Step 4). The AI also noted an expected sanity property of the alignment — "the paper states stimulation ends before go, providing a temporal sanity check" — and the `--show-processing` plot draws the photostim trace on the go-aligned axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the NWB file, so the AI derives it from two trials-table columns: `trials/outcome` (`'hit'`/`'miss'`/`'ignore'`) and `trials/trial_instruction` (`'left'`/`'right'`). A hit means the animal licked the instructed side, a miss means it licked the opposite side, and an ignore means it did not lick.

ii.
```python
outcome=decode_strings(trials['outcome'])[kept]
instruction=decode_strings(trials['trial_instruction'])[kept]
```
```python
'choice_derivation':'ignore=no lick; hit=instruction side; miss=opposite side, using curated NWB trial labels',
```

iii. Step 5 Key Decision 7: "**Choice**: Derive actual lick direction from the curated per-trial outcome/instruction labels: hits follow instruction, misses are opposite, ignores are no lick. A direct event comparison agreed overwhelmingly but had occasional timing/free-water edge cases, so curated labels are authoritative." The AI therefore explicitly considered and rejected deriving choice from the raw `left_lick_times`/`right_lick_times` streams.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. A nested `np.where` maps the two label columns to codes `0 = left`, `1 = right`, `2 = no lick`, stored `int8` in row 0 of the `(4, 80)` output array and repeated identically across all 80 bins (it is a per-trial quantity held in a time-varying container). `output_values[0] = ['left','right','no lick']`. Resulting distribution over the full dataset: left 0.428, right 0.422, no lick 0.150.

ii.
```python
choice=np.where(outcome=='ignore',2,np.where(outcome=='hit',
                 np.where(instruction=='left',0,1),
                 np.where(instruction=='left',1,0))).astype(np.int8)
```
```python
output_cube=np.empty((len(kept),4,N_BINS),dtype=np.int8)
output_cube[:,0,:]=choice[:,None]
```
```python
'output_names':['lick direction choice','outcome','early lick','tongue y-position'],
'output_values':[['left','right','no lick'], ...],
```

iii. Step 5 mapping table: "ignore→no lick; hit→instruction side; miss→opposite of instruction … Expanded across 80 bins so outputs share a time-varying matrix." Step 10 Check 2 verified "choice, outcome, and early-lick values for three specific raw trials" directly from the original NWB with `np.allclose`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `trials/outcome` column of the NWB trials table, which already contains exactly the three strings required by the instructions (`'ignore'`, `'miss'`, `'hit'`). No derivation from lick or reward streams.

ii.
```python
outcome=decode_strings(trials['outcome'])[kept]
```
```python
def decode_strings(dataset):
    """Robustly decode an HDF5 string-like 1-D dataset."""
    a = dataset[:]
    out=[]
    for x in a:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode('utf-8', errors='replace').strip())
        ...
```

iii. Step 2's variable table lists "`trials/outcome` | string per trial | `hit`, `miss`, `ignore`", and Step 5's mapping table records the transform simply as "map ignore/miss/hit to 0/1/2 | direct NWB label". The categories in the file are identical to those the Decoder Task asks for, so the AI treats the stored label as authoritative.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps the strings to `0 = ignore`, `1 = miss`, `2 = hit` (the order given in the instructions), stored `int8` in row 1 of the output array and repeated across all 80 bins. Resulting distribution: ignore 0.150, miss 0.167, hit 0.683 — the 68.3% hit rate is close to the data paper's reported mean correct rate of 84% once the retained early-lick/ignore/photostim trials that the paper excluded are taken into account.

ii.
```python
outcome_code=np.array([{'ignore':0,'miss':1,'hit':2}[x] for x in outcome],dtype=np.int8)
```
```python
output_cube[:,1,:]=outcome_code[:,None]
```
```python
'output_values':[..., ['ignore','miss','hit'], ...],
```

iii. Step 5: "Expanded across time." Step 9 checked the distribution: "Outcome: ignore, miss, and hit all represented; hit is the majority as expected." The dictionary lookup will raise `KeyError` on any unexpected label, which the AI treats as a deliberate fail-fast guard rather than silently coercing unknown values.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `trials/early_lick` column, which holds the strings `'no early'` and `'early'`. No re-derivation from the lick-time streams or from sample/delay replay events.

ii.
```python
early=decode_strings(trials['early_lick'])[kept]
```

iii. Step 2's variable table: "`trials/early_lick` | string per trial | `early`, `no early`". Step 1 noted that the reference loader likewise "propagates early-lick trial labels" from the behavioural table rather than recomputing it, so using the stored flag matches the reference pipeline.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A fixed dictionary maps `'no early' → 0`, `'early' → 1`, stored `int8` in row 2 of the output array and repeated across all 80 bins. Distribution: no 0.884, yes 0.116, consistent with the raw-data count the AI measured in Step 2 (no early 84,185 / early 10,805 over all 94,990 source trials).

ii.
```python
early_code=np.array([{'no early':0,'early':1}[x] for x in early],dtype=np.int8)
```
```python
output_cube[:,2,:]=early_code[:,None]
```
```python
'output_values':[..., ['no','yes'], ...],
```

iii. Step 5 mapping table: "no early→0, early→1 | loader early-lick field | Expanded across time." Step 12 revisited this output specifically because its validation accuracy (0.7476) sat just under the 1.5× chance heuristic: "Direct raw checks of three trials confirm early-lick labels and time-constant expansion exactly (`np.allclose`). Both classes are well represented across the full dataset; imbalance is handled by the supplied decoder's balanced loss … No conversion change is justified by the 0.0024 heuristic shortfall."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` — the series' own description string states the columns are `('tongue_x', 'tongue_y', 'tongue_likelihood')`. Column 1 (`tongue_y`) supplies the value and column 2 (the DeepLabCut likelihood) decides visibility; `timestamps` supplies the ~294 Hz frame times. Only the side camera is used.

ii.
```python
tg=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
video_t=tg['timestamps'][:]
...
tongue=np.asarray(tg['data'][:],dtype=np.float64)
visible_session=tongue[:,2]>=VISIBILITY_THRESHOLD
```

iii. Step 2: "`acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` exists in all sessions. Its data are `(n_video_frames,3)` float64 columns `(tongue_x, tongue_y, tongue_likelihood)`, with explicit timestamps; representative median interval is 3.4 ms (~294 Hz). All coordinates/likelihoods are finite, including low-confidence frames, so visibility must use likelihood rather than NaN detection." Step 3: "The method paper used side-view video only. This matches the Camera0 side-view stream present in every NWB session."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps. (1) Visibility: a frame counts as showing a protruded tongue when `tongue_likelihood >= 0.9`. (2) Session thresholds: the 40th and 60th percentiles of `tongue_y` are computed over the **visible raw frames of that session only** (invisible frames are excluded entirely rather than imputed). (3) Downsampling: each 50 ms bin takes the value of the single video frame **nearest its centre** — the signal is point-sampled, not averaged over the bin. The nearest frame's likelihood also decides that bin's visibility. Resulting distribution: 0.063 / 0.032 / 0.066 / 0.840, i.e. 84.0% of bins are 'not visible', against 75.0% in the human reference, which averages within the bin instead.

ii.
```python
VISIBILITY_THRESHOLD = 0.9
```
```python
def nearest_indices(sorted_t, query):
    idx=np.searchsorted(sorted_t, query)
    idx=np.clip(idx, 1, len(sorted_t)-1)
    left=idx-1
    choose_left=(query-sorted_t[left]) <= (sorted_t[idx]-query)
    return np.where(choose_left, left, idx)
```
```python
tongue=np.asarray(tg['data'][:],dtype=np.float64)
visible_session=tongue[:,2]>=VISIBILITY_THRESHOLD
if visible_session.sum()<2: raise ValueError(f'{path.name}: insufficient visible tongue frames')
q40,q60=np.percentile(tongue[visible_session,1],[40,60])
vi=nearest_indices(video_t,centers.ravel()).reshape(centers.shape)
y=tongue[vi,1]; likelihood=tongue[vi,2]
```

iii. Step 5 Key Decisions 8 and 9: "**Tongue visibility**: Use DLC likelihood >=0.9. Likelihood is strongly bimodal (near zero or one), and 0.9 is a standard conservative DLC confidence threshold. Percentiles use only visible frames, preventing occluded/noisy coordinates from corrupting thresholds." and "**Tongue temporal sampling**: Nearest video frame to each 50-ms bin center is sufficient at ~300 Hz (maximum typical mismatch ~1.7 ms), avoids smoothing across appearance boundaries, and preserves categorical visibility." Step 4 rejected the method paper's treatment: "Tongue occlusion | Mean-imputed for continuous marker models | … | Do not mean-impute invisible frames; classify visibility from DLC likelihood, then discretize visible y per session." Step 3/4 measured the likelihood distribution to support the threshold: "about 11.68% of frames have tongue likelihood ≥0.9."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Exactly as written in the Decoder Task: class 0 for `y < q40`, class 1 for `q40 <= y <= q60`, class 2 for `y > q60`, class 3 when the sampled frame is not visible (likelihood below 0.9). The three cases are exhaustive and mutually exclusive, and the array is pre-filled with 3 so that any bin never assigned a visible class remains 'not visible'. `q40`/`q60` are per-session and are recorded in `session_info` for every session.

ii.
```python
tongue_code=np.full(y.shape,3,dtype=np.int8)
vis=likelihood>=VISIBILITY_THRESHOLD
tongue_code[vis & (y<q40)]=0
tongue_code[vis & (y>=q40) & (y<=q60)]=1
tongue_code[vis & (y>q60)]=2
output_cube[:,3,:]=tongue_code
```
```python
'output_values':[..., ['below 40th percentile','40th to 60th percentile','above 60th percentile','not visible']],
'tongue_discretization':'session visible-frame y percentiles: <40, 40-60 inclusive, >60; likelihood<0.9 is not visible',
'tongue_y_percentile_40':float(q40), 'tongue_y_percentile_60':float(q60),
```

iii. Step 5 mapping table: "likelihood <0.9→3 not visible; visible y below session 40th percentile→0, 40th–60th→1, above 60th→2 … Percentiles computed from visible session frames only; preserve required invisible class." The per-session scope and the 40/60 split are taken verbatim from the Decoder Task specification. Step 10 Check 2 verified tongue classes against "nearest raw video frames, likelihood threshold, and raw-session visible-y percentiles" with `np.allclose`.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps share the session-absolute clock with the spikes and events, so alignment is a direct nearest-neighbour lookup of each absolute bin centre (`go + REL_CENTERS`) into the camera timestamp array — the same grid used for the firing rates, so bin *k* of the tongue output corresponds to bin *k* of the neural matrix. There is no interpolation and no offset correction. A trial-level guard requires the whole window to lie inside the session's video timestamp range, and such trials are dropped (802 dataset-wide). There is, however, **no maximum-distance guard** on the nearest-frame lookup: the video is trial-gated with inter-trial gaps up to ~1.15 s, and I measured that in typical sessions ~0.3% of bin centres take a frame up to ~0.36 s away (the AI's notes claim a "maximum typical mismatch ~1.7 ms", which holds for bins inside the video-on periods but not for bins falling in inter-trial gaps).

ii.
```python
video_mask=(go+OFF_START>=video_t[0]) & (go+OFF_END<=video_t[-1])
keep_trial=recorded_mask & video_mask
```
```python
go_k=go[kept]; centers=go_k[:,None]+REL_CENTERS[None,:]
...
vi=nearest_indices(video_t,centers.ravel()).reshape(centers.shape)
y=tongue[vi,1]; likelihood=tongue[vi,2]
```

iii. Step 5 Key Decision 9 (quoted above) and Key Decision 10: "`stop_time` may precede go+1.5 for ignore/aborted trials, but continuous streams remain valid and no requested window crosses the next trial. Do not clip solely to behavioral trial stop; exclude only genuine stream-coverage failures." Step 10 Check 2 independently re-derived the nearest-frame assignment from the raw NWB and Step 5's planned check "Assert no retained requested window crosses the next trial or falls outside video timestamps" is implemented as the `video_mask` guard.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Six distinct cases, each handled explicitly and fail-fast where the situation is unexpected:

- **Session never quality-controlled**: `units/classification` is float64 NaN instead of strings; detected by dtype and the whole session is skipped with a printed `SKIP` reason.
- **Non-string subject id**: falls back to the `sub-*` directory name.
- **Trials with no ephys**: excluded via the `obs_intervals` → `trials.start_time` mapping.
- **Trials with no spikes at all** (free-water and similar): excluded by the all-zero-neural test.
- **Trials whose window exceeds the video record**: excluded by `video_mask`.
- **Invisible tongue**: represented as an explicit fourth class (3) rather than imputed.
- **Missing/blank anatomical annotation**: relabelled `'Unknown'` so it gets a valid region index instead of an empty or `'nan'` string.

Anything the AI did *not* anticipate raises immediately — mismatched go/trial counts, inconsistent per-unit interval counts, unmappable observation intervals, go cue without a preceding tone, tone outside the trial, photostim on/off count mismatch, a window crossing the next trial, fewer than two usable trials, no stable good units, and unknown outcome/early-lick label strings (dictionary `KeyError`). Shape and finiteness assertions run on every session.

ii.
```python
if d.dtype.kind in 'OSU': kept.append(p)
else: skipped.append((p.name, 'missing classifier labels'))
```
```python
# Empty/missing native annotation gets an explicit name rather than an invalid index.
anno=[x if x and x.lower() not in ('nan','none') else 'Unknown' for x in anno]
```
```python
tongue_code=np.full(y.shape,3,dtype=np.int8)      # default: not visible
```
```python
assert neural_cube.shape==(len(kept),len(unit_ids),N_BINS)
assert input_cube.shape==(len(kept),2,N_BINS)
assert output_cube.shape==(len(kept),4,N_BINS)
assert np.isfinite(neural_cube).all() and np.isfinite(input_cube).all()
```

iii. Step 10 Check 5: "One source session has classifier labels entirely missing and is excluded. One retained session has only seven trials because its side video ends at 49.3 s; all seven retained trials have valid neural/video data and satisfy the format minimum. One other session has video only through part of the behavior; only jointly observed trials are retained. `classification` is non-string in the excluded session and is robustly detected." Step 9 documents the reasoning behind excluding rather than zero-filling: trials with no ephys "have no usable neural decoder input", and the initial version that treated them as normal produced "3,470 wholly zero neural trials" in the validator. The AI's general principle is to drop where nothing was recorded and to encode an explicit category where the measurement legitimately has no value.

## 10-a. What are the most time-consuming steps of the code?

i. Per-session HDF5 I/O dominates, followed by pickling. The full run took **154.6 s for 173 sessions** (~0.89 s/session, range ~0.20–1.29 s), scaling roughly with unit count. Within a session the costs are: reading the full tongue-tracking array (`tg['data'][:]`, e.g. 680,500 × 3 float64 ≈ 16 MB per session), the per-unit `spike_times` slice reads plus one `np.searchsorted` per unit over 81 × n_trials edges, and reading `is_good_trials` (n_units × n_trials bool). `eligible_files()` adds one extra open of all 174 files up front. Writing the 11.645 GB pickle is a substantial share of the wall clock beyond the 154.6 s of per-session processing. The script prints per-session timing and stores `processing_seconds` in each `session_info`.

ii.
```python
    for i,p in enumerate(files):
        ts=time.perf_counter(); print(f'[{i+1}/{len(files)}] {p.name}',flush=True)
        sess=process_session(p,show_processing=args.show_processing and i<2)
        print('  trials={retained_trials}/{source_trials}, neurons={retained_stable_good_units}, time={:.2f}s'.format(time.perf_counter()-ts,**sess['info']),flush=True)
```
```python
'processing_seconds':float(time.perf_counter()-t0)
```

iii. Step 6: "Code inefficiencies identified: Naive loops over trials x bins x spikes would be prohibitively slow. Repeatedly loading full video arrays or scanning spikes per trial would duplicate I/O." Step 7 estimated "Processing 0.86–1.29 s/session → ~3 minutes for 173 sessions" and "total conservatively <10 minutes"; Step 9 confirms "Full conversion took 154.6 s; runtime was well below the estimate and optimization threshold", so no further optimisation was pursued. The AI reads only the slices for selected good units rather than the whole concatenated spike buffer, which is why it beats the human reference's 247 s.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain:

1. **`bin_selected_units` per-unit loop** — one `searchsorted` per unit, but already vectorised across all trials and bins by flattening the edge array. This cannot be collapsed further because spike times are ragged (each unit has a different number of spikes), so it mirrors the human reference exactly.
2. **`photostim_series` per-event loop** — `for a,b in zip(on,off)` builds and ORs a full `(n_trials, 80)` boolean mask once per stimulation event, i.e. O(n_events × n_trials × 80). This one *is* genuinely vectorisable with a single `np.searchsorted` of the bin centres into the interleaved on/off boundaries. The AI did not identify it; with ~78–150 stim events per session it costs a few tens of ms per session, so it is not a bottleneck.
3. **`eligible_files` file loop and `decode_strings`' per-element Python loop** — inherently serial I/O and per-string decoding; the two dictionary-lookup list comprehensions for outcome/early-lick codes are similarly O(n_trials) Python but negligible.

ii.
```python
    for j,u in enumerate(unit_ids):
        st=np.asarray(unit_spike_slice(spikes, ends, int(u)), dtype=np.float64)
        cumulative=np.searchsorted(st, flat, side='left').reshape(edges.shape)
        counts[j]=np.diff(cumulative, axis=1)/BIN_S
```
```python
    result=np.zeros(absolute_centers.shape, dtype=bool)
    for a,b in zip(on,off): result |= ((absolute_centers>=a)&(absolute_centers<b))
```

iii. Step 6: "Code speedups added: For each unit, one vectorized `np.searchsorted` over all trial bin edges computes all 80-bin counts. Session tongue arrays are loaded once; nearest video indices and all classes are vectorized. Float32 neural/input storage and int8 categorical output storage reduce pickle size and I/O." The AI's stated position is that the remaining structure is inherent to the ragged spike storage, and that with a 154.6 s total run "runtime was well below the estimate and optimization threshold" (Step 9), so no further vectorisation was needed. It does not mention the photostim loop anywhere.

## 10-c. What processing does the code repeat multiple times?

i. One real repeat: **every NWB file is opened twice** — once in `eligible_files()` purely to read the dtype of `units/classification`, and again in `process_session()`, which re-reads the same column via `decode_strings`. The cost is one HDF5 open per file (~ms), so it is negligible, but the classifier column is genuinely read twice per session. Everything else is computed once per session: the tongue array is read once and reused for both the session percentiles and the per-bin lookup; `REL_EDGES`/`REL_CENTERS` are built once at module level; the `centers` matrix is built once and reused for the tone input, the photostim input and the tongue alignment. Some quantities are computed for all source trials and then subsetted by `[kept]` (`session_tone_onsets`, `decode_strings(trials['outcome'])`, etc.) — wasted work proportional to the ~4% of trials dropped, not a repeat.

ii.
```python
def eligible_files():
    for p in sorted(DATA_ROOT.rglob('*.nwb')):
        with h5py.File(p, 'r') as f:
            d=f['units/classification']          # opened here ...
```
```python
    with h5py.File(path,'r') as f:               # ... and again here
        ...
        classification=decode_strings(f['units/classification'])
```
```python
go_k=go[kept]; centers=go_k[:,None]+REL_CENTERS[None,:]   # built once, reused 3x
```

iii. Not discussed in CONVERSION_NOTES.md. The AI's stated efficiency principle (Step 6) is to avoid duplicating I/O — "Repeatedly loading full video arrays or scanning spikes per trial would duplicate I/O" — and it holds to that within `process_session`; the double open is a consequence of wanting session eligibility resolved before the conversion loop starts so that the `--sample` selection and the `SKIP` report are deterministic.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Very little, and nothing is silently thrown away:

- **`neural_window_mask`** is computed (whether each recorded trial's full window lies inside its observation interval) but is **not used for filtering** — its only consumer is the diagnostic field `excluded_incomplete_neural_window_trials` in `session_info`. This is the remnant of the rule the AI tried and rejected in Step 9.
- **`interval_counts`** and the several `raise ValueError` consistency checks are pure validation that produce no output.
- Tone onsets, outcome, instruction and early-lick strings are decoded for **all** source trials and then subsetted to `kept`, so work is done for the ~4% of dropped trials.
- `bin_selected_units` is run for *all* video/ephys-eligible trials before the all-zero-neural filter removes 2,414 of them, so those trials are binned and then discarded.
- `nearest_indices` and the `y`/`likelihood` gathers are computed for all 80 bins of every trial, including the 84% that turn out to be 'not visible' and whose `y` value is never used.
- `metadata['bin_centers_seconds']`, the per-session `q40`/`q60`, the counts in `session_info` and the `--show-processing` plots are all documentation the decoder never reads.

ii.
```python
neural_window_mask=np.zeros(nt,dtype=bool)
neural_window_mask[recorded_idx]=((go[recorded_idx]+OFF_START>=ref_intervals[:,0]) &
                                  (go[recorded_idx]+OFF_END<=ref_intervals[:,1]))
keep_trial=recorded_mask & video_mask          # neural_window_mask deliberately NOT used
```
```python
'excluded_incomplete_neural_window_trials':int(np.sum(recorded_mask & ~neural_window_mask)),
```
```python
tone_all=session_tone_onsets(f, starts, go)    # all trials, then subset
time_from_tone=(centers-tone_all[kept,None]).astype(np.float32)
```

iii. Step 9: "An intermediate overly strict rule requiring the entire requested window inside trial observation bounds disproportionately removed short ignore trials; it was rejected. Final logic preserves recorded short trials but excludes only wholly absent neural inputs." The AI kept the computation as a reported statistic rather than deleting it, so that the user can see how many trials *would* have been affected. The remaining items are validation and provenance metadata, which the instructions explicitly asked for ("Include sanity checks", "Add other relevant fields, e.g. `session_info`").
