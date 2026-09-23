# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is the DANDI NWB release, one `.nwb` file per session under `/app/data/sub-<id>/`. The AI enumerates every session with a single sorted glob over that layout and opens each file once with `pynwb.NWBHDF5IO`. Within a file it reads `nwb.subject`, `nwb.trials`, `nwb.units`, `nwb.acquisition['BehavioralEvents']` and `nwb.acquisition['BehavioralTimeSeries']`. 174 files are found; 173 are converted (one is skipped, see 2-c). All per-session results are accumulated in RAM and pickled at the end to a temp file that is then `os.replace`d onto `/app/converted_data.pkl`.

ii.
```python
ROOT='/app/data'; OUT='/app/converted_data.pkl'
...
def one_file(path):
    with NWBHDF5IO(path,'r',load_namespaces=True) as io:
        n=io.read(); tr=n.trials; u=n.units
        subject=s(n.subject.subject_id)
        ntr=len(tr)
...
def main():
    files=sorted(glob.glob(os.path.join(ROOT,'sub-*','*.nwb')))
    if not files: raise FileNotFoundError('No NWB files found')
    for z,path in enumerate(files,1):
        result=one_file(path)
```

iii. From the trajectory: the AI first inventoried `/app/data` (`174` NWBs, 28 subject directories, ~85 GB), confirmed `pynwb` was installed, and dumped the NWB schema (trial columns, unit columns, acquisition members) before writing any code. It concluded that "the NWB release … exposes `classification == 'good'`" and that the NWB file already contains the curated units and synchronized behavioral streams directly, so no auxiliary files (as used by the reference MATLAB/pickle pipeline in `/app/code`) were needed. It notes in the module docstring that "Spikes and all behavioral streams use the NWB master clock."

## 1-b. How are the data split into subjects?

i. The subject is read directly from `nwb.subject.subject_id` (a numeric string such as `'440956'`). `data['subjects']` is built in first-encounter order as sessions are processed, and `subject_idx` records each session's index into that list, cast to `int16`. This gives 28 subjects. Note the list is not sorted (the reference sorts it); because the file glob is sorted by subject directory, the encounter order happens to be the same sorted order anyway.

ii.
```python
subject=s(n.subject.subject_id)
...
if sub not in data['subjects']: data['subjects'].append(sub)
data['subject_idx'].append(data['subjects'].index(sub))
...
data['subject_idx']=np.asarray(data['subject_idx'],dtype=np.int16)
```

iii. No explicit justification is given in the trajectory beyond the schema inspection, which printed `SUBJECT 440956 SESSION None` and established that `subject_id` is the only animal identifier carried in the file. The AI did not attempt to recover the paper-facing mouse name (e.g. `SC015`) from `nwb.identifier`.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session; no grouping or splitting is performed. Session order follows the sorted file list (subject directory, then acquisition timestamp encoded in the filename, so chronological within a subject). The session identity recorded in the output is the file basename, stored per session in `metadata['session_info']` together with subject, trial count, unit count and the session's tongue percentiles.

ii.
```python
files=sorted(glob.glob(os.path.join(ROOT,'sub-*','*.nwb')))
...
info={'file':os.path.basename(path),'subject':subject,'n_trials':ntr,
      'n_good_units':len(keep),'tongue_y_percentiles':[float(q40),float(q60)],
      'dlc_visibility_threshold':DLC_VISIBLE}
```

iii. The AI verified from the schema dump that `nwb.session_id` is `None` and that each file holds a single trials table, so the file boundary is the session boundary. In step 23 it states the final dataset "contains the expected 173 analyzed sessions after excluding the single NWB whose classifier and anatomy labels are entirely missing", which it cross-checks against the 173 sessions reported in `methods.txt`.

## 1-d. How are the data split into trials?

i. Trials come straight from the NWB trials table (`nwb.trials`), one row per behavioural trial. The AI asserts that the number of `go_start_times` events equals the number of trial rows and raises if not, which establishes the one-go-cue-per-trial mapping. It explicitly does *not* use `sample_start_times` for trial boundaries, because that stream has more events than trials.

ii.
```python
ntr=len(tr)
go=np.asarray(ev['go_start_times'].timestamps[:],dtype=float)
sample=np.asarray(ev['sample_start_times'].timestamps[:],dtype=float)
if len(go)!=ntr:
    raise RuntimeError(f'{path}: {len(go)} go events != {ntr} trials')
```

iii. From step 13: "each NWB has explicit sample (tone) and go timestamps". From the code comment: "Occasional extra sample events are replays after an early lick" — the AI recognised (consistent with `methods.txt`: "Licking early during the sample/delay epoch triggered a replay of the epoch") that sample events are not 1:1 with trials, while go cues are, so the go-cue stream is the safe per-trial anchor.

## 1-e. How are trials filtered based on quality controls?

i. **No trial-level filtering is performed at all.** Every row of every retained session's trials table is emitted, including early-lick trials, `ignore` (no-response) trials, `free_water` trials, and trials that lie outside the units' `obs_intervals` (i.e. trials recorded before the ephys acquisition started). There is no code path that subsets `tr`, and the exported per-session trial count is always `ntr = len(tr)`.

ii. There is no filtering snippet to show; the loop runs over all trials unconditionally:
```python
        ntr=len(tr)
        ...
        for j,g in enumerate(go):
            fr=frcube[:,j,:].copy()
```
The only curation anywhere in the script is at the unit/session level:
```python
        cls=np.asarray([s(x).lower() for x in u['classification'][:]])
        keep=np.flatnonzero(cls=='good')
        if not len(keep):
            return None
```

iii. The stated justification is in the module docstring: "sessions in the release are already behaviorally curated". In step 12 the AI wrote that "The methods specify 173 preselected behavioral sessions and classifier-based good units; the NWB release matches this", i.e. it assumed the release had already been trial-curated upstream. It did notice a per-unit `is_good_trials` column ("indicating drift-based trial-level validity", step 11) but never used it, and it never inspected `units/obs_intervals` or the `free_water` trial column, both of which it had printed in its own schema dumps.

**Measured consequence (verified independently against the raw NWBs):** across the 173 retained sessions there are 94,370 trials; 3,510 of them (3.72%) have literally zero spikes on every good unit in the [-2.5, +1.5] s window — 3,510 is exactly the number the expert solution removes. Those trials are emitted as all-zero `(n_neurons, 80)` firing-rate matrices. For example in `sub-456773_ses-20191005T142316`, all 35 `free_water` trials have a total spike count of 0 versus a mean of 3,293 on ordinary trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (session-absolute seconds), restricted to the units whose `units/classification` is `'good'`, together with `BehavioralEvents/go_start_times` timestamps, which position the bin edges.

ii.
```python
cls=np.asarray([s(x).lower() for x in u['classification'][:]])
keep=np.flatnonzero(cls=='good')
...
go=np.asarray(ev['go_start_times'].timestamps[:],dtype=float)
...
spikes=[np.asarray(u['spike_times'][int(i)],float) for i in keep]
```

iii. The schema inspection established `spike_times` as the only neural representation in the file (there is no binned/rate representation, no `processing` module). The AI matched this to the reference pipeline, which also starts from raw spike times and bins them with `sliding_histogram`.

## 2-b. How is the `neural` data processed?

i. Per-bin spike counts converted to firing rate in Hz. For each good unit, `np.searchsorted(spike_times, all_edges, side='left')` gives the running spike count at every bin edge of every trial at once; `np.diff` along the edge axis yields the count per bin; dividing by `DT = 0.05` gives Hz. Results are stored `float32`. No smoothing, baseline subtraction, normalisation or z-scoring.

ii.
```python
        spikes=[np.asarray(u['spike_times'][int(i)],float) for i in keep]
        all_edges=go[:,None]+EDGES[None,:]
        frcube=np.empty((len(keep),ntr,len(CENTERS)),dtype=np.float32)
        for k,st in enumerate(spikes):
            frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT
```

iii. Code comment: "Using left insertion indices gives counts in [edge_i, edge_i+1), matching the half-open convention in the supplied reference implementation." The AI had read `sliding_histogram` in `/app/code/VideoAnalysisUtils/`, noting in step 6 that it uses "half-open 50-ms windows, returning rates in spikes/s" (`binSpikes/float(bin_width)`), and matched both the half-open convention and the Hz normalisation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only if `units/classification` (lower-cased) equals `'good'`; no individual QC metric (ISI violation, presence ratio, amplitude cutoff, drift) is thresholded, and the older `unit_quality` ('good'/'multi') column is deliberately not used. A session in which no unit is labelled `'good'` is dropped entirely — this removes exactly one file, `sub-440958_ses-20190216T162508`, whose `classification` and `anno_name` are NaN for all 1,852 units, leaving 173 sessions.

Separately (not a QC filter but a related curation choice): brain region for each neuron is taken from the *probe's* `electrode_group.location` JSON (`"left ALM"`, `"right Striatum"`, …) rather than from the per-unit CCF annotation `units/anno_name`.

ii.
```python
        # Classifier label is the QC criterion described in methods.txt.
        cls=np.asarray([s(x).lower() for x in u['classification'][:]])
        keep=np.flatnonzero(cls=='good')
        if not len(keep):
            return None
        regions=[probe_region(u['electrode_group'][int(i)]) for i in keep]
```
```python
def probe_region(unit_group):
    """Use broad recording location from probe metadata, not fine CCF layer."""
    loc=s(getattr(unit_group,'location','')).strip()
    try:
        d=json.loads(loc)
        return str(d.get('brain_regions',loc)).strip()
    except Exception:
        return loc or 'unknown'
```

iii. Step 11: "Unit curation has both spike-sorting `unit_quality` and classifier `classification`; the paper's newer QC is represented by `classification == 'good'`." Step 17, after the run crashed on the unlabelled file: "all 1,852 units have missing classifier labels and anatomy, explaining the 174 source files versus 173 paper sessions" — the AI queried that file directly (`classification: {'nan': 1852}`, `unit_quality: {'good': 1201, 'multi': 651}`) before deciding to skip it. For regions it reasoned (step 11) that "Probe `location` supplies broad side/region labels, while `anno_name` is fine anatomy" and chose the probe label to get the seven paper regions (ALM, Striatum, Thalamus, Midbrain, Medulla, ECT, BLA) × hemisphere.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No alignment/interpolation step is needed: spike times and go-cue timestamps share the NWB master clock. The trial-relative edge grid is added to each trial's go-cue time to build a `(n_trials, 81)` matrix of absolute edge times, and spikes are binned against it directly.

ii.
```python
DT=.05; OFF0=-2.5; OFF1=1.5
EDGES=np.arange(OFF0, OFF1+DT/2, DT, dtype=np.float64)
...
all_edges=go[:,None]+EDGES[None,:]
frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT
```

iii. Module docstring: "Spikes and all behavioral streams use the NWB master clock. Exactly 80 non-overlapping 50-ms bins cover [-2.5, 1.5) s around the go-cue onset." The AI verified from the schema dump that all event streams carry absolute `timestamps` arrays on one clock, so no per-stream offset correction is required.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins; 81 edges from -2.5 s to +1.5 s give exactly 80 non-overlapping bins per trial, identical for every trial and every session. `metadata['time_bin_size'] = 50.0` (ms), `n_time_bins = 80`, `off_start = -2.5`, `off_end = 1.5`. The grid is computed once at module scope and reused. No rebinning or resampling of an already-binned representation occurs — spikes are binned once at the target resolution. (Verified: this edge array is numerically identical to the expert's to within 1.4e-14.)

ii.
```python
DT=.05; OFF0=-2.5; OFF1=1.5
EDGES=np.arange(OFF0, OFF1+DT/2, DT, dtype=np.float64)
CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

iii. Step 12: "The requested 4-second window at 50 ms naturally gives 80 non-overlapping bins; unlike the reference sliding-window plotting code's center endpoints, decoder extraction should use edges from -2.5 to 1.5." The AI explicitly noticed that `sliding_histogram`'s bin-*center* convention would produce 81 centres with endpoint bins spilling outside the requested window, and chose edge-based binning to satisfy the decoder instructions instead.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times` timestamps (the sample-epoch/instruction-tone onsets), paired with each trial's go cue. For each trial the *last* sample onset at or before the go cue is taken as that trial's tone.

ii.
```python
        sample=np.asarray(ev['sample_start_times'].timestamps[:],dtype=float)
        ...
        # Occasional extra sample events are replays after an early lick. Associate
        # each trial with the last sample onset preceding its unique go cue.
        sample_for_trial=np.empty(ntr,float)
        for j,g in enumerate(go):
            q=sample[sample < g+1e-9]
            sample_for_trial[j]=q[-1] if len(q) else np.nan
```

iii. The code comment states the reason: early licks replay the sample epoch, so a trial may carry several sample onsets, and the last one before the go cue is the one the animal actually heard last. The AI confirmed from the schema dump that `sample_start_times` has more entries than trials (405 events for 368 trials in the first session) while `go_start_times` has exactly one per trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, absolute bin-centre times are formed as `go + CENTERS`, and the trial's tone time is subtracted, giving a continuous, monotonically increasing seconds-since-tone value per bin. Stored as `float32` as row 0 of the `(2, 80)` input array. No clipping, no zeroing before the tone, no discretisation.

ii.
```python
            times=g+CENTERS
            tone_elapsed=(times-sample_for_trial[j]).astype(np.float32)
            ...
            inputs.append(np.vstack((tone_elapsed,photo)).astype(np.float32))
```

iii. The instructions ask for "time from tone onset in seconds (continuous, time-varying)", and the AI implemented exactly that; no further justification was recorded. (This is algebraically identical to the expert's `CENTERS + (go - tone)`.)

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the centres of the same 80 bins used for the firing rates (`CENTERS` is derived from the same `EDGES` array), on the same absolute clock, so input bin *k* and neural bin *k* cover the same interval by construction.

ii.
```python
EDGES=np.arange(OFF0, OFF1+DT/2, DT, dtype=np.float64)
CENTERS=(EDGES[:-1]+EDGES[1:])/2
...
all_edges=go[:,None]+EDGES[None,:]     # neural
times=g+CENTERS                        # inputs/outputs
```

iii. Docstring: "Spikes and all behavioral streams use the NWB master clock." A single shared grid is used for every stream, so alignment is automatic.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The `BehavioralEvents/photostim_start_times` and `photostim_stop_times` timestamp arrays — i.e. the recorded stimulation intervals on the session clock — rather than the trials-table columns `photostim_onset`/`photostim_duration` that the expert used.

ii.
```python
        pstart=np.asarray(ev['photostim_start_times'].timestamps[:],float)
        pstop=np.asarray(ev['photostim_stop_times'].timestamps[:],float)
```

iii. Step 13: "each NWB has explicit sample (tone) and go timestamps, photostimulation intervals, lick events, and synchronized tongue tracking", and the plan was to use "binary photostimulation state" from those intervals. Using the event stream avoids parsing the `'N/A'`-padded string columns of the trials table. (Verified independently: the event onsets agree with `trial.start_time + photostim_onset` to within 9e-13 s, the durations are identical, and the resulting binary time series is bit-identical to the expert's in the sessions checked.)

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1, stored as `float32`) time series over the 80 bins: a bin is 1 if its centre falls in any half-open stimulation interval `[start, stop)`. Trials with no stimulation simply keep all-zero rows. It is a time-varying series, not a per-trial flag.

ii.
```python
            photo=np.zeros(len(CENTERS),dtype=np.float32)
            # State at bin centers for every stimulation interval.
            for a,b in zip(pstart,pstop): photo[(times>=a)&(times<b)]=1
```

iii. The instructions call for "whether photostimulation is on at every time point (discrete, time-varying)"; the code comment "State at bin centers for every stimulation interval" records the sampling convention. Because *all* session intervals are tested against each trial's absolute window (rather than only that trial's own stim), a stimulation that spills across a trial boundary into the 4 s window would still be captured.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Stimulation start/stop times and bin centres are both absolute session times, so the comparison is done directly on the shared clock, at the centres of the same 80 bins used for the firing rates.

ii.
```python
            times=g+CENTERS
            for a,b in zip(pstart,pstop): photo[(times>=a)&(times<b)]=1
```

iii. Same as 3-c: everything is on the NWB master clock, so no offset correction or resampling is needed.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the NWB trials table, so choice is derived from `trial_instruction` (`'left'`/`'right'`) combined with `outcome` (`'hit'`/`'miss'`/`'ignore'`): a hit means the animal licked the instructed side, a miss means it licked the opposite side, an ignore means it did not lick.

ii.
```python
        instr=np.asarray([s(x).lower() for x in tr['trial_instruction'][:]])
        outcome=np.asarray([s(x).lower() for x in tr['outcome'][:]])
        ...
            if outcome[j]=='ignore': choice=2
            elif outcome[j]=='hit': choice=0 if instr[j]=='left' else 1
            elif outcome[j]=='miss': choice=1 if instr[j]=='left' else 0
            else: raise ValueError(f'Unknown outcome {outcome[j]}')
```

iii. Step 12: "Choice should represent the actual response: right/left for hit or miss (derived from instruction plus correctness), and no lick for ignore." The AI had confirmed from the trial-column dump that the only available labels are instruction/outcome/early-lick, and it read in `methods.txt` that mice "reported the instruction by licking one of the two lick ports", so instruction × correctness determines the licked port.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0 = left`, `1 = right`, `2 = no lick`, written as an `int8` row broadcast across all 80 bins so that all four outputs share one `(4, 80)` per-trial array. `output_values[0] = ['left','right','no lick']`. An unrecognised outcome string raises, so the mapping is total.

ii.
```python
'output_names':['lick direction choice','outcome','early lick','tongue y-position'],
'output_values':[['left','right','no lick'],...]
...
out=np.vstack((np.full(len(CENTERS),choice,np.int8), ...))
```

iii. Step 12/13: "The decoder supports 2D time-varying outputs, which is necessary because tongue category varies over the 80 bins; trial-level outputs can be broadcast across those bins." The AI read `decoder.py`'s `SessionData` class to confirm that a `(doutput, T)` integer array is the expected form.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the NWB trials table, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome=np.asarray([s(x).lower() for x in tr['outcome'][:]])
```

iii. The AI's trial-table dump showed the column verbatim, so it saw that the source categories match the requested ones one-to-one; no derivation was needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped through a fixed dictionary to `0 = ignore`, `1 = miss`, `2 = hit` and broadcast as an `int8` row across all 80 bins (row 1 of the output array). The dictionary is re-created inside the per-trial loop.

ii.
```python
            omap={'ignore':0,'miss':1,'hit':2}
            ...
            out=np.vstack((np.full(len(CENTERS),choice,np.int8),
                           np.full(len(CENTERS),omap[outcome[j]],np.int8), ...))
```

iii. The code ordering follows the instructions' "Outcome (ignore, miss, hit)" listing, so `output_values[1]` names the codes in the requested order. No further justification recorded.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The `early_lick` column of the trials table, whose only values are `'no early'` and `'early'` (confirmed across all 173 sessions).

ii.
```python
early=np.asarray([s(x).lower() for x in tr['early_lick'][:]])
```

iii. The AI's trial-column dump showed this column directly; `methods.txt` explains that early licking during sample/delay triggers an epoch replay, which is why the flag exists.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `1` if the string starts with `'early'`, else `0`; broadcast as an `int8` row across all 80 bins (row 2). `output_values[2] = ['no','yes']`.

ii.
```python
            ecat=1 if early[j].startswith('early') else 0
```

iii. No explicit justification recorded. The `startswith` form is a defensive way to tolerate variants; because `'no early'` does not begin with `'early'`, the mapping is correct for both observed values.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, a `(n_frames, 3)` array of `(tongue_x, tongue_y, tongue_likelihood)` at ~294 Hz with matching absolute `timestamps`. Column 1 is the y-position; column 2 is the DeepLabCut likelihood used to decide visibility. Column 0 (x) is loaded but unused.

ii.
```python
        tongue=n.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
        tt=np.asarray(tongue.timestamps[:],float)
        td=np.asarray(tongue.data[:],dtype=np.float32)
```

iii. Step 9: "Tongue tracking has three columns that appear to be x, y, and confidence/likelihood." The AI inferred the layout from the printed sample values (two pixel-scale coordinates followed by a 0–1 value) and from the series description; it also grepped `/app/code` and `methods.txt` for an explicit tongue-visibility threshold and found none, so it fell back on a DeepLabCut convention (see 8-b).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Two steps. (1) A frame counts as showing a visible tongue when its y and likelihood are finite and the likelihood is `>= 0.9`. (2) The 40th and 60th percentiles of the y-values of *all visible raw frames in the session* are computed once per session and stored in `session_info`. No averaging, smoothing or interpolation of y is done; the per-bin value is a single raw frame's y (see 8-d).

ii.
```python
DLC_VISIBLE=.9
...
        visible=np.isfinite(td[:,1]) & np.isfinite(td[:,2]) & (td[:,2]>=DLC_VISIBLE)
        if visible.any(): q40,q60=np.percentile(td[visible,1],[40,60])
        else: q40,q60=np.nan,np.nan
```

iii. Module docstring: "DLC confidence >= .9 defines a visible tongue. Session-wide 40/60 percentiles are computed from all visible tongue-y samples, as requested." Step 13: "classify tongue as not visible when DLC confidence is below the conventional 0.9 threshold" — the AI adopted the standard DeepLabCut cutoff after failing to find one in the reference code or methods. (Verified: the likelihood distribution is strongly bimodal — 10.52% of frames are ≥0.9 versus 10.59% ≥0.5 — so the exact threshold is immaterial.)

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. `0` if `y < q40`, `1` if `q40 <= y <= q60`, `2` if `y > q60`, `3` if the sampled frame is not visible (default fill). `output_values[3] = ['below 40th percentile','40th to 60th percentile','above 60th percentile','not visible']`. Class 1 is closed on both ends here (the expert's `np.digitize` makes it `[q40, q60)`), which is a tie-breaking difference only.

ii.
```python
            ycat=np.full(len(CENTERS),3,dtype=np.int8)
            ycat[vis & (td[ix,1]<q40)]=0
            ycat[vis & (td[ix,1]>=q40) & (td[ix,1]<=q60)]=1
            ycat[vis & (td[ix,1]>q60)]=2
```

iii. This transcribes the instruction's discretisation literally ("0: < 40th percentile … 1: 40th to 60th … 2: > 60th … 3: not visible"), with per-session percentiles as specified.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Nearest-neighbour sampling at bin centres: for each of the 80 absolute bin-centre times, `searchsorted` on the camera timestamps finds the bracketing frames, the index is clipped to `[1, n_frames-1]`, and whichever of the two neighbouring frames is closer in time supplies both the y-value and the visibility verdict for that bin. Only that single frame is used; the other ~13 frames inside the 50 ms bin are ignored. Camera timestamps share the master clock with the spikes, so no offset correction is applied.

ii.
```python
            # Nearest camera frame at each bin center; no extrapolation beyond video.
            ix=np.searchsorted(tt,times); ix=np.clip(ix,1,len(tt)-1)
            ix-=((times-tt[ix-1]) <= (tt[ix]-times))
            vis=(td[ix,2]>=DLC_VISIBLE)&np.isfinite(td[ix,1])
```

iii. The code comment claims "no extrapolation beyond video", and the module docstring notes all behavioural streams are on the NWB master clock. No further reasoning is recorded in the trajectory.

**Measured consequence (verified independently):** sampling one frame instead of averaging the bin shifts the class balance markedly toward "not visible". For `sub-440956_ses-20190207T120657` the class distribution is `[0.046, 0.025, 0.055, 0.874]` versus `[0.075, 0.041, 0.090, 0.794]` for the expert's bin-mean method; for `sub-480927_ses-20210226T133429` it is `[0.096, 0.055, 0.117, 0.732]` versus `[0.157, 0.089, 0.195, 0.559]`. Also, the `np.clip` means a bin-centre outside the video's coverage silently snaps to the nearest available frame rather than being marked not-visible; in sessions with inter-trial video gaps up to 1.15 s this affects ~0.2–0.3% of bins, with nearest-frame distances up to 0.36 s.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three mechanisms, covering two of the three failure modes present in the dataset:

- **Session with no QC labels**: `classification` is NaN for all units; `s(x).lower()` renders it `'nan'`, no unit matches `'good'`, and `one_file` returns `None`, so the session is skipped with a printed message. (This was added reactively: the first run crashed with `RuntimeError: No classifier-good units`, and the AI patched the raise into a `return None`.)
- **Non-finite tracking values**: `np.isfinite` guards on both tongue y and likelihood before any comparison, so bad frames become "not visible" rather than propagating NaN.
- **No tone before the go cue**: `sample_for_trial` falls back to `np.nan`, which would silently produce NaN inputs; verified never to trigger (0 such trials across all 173 sessions).
- **Structural sanity check**: a mismatch between go-cue count and trial count raises rather than silently misaligning.
- **Not handled**: trials with no ephys coverage (outside `obs_intervals`) and `free_water` trials, which are emitted as all-zero neural matrices (see 1-e). There is also no NaN check on the firing rates or inputs before pickling.

ii.
```python
        if not len(keep):
            return None
...
        if len(go)!=ntr:
            raise RuntimeError(f'{path}: {len(go)} go events != {ntr} trials')
...
            sample_for_trial[j]=q[-1] if len(q) else np.nan
...
        visible=np.isfinite(td[:,1]) & np.isfinite(td[:,2]) & (td[:,2]>=DLC_VISIBLE)
        if visible.any(): q40,q60=np.percentile(td[visible,1],[40,60])
        else: q40,q60=np.nan,np.nan
...
        result=one_file(path)
        if result is None:
            print(f'[{z}/{len(files)}] SKIP {os.path.basename(path)}: no classifier-good units',flush=True)
            continue
```

iii. Step 16: "Conversion stopped at session 12 because that NWB has no units labeled `classification == good`. The source contains 174 NWBs while the paper reports 173 analyzed sessions, strongly suggesting this unitless/nonqualifying session should be excluded." Step 17 confirms it after querying the file. The writes go to `OUT+'.tmp'` and are then `os.replace`d, so a crash during serialisation cannot leave a corrupt pickle in place.

## 10-a. What are the most time-consuming steps of the code?

i. As written, the per-session costs are: reading the ragged `spike_times` (one HDF5 fancy read per good unit, up to ~900 units/session), reading the `(n_frames, 3)` tongue array (~680k × 3 doubles) and its timestamps, the `searchsorted` pass per unit over `(n_trials, 81)` edges, and the Python per-trial loop (which itself contains an inner loop over every stimulation interval). Serialising the 12.31 GB pickle at the end is a further one-off cost. Observed throughput after the optimisation was roughly 45–50 sessions/minute of processing.

The dominant cost in the *first* version was `np.histogram`, called once per unit per trial; the AI profiled this by interrupting the run and reading the traceback, then replaced it.

ii. The identified hot spot and its replacement:
```python
            for k,st in enumerate(spikes):
                fr[k]=np.histogram(st,bins=absedges)[0].astype(np.float32)/DT   # BEFORE
```
```python
        all_edges=go[:,None]+EDGES[None,:]
        frcube=np.empty((len(keep),ntr,len(CENTERS)),dtype=np.float32)
        for k,st in enumerate(spikes):
            frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT   # AFTER
```

iii. Step 17: "its nested unit-by-trial histogram loop is unnecessarily slow. Since spikes are on one continuous session clock, all trial-bin edge counts for a unit can be computed in one vectorized `searchsorted`, reducing hundreds of thousands of Python histogram calls per session to one operation per unit." Step 18 adds the mechanism: "`np.histogram` … sorts the full unit spike train each time. Spike times are already sorted." The AI verified the speed-up by observing it reached session 49 in one minute after the patch versus ~4 sessions/minute before.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorised the dominant loop (trial dimension of the spike binning) but left several others:

- `sample_for_trial`: a Python loop over trials that masks the *entire* `sample` array on every iteration — O(n_trials × n_samples). A single `np.searchsorted(sample, go, 'left') - 1` does the same work.
- The photostim inner loop `for a,b in zip(pstart,pstop)` runs inside the per-trial loop, so all ~150 session intervals are re-tested against all 80 bin centres for every one of ~550 trials (~7M element comparisons/session). A single `(n_trials, 80)` broadcast against the intervals, or an interval-membership `searchsorted`, would collapse it.
- The whole `for j,g in enumerate(go)` body (tone, photostim, choice, outcome, early lick, tongue, `np.vstack`) is per-trial Python; every one of these quantities is computable as an `(n_trials, 80)` array in one shot, as the expert's code does.
- The per-unit `regions=[probe_region(u['electrode_group'][int(i)]) for i in keep]` and `spikes=[np.asarray(u['spike_times'][int(i)],float) for i in keep]` do one HDF5/pynwb access per unit; the spike buffer can be read once in bulk (`spike_times.target.data` plus the index) as the expert does.
- The outer-loop region and subject bookkeeping uses `list.index`, a linear scan per unit.

ii. The loops in question:
```python
        for j,g in enumerate(go):
            q=sample[sample < g+1e-9]
            sample_for_trial[j]=q[-1] if len(q) else np.nan
```
```python
            for a,b in zip(pstart,pstop): photo[(times>=a)&(times<b)]=1
```
```python
        for r in regs:
            if r not in data['brain_regions']: data['brain_regions'].append(r)
            rid.append(data['brain_regions'].index(r))
```

iii. The AI's only recorded efficiency reasoning concerns the spike histogram (step 17/18). The remaining loops were never discussed; the per-unit loop over `spikes` is inherent to the ragged storage, but the per-trial loops are not.

## 10-c. What processing does the code repeat multiple times?

i. Several quantities are recomputed rather than hoisted:

- `probe_region` is called once per good unit and re-parses the probe's `location` JSON each time, even though a session has only 2–4 distinct probes (e.g. 459 JSON parses for 4 distinct strings in the first session).
- `omap={'ignore':0,'miss':1,'hit':2}` is rebuilt inside the per-trial loop.
- The photostim interval scan is repeated in full for every trial (10-b).
- `sample[sample < g+1e-9]` rescans the full session tone array for every trial.
- `data['brain_regions'].index(r)` / `data['subjects'].index(sub)` re-scan the accumulating lists per unit/session.
- `frcube[:,j,:].copy()` re-materialises each trial's slice after the cube was already built; `len(CENTERS)` is recomputed in ~6 places per trial.

Each NWB file is nevertheless opened and read exactly once, and the bin grid is built once at module scope, so there is no second pass over the data.

ii.
```python
        regions=[probe_region(u['electrode_group'][int(i)]) for i in keep]
...
            omap={'ignore':0,'miss':1,'hit':2}
...
            fr=frcube[:,j,:].copy()
```

iii. Not discussed in the trajectory. None of these affect correctness; they are constant-factor costs on top of the I/O-dominated runtime.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little, but not nothing:

- The whole `(n_frames, 3)` tongue array is cast to `float32` and held in memory although only columns 1 and 2 are ever read; column 0 (`tongue_x`) is loaded and discarded.
- `frcube` is built for the whole session and then copied trial-by-trial, so the neural data is materialised twice (peak ≈ 2× the session's rate cube).
- Firing rates, inputs and tongue categories are computed for the 3,510 trials that contain no spike data at all; that work — and the 12.31 GB pickle's share of it — is wasted at best and harmful at worst (see 1-e).
- `visible` is computed over every frame in the session purely to obtain two percentiles.
- Per-session `session_info` carries `tongue_y_percentiles` and `dlc_visibility_threshold`, which the decoder never reads (though they are useful provenance).

Everything else computed is written to the output and consumed by the decoder.

ii.
```python
        td=np.asarray(tongue.data[:],dtype=np.float32)   # x column never used
...
        frcube=np.empty((len(keep),ntr,len(CENTERS)),dtype=np.float32)
        ...
            fr=frcube[:,j,:].copy()
```

iii. Not discussed in the trajectory. The AI did choose `float32` for neural/inputs and `int8` for outputs, and `int16` for the index arrays, which keeps the output near the minimum size for the data it emits.
