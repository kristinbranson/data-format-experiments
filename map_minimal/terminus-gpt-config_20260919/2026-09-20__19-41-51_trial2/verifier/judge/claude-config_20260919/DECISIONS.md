# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session laid out as `/app/data/sub-<id>/*.nwb`. The AI finds every session with a single sorted glob over that layout (path hard-coded, not a CLI argument). Files are opened with **`h5py` directly** rather than `pynwb`, and the NWB internal paths are addressed by hand (`intervals/trials`, `units`, `acquisition/BehavioralTimeSeries/...`). Each file is opened **twice**: once in a pre-filter pass that reads only `units/classification` to decide whether the session has any QC-passing unit, and once in the main conversion loop. Within a session, trials come from the trials table, units/spikes from the `units` group, and video from `acquisition/BehavioralTimeSeries`. Subjects are parsed from the containing directory name.

ii.
```python
files=sorted(glob.glob('/app/data/sub-*/*.nwb'))
kept=[]
for p in files:
  with h5py.File(p,'r') as f:
    if np.any(text(f['units/classification'][:])=='good'): kept.append(p)
    else: print('Excluding session with no good units:',os.path.basename(p),flush=True)
files=kept
...
for fi,p in enumerate(files):
  with h5py.File(p,'r') as f:
    tr=f['intervals/trials']; starts=np.asarray(tr['start_time'][:],float); ntr=len(starts)
    ...
    u=f['units']; cls=text(u['classification'][:]); good=np.flatnonzero(cls=='good')
```

iii. From the trajectory, the AI first tried `pynwb` for schema discovery (steps 3–8) but switched to raw `h5py` for the conversion, reasoning that it needed to "stream efficiently and avoid loading all source data simultaneously" given a projected ~12 GB output (step 11: *"Implement `/app/convert_data.py` with direct HDF5 access"*). It counted the files up front (step 5: 174 NWB files, 50 GB, 28 subjects) and cross-checked totals against the paper (step 11: *"The dataset totals 174 sessions, 94,990 trials, and 69,453 good units; this nearly matches the paper's stated 69,943 good units across 173 sessions"*).

## 1-b. How are the data split into subjects?

i. Subjects are taken from the parent directory name of each NWB file, stripping the `sub-` prefix (e.g. `sub-440956` -> `'440956'`). `subjects` is the sorted set of unique ids over the retained files, and `subject_idx` is built per session from the per-session `session_info` records, so its order matches the session order in `neural`/`input`/`output`. The numeric DANDI subject id is used, not the mouse name (`SC015`, ...) used in the papers.

ii.
```python
subjects=sorted({os.path.basename(os.path.dirname(p)).removeprefix('sub-') for p in files})
subjmap={s:i for i,s in enumerate(subjects)}
...
sub=os.path.basename(os.path.dirname(p)).removeprefix('sub-')
sinfo.append({'file':os.path.basename(p),'subject':sub, ...})
...
'subject_idx':np.asarray([subjmap[x['subject']] for x in sinfo],dtype=np.int32),
```

iii. The AI never states an explicit justification for using the directory name over `nwb.subject.subject_id`; it verified early (step 5) that the directory layout enumerates subjects cleanly (`4 sub-440956`, `4 sub-440957`, ... 28 subjects) and used it as the grouping key. Result: 28 subjects, matching the dandiset.

## 1-c. How are the data split into sessions?

i. One NWB file == one session; no splitting or grouping is performed. Session order is the sorted file order (which is chronological within subject because the filename embeds the acquisition timestamp). Exactly one session is dropped: the recording with zero QC-passing units. Per-session provenance is stored in `metadata['session_info']` keyed on the file basename rather than on `nwb.identifier`.

ii.
```python
files=sorted(glob.glob('/app/data/sub-*/*.nwb'))
# A recording with no QC-passing units cannot furnish decoder input; the paper's
# analyzed set likewise contains 173 sessions rather than all 174 NWB assets.
kept=[]
for p in files:
  with h5py.File(p,'r') as f:
    if np.any(text(f['units/classification'][:])=='good'): kept.append(p)
...
sinfo.append({'file':os.path.basename(p),'subject':sub,'n_trials':ntr,'n_good_units':len(good), ...})
```

iii. The AI initially kept all 174 files, then the format verifier flagged an empty `brain_region_idx` (step 21: *"The full verifier found one or more sessions with zero retained good units... Such sessions cannot support neural decoding and should be excluded. This likely explains why the provided directory has 174 files while the paper reports 173 analyzed behavioral sessions."*). It then identified the single offender (step 22: *"Exactly one file has zero good units: `sub-440958_ses-20190216T162508...`. Excluding it yields 173 sessions, matching the paper"*) and regenerated the dataset.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table (`intervals/trials`), read once per session; `ntr = len(start_time)`. Every per-trial quantity (labels, photostim, tongue, neural window) is indexed by that row order. No cross-check is made against any event stream, because the AI concluded (incorrectly, see 2-d) that the file contained no per-trial cue events.

ii.
```python
tr=f['intervals/trials']; starts=np.asarray(tr['start_time'][:],float); ntr=len(starts)
outcome=text(tr['outcome'][:]); instr=text(tr['trial_instruction'][:]); early=text(tr['early_lick'][:])
pon=text(tr['photostim_onset'][:]); pdur=text(tr['photostim_duration'][:]); ppow=text(tr['photostim_power'][:])
```

iii. Step 7: *"Trial tables also lack explicit go/tone columns, so alignment likely relies on fixed task timing relative to `start_time` and trial protocol."* The trials table is treated as the authoritative trial definition, which it is; the AI simply never located `acquisition/BehavioralEvents`, so it had no second source to reconcile against.

## 1-e. How are trials filtered based on quality controls?

i. **No trial filtering is applied at all.** Every row of every retained session's trials table is emitted (94,370 trials over 173 sessions). Early-lick trials and `ignore` (no-response) trials are deliberately kept. Trials with no spike data are *not* excluded: `units/obs_intervals` is never read, and `free_water` is never consulted. The only curation is at the session level (zero good units).

ii.
```python
# nothing between reading the trial table and the per-trial loop filters rows
for ti in range(ntr):
    ...
# metadata records the rationale:
'trial_filter':'all trials retained because ignore and early lick are requested outputs; sessions with zero QC-passing units excluded'
```
Module docstring:
```python
* Keep every supplied session/trial: ignore and early-lick trials are required targets.
```

iii. Step 5: *"the paper analyzed 173 behavior sessions and excluded early-lick/no-response trials for its movement analysis, but this conversion must retain those trial types because they are explicitly requested decoder outputs."* This is a sound and explicitly reasoned departure from the data paper. However, the AI offers no justification for retaining trials that carry **no spike data**, because it never examined `obs_intervals` or `free_water`. Measured on the delivered pickle, 3,512 of 94,370 trials (3.7%) have an all-zero neural matrix — 1,060 trials in 8 sessions that fall before the ephys `obs_intervals` start, plus ~2,450 `free_water` trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Spike times from the ragged `units/spike_times` + `units/spike_times_index` datasets, restricted to units whose `units/classification == 'good'`. The trial window is defined from **`intervals/trials/start_time`** — the AI uses trial start plus a hard-coded 2.5 s constant as its stand-in for the go cue, rather than `acquisition/BehavioralEvents/go_start_times`, which it never found. `units/anno_name` supplies each retained unit's brain region.

ii.
```python
u=f['units']; cls=text(u['classification'][:]); good=np.flatnonzero(cls=='good')
if 'anno_name' in u:
  regs=text(u['anno_name'][:])[good]
else:
  regs=np.repeat('unknown',len(good))
...
allsp=np.asarray(u['spike_times'][:],float); ends=np.asarray(u['spike_times_index'][:],int)
begins=np.r_[0,ends[:-1]]
```

iii. Step 10: *"the requested 4-second interval likely begins at trial start and go cue/tone occurs 2.5 seconds later."* Step 11: *"Based on task timing and late-delay stimulation values, go cue is 2.5 s after trial start, making the requested window exactly trial start through 4.0 s."* Region annotation: code comment *"Paper analyses use anatomical annotation attached to each classified unit."*

## 2-b. How is the `neural` data processed?

i. Per good unit, spikes falling in the trial window are histogrammed into 80 fixed 50 ms bins with `np.bincount` on `floor((t - window_start)/0.05)`, then divided by the bin width to give firing rate in Hz (`float32`). Bins are left-closed / right-open. No smoothing, normalisation, baseline subtraction, or rate thresholding is applied. The result is stored per trial as `(n_good_units, 80)`.

ii.
```python
sess_n=[np.empty((len(good),NBIN),dtype=np.float32) for _ in range(ntr)]
for row,ui in enumerate(good):
  sp=allsp[begins[ui]:ends[ui]]
  for ti,s in enumerate(starts):
    # integer indexing is stable and gives [start,start+4) bins
    lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
    idx=np.floor((sp[lo:hi]-s)/DT).astype(np.int64)
    sess_n[ti][row]=np.bincount(idx,minlength=NBIN)[:NBIN]/DT
```
```python
'neural_units':'firing rate (spikes/s)','bin_convention':'left-closed, right-open; values sampled at bin centers',
```

iii. Step 9: *"implement an HDF5-based converter using only good units, 80 half-open 50-ms bins from go cue -2.5 to +1.5"*. The code comment justifies the integer-index form as *"integer indexing is stable and gives [start,start+4) bins"*. No justification is given for using raw firing rate rather than any transform; the instructions ask only for firing rates in 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept; all other classes (`unlabelled`, etc.) are dropped. No thresholds on any individual quality metric (ISI violation, presence ratio, amplitude, firing rate) are applied, and `units/unit_quality` is not used. A session with zero such units is dropped entirely (the one 174th file). This retains 69,453 units over 173 sessions.

ii.
```python
u=f['units']; cls=text(u['classification'][:]); good=np.flatnonzero(cls=='good')
```
```python
if np.any(text(f['units/classification'][:])=='good'): kept.append(p)
else: print('Excluding session with no good units:',os.path.basename(p),flush=True)
```
```python
'unit_filter':"NWB units.classification == 'good' (paper QC classifier)",
```

iii. Step 4/5: *"The methods establish the key curation rule for neurons: retain only units whose `classification` is `good`; the sample session has 459 good of 1,952 total clusters."* Step 10: *"paper-wide quality-control counts suggest retaining all `good` units."* Step 11 cross-checks the total: *"69,453 good units; this nearly matches the paper's stated 69,943 good units across 173 sessions."* The docstring records it as *"matching the paper's region-specific QC classifiers"* (i.e. the Chen/Liu et al. 2023 spike-sorting QC classifier).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instructions require alignment to **go cue onset**, window −2.5 s to +1.5 s. The AI does **not** use the recorded go-cue times. It assumes the go cue occurs at a **fixed 2.5 s after `trials.start_time`** on every trial, and therefore bins spikes over the absolute interval `[start_time, start_time + 4.0)`, labelling bin centres as `-2.475 … +1.475` s "relative to go". `metadata` records `temporal_alignment_event = 'auditory Go cue onset'`, `off_start=-2.5`, `off_end=1.5`, and `go_cue_from_trial_start_s = 2.5`.

ii.
```python
DT=.05; NBIN=80; GO_FROM_START=2.5; TONE_FROM_START=.5; DLC_THRESHOLD=.9
...
# Bin centers relative to go; tone starts at -2.0 s.
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
...
    lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')   # s = trials.start_time
```
```python
'temporal_alignment_event':'auditory Go cue onset','off_start':-2.5,'off_end':1.5,
'go_cue_from_trial_start_s':2.5,
```

iii. Step 7: *"Trial tables also lack explicit go/tone columns, so alignment likely relies on fixed task timing relative to `start_time` and trial protocol."* Step 9: *"The likely task timing is fixed: trial start is 2.5 seconds before the auditory go cue, matching the requested window exactly."* Step 11: *"Based on task timing and late-delay stimulation values, go cue is 2.5 s after trial start."* Step 13 acknowledges the tension: *"Reference preprocessing confirms go-cue alignment is the intended convention, though the NWB omits the explicit per-trial cue field."* The premise is false — every NWB file contains `acquisition/BehavioralEvents/go_start_times` (and `sample_start_times`, `delay_start_times`, `photostim_start_times`, `left/right_lick_times`). The string `go_start_times` never appears anywhere in the trajectory; the AI's HDF5 path scan filtered on the keywords `tongue|lick|stim|cue|tone|camera|video`, none of which match `go_start_times`.

Measured over all 94,990 trials: `go − start_time` has median 3.15 s, mean 3.24 s, range 2.11–11.73 s. **99.97 % of trials have a go cue that is not within ±50 ms of `start_time + 2.5`**, and on 5.6 % of trials the true go cue falls entirely outside the extracted 4 s window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins (`DT = .05`), 80 bins per trial spanning a 4 s window, identical for every trial and session; `metadata['time_bin_size'] = 50.0` (ms). Spikes are binned once directly from raw spike times — there is no intermediate resolution and therefore no rebinning. The tongue stream (~294 Hz video) is *downsampled* to the same grid by nearest-frame point sampling at bin centres rather than by averaging within the bin.

ii.
```python
DT=.05; NBIN=80
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
...
idx=np.floor((sp[lo:hi]-s)/DT).astype(np.int64)
sess_n[ti][row]=np.bincount(idx,minlength=NBIN)[:NBIN]/DT
```
```python
'time_bin_size':50.0, 'bin_convention':'left-closed, right-open; values sampled at bin centers',
```

iii. The 50 ms width and the ±window are taken directly from the Decoder Task spec (step 9: *"80 half-open 50-ms bins from go cue -2.5 to +1.5"*). Binning once from spike times avoids any resampling error; the fixed grid guarantees the constant 80 timepoints the target format requires.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. **None.** The value is derived entirely from two hard-coded constants, `GO_FROM_START = 2.5` and `TONE_FROM_START = 0.5`: the AI assumes the sample tone always starts 0.5 s after `trials.start_time`, i.e. exactly 2.0 s before its assumed go cue. `acquisition/BehavioralEvents/sample_start_times`, which records the actual tone onsets, is never read. Consequently `time_from_tone` is a single 80-element vector reused for every trial in every session.

ii.
```python
DT=.05; NBIN=80; GO_FROM_START=2.5; TONE_FROM_START=.5; DLC_THRESHOLD=.9
...
# Bin centers relative to go; tone starts at -2.0 s.
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
time_from_tone=rel_centers-(-GO_FROM_START+TONE_FROM_START)
...
sess_i.append(np.stack((time_from_tone.astype(np.float32),stim)))
```
```python
'tone_onset_from_trial_start_s':0.5,
```

iii. Step 9/11: the AI inferred the epoch structure from `methods.txt` (sample tone 3×150 ms with 100 ms gaps = 0.65 s, then a 1.2 s delay) and asserted *"Tone/sample onset is 0.5 s after trial start, or -2.0 s relative to go cue."* No data check of this constant was performed. In fact `sample_start_times` gives tone onset at median 1.29 s (range 1.16–4.95 s) after trial start — **100 % of trials are inconsistent with the assumed 0.5 s** — and go − tone is median ~1.90 s (range 0.95–10.4 s), not a constant 2.0 s, because an early lick replays the sample/delay epoch.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A single arithmetic shift of the bin-centre vector: `time_from_tone = rel_centers + 2.0`, evaluated once before the session loop and cast to `float32` when stacked into each trial's input array. The resulting continuous ramp runs from **−0.475 s to +3.475 s** in 50 ms steps and is byte-identical across all 94,370 trials (verified on the delivered pickle). It is stored as row 0 of `input`, with `input_names[0] = 'time from tone onset'`.

ii.
```python
time_from_tone=rel_centers-(-GO_FROM_START+TONE_FROM_START)
...
for ti in range(ntr):
    ...
    sess_i.append(np.stack((time_from_tone.astype(np.float32),stim)))
```

iii. The AI treats tone onset as a deterministic function of the trial clock, so the only "processing" required is an offset. No justification is given for keeping it constant across trials; step 11 simply lists *"Construct time-from-tone and binary photostimulation inputs."* The instructions call for a continuous, time-varying input, which the encoding technically satisfies in form, but because the offset carries no per-trial information the input contributes nothing beyond a fixed positional code — and the true offset (tone-to-window-start) varies by up to ~3.8 s across trials.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is defined on exactly the same 80-bin grid used for the spike histogram (`rel_centers`), so element *k* of the input corresponds to bin *k* of `neural` by construction. No resampling or interpolation. The alignment mechanism is therefore internally consistent, but the *event* being aligned to is an assumed fixed offset rather than the recorded tone onset.

ii.
```python
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
time_from_tone=rel_centers-(-GO_FROM_START+TONE_FROM_START)
```
```python
lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
idx=np.floor((sp[lo:hi]-s)/DT).astype(np.int64)
```

iii. Both streams are derived from `starts[ti]` plus a shared constant grid, so no explicit alignment step is needed. The AI documents the convention in `metadata['bin_convention'] = 'left-closed, right-open; values sampled at bin centers'`.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Three string columns of the trials table: `photostim_onset` (seconds from trial start), `photostim_duration` (seconds), and `photostim_power` (used purely as the present/absent flag). Control trials carry the literal string `'N/A'` in all three.

ii.
```python
pon=text(tr['photostim_onset'][:]); pdur=text(tr['photostim_duration'][:]); ppow=text(tr['photostim_power'][:])
...
if ppow[ti] != 'N/A':
    a=float(pon[ti])-GO_FROM_START; b=a+float(pdur[ti])
```

iii. Step 8: *"The eligibility calculation treated control photostimulation incorrectly: control power is evidently NaN rather than zero."* Step 9: *"`photostim_power` is nonnumeric (likely text such as `N/A`)... Inspect raw photostimulation values/dtypes."* Step 11: *"Photostimulation is encoded as `N/A` for control or power `5.500`, with onset/duration relative to trial start."* The AI verified the encoding empirically before coding. (I confirmed `photostim_onset != 'N/A'` and `photostim_power != 'N/A'` select the identical trials, and that `start_time + photostim_onset` reproduces `BehavioralEvents/photostim_start_times` exactly.)

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each stimulated trial the onset/offset are converted to the bin grid's coordinate system and a bin is set to 1 when its **centre** falls in `[onset, onset+duration)`, else 0 — a binary time series, not a per-trial flag. Control trials get an all-zero row. Stored as row 1 of `input` (`float32`), named `'photostimulation on'`. In the delivered pickle 18,248 of 94,370 trials carry a non-zero photostim row.

ii.
```python
stim=np.zeros(NBIN,dtype=np.float32)
if ppow[ti] != 'N/A':
    a=float(pon[ti])-GO_FROM_START; b=a+float(pdur[ti])
    # state at bin centers
    stim[(rel_centers>=a)&(rel_centers<b)]=1
sess_i.append(np.stack((time_from_tone.astype(np.float32),stim)))
```

iii. Step 11: *"Construct time-from-tone and binary photostimulation inputs."* The instructions ask for "whether photostimulation is on at every time point", i.e. a discrete time-varying signal; sampling the on/off state at bin centres is the matching encoding, and the code comment says exactly that (*"state at bin centers"*). The `'N/A'` guard keeps the string columns from being coerced.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. `photostim_onset` is trial-start-relative, and the neural bins are also laid out from `trials.start_time`; subtracting `GO_FROM_START` puts the onset into the same `rel_centers` coordinate as the bins, so bin *k* of the photostim row covers exactly the same absolute interval as bin *k* of the firing rates. Unlike the tone input, this row is therefore **correct in absolute time**, because both the stimulus times and the bin grid are anchored to the same variable (`start_time`); the 2.5 s constant cancels out.

ii.
```python
a=float(pon[ti])-GO_FROM_START; b=a+float(pdur[ti])
stim[(rel_centers>=a)&(rel_centers<b)]=1
```
where
```python
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT      # == (i+0.5)*DT - 2.5
lo=np.searchsorted(sp,s,'left')                          # bins start at s = start_time
```

iii. No separate justification is recorded; the AI established in step 11 that onset/duration are *"relative to trial start"* and consistently expressed both streams against trial start.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The trials table has no lick-direction column, so choice is reconstructed from the pair (`trial_instruction` ∈ {`left`,`right`}, `outcome` ∈ {`hit`,`miss`,`ignore`}): a hit means the animal licked the instructed port, a miss the opposite port, and an `ignore` means it did not lick. The recorded lick event streams (`left_lick_times`/`right_lick_times`) are not used.

ii.
```python
outcome=text(tr['outcome'][:]); instr=text(tr['trial_instruction'][:])
...
# Actual lick choice: correct instruction on hit, opposite on miss, no lick on ignore.
if outcome[ti]=='ignore': choice=2
elif outcome[ti]=='hit': choice=0 if instr[ti]=='left' else 1
else: choice=1 if instr[ti]=='left' else 0
```

iii. Step 10: *"Choice is not directly stored but can be reconstructed exactly: hit means lick in instructed direction, miss means opposite direction, and ignore means no lick."* The AI had earlier enumerated the category vocabularies from the data (step 6: `trial_instruction {'right': 228, 'left': 140}`, `outcome {'hit': 161, 'miss': 135, 'ignore': 72}`), so the mapping is grounded in the observed value sets.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0 = left`, `1 = right`, `2 = no lick`, per the instructions' ordering, and written into row 0 of the per-trial output array, tiled across all 80 bins so the output is time-varying in shape. `output_values[0] = ['left','right','no lick']`. Delivered class balance: 3,235,920 / 3,186,080 / 1,127,600 bin-samples (≈ 42.9 % / 42.2 % / 14.9 %).

ii.
```python
sess_o.append(np.vstack((np.full(NBIN,choice,dtype=np.int64),
                         np.full(NBIN,oc,dtype=np.int64),
                         np.full(NBIN,el,dtype=np.int64),
                         tongue[ti])))
```
```python
'output_names':['lick direction choice','outcome','early lick','tongue y-position'],
'output_values':[['left','right','no lick'],['ignore','miss','hit'],['no','yes'],[...]],
```

iii. The instruction lists the values as "left, right, no lick", and the AI's codes follow that order. Tiling the per-trial scalar across bins keeps all four outputs in one `(n_output, n_timepoints)` array, satisfying the format note "If at all possible, make it time-varying".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which already stores exactly the three strings `'ignore'`, `'miss'`, `'hit'`. No derivation.

ii.
```python
outcome=text(tr['outcome'][:])
```
(`text()` decodes the HDF5 byte strings.)

iii. Step 6: *"Trial labels are directly available: `early_lick` uses `no early`/`early`, and `outcome` uses `hit`/`miss`/`ignore`."* The stored vocabulary matches the requested categories one-for-one, so the AI used it as-is.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps the strings to `0 = ignore`, `1 = miss`, `2 = hit` (the order given in the instructions), written to row 1 of the output array and tiled across all 80 bins. `output_values[1] = ['ignore','miss','hit']`. Delivered class balance: 1,127,600 / 1,242,800 / 5,179,200 bin-samples (≈ 14.9 % / 16.5 % / 68.6 %).

ii.
```python
oc={'ignore':0,'miss':1,'hit':2}[outcome[ti]]
...
sess_o.append(np.vstack((..., np.full(NBIN,oc,dtype=np.int64), ...)))
```

iii. No separate justification beyond the ordering matching the instruction text ("Outcome (ignore, miss, hit, per-trial)"). Using a dict lookup with no default means an unexpected outcome string would raise rather than be silently miscoded.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `early_lick` column of the trials table, which stores `'no early'` / `'early'`.

ii.
```python
early=text(tr['early_lick'][:])
```

iii. Step 6: *"`early_lick` uses `no early`/`early`"* — verified from the value counts of a sample session (`{'no early': 350, 'early': 18}`). The flag is stored explicitly, so no derivation from lick times is needed.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to `0 = no`, `1 = yes` by an equality test against `'early'`, written to row 2 and tiled across all 80 bins. `output_values[2] = ['no','yes']`. Delivered balance: 6,690,080 / 859,520 bin-samples (≈ 88.6 % / 11.4 %).

ii.
```python
el=1 if early[ti]=='early' else 0
...
sess_o.append(np.vstack((..., np.full(NBIN,el,dtype=np.int64), ...)))
```

iii. Follows the instruction ordering ("Early lick (no, yes, per-trial)"). Unlike the outcome dict, this uses an `else 0` fallback, so any unexpected string would silently become "no"; in practice only the two documented values occur.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = (tongue_x, tongue_y, DeepLabCut likelihood) with a matching `timestamps` vector on the session clock (~294 Hz). Column 1 supplies the y value; column 2 gates visibility.

ii.
```python
# Tongue side view: columns x, y, DLC likelihood. Camera0 exists in all files.
tg=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
td=np.asarray(tg['data'][:],float); tt=np.asarray(tg['timestamps'][:],float)
```

iii. The AI initially concluded the video was missing (step 7: *"tracked tongue coordinates are not present in the provided NWBs"*) and corrected itself in step 8: *"The full path scan confirms tracked video data are present under `acquisition/BehavioralTimeSeries`, including side-camera tongue tracking and timestamps. Earlier acquisition inspection missed these because `BehavioralTimeSeries` is a container."* Step 11: *"Every session has side-camera tongue tracking. Tracking data columns are x, y, and DeepLabCut likelihood; the tongue is effectively not visible when confidence is low."*

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Point sampling, not averaging. For every (trial, bin) the **single nearest video frame to the bin centre** is selected by `searchsorted` plus a midpoint comparison; its y and likelihood are taken. A sample is "visible" iff y and likelihood are both finite **and** likelihood ≥ 0.9. The per-session 40th/60th percentiles are then computed over **only the visible sampled values inside the retained trial windows** (`visvals`), i.e. over the same quantity that is subsequently discretised. The ~15 other frames per bin are discarded.

ii.
```python
DLC_THRESHOLD=.9
...
# Nearest video frame to each 50-ms center (video is ~300 Hz).
sample_t=(starts[:,None]+(np.arange(NBIN)+.5)*DT).ravel()
jj=np.searchsorted(tt,sample_t); jj=np.clip(jj,1,len(tt)-1)
jj-=((sample_t-tt[jj-1]) <= (tt[jj]-sample_t))
yy=td[jj,1].reshape(ntr,NBIN); conf=td[jj,2].reshape(ntr,NBIN)
visible=np.isfinite(yy)&np.isfinite(conf)&(conf>=DLC_THRESHOLD)
visvals=yy[visible]
q40,q60=np.percentile(visvals,[40,60]) if len(visvals) else (np.nan,np.nan)
```

iii. Docstring: *"DLC tongue detections with likelihood < .9 are treated as not visible. Session percentiles are computed from all visible samples falling in retained windows."* Step 13: *"No reference code applies a DLC likelihood threshold, so the chosen 0.9 threshold is a documented, conventional visibility decision."* Step 14 addresses the resulting sparsity: *"The high not-visible fraction for tongue is plausible because the tongue appears only briefly during licking and low-confidence DLC detections are intentionally classified as not visible."* (The threshold choice is immaterial in practice — likelihood is bimodal; ≥0.9 selects 8.81 % of frames and ≥0.5 selects 8.89 %. The point-sampling choice does matter: it yields 11.3 % visible bins, versus ~25 % if any visible frame in the bin counts.)

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Exactly the four classes the instructions specify, using the per-session `q40`/`q60`: `0` if `y < q40`, `1` if `q40 ≤ y ≤ q60`, `2` if `y > q60`, and `3` for any bin whose sampled frame is not visible. The array is pre-filled with `3` so the not-visible case is the default. Delivered bin counts: 340,467 / 170,219 / 340,467 / 6,698,447 — i.e. an exact 40/20/40 split within the 11.3 % of bins that are visible.

ii.
```python
tongue=np.full((ntr,NBIN),3,dtype=np.int64)
tongue[visible&(yy<q40)]=0; tongue[visible&(yy>=q40)&(yy<=q60)]=1; tongue[visible&(yy>q60)]=2
```
```python
'output_values':[..., ['below 40th percentile','40th to 60th percentile','above 60th percentile','not visible']],
'tongue_processing':'nearest Camera0 side frame to each bin center; DLC likelihood >=0.9 visible; visible y discretized by per-session 40th/60th percentiles',
```
`session_info` also records the realised edges per session: `'tongue_y_percentiles':[float(q40),float(q60)]`.

iii. The cut points and the "per-session" scope are taken verbatim from the Decoder Task spec; the fourth class is the spec's "3: not visible". The AI chose to define "visible" by DLC likelihood rather than by presence of a coordinate, since the tracker emits a position even when the tongue is retracted. If a session had no visible sample at all, `q40/q60` become NaN, every comparison is False, and the whole session falls to class 3 rather than crashing.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps, spike times and trial start times all live on the same session-absolute clock, so the tongue is sampled on the *identical* grid used for the firing rates: `sample_t = start_time + (i+0.5)·0.05` for `i = 0..79`. Bin *k* of the tongue row therefore covers the same interval as bin *k* of `neural`, with no interpolation or offset correction. The alignment is relative rather than event-based, so it inherits the window offset described in 2-d but remains mutually consistent with the neural data.

ii.
```python
sample_t=(starts[:,None]+(np.arange(NBIN)+.5)*DT).ravel()
jj=np.searchsorted(tt,sample_t); jj=np.clip(jj,1,len(tt)-1)
jj-=((sample_t-tt[jj-1]) <= (tt[jj]-sample_t))
yy=td[jj,1].reshape(ntr,NBIN); conf=td[jj,2].reshape(ntr,NBIN)
```

iii. Code comment: *"Nearest video frame to each 50-ms center (video is ~300 Hz)."* The ~3.4 ms frame period is much finer than the 50 ms bin, so nearest-frame lookup introduces negligible timing error — I measured the nearest-frame distance at ≤ 2 ms for every sampled bin centre across spot-checked sessions, with no sample landing in a video gap.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases are handled explicitly, and one important case is not:

- **Session with no QC-passing units** — detected in a pre-pass and excluded before any output arrays are built (this was added only after the verifier complained about an empty `brain_region_idx`).
- **Missing `anno_name` column, or an empty annotation string** — the unit's region falls back to the literal `'unknown'`.
- **Control trials with `'N/A'` photostim strings** — guarded by a string comparison before any `float()` conversion, leaving an all-zero photostim row.
- **NaN / non-finite tongue coordinates or likelihoods** — excluded by `np.isfinite`, then represented as the explicit `'not visible'` category rather than imputed.
- **A session with no visible tongue frame at all** — percentiles fall back to `(nan, nan)`, which makes every comparison False so the session is entirely class 3 instead of raising.
- **Not handled: trials with no spike data.** `obs_intervals` and `free_water` are never consulted, so 3,512 trials (3.7 %) are emitted as 4 s of exactly 0 Hz across every unit rather than being dropped or marked.

ii.
```python
def text(a):
    return np.asarray([x.decode() if isinstance(x,(bytes,np.bytes_)) else str(x) for x in a])
```
```python
if np.any(text(f['units/classification'][:])=='good'): kept.append(p)
else: print('Excluding session with no good units:',os.path.basename(p),flush=True)
```
```python
if 'anno_name' in u: regs=text(u['anno_name'][:])[good]
else: regs=np.repeat('unknown',len(good))
regs=np.asarray([r if r else 'unknown' for r in regs])
```
```python
if ppow[ti] != 'N/A':
    a=float(pon[ti])-GO_FROM_START; b=a+float(pdur[ti])
```
```python
visible=np.isfinite(yy)&np.isfinite(conf)&(conf>=DLC_THRESHOLD)
visvals=yy[visible]
q40,q60=np.percentile(visvals,[40,60]) if len(visvals) else (np.nan,np.nan)
```

iii. The AI's stated principle is that anything that cannot support decoding is excluded at the session level, while a legitimately absent measurement becomes its own category. Step 21: *"Such sessions cannot support neural decoding and should be excluded."* Step 14: the not-visible class is *"plausible because the tongue appears only briefly during licking."* The AI never investigated whether the ephys covers every behavioural trial, so it offers no justification for the all-zero trials — they were not detected (the verifier passed with return code 0 and no warnings, step 28).

## 10-a. What are the most time-consuming steps of the code?

i. The AI never profiled the code and records no decision about it. Measured behaviour: the full 173-session conversion ran in roughly 5 minutes (trajectory steps 22–26, ~4½ min of 60 s polls plus serialization), plus ~1 min to write the 12.5 GB pickle. The dominant costs are (1) reading the ragged `units/spike_times` buffer and the `(n_frames, 3)` tongue array out of each 130–350 MB NWB, and (2) the **nested Python loop over (good unit × trial)** in the spike binning — 459 × 368 ≈ 169 k iterations for a typical session, each doing two `searchsorted` calls, a `floor`/`astype`, and a `bincount`. On a representative session that loop takes ~1.0 s versus ~0.2 s for the equivalent vectorised-over-trials formulation, i.e. ~5× the necessary time. There is also a full extra pass that opens all 174 files to read `units/classification`.

ii.
```python
for row,ui in enumerate(good):
  sp=allsp[begins[ui]:ends[ui]]
  for ti,s in enumerate(starts):
    lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
    idx=np.floor((sp[lo:hi]-s)/DT).astype(np.int64)
    sess_n[ti][row]=np.bincount(idx,minlength=NBIN)[:NBIN]/DT
```
```python
allsp=np.asarray(u['spike_times'][:],float); ends=np.asarray(u['spike_times_index'][:],int)
td=np.asarray(tg['data'][:],float); tt=np.asarray(tg['timestamps'][:],float)
```

iii. No explicit reasoning about runtime appears in the trajectory beyond step 11's concern about *memory*: *"The full dense float32 neural output will be about 12 GB, so conversion must stream efficiently and avoid loading all source data simultaneously."* The AI addressed that by processing one file at a time inside a `with` block, and accepted the per-session loop cost without comment. Total runtime was well inside budget, so the inefficiency had no practical consequence.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops are vectorizable, one of them meaningfully:

- **The inner `for ti,s in enumerate(starts)` trial loop in the spike binning** is the real one. Because all 80 bin edges for all trials can be flattened into one sorted-ish query array, the whole per-unit histogram can be done with a *single* `searchsorted` + `np.diff` instead of `n_trials` separate calls (this is exactly what the human reference does). I verified the two produce identical output, with the vectorised form ~5× faster.
- **The per-trial `for ti in range(ntr)` output/input loop** rebuilds `np.full(NBIN, ...)` tiles and re-stacks an identical `time_from_tone` vector once per trial; all four output rows could be built as one `(ntr, 4, 80)` array with broadcasting.
- **The `for r in regs` region-interning loop** is a Python loop over every good unit; `np.unique(..., return_inverse=True)` would do it in one call.

The outer per-unit loop genuinely cannot be collapsed, since `spike_times` is ragged.

ii.
```python
for row,ui in enumerate(good):
  sp=allsp[begins[ui]:ends[ui]]
  for ti,s in enumerate(starts):        # <- vectorizable over trials
    lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
```
```python
for ti in range(ntr):
  stim=np.zeros(NBIN,dtype=np.float32)
  ...
  sess_o.append(np.vstack((np.full(NBIN,choice,dtype=np.int64), ...)))
```
```python
rid=[]
for r in regs:
  if r not in region_map: region_map[r]=len(region_names); region_names.append(r)
  rid.append(region_map[r])
```

iii. No justification is offered — the AI made no statement about vectorisation anywhere in the trajectory. The code comment *"integer indexing is stable and gives [start,start+4) bins"* suggests the per-trial form was chosen for clarity/robustness of the bin-edge convention rather than for speed.

## 10-c. What processing does the code repeat multiple times?

i. Two genuine repeats:

- **Every NWB file is opened and `units/classification` decoded twice** — once in the session pre-filter and again in the main loop. This was bolted on in step 21 after the verifier failure rather than folded into the single pass.
- **`time_from_tone` is materialised once per trial.** The same 80-element vector is `.astype(np.float32)`-copied and stacked for all 94,370 trials, producing 94,370 identical copies in the pickle.

Additionally, `sample_t`/`jj` for the tongue and `rel_centers` for the inputs are each computed once (correctly, not repeated), and the percentile edges are computed once per session inside the same pass. Nothing else is recomputed.

ii.
```python
for p in files:
  with h5py.File(p,'r') as f:
    if np.any(text(f['units/classification'][:])=='good'): kept.append(p)   # pass 1
...
for fi,p in enumerate(files):
  with h5py.File(p,'r') as f:
    ...
    u=f['units']; cls=text(u['classification'][:]); good=np.flatnonzero(cls=='good')   # pass 2
```
```python
sess_i.append(np.stack((time_from_tone.astype(np.float32),stim)))   # identical row 0 copied per trial
```

iii. Step 21: *"Patch the converter to prefilter files lacking classifier-labeled good units before constructing subjects/session arrays."* The pre-pass exists because `subjects` and `subject_idx` are derived from the file list *before* the main loop runs, so the session list has to be final up front. That is a defensible reason for the second pass, and it is cheap (one small string column per file); it was simply not refactored afterwards.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Very little is computed and thrown away — almost everything the code produces ends up in the pickle. What is wasteful is work spent on data that carries no information, and storage that is larger than needed:

- **3,512 trials (3.7 %) are fully processed and written even though every unit is silent throughout** (no `obs_intervals`/`free_water` filtering). Their neural matrices are all zeros, so they consume ~0.45 GB of the output and are pure noise for the decoder.
- **Outputs are stored as `int64`** although every value is in `0..3`; `int8` would be 8× smaller (≈ 242 MB → 30 MB).
- **The `photostim_duration` and `photostim_power` columns are decoded for every trial** (`text()` over the whole column) but `ppow` is used only as an `'N/A'` flag, duplicating information already in `pon`.
- **Both tongue x and the full `(n_frames, 3)` array are read and cast to `float64`**, while only columns 1 and 2 are ever used.
- **`np.bincount(idx, minlength=NBIN)[:NBIN]` slices a result that can never be longer than `NBIN`**, since `idx` is already restricted to the window — a no-op guard on every one of the ~169 k inner iterations.

ii.
```python
sess_o.append(np.vstack((np.full(NBIN,choice,dtype=np.int64), np.full(NBIN,oc,dtype=np.int64),
                         np.full(NBIN,el,dtype=np.int64), tongue[ti])))
```
```python
pon=text(tr['photostim_onset'][:]); pdur=text(tr['photostim_duration'][:]); ppow=text(tr['photostim_power'][:])
```
```python
td=np.asarray(tg['data'][:],float)      # (n_frames, 3); only cols 1 and 2 used
```
```python
sess_n[ti][row]=np.bincount(idx,minlength=NBIN)[:NBIN]/DT
```

iii. The AI's only stated efficiency concern was output size (step 11: *"The full dense float32 neural output will be about 12 GB"*), which it addressed by keeping the neural array in `float32`; it did not apply the same reasoning to the categorical outputs. It offers no justification for retaining the silent trials because it never detected them — the format verifier reported no errors or warnings (step 28: *"passed `/app/train_decoder.py --verify-only` with return code 0"*), so no all-zero-trial check was ever triggered.
