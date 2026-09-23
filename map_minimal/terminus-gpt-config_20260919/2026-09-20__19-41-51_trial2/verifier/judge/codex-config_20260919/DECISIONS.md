# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed every NWB file under `/app/data/sub-*/*.nwb`, sorted the paths, pre-scanned them to exclude files with no `classification == 'good'` unit, and opened each retained file once with `h5py`. It read trials, units, spikes, and behavioral acquisition arrays directly from their HDF5 paths.

ii.
```python
files=sorted(glob.glob('/app/data/sub-*/*.nwb'))
kept=[]
for p in files:
  with h5py.File(p,'r') as f:
    if np.any(text(f['units/classification'][:])=='good'): kept.append(p)
...
with h5py.File(p,'r') as f:
  tr=f['intervals/trials']
  u=f['units']
```

iii. The trajectory says there were 174 NWBs from 28 subjects, one had no good units, and the 173 retained files matched the paper's analyzed-session count. Direct HDF5 access was chosen for efficient streaming of roughly 50 GB of source files and a roughly 12 GB result.

## 1-b. How are the data split into subjects?

i. The enclosing `sub-<id>` directory supplies the subject identifier. Unique identifiers are sorted, and each session receives an index into that list.

ii.
```python
subjects=sorted({os.path.basename(os.path.dirname(p)).removeprefix('sub-') for p in files})
subjmap={s:i for i,s in enumerate(subjects)}
...
'subject_idx':np.asarray([subjmap[x['subject']] for x in sinfo],dtype=np.int32)
```

iii. The agent treated the DANDI directory convention as the subject boundary. Its trajectory reports 28 subjects and does not describe an alternative grouping.

## 1-c. How are the data split into sessions?

i. Each retained NWB file is one output session, in sorted pathname order. The file with zero good units is excluded; `--limit` optionally truncates the retained session list for testing.

ii.
```python
for fi,p in enumerate(files):
  ...
  neural.append(sess_n); inputs.append(sess_i); outputs.append(sess_o)
```

iii. The trajectory inferred that the sole zero-good-unit recording explained the difference between 174 assets and 173 paper sessions. The full result was regenerated after the validator rejected the empty-neuron session.

## 1-d. How are the data split into trials?

i. Rows of `intervals/trials` define trials. The agent uses each row's `start_time` as the beginning of the four-second neural/behavioral window and does not read or validate recorded go-cue events.

ii.
```python
tr=f['intervals/trials']; starts=np.asarray(tr['start_time'][:],float); ntr=len(starts)
sess_n=[np.empty((len(good),NBIN),dtype=np.float32) for _ in range(ntr)]
for ti,s in enumerate(starts):
```

iii. The agent reasoned that trial start was exactly 2.5 seconds before the go cue, making the requested window identical to `[start_time, start_time + 4)`. This was inferred from task timing rather than taken from the NWB event timestamps.

## 1-e. How are trials filtered based on quality controls?

i. No trials are filtered. All trial-table rows in retained sessions are emitted, including early-lick, ignore, free-water, and trials without observed spikes.

ii.
```python
ntr=len(starts)
...
'trial_filter':'all trials retained because ignore and early lick are requested outputs; sessions with zero QC-passing units excluded'
```

iii. The agent explicitly kept ignore and early-lick trials because those are requested decoder targets. It did not identify the separate need to remove trials outside `units/obs_intervals` or `free_water` trials with no spike data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the ragged `units/spike_times` buffer and `units/spike_times_index`, restricted by `units/classification`. Trial `start_time` is used to position bins.

ii.
```python
allsp=np.asarray(u['spike_times'][:],float); ends=np.asarray(u['spike_times_index'][:],int)
begins=np.r_[0,ends[:-1]]
cls=text(u['classification'][:]); good=np.flatnonzero(cls=='good')
```

iii. The trajectory identified spike times as the neural source and the paper's classifier verdict as the appropriate QC field.

## 2-b. How is the `neural` data processed?

i. For every good unit and trial, spikes in the four-second window are assigned to 50 ms half-open bins with `floor`/`bincount`; counts are divided by 0.05 seconds to produce firing rates in Hz. No smoothing, normalization, or baseline correction is applied.

ii.
```python
lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
idx=np.floor((sp[lo:hi]-s)/DT).astype(np.int64)
sess_n[ti][row]=np.bincount(idx,minlength=NBIN)[:NBIN]/DT
```

iii. The agent aimed to reproduce 80 left-closed 50 ms firing-rate bins over the specified four-second interval.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `units/classification` text equals `good` are retained. A whole session is excluded if it has no such units.

ii.
```python
cls=text(u['classification'][:]); good=np.flatnonzero(cls=='good')
if np.any(text(f['units/classification'][:])=='good'): kept.append(p)
```

iii. The agent linked `classification == 'good'` to the paper's region-specific spike-sorting QC classifiers. The zero-good-unit session was removed because it cannot furnish decoder input and because doing so yields the paper's 173 sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is indirect: the agent assumes `go = trial start + 2.5 s`, then bins from trial start to four seconds later. It never reads `BehavioralEvents/go_start_times`.

ii.
```python
GO_FROM_START=2.5
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
for ti,s in enumerate(starts):
  lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
```

iii. The trajectory says the fixed offset was inferred from task timing and the requested window. It regarded the NWB as omitting an explicit per-trial cue field, although the reference accesses that event through the `BehavioralEvents` container.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50 ms bins over four seconds. Raw spikes are binned directly; no subsequent rebinning is performed.

ii.
```python
DT=.05; NBIN=80
...
'time_bin_size':50.0
```

iii. This directly follows the requested 50-ms resolution and -2.5 to +1.5 second window.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is not derived from a recorded tone event. It is derived entirely from constants: go is assumed 2.5 seconds after trial start and tone is assumed 0.5 seconds after trial start.

ii.
```python
GO_FROM_START=2.5; TONE_FROM_START=.5
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
time_from_tone=rel_centers-(-GO_FROM_START+TONE_FROM_START)
```

iii. The agent inferred these constants from the task protocol and believed the NWB did not expose explicit tone/go events.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The fixed tone offset (-2.0 seconds relative to assumed go) is subtracted from every bin center. Consequently every trial and session gets the same vector, from 0.025 to 3.975 seconds.

ii.
```python
time_from_tone=rel_centers-(-GO_FROM_START+TONE_FROM_START)
...
sess_i.append(np.stack((time_from_tone.astype(np.float32),stim)))
```

iii. The agent judged that no trial-specific calculation was needed under its fixed-timing assumption.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same 80 conceptual bin centers relative to trial start/assumed go. The time vector is stacked with the input for each trial.

ii.
```python
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
...
sess_i.append(np.stack((time_from_tone.astype(np.float32),stim)))
```

iii. The alignment is internally consistent with the agent's trial-start neural bins, but inherits the fixed event-timing assumption.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_power` to decide whether stimulation exists, and `photostim_onset` plus `photostim_duration` to define its interval.

ii.
```python
pon=text(tr['photostim_onset'][:]); pdur=text(tr['photostim_duration'][:]); ppow=text(tr['photostim_power'][:])
if ppow[ti] != 'N/A':
```

iii. The trajectory found `N/A` for control trials and numeric stimulation values, with onset/duration relative to trial start.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For stimulated trials, onset is shifted by 2.5 seconds onto the assumed go-relative axis, duration supplies the offset, and bins whose centers fall in `[onset, offset)` are set to one; other bins are zero.

ii.
```python
a=float(pon[ti])-GO_FROM_START; b=a+float(pdur[ti])
stim[(rel_centers>=a)&(rel_centers<b)]=1
```

iii. The agent intentionally represented stimulation as a time-varying binary state sampled at bin centers.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation and neural data share the same bin centers relative to trial start/assumed go.

ii.
```python
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
stim[(rel_centers>=a)&(rel_centers<b)]=1
```

iii. The agent considered onset relative to trial start plus the fixed 2.5-second shift sufficient for alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is reconstructed from trial `outcome` and `trial_instruction`: hit means instructed side, miss means opposite side, and ignore means no lick.

ii.
```python
outcome=text(tr['outcome'][:]); instr=text(tr['trial_instruction'][:])
if outcome[ti]=='ignore': choice=2
elif outcome[ti]=='hit': choice=0 if instr[ti]=='left' else 1
else: choice=1 if instr[ti]=='left' else 0
```

iii. The trajectory states that choice is not stored directly but is exactly recoverable from instruction and outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as 0 left, 1 right, or 2 no lick, then repeated over all 80 time bins.

ii.
```python
np.full(NBIN,choice,dtype=np.int64)
...
['left','right','no lick']
```

iii. The extra no-lick category preserves ignore trials, and repetition permits a common time-varying output array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from `intervals/trials/outcome`.

ii.
```python
outcome=text(tr['outcome'][:])
```

iii. The field already contains the requested hit, miss, and ignore labels.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to ignore=0, miss=1, hit=2 and repeated across 80 bins.

ii.
```python
oc={'ignore':0,'miss':1,'hit':2}[outcome[ti]]
np.full(NBIN,oc,dtype=np.int64)
```

iii. This supplies the requested categorical coding in the common output shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `intervals/trials/early_lick`.

ii.
```python
early=text(tr['early_lick'][:])
```

iii. The trial table explicitly labels `early` and `no early` trials.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `early` maps to 1 and every other expected value (`no early`) maps to 0; the value is repeated over 80 bins.

ii.
```python
el=1 if early[ti]=='early' else 0
np.full(NBIN,el,dtype=np.int64)
```

iii. This is the requested binary no/yes encoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: column 1 is y, column 2 is DLC likelihood, and the series timestamps locate frames.

ii.
```python
tg=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
td=np.asarray(tg['data'][:],float); tt=np.asarray(tg['timestamps'][:],float)
```

iii. The agent inspected the acquisition container and identified its x, y, likelihood layout and roughly 300 Hz timestamps.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The nearest video frame to each 50 ms neural-bin center is selected. A sample is visible only if y/likelihood are finite and likelihood is at least 0.9. Session thresholds are percentiles of all visible center-sampled values in retained trial windows; invisible samples get class 3.

ii.
```python
sample_t=(starts[:,None]+(np.arange(NBIN)+.5)*DT).ravel()
jj=np.searchsorted(tt,sample_t); jj=np.clip(jj,1,len(tt)-1)
jj-=((sample_t-tt[jj-1]) <= (tt[jj]-sample_t))
yy=td[jj,1].reshape(ntr,NBIN); conf=td[jj,2].reshape(ntr,NBIN)
visible=np.isfinite(yy)&np.isfinite(conf)&(conf>=DLC_THRESHOLD)
q40,q60=np.percentile(yy[visible],[40,60]) if np.any(visible) else (np.nan,np.nan)
```

iii. The trajectory found no likelihood rule in the method code and selected 0.9 as a documented conventional threshold. It considered the high invisible fraction plausible because protrusions are brief.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Visible values below q40 map to 0, q40 through q60 inclusive map to 1, and values above q60 map to 2. Invisible values remain 3.

ii.
```python
tongue=np.full((ntr,NBIN),3,dtype=np.int64)
tongue[visible&(yy<q40)]=0
tongue[visible&(yy>=q40)&(yy<=q60)]=1
tongue[visible&(yy>q60)]=2
```

iii. This implements the requested per-session 40th/60th percentile categories, with explicit boundary choices.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames nearest to `trial start + bin center` are sampled, matching the centers of the agent's neural bins under its fixed go-offset assumption.

ii.
```python
sample_t=(starts[:,None]+(np.arange(NBIN)+.5)*DT).ravel()
```

iii. The agent used timestamp-based nearest-neighbor sampling because video is much faster than the 50 ms output grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. A session with no good units is excluded; absent/low-confidence tongue samples become `not visible`; a session with no visible tongue values receives NaN thresholds and thus all tongue bins stay class 3. Missing `anno_name` would become `unknown`. The code does not remove trials with missing spike recordings.

ii.
```python
if np.any(text(f['units/classification'][:])=='good'): kept.append(p)
...
regs=np.asarray([r if r else 'unknown' for r in regs])
...
q40,q60=np.percentile(visvals,[40,60]) if len(visvals) else (np.nan,np.nan)
```

iii. The trajectory discovered and corrected the zero-neuron session after validation. It treated low-confidence tracking as legitimate invisibility, but did not investigate `obs_intervals`/free-water missing-spike trials.

## 10-a. What are the most time-consuming steps of the code?

i. The nested good-unit-by-trial spike histogram loop is the principal compute-heavy section; reading large spike/video arrays and serializing the roughly 12 GB pickle are also expensive. The full conversion was monitored across 173 sessions.

ii.
```python
for row,ui in enumerate(good):
  sp=allsp[begins[ui]:ends[ui]]
  for ti,s in enumerate(starts):
    lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
```

iii. The trajectory emphasized streaming and avoiding simultaneous source loads due to 50 GB of NWBs and a dense 12 GB output, though it did not provide a timing breakdown.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner per-trial spike loop could be replaced by one `searchsorted` over all trial edges per unit, as in the reference. The loops constructing inputs/outputs per trial and assigning region IDs could also be vectorized, though they are much smaller.

ii.
```python
for row,ui in enumerate(good):
  ...
  for ti,s in enumerate(starts):
...
for ti in range(ntr):
  ...
for r in regs:
```

iii. The agent did not discuss these vectorization opportunities; it only described its implementation as reading ragged spikes once and histogramming units across trials.

## 10-c. What processing does the code repeat multiple times?

i. It repeats two spike boundary searches and a histogram for every unit/trial pair, rebuilds the same time-from-tone array in every trial input, and allocates repeated constant arrays for three per-trial outputs.

ii.
```python
sess_i.append(np.stack((time_from_tone.astype(np.float32),stim)))
sess_o.append(np.vstack((np.full(NBIN,choice,dtype=np.int64),
                         np.full(NBIN,oc,dtype=np.int64),
                         np.full(NBIN,el,dtype=np.int64),tongue[ti])))
```

iii. The trajectory offers no explicit justification beyond producing uniform `(variable, time)` arrays.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little computed data is discarded. `photostim_power` is read only as a presence flag, and full anatomical annotation strings are retained as region categories rather than reduced region labels. Trial-level labels are redundantly expanded across 80 bins, increasing storage although the decoder targets are constant per trial.

ii.
```python
ppow=text(tr['photostim_power'][:])
if ppow[ti] != 'N/A':
...
np.full(NBIN,choice,dtype=np.int64)
```

iii. The agent justified expansion as a common time-varying output format, but did not discuss discarded or unnecessary work.
