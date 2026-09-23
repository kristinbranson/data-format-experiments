# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by globbing every NWB file under `/app/data/sub-*/*.nwb`, opening each file with `h5py`, and then reading HDF5 groups directly. Within each file it reads trials from `intervals/trials`, units from `units`, and tongue tracking from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`.

ii. 
```python
files=sorted(glob.glob('/app/data/sub-*/*.nwb'))
```

```python
with h5py.File(p,'r') as f:
    tr=f['intervals/trials']; starts=np.asarray(tr['start_time'][:],float); ntr=len(starts)
    outcome=text(tr['outcome'][:]); instr=text(tr['trial_instruction'][:]); early=text(tr['early_lick'][:])
    pon=text(tr['photostim_onset'][:]); pdur=text(tr['photostim_duration'][:]); ppow=text(tr['photostim_power'][:])
    u=f['units']; cls=text(u['classification'][:]); good=np.flatnonzero(cls=='good')
```

iii. In trajectory steps 2-11, the agent inspected the NWB schema repeatedly, concluded that the dataset was organized as one large NWB file per session under `sub-*` directories, and chose direct HDF5 access after probing field names and paths. The trajectory does not show a separate methodological reason for preferring `h5py` over `pynwb`; it mainly reflects a schema-discovery workflow.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the parent directory name, e.g. `sub-440956`, by stripping the `sub-` prefix. The script then builds a sorted unique subject list and a subject-to-index map.

ii. 
```python
subjects=sorted({os.path.basename(os.path.dirname(p)).removeprefix('sub-') for p in files})
subjmap={s:i for i,s in enumerate(subjects)}
```

```python
sub=os.path.basename(os.path.dirname(p)).removeprefix('sub-')
...
'subject_idx':np.asarray([subjmap[x['subject']] for x in sinfo],dtype=np.int32),
```

iii. In trajectory steps 5-7, the agent counted sessions per `sub-*` folder and treated those folders as the subject split. The trajectory does not mention reading `nwb.subject`; the folder structure itself was used as sufficient evidence.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The script iterates one file at a time and appends one session-level entry to `neural`, `input`, `output`, and `brain_region_idx`. One session with zero good units is excluded before this per-session loop.

ii. 
```python
files=sorted(glob.glob('/app/data/sub-*/*.nwb'))
...
for p in files:
  with h5py.File(p,'r') as f:
    if np.any(text(f['units/classification'][:])=='good'): kept.append(p)
    else: print('Excluding session with no good units:',os.path.basename(p),flush=True)
files=kept
```

```python
for fi,p in enumerate(files):
  print(f'[{fi+1}/{len(files)}] {os.path.basename(p)}',flush=True)
  with h5py.File(p,'r') as f:
      ...
      neural.append(sess_n); inputs.append(sess_i); outputs.append(sess_o)
```

iii. In trajectory steps 2, 5, and 21-27, the agent explicitly treated one NWB file as one session and later justified dropping exactly one file because it had zero `classification == 'good'` units, which it tied to the paper's 173 analyzed sessions.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the `intervals/trials` table. Each row is treated as one trial, and `start_time` is used as the trial anchor for neural binning, tongue alignment, and photostim alignment.

ii. 
```python
tr=f['intervals/trials']; starts=np.asarray(tr['start_time'][:],float); ntr=len(starts)
```

```python
sess_n=[np.empty((len(good),NBIN),dtype=np.float32) for _ in range(ntr)]
...
for ti,s in enumerate(starts):
```

iii. In trajectory steps 7-10, the agent reasoned that explicit per-trial go/tone events were not obvious from its HDF5 probes and therefore treated the trial table plus fixed task timing as the relevant structure. The justification was that the requested 4 s window likely began at trial start.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any per-trial quality-control filtering. It keeps all trials in each retained session, including ignore and early-lick trials, and only excludes sessions with zero good units.

ii. 
```python
"""Convert the MAP auditory delayed-response NWB files for neural decoding.

Decisions:
* Keep every supplied session/trial: ignore and early-lick trials are required targets.
...
"""
```

```python
tr=f['intervals/trials']; starts=np.asarray(tr['start_time'][:],float); ntr=len(starts)
...
sess_n=[np.empty((len(good),NBIN),dtype=np.float32) for _ in range(ntr)]
```

```python
'trial_filter':'all trials retained because ignore and early lick are requested outputs; sessions with zero QC-passing units excluded'
```

iii. In trajectory steps 5, 11, and 21, the agent explicitly decided to keep ignore and early-lick trials because they were required decoder outputs. There is no evidence in the trajectory that it discovered or used `obs_intervals` or `free_water` as trial-level filters.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `units/spike_times`, `units/spike_times_index`, and the subset of units with `units/classification == 'good'`. Trial alignment is based on `intervals/trials/start_time`, not on behavioral go-cue timestamps.

ii. 
```python
u=f['units']; cls=text(u['classification'][:]); good=np.flatnonzero(cls=='good')
...
allsp=np.asarray(u['spike_times'][:],float); ends=np.asarray(u['spike_times_index'][:],int)
begins=np.r_[0,ends[:-1]]
```

```python
tr=f['intervals/trials']; starts=np.asarray(tr['start_time'][:],float); ntr=len(starts)
```

iii. In trajectory steps 4-11, the agent identified `classification` as the QC field and `spike_times` as the raw neural signal. It also reasoned that go-cue timing would be reconstructed from fixed task timing relative to `start_time` rather than from behavioral event streams.

## 2-b. How is the `neural` data processed?

i. For each retained unit and each trial, the AI bins spikes into 80 consecutive 50 ms bins covering `[trial_start, trial_start + 4 s)`, converts counts to firing rates by dividing by `DT`, and stores one `(n_neurons, 80)` matrix per trial. There is no smoothing or baseline subtraction.

ii. 
```python
DT=.05; NBIN=80; GO_FROM_START=2.5
```

```python
sess_n=[np.empty((len(good),NBIN),dtype=np.float32) for _ in range(ntr)]
for row,ui in enumerate(good):
  sp=allsp[begins[ui]:ends[ui]]
  for ti,s in enumerate(starts):
    lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
    idx=np.floor((sp[lo:hi]-s)/DT).astype(np.int64)
    sess_n[ti][row]=np.bincount(idx,minlength=NBIN)[:NBIN]/DT
```

iii. In trajectory steps 11-13, the agent justified this by assuming the requested window started at trial start and by describing the output as firing rate in spikes/s with 50 ms bins. The trajectory does not mention any further normalization or smoothing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered solely by `units/classification == 'good'`. If a session has no such units, the whole session is excluded.

ii. 
```python
for p in files:
  with h5py.File(p,'r') as f:
    if np.any(text(f['units/classification'][:])=='good'): kept.append(p)
    else: print('Excluding session with no good units:',os.path.basename(p),flush=True)
```

```python
u=f['units']; cls=text(u['classification'][:]); good=np.flatnonzero(cls=='good')
```

iii. In trajectory steps 4-5 and 20-21, the agent connected `classification == 'good'` to the paper's QC classifier and used verifier failure on an empty-neuron session to justify excluding the single zero-good-unit session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not read go-cue timestamps. Instead, it assumes the go cue occurs exactly 2.5 s after `trial.start_time`, so the `[trial_start, trial_start + 4 s)` window is treated as equivalent to `[-2.5 s, +1.5 s]` around the go cue.

ii. 
```python
DT=.05; NBIN=80; GO_FROM_START=2.5; TONE_FROM_START=.5
```

```python
# Trial start is 2.5 s before the go cue; sample-tone onset is 0.5 s after trial
# start. Histograms use left-closed 50-ms bins over go-2.5 through go+1.5.
```

```python
for ti,s in enumerate(starts):
  lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
```

iii. In trajectory steps 7-13, the agent explicitly reasoned that the NWB files appeared to omit convenient per-trial cue fields and that the likely task timing was fixed, with trial start 2.5 s before the go cue. That assumption became the alignment rule used throughout the script.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 50 ms bins and 80 total bins per trial. There is no extra temporal rebinning beyond the initial spike histogramming and per-bin output/input construction.

ii. 
```python
DT=.05; NBIN=80
```

```python
idx=np.floor((sp[lo:hi]-s)/DT).astype(np.int64)
sess_n[ti][row]=np.bincount(idx,minlength=NBIN)[:NBIN]/DT
```

iii. In trajectory steps 11-14, the agent repeatedly described the target as 80 bins of width 50 ms and validated that shape with the decoder verifier.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is not derived from a raw event variable. The AI computes it from hard-coded constants: go cue assumed 2.5 s after trial start and tone assumed 0.5 s after trial start, making tone onset effectively fixed at `-2.0 s` relative to the assumed go cue.

ii. 
```python
DT=.05; NBIN=80; GO_FROM_START=2.5; TONE_FROM_START=.5
```

```python
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
time_from_tone=rel_centers-(-GO_FROM_START+TONE_FROM_START)
```

iii. In trajectory steps 7-13, the agent concluded that trial timing should be reconstructed from fixed task structure rather than from event timestamps. It therefore used a constant tone offset instead of reading `sample_start_times`.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes a single 80-element vector of bin centers relative to the assumed go cue and shifts it by 2.0 s so that it represents seconds since the assumed tone onset. The exact same vector is reused for every trial in every session.

ii. 
```python
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
time_from_tone=rel_centers-(-GO_FROM_START+TONE_FROM_START)
```

```python
sess_i.append(np.stack((time_from_tone.astype(np.float32),stim)))
```

iii. The trajectory justification is the same fixed-timing assumption from steps 7-13. There is no trial-specific computation beyond stacking this precomputed vector with the photostim signal.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by construction to the same 80-bin grid used for neural binning. Both the neural data and this input use bin centers derived from the assumed relation between trial start, tone, and go cue.

ii. 
```python
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
time_from_tone=rel_centers-(-GO_FROM_START+TONE_FROM_START)
```

```python
for ti,s in enumerate(starts):
  lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
  ...
sess_i.append(np.stack((time_from_tone.astype(np.float32),stim)))
```

iii. In trajectory steps 11-14, the agent described the sample as correctly aligned because both neural and inputs shared the same fixed 80-bin layout. The justification depends entirely on the fixed timing assumption.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from trial-table columns `photostim_onset`, `photostim_duration`, and `photostim_power`. `photostim_power` is used as the presence/absence flag, while onset and duration set the active interval.

ii. 
```python
pon=text(tr['photostim_onset'][:]); pdur=text(tr['photostim_duration'][:]); ppow=text(tr['photostim_power'][:])
```

```python
if ppow[ti] != 'N/A':
  a=float(pon[ti])-GO_FROM_START; b=a+float(pdur[ti])
```

iii. In trajectory steps 8-11, the agent inspected trial-table photostim fields and concluded that trials encoded control vs stimulation via `N/A`-style text. It then used the power field as the branch condition for whether stimulation was present.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the AI creates an 80-bin binary vector initialized to zeros. If `photostim_power` is not `N/A`, it converts `photostim_onset` and `photostim_duration` to floats, shifts onset by the fixed `GO_FROM_START`, and sets bins whose centers fall inside `[onset, offset)` to 1.

ii. 
```python
stim=np.zeros(NBIN,dtype=np.float32)
if ppow[ti] != 'N/A':
  a=float(pon[ti])-GO_FROM_START; b=a+float(pdur[ti])
  stim[(rel_centers>=a)&(rel_centers<b)]=1
```

iii. In trajectory steps 10-11, the agent reasoned that photostimulation fields were stored relative to trial start and therefore only needed conversion to a bin-centered on/off series on the assumed go-relative axis.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is aligned to the same fixed bin centers used for the neural data. The onset is shifted by the constant `GO_FROM_START`, not by an observed go-cue timestamp.

ii. 
```python
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
```

```python
if ppow[ti] != 'N/A':
  a=float(pon[ti])-GO_FROM_START; b=a+float(pdur[ti])
  stim[(rel_centers>=a)&(rel_centers<b)]=1
```

iii. The trajectory justification in steps 10-13 is that the neural data, time-from-tone input, and photostim input should all share one fixed 80-bin axis. The code implements that shared axis via `rel_centers`.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `outcome` and `trial_instruction`. It is not read from a dedicated lick-direction variable.

ii. 
```python
outcome=text(tr['outcome'][:]); instr=text(tr['trial_instruction'][:])
```

```python
if outcome[ti]=='ignore': choice=2
elif outcome[ti]=='hit': choice=0 if instr[ti]=='left' else 1
else: choice=1 if instr[ti]=='left' else 0
```

iii. In trajectory steps 10-11, the agent explicitly reasoned that actual lick choice was not directly stored but could be reconstructed exactly: instructed side on hits, opposite side on misses, and no lick on ignores.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The derived choice is encoded as `0=left`, `1=right`, `2=no lick`, then repeated across all 80 bins for that trial in the output tensor.

ii. 
```python
if outcome[ti]=='ignore': choice=2
elif outcome[ti]=='hit': choice=0 if instr[ti]=='left' else 1
else: choice=1 if instr[ti]=='left' else 0
...
sess_o.append(np.vstack((np.full(NBIN,choice,dtype=np.int64), ...
```

iii. The trajectory justification in steps 10-14 is that choice is a trial-level label, so repeating it across bins was acceptable for the decoder format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table `outcome` column.

ii. 
```python
outcome=text(tr['outcome'][:])
```

iii. In trajectory step 5, the agent inspected trial-column value counts and observed that `outcome` already contained the required `hit`, `miss`, and `ignore` labels.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped to integers `0=ignore`, `1=miss`, `2=hit`, and that trial-level code is repeated across the 80 bins.

ii. 
```python
oc={'ignore':0,'miss':1,'hit':2}[outcome[ti]]
...
sess_o.append(np.vstack((...,np.full(NBIN,oc,dtype=np.int64),...)))
```

iii. The trajectory justification is minimal here; the mapping follows the required categorical output format and uses the existing trial labels directly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table `early_lick` column.

ii. 
```python
early=text(tr['early_lick'][:])
```

iii. In trajectory step 5, the agent inspected the value counts for `early_lick` and found the two categories `no early` and `early`, so no further derivation was needed.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string value is converted to a binary code, with `1` for `'early'` and `0` otherwise, and repeated across all 80 bins.

ii. 
```python
el=1 if early[ti]=='early' else 0
...
sess_o.append(np.vstack((...,np.full(NBIN,el,dtype=np.int64),tongue[ti])))
```

iii. The trajectory justification is that early-lick trials had to be retained because early lick itself was a requested decoder output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` and `.../timestamps`. Column 1 of `data` is used as tongue y and column 2 as the DeepLabCut confidence.

ii. 
```python
tg=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
td=np.asarray(tg['data'][:],float); tt=np.asarray(tg['timestamps'][:],float)
```

```python
yy=td[jj,1].reshape(ntr,NBIN); conf=td[jj,2].reshape(ntr,NBIN)
```

iii. In trajectory steps 7-11, the agent first confirmed that tongue tracking existed in the NWB files under `BehavioralTimeSeries`, then described the columns as x, y, and DLC likelihood. That became the basis for the tongue output.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI samples the nearest video frame to each 50 ms bin center, rather than averaging all frames in the bin. Frames with nonfinite values or confidence below `0.9` are treated as invisible. It pools all visible sampled y-values across retained trial windows in the session, computes the 40th and 60th percentiles, and then classifies each sampled bin-center value against those thresholds.

ii. 
```python
DLC_THRESHOLD=.9
```

```python
sample_t=(starts[:,None]+(np.arange(NBIN)+.5)*DT).ravel()
jj=np.searchsorted(tt,sample_t); jj=np.clip(jj,1,len(tt)-1)
jj-=((sample_t-tt[jj-1]) <= (tt[jj]-sample_t))
yy=td[jj,1].reshape(ntr,NBIN); conf=td[jj,2].reshape(ntr,NBIN)
visible=np.isfinite(yy)&np.isfinite(conf)&(conf>=DLC_THRESHOLD)
visvals=yy[visible]
q40,q60=np.percentile(visvals,[40,60]) if len(visvals) else (np.nan,np.nan)
```

iii. In trajectory steps 12-13, the agent explicitly described this as a conventional visibility thresholding choice and noted that it had not found a repository rule forcing a different cutoff. The docstring at the top of the script also states the 0.9 threshold and that percentiles are computed from visible samples in retained windows.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI assigns category `0` below the session 40th percentile, category `1` between the 40th and 60th percentiles inclusive, category `2` above the 60th percentile, and category `3` when the tongue is not visible.

ii. 
```python
tongue=np.full((ntr,NBIN),3,dtype=np.int64)
tongue[visible&(yy<q40)]=0
tongue[visible&(yy>=q40)&(yy<=q60)]=1
tongue[visible&(yy>q60)]=2
```

iii. The trajectory justification is brief: the agent was trying to follow the instruction that tongue y-position be discretized by per-session 40th and 60th percentiles, while using an explicit fourth class for invisibility.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y-position is aligned to the neural data by sampling the nearest camera frame to each neural bin center. Those bin centers are defined relative to `trial.start_time`, which the AI assumes is 2.5 s before the go cue.

ii. 
```python
sample_t=(starts[:,None]+(np.arange(NBIN)+.5)*DT).ravel()
jj=np.searchsorted(tt,sample_t); jj=np.clip(jj,1,len(tt)-1)
jj-=((sample_t-tt[jj-1]) <= (tt[jj]-sample_t))
```

iii. In trajectory steps 11-14, the agent justified this by noting that video runs much faster than 50 ms bins, so taking the nearest frame to each bin center was a simple way to put tongue measurements on the same grid as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles a few missing-data cases, but narrowly. Sessions with no good units are dropped. Missing or empty `anno_name` values become `'unknown'`. Tongue bins with low-confidence or nonfinite video values are assigned the `'not visible'` class. There is no explicit handling for behavioral trials lacking spike coverage.

ii. 
```python
if np.any(text(f['units/classification'][:])=='good'): kept.append(p)
else: print('Excluding session with no good units:',os.path.basename(p),flush=True)
```

```python
if 'anno_name' in u:
  regs=text(u['anno_name'][:])[good]
else:
  regs=np.repeat('unknown',len(good))
regs=np.asarray([r if r else 'unknown' for r in regs])
```

```python
visible=np.isfinite(yy)&np.isfinite(conf)&(conf>=DLC_THRESHOLD)
...
tongue=np.full((ntr,NBIN),3,dtype=np.int64)
```

iii. In trajectory steps 12-13 and 20-21, the agent focused on two data-quality issues: low-confidence tongue tracking and the one session that had zero good units. The trajectory does not show discovery of the trial-level missing-spike cases handled in the reference solution.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are the per-session reads of large NWB arrays and the nested per-unit/per-trial spike histogram loop. The tongue-processing step also reads the full tracking arrays and performs nearest-frame indexing across all bins.

ii. 
```python
with h5py.File(p,'r') as f:
```

```python
for row,ui in enumerate(good):
  sp=allsp[begins[ui]:ends[ui]]
  for ti,s in enumerate(starts):
    lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
    idx=np.floor((sp[lo:hi]-s)/DT).astype(np.int64)
    sess_n[ti][row]=np.bincount(idx,minlength=NBIN)[:NBIN]/DT
```

```python
td=np.asarray(tg['data'][:],float); tt=np.asarray(tg['timestamps'][:],float)
sample_t=(starts[:,None]+(np.arange(NBIN)+.5)*DT).ravel()
```

iii. In trajectory steps 12-14 and 20, the agent repeatedly discussed the expected size of the full output file and the fact that full conversion over all sessions was substantial. It did not provide a formal runtime analysis, but the code structure makes the dominant costs clear.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The largest missed vectorization opportunity is the nested loop over good units and trials for spike binning. The loop that appends per-trial inputs/outputs could also be vectorized further, as could the loop building region indices.

ii. 
```python
for row,ui in enumerate(good):
  sp=allsp[begins[ui]:ends[ui]]
  for ti,s in enumerate(starts):
    ...
```

```python
rid=[]
for r in regs:
  if r not in region_map: region_map[r]=len(region_names); region_names.append(r)
  rid.append(region_map[r])
```

```python
sess_i=[]; sess_o=[]
for ti in range(ntr):
  ...
  sess_i.append(np.stack((time_from_tone.astype(np.float32),stim)))
  ...
  sess_o.append(np.vstack((...)))
```

iii. The trajectory does not show the agent analyzing vectorization tradeoffs in detail. The only implicit efficiency justification is in comments such as reading ragged spike vectors once and using integer bin indexing.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several computations: it re-histograms spikes separately for every trial of every unit, repeatedly casts the same `time_from_tone` vector to `float32` inside the per-trial loop, and repeatedly allocates full-length constant output rows for trial-level labels.

ii. 
```python
for row,ui in enumerate(good):
  sp=allsp[begins[ui]:ends[ui]]
  for ti,s in enumerate(starts):
    ...
```

```python
sess_i.append(np.stack((time_from_tone.astype(np.float32),stim)))
```

```python
sess_o.append(np.vstack((np.full(NBIN,choice,dtype=np.int64),
                         np.full(NBIN,oc,dtype=np.int64),
                         np.full(NBIN,el,dtype=np.int64),
                         tongue[ti])))
```

iii. There is no explicit trajectory justification for these repeated computations. They appear to be straightforward implementation choices rather than defended design decisions.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does little obviously discarded work, but it does read and branch on `photostim_power` even though the downstream input only stores on/off state, and it records extra session metadata such as tongue percentiles and the DLC threshold that are not used by the decoder itself.

ii. 
```python
pon=text(tr['photostim_onset'][:]); pdur=text(tr['photostim_duration'][:]); ppow=text(tr['photostim_power'][:])
...
if ppow[ti] != 'N/A':
```

```python
sinfo.append({'file':os.path.basename(p),'subject':sub,'n_trials':ntr,'n_good_units':len(good),
              'tongue_y_percentiles':[float(q40),float(q60)],
              'dlc_likelihood_threshold':DLC_THRESHOLD})
```

iii. The trajectory does not present these as intentional downstream features; they are incidental byproducts of how the converter was implemented and documented.
