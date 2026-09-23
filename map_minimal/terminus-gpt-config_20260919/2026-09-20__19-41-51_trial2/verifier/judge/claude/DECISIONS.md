# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `h5py` to directly read HDF5 paths within each NWB file. It finds all sessions via `glob.glob('/app/data/sub-*/*.nwb')`, iterates through them, and reads trials from `intervals/trials`, units from `units/`, and behavioral data from `acquisition/`. A pre-filtering pass first excludes sessions with no `classification == 'good'` units before the main processing loop.

ii.
```python
files=sorted(glob.glob('/app/data/sub-*/*.nwb'))
kept=[]
for p in files:
  with h5py.File(p,'r') as f:
    if np.any(text(f['units/classification'][:])=='good'): kept.append(p)
    else: print('Excluding session with no good units:',os.path.basename(p),flush=True)
files=kept
```

```python
for fi,p in enumerate(files):
  with h5py.File(p,'r') as f:
    tr=f['intervals/trials']; starts=np.asarray(tr['start_time'][:],float); ntr=len(starts)
```

iii. The agent initially explored the data with pynwb but switched to h5py for the production code, reasoning that direct HDF5 access would be more efficient for streaming through ~50 GB of source data and would give precise control over which arrays are loaded.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from directory names (e.g., `sub-440956` -> `440956`) rather than from within the NWB file. A sorted unique list of subjects is built, and each session is mapped to its subject index.

ii.
```python
subjects=sorted({os.path.basename(os.path.dirname(p)).removeprefix('sub-') for p in files})
subjmap={s:i for i,s in enumerate(subjects)}
...
sub=os.path.basename(os.path.dirname(p)).removeprefix('sub-')
```

iii. The agent used the directory structure as the source of subject IDs since the NWB files are organized into `sub-<id>/` directories. This yields the numeric subject IDs (e.g., `440956`) rather than mouse names (e.g., `SC015`).

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The sorted glob of NWB files defines the session list. Sessions with zero QC-passing units are excluded in a pre-filtering pass, yielding 173 sessions from the original 174.

ii.
```python
files=sorted(glob.glob('/app/data/sub-*/*.nwb'))
kept=[]
for p in files:
  with h5py.File(p,'r') as f:
    if np.any(text(f['units/classification'][:])=='good'): kept.append(p)
```

iii. The agent noted that the paper's analyzed set contains 173 sessions rather than all 174 NWB assets, and that excluding the one session with no QC-passing units matches this count.

## 1-d. How are the data split into trials?

i. Trials are taken directly from `intervals/trials` in each NWB file. The number of trials per session equals the number of rows in that table (`ntr=len(starts)`). No per-trial validation against go-cue events is performed.

ii.
```python
tr=f['intervals/trials']; starts=np.asarray(tr['start_time'][:],float); ntr=len(starts)
outcome=text(tr['outcome'][:]); instr=text(tr['trial_instruction'][:]); early=text(tr['early_lick'][:])
```

iii. The agent treated each row of the trials table as one trial, using `start_time` as the trial anchor.

## 1-e. How are trials filtered based on quality controls?

i. **No trial-level filtering is applied.** All trials from retained sessions are kept, including early-lick trials, ignore trials, and free-water trials. Only session-level filtering (excluding sessions with no good units) is performed.

ii.
```python
# No trial filtering code exists. All ntr trials are processed:
for ti,s in enumerate(starts):
    ...
```

iii. The agent explicitly reasoned that "this conversion must retain those trial types because they are explicitly requested decoder outputs" (referring to ignore and early-lick trials). However, it did not consider filtering based on `obs_intervals` (trials with no spike data) or `free_water` trials (which have no neural data in the recording window).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the sorted spike times of each unit) and `units/spike_times_index` (ragged array offsets). Only units with `classification == 'good'` contribute. Trial `start_time` values are used to place bin edges.

ii.
```python
allsp=np.asarray(u['spike_times'][:],float); ends=np.asarray(u['spike_times_index'][:],int)
begins=np.r_[0,ends[:-1]]
```

iii. The agent reads the flat spike_times array once per session and uses the index to slice per-unit spike trains.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50 ms bins. For each good unit and each trial, spikes within `[trial_start, trial_start + 4.0s)` are binned using `np.searchsorted` and `np.floor` to assign bin indices, then `np.bincount` gives counts per bin. Counts are divided by bin width (0.05s) to yield firing rates in Hz. **Critically, bins are aligned to trial start, not to the go cue.**

ii.
```python
for row,ui in enumerate(good):
  sp=allsp[begins[ui]:ends[ui]]
  for ti,s in enumerate(starts):
    lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
    idx=np.floor((sp[lo:hi]-s)/DT).astype(np.int64)
    sess_n[ti][row]=np.bincount(idx,minlength=NBIN)[:NBIN]/DT
```

iii. The agent assumed that trial start is always exactly 2.5s before the go cue (i.e., the task timing is fixed). It used `trial_start` as the alignment anchor rather than the actual go cue times.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept. Sessions with no good units are excluded entirely. No additional metric thresholds are applied.

ii.
```python
cls=text(u['classification'][:]); good=np.flatnonzero(cls=='good')
```

iii. The agent found that the paper used region-specific QC classifiers, and the NWB files store this classification directly. This is consistent with the reference approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. **The neural data is aligned to trial start, NOT to the go cue.** The AI assumes the go cue always occurs exactly 2.5s after trial start (`GO_FROM_START=2.5`), so the bin window `[trial_start, trial_start + 4.0s)` is treated as equivalent to `[go - 2.5s, go + 1.5s)`. In reality, early-lick trials replay the sample/delay epochs, shifting the go cue later relative to trial start.

ii.
```python
DT=.05; NBIN=80; GO_FROM_START=2.5
...
for ti,s in enumerate(starts):
    lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
    idx=np.floor((sp[lo:hi]-s)/DT).astype(np.int64)
```

iii. The agent reasoned that the task protocol has a fixed structure with the go cue at 2.5s after trial start. It did not account for early-lick trials where the sample or delay epoch is replayed, which extends the pre-go-cue period beyond 2.5s.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins (DT=0.05s), producing 80 bins over a 4-second window. No rebinning is applied — spike times are directly binned at this resolution.

ii.
```python
DT=.05; NBIN=80
```

iii. The 50 ms bin width matches the instructions. The 80 bins span 4 seconds as specified (-2.5s to +1.5s relative to go cue).

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI does NOT read actual tone onset times from the data. Instead, it uses a **fixed offset**: tone onset is assumed to always occur 0.5s after trial start (`TONE_FROM_START=0.5`), which is -2.0s relative to the go cue.

ii.
```python
TONE_FROM_START=.5
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
time_from_tone=rel_centers-(-GO_FROM_START+TONE_FROM_START)
```

iii. The agent noted that the task protocol specifies the sample epoch starts 0.5s after trial start. It treated this as a fixed offset rather than reading actual tone times from `sample_start_times`. This produces a deterministic ramp identical for all trials.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A fixed time ramp is computed once: bin centers relative to go cue minus the assumed tone-to-go-cue offset. The result is `rel_centers - (-2.5 + 0.5) = rel_centers + 2.0`, giving values from -0.475s to 3.475s. This is constant across all trials.

ii.
```python
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
time_from_tone=rel_centers-(-GO_FROM_START+TONE_FROM_START)
```

iii. Since a fixed offset is used, the time_from_tone input is identical for every trial. The reference solution computes per-trial values using actual `sample_start_times`, which differ across trials (especially on early-lick trials where the sample epoch is replayed).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both are computed on the same bin grid (80 bins of 50 ms), so they share the same time axis by construction. However, since the neural data is aligned to trial start and time_from_tone uses a fixed offset also from trial start, the alignment is internally consistent but misaligned relative to the actual go cue on early-lick trials.

ii.
```python
rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
time_from_tone=rel_centers-(-GO_FROM_START+TONE_FROM_START)
# same rel_centers used for neural binning via trial start anchor
```

iii. The alignment is self-consistent within the AI's framework but incorrect for early-lick trials where the go cue does not occur at the assumed fixed offset from trial start.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset`, `photostim_duration`, and `photostim_power` in the trials table. `photostim_power` is used to determine whether stimulation occurred (`!= 'N/A'`).

ii.
```python
pon=text(tr['photostim_onset'][:]); pdur=text(tr['photostim_duration'][:]); ppow=text(tr['photostim_power'][:])
...
if ppow[ti] != 'N/A':
    a=float(pon[ti])-GO_FROM_START; b=a+float(pdur[ti])
```

iii. The agent identified that photostim fields are stored as text strings with `'N/A'` for control trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time-varying signal is created. For stimulated trials, the onset (relative to trial start) is converted to go-cue-relative time by subtracting `GO_FROM_START=2.5`, and bins whose centers fall in `[onset, onset+duration)` are set to 1. Non-stimulated trials are all zeros.

ii.
```python
stim=np.zeros(NBIN,dtype=np.float32)
if ppow[ti] != 'N/A':
    a=float(pon[ti])-GO_FROM_START; b=a+float(pdur[ti])
    stim[(rel_centers>=a)&(rel_centers<b)]=1
```

iii. Uses `photostim_power` rather than `photostim_onset` to detect stimulation presence. The fixed GO_FROM_START offset is used instead of actual go cue times.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim onset is expressed relative to the assumed go cue position (`trial_start + 2.5s`), and bin centers are on the same assumed grid. This is internally consistent but subject to the same early-lick misalignment issue as the neural data.

ii.
```python
a=float(pon[ti])-GO_FROM_START  # onset relative to assumed go cue
stim[(rel_centers>=a)&(rel_centers<b)]=1
```

iii. Same fixed-offset assumption as the neural alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From `trial_instruction` ('left'/'right') and `outcome` ('hit'/'miss'/'ignore') in the trials table. The actual lick direction is inferred: hit = correct direction, miss = opposite direction, ignore = no lick.

ii.
```python
if outcome[ti]=='ignore': choice=2
elif outcome[ti]=='hit': choice=0 if instr[ti]=='left' else 1
else: choice=1 if instr[ti]=='left' else 0
```

iii. The agent correctly reasoned that lick direction is not stored directly but can be derived from instruction and outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as 0=left, 1=right, 2=no lick. Stored as a per-trial constant broadcast across all 80 time bins.

ii.
```python
sess_o.append(np.vstack((np.full(NBIN,choice,dtype=np.int64),...)))
```

iii. The encoding matches the instructions (left, right, no lick).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains 'ignore', 'miss', 'hit'.

ii.
```python
outcome=text(tr['outcome'][:])
...
oc={'ignore':0,'miss':1,'hit':2}[outcome[ti]]
```

iii. The trials table stores outcome explicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore=0, miss=1, hit=2. Broadcast across all 80 time bins.

ii.
```python
oc={'ignore':0,'miss':1,'hit':2}[outcome[ti]]
np.full(NBIN,oc,dtype=np.int64)
```

iii. Direct mapping matching the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'early' or 'no early'.

ii.
```python
early=text(tr['early_lick'][:])
...
el=1 if early[ti]=='early' else 0
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no, 1=yes. Broadcast across all 80 time bins.

ii.
```python
el=1 if early[ti]=='early' else 0
np.full(NBIN,el,dtype=np.int64)
```

iii. Simple binary encoding matching the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains columns [x, y, DLC_likelihood] with timestamps at ~300 Hz.

ii.
```python
tg=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
td=np.asarray(tg['data'][:],float); tt=np.asarray(tg['timestamps'][:],float)
```

iii. This is the only tongue measurement in the files.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each of the 80 bin centers, the **nearest video frame** is found (rather than averaging all frames in the bin). Frames with DLC likelihood < 0.9 are marked not visible. Visible y-values across all trials in the session are used to compute 40th and 60th percentiles. Values are then discretized: 0 (< 40th), 1 (40th-60th inclusive), 2 (> 60th), 3 (not visible).

ii.
```python
sample_t=(starts[:,None]+(np.arange(NBIN)+.5)*DT).ravel()
jj=np.searchsorted(tt,sample_t); jj=np.clip(jj,1,len(tt)-1)
jj-=((sample_t-tt[jj-1]) <= (tt[jj]-sample_t))
yy=td[jj,1].reshape(ntr,NBIN); conf=td[jj,2].reshape(ntr,NBIN)
visible=np.isfinite(yy)&np.isfinite(conf)&(conf>=DLC_THRESHOLD)
visvals=yy[visible]
q40,q60=np.percentile(visvals,[40,60]) if len(visvals) else (np.nan,np.nan)
tongue=np.full((ntr,NBIN),3,dtype=np.int64)
tongue[visible&(yy<q40)]=0; tongue[visible&(yy>=q40)&(yy<=q60)]=1; tongue[visible&(yy>q60)]=2
```

iii. The agent chose DLC likelihood threshold of 0.9 as a "conventional visibility decision" for DeepLabCut, noting no explicit threshold was found in the reference code.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Discretized using per-session 40th and 60th percentiles of visible y-values (from nearest-frame samples across all trials). Categories: 0 (< 40th), 1 (between 40th and 60th inclusive), 2 (> 60th), 3 (not visible). Note the middle bin uses `<=` for the upper boundary, which differs from the reference's use of `np.digitize` (which uses strict `<`).

ii.
```python
tongue[visible&(yy<q40)]=0
tongue[visible&(yy>=q40)&(yy<=q60)]=1
tongue[visible&(yy>q60)]=2
```

iii. The percentile boundaries (40th/60th) match the instructions. However, percentiles are computed from individual frame y-values at bin centers rather than from bin means, and the boundary logic uses `<=` for the upper edge of the middle bin.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue data uses the **nearest video frame to each bin center**, where bin centers are computed relative to trial start (same fixed-offset assumption as neural data). This differs from the reference which averages all frames within each bin.

ii.
```python
sample_t=(starts[:,None]+(np.arange(NBIN)+.5)*DT).ravel()
jj=np.searchsorted(tt,sample_t); jj=np.clip(jj,1,len(tt)-1)
jj-=((sample_t-tt[jj-1]) <= (tt[jj]-sample_t))
```

iii. Uses nearest-frame lookup rather than bin averaging. The agent reasoned this was appropriate given the ~300 Hz video rate (roughly 15 frames per 50 ms bin).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Session with zero good units**: Excluded entirely (1 session).
- **Missing tongue data**: DLC likelihood < 0.9 -> label 3 ("not visible").
- **Missing photostimulation**: `'N/A'` values produce an all-zeros signal.
- **Missing brain region**: Empty `anno_name` defaults to `'unknown'`.
- **No visible tongue samples**: Percentiles set to NaN, all tongue values default to 3.
- **Trials without spike data (obs_intervals)**: NOT handled. Free-water trials are NOT filtered.

ii.
```python
regs=np.asarray([r if r else 'unknown' for r in regs])
...
q40,q60=np.percentile(visvals,[40,60]) if len(visvals) else (np.nan,np.nan)
```

iii. The agent handled most missing data cases but did not address trials outside `obs_intervals` or free-water trials, which can have completely empty spike data.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file with h5py and the per-unit, per-trial spike binning loop dominate. The agent estimated ~3-4 seconds per session, with 174 sessions taking roughly 5 minutes total.

ii.
```python
for row,ui in enumerate(good):
    sp=allsp[begins[ui]:ends[ui]]
    for ti,s in enumerate(starts):
        lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
        idx=np.floor((sp[lo:hi]-s)/DT).astype(np.int64)
        sess_n[ti][row]=np.bincount(idx,minlength=NBIN)[:NBIN]/DT
```

iii. The double loop (units x trials) for spike binning is the computational bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main double loop over units and trials for spike binning could be partially vectorized. The reference solution vectorizes the trial dimension by flattening all trial bin edges into one array and doing a single `searchsorted` per unit, leaving only a per-unit loop.

ii.
```python
# AI's double loop:
for row,ui in enumerate(good):
    sp=allsp[begins[ui]:ends[ui]]
    for ti,s in enumerate(starts):
        lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
        ...
```

```python
# Reference's single loop (trial dimension vectorized):
edges = (go[:, None] + REL_EDGES[None, :]).ravel()
for r, u in enumerate(good):
    s = allst[starts[u]:offs[u]]
    pos = np.searchsorted(s, edges).reshape(n_trials, N_BINS + 1)
    rates[r] = np.diff(pos, axis=1)
```

iii. The reference eliminates the inner trial loop by building all bin edges at once, which significantly reduces Python overhead.

## 10-c. What processing does the code repeat multiple times?

i. No processing appears to be repeated. The pre-filtering pass reads `classification` from each file once, and the main pass reads it again along with all other data. So the classification check is done twice per session (once in the filter pass, once in the main loop).

ii.
```python
# Pre-filter pass:
for p in files:
  with h5py.File(p,'r') as f:
    if np.any(text(f['units/classification'][:])=='good'): kept.append(p)

# Main processing (reads classification again):
cls=text(u['classification'][:]); good=np.flatnonzero(cls=='good')
```

iii. The pre-filtering pass opens each file to check for good units, then the main pass opens each file again. This doubles the I/O for the classification check but avoids loading all data for the one excluded session.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code processes free-water trials and trials outside `obs_intervals` that have no spike data — these produce zero or near-zero firing rate rows that add noise to the dataset. Additionally, the `photostim_power` field is read but only used as a presence check (could use `photostim_onset != 'N/A'` instead).

ii.
```python
# No filtering of free_water or obs_intervals trials
# All trials processed including those with no spike data
```

iii. Processing trials with no neural data produces uninformative training examples for the decoder.
