# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads three of the four directories under `/app/data`. `beh/Imaging_Exp_info.npy` is loaded first as the master index; it is a dict keyed by experiment type, each value a list of recording entries. For every experiment type the matching `beh/Beh_<type>.npy` is loaded once and each of its per-session behavior dicts is stored in a flat dict `beh[base_session_id]`. Spike files are discovered by globbing `spk/*_neural_data.npy` rather than being constructed from the index, and the usable set of sessions is the intersection of the behavior keys and the spike-file keys (89 sessions). Retinotopy (`retinotopy/<mouse>_<date>_trans.npz`) is loaded lazily, once per session, at the end of that session's processing. `data/process_data` is never read. The AI then makes three passes over the sessions: one to compute global speed quartiles, one to infer the rewarded stimulus, and the main conversion pass; because the behavior dicts are all held in memory the behavior files themselves are read only once, while each spike file is read once in the main loop.

ii.
```python
info=np.load(os.path.join(ROOT,'beh','Imaging_Exp_info.npy'),allow_pickle=True).item()
beh={}; source_group={}; rowmap={}
for group,rows in info.items():
    bf=os.path.join(ROOT,'beh','Beh_'+group+'.npy')
    if not os.path.exists(bf): continue
    bd=np.load(bf,allow_pickle=True).item()
    for r in rows:
        full=sid_of(r); base=base_sid(r)
        key=full if full in bd else base
        if key in bd and base not in beh:
            beh[base]=bd[key]; source_group[base]=group; rowmap[base]=r
spkfiles={os.path.basename(f).replace('_neural_data.npy',''):f for f in glob.glob(os.path.join(ROOT,'spk','*_neural_data.npy'))}
sessions=sorted(set(beh)&set(spkfiles))
```
```python
b=beh[s]; raw=np.load(spkfiles[s],allow_pickle=True).item()['spks']; spk=np.concatenate(raw,axis=0); del raw
```
```python
def region_indices(sid,n):
    p=sid.rsplit('_',1)[0]
    f=os.path.join(ROOT,'retinotopy',p+'_trans.npz')
```

iii. From the trajectory (steps 9–11, 19): the agent first tried to find frame-wise behavior inside the spike files, discovered they contain only `spks`, and then established that "The behavioral dictionaries do contain the complete frame-aligned raw behavior needed for conversion: trial starts/ends, imaging-frame trial index, position, movement/speed, cue frame, lick frames, stimulus identity, reward mode, and corridor geometry. Thus direct trial reconstruction is feasible." It confirmed the loader convention by reading `utils.load_spk` directly: "The exact paper loader concatenates all three `spks` arrays along the neuron axis, so every session should use all recorded neurons. Brain-region indices can be derived from each matching retinotopy file via `neu_area_ID`."

## 1-b. How are the data split into subjects?

i. Subjects are mice. The AI does not read `mname` from the index entry; it takes the text before the first underscore of the session id, which is the same string. The subject list is the sorted set of those names and `subject_idx` is the position of each session's mouse in that list. This yields 19 subjects, matching the paper and the reference.

ii.
```python
... subj.append(s.split('_')[0])
```
```python
subjects=sorted(set(subj)); subject_idx=np.array([subjects.index(x) for x in subj],dtype=np.int16)
```

iii. Not discussed explicitly in the trajectory; the agent treated the mouse name as a self-evident component of the session id it had already parsed out of `Imaging_Exp_info` (step 11: "enumerate unique sessions while avoiding duplicate behavioral analysis files").

## 1-c. How are the data split into sessions?

i. A session is one mouse on one date in one block, i.e. the triple `mname_datexp_blk`. The AI builds two ids per index entry: a "full" id that appends `stimtype` when present, and a "base" id that does not. The behavior dict is looked up under the full id if present, otherwise the base id, but is always *stored* under the base id, and `if ... base not in beh` keeps only the first occurrence, so a recording listed under several experiment types (or several stimtypes) contributes exactly one session. Sessions are then restricted to those that also have a spike file, and sorted alphabetically (which groups them by mouse and then by date). Result: 89 sessions, identical to the reference.

ii.
```python
def sid_of(r): return f"{r['mname']}_{r['datexp']}_{r['blk']}" + (f"_{r['stimtype']}" if 'stimtype' in r else '')
def base_sid(r): return f"{r['mname']}_{r['datexp']}_{r['blk']}"
```
```python
        key=full if full in bd else base
        if key in bd and base not in beh:
            beh[base]=bd[key]; source_group[base]=group; rowmap[base]=r
```
```python
sessions=sorted(set(beh)&set(spkfiles))
```

iii. Step 11: "enumerate unique sessions while avoiding duplicate behavioral analysis files"; the code comment states "Behavior keyed by both full keys and base session keys. Preserve first occurrence; duplicate analysis files contain the same raw frame streams." Step 21 records the result of the intersection ("There are 76 behavior/neural-overlap sessions" at that point, later corrected to 89 once the key-matching fallback was in place; step 25: "Conversion correctly found all 89 sessions represented in both behavior and neural data").

## 1-d. How are the data split into trials?

i. Trials are the ones the behavior declares: `ntrials` of them, with every imaging frame labelled by `ft_trInd`. For trial `tr` the AI selects the frames where `ft_trInd == tr` **and** `ft_CorrSpc` is true (inside the texture), additionally requiring the position to be finite and within `[0, Texture_Length)`. That extra position condition turns out to be a no-op: across all 142 session entries, 0 of 2,154,916 corridor frames are removed by it, because `ft_CorrSpc` already implies `0 <= ft_Pos < 40`. So the frame set per trial is identical to the reference's `(ft_trInd == trial) & ft_CorrSpc`. Trials therefore run from corridor entry to the end of the 4 m texture, exclude the grey space, and have variable length. Before this selection, all frame-wise streams are truncated to `nfr = min(n_spike_frames, n_behavior_frames)`. Note that the module docstring still claims trials "end at corridor exit (GrayFr)", which is stale — the executed code ends them at the end of the texture.

ii.
```python
    nfr=min(spk.shape[1],len(b['ft_trInd']))
    tri=np.asarray(b['ft_trInd'][:nfr]); pos=np.asarray(b['ft_Pos'][:nfr]); speed=np.asarray(b['ft_RunSpeed'][:nfr]); ft=np.asarray(b['ft'][:nfr])
```
```python
    for tr in range(min(ntr,len(trialstim))):
        ix=np.where((tri==tr)&np.asarray(b['ft_CorrSpc'][:nfr],dtype=bool)&np.isfinite(pos)&(pos>=0)&(pos<float(b['Texture_Length'])))[0]
```

iii. Step 20: "A faithful conversion should therefore split concatenated deconvolved traces by `ft_trInd`, retaining corridor frames from `StartFr`/trial assignment through `GrayFr` (corridor exit), rather than position-interpolating neural activity, because the decoder explicitly requires temporal alignment and time-varying licking/speed." Step 43, after the first run: "stored sessions contain a 4 m visual corridor plus 2 m gray space (`Texture_Length=40`, `Corridor_Length=60`), so only `ft_CorrSpc` should be used for the requested four 1 m corridor bins" — this is why the grey space was dropped in the second run.

## 1-e. How are trials filtered based on quality controls?

i. Almost nothing is filtered. A trial is dropped only if fewer than two frames survive the corridor mask (`if ix.size<2: continue`), and a whole session is dropped if fewer than two trials survive (`if len(ns)<2`). In practice no session is dropped: the run kept all 89 sessions and 38,110 trials — i.e. every trial the data declares. There is **no** outlier/duration filter. The recorded metrics show `T_max = 1765` bins, i.e. a single "trial" of 1765 s ≈ 29 minutes, which is the animal standing still inside the corridor rather than traversing it. The reference drops 382 such trials (those above the 99th percentile of traversal length, 238.9 frames ≈ 75 s), leaving 37,729. The consequence in the delivered data is visible in the recorded input ranges: `time since trial start` runs to 1764.5 s and `time to sound cue` down to −1762.8 s.

ii.
```python
        if ix.size<2: continue
```
```python
    if len(ns)<2: print('skip',s,'too few trials'); continue
```

iii. Step 23: "We will retain all Suite2p-classified neurons and all valid trials from the 76 sessions, unless output size proves excessive." The agent's only stated criterion for a valid trial is that it has corridor frames; it never inspected the distribution of trial durations and never mentions stopped animals. Step 46 asserts the opposite of what the data shows — "Visual-corridor trial durations vary appropriately with running speed" — without measuring the tail.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`. That entry is a list of three arrays (one per imaging plane, e.g. 3 × (19408, 31707) for `DR10_2022_07_12_1`) which the AI concatenates along the neuron axis, exactly as `utils.load_spk` does — 58,224 neurons for that session, matching the `iarea` length. Region labels come from `retinotopy/<mouse>_<date>_trans.npz['iarea']`. Nothing else feeds the `neural` stream.

ii.
```python
    b=beh[s]; raw=np.load(spkfiles[s],allow_pickle=True).item()['spks']; spk=np.concatenate(raw,axis=0); del raw
```
```python
    a=np.asarray(np.load(f)['iarea']).ravel()
```

iii. The agent initially misread the three equal-shaped arrays (step 6: "The three matrices likely represent raw/deconvolved activity variants or processing conditions"; step 13: "three neural representations or independently generated variants"), then resolved it by reading the repository loader (step 19): "The exact paper loader concatenates all three `spks` arrays along the neuron axis, so every session should use all recorded neurons." The docstring records this: "All three Suite2p `spks` arrays are concatenated over neurons, exactly as code/utils.py:load_spk. These are non-negative deconvolved fluorescence traces."

## 2-b. How is the `neural` data processed?

i. No normalization, no dF/F, no deconvolution, no z-scoring, no position interpolation — the deconvolved traces are used as they are. The only transformation is temporal: within each trial, frames are grouped into non-overlapping 1 s bins measured from the first corridor frame, and the mean over the frames in each bin is taken per neuron. The result is stored as `float32` with shape (n_neurons, n_bins). If a bin happens to contain no frame, the previous bin's whole population vector is copied forward (zeros for the first bin). I measured this case to be very rare — 13 of 98,042 bins (0.013%) over 20 sessions — so it is a fabrication in principle but negligible in practice. The `float32` choice (against the reference's `float16`) is the main reason the output is 94.72 GB.

ii.
```python
        N=np.empty((spk.shape[0],nb),np.float32); P=np.empty(nb,np.float32); V=np.empty(nb,np.float32)
        L=np.zeros(nb,np.int16)
        for j in range(nb):
            jj=ix[bins==j]
            if jj.size: N[:,j]=spk[:,jj].mean(1,dtype=np.float32); P[j]=np.nanmean(pos[jj]); V[j]=np.nanmean(speed[jj])
            else: N[:,j]=N[:,j-1] if j else 0; P[j]=P[j-1] if j else 0; V[j]=V[j-1] if j else 0
```

iii. Step 17: "The shown downstream code computes d-prime from raw corridor frames and does not quality-filter neurons beyond Suite2p cell classification already applied upstream. It normalizes only for specific figures; the conversion should retain deconvolved traces rather than applying that figure-specific normalization." Step 22/23 give the reason for averaging: "a better compromise is fixed-width 1-second temporal bins from corridor entry to exit, averaging deconvolved activity and behavior within each bin... This preserves temporal ordering and cue/lick events while reducing each session substantially."

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered at all. Every neuron in every plane is kept — 4,691,034 in total, against the reference's 4,105,393 (ratio 1.14, which failed the neuron-count check). The AI does assign a region index per neuron, but the mapping it uses is wrong. The repository's `utils.neu_area_ID` maps `iarea == 8 → V1`, `{0,1,2,9} → mHV`, `{5,6} → lHV`, `{3,4} → aHV`. The AI's code claims to follow that function but maps `iarea == 1 → V1`, `{2,3} → medial`, `{4,5} → anterior`, `{6,7} → lateral`, everything else → "unassigned". Since `iarea == 8` is the single largest group (26,069 / 58,224 = 45% for `DR10_2022_07_12`), real V1 is entirely labelled "unassigned": the run reported only 147,071 "V1" neurons and ~3.05 M (65%) "unassigned". There is also a silent fallback that returns all zeros if the retinotopy file is missing and zero-pads if the lengths disagree.

ii.
```python
def region_indices(sid,n):
    p=sid.rsplit('_',1)[0]
    f=os.path.join(ROOT,'retinotopy',p+'_trans.npz')
    if not os.path.exists(f): return np.zeros(n,dtype=np.int16)
    a=np.asarray(np.load(f)['iarea']).ravel()
    # Repository neu_area_ID: 1=V1, 2=medial, 3=anterior, 4=lateral; 0=unassigned.
    out=np.zeros(len(a),dtype=np.int16)
    out[a==1]=1; out[np.isin(a,[2,3])]=2; out[np.isin(a,[4,5])]=3; out[np.isin(a,[6,7])]=4
    if len(out)!=n:
        # Retinotopy corresponds to concatenated planes; guard malformed metadata.
        z=np.zeros(n,dtype=np.int16); z[:min(n,len(out))]=out[:min(n,len(out))]; out=z
    return out
```
```python
 'brain_regions':['unassigned','V1','medial visual areas','anterior visual areas','lateral visual areas'],'brain_region_idx':bridx,
```

iii. Docstring: "Suite2p cell classification is the supplied curation; no post-hoc selectivity filter." Step 21 shows the agent *considered* area-based exclusion purely for size reasons — "This determines whether excluding `iarea==0` unassigned/non-visual ROIs—as paper area-specific analyses may do—makes the dataset tractable" — and step 22, "If all neurons are consumed, use the paper's four named visual regions and exclude unassigned ROIs", but step 23 settled on "We will retain all Suite2p-classified neurons". Step 43 identified only the filename bug ("every neuron was labeled unassigned because the retinotopy filename was constructed without the day component"), not the code-value mapping; step 56 then reports the wrong counts as a success: "populated retinotopic region assignments: 147,071 V1, 708,036 medial, 441,492 anterior, 341,596 lateral, plus unassigned neurons."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry / trial start. The AI sets the trial origin `t0` to the timestamp of the *first corridor frame* of that trial (rather than interpolating the fractional `StartFr`), and bins forward from there, so bin 0 begins at corridor entry. Trials are left at their natural, variable length; nothing is padded or truncated to a common window. The metadata reports `temporal_alignment_event = 'trial start / corridor entry'`, `off_start = 0.0`, `off_end = None`. This is conceptually the same alignment as the reference; the only difference is the sub-frame offset between the first corridor frame and the fractional `StartFr`.

ii.
```python
        t0=tsec[ix[0]]; rel=tsec[ix]-t0; nb=max(1,int(np.floor(rel[-1]/BIN_S))+1); bins=np.minimum((rel/BIN_S).astype(int),nb-1)
```
```python
'temporal_alignment_event':'trial start / corridor entry','off_start':0.0,'off_end':None,
```

iii. Step 20: "the decoder explicitly requires temporal alignment"; step 22: "the task explicitly asks temporal alignment to corridor entry and time-varying cue/lick/speed, so a better compromise is fixed-width 1-second temporal bins from corridor entry to exit". Step 23: "bins frames in 1-second bins from trial/corridor entry through corridor exit".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes, rebinning is applied. The native imaging rate is ~3.17 Hz (315 ms per frame), and the AI averages roughly three native frames into each non-overlapping 1000 ms bin, indexed by `floor(rel / 1 s)` from corridor entry. `metadata['time_bin_size']` is 1000.0 ms and `metadata['temporal_binning']` says "non-overlapping 1 s means from corridor entry to exit". The consequence is that the median trial is 10.4 bins long against the reference's 31.0 native frames (ratio 0.33), which failed the median-trial-length check. The stated motivation — tractability — is not really borne out: the 3× reduction was offset by keeping all 4.69 M neurons in `float32`, and the output is still 94.72 GB, comparable to a native-resolution `float16` conversion.

ii.
```python
ROOT='/app/data'; BIN_S=1.0
```
```python
        t0=tsec[ix[0]]; rel=tsec[ix]-t0; nb=max(1,int(np.floor(rel[-1]/BIN_S))+1); bins=np.minimum((rel/BIN_S).astype(int),nb-1)
```
```python
'time_bin_size':1000.0, ... 'source_frame_rate_hz':'approximately 3.2, variable','temporal_binning':'non-overlapping 1 s means from corridor entry to exit',
```

iii. Step 13: "Imaging timestamps are MATLAB serial-day values: median spacing × 86,400 is ~0.315 s, i.e. ~3.18 Hz." Step 21–23 give the justification: "Fully materializing every neuron and native frame into a pickle would remain hundreds of GB and make decoder training impractical"; "a better compromise is fixed-width 1-second temporal bins from corridor entry to exit, averaging deconvolved activity and behavior within each bin. This reduces size by about 3× only and may remain large"; "We will use 1-second non-overlapping temporal bins aligned to corridor entry, averaging deconvolved traces. This preserves temporal ordering and cue/lick events while reducing each session substantially."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the fractional imaging-frame index of the cue on each trial) and `ft` (the MATLAB datenum timestamp of every imaging frame). No session has a non-finite `SoundFr` (checked across all 142 session entries), so the NaN fallback never fires.

ii.
```python
    ntr=int(b['ntrials']); trialstim=np.asarray(b['TrialStim']); sound=np.asarray(b['SoundFr']); lick=np.asarray(b['LickFr'])
```
```python
        cue_rel=(np.interp(sound[tr],np.arange(nfr),tsec)-t0) if tr<len(sound) and np.isfinite(sound[tr]) else np.nan
```

iii. Step 13: "Trial event fields such as `StartFr`, `SoundFr`, and `EndFr` are fractional imaging-frame indices, while `ft_trInd` directly assigns frames to trials." Code comment: "Cue event is a fractional global imaging-frame index."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Frame timestamps are converted from MATLAB datenums to elapsed seconds (`(ft - ft[0]) * 86400`). The fractional cue frame is interpolated onto that time axis and expressed relative to the trial origin, giving `cue_rel`. The input is then `cue_rel − elapsed`, where `elapsed` is the bin-centre time, so the value is **positive before the cue and negative after** — the same sign convention as the reference. It is stored as `float32`. Because no duration filter is applied, this variable reaches −1762.8 s in the delivered data.

ii.
```python
    tsec=(ft-ft[0])*86400.0
```
```python
        elapsed=(np.arange(nb,dtype=np.float32)+.5)*BIN_S
        tcue=(cue_rel-elapsed).astype(np.float32) if np.isfinite(cue_rel) else np.full(nb,np.nan,np.float32)
```

iii. Step 12: "Frame timestamps are not in seconds—the naïve reciprocal-difference calculation produced ~274 kHz—so they likely use MATLAB day units or another scale and must be converted before temporal binning", resolved in step 13. Step 23: "creates continuous time-to-cue, day, elapsed time, and rewarded-corridor inputs".

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is defined on exactly the same 1 s bin grid as the neural data for that trial: the same `t0`, the same `nb`, and bin centres `(j + 0.5) s`. Since the cue time is placed on the same seconds axis derived from `ft`, the two streams share one clock.

ii.
```python
        t0=tsec[ix[0]]; rel=tsec[ix]-t0; nb=max(1,int(np.floor(rel[-1]/BIN_S))+1)
```
```python
        elapsed=(np.arange(nb,dtype=np.float32)+.5)*BIN_S
        tcue=(cue_rel-elapsed).astype(np.float32) ...
        inp=np.vstack([tcue,day,elapsed,rewarded]).astype(np.float32)
```

iii. Implicit in step 22/23: everything is put on the common "1-second temporal bins aligned to corridor entry" grid, so no separate alignment step is needed.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp` in the `Imaging_Exp_info` entry for the session (kept in `rowmap`), and from the mouse name used to find that mouse's earliest recorded date. No other field is used.

ii.
```python
def date_ordinal(date):
    return float(np.datetime64(date.replace('_','-'),'D').astype(int))
```
```python
subject_day0={}
for s in sessions:
    mouse=s.split('_')[0]; d=date_ordinal(rowmap[s]['datexp'])
    subject_day0[mouse]=min(subject_day0.get(mouse,d),d)
```

iii. Step 19: "need make final choices for rewarded-corridor identification, training-day values, trial inclusion, and practical time resampling. These should be derived directly from metadata and methods rather than guessed."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The date string `YYYY_MM_DD` is converted to a day ordinal; the value is `ordinal − (earliest ordinal for that mouse) + 1`, i.e. **calendar days elapsed** since the mouse's first included imaging session, starting at 1. It is broadcast as a constant across all bins of every trial in that session, as `float32`. The delivered range is 1–93. The reference instead counts *recorded sessions* per mouse, giving 0–7; the two encode the same ordering but on very different scales, and the AI's version is dominated by the multi-week gaps between pre- and post-learning blocks. This difference is what produced the large `input_range_error_day_of_training` in the metrics.

ii.
```python
        day=np.full(nb,date_ordinal(rowmap[s]['datexp'])-subject_day0[s.split('_')[0]]+1.0,np.float32)
```

iii. This was a deliberate second-pass correction. Step 43: "training day should be relative to each subject's first imaging date rather than an absolute date ordinal"; step 44: "The script was patched with the corrected visual-corridor mask, 4 m position scale, relative training days, and retinotopy filename." (The first run had used the raw absolute date ordinal, which was clearly wrong.)

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Strictly speaking, only the bin index — the value is constructed analytically from the bin grid rather than from a raw field. The bin grid itself comes from `ft` (frame timestamps) and `ft_trInd`/`ft_CorrSpc` (which frames belong to the trial's corridor traversal). `StartFr` is **not** used, despite the module docstring mentioning it.

ii.
```python
        t0=tsec[ix[0]]; rel=tsec[ix]-t0; nb=max(1,int(np.floor(rel[-1]/BIN_S))+1); bins=np.minimum((rel/BIN_S).astype(int),nb-1)
```
```python
        elapsed=(np.arange(nb,dtype=np.float32)+.5)*BIN_S
```

iii. Step 23: "creates continuous time-to-cue, day, elapsed time, and rewarded-corridor inputs". The agent treated elapsed time as a property of the bin grid it had just defined.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The trial origin is the timestamp of the first corridor frame; the input is the bin-centre time, `(j + 0.5) × 1 s`, so it starts at 0.5 s and increases in exact 1 s steps. It is always non-negative (there are no pre-trial bins) and is stored as `float32`. Because no duration filter is applied, it reaches 1764.5 s on the stalled trials. The reference instead takes each frame's true timestamp minus the interpolated `StartFr`, which is the same quantity up to the sub-frame origin offset and to within-bin jitter.

ii.
```python
        elapsed=(np.arange(nb,dtype=np.float32)+.5)*BIN_S
        ...
        inp=np.vstack([tcue,day,elapsed,rewarded]).astype(np.float32)
```

iii. As above (step 23). No separate justification is given for using idealised bin centres rather than measured frame times; it follows from the decision to average into fixed 1 s bins.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. By construction: `elapsed` has exactly `nb` entries, one per neural bin, on the same origin `t0` as the neural data.

ii.
```python
        N=np.empty((spk.shape[0],nb),np.float32)
        ...
        elapsed=(np.arange(nb,dtype=np.float32)+.5)*BIN_S
        inp=np.vstack([tcue,day,elapsed,rewarded]).astype(np.float32)
```

iii. Implicit in the single shared bin grid (step 22/23).

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `TrialStim` (the per-trial stimulus label) and `RewardFr` (the frame at which reward was delivered, NaN when none). The AI does **not** use `isRew`, the boolean per-trial field that directly states whether the trial was run in the rewarded corridor, even though `isRew` is present in every one of the 142 session entries.

ii.
```python
reward_stim={}
for s in sessions:
    b=beh[s]; st=np.asarray(b['TrialStim']); rw=np.asarray(b['RewardFr'])
    counts={x:int(np.isfinite(rw[st==x]).sum()) for x in np.unique(st)}
    reward_stim[s]=max(counts,key=counts.get) if counts and max(counts.values())>0 else None
```

iii. Code comment: "Rewarded corridor is an experimental property. Infer the rewarded visual category only where rewards exist; task sessions consistently identify it. Unsupervised/naive sessions have no rewarded corridor, hence all zeros." Step 19 lists "rewarded-corridor identification" as something to resolve; step 23: "creates ... rewarded-corridor inputs". The trajectory never mentions `isRew`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Per session, the AI counts, for each distinct `TrialStim` value, how many of its trials have a finite `RewardFr`, and takes the arg-max as "the rewarded stimulus" (None if no reward was ever delivered — unsupervised and naive sessions). A trial is then flagged 1 if its `TrialStim` equals that stimulus, else 0, and the flag is broadcast across all bins as `float32`. I checked this reconstruction against `isRew` over all 63,177 trials in the behavior files: it disagrees on 155 trials (0.25%). So it is close to, but not identical to, the ground-truth field, and it inherits any weakness of `TrialStim` (see 7-a).

ii.
```python
        rewarded=np.full(nb,int(reward_stim[s] is not None and str(trialstim[tr])==reward_stim[s]),np.float32)
```

iii. Same as 6-a: the comment asserts the inference is reliable ("task sessions consistently identify it") and that unsupervised/naive sessions correctly come out all-zero.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `TrialStim`. This is the wrong field. `TrialStim` is an *anonymised/remapped* analysis label, not the physical texture: in `TX119_2023_12_12_1`, for instance, wall texture `rock1` is labelled `circle1`, `rock2` → `circle2`, `wood1` → `leaf1`, `wood2` → `leaf2`, and `wood5` → the literal placeholder string `'stimulus_of_trial'`. Across the dataset `TrialStim` takes 8 values (`circle1, circle2, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, stimulus_of_trial`), and 8,594 of 63,177 trials (13.6%) carry the placeholder; 64 of the 142 session entries are partially masked. The field that actually names the texture is `WallName` (15 values, resolving to the four base textures circle / leaf / rock / wood), which the reference uses.

ii.
```python
stim_names=sorted({str(x) for s in sessions for x in np.unique(beh[s]['TrialStim'])}); stim_id={x:i for i,x in enumerate(stim_names)}
```
```python
        cat=np.full(nb,stim_id[str(trialstim[tr])],np.int16)
```

iii. Step 11 lists "stimulus identity" among the usable behavior fields, and step 19 plans to "inspect ... one session's trial-level relationships among stimulus, cue, rewards, and licks". The agent never compared `TrialStim` with `WallName` and never reports noticing the `'stimulus_of_trial'` placeholder; step 56 simply states the final data includes "stimulus identity".

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. A single global vocabulary is built as the sorted set of all `TrialStim` strings seen across all 89 sessions, giving 8 classes, and each trial's label is the index into that list, broadcast across all of its bins as `int16`. No mapping of crop variants onto base textures is applied. `output_values[0]` is therefore `['circle1','circle2','leaf1','leaf1_swap1','leaf1_swap2','leaf2','leaf3','stimulus_of_trial']`. The recorded metrics confirm 8 classes against the reference's 4, and the class-fraction comparison for this variable failed. The practical damage is twofold: rock/wood trials are labelled as circle/leaf, and 13.6% of trials are collapsed into a meaningless placeholder class that mixes several distinct textures.

ii.
```python
stim_names=sorted({str(x) for s in sessions for x in np.unique(beh[s]['TrialStim'])}); stim_id={x:i for i,x in enumerate(stim_names)}
```
```python
        cat=np.full(nb,stim_id[str(trialstim[tr])],np.int16)
```
```python
 'output_values':[stim_names,['not licking','licking'],['0-1 m','1-2 m','2-3 m','3-4 m'],['Q1','Q2','Q3','Q4']],
```

iii. No explicit justification for the granularity is given. The global vocabulary is a deliberate choice so that class indices are comparable across sessions. The agent read the paper's stimulus description (step 8: "For simplicity, we denote the stimuli as 'leaf' and 'circle', even though other visual stimuli were also used ('rock' and 'bricks')" appears in the methods it summarised) but did not reconcile that with the field it chose.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the fractional imaging-frame index of every lick in the session, together with `ft` for the time conversion. This matches the reference.

ii.
```python
    ntr=int(b['ntrials']); trialstim=np.asarray(b['TrialStim']); sound=np.asarray(b['SoundFr']); lick=np.asarray(b['LickFr'])
```

iii. Step 11 identified "lick frames" as directly available in the behavior dict.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Non-finite lick frames are dropped. The remaining fractional frame indices are interpolated onto the seconds axis, expressed relative to the trial origin `t0`, floored to a bin index, restricted to `[0, nb)`, and the corresponding bins are set to 1; all other bins are 0. So a bin is 1 if at least one lick falls in it. Stored as `int16`, two classes. One small wrinkle: `np.interp` *clamps* rather than drops lick frames beyond the last imaged frame, mapping them to the final frame time, whereas the reference discards them — this can add at most a spurious lick to the last trial of a session. The delivered class fractions are within tolerance of the reference (fraction error 0.037).

ii.
```python
        lf=lick[np.isfinite(lick)] if lick.size else lick
        if lf.size:
            lt=np.interp(lf,np.arange(nfr),tsec)-t0; lb=(lt/BIN_S).astype(int); lb=lb[(lb>=0)&(lb<nb)]; L[np.unique(lb)]=1
```

iii. Code comment: "Any lick in a temporal bin -> binary licking." Step 22: the 1 s temporal binning was chosen specifically because "the task explicitly asks temporal alignment to corridor entry and time-varying cue/lick/speed".

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Through the same trial origin `t0` and the same 1 s bin grid of length `nb` as the neural array, with licks assigned to bins by their interpolated time. Because the window is the contiguous corridor traversal, only licks that fall inside the trial's temporal span are kept.

ii.
```python
        L=np.zeros(nb,np.int16)
        ...
            lt=np.interp(lf,np.arange(nfr),tsec)-t0; lb=(lt/BIN_S).astype(int); lb=lb[(lb>=0)&(lb<nb)]; L[np.unique(lb)]=1
        out=np.vstack([cat,L,pcat,vcat]).astype(np.int16)
```

iii. Implicit in the single shared bin grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the corridor position at each imaging frame (in virtual units; `Texture_Length = 40` units spans the 4 m textured corridor, `Corridor_Length = 60` including the 2 m grey space), plus the per-session `Texture_Length` scalar used as the scale.

ii.
```python
    tri=np.asarray(b['ft_trInd'][:nfr]); pos=np.asarray(b['ft_Pos'][:nfr]); ...
```
```python
        pcat=np.clip((P/float(b['Texture_Length'])*4).astype(int),0,3).astype(np.int16)
```

iii. Step 43: "stored sessions contain a 4 m visual corridor plus 2 m gray space (`Texture_Length=40`, `Corridor_Length=60`), so only `ft_CorrSpc` should be used for the requested four 1 m corridor bins". Note the module docstring was never updated and still says "Stored corridor coordinates span 60 units = 4 m", which contradicts both the data and the implemented code.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Within each 1 s bin, the mean position over the frames in that bin is taken (`np.nanmean`), producing a continuous `float32` value per bin; that mean is then discretised. Empty bins carry the previous bin's value forward (0 for the first bin). The reference discretises the raw per-frame position instead, but since position increases monotonically within a traversal and a bin spans ~3 frames, the two agree closely — the recorded position class-fraction error is 0.018, well inside tolerance.

ii.
```python
            if jj.size: N[:,j]=spk[:,jj].mean(1,dtype=np.float32); P[j]=np.nanmean(pos[jj]); V[j]=np.nanmean(speed[jj])
            else: N[:,j]=N[:,j-1] if j else 0; P[j]=P[j-1] if j else 0; V[j]=V[j-1] if j else 0
```

iii. Step 22/23: behavior is averaged in the same 1 s bins as the neural data.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. `clip(int(P / Texture_Length × 4), 0, 3)` — a linear map of the 0–40 unit textured corridor onto four equal 1 m bins, labelled `['0-1 m','1-2 m','2-3 m','3-4 m']`. With `Texture_Length = 40` this is arithmetically identical to the reference's `clip(ft_Pos // 10, 0, 3)`. The grey space is excluded upstream by the `ft_CorrSpc` mask, so the clip almost never binds. The recorded output has exactly 4 classes with fractions matching the reference.

ii.
```python
        # Linear mapping required by task: 4 equal 1 m bins over each session corridor.
        pcat=np.clip((P/float(b['Texture_Length'])*4).astype(int),0,3).astype(np.int16)
```
```python
'position_mapping':'Visual-corridor coordinates (ft_CorrSpc; Texture_Length=40 units) mapped to four equal 1 m bins; gray space excluded',
```

iii. Step 16 flagged the ambiguity — "The notebook comment says 60 bins over a 6 m corridor, conflicting with the task's explicit 4 equal 1 m bins and methods' apparent 4 m wording" — and step 43 resolved it correctly against `Texture_Length`. Step 54: "reduced by excluding the 2 m gray-space segment while preserving the 4 m visual corridor required by the decoder task."

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is sampled per imaging frame, and the per-bin mean is computed in the same loop, over the same frame indices `jj`, as the per-bin neural mean. So position and neural activity share the bin grid exactly.

ii.
```python
        for j in range(nb):
            jj=ix[bins==j]
            if jj.size: N[:,j]=spk[:,jj].mean(1,dtype=np.float32); P[j]=np.nanmean(pos[jj]); V[j]=np.nanmean(speed[jj])
```

iii. Implicit in the single shared bin grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame. Same source as the reference.

ii.
```python
    ... speed=np.asarray(b['ft_RunSpeed'][:nfr]); ...
```
```python
    b=beh[s]; tri=np.asarray(b['ft_trInd']); pos=np.asarray(b['ft_Pos']); v=np.asarray(b['ft_RunSpeed'])
```

iii. Step 11 identified "movement/speed" as directly available per frame.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Two steps. First, a pre-pass over all 89 sessions concatenates the per-frame speeds of all frames that will be kept (corridor frames with finite position/speed/trial index) and computes the 25/50/75% quantile edges *globally*, i.e. pooled across sessions. Second, in the main loop, the per-bin mean speed is computed alongside the per-bin mean neural activity and position, and that bin-averaged speed is what gets discretised. The reference instead computes the split per session and on the raw per-frame speeds.

ii.
```python
speeds=[]
for s in sessions:
    b=beh[s]; tri=np.asarray(b['ft_trInd']); pos=np.asarray(b['ft_Pos']); v=np.asarray(b['ft_RunSpeed'])
    ok=np.isfinite(tri)&np.isfinite(pos)&np.isfinite(v)&np.asarray(b['ft_CorrSpc'],dtype=bool)&(pos>=0)&(pos<float(b['Texture_Length']))
    speeds.append(v[ok].astype(np.float32))
q=np.quantile(np.concatenate(speeds),[.25,.5,.75]).astype(np.float32); del speeds
```
```python
            if jj.size: ... V[j]=np.nanmean(speed[jj])
```

iii. Step 23: "To avoid loading all sessions twice for speed quartiles, first derive global quartiles from framewise speed over usable trial frames." The comment in the code reads "Global speed quartiles from the same valid corridor frames included below."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(V, q, right=False)` against the three global quantile edges, labelled `['Q1','Q2','Q3','Q4']`. This does **not** produce four bins of 25% of the data, which is what the task asks for. Roughly 27.5% of corridor frames have a speed of exactly 0 (the VR is stationary below the 6 cm/s running threshold), so the 25th-percentile edge is `q[0] = 0.0`; with `right=False`, `digitize(0.0, [0.0, ...])` returns 1, so the entire zero-speed mass lands in class 1 and class 0 is left with only the (empty) strictly-negative range. On a 25-session sample I measured the resulting frame-level split as 9.3% / 40.7% / 25.0% / 25.0%. The delivered file shows the same problem: `output_fraction_error_running_speed = 0.32`, the only output class-fraction check that failed besides the stimulus. Averaging speeds within a bin before applying frame-level edges shifts the split further.

ii.
```python
q=np.quantile(np.concatenate(speeds),[.25,.5,.75]).astype(np.float32); del speeds
```
```python
        vcat=np.digitize(V,q,right=False).astype(np.int16)
```
```python
 'output_values':[stim_names,['not licking','licking'],['0-1 m','1-2 m','2-3 m','3-4 m'],['Q1','Q2','Q3','Q4']],
```

iii. The agent's only stated reasoning is that quartiles should be global so that classes are comparable across sessions (step 23, and the code comment). It never checked the resulting class balance, and never noticed the tie at zero. By contrast the reference explicitly addressed it: "Up to a third of the frames of a session sit at exactly zero speed, so we take a rank ordering and divide into 4 equal bins."

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is per imaging frame, and the per-bin mean is computed in the same loop and over the same frame indices `jj` as the neural mean, so the speed output shares the trial's bin grid exactly.

ii.
```python
        for j in range(nb):
            jj=ix[bins==j]
            if jj.size: N[:,j]=spk[:,jj].mean(1,dtype=np.float32); P[j]=np.nanmean(pos[jj]); V[j]=np.nanmean(speed[jj])
```

iii. Implicit in the single shared bin grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several small guards, most of which never actually fire on this dataset:
- Behavior can run past the imaging, so all frame-wise streams are truncated to `nfr = min(n_spike_frames, n_behavior_frames)` — the same cut the reference makes.
- Frames with non-finite `ft_trInd`, `ft_Pos` or out-of-range position are excluded from trials (a no-op here: `ft_CorrSpc` already implies `0 <= ft_Pos < 40`).
- Non-finite `LickFr` entries are dropped; licks outside the trial window are dropped (but licks past the last imaged frame are clamped by `np.interp` rather than dropped).
- A trial with a non-finite `SoundFr` would get an all-NaN cue input; no session has one, so this never fires.
- Within-bin averages use `np.nanmean`; a bin with no frames copies the previous bin forward for neural, position and speed (or zeros at bin 0). I measured this at 13 of 98,042 bins over 20 sessions (0.013%).
- A missing retinotopy file silently returns all-zero region indices, and a length mismatch between `iarea` and the neuron count silently zero-pads. Neither condition occurs (`iarea` length equals the concatenated neuron count).
- Sessions listed under several experiment types are deduplicated; sessions without a spike file are skipped; sessions ending with fewer than 2 usable trials are skipped (none are).
There is no handling of the *large* data problem in this dataset — the stalled multi-minute "traversals" (see 1-e).

ii.
```python
    nfr=min(spk.shape[1],len(b['ft_trInd']))
```
```python
        ix=np.where((tri==tr)&np.asarray(b['ft_CorrSpc'][:nfr],dtype=bool)&np.isfinite(pos)&(pos>=0)&(pos<float(b['Texture_Length'])))[0]
        if ix.size<2: continue
```
```python
            else: N[:,j]=N[:,j-1] if j else 0; P[j]=P[j-1] if j else 0; V[j]=V[j-1] if j else 0
```
```python
    if not os.path.exists(f): return np.zeros(n,dtype=np.int16)
    ...
    if len(out)!=n:
        z=np.zeros(n,dtype=np.int16); z[:min(n,len(out))]=out[:min(n,len(out))]; out=z
```

iii. Code comment: "Retinotopy corresponds to concatenated planes; guard malformed metadata." The agent's general posture (steps 3–6) was defensive because of the file sizes: "Blind loading would be unsafe, so we need identify array shapes/dtypes and paper-defined session metadata and processing first." It did not report encountering any actual data defects — step 49: "No trial, region, or alignment errors have occurred."

## 12-a. What are the most time-consuming steps of the code?

i. Three things dominate, in rough order:
1. Reading the spike files — 89 pickled `.npy` object files totalling ~405 GB (the agent measured "374 GB of source activity" for the subset it had enumerated at the time). These cannot be memory-mapped because they are object arrays, so each is fully unpickled, and `np.concatenate(raw, axis=0)` then makes a second full-size copy of the session in RAM before `del raw`.
2. The inner per-bin Python loop, which for every one of the ~490,000 bins in the dataset does a fancy-index `spk[:, jj]` and a mean over ~50,000 neurons. This is a pure-Python loop wrapping small NumPy calls over the largest array in the program.
3. Writing and re-reading the 94.72 GB pickle. The agent's own polling shows the write alone took several minutes, and the validator took long enough to need repeated polling.
On top of this, the whole conversion was run twice end to end (once producing a 137.48 GB file that was then discarded after the gray-space/retinotopy/day fixes).

ii.
```python
    b=beh[s]; raw=np.load(spkfiles[s],allow_pickle=True).item()['spks']; spk=np.concatenate(raw,axis=0); del raw
```
```python
        for j in range(nb):
            jj=ix[bins==j]
            if jj.size: N[:,j]=spk[:,jj].mean(1,dtype=np.float32); ...
```
```python
with open('/app/converted_data.pkl','wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
```

iii. Step 5: "All neural files are object-dtype `.npy` containers, so NumPy cannot memory-map them." Step 21: "There are 76 behavior/neural-overlap sessions, 31,443 trials, and 374 GB of source activity." Steps 24–41 and 44–54 are almost entirely the agent polling the two long conversion runs.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Four, the first of which is the significant one:
1. **The per-bin averaging loop** (`for j in range(nb)`). Because the trial's frames are contiguous and sorted, the whole per-bin mean could be done in one call with `np.add.reduceat(spk[:, ix], starts, axis=1)` divided by the bin counts, or as a single matrix multiply against a sparse bin-indicator matrix. As written it performs one fancy-index and one reduction per bin over all ~50,000 neurons, plus `ix[bins==j]`, which itself rescans the whole trial index once per bin (an O(n_bins × n_frames) scan hidden inside the loop).
2. **The per-trial frame search** (`np.where((tri==tr) & ...)` inside `for tr in ...`), which rescans the entire session's frame index once per trial — O(n_trials × n_frames). Grouping all frames by `ft_trInd` in one pass (e.g. `np.argsort` or `np.bincount`-based splitting) would make this a single sweep. The reference has the same pattern and calls it out as negligible; here it is amplified because the mask is rebuilt from `b['ft_CorrSpc'][:nfr]` each time.
3. **`subjects.index(x)`** inside a list comprehension over sessions — O(n_sessions × n_subjects); trivial in absolute terms.
4. The three separate full passes over `sessions` (speed quartiles, reward inference, conversion) could be folded into fewer passes.

ii.
```python
        for j in range(nb):
            jj=ix[bins==j]
```
```python
    for tr in range(min(ntr,len(trialstim))):
        ix=np.where((tri==tr)&np.asarray(b['ft_CorrSpc'][:nfr],dtype=bool)&np.isfinite(pos)&(pos>=0)&(pos<float(b['Texture_Length'])))[0]
```
```python
subjects=sorted(set(subj)); subject_idx=np.array([subjects.index(x) for x in subj],dtype=np.int16)
```

iii. The agent never discusses vectorisation. Its performance reasoning is entirely about output *size* (steps 21–23) and it treats the long runtime as expected, polling patiently through ~30 steps without profiling.

## 12-c. What processing does the code repeat multiple times?

i.
- `np.asarray(b['ft_CorrSpc'][:nfr], dtype=bool)` is rebuilt **inside the trial loop**, i.e. once per trial (up to ~600 times per session) instead of once per session, even though `tri`, `pos`, `speed` and `ft` were hoisted out on the line above.
- The lick interpolation `np.interp(lf, np.arange(nfr), tsec)` is recomputed **for every trial** over the *entire session's* lick vector (thousands of licks), and `np.arange(nfr)` is reallocated each time, when one session-level computation would do.
- `date_ordinal(rowmap[s]['datexp'])` and `s.split('_')[0]` are recomputed inside the trial loop although they are session constants.
- `float(b['Texture_Length'])` is re-extracted per trial (twice: in the mask and in `pcat`).
- Every session is iterated over three times at the top level (speed quartiles, reward inference, main loop).
- `np.unique(beh[s]['TrialStim'])` is computed once for the vocabulary and the labels are looked up again per trial.
- Most expensively of all: the entire conversion was executed twice (137.48 GB run, then the corrected 94.72 GB run), so all the I/O was paid twice.

ii.
```python
    for tr in range(min(ntr,len(trialstim))):
        ix=np.where((tri==tr)&np.asarray(b['ft_CorrSpc'][:nfr],dtype=bool)&np.isfinite(pos)&(pos>=0)&(pos<float(b['Texture_Length'])))[0]
```
```python
        lf=lick[np.isfinite(lick)] if lick.size else lick
        if lf.size:
            lt=np.interp(lf,np.arange(nfr),tsec)-t0; lb=(lt/BIN_S).astype(int); ...
```
```python
        day=np.full(nb,date_ordinal(rowmap[s]['datexp'])-subject_day0[s.split('_')[0]]+1.0,np.float32)
```

iii. Not discussed. Step 43 documents why the second full run was needed: "It exposed an important metadata bug: every neuron was labeled unassigned because the retinotopy filename was constructed without the day component. It also highlighted two scientific corrections needed before finalizing."

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Storing neurons that the reference discards.** All 4,691,034 neurons are written out; 585,641 of them fall outside the four visual areas, and with the AI's (incorrect) mapping ~3.05 M end up in an "unassigned" bucket. This is pure output bloat relative to the reference's 4,105,393, and it failed the neuron-count check.
- **`float32` instead of `float16` for the neural array**, which doubles the size of the dominant field for no analytic benefit; combined with the above this is the main reason the pickle is 94.72 GB.
- **The whole `reward_stim` inference pass** over all 89 sessions, which reconstructs — with 0.25% error — a value that is directly available as `isRew`.
- **Keeping the 382-odd stalled trials** whose bins are all the same position and zero speed; they contribute thousands of near-duplicate samples and distort the input ranges, and any sensible downstream analysis would drop them.
- `import re` is never used; `source_group`, `n_source_trials` and `trial_indices` are computed and stored per session but are not used by the decoder.
- `del spk; gc.collect()` is placed *after* the `continue` for skipped sessions, so it is bypassed in exactly the case where freeing memory would matter; and `gc.collect()` on every session is itself an avoidable full-heap scan over a multi-gigabyte working set.
- The first complete 137.48 GB conversion was discarded wholesale.

ii.
```python
import os, glob, pickle, re, gc
```
```python
        N=np.empty((spk.shape[0],nb),np.float32)
```
```python
reward_stim={}
for s in sessions:
    b=beh[s]; st=np.asarray(b['TrialStim']); rw=np.asarray(b['RewardFr'])
    counts={x:int(np.isfinite(rw[st==x]).sum()) for x in np.unique(st)}
    reward_stim[s]=max(counts,key=counts.get) if counts and max(counts.values())>0 else None
```
```python
    if len(ns)<2: print('skip',s,'too few trials'); continue
    neural.append(ns); ...
    del spk; gc.collect()
```

iii. The agent's justification for retaining everything is faithfulness to the reference loader (step 21): "The unusually high neuron count is explained by three imaging planes concatenated as required by the paper"; step 26: "Retaining all paper-loaded neurons yields tens of thousands per session and likely a roughly 100–130 GB pickle; this is large but within available disk and memory, and it most faithfully matches `utils.load_spk`." It briefly considered dropping unassigned ROIs for tractability (steps 21–22) but rejected it (step 23). Storage dtype is never discussed.
