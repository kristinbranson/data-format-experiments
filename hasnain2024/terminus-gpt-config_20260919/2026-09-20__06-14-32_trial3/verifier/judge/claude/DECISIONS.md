# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. One MATLAB file per session, `data_structure_<anm>_<date>.mat`, living in one of the two ephys folders (`Ephys_Behavior`, `RandomizedDelay_Ephys_Behavior`), with its motion energy in a sibling `motionEnergy_<anm>_<date>.mat`. The session list is **not** hard-coded and **not** globbed blindly: `parse_sessions()` reads the authors' own loader scripts in `code/DataLoadingScripts/Recording and video/*.m`, regex-parses every `meta(end+1)` block for `anm`, `date`, and `probe`, skips commented-out (`%`) lines, and then intersects that "active" list with the files actually present on disk. Sessions with no probe entry are dropped. This yields 44 sessions from 14 mice, with the probe(s) the authors specified (three JEB15 sessions use both probes). Each session file is opened exactly once, with `h5py` for MATLAB v7.3 files and `scipy.io.loadmat` for the v5 files, dispatched on `h5py.is_hdf5`. Cluster spike arrays for units labelled `garbage`/`gabrga` are never dereferenced, so the largest part of the file is skipped.

ii.
```python
def parse_sessions():
    """Parse active author loader entries, including scalar or dual-probe choices."""
    rows=[]
    p=CODE/'DataLoadingScripts'/'Recording and video'
    for f in sorted(p.glob('*.m')):
        cur={}; active=False
        for raw in f.read_text(errors='ignore').splitlines():
            line=raw.strip()
            if not line or line.startswith('%'): continue
            if re.match(r'meta\(end\+1\)',line):
                if active and {'anm','date'}<=cur.keys(): rows.append((cur['anm'],cur['date'],cur.get('probe',[])))
                cur=cur.copy(); active=True
            ...
    active={f'{a}_{d}':pr for a,d,pr in rows}
    out=[]
    for f in sorted(DATA.glob('*Ephys_Behavior/data_structure_*.mat')):
        key=f.stem.removeprefix('data_structure_')
        if key in active and active[key]: out.append((key,f,active[key]))
    return out
```

```python
holder,B,units,traj=load_h5(f,probes) if is_h5 else load_v5(f,probes)
```

iii. From CONVERSION_NOTES Step 4: "Active recording loaders select session/date/probe combinations ... 47 ephys-directory files; 44 match active loaders; JEB23 2023-10-20 is commented out and JEB24 2023-10-03/04 lack clusters ... Use the 44 active, available sessions: exactly 25 + 19." The AI notes that six JEB4/JEB5 sessions referenced by the loaders are absent from the supplied data, and that the resulting 25 fixed-delay + 19 randomized-delay split reproduces the paper's session counts exactly. Both MATLAB readers are needed because "earlier sessions are MATLAB v7.3/HDF5; 11 later randomized-delay sessions are MATLAB v5".

**Verified:** the 44 (session, probe) pairs produced by `parse_sessions()` are *identical* to the expert's hand-transcribed `SESSIONS` dict — same keys, same probe lists, no differences.

## 1-b. How are the data split into subjects?

i. The animal ID is the filename stem before the first underscore. Each session contributes one entry to an `animals` list; `subjects` is the sorted unique set and `subject_idx` is each session's index into it. 14 subjects (EKH1, EKH3, JEB6, JEB7, JEB11–15, JEB19, JEB23, JEB24, JGR2, JGR3).

ii.
```python
animals.append(key.split('_')[0])
...
subjects=sorted(set(animals)); subject_idx=np.asarray([subjects.index(a) for a in animals],dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5 mapping row: "filename animal ID → `subjects`, `subject_idx`; unique stable subject list and per-session index; 14 available recording subjects expected." The notes record that the in-file metadata is unreliable/absent in places, whereas the filename always carries the animal, and that the authors' loader scripts identify animals the same way.

## 1-c. How are the data split into sessions?

i. One `data_structure_*.mat` file = one session = one element of `neural`/`input`/`output`. The glob `DATA.glob('*Ephys_Behavior/data_structure_*.mat')` covers both the fixed-delay and randomized-delay folders in one pass, so the two task variants are treated as one pool of sessions; the task family is recorded per session in metadata (`'task_family': f.parent.name`). Sessions are ordered by the sorted glob (fixed-delay folder first, then randomized delay). A session is dropped entirely if fewer than 2 trials survive curation (never triggered) or if no unit survives curation (`RuntimeError`).

ii.
```python
for f in sorted(DATA.glob('*Ephys_Behavior/data_structure_*.mat')):
    key=f.stem.removeprefix('data_structure_')
    if key in active and active[key]: out.append((key,f,active[key]))
```
```python
n,x,y,info=convert_session(key,f,probes,args.show_processing and i<2)
if len(n)<2: print('Skipping session with <2 valid trials:',key); continue
```
```python
info={'session_id':key,'source_file':str(f),'probes':probes,...,'task_family':f.parent.name}
```

iii. Step 4/5 notes: the 44 active-and-available sessions "exactly gives 25 fixed-delay and 19 randomized-delay", matching the paper; the two task families are pooled because both are ALM recordings aligned to the same go cue, and the family is retained in `metadata['session_info']` so the split can be recovered downstream.

## 1-d. How are the data split into trials?

i. The trial count is taken as `len(bp.ev.goCue)` — one go cue per trial — and every per-trial behaviour field (`R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`, `stim.enable`) is read as a same-length vector. Stream-validity flags `obj.trials.bp.haveEphys` and `haveVid` are explicitly truncated to that length (`[:B['n']]`) because they can be stored longer. Spikes carry their own 1-based `trial` index, converted once to 0-based and range-checked; camera frames are already stored per trial in `obj.traj{view}(trial)`; motion energy is one cell per trial. So no trial boundary is ever reconstructed.

ii.
```python
B['go']=scalar('obj/bp/ev/goCue').astype(float); B['n']=len(B['go'])
B['haveE']=scalar('obj/trials/bp/haveEphys').astype(bool)[:B['n']]
B['haveV']=scalar('obj/trials/bp/haveVid').astype(bool)[:B['n']]
```
```python
tr=np.asarray(h[g['trial'][i,0]],int).ravel()-1
...
ok=(tr>=0)&(tr<n); tr=tr[ok]; al=tm[ok]-B['go'][tr]
```

iii. Step 2 notes: "`obj.trials.bp` supplies `haveEphys`, `haveVid`, `sglxFileNum`, and `vidFileNum`; these explicitly identify valid/matched streams and must be honored", and the notes flag that `haveVid` summed to one more than `haveEphys` across the raw data, which is why both are truncated and re-checked per trial.

**Verified:** across all 44 sessions `len(goCue) == bp.Ntrials == len(bp.early)` (14,972 trials total), so using `len(goCue)` is equivalent to the expert's `Ntrials` truncation here.

## 1-e. How are trials filtered based on quality controls?

i. Four conditions, ANDed into one mask before anything is computed: `haveEphys` true, `goCue` finite and > 0, not an early-lick trial (`~bp.early`), and not a photostimulation trial (`~bp.stim.enable`). Ignore (no-response) trials are deliberately **kept**, because the requested `outcome` output needs an `ignore` class. Then a second, data-driven pass drops any surviving trial whose entire curated population matrix is exactly zero — these are trailing trials in two JEB24 sessions where the behaviour kept running after the probe stopped and the native `haveEphys`/`haveVid` flags are wrong. Their indices are recorded in the session metadata. Result: 13,762 of 14,972 trials.

ii.
```python
valid=B['haveE']&np.isfinite(B['go'])&(B['go']>0)&(~B['early'])&(~B['stim'])
tids=np.flatnonzero(valid)
neural,rates=neural_arrays(B,units,tids)
# Exclude native trailing/invalid periods with no activity in any retained
# unit; stream flags are incorrect for these periods in two JEB24 files.
neural_valid=np.any(neural!=0,axis=(1,2))
invalid_zero_trials=tids[~neural_valid].tolist()
tids=tids[neural_valid]; neural=neural[neural_valid]
```

iii. Step 5 Key Decision 3: "require valid ephys mapping, finite positive go cue, and `~early`. Retain no/ignore trials because the target requires them. Exclude stimulation trials if any are present, matching canonical no-stimulation reference conditions." Step 9/10: "Sixty-one trailing trials in two JEB24 sessions had no neural samples despite incorrect native validity flags; these invalid periods were excluded. Revalidation then reported no warnings" (28 trials in JEB24 2023-10-23, 33 in JEB24 2023-11-03). Early-lick and stim exclusion follows `params.condition` in `getDefaultParams.m` and the paper's "excluding early lick ... trials, which were omitted from all analyses".

**Verified:** the raw pool contains 976 early, 187 stim, and 1 `~haveEphys` trial; the final count of 13,762 kept trials is identical to the expert's.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the probe(s) named in the authors' loader: per cluster, `quality` (curation label), `trial` (1-based trial of each spike), and `trialtm` (spike time relative to that trial's start). `obj.bp.ev.goCue` supplies the alignment time. Units from all requested probes are concatenated into one population (`for probe in probes: ... units.append(...)`), and `brain_region_idx` maps every unit to the single region `ALM`. `clu.tm`, `clu.spkWavs`, and `clu.channel` are not read. For v5 files `obj.clu` already stores only the selected probe's cluster struct array, so it is iterated directly.

ii.
```python
for probe in probes:
    g=h[h['obj/clu'][probe-1,0]]
    if not isinstance(g,h5py.Group) or 'quality' not in g: continue
    for i in range(g['quality'].shape[0]):
        q=hchars(h,g['quality'][i,0]).strip().lower()
        if q in BADQ: continue
        tr=np.asarray(h[g['trial'][i,0]],int).ravel()-1
        tm=np.asarray(h[g['trialtm'][i,0]],float).ravel()
        units.append((q,tr,tm))
```
```python
for c in np.asarray(o.clu).flat: # later v5 objects already contain selected probe
    q=str(c.quality).strip().lower()
    if q in BADQ: continue
    units.append((q,np.asarray(c.trial,int).ravel()-1,np.asarray(c.trialtm,float).ravel()))
```

iii. Step 1/5 notes: `alignSpikes.m` does `trialtm_aligned = trialtm - event` with `params.alignEvent='goCue'`, and `getDefaultParams.m` selects probe/area (`params.probe`, `params.probeArea{1}='ALM'`), so the loader's probe choice is followed exactly and "Recordings are from right and left ALM" → `brain_regions=['ALM']`.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned (2-d), histogrammed into the fixed 5 ms grid per trial, divided by the bin width to give spikes/s, and then smoothed along time with a **re-implementation of the reference `mySmooth`**: a 15-point `gausswin` with alpha 2.5, whose first `floor(15/2)=7` taps are zeroed to make it *causal*, normalised to sum 1, and applied with `conv(..., 'same')` per trial. Nothing else — no z-scoring, no baseline subtraction, no normalisation — so stored values are firing rates in Hz (`float32`, range 0–~330 Hz). Smoothing is applied row-wise within a trial, never across trial boundaries.

ii.
```python
mat=np.zeros((len(trial_ids),len(TIME)),np.float32)
for oi,tid in enumerate(trial_ids):
    vals=al[tr==tid]; mat[oi]=np.histogram(vals,EDGES)[0].astype(np.float32)/DT
# Exact reference mySmooth: gausswin(15, alpha=2.5), first half
# zeroed for a causal kernel, normalized, conv(...,'same').
N=15; alpha=2.5
nn=np.arange(N,dtype=float)-(N-1)/2
kern=np.exp(-0.5*(alpha*nn/((N-1)/2))**2)
kern[:N//2]=0; kern/=kern.sum()
mat=np.stack([np.convolve(row,kern,mode='same') for row in mat]).astype(np.float32)
```

iii. Step 5 Key Decision 5: "single-trial binned firing rates in spikes/s, float32, with the same temporal smoothing convention as reference PSTHs applied along time. No z-scoring, because reference trial data are rates and the target asks for neural activity." Step 10/12 record this as a fixed bug: "Incorrect symmetric sigma-15 smoothing was replaced with exact reference 15-point causal Gaussian-window convolution ... Initial over-smoothed conversion produced artificially higher sample accuracies; exact reference smoothing lowered them but remained robustly above chance."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) The manual curation label `clu.quality` is stripped and lower-cased and the unit is dropped if it is `garbage` or `gabrga` (the authors' typo); every other label is kept, including `multi`, `poor`, `fair`, `good`, `great`, `excellent`, and blank. (2) Mean firing rate over the −2.5→2.5 s window across all trials must be **> 0.5 Hz**, computed as total in-window spikes ÷ (n_trials × 5 s). A session with no surviving unit raises. Result: 2,498 units, 17–142 per session (mean 56.8).

ii.
```python
BADQ={'garbage','gabrga'}
...
q=hchars(h,g['quality'][i,0]).strip().lower()
if q in BADQ: continue
```
```python
ok=(tr>=0)&(tr<n); tr=tr[ok]; al=tm[ok]-B['go'][tr]
rate=np.count_nonzero((al>=TMIN)&(al<TMAX))/(n*(TMAX-TMIN))
if rate<=0.5: continue
...
if not binned: raise RuntimeError('No neurons survive curation')
```

iii. Step 4 Discrepancies table: "`quality='all'` in defaults, but curated analysis data exclude garbage; 10,931/14,297 raw clusters explicitly labeled garbage → Exclude explicit garbage/corrupted-garbage labels; retain curated multi/fair/poor/good/great/excellent labels as reference 'all units'." And on the threshold: "Executable default is >0.5 Hz ... Paper prose says >1 Hz and reports 2,496 total units (1,651 + 845) → Use >0.5 Hz because it reproduces reported aggregate within 2 units; >1 Hz yields only 2,459. Document text/code mismatch."

**Verified:** `findClusters.m` with `quality='all'` excludes `garbage`, `gabrga`, `noisy`, **and** `real?`. Across the 44 selected sessions there is exactly 1 `noisy` and 1 `real?` unit, so the AI's shorter drop list differs from the reference code by 2 units. Recomputing the label census independently gives 2,496 units at > 0.5 Hz — exactly the paper's reported 1,651 + 845 total.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction. `clu.trialtm` is already on the behaviour clock and relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so each spike's time from the go cue is `trialtm − goCue[trial]`, using the spike's own (0-based) trial index. Spikes whose trial index falls outside `[0, n)` are discarded first. No interpolation, no per-session constant, no resampling. Alignment is verified in the notes against an independent raw reload.

ii.
```python
ok=(tr>=0)&(tr<n); tr=tr[ok]; al=tm[ok]-B['go'][tr]
```
```python
'temporal_alignment_event':'go cue onset (trial-specific Bpod goCue)'
```

iii. Step 4: "Code defaults to -2.5..+2.5 s, dt=5 ms, go-cue alignment; raw spike `trialtm` is from trial start and go cue varies by trial → Subtract trial-specific go cue and bin on the common reference axis." Step 10 Check 2: an independent reload of EKH1 native HDF5 "without importing conversion code", re-aligned and re-histogrammed, matched the converted 1000-bin trace exactly (`np.allclose` True, max difference 0.0).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins (200 Hz), 1000 non-overlapping bins spanning [−2.5, +2.5) s from the go cue, identical for every trial and session. The edge grid and the bin-centre time axis are module-level constants shared by the neural, input, and all three video streams, so there is exactly one time axis in the dataset. No rebinning: spikes are histogrammed straight into the 5 ms grid (never a finer grid first), and the 400 Hz video streams are resampled onto the same grid once by linear interpolation at bin centres. `metadata['time_bin_size']` is 5.0 ms, `off_start`/`off_end` are −2.5/+2.5.

ii.
```python
DT=0.005; TMIN=-2.5; TMAX=2.5
EDGES=np.arange(TMIN,TMAX+DT/2,DT,dtype=np.float64)
TIME=((EDGES[:-1]+EDGES[1:])/2).astype(np.float32)
```
```python
assert all(a.shape[1]==len(TIME) for s in neural for a in s)
```

iii. Step 5 Key Decision 4: "use 1000 half-open 5-ms bins spanning [-2.5, 2.5) s, represented by centers from -2.4975 to 2.4975 s. This avoids a 1001-point edge/center ambiguity and matches histogram bin width." Step 3/4: `params.dt = 1/200`, `params.tmin=-2.5`, `params.tmax=2.5` from `getDefaultParams.m`; "Video rate: raw video is ~400 Hz with per-trial frame times → Interpolate derived behavior to the common 5 ms axis as reference code does."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing from the raw files — it is the conversion's own time axis. It is the vector of centres of the 1000 bins of the go-cue-aligned window, i.e. −2.4975 … +2.4975 s, and it is byte-identical for every trial of every session (only the alignment event, `bp.ev.goCue`, is raw). It is stored as the single input channel, shape `(1, 1000)`, named `time_from_go_cue_s`.

ii.
```python
TIME=((EDGES[:-1]+EDGES[1:])/2).astype(np.float32)
...
trialsI.append(TIME[None,:].copy())
...
'input_names':['time_from_go_cue_s']
```

iii. Step 5 mapping row: "Common bin centers → `input[0]`: Broadcast signed seconds from go cue across every trial (`getDefaultParams`). Continuous time-varying decoder input, shape 1 x 1000." The Decoder Task specifies this input as continuous and time-varying, so it is a ramp rather than a binary event marker.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The bin-centre vector is computed once at module level from the window constants and copied per trial (`TIME[None,:].copy()`); it is never scaled, centred, or normalised. The only shaping is the leading singleton axis so the array is `(n_input, n_timepoints)`.

ii.
```python
EDGES=np.arange(TMIN,TMAX+DT/2,DT,dtype=np.float64)
TIME=((EDGES[:-1]+EDGES[1:])/2).astype(np.float32)
...
assert all(x.shape==(1,len(TIME)) for s in inputs for x in s)
```

iii. Step 10 Check 3: "Independent input sanity (`np.allclose`): independently generated 5-ms bin centers; full input trace matched (`True`)." No processing is required because the quantity is defined by the binning itself.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. Spike times relative to the go cue are histogrammed into `EDGES`, and the input is the midpoint of those same edges, so input sample *k* labels exactly the interval that produced neural column *k*. No shift or resampling is possible between the two. An assertion enforces that every input array has the same length as the neural time axis.

ii.
```python
mat[oi]=np.histogram(vals,EDGES)[0].astype(np.float32)/DT
```
```python
TIME=((EDGES[:-1]+EDGES[1:])/2).astype(np.float32)
trialsI.append(TIME[None,:].copy())
```

iii. Step 5 Key Decision 4 (shared axis, half-open bins) and Step 10 Check 7: "verified 1000 bins with centers -2.4975 and +2.4975; half-open final edge at +2.5". The notes treat the common axis as the mechanism guaranteeing alignment between all streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial Bpod flags: `bp.R` and `bp.L` (the instructed port) together with `bp.hit` and `bp.miss` (the outcome). The lick side itself is not recorded anywhere, so it is inferred from the instructed side and whether the animal was correct. Trials that are neither hit nor miss get the third class, `none`.

ii.
```python
B={k:scalar('obj/bp/'+k).astype(bool) for k in ('R','L','hit','miss','no','early','autowater')}
...
lick=0 if (B['L'][tid] and B['hit'][tid]) or (B['R'][tid] and B['miss'][tid]) else 1 if (B['R'][tid] and B['hit'][tid]) or (B['L'][tid] and B['miss'][tid]) else 2
```

iii. Step 5 mapping row: "`bp.R/L`, `bp.hit/miss/no` → `output[0]` lick direction; right=`R&hit` or `L&miss`; left=`L&hit` or `R&miss`; none=`no`" — justified by the reference condition expressions (`'R&hit&~stim.enable&...'`) and the paper's choice decoders, which define choice from the instructed side crossed with outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A pure relabelling into three codes — `left`=0, `right`=1, `none`=2 — with `output_values[0] = ['left','right','none']`. The per-trial scalar is then broadcast across all 1000 bins so that all six outputs live in one `(6, 1000)` int8 array per trial (the format prefers time-varying outputs). Hit → instructed port; miss → the opposite port; anything else → `none`.

ii.
```python
lick=0 if (B['L'][tid] and B['hit'][tid]) or (B['R'][tid] and B['miss'][tid]) else 1 if (B['R'][tid] and B['hit'][tid]) or (B['L'][tid] and B['miss'][tid]) else 2
...
trialsO.append(np.vstack([np.full(len(TIME),lick,np.int8), ...]))
...
'output_values':[['left','right','none'], ...]
```

iii. Step 5 Key Decision 6: "lick direction, context, and outcome are per-trial scalars" (replicated over time for the required array shape). Step 10 Check 4 verified the mapping against independently reloaded raw `R/L/hit/miss/no` flags for specific hit/miss/ignore trials (`np.allclose` True) and confirmed "All per-trial target rows were verified constant over time".

**Verified:** `R` and `L` are exact complements in all 14,972 raw trials and no hit/miss trial has both flags clear, so this formulation is exactly equivalent to the expert's `right`-only version.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. A single per-trial flag, `obj.bp.autowater`, which marks trials where water was delivered from a random port with no sample tone, delay, or go cue — the water-cued (WC) block. Everything else is the delayed-response (DR) context. No other field is consulted.

ii.
```python
B={k:scalar('obj/bp/'+k).astype(bool) for k in ('R','L','hit','miss','no','early','autowater')}
...
np.full(len(TIME),int(B['autowater'][tid]),np.int8)
```

iii. Step 5 mapping row: "`bp.autowater` → `output[1]` behavioral context: 0=DR (`false`), 1=WC (`true`); default condition expressions" — the reference `params.condition` strings distinguish contexts purely by `autowater`/`~autowater`, so the flag can be read straight off the trial table.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct cast of the boolean to an integer code: DR = 0, WC = 1, with `output_values[1] = ['DR','WC']` documenting the mapping. Broadcast across all 1000 bins like the other per-trial variables. (The expert used the opposite numeric convention, WC = 0, DR = 1, following the order the prompt lists them; the AI's labels are self-consistent with its codes.) Session-level distribution: DR 0.903, WC 0.097.

ii.
```python
np.full(len(TIME),int(B['autowater'][tid]),np.int8)
...
'output_names':['lick_direction','behavioral_context','outcome',...]
'output_values':[...,['DR','WC'],...]
```

iii. Step 5 mapping row as above. The notes give no separate rationale for the 0/1 ordering; the requirement they address is that `output_values` names each code in order, which the verification output confirms ("behavioral_context: {DR (0.903), WC (0.097)}").

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Two per-trial flags, `bp.hit` and `bp.miss`. `bp.no` is loaded but not used for this output: a trial that is neither a hit nor a miss is an ignore by elimination, so `ignore` is the fall-through class.

ii.
```python
B={k:scalar('obj/bp/'+k).astype(bool) for k in ('R','L','hit','miss','no','early','autowater')}
...
outcome=1 if B['hit'][tid] else 0 if B['miss'][tid] else 2
```

iii. Step 5 mapping row: "`bp.hit/miss/no` → `output[2]` outcome: 0=incorrect/miss, 1=correct/hit, 2=ignore/no (`getOutcome`, behavior fields); per-trial categorical scalar; retain ignore trials by task requirement."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-way relabelling — `incorrect`=0 (miss), `correct`=1 (hit), `ignore`=2 (neither) — broadcast across the 1000 bins. Ignore trials are kept in the dataset rather than dropped, which is a deliberate deviation from the paper's analyses, because the Decoder Task asks for an `ignore` class. Distribution: incorrect 0.120, correct 0.749, ignore 0.131.

ii.
```python
outcome=1 if B['hit'][tid] else 0 if B['miss'][tid] else 2
...
'output_values':[...,['incorrect','correct','ignore'],...]
```

iii. Step 4: "Paper omits early and ignore → Exclude early trials. Retain ignore because requested target explicitly requires an ignore class. Do not balance/drop valid trials for the converted dataset." Step 1 also notes the reference decoders balance conditions, and that "This balancing is analysis-specific and should not discard other valid trials required for the requested outcome/lick-direction targets."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`. The AI uses **only the side camera** (`view = 0`) and only its `tongue` feature: `ts[:, :2, feat]` for x/y, `ts[:, 2, feat]` implicitly through the NaN pattern, plus that view's per-trial `frameTimes` and `featNames`. `bp.ev.goCue` and `obj.trials.bp.haveVid` are the other inputs. The bottom-camera tongue markers (`top_tongue`, `bottom_tongue`, …) are not used, so no cross-camera scale reconciliation is needed. Visibility comes from the raw coordinates being NaN.

ii.
```python
tongue=feature_speed(traj,tids,0,'tongue',B)
```
```python
def feature_speed(traj,trial_ids,view,name,B):
    trials=traj[view] if view<len(traj) else []
    for tid in trial_ids:
        z=trials[tid] if tid<len(trials) else None
        if z is None or not B['haveV'][tid]: out.append(np.full(len(TIME),np.nan,np.float32)); continue
        ft,ts,names=z
        try: fi=names.index(name)
        except ValueError: out.append(np.full(len(TIME),np.nan,np.float32)); continue
        ...
        m=min(len(ft),a.shape[0]); ft=np.asarray(ft[:m],float); xy=a[:m,:2,fi]
        # Raw DLC stores invisible coordinates as NaN (reference findPosition).
        vis=np.isfinite(xy).all(1)
```

iii. Step 5 mapping row: "DLC tongue x/y coordinates + visibility → `output[3]`: Use tracked tongue point consistently across sessions/views; visibility from finite coordinates/likelihood-derived NaNs, never infer visibility from zero speed", and Key Decision 9: "use a consistently named tongue feature and bottom-view paw feature. If multiple paw/tongue points exist, use a fixed anatomically corresponding point and speed magnitude, not separate x/y categories." Step 1 notes that the reference sets tongue NaN velocity to zero only for feature construction, so "target class `not visible` must be reconstructed from masks rather than treating zero as visible low velocity".

**Verified:** NaN x/y coincides exactly with likelihood ≤ 0.9 for every feature, so the NaN test is equivalent to the reference's explicit likelihood cut. `view 0` contains `tongue` and `view 1` contains `top_paw` in all 44 sessions, so the hard-coded view indices resolve correctly everywhere.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Five steps, all at raw frame resolution first. (1) The `ts` array is normalised to `(frames, 3, features)` regardless of MATLAB layout, and truncated to `min(len(frameTimes), n_frames)`. (2) Visibility is `isfinite(x) & isfinite(y)`. (3) Speed is `hypot(np.gradient(x), np.gradient(y))` evaluated at visible frames — **no position smoothing** and **no division by the frame interval**, so units are pixels/frame. (4) The speed trace is linearly interpolated onto the 1000 bin centres by `interp_visible`, which sorts, de-duplicates, and refuses to extrapolate (`left=np.nan, right=np.nan`). (5) Because interpolation would otherwise bridge long invisible stretches, the 0/1 visibility mask is separately interpolated onto the bin centres and thresholded at 0.5, and any bin judged invisible is forced back to NaN.

ii.
```python
vis=np.isfinite(xy).all(1)
vel=np.full(m,np.nan); dx=np.gradient(xy[:,0]); dy=np.gradient(xy[:,1]); vel[vis]=np.hypot(dx[vis],dy[vis])
rel=ft-B['go'][tid]
speed=interp_visible(rel,vel,TIME)
# Visibility interpolation must not bridge long invisible intervals: nearest raw-frame mask.
vok=np.isfinite(rel)
if vok.sum()>1:
    order=np.argsort(rel[vok]); rr=rel[vok][order]; vv=vis[vok][order].astype(float)
    vi=np.interp(TIME,rr,vv,left=0,right=0)>=0.5; speed[~vi]=np.nan
```
```python
def interp_visible(t,y,target):
    """Linear interpolation inside support; leave outside/insufficient samples NaN."""
    ...
    return np.interp(target,t,y,left=np.nan,right=np.nan) if t.size>1 else np.full(target.shape,np.nan)
```

iii. Step 5 mapping row: "derive speed magnitude from first derivative; align/interpolate to 5-ms axis". Step 1 records the reference behaviour being imitated: `findVelocity` takes `gradient` of position without dividing by dt and does not smooth tongue position (`findPosition` skips `mySmooth` for tongue features), and the reference nearest-fills only non-tongue features. Key Decision 10: "Never treat missing tongue/paw values as low velocity. Interpolate only within valid visible stretches; outside video support remains not visible."

**Verified issue:** `np.gradient` is taken over the *whole* frame vector, which still contains NaNs at invisible frames, so the central difference at the first and last frame of every visible run evaluates to NaN. Measured on EKH3 (200 trials): median visible tongue run is 8 frames and **14.8 % of visible tongue frames end up with NaN velocity** and are therefore coded `not visible`. The expert avoids this by differentiating within each contiguous run.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One threshold per session: the median of all finite binned tongue speeds pooled over every included trial and every bin of that session. Bins strictly below it get 0, bins at or above it get 1, and bins where the speed is NaN (tongue untracked, no video, or dropped by the run-edge effect) get 2. The threshold is stored in `metadata['session_info'][i]['thresholds']['tongue_velocity']` and drawn on the `--show-processing` plots. Resulting distribution: 0.047 / 0.047 / 0.907.

ii.
```python
def discretize(x,missing_class=True):
    med=float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
    y=np.full(x.shape,2 if missing_class else 0,np.int8); ok=np.isfinite(x)
    if np.isfinite(med): y[ok]=(x[ok]>=med).astype(np.int8)
    return y,med
...
td,tmed=discretize(tongue); pd,pmed=discretize(paw); md,mmed=discretize(motion)
```

iii. Step 5 Key Decision 7: "calculate one median per session using all finite/visible values across included trials and time bins. Values below median=0, >=median=1, not visible=2." Step 4 records that this overrides the paper: "Paper manually separates bimodal movement distributions ... Decoder task mandates 50th percentile → Use per-session median threshold, overriding the paper manual movement threshold exactly as requested." Step 10 Check 8 notes the visible classes come out balanced to within rounding, and that exact ties at the median explain the residual asymmetry.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The raw camera `frameTimes` of the side view have only the trial's go cue subtracted; the speed trace is then linearly interpolated onto the shared 1000-bin grid. **No video-clock correction is applied.** The reference pipeline (`findVideoOffset.m`, used by `findPosition.m` and `loadMotionEnergy.m`) subtracts a per-session `vidshift = mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)` before the go cue, because the video file starts before the behaviour clock; the AI's code omits that term.

ii.
```python
rel=ft-B['go'][tid]
speed=interp_visible(rel,vel,TIME)
```

iii. Step 4: "Time base: ... Subtract trial-specific go cue and bin on the common reference axis"; Step 5 mapping row: "align/interpolate to 5-ms axis". The AI did read `findVideoOffset.m` (trajectory step 51 states the alignment "uses frame times minus video offset and trial-specific go cue"), but the offset never reached `convert_data.py` and the final notes do not mention it. Step 10 Check 5 nonetheless asserts alignment was compared against the reference code and matched.

**Verified defect:** `vidshift` is 0.4900 s in every session checked (both v7.3 and v5 files). Omitting it puts all video-derived streams **490 ms late**. Independently recomputing the motion-energy population average for EKH1, the half-rise of movement after the go cue is at **+0.542 s without** the offset and **+0.052 s with** it — the latter is the physiologically correct reaction time. In the AI's own sample output, tongue visibility and motion energy only rise between +0.5 and +0.75 s, and no bin before t = −1.95 s has any camera frame (the first ~100 bins of every trial are spuriously `not visible`).

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The bottom camera (`view = 1`) of `obj.traj` and its `top_paw` feature only — x/y from `ts[:, :2, feat]`, that view's `frameTimes`, plus `bp.ev.goCue` and `haveVid`. `bottom_paw` is not used. Same code path as the tongue, with different view and feature name.

ii.
```python
paw=feature_speed(traj,tids,1,'top_paw',B)
```
```python
try: fi=names.index(name)
except ValueError: out.append(np.full(len(TIME),np.nan,np.float32)); continue
```

iii. Step 5 mapping row: "DLC paw x/y coordinates + visibility → `output[4]` paw velocity ... Bottom-view paw. Time-varying categorical series", and Step 3: "paws only [tracked] in bottom view". Key Decision 9 fixes a single anatomically consistent point per feature rather than averaging several.

**Verified:** `top_paw` has likelihood > 0.9 on essentially every frame (1.000 in the inspected trial) whereas `bottom_paw` is tracked in only ~19 % of frames — the same reason the expert gives for using `top_paw` alone.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to the tongue: layout normalisation, NaN-based visibility, `hypot` of `np.gradient(x)`, `np.gradient(y)` in pixels/frame with no smoothing and no dt division, linear interpolation to the 1000 bin centres with no extrapolation, and re-masking with the interpolated visibility. No normalisation and no nearest-filling — untracked paw bins stay `not visible` (0.268 of bins) rather than being filled as the reference MATLAB does for non-tongue features.

ii.
```python
vel=np.full(m,np.nan); dx=np.gradient(xy[:,0]); dy=np.gradient(xy[:,1]); vel[vis]=np.hypot(dx[vis],dy[vis])
rel=ft-B['go'][tid]
speed=interp_visible(rel,vel,TIME)
```

iii. Step 5 mapping row: "derive speed magnitude, align/interpolate, median split visible samples; class 2 where paw not visible ... same kinematics functions". Key Decision 10: "preserve categorical missingness. Never treat missing tongue/paw values as low velocity", which is why the reference's `fillmissing(...,'nearest')` for non-tongue features is deliberately not copied. The same run-edge `np.gradient` artifact as 7-b applies, but is immaterial here because the paw is tracked almost continuously (long runs, so only ~2 frames lost per run).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same rule and same function as the tongue: one per-session median over all finite binned paw speeds across included trials and bins; `< median` → 0, `>= median` → 1, NaN → 2 (`not_visible`). Threshold stored per session in metadata and plotted. Distribution: 0.366 / 0.366 / 0.268.

ii.
```python
td,tmed=discretize(tongue); pd,pmed=discretize(paw); md,mmed=discretize(motion)
...
'thresholds':{'tongue_velocity':tmed,'paw_velocity':pmed,'motion_energy':mmed}
```

iii. Step 5 Key Decision 7 (one median per session over all finite values) and Step 4 (per-session median mandated by the Decoder Task in place of the paper's hand-chosen movement threshold). Step 10 Check 8 confirms the two visible classes are balanced within rounding.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. `frameTimes` of the bottom view minus the trial's go cue, then interpolation onto the shared grid — the same code as the tongue, and the same omission of the per-session video-clock offset. Because each feature is timed by its own camera's `frameTimes`, a frame-count difference between the two views cannot misalign the paw.

ii.
```python
rel=ft-B['go'][tid]
speed=interp_visible(rel,vel,TIME)
```
```python
m=min(len(ft),a.shape[0]); ft=np.asarray(ft[:m],float); xy=a[:m,:2,fi]
```

iii. As for 7-d: Step 4/5 describe alignment as frame times minus the trial go cue interpolated onto the 5 ms axis, with no mention of `vidshift`.

**Verified defect:** the same 0.49 s lag. In the sample output, paw visibility is exactly 0 for all bins before t ≈ −1.95 s and 0.87 immediately after, i.e. ~10 % of every trial's bins are spuriously `not visible` purely because the un-shifted frame times do not reach the start of the window. This also inflates the paw `not_visible` fraction to 0.268 versus the expert's 0.186.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The companion file `motionEnergy_<anm>_<date>.mat` beside each data structure, loaded with `scipy.io.loadmat`; `me.data` is a cell array with one trace per trial, one value per camera frame. Legacy nesting (`me.data.data`) is unwrapped in a `while` loop. The frame times come from the **side** camera (`traj[0][tid][0]`), and `haveVid` gates each trial. `obj.me`, where present, is not used. If the companion file is absent the whole session is NaN → class 2 (`no_video`).

ii.
```python
def motion_arrays(f,traj,B,trial_ids):
    mf=f.with_name(f.name.replace('data_structure_','motionEnergy_'))
    if not mf.exists(): return np.full((len(trial_ids),len(TIME)),np.nan,np.float32),False
    me=loadmat(mf,squeeze_me=True,struct_as_record=False)['me']
    raw=me.data
    # Reference loadMotionEnergy unwraps legacy struct-valued me.data.
    while hasattr(raw,'data'):
        raw=raw.data
    vals=np.asarray(raw).flat
```

iii. Step 1: "`loadMotionEnergy` loads companion motion-energy files, interpolates video values to neural/object time relative to go cue, nearest-fills gaps, retains missing sessions as NaN." Step 2: "All 45 neural-bearing sessions have motion energy". Step 10 Issues: "Nested motion-energy struct: unwrapped legacy `me.data.data`, matching reference loader."

**Verified:** all 44 selected companion files expose `me.data`, so the attribute-based unwrap succeeds everywhere.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Almost none — the file already stores one scalar per frame (the paper's per-pixel median difference reduced by a 99th percentile across pixels). The trace is truncated to `min(len(frameTimes), len(values))`, linearly interpolated onto the 1000 bin centres with no extrapolation, and then `nearest_fill` replaces any remaining NaN by nearest-neighbour interpolation along time, mirroring the reference's `fillmissing(me.data,'nearest')`. So class 2 is reached only when the trial has no usable video at all; the observed `no_video` fraction is 0.00007 (one trial).

ii.
```python
ft=traj[0][tid][0]; y=np.asarray(vals[tid],float).ravel(); m=min(len(ft),len(y)); rel=np.asarray(ft[:m])-B['go'][tid]
z=interp_visible(rel,y[:m],TIME); z=nearest_fill(z).astype(np.float32); out.append(z)
```
```python
def nearest_fill(x):
    x=np.asarray(x,float).copy(); good=np.isfinite(x)
    if not good.any(): return x
    ix=np.arange(x.size); x[~good]=np.interp(ix[~good],ix[good],x[good]); return x
```

iii. Step 5 mapping row: "companion `me.data` + frame times → `output[5]` motion energy: align to go cue and interpolate to 5-ms axis; per-session median split; class 2 only if session has no video/motion stream (`loadMotionEnergy`). Do not encode ordinary interpolation gaps as no-video." This matches the Decoder Task, which labels class 2 for motion energy specifically as "no video" rather than "not visible".

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same `discretize` call: one per-session median over all finite binned motion-energy values, `< median` → 0, `>= median` → 1, NaN → 2 (`no_video`). Distribution 0.499 / 0.501 / 0.00007, i.e. essentially a clean median split because the nearest-fill leaves almost no missing bins.

ii.
```python
md,mmed=discretize(motion)
...
'output_values':[...,['below_session_median','at_or_above_session_median','no_video']]
```

iii. Step 5 Key Decision 8: "calculate one median per session over finite aligned motion-energy values. Values below median=0, >=median=1; class 2 is reserved for sessions/trials with no video stream as specified." Step 4 again notes this replaces the paper's manual per-session bimodal threshold because the Decoder Task mandates the 50th percentile.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy has one value per side-camera frame, so it is timed by `traj[0][tid].frameTimes` minus the trial's go cue and interpolated onto the shared 1000-bin grid — the same route, and the same missing video-clock offset, as the tongue and paw. A trial is skipped (all NaN) if the side-view trajectory entry is missing or `haveVid` is false.

ii.
```python
if tid>=len(vals) or not B['haveV'][tid] or not traj or tid>=len(traj[0]) or traj[0][tid] is None:
    out.append(np.full(len(TIME),np.nan,np.float32)); continue
ft=traj[0][tid][0]; ... rel=np.asarray(ft[:m])-B['go'][tid]
z=interp_visible(rel,y[:m],TIME); z=nearest_fill(z).astype(np.float32)
```

iii. Step 1 documents that `loadMotionEnergy.m` interpolates "video values to neural/object time relative to go cue" and nearest-fills; the AI reproduces the interpolation and the fill but not the `-vidshift` term in the same MATLAB expression (`interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix), ...)`).

**Verified defect:** the 0.49 s lag applies here too, and `nearest_fill` then hides it — the first ~100 bins of every trial are filled by copying the earliest real frame value, so motion energy shows as 100 % "available" (class 0/1) even though no frame exists in that interval.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missingness is represented, not invented, in six places. (1) Heterogeneous MATLAB formats: `h5py.is_hdf5` dispatch, plus `ts` layout normalisation that accepts `(frames,3,feat)`, `(3,·,·)`, or `(·,·,3)` and yields NaN if none matches. (2) Absent `bp.stim` struct → all-false stim mask rather than a crash. (3) Per-trial trajectory reads are wrapped in `try/except` → `None` → an all-NaN trial → `not visible`. (4) Stream flags `haveEphys`/`haveVid` are honoured per trial and truncated to the trial count. (5) Unreliable flags are backstopped by data: trials whose whole curated population matrix is zero are dropped and their indices logged. (6) Velocity is never extrapolated beyond the frame support and never bridged across invisible gaps; motion energy alone is nearest-filled, following the reference loader. Spikes with out-of-range trial indices are discarded, arrays are truncated to `min(len(frameTimes), n_frames)`, `nanmedian` is guarded by `np.isfinite(x).any()`, and four shape/dtype assertions run before the pickle is written.

ii.
```python
if 'stim' in bp and isinstance(bp['stim'],h5py.Group) and 'enable' in bp['stim']:
    B['stim']=np.asarray(bp['stim/enable']).ravel().astype(bool)
else: B['stim']=np.zeros_like(B['hit'])
```
```python
try: ... trials.append((ft,ts,names))
except Exception: trials.append(None)
```
```python
neural_valid=np.any(neural!=0,axis=(1,2))
invalid_zero_trials=tids[~neural_valid].tolist()
```
```python
med=float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
```
```python
assert all(y.shape==(6,len(TIME)) and np.issubdtype(y.dtype,np.integer) for s in outputs for y in s)
```

iii. Key Decision 10: "preserve categorical missingness. Never treat missing tongue/paw values as low velocity. Interpolate only within valid visible stretches/reference behavior; outside video support remains not visible (tongue/paw) or no-video where applicable." Step 10 Issues: "Invalid trailing neural periods: excluded 28 trials from JEB24 2023-10-23 and 33 from JEB24 2023-11-03 after native clusters ended; repeated all checks and eliminated validator warnings." Step 12 explains the one retained sklearn warning as a rare class missing from a held-out fold rather than a data defect.

## 11-a. What are the most time-consuming steps of the code?

i. File reading, then spike binning. Per-session timing is printed for every session; the full 44-session conversion took ≈143 s in total (1.5–5.6 s per session). Profiling the two representative sessions gives: load 3.10 s / neural 1.18 s / video 0.11 s / motion energy 0.05 s (EKH3, v7.3) and load 1.38 s / 0.61 s / 0.09 s / 0.04 s (JEB24, v5) — so ~70 % loading and ~25 % spike binning and smoothing. Within loading, dereferencing the DeepLabCut trajectories dominates: 2.73 s of EKH3's 3.10 s load, versus 0.21 s for the spike clusters.

ii.
```python
print(f"{key}: trials {len(tids)}/{B['n']}, neurons {len(rates)}, {time.time()-t0:.2f}s",flush=True)
```
```python
for probe in probes:
    ...
    q=hchars(h,g['quality'][i,0]).strip().lower()
    if q in BADQ: continue      # garbage units' spike arrays are never read
```

iii. Step 6: "Code inefficiencies identified: Raw trajectory data are large and per-trial MATLAB references require iteration. Single-trial spike histograms require unit/trial grouping." Step 7: "Conversion 2.9–4.8 s for first two sessions → approximately 3–6 min" full-run estimate, comfortably under the 15-minute budget, so no further optimisation was pursued.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three. (1) The inner trial loop in `neural_arrays` calls `np.histogram` once per (unit, trial) — roughly 2,500 units × ~300 trials ≈ 750k calls over the dataset — where a single `np.histogram2d` over (trial, aligned time) per unit would do the same work in one pass (this is exactly what the expert's `spike_count` does). (2) The per-unit smoothing loops `np.convolve` over rows one at a time and rebuilds the 15-tap kernel inside the unit loop; `scipy.ndimage.convolve1d(mat, kern, axis=1)` would handle the whole 2-D block. (3) `feature_speed` and `motion_arrays` loop over trials, which is unavoidable because each trial has a different number of camera frames. The final per-trial assembly loop is list construction, not arithmetic.

ii.
```python
for oi,tid in enumerate(trial_ids):
    vals=al[tr==tid]; mat[oi]=np.histogram(vals,EDGES)[0].astype(np.float32)/DT
```
```python
N=15; alpha=2.5
nn=np.arange(N,dtype=float)-(N-1)/2
kern=np.exp(-0.5*(alpha*nn/((N-1)/2))**2)
kern[:N//2]=0; kern/=kern.sum()
mat=np.stack([np.convolve(row,kern,mode='same') for row in mat]).astype(np.float32)
```

iii. Step 6 lists "vectorized alignment" among the speedups and identifies "single-trial spike histograms" as an inefficiency, but the notes argue no further work was needed because the projected full runtime was already minutes, not the 15-minute threshold. The alignment `al=tm[ok]-B['go'][tr]` and the boolean trial masks are indeed vectorised; only the per-trial histogram call is not.

## 11-c. What processing does the code repeat multiple times?

i. Four repetitions, all cheap individually. (1) The 15-tap smoothing kernel is recomputed for every unit (~2,500 times) although it is a constant. (2) `featNames` are decoded character-by-character through `hchars` for **every trial and every view** during loading (2 × n_trials × 7–10 strings per session) even though the names are constant within a view. (3) `interp_visible` re-sorts and de-duplicates the frame-time vector on every call, so each trial's frame times are sorted three times (tongue, paw, motion energy). (4) Neural binning and smoothing are performed for all 13,823 flag-valid trials and the 61 all-zero trials are discarded afterwards. The video offset is not recomputed because it is never computed at all; the bin grid is built once at module level and shared by every stream.

ii.
```python
for i in range(g['quality'].shape[0]):
    ...
    N=15; alpha=2.5           # rebuilt per unit
    kern=np.exp(-0.5*(alpha*nn/((N-1)/2))**2)
```
```python
for j in range(B['n']):
    ...
    names=[hchars(h,r) for r in np.asarray(fn).flat]   # same names every trial
```
```python
t=t[ok]; y=y[ok]; order=np.argsort(t); t=t[order]; y=y[order]
u=np.r_[True,np.diff(t)>0]; t=t[u]; y=y[u]
```

iii. The notes do not enumerate these; Step 6 claims only "Lazy HDF5 dereferencing, float32/int8 storage, vectorized alignment, and processing one session at a time" as the efficiency measures, and Step 7 concludes the runtime was acceptable without further profiling of repeated work.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly in loading. `load_h5` walks **both** camera views and **every trial**, materialising the full `ts` array (all 7 features on the side view, all 10 on the bottom view, x/y/likelihood, all frames), its `frameTimes`, and its decoded `featNames` — but only `tongue` from view 0 and `top_paw` from view 1 are ever used, i.e. 2 of 17 tracked features. Measured on EKH3 this is 2.73 s of a 3.10 s load, the single largest cost in the whole conversion. `load_v5` calls `loadmat` on the entire `obj`, including `clu.tm` and `clu.spkWavs`. Smaller items: `bp.no` and `bp.L` are loaded although `hit`/`miss`/`R` suffice; neural rates are computed for the 61 trials later dropped as all-zero; the per-unit mean rates are computed for thresholding and also stored in `metadata` (used only for documentation); and the `--show-processing` plots recompute nothing but retain the full raw speed traces.

ii.
```python
for ref in h['obj/traj'][:,0]:
    g=h[ref]; trials=[]
    for j in range(B['n']):
        ts=np.asarray(h[g['ts'][j,0]],float).transpose(2,1,0) # frames, xyz, features
        ft=np.asarray(h[g['frameTimes'][j,0]],float).ravel()
        fn=h[g['featNames'][j,0]]
        names=[hchars(h,r) for r in np.asarray(fn).flat]
        trials.append((ft,ts,names))
```
```python
o=loadmat(f,squeeze_me=True,struct_as_record=False,variable_names=['obj'])['obj']
```

iii. Step 6 identifies "Raw trajectory data are large and per-trial MATLAB references require iteration" as a known inefficiency and lists "Lazy HDF5 dereferencing" as a mitigation. That claim holds for the spike clusters — garbage-labelled units are skipped before their spike arrays are dereferenced, avoiding ~7,800 of 10,300 clusters — but not for the trajectories, which are read in full for every feature and both views before any feature selection happens.
