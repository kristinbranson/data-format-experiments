# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI restricted the dataset to the **two-context electrophysiology cohort only**: the 12 sessions (7 released animal IDs) that `Scripts/Figure 8/Figure8a_thru_c.m` loads via `loadJEB6/JEB7/EKH1/EKH3/JGR2/JGR3/JEB19_ALMVideo.m`. The session/date/probe triples are hard-coded, not globbed. Each session is one `data_structure_<anm>_<date>.mat` in `/app/data/Ephys_Behavior`, opened once with `h5py` (all 12 are MATLAB v7.3; no scipy fallback is provided), with a companion `motionEnergy_<anm>_<date>.mat` read with `scipy.io.loadmat`. Behavior, clusters, video trajectories and motion energy are all read inside a single `with h5py.File(...)` block per session. The other 33 cluster-bearing sessions in the release (13 DR-only `Ephys_Behavior` sessions plus all 22 `RandomizedDelay_Ephys_Behavior` sessions) are **not** loaded; the two behavior-only photoinactivation folders are excluded because they contain no `obj.clu`.

ii.
```python
ROOT=Path('/app/data/Ephys_Behavior')
# Exact active records from Figure 8 metadata loaders; MATLAB probe indices are 1-based.
SESSIONS=[
 ('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),
 ('JGR2','2021-11-16',1),('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),
 ('JEB19','2023-04-21',1),('JEB19','2023-04-20',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-18',1)]
```
```python
def process_session(subject,date,probe,show=False):
    t0=time.time(); p=ROOT/f'data_structure_{subject}_{date}.mat'
    with h5py.File(p,'r') as f:
        b=behavior(f); n=len(b['go'])
        ...
        neural,qualities,_=load_clusters(f,probe,b['go'],ids)
        vid=video_continuous(f,ids,b['go'])
        motion=np.stack(motion_continuous(subject,date,ids,b['go'],fts))
```

iii. From CONVERSION_NOTES Step 4/5: "Use the exact 12-session Figure 8/two-context cohort… Context is a required output and this cohort exactly matches the paper's 12-session WC/DR analysis" (paper: "In 12 sessions from six mice, animals performed the two-context task. In total, 522 units (214 well-isolated single units)"). It also notes the behavior-only folders "contain behavior/video but no `obj.clu` neural records, so they cannot be sessions in a neural-input decoder". I verified the AI's premise: the 12 chosen sessions have 22–38% autowater (WC) trials, whereas the 33 excluded cluster-bearing sessions have 0–12% (24 of them ≤2.3%, 14 exactly zero), so behavioral context is genuinely near-degenerate outside this cohort.

## 1-b. How are the data split into subjects?

i. The subject is the animal ID field of each hard-coded session tuple (equivalently the `<anm>` part of the filename). `subjects` is built by first-encounter order as sessions are processed, and `subject_idx` is the index of each session's animal into that list. Result: 7 subjects over 12 sessions (JEB6 1, JEB7 2, EKH1 1, EKH3 1, JGR2 2, JGR3 1, JEB19 4). The AI deliberately did not merge IDs to reach the paper's "six mice".

ii.
```python
for si,(sub,date,probe) in enumerate(sessions):
    n,x,y,info=process_session(sub,date,probe,a.show_processing and si<2)
    ...
    if sub not in subjects:subjects.append(sub)
    ...;subject_idx.append(subjects.index(sub))
```

iii. "Preserve released IDs; do not merge without evidence" (Step 5 mapping table). Step 4 records the 7-vs-6 discrepancy explicitly: "Seven released IDs versus six paper mice is documented as a source-identifier discrepancy; do not silently merge IDs without evidence."

## 1-c. How are the data split into sessions?

i. One session = one `data_structure_*.mat` file = one entry of `SESSIONS` = one element of `neural`/`input`/`output`/`brain_region_idx`/`subject_idx`. Each session also carries one hard-coded probe number; the per-session record in `metadata['session_info']` stores id, source/retained trial counts, neuron count, probe, quality-label histogram, discretization thresholds and wall time. A session is only accepted if it yields ≥2 trials and ≥10 neurons, otherwise the run aborts.

ii.
```python
    if len(n)<2 or n[0].shape[0]<10: raise ValueError(f'insufficient data {sub}_{date}')
    ...
    neural.append(n);inputs.append(x);outputs.append(y);infos.append(info)
    subject_idx.append(subjects.index(sub));regions.append(np.zeros(n[0].shape[0],np.int64))
```
```python
info={'id':f'{subject}_{date}','source_trials':n,'trials':len(ids),'neurons':neural.shape[1],
      'probe':probe,'quality_counts':{...},'thresholds':{...},'seconds':time.time()-t0}
```

iii. Sessions and probes were transcribed from the authors' `load<ANM>_ALMVideo.m` metadata loaders (I confirmed all 12 dates and all 12 probe numbers match the *uncommented* entries exactly, and that none of these sessions is a two-probe session). The ≥10-unit rule is the paper's: "Recording sessions were included for analysis only if they had at least 10 units."

## 1-d. How are the data split into trials?

i. A trial is one index into the per-trial `obj.bp` fields, and the go cue of that trial is `bp.ev.goCue[trial]`. The AI takes the trial count as the length of the `bp` vectors (`n=len(b['go'])`) rather than `bp.Ntrials`. Spikes carry their own 1-based trial index (`clu.trial`), converted to 0-based; camera trajectories and motion energy are indexed per trial by the same original trial number. Retained trials are the sorted original indices `ids=np.flatnonzero(valid)`, and every stream is gathered by that same index array, so no trial boundary is reconstructed and neural/video/label rows stay in register.

ii.
```python
def behavior(f):
    bp=f['obj/bp']; n=np.asarray(bp['L']).size
    get=lambda k:np.asarray(bp[k]).squeeze()
    ...
    d['go']=np.asarray(ev['goCue']).squeeze()
```
```python
ids=np.flatnonzero(valid)
...
pos={int(t):i for i,t in enumerate(trial_keep)}
for old in np.unique(tr):
    if old not in pos or old<0 or old>=len(go): continue
    z=tm[tr==old]-go[old]
    mat[pos[old]]=np.histogram(z,EDGES)[0]/DT
```

iii. Step 2 notes "All representative trial-level fields had matching lengths", and a planned sanity check was "trial-field length equality". I confirmed that for all 12 sessions every field used (`L,R,hit,miss,no,early,autowater,stim.enable,ev.goCue,trials.bp.haveEphys`) has length exactly `bp.Ntrials`, so not truncating to `Ntrials` is harmless here.

## 1-e. How are trials filtered based on quality controls?

i. A trial is kept if it has valid ephys (`trials.bp.haveEphys`), a finite go cue, is **not** an early-lick trial (`bp.early==0`), has **no photostimulation** (`bp.stim.enable==0`), and has exactly one of `hit`/`miss`/`no` set. Ignore (`no`) trials are deliberately retained because the decoder spec demands an `ignore` outcome class and a `none` lick class. 3,116 of 3,626 source trials are kept. I confirmed the only two filters that actually bite are early (389) and stim (135); `haveEphys`, finite-go and the mutual-exclusivity guard remove nothing in this cohort. No trial is dropped for missing video.

ii.
```python
valid=b['haveEphys']&np.isfinite(b['go'])&(np.asarray(b['early'])==0)&(np.asarray(b['stim'])==0)
outcome_sum=np.asarray(b['hit'])+np.asarray(b['miss'])+np.asarray(b['no'])
valid &= outcome_sum==1
ids=np.flatnonzero(valid)
```

iii. "Exclude early trials as reference curation. Retain technically valid `no` trials because decoder explicitly requires ignore outcome and none lick direction… stimulation is absent/not relevant in this ephys cohort or will be excluded if enabled" (Step 4), following the methods: "Trials in which the animal contacted the lickport before the reward ('early lick') were omitted from analyses." Missing video "does not justify dropping an otherwise valid neural trial because required outputs define explicit unavailable classes" (Step 3).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` of the single metadata-selected probe: per cluster, `trialtm` (spike time relative to that trial's start), `trial` (1-based trial index), and `quality` (manual curation label). `obj.bp.ev.goCue` supplies the alignment time. Nothing else (no `tm`, no `spkWavs`, no `site`) enters the neural array; `quality` is also kept for the metadata histogram.

ii.
```python
pref=np.asarray(f['obj/clu']).ravel()[probe-1]; g=f[pref]
if not isinstance(g,h5py.Group): raise ValueError('selected probe is empty')
for rt,ri,rq in zip(np.asarray(g['trialtm']).ravel(),np.asarray(g['trial']).ravel(),np.asarray(g['quality']).ravel()):
    q=h5str(f,rq).lower().strip()
    if q not in VALID_QUAL: continue
    tm=np.atleast_1d(h5arr(f,rt)).astype(float); tr=np.atleast_1d(h5arr(f,ri)).astype(int)-1
```

iii. Step 1 identified `alignSpikes.m` ("`trialtm_aligned = trialtm - event`") and Step 2 confirmed "Spike data: per cluster, continuous spike time (`tm`), trial index (`trial`), within-trial spike time (`trialtm`), waveform, site, and quality. No calcium-imaging data are present" — so no ΔF/F step is applicable. Probe selection follows the Figure 8 `meta.probe`.

## 2-b. How is the `neural` data processed?

i. Spikes of each surviving cluster are histogrammed into the 10 ms bin grid and divided by the bin width to give spikes/s, then smoothed along time with an **exact translation of the reference `mySmooth(x,15,'reflect')`**: a 15-sample MATLAB `gausswin` (α = 2.5), with the first `floor(15/2)=7` taps zeroed to make it causal, normalised to sum 1, applied by `conv(...,'same')` after reflect-padding the first 15 samples and then trimming them off. No normalisation, z-scoring or baseline subtraction; values are firing rates in Hz stored as `float32`. Only one probe per session, so no probe concatenation is needed.

ii.
```python
def causal_smooth(x, n=15):
    """Exact translation of reference mySmooth(x,15): reflect-pad and causal gausswin."""
    idx=np.linspace(-1.0,1.0,int(n),dtype=float)
    k=np.exp(-0.5*(2.5*idx)**2)
    k[:len(k)//2]=0.0; k/=k.sum()
    out=np.empty_like(x,dtype=np.float32)
    for i,row in enumerate(x):
        padded=np.concatenate([row[:n][::-1],row])
        y=np.convolve(padded,k,mode='same')
        out[i]=y[n:]
    return out
```
```python
mat[pos[old]]=np.histogram(z,EDGES)[0]/DT
...
raw=np.stack(mats,axis=1) # retained decoder trials, neurons, time
sm=np.stack([causal_smooth(x) for x in raw],axis=0)
```

iii. Step 10 records this as a corrected error: "Initial code interpreted 15 as Gaussian sigma in milliseconds. Review of `mySmooth.m` showed it is a 15-sample MATLAB `gausswin` with the first half zeroed, reflect padding, and same convolution. The implementation was corrected, sample/full data regenerated." `params.smooth = 15` with `params.bctype = 'reflect'` in `Figure8a_thru_c.m` and the comment "% smooth with causal gaussian kernel" support this. I re-implemented the histogram + causal smoothing independently from the raw HDF5 and reproduced session 0 / trial 10 / neuron 0 to `np.allclose`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters plus a session rule. (1) The manual `clu.quality` string is stripped and lower-cased and must be in the whitelist `{poor, fair, good, great, excellent, multi}`; everything else (garbage, gabrga, noisy, real?, unlabeled, and the typos `ood`/`mutli`) is dropped. (2) Each surviving cluster's **reference-style PSTH** — all spikes over all trials histogrammed, divided by (n_trials × dt), causally smoothed, then averaged over time — must exceed 1.0 Hz. (3) A session must retain ≥10 units. 515 units survive (27–67 per session, mean 42.9), out of 2,287 clusters on the selected probes.

ii.
```python
VALID_QUAL={'poor','fair','good','great','excellent','multi'}
...
# Reference low-FR curation uses condition 1 (all hit|miss|no trials),
# computes a trial-averaged PSTH, applies mySmooth, then averages over time.
rates=[]
for rt,ri,rq in zip(...):
    q=h5str(f,rq).lower().strip()
    if q not in VALID_QUAL: continue
    ...
    good=(tr>=0)&(tr<len(go)); aligned=tm[good]-go[tr[good]]
    psth=np.histogram(aligned,EDGES)[0].astype(np.float32)/(len(go)*DT)
    rates.append(float(np.mean(causal_smooth(psth[None,:])[0])))
use=np.asarray(rates)>1.0
return sm[:,use,:],np.asarray(qualities)[use],use
```

iii. Step 4: "Exclude `garbage`, `noisy`, null/unknown and `real?`; retain curated poor/fair/good/great/excellent/multi labels, then apply >1 Hz… Good/fair/great/excellent gives exactly 214 singles at >1 Hz, independently validating label interpretation." The 1 Hz threshold is "the directly applicable Figure 8 override (`params.lowFR = 1`) and paper rule: strictly >1 Hz" ("All units with firing rates exceeding 1 Hz were included in all other analyses"). Step 10 documents the residual 515-vs-522 gap and refuses to relax the rules to close it. I verified: re-running the reference's own blacklist semantics (`findClusters` with `quality={'all'}` drops only garbage/gabrga/noisy/real?) gives 518 units, i.e. the whitelist costs 3 units with typo'd/empty labels; and the single-unit subset reproduces the paper's 214 exactly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction: for each spike, `trialtm − goCue[trial of that spike]`, then histogram into the fixed −3.0…+2.5 s edge grid. Spikes outside the window fall outside the histogram range and are discarded; trials with no spikes stay as zeros. No interpolation, no offset, no time warping (`params.timeWarp = 0`, `params.advance_movement = 0` in the reference are both respected).

ii.
```python
tm=np.atleast_1d(h5arr(f,rt)).astype(float); tr=np.atleast_1d(h5arr(f,ri)).astype(int)-1
...
z=tm[tr==old]-go[old]
mat[pos[old]]=np.histogram(z,EDGES)[0]/DT
```

iii. Step 1: "Verified alignment rule: `obj.bp.ev.(params.alignEvent)` is indexed by each cluster spike's trial, and that event timestamp is subtracted from the spike's within-trial time. The decoder requirement therefore maps naturally to `params.alignEvent = 'goCue'`… no derived movement alignment is appropriate." Step 10 lists an independent raw-spike reconstruction that matched all 550 bins with `np.allclose` (I reproduced this myself).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins (`time_bin_size: 10.0`) over a fixed window of −3.0 to +2.5 s from the go cue = 550 bins, identical for every trial and session. Spikes are histogrammed **directly** at 10 ms, so there is no rebinning of the neural data. The video streams are resampled onto the same 550-bin grid (nearest frame), so all streams share one time axis.

ii.
```python
TMIN,TMAX,DT=-3.0,2.5,0.01
EDGES=np.arange(TMIN,TMAX+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
```
```python
'time_bin_size':10.0,'temporal_alignment_event':'go cue onset (water drop in WC)',
'off_start':TMIN,'off_end':TMAX,
```

iii. Step 4 explicitly chose between two parameter sets: "Generic defaults: −2.5..2.5 s, 5 ms; Figure 8 two-context: −3..2.5 s, 10 ms… Use Figure 8 parameters: go-cue alignment, −3 to +2.5 s, 10 ms bins, causal Gaussian smoothing parameter 15, no time warping/movement advance," on the grounds that "Context analyses are the closest match". I confirmed `Figure8a_thru_c.m` sets `params.tmin=-3; params.tmax=2.5; params.dt=1/100`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable. It is the bin-centre vector of the conversion grid defined by the AI (−2.995 … +2.495 s in 10 ms steps), which exists only because every trial is aligned to `bp.ev.goCue`. The same `(1, 550)` float32 row is emitted for every trial of every session, named `time_from_go_cue_s`.

ii.
```python
EDGES=np.arange(TMIN,TMAX+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
...
inputs.append(CENTERS[None,:].astype(np.float32))
...
'input_names':['time_from_go_cue_s']
```

iii. Step 5: "`input[0]`: Continuous signed seconds from go cue, repeated identically for every trial as shape `(1,550)`… Values −2.995 through +2.495 s", and "No leakage from video into neural/input: Video-derived quantities are outputs only. Input is solely signed aligned time as requested."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the grid: mid-points of the histogram edges, cast to float32, tiled per trial. It is left continuous (a ramp), not binarised, because the decoder spec calls it "continuous, time-varying".

ii.
```python
CENTERS=(EDGES[:-1]+EDGES[1:])/2
...
inputs.append(CENTERS[None,:].astype(np.float32))
```

iii. Implicit in Step 5's mapping table ("Bin centers → `input[0]`"); the validator's reported input range `[-3.0, 2.5]` (rounded) was checked in Step 7, and Step 10 check 3 compared "every converted input… with analytically generated 10 ms bin centers using `np.allclose`".

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural grid. Spike times are expressed relative to the trial's own go cue and histogrammed into `EDGES`; the input is `(EDGES[:-1]+EDGES[1:])/2`, so input column *k* is the centre of the exact interval whose spikes produced neural column *k*. Both are length 550 for every trial.

ii.
```python
mat[pos[old]]=np.histogram(z,EDGES)[0]/DT      # neural: counts in EDGES
...
CENTERS=(EDGES[:-1]+EDGES[1:])/2               # input: centres of the same EDGES
inputs.append(CENTERS[None,:].astype(np.float32))
```

iii. Step 5's boundary sanity check: "exactly 550 bins; centers −2.995..2.495; no spike counted twice"; Step 10: "EDGE/window checks: PASS".

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The **recorded lick events**: `obj.bp.ev.lickL` and `obj.bp.ev.lickR`, which are per-trial cell arrays of left- and right-port contact times, plus `bp.ev.goCue` to define "post go cue". The instructed side `bp.L`/`bp.R` is *not* used to assign the label (it is loaded and kept only as a cross-check).

ii.
```python
d={k:get(k) for k in ['L','R','hit','miss','no','early','autowater']}
d['lickL']=cell_arrays(f,ev['lickL']); d['lickR']=cell_arrays(f,ev['lickR'])
```

iii. Step 5 key decision 3: "**Use actual lick events**: Decoder asks for lick direction, so target side cannot stand in for emitted behavior. No post-go lick is `none`" and "`bp.L/R` is a target-direction check, not substituted for actual licking."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each retained trial, the first finite left-lick time at or after the go cue and the first finite right-lick time at or after the go cue are found (`inf` if none). Whichever is earlier gives the class: left = 0, right = 1; if neither exists (both `inf`, i.e. a tie) the class is 2 = `none`. The per-trial scalar is then broadcast across all 550 bins so the output tensor is uniform. Resulting distribution: left 0.400, right 0.377, none 0.223 (the `none` fraction equals the ignore fraction 0.225 up to 3 trials).

ii.
```python
def first_post_go(a,go):
    x=np.atleast_1d(a).astype(float); x=x[np.isfinite(x)&(x>=go)]
    return np.min(x) if x.size else np.inf
...
l=first_post_go(b['lickL'][old],b['go'][old]);r=first_post_go(b['lickR'][old],b['go'][old])
lick.append(0 if l<r else (1 if r<l else 2))
...
o=np.vstack([np.full(CENTERS.size,lick[i]),...]).astype(np.int64)
```
```python
'output_values':[['left','right','none'], ...]
```

iii. Same as 4-a; per-trial labels are "repeated across time for decoder compatibility" (Step 5), matching the spec's "per-trial" annotation while keeping one uniform `(6,550)` tensor. I checked the AI's labels against the instructed-side + hit/miss derivation on 4 sessions: 1,080/1,083 trials agree, and all 3 disagreements are `ignore` trials on which the animal did emit a late post-go lick.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The single per-trial flag `obj.bp.autowater`.

ii.
```python
d={k:get(k) for k in ['L','R','hit','miss','no','early','autowater']}
...
context.append(1 if b['autowater'][old] else 0)
```

iii. Step 4: "Figure code calls AFC/2AFC `~autowater` and AW `autowater`… Map `autowater=0` to DR and `autowater=1` to WC." Step 1 also notes "Comments explicitly equate AFC/2AFC with DR and AW with WC" (e.g. `params.condition = {'hit&~stim.enable&~autowater'}` commented "all 2AFC hits").

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabel, broadcast over time: `autowater != 0` → 1, else 0, with `output_values[1] = ['DR','WC']` so 0 = DR and 1 = WC. (This is the opposite numeric coding from the prompt's "WC, DR" listing order, but is self-consistent with the declared value names.) Distribution 0.685 DR / 0.315 WC; all 12 sessions contain both.

ii.
```python
context.append(1 if b['autowater'][old] else 0)
...
'output_values':[...,['DR','WC'],...]
```

iii. Step 12 records the verification: three raw trials (DR, WC, ignore) checked directly from the HDF5 file against the converted labels, and a per-session DR/WC count table showing "CONTEXT VARIATION EVERY SESSION: PASS".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial flags of `obj.bp`: `miss`, `hit`, and `no`. All three are read explicitly, and their mutual exclusivity is enforced as a trial filter (`hit+miss+no == 1`).

ii.
```python
d={k:get(k) for k in ['L','R','hit','miss','no','early','autowater']}
...
outcome_sum=np.asarray(b['hit'])+np.asarray(b['miss'])+np.asarray(b['no'])
valid &= outcome_sum==1
```

iii. Step 5: "`bp.miss`, `bp.hit`, `bp.no` → `output[2]` outcome… Mutually exclusive assertion; per-trial label repeated across time." Step 4 notes the paper omits ignore trials but the decoder spec requires the class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-way relabel, broadcast over time: `miss` → 0 (incorrect), else `hit` → 1 (correct), else 2 (ignore). Because the trial filter guarantees exactly one flag is set, the `if/elif` chain is unambiguous. Distribution: incorrect 0.106, correct 0.669, ignore 0.225.

ii.
```python
outcome.append(0 if b['miss'][old] else (1 if b['hit'][old] else 2))
...
'output_values':[...,['incorrect','correct','ignore'],...]
```

iii. Ordering follows the decoder spec's "(incorrect, correct, ignore)". Step 3: ignore trials are kept only because "the requested decoder… explicitly requires an `ignore` outcome".

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. `obj.traj{1}` (camera 0, the **side** view) only: `ts` (feature × [x, y, likelihood] × frame), `frameTimes`, and `featNames`. Every feature whose lower-cased name contains "tongue" is used — on the side camera that is `tongue`, `left_tongue`, `right_tongue`. The bottom camera's `top_tongue`/`bottom_tongue` are **not** used for the tongue. `bp.ev.goCue` provides the alignment.

ii.
```python
for cref in cams:
    g=f[cref]; decoded.append((g, np.asarray(g['ts']).ravel(),np.asarray(g['frameTimes']).ravel(),np.asarray(g['featNames']).ravel()))
...
def speed_for(cam,patterns):
    arr,ft,names=vals[cam]
    inds=[i for i,n in enumerate(names) if any(p in n for p in patterns)]
    if not inds:return np.full(CENTERS.size,np.nan)
...
tongue=speed_for(0,['tongue'])
```

iii. Step 5: "Side-camera tongue DLC coordinates/likelihood and frame times → `output[3]` tongue velocity… Visibility requires valid tongue coordinates/tracking; no imputation across invisible spans", referencing `getKinematics`. The reference's `params.traj_features{1}` does list `tongue, left_tongue, right_tongue` for camera 0. The AI gives no explicit rationale for excluding the bottom-camera tongue landmarks.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Per trial and per tongue landmark: (1) mark a frame visible iff x and y are finite **and** likelihood > 0.9; (2) differentiate x and y against the real frame-time vector with `np.gradient` (no positional smoothing, matching the reference's "if tongue, don't smooth"); (3) speed = `hypot(dx, dy)` in pixels/s, set to NaN on non-visible frames; (4) resample onto the 550-bin grid by nearest frame (`nearest_bin`), NaN outside the frames' own time support; (5) average the three landmarks bin-wise over whichever are finite, NaN where none is. No cross-camera or percentile normalisation is applied (single camera, so scales are already common).

ii.
```python
for j in inds:
    x,y=arr[j,0],arr[j,1]
    visible=np.isfinite(x)&np.isfinite(y)
    if arr.shape[1]>2: visible &= np.isfinite(arr[j,2])&(arr[j,2]>0.9)
    dx=np.gradient(x,ft);dy=np.gradient(y,ft); sp=np.hypot(dx,dy);sp[~visible]=np.nan
    speeds.append(nearest_bin(ft,sp))
z=np.stack(speeds); finite=np.isfinite(z); count=finite.sum(axis=0)
return np.divide(np.nansum(z,axis=0),count,out=np.full(z.shape[1],np.nan),where=count>0)
```
```python
def nearest_bin(times, values):
    """Nearest-frame resampling, unavailable outside source support or at nonfinite values."""
    ...
    ix=np.searchsorted(t,CENTERS); ix=np.clip(ix,1,len(t)-1)
    lo=ix-1; pick=np.where(np.abs(t[ix]-CENTERS)<np.abs(t[lo]-CENTERS),ix,lo)
    inside=(CENTERS>=t[0])&(CENTERS<=t[-1]); out[inside]=v[pick[inside]]
```

iii. Step 5: "Reference kinematic velocity magnitude aligned/interpolated to neural bins". Step 10: "All-NaN landmark warning: Replaced `nanmean` on entirely invisible bins with explicit finite counts; regenerated data have no conversion warnings." The 0.9 likelihood cut is the DeepLabCut visibility criterion; Step 3 notes "Visibility is based on whether the relevant landmark is labeled/available… missing visibility must remain an explicit output class rather than imputation."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session: the 50th percentile of **all finite** binned tongue-speed values pooled over every retained trial and every bin. Bins strictly below it → 0, at or above → 1, non-finite (untracked) → 2 (`not_visible`). The threshold is stored in `metadata.session_info[i]['thresholds']['tongue_velocity']`. Resulting distribution 0.038 / 0.038 / 0.923.

ii.
```python
def discretize_session(a):
    a=np.asarray(a,float); valid=np.isfinite(a); med=float(np.nanpercentile(a,50)) if valid.any() else np.nan
    y=np.full(a.shape,2,np.int64); y[valid & (a<med)]=0; y[valid & (a>=med)]=1
    return y,med
...
tv,tmed=discretize_session(tongue);pv,pmed=discretize_session(paw);mv,mmed=discretize_session(motion)
```

iii. Step 5 key decision 4: "**Median thresholds pool valid time bins within each session**: This implements 'per-session threshold' with maximal stable sample size; missing values are excluded from percentile calculation and assigned class 2." Step 4 notes this deliberately supersedes the paper's manual bimodal `moveThresh`: "Decoder specification overrides this: use per-session 50th percentile of valid aligned values."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are taken from the same camera the feature comes from and the trial's go cue is subtracted: `frameTimes − goCue[trial]`. The resulting speed trace is then nearest-frame resampled onto the shared 550-bin grid. **No video-clock offset is applied.** The reference code (`findPosition.m`, `loadMotionEnergy.m`) uses `frameTimes − vidshift − alignEvent` with `vidshift = findVideoOffset(obj) = mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`; the AI's converter never computes or subtracts that term.

ii.
```python
for ci,(g,tsrefs,ftrefs,nrefs) in enumerate(decoded):
    arr=np.asarray(f[tsrefs[old]],float) # feature, xyz, frame
    ft=np.asarray(f[ftrefs[old]],float).squeeze()-go[old]
    names=[x.lower() for x in feat_names(f,nrefs[old])]
    vals.append((arr,ft,names))
```

iii. Step 4/5 state only that "Kinematics and motion energy use native trial/video mappings and timestamps, aligned to go cue"; Step 10 check 7 and Step 12 check 6 assert alignment was verified, but the evidence cited there (`np.allclose` on reconstructed *spikes*, and the shared −3…2.5 s plot axis) tests only the neural stream. The AI never discusses the bitcode offset; `findVideoOffset` appears nowhere in its notes, code, or trajectory. I measured the offset directly: 0.49 s for the 8 non-JEB19 sessions and 0.99 s for the 4 JEB19 sessions, and in the converted file video-derived bins first become available at t = −1.975 s (and −1.475 s for JEB19) instead of ≈ −2.47 s, with peak tongue visibility at +0.63…+1.27 s instead of ≈ +0.2 s.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. `obj.traj{2}` (camera 1, the **bottom** view) only: `ts`, `frameTimes`, `featNames`. Every feature whose name contains "paw" is used — on the bottom camera that is `top_paw` **and** `bottom_paw`, which are averaged bin-wise over whichever is tracked.

ii.
```python
paw=speed_for(1,['paw'])
```
(with the same `speed_for` body quoted in 7-b, which averages `inds` over finite entries)

iii. Step 5: "Bottom-camera paw DLC coordinates/likelihood and frame times → `output[4]` paw velocity… select top/bottom paw features available in camera 2 and combine consistently with reference feature processing." The reference's `params.traj_features{2}` does include both `top_paw` and `bottom_paw`. The AI does not discuss the differing tracking reliability of the two paws.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to the tongue and by the same `speed_for` code path: likelihood > 0.9 and finite x/y for visibility, `np.gradient` of x and y against real frame times, speed = `hypot`, NaN where not visible, nearest-frame resampling to the 550-bin grid, then a bin-wise mean over whichever of the two paw features is available. No positional smoothing (the reference's `mySmooth(ts,1,...)` for non-tongue features is a no-op, since `mySmooth` returns the input unchanged when N == 1), no normalisation, no baseline-derivative subtraction. Units remain pixels/s.

ii.
```python
dx=np.gradient(x,ft);dy=np.gradient(y,ft); sp=np.hypot(dx,dy);sp[~visible]=np.nan
speeds.append(nearest_bin(ft,sp))
z=np.stack(speeds); finite=np.isfinite(z); count=finite.sum(axis=0)
return np.divide(np.nansum(z,axis=0),count,out=np.full(z.shape[1],np.nan),where=count>0)
```

iii. Same Step 5 mapping entry as 8-a; the shared implementation is justified as "Reference kinematic velocity magnitude aligned/interpolated to neural bins".

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Exactly as the tongue: per-session 50th percentile of all finite binned paw speeds pooled across trials and bins; below → 0, at-or-above → 1, non-finite → 2 (`not_visible`); threshold saved in the session metadata. Distribution 0.388 / 0.388 / 0.223.

ii.
```python
pv,pmed=discretize_session(paw)
...
'thresholds':{'tongue_velocity':tmed,'paw_velocity':pmed,'motion_energy':mmed}
```

iii. Same as 7-c (Step 5 key decision 4). Step 7 verified "Valid bins for each median-discretized stream divide approximately 50/50, confirming threshold logic."

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The bottom camera's own `frameTimes` minus the trial's go cue, then nearest-frame resampling onto the 550-bin grid. Using each camera's own frame times means differing frame counts between views are handled. As with the tongue, **the reference video-clock offset (`findVideoOffset`) is not subtracted**, so the paw stream is late by 0.49 s (0.99 s for JEB19) relative to the neural data and to the time-from-go-cue input.

ii.
```python
ft=np.asarray(f[ftrefs[old]],float).squeeze()-go[old]
...
def speed_for(cam,patterns):
    arr,ft,names=vals[cam]
```

iii. Same as 7-d: the notes claim go-cue alignment of video streams and claim alignment was checked, but the checks performed exercised only the spike stream. Confirmed empirically: paw availability in the converted file begins at t = −1.975 s (−1.475 s for JEB19) rather than ≈ −2.47 s.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone companion file `motionEnergy_<subject>_<date>.mat`, field `me.data`, which holds one trace per trial with one value per camera frame. Frame times come from `obj.traj{1}` (side camera), the camera motion energy is computed from. `obj.me` (present in only some sessions) is not used.

ii.
```python
def motion_continuous(subject,date,trial_ids,go,frame_times):
    p=ROOT/f'motionEnergy_{subject}_{date}.mat'; out=[]
    if not p.exists(): return [np.full(CENTERS.size,np.nan) for _ in trial_ids]
    me=loadmat(p,simplify_cells=True)['me']; data=np.atleast_1d(me['data'])
...
motion=np.stack(motion_continuous(subject,date,ids,b['go'],fts))
```

iii. Step 1 identified `loadMotionEnergy` as a separate loader called after `loadSessionData`; Step 2 noted "Separate motion-energy files contain session motion-energy data and movement threshold metadata" and that some sessions elsewhere in the release lack a companion file, so "missing motion energy can therefore be represented by the required `no video`/unavailable class rather than fabricating values". All 12 cohort sessions have a companion file (I confirmed all 12 use the simple `{data, moveThresh}` layout, so the single-level `me['data']` access is sufficient here).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling. The stored value is already one scalar per frame (the paper's per-pixel median-difference reduced by the 99th percentile across pixels), so the trace is simply paired with the side camera's frame times, truncated to the shorter of the two lengths, and nearest-frame resampled onto the 550-bin grid; bins outside the frames' time support become NaN. Trials whose index exceeds the length of `me.data` yield an all-NaN row.

ii.
```python
for old,ft in zip(trial_ids,frame_times):
    if old>=len(data):out.append(np.full(CENTERS.size,np.nan));continue
    v=np.asarray(data[old],float).squeeze(); t=np.asarray(ft,float).squeeze()
    n=min(len(v),len(t)); out.append(nearest_bin(t[:n],v[:n]))
```

iii. Step 3 recorded the paper's motion-energy definition ("difference of medians over next vs previous five frames… then framewise 99th pixel percentile") as already-computed upstream, and Step 5 maps it as "Align native values to go cue, aggregate/interpolate to 10 ms bins, session median split; no video/nonfinite=2."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretize_session`: per-session 50th percentile of all finite binned values pooled over trials and bins; below → 0, at-or-above → 1, non-finite → 2, whose value name here is `no_video` rather than `not_visible`. Distribution 0.385 / 0.399 / 0.216 (the small 0/1 asymmetry comes from ties at the median being assigned to class 1).

ii
```python
mv,mmed=discretize_session(motion)
...
'output_values':[...,['below_session_median','at_or_above_session_median','no_video']]
```

iii. Step 4: "Paper/reference stores manually chosen per-session movement threshold… Decoder specification overrides this: use per-session 50th percentile of valid aligned values. Preserve unavailable as class 2." The stored `me.moveThresh` is deliberately ignored.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Via the side camera's `frameTimes` for the same trial, already reduced by the go cue inside `video_continuous` and passed out as `fts`; then nearest-frame resampling onto the 550-bin grid. **The video-clock offset is again not applied**, so motion energy carries the same 0.49 s / 0.99 s lag as the tongue and paw. (The `go` argument of `motion_continuous` is accepted but never used, because the frame times handed in are already go-cue-relative.)

ii.
```python
tongue=np.stack([z[0] for z in vid]);paw=np.stack([z[1] for z in vid]);fts=[z[2] for z in vid]
motion=np.stack(motion_continuous(subject,date,ids,b['go'],fts))
...
results.append((tongue,paw,vals[0][1]))   # vals[0][1] == side-camera frameTimes - goCue
```

iii. Same as 7-d/8-d. Notably the reference `loadMotionEnergy.m` line the AI's notes cite does exactly `interp1(obj.traj{1}(trix).frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)`, i.e. it includes the offset the converter omits.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is always represented, never imputed, and never causes a trial to be dropped. Specifically: (a) frames with likelihood ≤ 0.9 or non-finite x/y → NaN speed → class 2; (b) bins outside a trial's frame-time support → NaN → class 2; (c) non-finite frame times are filtered out in `nearest_bin`, and a trial with fewer than 2 usable frames yields an all-NaN row; (d) bins where every landmark of a feature is missing use an explicit finite-count divide instead of `nanmean`, avoiding all-NaN warnings; (e) a missing `motionEnergy_*.mat` file, or a trial index beyond the end of `me.data`, yields an all-NaN row → class 2 `no_video`; (f) motion-energy traces and frame-time vectors of unequal length are truncated to the shorter; (g) an empty probe cell raises rather than being counted as zero neurons; (h) a camera whose `featNames` lack the requested feature yields an all-NaN row; (i) each camera is timed by its own `frameTimes`, so differing frame counts across views are tolerated; (j) NaN discretisation thresholds are handled (`med=np.nan` when a session has no valid values at all). Quality strings are stripped, lower-cased and NUL-trimmed before matching, absorbing the case and padding variants in the files.

ii.
```python
def h5str(f, obj):
    a=h5arr(f,obj)
    if a.dtype.kind in 'ui': return ''.join(chr(int(z)) for z in np.asarray(a).ravel(order='F')).rstrip('\x00')
```
```python
ok=np.isfinite(t); t=t[ok]; v=v[ok]
out=np.full(CENTERS.size,np.nan)
if t.size<2:return out
```
```python
if not p.exists(): return [np.full(CENTERS.size,np.nan) for _ in trial_ids]
...
if old>=len(data):out.append(np.full(CENTERS.size,np.nan));continue
n=min(len(v),len(t))
```
```python
med=float(np.nanpercentile(a,50)) if valid.any() else np.nan
y=np.full(a.shape,2,np.int64)
```

iii. Step 3: "missing visibility must remain an explicit output class rather than imputation"; Step 5: "Do not exclude for missing video; encode class 2 for affected video outputs"; Step 10 documents the all-NaN warning fix. Note that a large part of the class-2 mass is structural rather than genuinely missing data: the −3.0 s window start precedes the first camera frame of every trial (≈ −2.47 s even after correct offset correction), and the un-corrected video clock (7-d) pushes that boundary to −1.975/−1.475 s, which accounts for essentially all of the 0.223 paw and 0.216 motion-energy "unavailable" fractions.

## 11-a. What are the most time-consuming steps of the code?

i. Reading and dereferencing the HDF5 files dominates. Total full conversion is 31 s for 12 sessions (1.9–4.3 s per session), well within the 15-minute budget, and per-session timings are printed and stored in `metadata`. Profiling one session (JEB6, 3.37 s) attributes 2.59 s (77%) to `video_continuous` — almost all of it `h5py` object-reference reads (`read_direct` 1.43 s over 7,932 dataset reads), including 0.93 s inside `feat_names` decoding per-trial `featNames` strings — 0.58 s to `load_clusters`, 0.18 s to `np.gradient`, 0.20 s to `np.histogram`, and only 0.13 s to `causal_smooth`.

ii.
```python
print(f"{info['id']}: {info['trials']}/{n} trials, {info['neurons']} neurons, {info['seconds']:.1f}s",flush=True)
```
```python
def feat_names(f, ref):
    cell=f[ref]
    return [h5str(f,r) for r in np.asarray(cell).ravel()]
```

iii. Step 6: "MATLAB reference dereferencing and per-unit/per-trial spike histograms are inherently irregular. Video trials have variable frame counts and require per-trial resampling." Step 7 estimated "<1 minute for 12 sessions" from a 2.6–2.8 s/session sample rate, and the full run confirmed 31 s.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain. (1) Spike binning loops over clusters **and then over each cluster's trials** calling `np.histogram` once per (cluster, trial) — 8,907 histogram calls for one session — where a single `np.histogram2d` over a (trial × time) edge grid per cluster, or one `bincount` over a flattened index, would do the whole cluster at once; this is genuinely rectangular and vectorizable. (2) `causal_smooth` loops over rows calling `np.convolve` per neuron instead of a single 2-D convolution (`scipy.ndimage.convolve1d`/`fftconvolve` along axis 1). (3) `video_continuous` loops over trials × cameras × features; this one is irregular because each trial has a different frame count, and `nearest_bin` is already vectorized internally. The AI kept all three as loops.

ii.
```python
for old in np.unique(tr):
    if old not in pos or old<0 or old>=len(go): continue
    z=tm[tr==old]-go[old]
    mat[pos[old]]=np.histogram(z,EDGES)[0]/DT
```
```python
    for i,row in enumerate(x):
        padded=np.concatenate([row[:n][::-1],row])
        y=np.convolve(padded,k,mode='same')
        out[i]=y[n:]
```

iii. Step 6 claims the remaining loops are irreducible ("inherently irregular", "variable frame counts"), and reports the speed-ups it did add: "Session files are opened once…; Numeric arrays use float32/int64 target dtypes; Sample mode limits work to two sessions; Neural output is stacked before smoothing/filtering and trial dictionaries avoid repeated index searches." Because the total runtime is 31 s, none of these were pursued further.

## 11-c. What processing does the code repeat multiple times?

i. Two real repetitions, both contradicting the notes' claim of a single pass. (1) `load_clusters` iterates over the quality-passing clusters **twice**, re-dereferencing and re-reading each cluster's `trialtm`/`trial`/`quality` datasets and re-histogramming every spike: once to build the per-trial matrices, once to build the all-trial PSTH used for the >1 Hz filter. The quality string of every cluster is likewise decoded twice. (2) `feat_names` is called once per trial **per camera** (604 calls for a 302-trial session) even though `featNames` is identical on every trial — 0.93 s, ~28% of that session's runtime. Smaller repetitions: the `pos={int(t):i for ...}` trial-index dictionary is rebuilt inside the per-cluster loop instead of once; the feature-name pattern match in `speed_for` is redone per trial; `np.asarray(g['ts'])`/`frameTimes` reference tables are, correctly, read once per camera.

ii.
```python
    raw=np.stack(mats,axis=1) # retained decoder trials, neurons, time
    sm=np.stack([causal_smooth(x) for x in raw],axis=0)
    # Reference low-FR curation uses condition 1 (all hit|miss|no trials), ...
    rates=[]
    for rt,ri,rq in zip(np.asarray(g['trialtm']).ravel(),np.asarray(g['trial']).ravel(),np.asarray(g['quality']).ravel()):
        q=h5str(f,rq).lower().strip()
        if q not in VALID_QUAL: continue
        tm=np.atleast_1d(h5arr(f,rt)).astype(float); tr=np.atleast_1d(h5arr(f,ri)).astype(int)-1
```
```python
        mat=np.zeros((trial_keep.size,CENTERS.size),np.float32)
        pos={int(t):i for i,t in enumerate(trial_keep)}
```
```python
            names=[x.lower() for x in feat_names(f,nrefs[old])]
```

iii. Step 6 asserts the opposite: "Session files are opened once; behavior, clusters, and video are processed in one pass… trial dictionaries avoid repeated index searches." The second cluster pass was an intentional design choice for fidelity — the inline comment explains the low-FR threshold must be computed from the reference's all-trial PSTH, not from the decoder-retained trials — but it was implemented as a second full read rather than by reusing the spikes already in memory.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest amounts. (1) The per-trial rate matrices are built **and causally smoothed** for every quality-passing cluster before the >1 Hz mask is applied, so the smoothing of the sub-threshold units is thrown away (32 → 29 units in JEB6; 2,287 clusters are quality-screened first, so the waste is ~5% of units, not of all clusters). (2) `behavior()` loads `bp.L` and `trials.bp.haveVid`, neither of which is ever used (`haveVid` is not even consulted for the video outputs). (3) `motion_continuous` takes a `go` argument it never uses. (4) `load_clusters` returns the boolean `use` mask, which the caller discards (`neural,qualities,_=`). (5) `qualities` is carried through only to build a metadata histogram. (6) Under `--show-processing`, plots are rendered for two sessions. (7) `h5str`'s string branch has a dead `str(a.item() ...)` fallback for non-integer dtypes. Everything else computed per trial ends up in the output.

ii.
```python
    raw=np.stack(mats,axis=1)
    sm=np.stack([causal_smooth(x) for x in raw],axis=0)   # smoothed before the >1 Hz mask
    ...
    use=np.asarray(rates)>1.0
    return sm[:,use,:],np.asarray(qualities)[use],use
```
```python
d['haveEphys']=np.asarray(trials['haveEphys']).squeeze().astype(bool)
d['haveVid']=np.asarray(trials['haveVid']).squeeze().astype(bool)
```
```python
def motion_continuous(subject,date,trial_ids,go,frame_times):
```
```python
neural,qualities,_=load_clusters(f,probe,b['go'],ids)
```

iii. Not addressed in CONVERSION_NOTES; Step 6 discusses only the loop irregularity and the speed-ups it added. The waste is small in absolute terms (`causal_smooth` is 0.13 s of a 3.4 s session), which is consistent with the AI's decision not to optimise further once the full run came in at 31 s.
