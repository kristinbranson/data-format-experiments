# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not glob the data directory. It hard-codes a list of 12 `(animal, date, probe)` tuples taken from the authors' Figure 8 loading script (`Scripts/Figure 8/Figure8a_thru_c.m`, which calls `loadJEB6/JEB7/EKH1/EKH3/JGR2/JGR3/JEB19_ALMVideo.m`), and only reads `/app/data/Ephys_Behavior/data_structure_<anm>_<date>.mat`. Each file is read once, whole, with `mat73.loadmat(...)['obj']` (all files in that folder are MATLAB v7.3). The `RandomizedDelay_Ephys_Behavior` folder (22 sessions) and the 13 remaining `Ephys_Behavior` sessions (JEB13, JEB14, JEB15) are never opened, and the stand-alone `motionEnergy_<anm>_<date>.mat` files that sit beside every data structure are never opened either — motion energy is taken only from the embedded `obj.me` field. Everything downstream (trials, spikes, video) is pulled out of the single loaded `obj`. Result: 12 sessions, 7 mice, 3,116 trials, 516 units.

ii.
```python
DATA='/app/data/Ephys_Behavior'
SESS=[
 ('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),
 ('JGR2','2021-11-16',1),('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),
 ('JEB19','2023-04-21',1),('JEB19','2023-04-20',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-18',1)]
...
def process(path,probe):
    print('Loading',os.path.basename(path),flush=True)
    o=mat73.loadmat(path)['obj']; b=o['bp']; n=int(float(np.asarray(b['Ntrials'])))
...
for anm,date,probe in SESS:
    p=f'{DATA}/data_structure_{anm}_{date}.mat'
    neu,inp,out,thr=process(p,probe)
```

iii. From the trajectory: the AI reasoned early (step 4) that "The requested WC/DR context output restricts conversion to the two-context electrophysiology sessions (reported as 12 sessions from six mice, 522 total units before the >1 Hz filter), rather than randomized-delay or DR-only sessions", and then spent several steps locating the authoritative cohort, concluding (step 23-24) that "The authoritative Figure 8 cohort loads seven named animal scripts (JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19) … This is preferable to inferring context from trial labels" and "The exact expert cohort is now established: 12 sessions from seven mice, matching the paper's 12 two-context sessions (the methods' 'six mice' appears inconsistent with the released Figure 8 script)." It verified its pre-filter unit count (526) against the paper's 522. It used `mat73` after finding that `scipy.io` cannot read v7.3 files and that raw `h5py` reference-chasing was awkward (steps 6-9), and processes sessions one at a time "to keep memory manageable" (step 23).

## 1-b. How are the data split into subjects?

i. The animal id is the first element of each hard-coded session tuple; it is never read from the file. `subjects` is built in first-encounter order as sessions are processed, and `subject_idx` is the index of that session's animal into `subjects`. This yields 7 subjects (`['JEB6','JEB7','EKH1','EKH3','JGR2','JGR3','JEB19']`) for 12 sessions.

ii.
```python
for anm,date,probe in SESS:
    ...
    if anm not in D['subjects']:D['subjects'].append(anm)
    D['subject_idx'].append(D['subjects'].index(anm))
...
D['subject_idx']=np.asarray(D['subject_idx'],dtype=np.int16)
```

iii. Not discussed explicitly in the trajectory beyond the observation that the loaders enumerate "animal/date/probe tuples" (step 23), which the AI parsed programmatically from the MATLAB loaders and then transcribed into `SESS`. The animal name is part of the filename and of the authors' `meta.anm`, so no in-file lookup is needed.

## 1-c. How are the data split into sessions?

i. One `(anm, date)` pair = one file = one session = one element of `neural`/`input`/`output`/`brain_region_idx`/`subject_idx`, in the order of the hard-coded `SESS` list. A session is dropped only if it would yield fewer than two trials or zero units (a guard that never fires here). All 12 sessions have a single selected probe, so no probe concatenation is performed.

ii.
```python
for anm,date,probe in SESS:
    p=f'{DATA}/data_structure_{anm}_{date}.mat'
    neu,inp,out,thr=process(p,probe)
    if len(neu)<2 or not neu or neu[0].shape[0]==0:
        warnings.warn('dropping unusable session '+p); continue
    ...
    D['neural'].append(neu); D['input'].append(inp); D['output'].append(out)
    D['metadata']['session_info'].append({'subject':anm,'date':date,'probe':probe, ...})
```

iii. Same justification as 1-a: sessions are those the released Figure 8 script loads, with the probe each loader specifies. The AI explicitly rejected discovering two-context sessions by inspecting trial labels in favour of the authors' enumeration (step 23).

## 1-d. How are the data split into trials?

i. Trials are the rows of the Bpod table `obj.bp`: `bp.Ntrials` trials, with one entry per trial in `bp.ev.goCue`, `bp.stim.enable`, `bp.early`, `bp.L`, `bp.R`, `bp.autowater`, `bp.hit`, `bp.miss`, `bp.no`. Spikes carry their own 1-based trial index (`clu.trial`), so a trial's spikes are selected by `tri==t`; video is indexed per trial through `traj[cam]['ts'][i]`, `['frameTimes'][i]`, `['featNames'][i]` and `me['data'][i]`. No trial boundaries are reconstructed.

ii.
```python
o=mat73.loadmat(path)['obj']; b=o['bp']; n=int(float(np.asarray(b['Ntrials'])))
go=arr(b['ev']['goCue']); stim=arr(b['stim']['enable']); early=arr(b['early'])
...
st=arr(clu['trialtm'][u]); tri=arr(clu['trial'][u]).astype(int)-1
for ii,t in enumerate(valid):
    z=st[tri==t]-go[t]
...
def trial_item(x,i):
    if isinstance(x,list): return x[i]
    a=np.asarray(x, dtype=object)
    return a.reshape(-1)[i]
```

iii. Step 16: "The representative context session has all needed trial labels: direction (`L`,`R`), context (`autowater`), outcomes (`hit`,`miss`,`no`), early-lick flag, stimulation flag, and go-cue event. Trials have variable delays but go-cue alignment removes that variability." Step 21: clusters "contain per-unit quality and spike timestamps plus 1-based trial assignments".

## 1-e. How are trials filtered based on quality controls?

i. A trial is kept if (a) its go cue is finite, (b) `bp.stim.enable == 0` (no photostimulation) and (c) `bp.early == 0` (no early lick). Outcome is *not* filtered: hit, miss and ignore trials are all retained, deliberately departing from the paper's figure conditions, because outcome and lick direction are requested decoder targets. 3,116 of 3,726 trials survive. No recording-length / trailing-trial filter is applied, and no minimum-units-per-session criterion is applied (the paper's ≥10-unit rule is satisfied anyway by all 12 sessions).

ii.
```python
go=arr(b['ev']['goCue']); stim=arr(b['stim']['enable']); early=arr(b['early'])
valid=np.flatnonzero(np.isfinite(go)&(stim==0)&(early==0))
```
and in the metadata:
```python
'trial_filter':'no photostimulation, no early lick; includes correct, incorrect and ignore trials',
```

iii. Step 15: "the requested outputs include incorrect and ignore trials, so conversion must retain all valid non-stimulation trials rather than the paper figure's correct-only subsets." The conditions in `getDefaultParams.m`/`Figure8a_thru_c.m` all carry `~stim.enable&~early`, and the methods state early-lick and ignore trials "were omitted from all analyses"; the AI keeps the first two exclusions and overrides the ignore exclusion because `outcome` must contain an `ignore` class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the one probe the authors' loader selects for that session: `clu.trialtm` (spike time relative to trial start), `clu.trial` (1-based trial of each spike) and `clu.quality` (manual curation label). `obj.bp.ev.goCue` supplies the alignment time. Nothing precomputed (`obj.trialdat`/`psth`) is used — those do not exist in the released objects.

ii.
```python
clu=o['clu'][probe-1] if isinstance(o['clu'],list) else o['clu']
quals=clu['quality']
...
st=arr(clu['trialtm'][u]); tri=arr(clu['trial'][u]).astype(int)-1
for ii,t in enumerate(valid):
    z=st[tri==t]-go[t]
```

iii. Step 19: "Neural processing is confirmed in `getSeq.m`: trial spike histograms are converted to Hz (`N / dt`) and smoothed with the repository's `mySmooth`". Step 21: "Cluster schema is straightforward: each probe contains per-unit quality and spike timestamps plus 1-based trial assignments." The comment in the code notes that "selected MATLAB probe numbers are 1-based", hence `probe-1`.

## 2-b. How is the `neural` data processed?

i. For each kept unit and each kept trial: spikes are histogrammed into 5 ms bins spanning the window, divided by the bin width to give Hz, and smoothed along time with the paper's *causal* Gaussian kernel — a reimplementation of `utils/mySmooth.m` with `params.smooth = 15`: `gausswin(15)` with the first `floor(15/2)=7` taps zeroed, renormalised to sum 1, convolved `'same'`. No boundary handling is applied (`mySmooth`'s `'reflect'` padding, used in the Figure 8 script, is omitted). No normalisation, baseline subtraction or z-scoring; stored as `float32` Hz.

ii.
```python
def matlab_smooth(x):
    # mySmooth: gausswin(15), first floor(15/2) entries zero, normalize,
    # conv(...,'same'). scipy convolve1d needs the equivalent correlation origin.
    k=gaussian(15,std=(15-1)/6, sym=True) # MATLAB gausswin default alpha=2.5: corrected below
    # Exact MATLAB gausswin(N): exp(-.5*(2.5*n/((N-1)/2))^2)
    n=np.arange(15)-(14/2); k=np.exp(-.5*(2.5*n/(14/2))**2)
    k[:7]=0; k/=k.sum()
    return np.stack([np.convolve(row,k,'same') for row in x],axis=0)
...
edges=np.r_[TIME-DT/2,TIME[-1]+DT/2]
neural=np.zeros((len(valid),len(keep0),TIME.size),np.float32)
for jj,u in enumerate(keep0):
    st=arr(clu['trialtm'][u]); tri=arr(clu['trial'][u]).astype(int)-1
    for ii,t in enumerate(valid):
        z=st[tri==t]-go[t]
        neural[ii,jj]=matlab_smooth((np.histogram(z,edges)[0]/DT)[None,:])[0]
```

iii. Step 20: "The smoothing implementation is now known exactly: a length-15 Gaussian window, its first half zeroed and renormalized, convolved with `same` and no boundary padding." Step 19 established the `N/dt` → Hz conversion from `getSeq.m`. The AI took the smoothing from `getSeq`'s single-trial branch (`obj.trialdat`), i.e. per trial and per unit, which is what the decoder needs (single-trial rates, not condition PSTHs).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two stages. (1) Curation label: a cluster is kept unless its lower-cased `quality` string is one of `garbage`, `noise`, `nan`, `none`, `''` — the AI's reading of `findClusters.m` with `params.quality = {'all'}`, i.e. "all non-garbage clusters", keeping single units and multiunits alike. (2) Rate: units whose mean smoothed rate over all kept trials and all bins is not `> 1 Hz` are dropped. 530 → 516 units over the 12 sessions (the AI's own printout reports 526 curated → 516). Sessions are not dropped on unit count.

ii.
```python
quals=clu['quality']; keep0=[]
for u,q in enumerate(quals):
    q=str(q).lower()
    # findClusters quality='all' means all non-garbage clusters.
    if q not in ('garbage','noise','nan','none',''): keep0.append(u)
...
# Paper's general-analysis criterion: average firing rate strictly >1 Hz.
unitkeep=np.nanmean(neural,axis=(0,2))>1
neural=neural[:,unitkeep,:]
```
metadata: `'neural_filter':'released Figure 8 selected probe; non-garbage curated units with mean firing rate >1 Hz'`.

iii. Step 24: "one small but important detail remains: whether `quality={'all'}` includes clusters labeled garbage. The reported 522 units suggests `findClusters.m` applies a meaningful exclusion" — the AI then read `findClusters.m` and implemented a non-garbage rule. Step 4 and step 15: the methods say "All units with firing rates exceeding 1 Hz were included in all other analyses", which the AI used to override the loader default `params.lowFR = 0.5` (Figure 8's own script also sets `params.lowFR = 1`). Step 26: it checked its 526 pre-filter units against the paper's 522 as a sanity check.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. By subtracting the trial's go cue from the spike times of that trial: `trialtm` is already on the Bpod clock relative to trial start and `bp.ev.goCue` is on the same clock, so one subtraction places every spike in seconds from go-cue onset. In WC (autowater) trials `bp.ev.goCue` is the water-delivery time, which the metadata records. No interpolation or resampling.

ii.
```python
z=st[tri==t]-go[t]
neural[ii,jj]=matlab_smooth((np.histogram(z,edges)[0]/DT)[None,:])[0]
```
```python
'temporal_alignment_event':'go cue onset (water delivery in WC trials)',
```

iii. Step 6: "Repository code confirms spikes are aligned by subtracting the configured event" (`alignSpikes.m`: `trialtm_aligned = trialtm - event`), and `params.alignEvent = 'goCue'` in `getDefaultParams.m` and in the Figure 8 script, matching the decoder task's requested alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins (`params.dt = 1/200`), window −2.5 s to +2.5 s (`params.tmin/tmax`), giving **1001** bins per trial, not 1000: the AI builds the time axis as the MATLAB colon vector `-2.5:0.005:2.5` (1001 points, treated as bin *centres*) and derives histogram edges as those centres ±2.5 ms, so the window actually spans [−2.5025, 2.5025]. Spikes are binned once at this resolution; there is no rebinning or downsampling afterwards. All other streams (input, tongue, paw, motion energy) are placed on this same 1001-point grid, so every trial in every session has T = 1001. `metadata['time_bin_size'] = 5.0` ms.

ii.
```python
DT=.005; TMIN=-2.5; TMAX=2.5
# MATLAB colon -2.5:0.005:2.5 gives 1001 sample centers. getSeq histogram
# uses edges centered on these samples; output therefore has 1001 points.
TIME=np.arange(TMIN,TMAX+DT/2,DT)
...
edges=np.r_[TIME-DT/2,TIME[-1]+DT/2]
```

iii. Step 15/24: "Authoritative defaults are go-cue alignment, -2.5 to +2.5 s, 5 ms bins, causal Gaussian smoothing (`smooth=15`)"; the AI inspected `getSeq.m`'s bin edges (step 24) and concluded — incorrectly, since `getSeq` does `edges = tmin:dt:tmax; obj.time = edges + dt/2; obj.time = obj.time(1:end-1)`, i.e. 1000 bins — that the reference output has 1001 points.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable: it is the analysis time axis itself, defined by the conversion as the bin centres of the go-cue-aligned window. Implicitly it is derived from `bp.ev.goCue`, which defines t = 0 for every trial. The same 1001-sample vector is stored for every trial of every session, as a `(1, 1001)` float32 array, named `time from go cue onset`.

ii.
```python
TIME=np.arange(TMIN,TMAX+DT/2,DT)
...
for t in valid:
    inp.append(TIME[None,:].astype(np.float32))
...
'input_names':['time from go cue onset'],
```

iii. Not discussed as a decision in the trajectory; the decoder task specifies "Time from go cue onset in seconds (continuous, time-varying)" as the single input, and the window/bin size come from the paper's parameters (step 15).

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the vector and casting to float32. Values run from −2.5 to +2.5 in 5 ms steps, identically on every trial.

ii.
```python
inp.append(TIME[None,:].astype(np.float32))
```

iii. N/A — the quantity is defined, not measured.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the very grid used to bin the spikes: `edges` are built from `TIME`, so input sample *k* is the centre of neural bin *k* by construction. Perfect alignment, no interpolation.

ii.
```python
TIME=np.arange(TMIN,TMAX+DT/2,DT)
edges=np.r_[TIME-DT/2,TIME[-1]+DT/2]
neural[ii,jj]=matlab_smooth((np.histogram(z,edges)[0]/DT)[None,:])[0]
...
inp.append(TIME[None,:].astype(np.float32))
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Three per-trial Bpod fields: the instructed side `bp.L` (with `bp.R` loaded but unused), and the outcome flags `bp.hit` and `bp.miss`. The actual licked port is not recorded, so direction is inferred from instructed side × outcome.

ii.
```python
L=arr(b['L']); R=arr(b['R']); aw=arr(b['autowater'])
hit=arr(b['hit']); miss=arr(b['miss']); no=arr(b['no'])
```

iii. Step 27: "`lick direction` currently uses instructed trial side, so it has no `none` values and labels incorrect responses with the wrong side; actual lick direction should be instructed side for hits, opposite side for misses, and none for ignores." This was a mid-course correction: the first version coded `L`/`R` directly and was patched at step 29.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Per trial: if the trial is neither hit nor miss (i.e. an ignore) → `2` (`none`); if hit → the instructed side (`L` → `0` left, else `1` right); if miss → the opposite of the instructed side. The scalar is then broadcast to all 1001 time bins as an `int16` row. `output_values[0] = ['left','right','none']`. Observed distribution: 39.8% left, 37.7% right, 22.5% none.

ii.
```python
outcome=1 if hit[t]>0 else (0 if miss[t]>0 else 2)
if outcome==2:
    lick=2
elif hit[t]>0:
    lick=0 if L[t]>0 else 1
else:
    lick=1 if L[t]>0 else 0
...
rows=[np.full(TIME.size,x,dtype=np.int16) for x in base]
```

iii. Step 29: "Lick direction must be based on actual response: instructed side on hits, opposite side on misses, none on ignores." The instruction's `left, right, none` value set forced the third class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The single per-trial flag `bp.autowater`, which marks water-cued (WC) block trials.

ii.
```python
aw=arr(b['autowater'])
```

iii. Step 16 identified `autowater` as the context label; the paper's conditions (`...&autowater` vs `...&~autowater`) and Figure 8's DR/WC conditions use exactly this field.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A relabelling: autowater → `0` (WC), otherwise → `1` (DR), broadcast over all 1001 bins. Observed: 31.5% WC, 68.5% DR.

ii.
```python
outs.append([lick,int(not (aw[t]>0)),outcome])
...
'output_values':[...,['WC','DR'],...]
```

iii. Step 25: "categorical ordering: `output_values` declares WC=0 and DR=1, but current encoding uses `autowater` directly (WC=1). That must be corrected for metadata/value consistency." The AI patched the encoding so the codes match the declared `output_values` order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit` and `bp.miss`. `bp.no` is loaded but not used — a trial that is neither hit nor miss is treated as an ignore by construction.

ii.
```python
hit=arr(b['hit']); miss=arr(b['miss']); no=arr(b['no'])
```

iii. Step 16 lists "outcomes (`hit`,`miss`,`no`)" as the available labels; the AI derived the three-way class from the first two flags in the same expression that drives lick direction.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `hit` → `1` (correct), `miss` → `0` (incorrect), otherwise → `2` (ignore), broadcast over all 1001 bins, matching the requested value order `['incorrect','correct','ignore']`. Observed: 10.6% incorrect, 66.9% correct, 22.5% ignore.

ii.
```python
outcome=1 if hit[t]>0 else (0 if miss[t]>0 else 2)
...
'output_values':[...,['incorrect','correct','ignore'],...]
```

iii. Step 15/18: ignore and error trials are excluded from the paper's figure conditions but "the decoder specification explicitly requires" them, so all outcomes are retained as classes rather than filtered.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DeepLabCut tracking in `obj.traj{1}` (camera index 0, the side view): the feature named `tongue`, via `traj[0]['ts'][trial][:, 0:2, j]` (x, y) and `[:, 2, j]` (likelihood), with `traj[0]['featNames'][trial]` giving the feature order and `traj[0]['frameTimes'][trial]` the frame clock. Only the side camera is used; the bottom-camera tongue landmarks (`top_tongue`, etc.) are ignored. `bp.ev.goCue` is used for alignment.

ii.
```python
for cam, feats in [(0,['tongue']),(1,['top_paw','bottom_paw'])]:
    tr=o['traj'][cam]; ts=np.asarray(trial_item(tr['ts'],i),float)
    ft=arr(trial_item(tr['frameTimes'],i))-go
    nm=names_at(tr,i)
    coords=[]
    for feat in feats:
        if feat not in nm: continue
        j=nm.index(feat); xy=ts[:,0:2,j].copy(); lk=ts[:,2,j]
```

iii. Step 11: "each trial has two-camera DLC arrays `(frames, 3, features)` (x, y, likelihood), camera 0 includes side-view tongue/paw landmarks, camera 1 includes bottom-view landmarks". Step 18: "Landmarks are identified: side-view tongue (`tongue`) and bottom-view paw landmarks (`top_paw`, `bottom_paw`)." This matches `params.traj_features` in `getDefaultParams.m`, where `tongue` is a cam-0 feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps. (1) Frames with DLC likelihood `< 0.9` have their x, y set to NaN ("low-confidence labels are missing"). (2) Speed at each frame is `hypot(np.gradient(x), np.gradient(y))` — a first-order derivative with respect to *frame index*, i.e. pixels per frame, with no smoothing of the position trace; frames that are not visible are then re-set to NaN. (3) The frame-resolution speed trace is resampled onto the 1001-point 5 ms grid with `np.interp`, keeping only frames where both the frame time and the speed are finite, and returning NaN outside the covered range (`left=nan, right=nan`). (4) The result is thresholded (7-c). Note that because step 3 interpolates between the surviving finite samples, gaps *inside* the tracked range — the intervals between licks where the tongue is retracted — are filled with linearly interpolated velocities rather than left missing; in the saved file every trial's `not visible` bins form only one or two runs touching the trial edges.

ii.
```python
xy[lk<.9]=np.nan; coords.append(xy)
...
xy=np.nanmean(np.stack(coords),axis=0)
# Velocity magnitude in pixels/frame, matching findVelocity's gradient of
# interpolated coordinates. Interpolate positions first onto video frames;
# preserve visibility separately through NaNs.
visible=np.all(np.isfinite(xy),axis=1)
vx=np.gradient(xy[:,0]); vy=np.gradient(xy[:,1])
speed=np.hypot(vx,vy); speed[~visible]=np.nan
out.append(interp_stream(ft,speed))
...
def interp_stream(ft,val):
    ft=arr(ft); val=arr(val)
    ok=np.isfinite(ft)&np.isfinite(val)
    if ok.sum()<2:return np.full(TIME.size,np.nan)
    order=np.argsort(ft[ok]); xx=ft[ok][order]; yy=val[ok][order]
    return np.interp(TIME,xx,yy,left=np.nan,right=np.nan)
```

iii. Step 18: "The paper velocity routine computes coordinate gradients after interpolation; tongue NaNs are set to zero in paper analyses, but the decoder specification explicitly requires a separate not-visible class, so visibility must be retained before filling." The docstring says video missingness "is preserved as an explicit category rather than filled as in plotting analyses", and the code comment claims the gradient matches `findVelocity.m` (which indeed uses bare `gradient` on the position, i.e. per frame).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session, the 50th percentile is taken over all finite tongue-speed samples pooled across all trials and all time bins of the session. Bins with a finite value `>=` that threshold are `1`, finite values below are `0`, and bins whose value is NaN are `2` (`not visible`). The three thresholds per session are stored in `metadata['session_info'][k]['video_median_thresholds']`. Observed overall: 12.2% / 12.2% / 75.6%.

ii.
```python
# Per-session medians over all visible samples, as requested.
thresholds=[]
for k in range(3):
    z=np.concatenate([x[k][np.isfinite(x[k])] for x in continuous])
    thresholds.append(float(np.percentile(z,50)) if z.size else np.nan)
...
for k,z in enumerate(v):
    y=np.full(TIME.size,2,dtype=np.int16); ok=np.isfinite(z)
    y[ok]=(z[ok]>=thresholds[k]).astype(np.int16); rows.append(y)
```
```python
'output_values':[..., ['below session median','at or above session median','not visible'], ...]
```

iii. The decoder task specifies exactly this rule ("discretized with per-session threshold: 0 < 50th percentile, 1 >= 50th percentile, 2 not visible"); the code comment "Per-session medians over all visible samples, as requested" points at the instructions, and step 15's plan is to "discretize each session at its visible-value median".

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are taken from `traj[0]['frameTimes'][trial]` and the trial's go cue is subtracted; the resulting trace is interpolated onto the shared 1001-point neural grid. **The video-to-behaviour clock offset is not applied.** The repository's `findVideoOffset.m` (`mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`) and `loadMotionEnergy.m` (`interp1(obj.traj{1}(trix).frameTimes - vidshift - alignTimes(trix), ...)`) both subtract it. I measured this offset directly in the data: 0.490 s for the 2021 sessions (JEB6, JEB7, EKH1, JGR2, …) and 0.990 s for the JEB19 sessions. Consequently all video-derived outputs are shifted late by 0.49–0.99 s relative to the neural data, which also shows up as the constant 10.4% (2021) / 20.4% (JEB19) of bins at the start of every trial that fall outside the (shifted) frame coverage and are labelled `not visible`.

ii.
```python
def video_trial(o,i,go):
    """Return tongue speed, paw speed, ME on the common go-aligned grid."""
    ...
    tr=o['traj'][cam]; ts=np.asarray(trial_item(tr['ts'],i),float)
    ft=arr(trial_item(tr['frameTimes'],i))-go
    ...
    out.append(interp_stream(ft,speed))
...
v=video_trial(o,t,go[t])
```

iii. The trajectory never mentions `findVideoOffset` or `vidshift`. The AI did read `loadMotionEnergy.m` (step 28: "The repository confirms motion energy is normally loaded from a separate `motionEnergy*` file and resampled to the common neural time axis") but took only the resampling idea from it, not the clock correction on the line it resamples with. Its stated plan (step 15) was simply to "interpolate video outputs to 5 ms" using `frameTimes`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DeepLabCut tracking in `obj.traj{2}` (camera index 1, bottom view): **both** `top_paw` and `bottom_paw`, with their x, y coordinates averaged together, plus that camera's `frameTimes` and `featNames`, and `bp.ev.goCue`.

ii.
```python
for cam, feats in [(0,['tongue']),(1,['top_paw','bottom_paw'])]:
    ...
    for feat in feats:
        if feat not in nm: continue
        j=nm.index(feat); xy=ts[:,0:2,j].copy(); lk=ts[:,2,j]
        xy[lk<.9]=np.nan; coords.append(xy)
    if not coords: out.append(np.full(TIME.size,np.nan)); continue
    xy=np.nanmean(np.stack(coords),axis=0)
```

iii. Step 18: "Landmarks are identified: side-view tongue (`tongue`) and bottom-view paw landmarks (`top_paw`, `bottom_paw`)", i.e. the AI took both paw features listed for cam 1 in `params.traj_features` and combined them into a single "paw" signal. The paper states the paws are tracked only from the bottom view, which is consistent with the camera choice.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical pipeline to the tongue: likelihood `< 0.9` → NaN; the two paws' coordinates averaged with `np.nanmean` (so when one paw is untracked the trace silently switches to the other paw's position); speed = `hypot(gradient(x), gradient(y))` in pixels per frame with no positional smoothing; NaN at invisible frames; then `np.interp` onto the 5 ms grid with NaN only outside the covered range. No normalisation. As with the tongue, interior missing stretches are bridged by the interpolation rather than kept as missing.

ii.
```python
xy=np.nanmean(np.stack(coords),axis=0)
visible=np.all(np.isfinite(xy),axis=1)
vx=np.gradient(xy[:,0]); vy=np.gradient(xy[:,1])
speed=np.hypot(vx,vy); speed[~visible]=np.nan
out.append(interp_stream(ft,speed))
```

iii. Same justification as 7-b (step 18): reproduce `findVelocity.m`'s gradient of position, but keep visibility information instead of the paper's `fillmissing(...,'nearest')`, because the decoder spec requires a `not visible` class.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Exactly as the tongue: session-wide 50th percentile of all finite paw-speed samples; `>=` → 1, `<` → 0, NaN → 2. Observed: 43.0% / 43.0% / 14.0%.

ii.
```python
z=np.concatenate([x[k][np.isfinite(x[k])] for x in continuous])
thresholds.append(float(np.percentile(z,50)) if z.size else np.nan)
...
y=np.full(TIME.size,2,dtype=np.int16); ok=np.isfinite(z)
y[ok]=(z[ok]>=thresholds[k]).astype(np.int16)
```

iii. As specified by the decoder task ("0: < 50th percentile, 1: >= 50th percentile, 2: not visible").

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Via the bottom camera's own `frameTimes` minus the trial's go cue, interpolated onto the shared 1001-bin grid — and, as for the tongue, **without** the `findVideoOffset` clock correction, so the paw signal is shifted by 0.49 s (2021 sessions) or 0.99 s (JEB19 sessions) relative to the spikes. The uniform `not visible` fraction of 10.4%/20.4% per session is exactly the leading part of the window left uncovered by this shift.

ii.
```python
tr=o['traj'][cam]
ft=arr(trial_item(tr['frameTimes'],i))-go
...
return np.interp(TIME,xx,yy,left=np.nan,right=np.nan)
```

iii. Same as 7-d: no mention of the video/ephys clock offset anywhere in the trajectory. Using each camera's own `frameTimes` (rather than always camera 0's) is a deliberate detail, since the two cameras can have different frame counts.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Only the embedded `obj.me` field (`obj.me['data'][trial]`, one value per side-camera frame), unwrapped through possible list nesting, with `obj.traj[0]['frameTimes']` as its time base. The stand-alone `motionEnergy_<anm>_<date>.mat` files — which exist for all 25 sessions in `Ephys_Behavior` (and all of `RandomizedDelay_Ephys_Behavior`) — are never read. `obj.me` is absent from the eight 2021 sessions, so motion energy is entirely missing for 8 of 12 sessions and those trials are filled with class `2` (`no video`) at every bin. Overall 73.3% of bins are `no video`.

ii.
```python
meobj=o.get('me',{})
me=meobj.get('data',[]) if isinstance(meobj,dict) else (meobj if isinstance(meobj,list) else [])
try:
    m=trial_item(me,i); ft=arr(trial_item(o['traj'][0]['frameTimes'],i))-go
    while isinstance(m,list) and len(m)==1: m=m[0]
    out.append(interp_stream(ft,m))
except Exception: out.append(np.full(TIME.size,np.nan))
```

iii. Step 27: "motion energy is entirely missing. The context files may store `me` differently or omit the precomputed stream even though video exists; we need inspect its representation and, if absent, reproduce motion energy from video rather than mark every sample `no video`." Step 28 then read `loadMotionEnergy.m` and noted it loads a separate file. Step 29 concluded: "Motion energy is genuinely absent from the released objects for older sessions (`None`), matching the repository's behavior of continuing with NaN when the separate file cannot be located. JEB19 sessions do embed per-trial motion energy as nested lists, so the converter must unwrap those… Older sessions still have tracked video but no motion-energy stream; class 2 is the only target-supported missing category." The AI checked only `obj.me` and never listed `/app/data/*/motionEnergy_*.mat`, so its premise that the stream is unavailable is false.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling: the per-frame scalar (already reduced to the 99th percentile across pixels upstream, per the paper's Methods) is linearly interpolated from side-camera frame times onto the 5 ms grid, NaN outside the covered range, then thresholded. Where `obj.me` is missing, the trace is all-NaN. Nested single-element list wrappers are unwrapped in a `while` loop; any exception during extraction is caught and converted to an all-NaN trace.

ii.
```python
while isinstance(m,list) and len(m)==1: m=m[0]
out.append(interp_stream(ft,m))
```

iii. Step 28/29, as quoted above; `loadMotionEnergy.m` itself resamples with `interp1(frameTimes-vidshift-alignTimes, me.data{trix}, taxis)`, and the AI mirrored the `interp1` step (though not the `vidshift`, nor the subsequent `fillmissing`).

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same rule as the other two streams: per-session 50th percentile over all finite samples, `>=` → 1, `<` → 0, NaN → 2 (`no video`). For the eight sessions without `obj.me` the pooled sample is empty, the threshold is stored as `nan`, and every bin is 2. Observed overall: 13.3% / 13.4% / 73.3% (and 100% class 2 in 8 of 12 sessions).

ii.
```python
z=np.concatenate([x[k][np.isfinite(x[k])] for x in continuous])
thresholds.append(float(np.percentile(z,50)) if z.size else np.nan)
...
y=np.full(TIME.size,2,dtype=np.int16); ok=np.isfinite(z)
y[ok]=(z[ok]>=thresholds[k]).astype(np.int16)
```
```python
'video_median_thresholds':dict(zip(['tongue_velocity','paw_velocity','motion_energy'],thr))
```

iii. The threshold rule is taken from the decoder task; the "no video" class is the instruction's third category, which the AI used for sessions lacking `obj.me` (step 29: "class 2 is the only target-supported missing category").

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Using `obj.traj[0]['frameTimes']` (the side camera, which is the camera motion energy is computed from and the one `loadMotionEnergy.m` uses) minus the trial's go cue, interpolated onto the shared grid — again **without** the `findVideoOffset` correction, so the four JEB19 sessions that do have motion energy are shifted by 0.99 s, leaving a constant 20.4% of leading bins uncovered.

ii.
```python
m=trial_item(me,i); ft=arr(trial_item(o['traj'][0]['frameTimes'],i))-go
out.append(interp_stream(ft,m))
```

iii. Choosing camera 0's frame times follows `loadMotionEnergy.m` (`obj.traj{1}(trix).frameTimes`); the omission of `vidshift` from that same line is not discussed anywhere in the trajectory.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several small guards, all of which keep the trial and mark the gap rather than drop or impute:
- Low-confidence DLC labels (`likelihood < 0.9`) → NaN coordinates → NaN speed → class `2`.
- Frames whose `frameTimes` or speed are non-finite are excluded from the interpolation; if fewer than two usable samples remain, the whole trial's stream is NaN (class `2`).
- Absent/oddly-wrapped `obj.me` (`None`, list, dict, nested single-element lists) is normalised, and any failure is caught by a bare `except Exception` producing an all-NaN trace.
- Empty threshold pools produce `nan` thresholds and an all-`2` output.
- `trial_item`/`names_at` normalise mat73's inconsistent list-vs-array and nested-string representations.
- A session yielding <2 trials or 0 units would be dropped with a warning (never triggered).
- `np.nanmean` is used when averaging the two paw features, so one paw suffices (this emits the `RuntimeWarning: Mean of empty slice` seen in the run log when both are missing).
Two caveats: `np.interp` silently *fills* interior gaps, so the "mark the gap" policy only holds at the two ends of a trial; and the bare `except Exception` would also convert a genuine extraction bug into "no video" without any message.

ii.
```python
xy[lk<.9]=np.nan
...
def interp_stream(ft,val):
    ok=np.isfinite(ft)&np.isfinite(val)
    if ok.sum()<2:return np.full(TIME.size,np.nan)
    ...
    return np.interp(TIME,xx,yy,left=np.nan,right=np.nan)
...
except Exception: out.append(np.full(TIME.size,np.nan))
...
thresholds.append(float(np.percentile(z,50)) if z.size else np.nan)
...
if len(neu)<2 or not neu or neu[0].shape[0]==0:
    warnings.warn('dropping unusable session '+p); continue
```

iii. Step 18: visibility "must be retained before filling" because the decoder spec requires a not-visible class; step 25: "those files represent absent motion energy as a list rather than a dictionary; this should map to the required `no video` category"; step 29: unwrap nested lists for JEB19 rather than discarding them. The module docstring states the overall policy: "Video missingness is preserved as an explicit category rather than filled as in plotting analyses."

## 11-a. What are the most time-consuming steps of the code?

i. Two steps dominate. (1) `mat73.loadmat` reads each 100–300 MB v7.3 file in full, materialising every field of `obj` — including spike waveforms, all SpikeGLX index arrays, and all tracked features — even though only a handful are used. (2) The neural binning block, a Python double loop over units × kept trials (~140,000 iterations over the 12 sessions), each iteration doing one `np.histogram` of 1001 bins and one 15-tap convolution, with the Gaussian kernel rebuilt inside `matlab_smooth` on every single call. The whole conversion took about 110 s of wall clock for the 12 sessions (22:28:38 → 22:30:27 in the trajectory), i.e. ~9 s per session, so file loading is the larger share. Video processing is comparatively cheap (three interpolations per trial).

ii.
```python
o=mat73.loadmat(path)['obj']
...
for jj,u in enumerate(keep0):
    st=arr(clu['trialtm'][u]); tri=arr(clu['trial'][u]).astype(int)-1
    for ii,t in enumerate(valid):
        z=st[tri==t]-go[t]
        neural[ii,jj]=matlab_smooth((np.histogram(z,edges)[0]/DT)[None,:])[0]
```

iii. Step 23: "To keep memory manageable, sessions will be loaded and processed one at a time." Step 16: "Loading all 50 huge files with mat73 may be slow, so context sessions should first be identified from lightweight HDF5 dereferencing or known Figure 8 metadata" — the AI's main efficiency measure was to reduce the number of files it opens rather than to speed up per-file processing.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest one is the unit × trial neural loop: the whole session could be binned in one `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, edges])` per unit (or per session) and smoothed with a single `convolve1d`/`gaussian_filter1d` call over the `(trials, units, time)` array, instead of ~140k separate `np.histogram` + `np.convolve` calls. Inside that loop, `st[tri==t]` re-scans the unit's full spike vector once per trial, which is O(units × trials × spikes) — sorting spikes by trial once would remove it. `matlab_smooth` is also called on one row at a time although it already accepts a 2-D array, and it rebuilds the kernel every call. The per-trial `video_trial` loop is harder to vectorise because each trial has its own frame count, but `names_at` is re-parsed per trial per camera even though the feature list is constant within a session.

ii.
```python
for jj,u in enumerate(keep0):
    ...
    for ii,t in enumerate(valid):
        z=st[tri==t]-go[t]
        neural[ii,jj]=matlab_smooth((np.histogram(z,edges)[0]/DT)[None,:])[0]
...
def matlab_smooth(x):
    ...
    return np.stack([np.convolve(row,k,'same') for row in x],axis=0)
```

iii. Not discussed in the trajectory; the AI's stated performance concern was memory and number of files loaded, not inner-loop speed.

## 11-c. What processing does the code repeat multiple times?

i. (1) The smoothing kernel is recomputed from scratch on each of the ~140k `matlab_smooth` calls — and the first, `scipy.signal.windows.gaussian(...)` line is computed and then immediately overwritten by the correct formula, so it is pure waste. (2) `st[tri==t]` re-evaluates a full boolean mask over the unit's spike train for every trial. (3) `video_trial` re-fetches `o['traj'][cam]` and re-decodes `featNames` (`names_at`) for every trial and camera, although the names are fixed within a session. (4) The `>1 Hz` filter is applied *after* smoothing all units, so the ~2% of units that are discarded were binned and smoothed for nothing (and in JEB19 sessions with 400 garbage clusters, the curation filter at least runs first). (5) The same `TIME` vector is re-materialised as a new array for every one of the 3,116 trials in `input`.

ii.
```python
def matlab_smooth(x):
    k=gaussian(15,std=(15-1)/6, sym=True) # discarded on the next line
    n=np.arange(15)-(14/2); k=np.exp(-.5*(2.5*n/(14/2))**2)
    k[:7]=0; k/=k.sum()
...
unitkeep=np.nanmean(neural,axis=(0,2))>1
neural=neural[:,unitkeep,:]
...
inp.append(TIME[None,:].astype(np.float32))
```

iii. Not discussed in the trajectory.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `mat73.loadmat` deserialises the entire `obj` — `clu.spkWavs`, `clu.tm`, `sglx`, `ex`, `trials`, and every tracked feature other than `tongue`, `top_paw`, `bottom_paw` — of which only a few fields are ever touched; with 100–300 MB files this is the single largest piece of wasted work. (2) The dead `gaussian(15, ...)` call in `matlab_smooth`. (3) `R=arr(b['R'])` and `no=arr(b['no'])` are loaded but never used (lick direction uses `L` only, outcome uses `hit`/`miss` only), and `n = bp.Ntrials` is only printed. (4) Units below 1 Hz are fully binned and smoothed before being thrown away. (5) Outputs are stored as `int16` where the six variables only take values 0–2, doubling the output array's footprint versus `int8`; the input stores 3,116 separate copies of the same 1001-value time vector. (6) Motion energy is interpolated for JEB19 but the eight sessions without `obj.me` still run the full (trivially all-NaN) threshold machinery. Notably absent from the wasted-work list: nothing in the video pipeline is computed and then dropped — including the interpolated values that fill invisible stretches, which are *not* discarded but instead become spurious 0/1 labels.

ii.
```python
o=mat73.loadmat(path)['obj']          # whole object, including spike waveforms
k=gaussian(15,std=(15-1)/6, sym=True) # overwritten two lines later
L=arr(b['L']); R=arr(b['R']); aw=arr(b['autowater'])
hit=arr(b['hit']); miss=arr(b['miss']); no=arr(b['no'])
...
rows=[np.full(TIME.size,x,dtype=np.int16) for x in base]
```

iii. Step 24 mentions saving "float32/int arrays to control pickle size", which is the one place the AI explicitly traded precision for size (neural stored as `float32`, 605 MB total). The remaining redundancies are not discussed.
