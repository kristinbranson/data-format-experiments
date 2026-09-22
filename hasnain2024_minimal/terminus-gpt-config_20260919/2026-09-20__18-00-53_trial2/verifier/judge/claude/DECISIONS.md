# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI restricted the conversion to a single data folder, `/app/data/Ephys_Behavior`, and within it to the 11 released sessions belonging to the six mice that the paper's "two-context" (WC/DR) electrophysiology dataset is built from (`JEB6, JEB7, EKH3, JGR2, JGR3, JEB19`). The 33 other released sessions (the DR-only fixed-delay mice `EKH1, JEB13, JEB14, JEB15` and the entire `RandomizedDelay_Ephys_Behavior` folder) are never opened. Session discovery is a glob over `data_structure_*.mat`, but each hit is only kept if its `(mouse, date)` key appears in a hard-coded `PROBES` dictionary, which simultaneously fixes the ALM probe for that session (transcribed from the authors' `load<ANM>_ALMVideo.m` metadata scripts). Each session file is read once, in full, with `mat73.loadmat` (all 11 files are MATLAB v7.3); the companion motion-energy file is read with `scipy.io.loadmat`.

ii.
```python
ROOT='/app/data/Ephys_Behavior'
# Figure 8's six mice; all their released sessions (11 files; paper reports 12).
MICE={'JEB6','JEB7','EKH3','JGR2','JGR3','JEB19'}
PROBES={
 ('EKH3','2021-08-11'):[2], ('JEB6','2021-04-18'):[2],
 ('JEB7','2021-04-29'):[1], ('JEB7','2021-04-30'):[1],
 ('JGR2','2021-11-16'):[1], ('JGR2','2021-11-17'):[1],
 ('JGR3','2021-11-18'):[1],
 ('JEB19','2023-04-18'):[1], ('JEB19','2023-04-19'):[1],
 ('JEB19','2023-04-20'):[1], ('JEB19','2023-04-21'):[1]}

files=[]
for f in glob.glob(ROOT+'/data_structure_*.mat'):
    base=os.path.basename(f)[15:-4]; mouse,day=base.split('_',1)
    if (mouse,day) in PROBES: files.append((mouse,day,f))
files.sort()
for mouse,day,f in files:
    print('loading',mouse,day,flush=True); o=mat73.loadmat(f)['obj']; b=o['bp']; n=int(b['Ntrials'])
```

iii. From the trajectory: at step 4 the AI decided `Ephys_Behavior` was "almost certainly" the relevant source because it holds the paper's 25 ALM sessions, and dismissed the randomized-delay folder because it "lacks the required WC context" and the optogenetic folders because they are "primarily behavioral". At step 26 it settled the mouse list by reading the repository's Figure 8 scripts: "Figure 8 explicitly loads JEB6, JEB7, EKH3, JGR2, JGR3, and JEB19 — the paper's six two-context mice. In the provided data this corresponds to 11 available sessions (1+2+1+2+1+4), suggesting one of the paper's 12 sessions is not included in this data release." At step 18 it parsed the `load<ANM>_ALMVideo.m` metadata scripts to build the probe map, noting these "provide the authoritative ALM probe mapping". At step 25 it had first tried the paper's behavioural inclusion criterion (≥40 correct DR trials/direction, ≥20 correct WC trials/direction), obtained only seven sessions, and concluded at step 27 that "the paper's behavioral threshold was an analysis-specific criterion rather than a session-selection rule we should reapply here".

## 1-b. How are the data split into subjects?

i. The subject is the part of the filename between `data_structure_` and the date, recovered by `base.split('_',1)`. The `subjects` list is the sorted hard-coded `MICE` set (six animals), and `subject_idx` is `subjects.index(mouse)` appended once per session, in the same order as `neural`/`input`/`output`. Each of the 11 sessions therefore maps to one of six subjects (`['EKH3','JEB19','JEB6','JEB7','JGR2','JGR3']`, with `subject_idx = [0 1 1 1 1 2 3 3 4 4 5]`).

ii.
```python
base=os.path.basename(f)[15:-4]; mouse,day=base.split('_',1)
...
subjects=sorted(MICE); subject_idx=[]
...
subject_idx.append(subjects.index(mouse))
...
'subjects':subjects,'subject_idx':np.asarray(subject_idx,int)
```

iii. The trajectory does not discuss this explicitly; the AI took the animal id from the filename throughout (its probe-map parser at step 18 already keyed sessions as `anm`/`date` pairs read from the filenames and metadata scripts), and its `MICE` set is the six-mouse list it read off the Figure 8 loading script at step 26.

## 1-c. How are the data split into sessions?

i. One `data_structure_<anm>_<date>.mat` file is one session and becomes one element of `neural`, `input`, `output`, `brain_region_idx`, `subject_idx` and `metadata['session_info']`. Sessions are ordered by `files.sort()`, i.e. alphabetically by `(mouse, date, path)`. All 11 sessions come from the single fixed-delay folder; the fixed-delay and randomized-delay experiments are not merged because the latter is excluded entirely. Per session the ALM probe(s) to use are looked up in `PROBES` and their clusters concatenated (all 11 entries happen to name exactly one probe).

ii.
```python
files.sort()
for mouse,day,f in files:
    ...
    units=[]
    for p in PROBES[(mouse,day)]:
        c=o['clu'][p-1]
        for tr,tm in zip(c['trial'],c['trialtm']): units.append(...)
    ...
    neural.append(ns);inputs.append(ins);outputs.append(outs)
    region_idx.append(np.zeros(rates.shape[0],int));subject_idx.append(subjects.index(mouse))
    session_info.append({'subject':mouse,'date':day, ...})
```

iii. Step 9: "Metadata scripts explicitly identify which probe corresponds to ALM (for example EKH1 uses probe 2), so neural extraction must respect `obj.probe` or the repository metadata rather than blindly combining probes." Step 26 fixed the session list to the Figure 8 mice, as in 1-a.

## 1-d. How are the data split into trials?

i. A trial is one row of the `obj.bp` per-trial table (`bp.Ntrials` entries) with one go cue in `bp.ev.goCue`. Spikes carry their own trial index (`clu.trial`, 1-based, converted with `-1`) and DeepLabCut tracking and motion energy are stored as one cell per trial, so no trial boundaries have to be reconstructed. The AI reads the per-trial flag vectors whole, without truncating them to `Ntrials` (in the 11 sessions used, all of these vectors are exactly `Ntrials` long, so nothing is over-read).

ii.
```python
b=o['bp']; n=int(b['Ntrials'])
early=np.asarray(b['early']).astype(bool); keep=np.flatnonzero(~early)
go=np.asarray(b['ev']['goCue'],float).ravel()
...
for ui,(tr,tm) in enumerate(units):
    for t in np.unique(tr):
        j=keepmap.get(int(t))
        ...
        h=np.histogram(tm[tr==t]-go[t],EDGES)[0]
```
and, for trial indices, `tr=np.asarray(tr,int).ravel()-1`.

iii. Step 28: "Cluster spike times are stored per unit as trial number and `trialtm`, where `trialtm` is relative to trial start/bit start, not go cue." Step 11: "Video tracking is stored as per-trial arrays with frame times and `ts` shaped `(frames, 3, 7)` ... Motion energy is a variable-length 1D trace per trial with matching trial count."

## 1-e. How are trials filtered based on quality controls?

i. Exactly one filter: early-lick trials (`bp.early`) are dropped. Trials where the animal did not respond (`bp.no`, "ignore") are deliberately kept because `ignore` is a required output class. Trials whose go cue is not finite are silently skipped when spikes are binned (their neural row stays zero) but are still emitted as trials. **Photostimulation trials (`bp.stim.enable`) are not excluded**, and no check is made for trials that run past the end of the ephys recording. Across the 11 sessions this keeps 2,985 of 3,067 non-early trials, of which 121 (≈4%) are optogenetic photoinactivation trials — including 55 in `JEB6_2021-04-18`, 39 in `JEB7_2021-04-29` and 26 in `JEB7_2021-04-30`, i.e. 9–15% of those sessions.

ii.
```python
early=np.asarray(b['early']).astype(bool); keep=np.flatnonzero(~early)
...
for t in np.unique(tr):
    j=keepmap.get(int(t));
    if j is None or not np.isfinite(go[t]): continue
```
and in the metadata:
```python
'trial_filter':'Figure-8 two-context sessions; early-lick trials excluded; ignore trials retained'
```

iii. Step 3: "Early-lick and ignore trials were omitted in the paper's behavioral analyses, but the decoder explicitly requires an ignore outcome, so trial inclusion needs closer inspection." Step 17: "Early-lick trials should be excluded consistent with the paper, while ignore trials must remain because explicitly requested as an output class." Photostimulation is never mentioned anywhere in the trajectory — the AI's own session scan printed only `hit/miss/no/early/autowater` counts and never inspected `bp.stim.enable`, so no justification exists for retaining those trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the ALM probe named in `PROBES`, specifically the two columnar fields `trial` (1-based trial index of each spike) and `trialtm` (spike time relative to trial start, on the behaviour clock). `obj.bp.ev.goCue` supplies the alignment time. `clu.quality` and `clu.spkWavs` are loaded by `mat73` but never used. Two-probe sessions would concatenate both probes' clusters into one unit list.

ii.
```python
units=[]
for p in PROBES[(mouse,day)]:
    c=o['clu'][p-1]
    for tr,tm in zip(c['trial'],c['trialtm']):
        units.append((np.asarray(tr,int).ravel()-1,np.asarray(tm,float).ravel()))
go=np.asarray(b['ev']['goCue'],float).ravel()
```

iii. Step 28: "Cluster spike times are stored per unit as trial number and `trialtm` ... The repository's `getSeq` aligns these times using an `alignEvent`; for this task we must subtract each trial's `bp.ev.goCue`." Step 17: "The saved `data_structure` files are raw session objects, not precomputed `trialdat`; conversion must reproduce repository processing from spikes."

## 2-b. How is the `neural` data processed?

i. For every unit and every kept trial, spikes are histogrammed into the fixed 5 ms edge grid, divided by the bin width to give spikes/s, and convolved along time with the repository's **causal** Gaussian: a 15-tap `gausswin` with `std = (15-1)/(2·2.5) = 2.8` samples, whose first 7 taps are zeroed and which is then renormalised to sum to 1. Convolution is `np.convolve(..., mode='same')`, i.e. zero-padded boundaries, matching `processData.m`'s default `params.bctype = 'none'`. No normalisation, baseline subtraction or z-scoring; values are stored as `float32` Hz. Units from the selected probe(s) are pooled into one population.

ii.
```python
def smooth_rates(x):
    # MATLAB mySmooth: gausswin(15), first floor(N/2) samples set to zero.
    k=gaussian(15, std=(15-1)/(2*2.5), sym=True); k[:7]=0; k/=k.sum()
    return np.apply_along_axis(lambda z:np.convolve(z,k,mode='same'),1,x)
...
h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT
rates[ui,j]=smooth_rates(h[None,:])[0]
```

iii. Step 14: "`getSeq.m` bins spikes on the common time grid, divides by `dt` to produce firing rates in Hz, and applies `mySmooth`." Step 15: "The smoothing kernel is explicitly causal: a Gaussian window of length 15 has its first half zeroed, is normalized, and is convolved with each binned spike train." Step 29/30: the AI revised its kernel mid-run — "adjust the Gaussian standard deviation to MATLAB `gausswin(15)`'s default alpha=2.5 equivalent for closer processing parity ... use the standard deviation equivalent to MATLAB `gausswin(15)` with default alpha 2.5 (`std=2.8`)".

## 2-c. How is the `neural` data filtered based on quality controls?

i. **Only one filter: mean firing rate.** After smoothing, a unit is kept if its mean rate over all kept trials and all 1000 bins exceeds 1 Hz. The manual spike-sorting label `clu.quality` is never read, so the repository's `findClusters.m` exclusion of `garbage` / `gabrga` / `noisy` / `real?` clusters is not applied. This retains 912 units over 11 sessions. The consequence is large in the four `JEB19` sessions, whose files contain 400–529 clusters manually labelled `garbage`: e.g. in `JEB19_2023-04-19` 96 of the 132 retained "units" (73%) are `garbage`, and in `JEB19_2023-04-20` 137 of 204 (67%). The paper reports 522 units in total for the 12 two-context sessions; the AI's 11 sessions yield 912, 613 of them from `JEB19` alone.

ii.
```python
use=rates.mean(axis=(1,2))>1.0; rates=rates[use]
```
and the metadata claim:
```python
'neural_representation':'5 ms firing rates smoothed with repository 15-bin causal Gaussian; units with mean rate >1 Hz'
```

iii. Step 3: "recording sessions require at least 10 units, and all analyses other than specified single-unit analyses include all units (single and multiunits) with firing rate >1 Hz." Step 12 read the reference defaults and concluded "cluster quality is unrestricted at this loading stage" from `params.quality = {'all'}` — the AI did not notice that `findClusters.m` still excludes `garbage`/`gabrga`/`noisy`/`real?` when `'all'` is requested. Step 28: "Unit filtering is based on mean trial-averaged PSTH across conditions exceeding 1 Hz." The session-level "at least 10 units" criterion is also not implemented (all 11 sessions pass it anyway).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. A single subtraction. `clu.trialtm` is already on the behaviour clock relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm − goCue[trial]` gives seconds from go cue directly; those values are then histogrammed into `EDGES`. Spikes outside ±2.5 s fall outside the histogram range and are discarded. Trials with a non-finite go cue are skipped.

ii.
```python
h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT
```
with
```python
'temporal_alignment_event':'go cue onset (water delivery in WC trials)'
```

iii. Step 28: "The repository's `getSeq` aligns these times using an `alignEvent`; for this task we must subtract each trial's `bp.ev.goCue`." Step 19 also checked the convention empirically: "Go cue occurs 2.5 s after trial start, matching the repository's ±2.5 s go-cue-aligned window."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins (`DT = 0.005`, 200 Hz) over a window of −2.5 s to +2.5 s from the go cue, giving 1000 non-overlapping bins per trial. The edge grid `EDGES` and bin-centre vector `TIME` are built once at module level and shared by every unit, trial, session and data stream. Spikes are counted directly into this grid, so there is no rebinning of an intermediate representation; the camera streams (400 Hz) are resampled onto the same 1000 bin centres by linear interpolation. `metadata['time_bin_size']` is reported as 5.0 ms and `off_start`/`off_end` as −2.5/+2.5.

ii.
```python
DT=.005; T0=-2.5; T1=2.5
EDGES=np.arange(T0,T1+DT/2,DT); TIME=EDGES[:-1]+DT/2
...
'time_bin_size':DT*1000,'off_start':T0,'off_end':T1
```

iii. Step 12: "The reference defaults establish the core representation: trials are aligned over -2.5 to +2.5 seconds at 200 Hz (`dt=0.005 s`, 1000 bins)". Step 28: "it will use 5 ms bins over [-2.5, 2.5), causal 15-bin Gaussian smoothing".

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable. The input is the time axis the AI defined: the centres of the 1000 bins of the go-cue-aligned window. It is therefore implicitly derived from `bp.ev.goCue`, which defines the origin of that axis, plus the chosen window and bin size.

ii.
```python
EDGES=np.arange(T0,T1+DT/2,DT); TIME=EDGES[:-1]+DT/2
...
'input_names':['time from go cue']
```

iii. Step 28: the converter uses "5 ms bins over [-2.5, 2.5)", the go-cue-aligned window taken from `getDefaultParams.m` (step 12).

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond building the bin-centre vector once and broadcasting it to every trial. Every trial gets the identical `(1, 1000)` `float32` array of bin centres from −2.4975 s to +2.4975 s. Note the array is not copied per trial — the same object is appended repeatedly.

ii.
```python
ins.append(TIME[None,:].astype(np.float32))
```

iii. Not discussed separately in the trajectory; it follows from the window/binning choice above. The decoder task specified "Time from go cue onset in seconds (continuous, time-varying)", which the AI satisfied by emitting the continuous bin-centre time rather than a binary onset marker.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. `TIME` is `EDGES[:-1] + DT/2`, and spikes are counted into `EDGES` after subtracting the trial's go cue, so bin *k* of `input` and bin *k* of `neural` describe exactly the same 5 ms interval relative to the go cue. Zero crossing of the input falls between bins 499 and 500, the go cue.

ii.
```python
EDGES=np.arange(T0,T1+DT/2,DT); TIME=EDGES[:-1]+DT/2
...
h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT
...
ins.append(TIME[None,:].astype(np.float32))
```

iii. Implicit in the design: one grid is defined at module level and reused by every stream (step 28: "a common go-cue-centered grid").

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial flags of `obj.bp`: the instructed side `L` and `R`, and the outcome flags `hit` and `miss` (with `no` used to detect ignore trials). The lick actually performed is not recorded anywhere, so it is inferred from instructed side × outcome.

ii.
```python
L=np.asarray(b['L']).astype(bool); R=np.asarray(b['R']).astype(bool)
hit=np.asarray(b['hit']).astype(bool); miss=np.asarray(b['miss']).astype(bool)
no=np.asarray(b['no']).astype(bool)
```

iii. Step 17: "Behavioral encoding is now clear at the top level: instructed side (`L`/`R`), outcomes (`hit`, `miss`, `no`) ... actual lick direction is instructed side on hits, opposite side on misses, and none on ignores."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A three-way relabelling per trial, then tiled across all 1000 bins so the variable is stored as a (constant) time series. A hit means the animal licked the instructed port; a miss means it licked the other one; a `no` trial (or any trial that is neither hit nor miss) is `none`. Codes are left 0, right 1, none 2, matching `output_values = ['left','right','none']`.

ii.
```python
# Actual lick: instructed direction on hit, opposite on miss, none on no/ignore.
lick=2 if no[t] or (not hit[t] and not miss[t]) else (0 if (hit[t] and L[t]) or (miss[t] and R[t]) else 1)
...
const=np.vstack([np.full(TIME.size,lick),np.full(TIME.size,int(not aw[t])),np.full(TIME.size,outcome)])
outs.append(np.vstack([const,tc[j],pc[j],mc[j]]).astype(np.int8))
```

iii. Step 17: "A sensible required mapping is outcome hit→correct, miss→incorrect, no→ignore; actual lick direction is instructed side on hits, opposite side on misses, and none on ignores."

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The single per-trial flag `obj.bp.autowater`. The AI first considered deriving context from `bp.protocol.nums` or from the repository's block-numbering helper `getBlockNum_AltContextTask`, then confirmed from the Figure 8 code that `autowater` is the label the authors themselves use.

ii.
```python
aw=np.asarray(b['autowater']).astype(bool)
```

iii. Step 19: "WC trials are likely encoded by `autowater` rather than protocol number", after finding that `protocol.nums` was constant (3) across all trials of a session. Step 27 is the decisive one: "Repository code confirms that `bp.autowater` is directly used as the WC-versus-DR trial label (`triallabel = obj.bp.autowater` and classifier target `Y`) ... The block helper is only needed for block numbering/nonstationarity checks, not for basic context labels."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct binary relabelling, tiled across the 1000 bins: `autowater` → WC (0), everything else → DR (1), encoded as `int(not aw[t])`. `output_values[1] = ['WC','DR']`. The realised distribution is 31.8% WC / 68.2% DR.

ii.
```python
const=np.vstack([np.full(TIME.size,lick),np.full(TIME.size,int(not aw[t])),np.full(TIME.size,outcome)])
...
'output_values':[...,['WC','DR'],...]
```

iii. Step 29: the AI caught a polarity bug in its first run — "the current context numeric value is `autowater` (0=DR, 1=WC), but `output_values` is listed as WC then DR" — and at step 30 chose to fix the encoding rather than the labels: "encode WC as 0 and DR as 1 to match the declared `['WC','DR']` value order". The converter was re-run after this patch.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial flags of `obj.bp`: `hit`, `miss` and `no`. The same arrays already read for lick direction.

ii.
```python
hit=np.asarray(b['hit']).astype(bool); miss=np.asarray(b['miss']).astype(bool); no=np.asarray(b['no']).astype(bool)
```

iii. Step 17: "Behavioral encoding is now clear at the top level: ... outcomes (`hit`, `miss`, `no`)".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-way relabelling tiled across bins: `no` (or any trial that is neither hit nor miss) → ignore 2, `hit` → correct 1, otherwise (miss) → incorrect 0. `output_values[2] = ['incorrect','correct','ignore']`. Realised distribution 11.5% incorrect / 64.9% correct / 23.7% ignore, consistent with lick direction's 23.7% `none`.

ii.
```python
outcome=2 if no[t] or (not hit[t] and not miss[t]) else (1 if hit[t] else 0)
```

iii. Step 17: "outcome hit→correct, miss→incorrect, no→ignore". Step 3/17: ignore trials are retained rather than dropped "because explicitly requested as an output class", in deliberate departure from the paper, which omits them from its own analyses.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. `obj.traj[1]` — the **bottom** camera — landmark indices 0–3, i.e. `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue`, taken from `ts[:, 0:2, ids]` (x, y) and `ts[:, 2, ids]` (DeepLabCut likelihood), together with that camera's `frameTimes` and the trial's `bp.ev.goCue`. The indices are hard-coded rather than looked up in `featNames` (they are in fact stable across all 11 sessions, which I verified). The side camera (`traj[0]`, feature `tongue`) is not used. The code comment labelling `traj[1]` as the "Side camera" is wrong — `traj[1]` is the bottom view; the landmark names make the intent unambiguous.

ii.
```python
def trial_kin(traj, trial, go):
    # Side camera (cam 1): first four tongue landmarks, top/bottom paw landmarks.
    if len(traj)<2 or trial>=len(traj[1]['ts']):
        return ...
    cam=traj[1]; ts=np.asarray(cam['ts'][trial],float); ft=np.asarray(cam['frameTimes'][trial],float).ravel()-go
    ...
    tongue,tvis=speed([0,1,2,3]); paw,pvis=speed([4,5])
```
and inside `speed`:
```python
xy=ts[:,0:2,ids]; lk=ts[:,2,ids] if ts.shape[1]>2 else np.ones((len(ft),len(ids)))
```

iii. Step 24 established the layout — "CAM 1 FIRST_FEATURES [['top_tongue'], ['topleft_tongue'], ['bottom_tongue'], ['bottomleft_tongue'], ['top_paw'], ['bottom_paw'], ...]" and "TS0 (1792, 3, 10)". Step 28: "compute tongue/paw speed from camera-1 landmarks and visibility". The AI never justified using only the bottom view; it chose the camera that carries both the tongue and the paws so a single `trial_kin` call yields all three streams.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps. **(1)** Landmarks with likelihood < 0.9 are masked to NaN, and a frame counts as "tongue visible" if *any* of the four landmarks is at or above 0.9. **(2)** A single position trace is formed as the `nanmean` centroid of the currently visible landmarks. **(3)** Speed is `hypot(np.gradient(x), np.gradient(y))` — the frame-to-frame gradient in **pixels per frame**, not per second, and with no smoothing of position first. Speed is set to NaN on invisible frames. **(4)** Both the speed trace and the visibility trace are resampled onto the 1000 bin centres by linear interpolation (`np.interp`, despite the helper being named `interp_nearest`) with NaN/0 fill outside the frame range; the visibility trace is thresholded at 0.5. The speed interpolation bridges invisible gaps, but those bins are overwritten by the `not visible` class later, so the bridging does not leak in.

ii.
```python
def speed(ids):
    ids=[i for i in ids if i<ts.shape[2]]
    xy=ts[:,0:2,ids]; lk=ts[:,2,ids] if ts.shape[1]>2 else np.ones((len(ft),len(ids)))
    vis=np.any(lk>=.9,axis=1)
    # centroid of visible landmarks; preserve invisibility rather than paper's zero fill
    xy=np.where((lk>=.9)[:,None,:],xy,np.nan)
    pos=np.nanmean(xy,axis=2)
    # MATLAB findVelocity uses gradient per frame (pixel/frame), not /dt.
    vx=np.gradient(pos[:,0]); vy=np.gradient(pos[:,1]); sp=np.hypot(vx,vy)
    sp[~vis]=np.nan
    return interp_nearest(ft,sp,TIME), interp_nearest(ft,vis.astype(float),TIME,0)>=.5
```

iii. Step 6: "velocity is computed with MATLAB `gradient` on x/y positions; non-tongue features have median framewise displacement subtracted and missing velocities nearest-filled, while tongue missing values are set to zero." The AI deliberately departed from the zero-fill (in-code comment: "preserve invisibility rather than paper's zero fill") because the decoder spec demands an explicit `not visible` class. Step 29 notes the resulting all-NaN warning is expected: "A benign warning occurs when no tongue/paw landmarks are visible in a frame; those samples are intentionally retained as the explicit not-visible class."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A per-session threshold, exactly as the decoder spec requires: the 50th percentile of the tongue speed pooled over **all visible bins of all trials in that session** (`np.nanpercentile(x[vis],50)`). Visible bins at or above the threshold are class 1, visible bins below are class 0, and every bin where the tongue is not visible is class 2. The threshold is recorded per session in `metadata['session_info'][i]['tongue_median']`. Because the tongue is out of view in 92.7% of all bins, the realised distribution is 3.7% / 3.6% / 92.7%.

ii.
```python
def classes(x,vis):
    th=np.nanpercentile(x[vis],50) if np.any(vis) else np.nan
    z=np.full(x.shape,2,np.int8); z[vis]=(x[vis]>=th).astype(np.int8); return z,float(th)
tc,tth=classes(tongue,tvis); pc,pth=classes(paw,pvis); mc,mth=classes(mes,mvis)
```
with `'output_values'` entry `['below session median','at or above session median','not visible']`.

iii. Step 10: "Motion energy contains ... a movement threshold of 10, but the decoder task instead requires session-median discretization." Step 28: "apply session-wide medians over visible samples."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The frame times are taken straight from `traj[1]['frameTimes']` and only the trial's go cue is subtracted; the result is interpolated onto the same 1000 bin centres as the neural data. **The video-clock offset is not applied.** The repository corrects camera frame times with `findVideoOffset.m` (`vidshift = mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`) before subtracting the alignment event, in both `findPosition.m` and `loadMotionEnergy.m`. That offset is 0.49 s in the seven EKH3/JEB6/JEB7/JGR2/JGR3 sessions and 0.99 s in the four JEB19 sessions, so every camera-derived output is shifted late by 0.49–0.99 s relative to the spikes. The effect is visible in the converted file: the fraction of trials with a visible tongue stays below 1% until +0.5 s after the nominal go cue and only rises at +0.75 s, whereas the paper states responses "were typically registered within 300 ms of the go cue".

ii.
```python
cam=traj[1]; ts=np.asarray(cam['ts'][trial],float)
ft=np.asarray(cam['frameTimes'][trial],float).ravel()-go
...
return interp_nearest(ft,sp,TIME), interp_nearest(ft,vis.astype(float),TIME,0)>=.5
...
tv,tvok,pv,pvok,ft=trial_kin(o['traj'],int(t),go[t])
```

iii. Step 28: "The available video coordinates include likelihood in the third channel and absolute frame times within each trial, so they can be shifted by go cue and interpolated." The AI treated `frameTimes` as already being on the behaviour clock. `findVideoOffset`, `vidshift` and `bitcode` appear nowhere in the trajectory, so there is no justification for the omission — it was not a considered trade-off.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same bottom-camera tracking, landmark indices 4 and 5 — `top_paw` **and** `bottom_paw` — plus their likelihoods, that camera's `frameTimes`, and `bp.ev.goCue`. The two paws are pooled into one stream.

ii.
```python
tongue,tvis=speed([0,1,2,3]); paw,pvis=speed([4,5])
```

iii. Step 24 identified `top_paw` and `bottom_paw` as cam-1 features 5 and 6 (0-based 4 and 5); step 28 states the plan to "compute tongue/paw speed from camera-1 landmarks and visibility". The choice to pool both paws rather than use the more reliably tracked one is not discussed.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to the tongue and produced by the same `speed()` call: likelihood ≥ 0.9 defines visibility, the position is the `nanmean` centroid of whichever of the two paws is currently visible, speed is `hypot` of the per-frame `np.gradient` of that centroid (pixels/frame, unsmoothed), NaN on invisible frames, then linear interpolation onto the 1000 bin centres. No normalisation, no baseline-displacement subtraction (the repository's `findVelocity.m` subtracts the median frame-wise displacement for non-tongue features), and no nearest-fill of missing values (the repository nearest-fills non-tongue features; the AI keeps the gap and encodes it as class 2). Because the centroid is taken over *whichever* paws are visible, the position jumps discontinuously between `top_paw` and the two-paw midpoint whenever `bottom_paw` flickers across the 0.9 likelihood cut, which injects spurious velocity transients.

ii.
```python
vis=np.any(lk>=.9,axis=1)
xy=np.where((lk>=.9)[:,None,:],xy,np.nan)
pos=np.nanmean(xy,axis=2)
vx=np.gradient(pos[:,0]); vy=np.gradient(pos[:,1]); sp=np.hypot(vx,vy)
sp[~vis]=np.nan
```

iii. Step 6 records that the AI read `findVelocity.m` and saw the baseline-subtraction and nearest-fill rules for non-tongue features; it nevertheless applied one common code path to both features, with the in-code justification "preserve invisibility rather than paper's zero fill" (step 29: the not-visible class is "intentionally retained").

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same `classes()` helper and same rule: the 50th percentile of paw speed over all visible bins of the session splits classes 0 and 1, and bins with no visible paw are class 2. The per-session threshold is stored as `paw_median`. Realised distribution 43.0% / 43.0% / 14.1%.

ii.
```python
tc,tth=classes(tongue,tvis); pc,pth=classes(paw,pvis); mc,mth=classes(mes,mvis)
...
session_info.append({... ,'paw_median':pth, ...})
```

iii. As 7-c — step 28: "apply session-wide medians over visible samples", following the decoder spec's per-session 50th-percentile rule rather than the paper's manual movement threshold.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, and with the same defect: `frameTimes − goCue[trial]` with no `vidshift` correction, then linear interpolation onto the shared 1000-bin grid. The 14.1% `not visible` rate for the paw is largely an artefact of this — the uncorrected frames cover only about −2.0 to +2.5 s of the window, so roughly the first 100 bins of every trial fall outside the camera's apparent coverage and are marked class 2.

ii.
```python
ft=np.asarray(cam['frameTimes'][trial],float).ravel()-go
...
def interp_nearest(t,y,target,fill=np.nan):
    ...
    return np.interp(target,t,y,left=fill,right=fill)
```

iii. Same as 7-d: no consideration of the bitcode-derived video offset appears in the trajectory.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone file `motionEnergy_<anm>_<date>.mat` sitting beside each data structure, read with `scipy.io.loadmat(..., squeeze_me=True, struct_as_record=False)` and unwrapped one level as `['me'].data` — a list of one variable-length trace per trial, one value per camera frame. The in-object copy `obj.me` is not used. The frame times used to place those values in time come from `obj.traj[1]` (bottom camera); the repository's `loadMotionEnergy.m` uses `obj.traj{1}`, i.e. the *side* camera (in practice the two cameras share frame times for these sessions). If the file is absent the whole session's motion energy becomes class 2.

ii.
```python
mef=os.path.join(ROOT,f'motionEnergy_{mouse}_{day}.mat'); me=None
if os.path.exists(mef): me=loadmat(mef,squeeze_me=True,struct_as_record=False)['me'].data
```

iii. Step 8: "Motion-energy files use an older MAT format and should be read with `scipy.io.loadmat`." Step 10: "Motion energy contains one object per trial and a movement threshold of 10." Step 11: "Motion energy is a variable-length 1D trace per trial with matching trial count."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling. The value is already one number per frame (the paper reduces each frame's per-pixel motion energy to its 99th percentile upstream), so the trace is simply truncated to the shorter of the trace and the frame-time vector and linearly interpolated onto the 1000 bin centres, with NaN outside the frame range. Bins where the interpolation returns NaN are marked "no video".

ii.
```python
if me is not None and ft is not None and t<len(me):
    mv=np.asarray(me[t],float).ravel(); m=min(len(mv),len(ft))
    z=interp_nearest(ft[:m],mv[:m],TIME); mes.append(z);mvis.append(np.isfinite(z))
else: mes.append(np.full(TIME.size,np.nan));mvis.append(np.zeros(TIME.size,bool))
```

iii. Step 10 and step 28: the AI treated motion energy as a ready-made per-frame scalar needing only alignment and interpolation ("align motion energy using camera frame times"), and explicitly rejected the file's own `moveThresh = 10` in favour of the spec's session median (step 10).

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same `classes()` helper: the 50th percentile of motion energy over all bins with finite interpolated values in the session splits classes 0 and 1; bins with no video (outside the frame range, or a missing motion-energy file) are class 2. The threshold is stored as `motion_energy_median`. Realised distribution 43.0% / 43.1% / 13.9%. The stored `output_values` correctly name the third class "no video" here rather than "not visible".

ii.
```python
mc,mth=classes(mes,mvis)
...
['below session median','at or above session median','no video']
```

iii. Step 10: "Motion energy contains one object per trial and a movement threshold of 10, but the decoder task instead requires session-median discretization."

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It reuses the very same `ft` vector returned by `trial_kin` — bottom-camera `frameTimes` minus the trial's go cue, **without the video offset** — and interpolates onto the shared 1000-bin grid. So motion energy carries the same 0.49 s / 0.99 s late shift as the tongue and paw, and the same ≈14% of bins falsely marked "no video" because the uncorrected frame coverage starts about 0.5 s into the window.

ii.
```python
tv,tvok,pv,pvok,ft=trial_kin(o['traj'],int(t),go[t])
...
if me is not None and ft is not None and t<len(me):
    mv=np.asarray(me[t],float).ravel(); m=min(len(mv),len(ft)); z=interp_nearest(ft[:m],mv[:m],TIME)
```

iii. As 7-d/8-d: the trajectory contains no mention of `findVideoOffset`/`vidshift`, even though `loadMotionEnergy.m` (which the AI's plan at step 28 says it followed for "camera frame times") applies it on the very line that interpolates motion energy.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Defensively, and always by keeping the trial and marking the gap rather than by filling it in or dropping data. Specific guards: (a) a missing `traj` entry, fewer than two cameras, or a trial index beyond the tracking list returns all-NaN velocity and all-False visibility, so the trial becomes 1000 `not visible` bins; (b) a `ts` array whose shape does not match the frame-time vector triggers the same fallback; (c) `interp_nearest` drops non-finite times and values, returns all-`fill` if fewer than two valid samples remain, and sorts by time before interpolating; (d) values outside the frame range get `left`/`right` fill (NaN, or 0 for visibility) rather than edge extrapolation; (e) a missing `motionEnergy_*.mat` file, or a trial index beyond the motion-energy list, yields an all-NaN trace → class 2; (f) motion-energy trace and frame-time vector are truncated to their common length; (g) spikes in trials with a non-finite go cue are skipped; (h) `classes()` guards against a session with no visible frames at all. NaN never reaches the saved arrays — every NaN becomes the trailing category. The one filter the AI does *not* implement is the reference's "trial runs past the end of the recording" check; I verified that no trial in these 11 sessions is all-zero across all units, so the omission is harmless here.

ii.
```python
def interp_nearest(t,y,target,fill=np.nan):
    t=np.asarray(t,float).ravel(); y=np.asarray(y,float).ravel()
    ok=np.isfinite(t)&np.isfinite(y)
    if ok.sum()<2:return np.full(target.shape,fill,float)
    t=t[ok]; y=y[ok]; ix=np.argsort(t); t=t[ix]; y=y[ix]
    return np.interp(target,t,y,left=fill,right=fill)

def trial_kin(traj, trial, go):
    if len(traj)<2 or trial>=len(traj[1]['ts']):
        return np.full(TIME.size,np.nan),np.zeros(TIME.size,bool),np.full(TIME.size,np.nan),np.zeros(TIME.size,bool),None
    ...
    if ts.ndim!=3 or ts.shape[0]!=ft.size:return ...
...
def classes(x,vis):
    th=np.nanpercentile(x[vis],50) if np.any(vis) else np.nan
```

iii. Step 29: "A benign warning occurs when no tongue/paw landmarks are visible in a frame; those samples are intentionally retained as the explicit not-visible class." Step 28: "mark missing video appropriately" / step 23: "Trial maps include `haveEphys` and `haveVid`, enabling explicit missing-stream handling" (in the end the AI used structural guards rather than those flags).

## 11-a. What are the most time-consuming steps of the code?

i. Two things dominate. **(1)** `mat73.loadmat` reads each v7.3 file eagerly and in full — including `clu.spkWavs`, `clu.tm`, all ten tracked features of both cameras and the `sglx` index arrays — none of which the conversion needs. The AI hit this directly: its all-session scan at steps 20–22 had to be abandoned as too slow ("`mat73` eagerly loads all spike and video arrays for every session"). **(2)** The spike binning/smoothing double loop: for every unit and every trial with spikes it does a boolean mask `tm[tr==t]` over the unit's whole spike list, a separate `np.histogram`, and a separate `smooth_rates` call that rebuilds the 15-tap Gaussian kernel and dispatches `np.apply_along_axis` on a single 1000-sample row. In the `JEB19` sessions that is 436–567 clusters × ~250 trials ≈ 10⁵ kernel constructions and 10⁵ single-row `apply_along_axis` calls per session. The full conversion took several minutes per run (the AI ran it twice, waiting across steps 29–31).

ii.
```python
o=mat73.loadmat(f)['obj']
...
for ui,(tr,tm) in enumerate(units):
    for t in np.unique(tr):
        j=keepmap.get(int(t));
        if j is None or not np.isfinite(go[t]): continue
        h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT
        rates[ui,j]=smooth_rates(h[None,:])[0]
```

iii. The AI never documented runtime in `DECISIONS`-style terms, but the trajectory shows it was aware of the loading cost (steps 20–22: "The full-object scan is too slow because `mat73` eagerly loads all spike and video arrays for every session ... A more efficient approach is to use h5py and dereference only the needed small datasets"). It used `h5py` for its exploratory scans but reverted to `mat73` for the converter itself, and at step 27 said the design was chosen "to control output size" rather than runtime.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three. **(1)** The unit × trial spike loop is the big one: the whole thing is a single `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, EDGES])` per unit (what the reference does), removing the inner Python loop and the repeated `tr==t` scans entirely. **(2)** `smooth_rates` is called once per (unit, trial) on a `(1,1000)` array; it could be called once per unit on the whole `(n_trials, 1000)` matrix — or once per session on the `(n_units·n_trials, 1000)` matrix — and the kernel should be built once at module level instead of on every call. `np.apply_along_axis` over a one-row array is pure overhead; `scipy.ndimage.convolve1d` (already imported but never used) would vectorise it. **(3)** The per-trial `trial_kin` loop re-derives `cam = traj[1]`, re-reads `ts`/`frameTimes` and re-filters the landmark list for every trial; the landmark selection and likelihood mask are trial-invariant work. This loop cannot be fully vectorised because trials have different frame counts, but the per-trial overhead could be cut substantially.

ii.
```python
for ui,(tr,tm) in enumerate(units):
    for t in np.unique(tr):
        ...
        h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT
        rates[ui,j]=smooth_rates(h[None,:])[0]
```
```python
def smooth_rates(x):
    k=gaussian(15, std=(15-1)/(2*2.5), sym=True); k[:7]=0; k/=k.sum()   # rebuilt every call
    return np.apply_along_axis(lambda z:np.convolve(z,k,mode='same'),1,x)
```

iii. Not discussed in the trajectory. The AI's stated optimisation concerns were memory and pickle size (step 27: "To control output size, the converter will store float32 neural matrices"), not loop structure.

## 11-c. What processing does the code repeat multiple times?

i. **The Gaussian kernel** is constructed from scratch on every one of the ~10⁵–10⁶ `smooth_rates` calls. **The spike-mask scan** `tm[tr==t]` walks the unit's entire spike array once per trial, so a unit with *T* active trials is scanned *T* times. **`np.unique(tr)`** is recomputed per unit (unavoidable) but `keepmap` lookups repeat the same `int()` conversion. **The camera indirection** — `traj[1]`, the `ids` filtering `[i for i in ids if i<ts.shape[2]]`, and the likelihood threshold `lk>=.9` (computed twice inside `speed`, once for `vis` and once for the `np.where` mask) — is repeated for every trial and for both the tongue and the paw call. **Visibility interpolation** is run as a separate `np.interp` pass alongside the speed interpolation. Nothing genuinely expensive is computed twice at the session level: each file is read once, the motion-energy file is read once, and `TIME`/`EDGES` are module-level constants shared by every stream.

ii.
```python
def smooth_rates(x):
    k=gaussian(15, std=(15-1)/(2*2.5), sym=True); k[:7]=0; k/=k.sum()
```
```python
vis=np.any(lk>=.9,axis=1)
xy=np.where((lk>=.9)[:,None,:],xy,np.nan)
```

iii. Not discussed in the trajectory.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. **Loading**: `mat73.loadmat` materialises the whole `obj` tree — `clu.spkWavs` (spike waveforms), `clu.tm`, `clu.site`, `clu.quality`, `obj.sglx`, `obj.me`, `obj.trials`, the six side-camera features and the four unused bottom-camera features (`lickport`, `jaw`, `top_nostril`, `bottom_nostril`) — all of which are thrown away. On the JEB19 files this is the single largest cost. **Neural**: firing rates are computed and smoothed for *every* cluster before the >1 Hz test, so in `JEB19_2023-04-19` 567 clusters are fully binned and smoothed to keep 132 — and, because cluster quality is never checked, 435 of the 567 are clusters the authors labelled `garbage`, whose work is wasted twice over (either discarded by the rate cut, or kept when it should not be). The `rates` array is allocated at full cluster count `(n_units_raw, n_trials, 1000)` before filtering. **Memory**: `del o,rates` at the end of each session does not free the rate buffer, because `ns.append(rates[:,j,:])` stores non-contiguous *views* that keep the post-filter base array alive for the whole run. **Kinematics**: the speed trace is interpolated across invisible gaps (`interp_nearest` drops NaNs before `np.interp`), and every one of those interpolated values is then overwritten by class 2; the same applies to motion energy. **Metadata**: the per-session thresholds `tongue_median`/`paw_median`/`motion_energy_median` are stored but unused downstream (harmless, and arguably useful provenance).

ii.
```python
o=mat73.loadmat(f)['obj']
...
rates=np.zeros((len(units),len(keep),TIME.size),np.float32)   # all clusters, pre-filter
...
use=rates.mean(axis=(1,2))>1.0; rates=rates[use]
...
ns.append(rates[:,j,:])     # views keep the base array alive
...
del o,rates                 # frees the name, not the buffer
```

iii. Step 27: the AI's only stated efficiency decision was output size — "To control output size, the converter will store float32 neural matrices over the repository's 1000-point grid". The resulting pickle is 1.03 GB. Nothing else about wasted work is discussed.
