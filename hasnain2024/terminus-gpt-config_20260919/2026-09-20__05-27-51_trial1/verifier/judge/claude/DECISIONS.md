# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a list of 12 `(subject, date, probe)` tuples, all from `/app/data/Ephys_Behavior`, transcribed from the reference `Scripts/Figure 8/Figure8a_thru_c.m` loader calls (`loadJEB6_ALMVideo`, `loadJEB7_…`, `loadEKH1_…`, `loadEKH3_…`, `loadJGR2_…`, `loadJGR3_…`, `loadJEB19_ALMVideo`), including each loader's selected probe. It deliberately excludes the other three data cohorts in `/app/data` (`RandomizedDelay_Ephys_Behavior`, `DelayInhibition_BilatMC_Behavior`, `GoCueInhibition_BilatMC_Behavior`) and the 13 remaining `Ephys_Behavior` sessions (JEB13/JEB14/JEB15), which are not loaded by the Figure 8 two-context script. Each session file is opened once with `h5py` (MATLAB v7.3), and only the needed HDF5 references are dereferenced (`obj/bp`, the selected `obj/clu` probe entry, `obj/traj`); motion energy is read separately with `scipy.io.loadmat`. There is no v5/`scipy` fallback for `data_structure` files — all 12 chosen files are v7.3. Result: 12 sessions, 7 subject IDs, 3,626 native → 3,491 retained trials, 515 units.

ii.
```python
ROOT=Path('/app/data/Ephys_Behavior')
SESSIONS=[('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),('JGR2','2021-11-16',1),
 ('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),('JEB19','2023-04-18',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-20',1),('JEB19','2023-04-21',1)]
...
def convert_session(subject,date,probe,show=False):
    t0=time.time(); path=ROOT/f'data_structure_{subject}_{date}.mat'
    with h5py.File(path,'r') as f:
        b=f['obj/bp']; ntr=int(np.asarray(b['Ntrials'][()]).squeeze())
```

iii. From CONVERSION_NOTES Step 4/5: "The relevant source subset is the 12 two-context electrophysiology sessions (six mice, 522 pre-rate-filter units), because behavioral context WC/DR is required. Randomized-delay sessions cannot supply WC labels even though their schema includes `goCue`." The AI further argues this subset reproduces the paper's two-context statistics exactly: 12 sessions and, after the >1 Hz filter, "**214 well-isolated single units**, exactly the paper value." It documents that the loaders name seven animal IDs while the manuscript says six mice, and keeps seven rather than dropping an animal. Targeted HDF5 traversal is justified as a speed-up after "initial `mat73` recursive loading stalled on large spike/video branches."

## 1-b. How are the data split into subjects?

i. The subject is the animal ID from the hard-coded session tuple (equivalently the filename prefix). `subjects` is built in first-encounter order as sessions are processed, and `subject_idx` is that list index per session. 7 subjects for 12 sessions (JEB6:1, JEB7:2, EKH1:1, EKH3:1, JGR2:2, JGR3:1, JEB19:4). The AI verified that `obj.meta` animal IDs are not relied upon.

ii.
```python
for i,(sub,date,probe) in enumerate(sessions):
    n,x,y,info,cont=convert_session(sub,date,probe,args.show_processing)
    ...
    if sub not in subjects:subjects.append(sub)
    sidx.append(subjects.index(sub));bridx.append(np.zeros(n[0].shape[0],dtype=np.int64))
```

iii. Notes Step 5 mapping table: "Loader animal ID → `subjects`, `subject_idx`; unique ID and per-session index; `load<animal>_ALMVideo`. Seven released IDs; manuscript reports six mice, documented discrepancy." The AI explicitly chose not to merge or drop animals to force the paper's count of six.

## 1-c. How are the data split into sessions?

i. One session = one `(animal, date, probe)` entry = one `data_structure_<anm>_<date>.mat` file = one element of `neural`/`input`/`output`/`brain_region_idx` and one row of `subject_idx`. Two-probe sessions do not occur in this subset, so each session uses the single probe named by its loader. All 12 sessions come from the fixed-delay two-context task folder, so no cross-folder path resolution is implemented.

ii.
```python
sessions=SESSIONS[:2] if args.sample else SESSIONS
for i,(sub,date,probe) in enumerate(sessions):
    n,x,y,info,cont=convert_session(sub,date,probe,args.show_processing)
    neural.append(n);inputs.append(x);outputs.append(y);infos.append(info)
...
info={'session_id':f'{subject}_{date}','native_trials':ntr,'retained_trials':len(keep_trials),
      'excluded_stim_trials':int(stim.sum()),'n_neurons':len(unit_counts),...,'probe':probe}
```

iii. Notes Step 5 Key Decision 1: "Use the 12 available sessions selected by Figure 8 ALM/video loaders … This matches the paper's session count and exact 214 single-unit statistic. Use each loader's selected probe." Sessions the loaders comment out, and `JEB7 2021-04-17` (absent from `/app/data`), are excluded and documented.

## 1-d. How are the data split into trials?

i. A trial is one index into the per-trial `obj.bp` arrays; `bp.Ntrials` sets the count and every behavioral flag is read as a flat vector indexed by that trial number. Spike times carry their own `trial` index (1-based, converted to 0-based and range-checked), and `obj.traj[view]` holds one cell per native trial, so trial boundaries are never reconstructed. The surviving native trial indices are kept in `keep_trials`, and a `remap` array maps native index → row in the output matrices, so neural, video and behavioral rows stay in register.

ii.
```python
ntr=int(np.asarray(b['Ntrials'][()]).squeeze())
get=lambda k:vec(b[k],bool)
R,L,hit,miss,no,early,aw=map(get,['R','L','hit','miss','no','early','autowater'])
go=vec(b['ev/goCue'],float)
...
keep_trials=np.flatnonzero((~stim)&np.isfinite(go))
...
tr=np.asarray(f[rr][()]).ravel(order='F').astype(int)-1
valid=(tr>=0)&(tr<ntr); tr=tr[valid]; tm=tm[valid]
remap=np.full(ntr,-1,int);remap[keep_trials]=np.arange(len(keep_trials))
```

iii. Notes Step 5 Key Decision 2: "Require native trial index correspondence; do not invent trials." Step 10 Check 5 reports verification that "every native trial equals retained plus excluded stimulation" and that the two trajectory views have one entry per native trial.

## 1-e. How are trials filtered based on quality controls?

i. Exactly two conditions: the trial must not be a photostimulation trial (`bp.stim.enable`) and its go cue must be finite. 135 stim trials are dropped across the 12 sessions (3,626 → 3,491). Early-lick trials (`bp.early`, 389 trials ≈ 11%) are **kept** — the `early` flag is loaded but never used. Ignore/no-response and miss trials are kept on purpose because they are required output classes. No filter for trials past the end of the recording exists (independent checking confirms it is not needed here: in all 12 sessions the last spiking trial equals `Ntrials`). No minimum-trial or minimum-unit session filter is applied (all sessions have ≥227 trials and ≥27 units).

ii.
```python
stim=np.zeros(ntr,bool)
if isinstance(b.get('stim'),h5py.Group) and 'enable' in b['stim']:stim=vec(b['stim/enable'],bool)
keep_trials=np.flatnonzero((~stim)&np.isfinite(go))
```

iii. Notes Step 5 Key Decision 2: "Exclude 135 stimulation trials because every Figure 8 context condition requires `~stim.enable`. Retain hit, miss, no-response, and early trials with valid go cues because the requested output explicitly includes incorrect and ignore. All selected sessions have finite go cues." Step 3 adds: "The paper omits early-lick and ignore trials from several analyses, but the requested decoder explicitly includes outcome `ignore`; therefore ignore trials must be retained."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The selected probe's spike-sorted clusters, `obj.clu{probe}`: per-cluster `trial` (1-based trial of each spike), `trialtm` (spike time relative to trial start) and `quality` (manual curation string). `bp.ev.goCue` supplies the alignment time. Waveforms, channels and global spike times are not read.

ii.
```python
clu=f['obj/clu']; cg=f[clu[()].ravel(order='F')[probe-1]]
qualities=np.array([chars(f,r).lower() for r in cg['quality'][()].ravel(order='F')])
trialrefs=cg['trial'][()].ravel(order='F'); tmrefs=cg['trialtm'][()].ravel(order='F')
for q,rr,rt in zip(qualities,trialrefs,tmrefs):
    tr=np.asarray(f[rr][()]).ravel(order='F').astype(int)-1
    tm=np.asarray(f[rt][()]).ravel(order='F').astype(float)
```

iii. Notes Step 5 mapping table: "Selected probe `clu[].trial`, `clu[].trialtm` → `neural`; subtract each trial's `bp.ev.goCue` … Reference: `alignSpikes`, `getSeq`, `processData`, `removeLowFRClusters`." Step 1 records that these are extracellular spikes, so "no delta-F/F calculation applies".

## 2-b. How is the `neural` data processed?

i. Per unit and per retained trial, go-cue-aligned spike times are histogrammed into 500 bins of 10 ms spanning −2.5 to +2.5 s, divided by the bin width to give Hz, stacked as (trial, neuron, time) and then smoothed along time only with a **centered 15-bin boxcar** (`uniform_filter1d(size=15, mode='nearest')`, i.e. a 150 ms moving average). No normalization, z-scoring or baseline subtraction; stored as `float32` Hz. Units from the one selected probe form the population.

ii.
```python
def smooth_rates(x,width=15):
    # Reference mySmooth is a centered boxcar; nearest avoids artificial zero edges.
    return uniform_filter1d(x,size=width,axis=1,mode='nearest').astype(np.float32)
...
for old in np.unique(tr[use]):
    z=tm[(tr==old)]-go[old]
    mats[remap[old]]=np.histogram(z,EDGES)[0]/DT
unit_counts.append(mats);unit_rates.append(rate)
...
# Stack directly as trial x neuron x time and smooth only the time axis.
neural=np.stack(unit_counts,axis=1).astype(np.float32)
neural=uniform_filter1d(neural,size=15,axis=2,mode='nearest').astype(np.float32)
```

iii. Notes Step 4/5: "Correct reference population parameters are go-cue alignment, no time warping, −2.5 to +2.5 s, 10 ms bins, smoothing parameter 15, and mean firing rate >1 Hz"; Key Decision 4: "Use firing rate rather than raw counts, matching PSTH processing … Smoothing width 15 is applied in bins with edge-aware normalization." The in-code comment asserts the reference `mySmooth` is a centered boxcar (it is in fact a *causal* `gausswin(15)`, with the first half of the kernel zeroed). Step 10 lists a fixed bug: an earlier reshape-based smoothing "could scramble trial/unit correspondence", replaced by direct `(trial, neuron, time)` stacking.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two stages. (1) Quality label, lower-cased and whitespace-stripped, must not be in `{'garbage', '' (unlabeled), 'noisy', 'real?'}` — so `poor`, `fair`, `good`, `great`, `excellent` and `multi` units are all retained. (2) Mean firing rate must be strictly >1 Hz, computed as all in-window spikes of that unit divided by `Ntrials × 5 s`, i.e. over **all native valid trials** (before stim-trial exclusion). 515 units survive across 12 sessions (27–67 per session). Only the loader-selected probe is read.

ii.
```python
ARTIFACT={'garbage','','noisy','real?'}
...
# Reference low-FR curation is based on all native valid trials, before
# the decoder-specific exclusion of stimulation trials.
aligned_all=tm-go[tr]
rate=np.sum((aligned_all>=TMIN)&(aligned_all<TMAX))/(ntr*(TMAX-TMIN))
if q not in ARTIFACT and rate>1:
```

iii. Notes Step 5 Key Decision 3: "Exclude clear artifacts (`garbage`, blank/unlabeled, `noisy`, and questionable `real?`), then require mean firing rate strictly >1 Hz over the reference five-second window … Retain poor/fair/good/great/excellent/multi categories because population analyses use all units rather than only well-isolated single units." This tracks `findClusters.m`'s `'all'` branch (which drops `garbage`, `gabrga`, `noisy`, `real?`) plus the paper's ">1 Hz" criterion, and the AI reports reproducing the paper's 214 well-isolated single units exactly as a sanity check. The paper's 522 total is explained as a pre-rate-filter count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction: each spike's within-trial time minus that trial's `bp.ev.goCue`, then histogramming into the fixed −2.5…2.5 s grid. `trialtm` is already relative to trial start on the behavior clock, so no clock correction or interpolation is used. Spikes outside the window simply fall outside the histogram edges.

ii.
```python
go=vec(b['ev/goCue'],float)
...
for old in np.unique(tr[use]):
    z=tm[(tr==old)]-go[old]
    mats[remap[old]]=np.histogram(z,EDGES)[0]/DT
```

iii. Notes Step 1: "`alignSpikes` stores `trialtm_aligned = trialtm - event`; therefore go-cue alignment must use the session behavioral event corresponding to go cue, preserving negative pre-cue and positive post-cue times." Step 10 Check 3: "source within-trial times subtract `bp.ev.goCue`, exactly matching `alignSpikes`; all 3,626 native go cues are finite." An independent raw-file `np.allclose` spot check of one unit/trial is reported (I reproduced this check independently and it passes exactly).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins (`DT=.01`), 500 bins per trial over a fixed [−2.5, +2.5] s window; `time_bin_size` is recorded as 10.0 ms. Binning happens once, directly from spike times (no rebinning of an intermediate representation). The same 500-bin grid is used for the input and for all video-derived outputs, which are resampled onto it by linear interpolation. No time warping.

ii.
```python
DT=.01; TMIN=-2.5; TMAX=2.5
EDGES=np.arange(TMIN,TMAX+DT/2,DT); TIME=(EDGES[:-1]+EDGES[1:])/2
```

iii. Notes Step 4: "Figure 8a-c population context analysis uses 10 ms bins and smoothing parameter 15 around go cue (Figure 8d uses 5 ms only for its single-unit/subspace analysis) … Figure 8d's 5 ms/single-unit settings are not used for this population-decoder conversion." Step 10 lists this as a resolved item: "Figure 8 bin discrepancy: selected 10 ms population settings from Figure 8a-c rather than 5 ms single-unit Figure 8d settings." The window is taken from the reference `tmin=-2.5`, `tmax=2.5`, and no warping "because the decoder task requests direct go-cue temporal alignment."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing from the raw files — it is defined by the conversion as the centers of the analysis bins, i.e. time relative to `bp.ev.goCue` by construction. Every trial in every session gets the identical vector, −2.495 … 2.495 s.

ii.
```python
EDGES=np.arange(TMIN,TMAX+DT/2,DT); TIME=(EDGES[:-1]+EDGES[1:])/2
...
return ..., [TIME[None,:].astype(np.float32).copy() for _ in keep_trials], ...
```

iii. Notes Step 5 mapping table: "Bin centers → `input[0]`: `-2.495, ..., 2.495` seconds repeated per trial; reference `getSeq`; decoder input is continuous time from go cue, shape `(1,500)`." This mirrors `getSeq.m`'s `obj.time = edges + params.dt/2`.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing bin centers and tiling them once per trial as a `(1, 500)` float32 array. The value is kept continuous (not binarized), as the Decoder Task specifies a continuous time-varying input.

ii.
```python
[TIME[None,:].astype(np.float32).copy() for _ in keep_trials]
```

iii. The AI's Step 10 Check 3 states: "Input: bin centers exactly match `edges + dt/2`; direct `np.allclose` passed." The decoder-format spec calls for a continuous, time-varying "time from go cue onset in seconds", so no further transform is applied.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid: `TIME` holds the centers of the same `EDGES` used to histogram the spikes, so input bin *k* and neural bin *k* are the same 10 ms interval relative to the go cue. Verified in the saved pickle: input spans −2.495 to 2.495 with 500 bins, matching neural `(n_neurons, 500)`.

ii.
```python
EDGES=np.arange(TMIN,TMAX+DT/2,DT); TIME=(EDGES[:-1]+EDGES[1:])/2
...
mats[remap[old]]=np.histogram(z,EDGES)[0]/DT     # neural on EDGES
[TIME[None,:] ...]                                # input = centers of EDGES
```

iii. Implied by the single shared grid; the AI's Step 10 check confirmed the analytical bin-center vector matches, and the metadata records `off_start=-2.5`, `off_end=2.5`, `time_bin_size=10.0`.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial `obj.bp` flags: the instructed side `R` (and its complement `L`, loaded but unused), and the outcome flags `hit`, `miss`, `no`. The actual licked port is not recorded, so it is inferred from instructed side × outcome.

ii.
```python
R,L,hit,miss,no,early,aw=map(get,['R','L','hit','miss','no','early','autowater'])
```

iii. Notes Step 5 Key Decision 7: "R/L encode instructed direction, so misses must be inverted to represent actual lick direction. `no` maps to none. This avoids incorrectly treating instructed side as behavioral choice."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A per-trial relabeling, then broadcast across all 500 bins: no-response trials → `none` (2); hits → the instructed side (`R`→right 1, else left 0); misses → the opposite of the instructed side; anything else → `none`. Codes are left 0, right 1, none 2, matching `output_values` `['left','right','none']`. Full-data distribution: left 0.417, right 0.376, none 0.207.

ii.
```python
for j,tr in enumerate(keep_trials):
    if no[tr]:lick=2
    elif hit[tr]:lick=1 if R[tr] else 0
    elif miss[tr]:lick=0 if R[tr] else 1
    else:lick=2
    ...
    y=np.vstack([np.full(len(TIME),lick),...]).astype(np.int64)
```

iii. Notes Step 5 Key Decision 5: per-trial labels are "repeated over 500 time points so all six outputs form a single categorical `(6,500)` array. This is semantically constant per trial and compatible with time-varying video labels." Step 10 reports a raw-data spot check of explicit hit, miss and no-response trials matching the converted labels.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The single per-trial flag `obj.bp.autowater`: true on water-cued (WC) trials where a reward is delivered from a random port with no cues, false on delayed-response (DR) trials.

ii.
```python
R,L,hit,miss,no,early,aw=map(get,['R','L','hit','miss','no','early','autowater'])
```

iii. Notes Step 4/5 Key Decision 6: "`autowater` is the code's WC/AW context indicator, not merely an outcome flag; its alternating blocks match the paper. `~autowater` is DR." This is read off the Figure 8 condition strings (`hit&~stim.enable&~autowater` = DR, `…&autowater` = WC).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabeling, broadcast over time: WC = 0 where `autowater`, DR = 1 otherwise. Distribution: WC 0.322, DR 0.678 (per session 0.22–0.38 WC), consistent with a session of ~100 DR trials followed by alternating blocks.

ii.
```python
context=0 if aw[tr] else 1
...
'output_values':[...,['WC','DR'],...]
```

iii. Codes follow the Decoder Task ordering (WC, DR). Notes Step 9 records the WC/DR split as "Sensible" against the paper's block design.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The per-trial `obj.bp` flags `hit` and `miss` (with `no` loaded and used for the lick-direction branch but not needed for outcome, since "neither hit nor miss" ⇒ ignore).

ii.
```python
R,L,hit,miss,no,early,aw=map(get,['R','L','hit','miss','no','early','autowater'])
...
outcome=1 if hit[tr] else (0 if miss[tr] else 2)
```

iii. Same source flags already used for lick direction; Notes Step 5 mapping table: "`hit`, `miss`, `no`, `early` → `output[2]` outcome: miss=incorrect, hit=correct, no/otherwise early nonresponse=ignore."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Three-class relabeling broadcast over time: correct = 1 on hits, incorrect = 0 on misses, ignore = 2 otherwise. Distribution: incorrect 0.113, correct 0.680, ignore 0.207. Ignore trials are retained as a class rather than dropped.

ii.
```python
outcome=1 if hit[tr] else (0 if miss[tr] else 2)
y=np.vstack([...,np.full(len(TIME),outcome),...]).astype(np.int64)
```

iii. Codes follow the Decoder Task (incorrect 0, correct 1, ignore 2). Notes Step 5 Key Decision 2 / Step 3: the paper omits ignore trials from its analyses, but "the requested decoder explicitly includes outcome `ignore`; therefore ignore trials must be retained when neural/video alignment is valid and encoded as an output class."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DeepLabCut tracking in `obj.traj`, **side camera only** (`views[0]`), feature named `tongue`: its x, y and likelihood per frame from `traj.ts`, plus `traj.frameTimes` and `bp.ev.goCue`. Feature names are resolved by name per trial (`featNames`), not by fixed column. The bottom camera's `top_tongue` view is not used, and `obj.sglx.bitcode`/`bp.ev.bitStart` (the video-clock reference) are not read.

ii.
```python
def feature_trial(f,view,trial,name):
    names_obj=f[view['featNames'][()].ravel(order='F')[trial]]
    names=[chars(f,r) for r in names_obj[()].ravel(order='F')]
    if name not in names:return None,None,None
    a=deref_array(f,view['ts'],trial)
    # MATLAB (frames, coordinates, features) appears reversed by h5py: feature,coord,frame.
    fi=names.index(name); x=np.asarray(a[fi,0,:],float); y=np.asarray(a[fi,1,:],float)
    likelihood=np.asarray(a[fi,2,:],float) if a.shape[1]>2 else np.ones_like(x)
    ft=deref_array(f,view['frameTimes'],trial).ravel(order='F').astype(float)
...
ft,pos,lk=feature_trial(f,views[0],tr,'tongue');tongue[j]=velocity_on_grid(ft,pos,lk,go[tr])
```

iii. Notes Step 5: "tongue velocity needs a named tongue landmark from the side view", citing `getKinematicsFromVideo`, `findVelocity`, `findDLCFeatIndex`; the reference `params.traj_features` lists `tongue` as a view-1 (side) feature and treats each view's features separately. Step 12 notes sparse tongue visibility is expected and is represented by the required class 2.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Per trial: (1) build a visibility mask requiring finite x, y, frame time and likelihood > 0.9; (2) fill non-finite x and y by linear interpolation over frame index, then smooth each coordinate with a **21-frame centered boxcar** (≈52 ms at 400 fps); (3) take the first difference of the smoothed position divided by the true inter-frame interval, giving speed in px/s (`hypot` of the two coordinate differences); (4) set speed to NaN wherever the frame was not visible; (5) linearly interpolate the surviving speed samples onto the 500-bin grid, restricted to the interpolation support, and then re-impose NaN on bins whose nearest frame was invisible. No cross-view normalization (only one view is used) and no zero-filling of invisible frames.

ii.
```python
def velocity_on_grid(ft,pos,likelihood,go):
    out=np.full(TIME.shape,np.nan,float)
    if ft is None or len(ft)<2:return out
    visible=np.isfinite(pos).all(1)&np.isfinite(ft)&np.isfinite(likelihood)&(likelihood>0.9)
    # Reference smooths positions over 21 frames before finite differencing.
    pp=pos.copy()
    for j in range(2):
        good=np.isfinite(pp[:,j])
        if good.sum()>=2:
            fill=np.interp(np.arange(len(pp)),np.flatnonzero(good),pp[good,j])
            pp[:,j]=uniform_filter1d(fill,size=21,mode='nearest')
    dt=np.diff(ft); speed=np.r_[np.nan,np.sqrt(np.sum(np.diff(pp,axis=0)**2,axis=1))/np.where(dt>0,dt,np.nan)]
    speed[~visible]=np.nan; rel=ft-go
    good=np.isfinite(speed)&np.isfinite(rel)
    if good.sum()>=2:
        inside=(TIME>=rel[good].min())&(TIME<=rel[good].max())
        out[inside]=np.interp(TIME[inside],rel[good],speed[good])
        # Restore visibility gaps by nearest-frame visibility.
        ix=np.searchsorted(rel,TIME[inside]).clip(1,len(rel)-1)
        left=ix-1; near=np.where(abs(TIME[inside]-rel[left])<=abs(rel[ix]-TIME[inside]),left,ix)
        out[np.flatnonzero(inside)[~visible[near]]]=np.nan
    return out
```

iii. Notes Step 5 mapping: "Smooth position as reference, finite difference magnitude; interpolate to bin centers"; Key Decision 8: "Never turn invisible tongue/paw or absent video into low velocity. The decoder specification overrides reference plotting code that zero-fills invisible tongue velocity" (i.e., deliberately not following `findVelocity.m`'s `tempx(isnan(tempx))=0` for the tongue). The likelihood > 0.9 cut is applied as the visibility rule. The in-code comment attributes the 21-frame position smoothing to the reference.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One threshold per session per stream: the median of all finite binned velocity values pooled over all retained trials and bins of that session. Bins ≥ median → 1, < median → 0, NaN (not visible) → 2. The median and the resulting class fractions are stored in the session metadata.

ii.
```python
def disc(a):
    out=np.full(a.shape,2,np.int64); finite=np.isfinite(a)
    if finite.any():
        med=float(np.median(a[finite]));out[finite]=(a[finite]>=med).astype(np.int64)
    else:med=np.nan
    return out,med
td,tmed=disc(tongue);pd,pmed=disc(paw);md,mmed=disc(me)
```

iii. Notes Step 5 Key Decision 9: "Compute one median per session and variable from all finite available time samples after alignment/interpolation and trial filtering; equality belongs to class 1 exactly as specified. Missing samples do not affect medians." This follows the Decoder Task's per-session 50th-percentile rule; the verification output confirms exactly balanced 0/1 fractions (e.g. 0.059/0.059 with 0.882 not visible in session 1).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are converted to time from the go cue by a single subtraction, `rel = frameTimes − goCue[trial]`, and the speed trace is linearly interpolated onto the shared 500-bin grid. **No video-clock offset is applied**: the reference `findVideoOffset.m` correction (`mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`) used by `findPosition.m` is absent from the script, and `bitStart`/`bitcode` are never read. Measuring that offset directly from the raw files gives 0.490 s for the eight 2021 sessions and 0.990 s for the four JEB19 sessions, so all camera-derived outputs sit that much too late on the trial axis. Consequences visible in the converted file: the first 49 (resp. 99) bins of every trial fall outside the interpolation support, and the tongue first becomes visible at +0.58 s (JEB6) and +1.23 s (JEB19) after the nominal go cue, although the methods state responses "were typically registered within 300 ms of the go cue".

ii.
```python
speed[~visible]=np.nan; rel=ft-go
...
inside=(TIME>=rel[good].min())&(TIME<=rel[good].max())
out[inside]=np.interp(TIME[inside],rel[good],speed[good])
```
(compare reference `findPosition.m`: `interp1(traj(trix).frameTimes-vidshift-obj.bp.ev.(alignEv)(trix), ts, taxis)`)

iii. Notes Step 5 mapping claims the video streams are "interpolate[d] to bin centers" following `getKinematicsFromVideo`/`findVelocity`, and Step 10 Check 3 states "Video uses named DLC features/frame timestamps"; the trajectory log contains no mention of `findVideoOffset`, `vidshift`, `bitStart` or `findPosition.m`, so the offset appears never to have been considered. Step 7's plot review concluded "Velocity and motion-energy traces overlap the go-cue window; tongue visibility is sparse outside licking, as expected", which did not catch the shift.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DeepLabCut tracking from the **bottom camera** (`views[1]`), feature `top_paw` — x, y, likelihood from `traj.ts`, plus that view's `frameTimes` and the trial's go cue. `bottom_paw` is not used.

ii.
```python
ft,pos,lk=feature_trial(f,views[1],tr,'top_paw');paw[j]=velocity_on_grid(ft,pos,lk,go[tr])
```

iii. Notes Step 5: "Paw velocity should use the tracked top-paw feature (view 2 in reference usage)", citing the reference `params.traj_features` view-2 list (`… 'top_paw','bottom_paw' …`) and Figure 1 feature usage.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical pipeline to the tongue (`velocity_on_grid`): likelihood > 0.9 visibility mask, gap-interpolation + 21-frame boxcar smoothing of x and y, first difference divided by the real inter-frame interval → px/s, NaN on invisible frames, linear interpolation onto the 500-bin grid, NaN restored by nearest-frame visibility. No normalization (single view) and no nearest-fill of missing values (unlike `findVelocity.m`, which does `fillmissing(...,'nearest')` for non-tongue features).

ii.
```python
ft,pos,lk=feature_trial(f,views[1],tr,'top_paw');paw[j]=velocity_on_grid(ft,pos,lk,go[tr])
```
(same `velocity_on_grid` body quoted in 7-b)

iii. Notes Step 5 mapping: "Same velocity/interpolation and per-session visible-sample median split" as the tongue, and Key Decision 8: missing tracking must not become low velocity, which is why the reference's nearest-fill is dropped in favour of the explicit `not visible` class.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same `disc()` rule: per-session median of all finite binned paw speeds, ≥ median → 1, < median → 0, NaN → 2. Observed fractions: 0.383/0.383/0.233 overall, and exactly balanced 0/1 within each session.

ii.
```python
td,tmed=disc(tongue);pd,pmed=disc(paw);md,mmed=disc(me)
info={...,'thresholds':{'tongue_velocity':tmed,'paw_velocity':pmed,'motion_energy':mmed},...}
```

iii. Same as 7-c — Notes Key Decision 9, per-session median with ties assigned to class 1, computed only from available samples.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as the tongue: `rel = frameTimes − goCue[trial]` and interpolation onto the shared grid, with the frame times taken from the paw's own (bottom) camera. The `findVideoOffset` correction is again missing, so the paw stream is late by 0.49 s (eight sessions) / 0.99 s (four sessions), and the first 0.49/0.99 s of each trial is outside the interpolation support and becomes class 2 (this is part of why `not visible` fractions are 0.10–0.38 per session).

ii.
```python
speed[~visible]=np.nan; rel=ft-go
inside=(TIME>=rel[good].min())&(TIME<=rel[good].max())
out[inside]=np.interp(TIME[inside],rel[good],speed[good])
```

iii. The AI documents only "aligned with camera timestamps" and per-trial go cue; no discussion of the video/behavior clock offset appears in CONVERSION_NOTES.md or the trajectory.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file in the same folder, read with `scipy.io.loadmat(..., struct_as_record=False)` and unwrapped one level (`me.data`) into one ragged trace per native trial; the copy embedded in `obj.me` is not used. The side camera's `frameTimes` (collected during the neural/video pass) provide the time base, on the assumption of one motion-energy sample per frame. If the trace count does not equal `Ntrials`, or loading raises, the whole session's motion energy is set to missing with a warning.

ii.
```python
def load_motion(subject,date,ntrials):
    p=ROOT/f'motionEnergy_{subject}_{date}.mat'
    if not p.exists():return None
    try:
        m=loadmat(p,squeeze_me=True,struct_as_record=False)['me']
        raw=np.asarray(m.data,dtype=object).ravel(order='F')
        if len(raw)!=ntrials:return None
        return [np.asarray(x,float).ravel(order='F') for x in raw]
    except Exception as e:
        warnings.warn(f'Could not load motion energy {p.name}: {e}');return None
...
video_times.append(deref_array(f,views[0]['frameTimes'],tr).ravel(order='F').astype(float))
```

iii. Notes Step 2 records that "motion energy is represented as either a struct or trial cell array" and that separate `motionEnergy_*.mat` files exist alongside embedded `obj.me`; Step 6 lists "Ragged motion-energy traces are interpolated trial-by-trial using actual camera frame timestamps without redundant file reads" and Step 10 records the fix after the first attempt at a dense cast failed. The mapping table cites `loadMotionEnergy`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling: the per-frame value is taken as-is (the paper already reduces each frame to a scalar), truncated to `min(len(trace), len(frameTimes))`, converted to time from the go cue, and linearly interpolated onto the 500-bin grid within its support; bins outside support or with no data stay NaN → class 2 (`no video`).

ii.
```python
# Motion-energy samples correspond one-to-one with camera frames.
for j,tr in enumerate(keep_trials):
    y=me_raw[tr]; ft=video_times[j]; n=min(len(y),len(ft)); y=y[:n]; rel=ft[:n]-go[tr]
    good=np.isfinite(y)&np.isfinite(rel)
    if good.sum()>1:
        inside=(TIME>=rel[good].min())&(TIME<=rel[good].max())
        me[j,inside]=np.interp(TIME[inside],rel[good],y[good])
```

iii. Notes Step 5 mapping: "Session/trial motion energy → `output[5]`: interpolate native trace to go-cue grid; per-session median split over available finite samples", citing `loadMotionEnergy`. Step 2/3 note motion energy is a video-derived scalar per frame, so no further derivation is performed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same `disc()` call: per-session median over all finite interpolated values, ≥ median → 1, < median → 0, NaN → 2 (`no video`). Observed: 0.436 / 0.437 / 0.127 overall, with a per-session 0/1 split that is balanced to within one bin.

ii.
```python
md,mmed=disc(me)
...
'output_values':[...,['below session median','at or above session median','no video']]
```

iii. Same rationale as 7-c (Key Decision 9); the third class is reserved for genuinely absent video/data rather than low movement (Key Decision 8).

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Via the side camera's `frameTimes` minus the trial's go cue, interpolated onto the shared grid, assuming sample *i* of the trace corresponds to frame *i*. The `findVideoOffset` correction is again omitted. This is directly visible in the converted data: the `no video` bins are exactly the leading 0.49 s of each trial in the eight 2021 sessions (fraction 0.086–0.098 ≈ 0.49/5) and the leading 0.99 s in the four JEB19 sessions (0.191–0.199 ≈ 0.99/5), and session-averaged motion energy peaks at +1.12 s / +1.42 s after the nominal go cue.

ii.
```python
y=me_raw[tr]; ft=video_times[j]; n=min(len(y),len(ft)); y=y[:n]; rel=ft[:n]-go[tr]
...
me[j,inside]=np.interp(TIME[inside],rel[good],y[good])
```

iii. As in 7-d/8-d, the AI documents only "interpolate native trace to go-cue grid" using "actual camera frame timestamps". Notes Step 9 accepted the resulting `no video` fractions as "Median split/missing correct" and Step 12 treated them as reflecting "source visibility/video coverage rather than imbalance bugs".

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several targeted guards, all of which keep the trial and mark the gap rather than fabricating a value. Spike trial indices outside `[1, Ntrials]` are dropped (`valid=(tr>=0)&(tr<ntr)`). Quality labels are stripped of padding, lower-cased, and unlabeled/`real?` entries are treated as artifacts. A missing `bp.stim` group is treated as no stimulation. Trials with non-finite go cues are dropped (none occur). Missing/short video: `feature_trial` returns `None` when the named feature is absent, `velocity_on_grid` returns an all-NaN row when there are <2 frames, ragged x/y are truncated to `min(len(frameTimes), len(ts))`, and frames below the likelihood cut or outside the interpolation support stay NaN → class 2. Motion energy missing at the session level (absent file, load error, or trace count ≠ `Ntrials`) yields an all-NaN matrix → class 2 for that session, with a warning. NaNs never reach the saved arrays: outputs are `int64` categories and neural values are finite by construction. Position gaps *are* filled by linear interpolation before smoothing, but the interpolated frames are re-masked as invisible afterwards.

ii.
```python
valid=(tr>=0)&(tr<ntr); tr=tr[valid]; tm=tm[valid]
stim=np.zeros(ntr,bool)
if isinstance(b.get('stim'),h5py.Group) and 'enable' in b['stim']:stim=vec(b['stim/enable'],bool)
...
if name not in names:return None,None,None
n=min(len(ft),len(x)); return ft[:n],np.c_[x[:n],y[:n]],likelihood[:n]
...
if ft is None or len(ft)<2:return out
...
if len(raw)!=ntrials:return None
except Exception as e:
    warnings.warn(f'Could not load motion energy {p.name}: {e}');return None
```

iii. Notes Step 5 Key Decision 8: "Never turn invisible tongue/paw or absent video into low velocity. The decoder specification overrides reference plotting code that zero-fills invisible tongue velocity." Step 2 lists the schema edge cases the guards target ("cluster data can be a struct array or a cell array by probe; quality strings are space-padded and include one source typo `mutli`; older sessions omit newer fields such as `autolearn`; motion energy is represented as either a struct or trial cell array"), and Step 10 Check 5 reports verifying "nested per-probe cluster cells, variable-length video/ME traces, capitalization/padding of quality labels, absent/questionable labels, trial starts/ends, exact bin count, per-session thresholds, and missing-video classes."

## 11-a. What are the most time-consuming steps of the code?

i. The AI reports HDF5 reading/deserialization as the dominant cost, at roughly 2.1–4.2 s per session and ~35 s for the full 12-session conversion (per-session timing is printed). Within a session the two heavy loops are the per-cluster × per-trial spike histogram loop (JEB19 sessions carry 436–567 clusters on the selected probe, each of which has its `trial`/`trialtm` arrays dereferenced before the quality test) and the per-trial video loop, which dereferences `featNames`, `ts` and `frameTimes` three times per trial.

ii.
```python
def convert_session(subject,date,probe,show=False):
    t0=time.time(); path=ROOT/f'data_structure_{subject}_{date}.mat'
...
print(f"{info['session_id']}: {ntr}->{len(keep_trials)} trials, {len(unit_counts)} neurons, {time.time()-t0:.2f}s",flush=True)
```

iii. Notes Step 6: "Initial `mat73` recursive loading stalled on large spike/video branches"; the fix was "Direct HDF5 reference traversal loads only selected probe/unit arrays", after which "conversion takes about 3 seconds/session in the two-session sample, predicting under one minute for all 12 sessions." Step 7 records the estimate (~3 s/session, <1 min total) which the full run confirmed (35 s).

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain. (1) The spike-binning loop is `for cluster: for trial in np.unique(tr[use])`, and inside it builds a boolean mask `tm[(tr==old)]` over the whole spike vector for every trial — an O(n_units × n_trials × n_spikes) pattern that a single `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, EDGES])` would replace (this is how the reference solution does it). (2) The per-trial video loop calls `feature_trial`/`velocity_on_grid` separately per trial and per feature; because trials have different frame counts this is hard to fully vectorize, though the three `deref_array` calls per trial could be hoisted. (3) The per-trial output-assembly loop builds a `(6,500)` array with `np.vstack` per trial instead of filling one `(n_trials,6,500)` array. The AI's notes claim "Vectorized histogram construction where possible" but do not identify any remaining vectorizable loop.

ii.
```python
for old in np.unique(tr[use]):
    z=tm[(tr==old)]-go[old]
    mats[remap[old]]=np.histogram(z,EDGES)[0]/DT
...
for j,tr in enumerate(keep_trials):
    ft,pos,lk=feature_trial(f,views[0],tr,'tongue');tongue[j]=velocity_on_grid(ft,pos,lk,go[tr])
    ft,pos,lk=feature_trial(f,views[1],tr,'top_paw');paw[j]=velocity_on_grid(ft,pos,lk,go[tr])
    video_times.append(deref_array(f,views[0]['frameTimes'],tr).ravel(order='F').astype(float))
...
for j,tr in enumerate(keep_trials):
    ...
    y=np.vstack([np.full(len(TIME),lick),...]).astype(np.int64)
    outputs.append(y)
```

iii. Notes Step 6 under "Code speedups added": "Vectorized histogram construction where possible; conversion takes about 3 seconds/session… Ragged motion-energy traces are interpolated trial-by-trial using actual camera frame timestamps without redundant file reads." The implicit justification is that runtime (35 s total) is far inside the 15-minute budget, so no further vectorization was pursued.

## 11-c. What processing does the code repeat multiple times?

i. Small, bounded repetition. `remap` (an `ntr`-length array) is rebuilt inside the per-cluster loop instead of once per session. `tr`/`tm` are dereferenced and range-filtered for *every* cluster on the probe, including artifact-labelled ones that are then discarded. `feature_trial` re-reads and re-decodes `featNames` per trial and per view, and the side camera's `frameTimes` are dereferenced both inside `feature_trial` and again into `video_times`. The boolean mask `tr==old` is recomputed for each trial of each unit. Things that are *not* repeated: each file is opened once, motion energy is loaded once per session, the bin grid is built once at module level, and the continuous video traces are computed once and reused for both the median and the discretization.

ii.
```python
for q,rr,rt in zip(qualities,trialrefs,tmrefs):
    tr=np.asarray(f[rr][()]).ravel(order='F').astype(int)-1
    tm=np.asarray(f[rt][()]).ravel(order='F').astype(float)
    ...
    if q not in ARTIFACT and rate>1:
        mats=np.zeros((len(keep_trials),len(TIME)),np.float32)
        remap=np.full(ntr,-1,int);remap[keep_trials]=np.arange(len(keep_trials))
```

iii. The AI's notes assert the opposite for file I/O ("Avoid unnecessary file I/O … without redundant file reads", "Direct HDF5 reference traversal loads only selected probe/unit arrays") and do not discuss these residual recomputations; the stated justification for not optimizing further is the acceptable total runtime.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly in loading/curation: spike `trial`/`trialtm` arrays are read and aligned for all clusters of the probe (up to 567 in JEB19) before the quality/rate test discards most of them; `L` and `early` are loaded but never used (`early` because early-lick trials are kept); `no` is used only for one lick-direction branch that `hit`/`miss` already determine. Per-unit mean firing rates (`unit_rates`) and per-session thresholds are computed and stored in metadata but unused downstream. `velocity_on_grid` interpolates a full 500-bin speed trace for trials/features that are then almost entirely masked back to `not visible` (≈91% of tongue bins). `--show-processing` plots only the three continuous video streams. The returned continuous `(tongue,paw,me)` tuple is discarded unless plotting. Conversely, nothing that is computed in the neural path is thrown away.

ii.
```python
R,L,hit,miss,no,early,aw=map(get,['R','L','hit','miss','no','early','autowater'])
...
for q,rr,rt in zip(qualities,trialrefs,tmrefs):
    tr=np.asarray(f[rr][()]).ravel(order='F').astype(int)-1
    tm=np.asarray(f[rt][()]).ravel(order='F').astype(float)
    ...
    aligned_all=tm-go[tr]
    rate=np.sum((aligned_all>=TMIN)&(aligned_all<TMAX))/(ntr*(TMAX-TMIN))
    if q not in ARTIFACT and rate>1:
...
info={...,'mean_rates_hz':unit_rates,'thresholds':{...},'probe':probe}
return [x.astype(np.float32) for x in neural], [...], outputs, info, (tongue,paw,me)
```

iii. The notes do not address discarded processing explicitly; the relevant claims are Step 6's "targeted h5py access … avoiding full deserialization of multi-gigabyte reference arrays" and "every output is stored as compact float32/int64 arrays". Reading each cluster's spikes before the quality test is unavoidable for the >1 Hz criterion but not for the label test, which could be applied first.
