# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Only one of the four data folders is used, `/app/data/Ephys_Behavior`, and within it only 12 hard-coded `(animal, date, probe)` triples. The list was transcribed from the `load<ANM>_ALMVideo.m` metadata calls made by the paper's `Scripts/Figure 8/Figure8a_thru_c.m`, i.e. the authors' curated "two-context" ephys subset. Nothing is discovered by globbing, and the `RandomizedDelay_Ephys_Behavior` folder (19 more sessions), the remaining 13 fixed-delay sessions, and the two behaviour-only folders are never opened. Each session file is opened once with `h5py` (all 12 are MATLAB v7.3) and only three branches are dereferenced: `obj/bp` (trial table and events), `obj/clu{probe}` (spike-sorted clusters of the single selected probe) and `obj/traj` (the two DeepLabCut cameras). There is no scipy/v5 fallback reader, and the companion `motionEnergy_<anm>_<date>.mat` files that sit in the same folder are never read.

ii.
```python
ROOT='/app/data/Ephys_Behavior'
SESSIONS=[
 ('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),
 ('JGR2','2021-11-16',1),('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),
 ('JEB19','2023-04-21',1),('JEB19','2023-04-20',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-18',1)]
```

```python
def convert_session(animal,date,probe):
    path=f'{ROOT}/data_structure_{animal}_{date}.mat'
    with h5py.File(path,'r') as f:
        bp=f['obj/bp']; n=int(np.asarray(bp['Ntrials']).squeeze())
        ...
        clu=f[refs(f['obj/clu'])[probe-1]]
        ...
        rawvid=session_video(f,tids,go)

def main():
    for s in SESSIONS:
        a,b,c,i=convert_session(*s); neural.append(a); ...
```

Reference dereferencing helpers:
```python
def refs(a): return np.asarray(a).ravel(order='F')
def arr(f, ref): return np.asarray(f[ref]).squeeze()
def matlab_text(a): return ''.join(chr(int(x)) for x in np.asarray(a).ravel(order='F') if x)
```

iii. From the trajectory: at step 6 the agent decided from `methods.txt` that "the two-context electrophysiology subset consists of 12 sessions from six mice and 522 recorded units, making it the likely intended source because the decoder requires WC versus DR context". It first scanned every `Ephys_Behavior` file for sessions containing both `autowater==0` and `autowater==1` trials (22 candidates), then rejected inference-from-labels in favour of the authors' own list: "many JEB13–15 sessions have very few WC trials and do not meet the paper's behavioral criterion of at least 20 correct WC trials per direction ... The Figure 8 script's metadata loader calls should identify those exact sessions and are preferable" (step 21). At step 24 it reported "The exact expert two-context subset is recovered ... Each session uses one specified probe, matching paper curation." The randomized-delay folder was never considered after step 6, where it concluded the Ephys_Behavior folder was "the intended dataset".

## 1-b. How are the data split into subjects?

i. The subject is the animal string in the hard-coded session tuple. `subjects` is built by first appearance while walking `SESSIONS` (so it is in session order, not sorted) and `subject_idx` is the index of each session's animal into that list. The result is 7 subjects — `['JEB6','JEB7','EKH1','EKH3','JGR2','JGR3','JEB19']` — for the 12 sessions. No field inside the .mat file (`obj.meta.anm`) is consulted.

ii.
```python
subjects=[]; subject_idx=[]
for animal,_,_ in SESSIONS:
    if animal not in subjects:subjects.append(animal)
    subject_idx.append(subjects.index(animal))
...
'subjects':subjects,'subject_idx':np.asarray(subject_idx,dtype=np.int64),
```

iii. No explicit justification is given in the trajectory beyond the session list itself carrying the animal name. At step 24 the agent wrote "12 sessions from seven loader names but six mice?" — it noticed that its animal count (7) does not match the paper's "12 sessions, six mice", flagged it with a question mark, and did not resolve it.

## 1-c. How are the data split into sessions?

i. One session = one `(animal, date, probe)` tuple = one `data_structure_<anm>_<date>.mat` file = one element of `neural`, `input`, `output`, `brain_region_idx` and `metadata['session_info']`. 12 sessions are produced, in the fixed order of the `SESSIONS` list. Where a session was recorded with two probes, only the single probe named in the authors' loader file is used; the other probe's units are discarded rather than concatenated.

ii.
```python
for s in SESSIONS:
    a,b,c,i=convert_session(*s); neural.append(a); inp.append(b); out.append(c); infos.append(i)
```
```python
clu=f[refs(f['obj/clu'])[probe-1]]
```

iii. Step 23: "The metadata snippets indicate Figure 8 selects individual probes rather than concatenating both probes." Step 24: "Each session uses one specified probe, matching paper curation."

## 1-d. How are the data split into trials?

i. A trial is one entry of the per-trial arrays of `obj.bp` (`R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`) and one entry of `bp.ev.goCue`; `bp.Ntrials` gives the count. Spikes carry their trial number in `clu.trial` (1-based, converted with `-1`), and the camera data are stored as one cell per trial (`traj{cam}.ts{trial}`, `traj{cam}.frameTimes{trial}`), so trial boundaries never have to be reconstructed. Behaviour flags are read with `np.asarray(...).ravel()` without truncating to `Ntrials`, but the trial list is built from `np.flatnonzero` over the mask, so only indices that exist are used.

ii.
```python
bp=f['obj/bp']; n=int(np.asarray(bp['Ntrials']).squeeze())
label={k:np.asarray(bp[k]).ravel().astype(bool) for k in ['R','L','hit','miss','no','early','autowater']}
go=np.asarray(bp['ev/goCue']).ravel().astype(float)
keep=(label['hit']|label['miss']|label['no']) & np.isfinite(go)
tids=np.flatnonzero(keep)
```
```python
for ni,ui in enumerate(good):
    tr=arr(f,trial_cells[ui]).astype(int)-1; rel=arr(f,tm_cells[ui]).astype(float)
    for ti,oj in tid_to_out.items():
        sp=rel[tr==ti]-go[ti]
```

iii. Step 26: "spike arrays carry one-based trial IDs, and `trialtm` is relative to trial start. Go-cue alignment therefore requires subtracting each trial's `bp.ev.goCue` time (which is in the same trial-relative basis), exactly as paper preprocessing does."

## 1-e. How are trials filtered based on quality controls?

i. Effectively no trial curation is performed. The only mask is `(hit | miss | no) & isfinite(goCue)`, which is condition 1 of `Figure8a_thru_c.m`. Because `hit`, `miss` and `no` are mutually exclusive and exhaustive in these files, this keeps **every** trial of every session: all 3,626 source trials survive (382+346+254+305+426+267+254+261+276+324+230+301). In particular, early-lick trials are deliberately kept (389 of them), and photostimulation trials (`bp.stim.enable`, 135 trials across JEB6, JEB7 ×2 and JGR3, up to 15% of a session) are never even read. No check is made that the probe was still recording at the end of the session.

ii.
```python
# Figure 8 condition 1: hit|miss|no. Keep early trials because that exact
# all-trial condition does not exclude them and outcome decoding needs no.
keep=(label['hit']|label['miss']|label['no']) & np.isfinite(go)
tids=np.flatnonzero(keep)
```

iii. Step 6: "Early-lick and ignore trials were omitted from the paper's behavioral analyses, but the requested decoder explicitly requires an `ignore` outcome class, so ignore trials should be retained when alignable; early-lick handling needs investigation." Step 22 settled it: "Trial condition 1 includes hit, miss, and no outcomes, so ignore/no trials should indeed be retained." The in-code comment states the early-lick rationale. Photoinactivation trials are not discussed anywhere in the trajectory, even though the agent had earlier grepped `stim.enable` out of the authors' condition strings (step 11).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the single selected probe: the per-cluster cells `trial` (1-based trial of each spike), `trialtm` (spike time relative to trial start) and `tm` (used only to count how many clusters exist). `bp.ev.goCue` supplies the alignment time, and `bp.Ntrials` the denominator of the firing-rate filter. `clu.quality` is never read.

ii.
```python
clu=f[refs(f['obj/clu'])[probe-1]]
trial_cells=refs(clu['trial']); tm_cells=refs(clu['trialtm'])
alltm=refs(clu['tm'])
```
```python
tr=arr(f,trial_cells[ui]).astype(int)-1; rel=arr(f,tm_cells[ui]).astype(float)
sp=rel[tr==ti]-go[ti]
```

iii. Step 17: "cluster fields `quality`, `site`, `tm`, `trial`, and `trialtm`; ... `trialtm` likely already contains spike times relative to each trial and is ideal for binning." Step 26 confirmed the 1-based trial ids and the trial-relative time base.

## 2-b. How is the `neural` data processed?

i. Per unit and per trial, go-cue-relative spike times are histogrammed into the 550 fixed 10 ms bins spanning [−3, 2.5) s, divided by the bin width to give spikes/s, and then smoothed along time with the paper's *causal* Gaussian: `gausswin(15)` (σ = (15−1)/(2·2.5) = 2.8 bins), first `floor(15/2)=7` taps set to zero, renormalised to unit sum, convolved 'same' with zero padding at the edges. Nothing else is done — no z-scoring, no baseline subtraction, no trial averaging. Values are stored as `float32` Hz.

ii.
```python
def causal_smooth(x, n=15):
    # MATLAB gausswin(15), first floor(15/2) entries zeroed, conv(...,'same').
    k=gaussian(n, std=(n-1)/2/2.5, sym=True)
    k[:n//2]=0; k/=k.sum()
    return convolve1d(x, k, axis=-1, mode='constant', origin=0)
```
```python
counts=np.histogram(sp,bins=EDGES)[0].astype(np.float32)/DT
neural_trials[oj][ni]=causal_smooth(counts)
```

iii. Step 25: "The paper's `getPSTHs` bins each unit's event-aligned spike times at 10 ms, converts counts to Hz, and applies `MySmooth` with width 15 to each single-trial trace." The code comment names the MATLAB construction it is reproducing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single filter: a unit is kept if its mean firing rate over the analysis window, computed as (number of its spikes falling in [−3, 2.5) s of any trial's go cue) / (`Ntrials` × 5.5 s), exceeds 1 Hz. The manual curation label `clu.quality` is **not** used at all, so clusters labelled `garbage`, `noisy`, `poor`, etc. are retained if they fire fast enough. This yields 956 units over the 12 sessions (30, 66, 27, 48, 66, 29, 53, 28, 144, 202, 130, 133), against the 522 units the paper reports for exactly this subset. In `JEB19_2023-04-18` probe 1, 400 of the 436 clusters carry the label `garbage`, and 133 units are kept. No per-session minimum unit count is enforced (all sessions happen to exceed 10).

ii.
```python
# removeLowFRClusters calls getFiringRate on trial-aligned data: mean
# over every time bin in [-3,2.5) and every source trial. Reproduce
# that denominator rather than including inter-trial wall-clock time.
good=[]
for ui in range(len(alltm)):
    tr_all=arr(f,trial_cells[ui]).astype(int)-1
    rel_all=arr(f,tm_cells[ui]).astype(float)
    aligned=rel_all-go[tr_all]
    nwin=np.count_nonzero((aligned>=TMIN)&(aligned<TMAX))
    rate=nwin/(n*(TMAX-TMIN))
    if rate>1.: good.append(ui)
```

iii. Two distinct decisions. The 1 Hz cut follows `params.lowFR = 1` in `Figure8a_thru_c.m` and the methods sentence "All units with firing rates exceeding 1 Hz were included in all other analyses"; the denominator was corrected mid-run after the agent noticed too few units survived (steps 33–36: "Our wall-clock rate denominator incorrectly includes inter-trial time, explaining the artificially low counts in JEB6 and JEB19"). The absence of a quality filter was an explicit conclusion: the agent flagged the risk at step 35 ("Unit-quality initialization may also exclude garbage/poor clusters even when `params.quality={'all'}`, so inspect that path before regenerating") and then concluded at step 36 "With `quality={'all'}`, all cluster qualities are eligible before this rate filter." It had noticed at step 33 that its 912/956 units far exceed the paper's 522 and attributed the gap to the rate calculation or "the published count predating those sessions".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. By subtracting the trial's go-cue time from the trial-relative spike times: `trialtm − goCue[trial]`. Both are on the Bpod clock and relative to the start of the same trial, so no other correction, interpolation or warping is applied. Histogram edges then run from −3 to 2.5 s about that zero.

ii.
```python
go=np.asarray(bp['ev/goCue']).ravel().astype(float)
...
sp=rel[tr==ti]-go[ti]
counts=np.histogram(sp,bins=EDGES)[0].astype(np.float32)/DT
```

iii. Step 26 (quoted in 1-d): the agent verified `trialtm` is trial-relative and that `goCue` is on the same basis, matching the paper's `alignSpikes.m` with `params.alignEvent='goCue'`. Step 12: "Go-cue alignment is explicitly used throughout the paper code."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins over [−3.0, 2.5) s from the go cue, i.e. 550 bins per trial, identical for every trial and session, and shared by the neural, input and all three video outputs. This is taken from `Figure8a_thru_c.m` (`params.dt = 1/100`, `params.tmin = -3`, `params.tmax = 2.5`) rather than from `getDefaultParams.m` (5 ms, ±2.5 s). Spikes are binned directly at 10 ms, so no rebinning of an intermediate resolution happens; the ~400 Hz camera streams are resampled onto this grid by linear interpolation (not by averaging frames within a bin). `metadata` records `time_bin_size: 10.0`, `off_start: -3.0`, `off_end: 2.5`.

ii.
```python
DT=.01; TMIN=-3.; TMAX=2.5
TIME=np.arange(TMIN,TMAX,DT); EDGES=np.arange(TMIN,TMAX+DT/2,DT)
```
```python
'metadata':{... 'time_bin_size':10.0,'temporal_alignment_event':'go cue onset',
            'off_start':-3.0,'off_end':2.5, ...}
```

iii. Step 22: "The expert Figure 8 parameters are now explicit: align to go cue, use −3 to +2.5 s at 10 ms bins, remove units below 1 Hz, include all unit qualities, apply causal Gaussian smoothing of width 15 bins". The agent chose the Figure 8 parameter set throughout because Figure 8 is the two-context analysis it had selected the sessions from.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read from the raw data; it is the analysis grid itself, defined by the alignment window and bin size (`TIME = np.arange(-3, 2.5, 0.01)`). `bp.ev.goCue` enters only implicitly, in that the grid is defined relative to it.

ii.
```python
TIME=np.arange(TMIN,TMAX,DT)
...
time_row=TIME.astype(np.float32)[None,:]
'input_names':['time from go cue onset'],
```

iii. The prompt defines this input ("Time from go cue onset in seconds (continuous, time-varying)") and the agent's plan at step 28 lists it as a derived time axis rather than a data field.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond casting to `float32` and copying the same (1, 550) row into every trial of every session. The values are the *left edges* of the bins (−3.00, −2.99, …, 2.49), not bin centres, so the input is offset by half a bin (5 ms) from the centre of the spike-count bin it labels.

ii.
```python
time_row=TIME.astype(np.float32)[None,:]
for j,ti in enumerate(tids):
    inputs.append(time_row.copy())
```

iii. No explicit justification in the trajectory; the agent's step-28 plan simply states the streams are put on the common grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. Perfectly by construction: `EDGES` is `TIME` extended by one bin, so `TIME[k]` is the left edge of the same interval that produced `neural[:, k]`, and the video traces are interpolated at exactly the points of `TIME`. All four streams therefore share one index axis.

ii.
```python
TIME=np.arange(TMIN,TMAX,DT); EDGES=np.arange(TMIN,TMAX+DT/2,DT)
...
counts=np.histogram(sp,bins=EDGES)[0]
...
out[:]=np.interp(TIME,old_t[good],val[good]).astype(np.float32)
```

iii. Implicit; the shared module-level grid is the mechanism.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Three per-trial flags of `obj.bp`: `no` (no response), `hit` (correct), and `L` (instructed left; `R` is its complement). The actual licked port is not recorded, so it is inferred from instructed side plus outcome.

ii.
```python
label={k:np.asarray(bp[k]).ravel().astype(bool) for k in ['R','L','hit','miss','no','early','autowater']}
```

iii. Step 11: "paper code uses boolean fields such as `R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`, and `stim.enable`. Thus lick direction can come from R/L, outcome from hit/miss/no."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A three-way per-trial relabelling: a no-response trial is `none` (2); a hit means the animal licked the instructed port, so the class is the instructed side; anything else (a miss, since only hit/miss/no trials are kept) means it licked the other port, so the class is the opposite side. Codes are left 0, right 1, none 2, matching `output_values[0] = ['left','right','none']`. The scalar is then tiled across all 550 bins so the output is time-varying in shape.

ii.
```python
# Actual lick direction: correct follows instructed side; incorrect is
# opposite; ignored/no-response has no lick. left=0,right=1,none=2.
if label['no'][ti]: lick=2
elif label['hit'][ti]: lick=0 if label['L'][ti] else 1
else: lick=1 if label['L'][ti] else 0
...
const=np.array([lick,context,outcome],dtype=np.int16)[:,None]
const=np.repeat(const,TIME.size,axis=1)
```

iii. The inline comment is the justification; the trajectory (step 11) records the same reasoning from the authors' condition strings.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The single per-trial flag `obj.bp.autowater`.

ii.
```python
label={k:np.asarray(bp[k]).ravel().astype(bool) for k in [...,'autowater']}
```

iii. Step 9: "The context code reveals that `obj.bp.autowater` is the context label: 0 denotes the delayed-response/2AFC context and 1 denotes the autowater (WC) context."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling, `autowater → 1 (WC)`, otherwise `0 (DR)`, tiled across the 550 bins. `output_values[1] = ['DR','WC']` so the codes and the names agree. The result is 2,460 DR and 1,166 WC trials.

ii.
```python
context=1 if label['autowater'][ti] else 0 # DR=0, WC=1
'output_values':[..., ['DR','WC'], ...]
```

iii. Same as 5-a; the agent verified the polarity of `autowater` against the authors' analysis code before coding it.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Two per-trial flags, `obj.bp.no` and `obj.bp.hit`. `miss` is not needed because only hit/miss/no trials are kept, so "neither no nor hit" implies miss.

ii.
```python
outcome=2 if label['no'][ti] else (1 if label['hit'][ti] else 0)
```

iii. Step 11: "outcome from hit/miss/no (mapped to correct/incorrect/ignore)".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Relabelling into the three classes required by the prompt — incorrect 0, correct 1, ignore 2 — then tiling across the 550 bins. The dataset ends up with 429 incorrect, 2,430 correct and 767 ignore trials.

ii.
```python
outcome=2 if label['no'][ti] else (1 if label['hit'][ti] else 0)
...
'output_values':[..., ['incorrect','correct','ignore'], ...]
```

iii. The prompt fixes the class order; the agent noted at step 6 that ignore trials must be retained because the decoder requires an `ignore` class, even though the paper drops them.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`. Both cameras are opened; within each, every feature whose name contains the substring `tongue` is used (side camera: `tongue`, `left_tongue`, `right_tongue`; bottom camera: `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue`). For each of those the x, y and likelihood rows of `traj{cam}.ts{trial}` (read as `(feature, xyl, frame)`) are used, together with `traj{cam}.frameTimes{trial}` and `bp.ev.goCue`. The side camera's trace is the one exported; the bottom camera is used only if the side trace is entirely missing.

ii.
```python
def camera_trial(f,traj,ti):
    ts=arr(f,refs(traj['ts'])[ti]).astype(float)
    ft=arr(f,refs(traj['frameTimes'])[ti]).astype(float)
    if ts.ndim!=3:return None,None,None
    # HDF5 dimensions are feature, coordinate(x,y,likelihood), frame.
    if ts.shape[1]!=3:return None,None,None
    names_obj=f[refs(traj['featNames'])[ti]]
    names=[matlab_text(f[r]) for r in refs(names_obj)]
    return ts,names,np.ravel(ft)
```
```python
tong=[j for j,n in enumerate(names) if 'tongue' in n]
s,_=landmark_speed(ts,tong)
if s is not None:
    q=interp_trace(old,s); components.append(q)
    if cami==0 or np.all(~np.isfinite(tv)): tv=q
```

iii. Step 28: "side camera has tongue landmarks; top camera has tongue and paw landmarks. Each trial matrix is features × 3 × frames, corresponding to x, y, and DLC likelihood, sampled near 400 Hz with frame timestamps." The agent verified the layout and feature names empirically (steps 17–18) before writing the converter.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps. (1) Frames with likelihood < 0.9 are masked to NaN, per landmark. (2) The visible landmarks are collapsed to a centroid with `nanmean` over x and over y. (3) Speed is a one-sample forward difference of that centroid, `hypot(dx, dy) × 400`, i.e. a hard-coded 400 fps is assumed rather than using the actual frame times (measured median Δt is 2.52 ms ≈ 397 Hz); the first sample is NaN, and samples where no landmark was visible are set back to NaN. (4) The frame-resolution trace is resampled onto the 550-bin grid with `np.interp`, keeping only the span covered by finite samples. No smoothing of the position, no per-camera normalisation, and no averaging of the two views. Because step 4 interpolates over the NaN gaps *inside* the covered span, a bin in which the tongue was not tracked still receives an interpolated velocity: `not visible` marks only bins outside the first/last visible frame. Measured over the converted file, the tongue is `not visible` in 69.4% of bins, and within a trial the classes run continuously from the first protrusion to the last with no interruptions between licks.

ii.
```python
def landmark_speed(ts, indices):
    xy=ts[indices,:2,:]
    lk=ts[indices,2,:]
    vis=lk>=0.9
    xy=np.where(vis[:,None,:],xy,np.nan)
    cx=np.nanmean(xy[:,0,:],axis=0); cy=np.nanmean(xy[:,1,:],axis=0)
    visible=np.any(vis,axis=0)
    speed=np.r_[np.nan, np.hypot(np.diff(cx),np.diff(cy))*400.]
    speed[~visible]=np.nan
    return speed,visible

def interp_trace(old_t, val):
    good=np.isfinite(old_t)&np.isfinite(val)
    out=np.full(TIME.size,np.nan,dtype=np.float32)
    if good.sum()>=2:
        out[:]=np.interp(TIME,old_t[good],val[good]).astype(np.float32)
        out[TIME < old_t[good].min()]=np.nan; out[TIME > old_t[good].max()]=np.nan
    return out
```

iii. Step 28: "A justified substitute is aggregate visible-landmark motion ... Neural and kinematic streams can otherwise follow the paper exactly", with the plan to "interpolate DLC-derived tongue/paw velocities ... onto the same grid". Step 29 explains the NaNs: "The warnings are expected at frames where no selected DLC landmark is visible; these locations are intentionally preserved as NaN and later assigned class 2" — which is true only for frames outside the interpolation span. Step 34 explains the high not-visible rate as "the −3 to +2.5 s window extends beyond camera coverage and tongue DLC confidence is often low". The 0.9 likelihood cut is the value used in the authors' kinematics code, which the agent read at step 10.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One threshold per session, the 50th percentile of all finite values pooled over every trial and every bin of that session. Bins below it become 0, bins at or above become 1, and every non-finite bin becomes 2. The threshold is recorded per session in `metadata['session_info'][i]['video_median_thresholds']`.

ii.
```python
def discretize_session(stream):
    finite=np.isfinite(stream)
    threshold=float(np.nanpercentile(stream,50)) if finite.any() else np.nan
    out=np.full(stream.shape,2,dtype=np.int16)
    out[finite & (stream<threshold)]=0; out[finite & (stream>=threshold)]=1
    return out,threshold
```
```python
'output_values':[..., ['below session median','at or above session median','not visible'], ...]
```

iii. The prompt specifies the per-session 50th-percentile rule and the `not visible` class; the code and the `output_values` strings follow it literally.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are converted to seconds-from-go-cue with a **fixed 0.5 s** video offset, `old = frameTimes − 0.5 − goCue[trial]`, and the trace is then linearly interpolated onto the same 550-bin grid as the spikes. The session-specific offset returned by the authors' `findVideoOffset.m` (mode of `sglx.bitcode.bitstart / sglx.fs` minus mode of `bp.ev.bitStart`) is never computed. For the eight 2021 sessions the true offset is 0.490 s, so the error is one bin; for the four JEB19 2023 sessions it is 0.990 s, so those sessions are shifted by 0.49 s (49 bins). The consequence is visible in the converted file: in `JEB19_2023-04-21` the paw is `not visible` in 91% of trials at t = −2.0 s (camera coverage starts at −1.98 s instead of −2.47 s), and the tongue-visibility peak moves from 0.27 s to 0.77 s after the go cue.

ii.
```python
old=ft-0.5-go[ti] # exact offset used by paper loadMotionEnergy/getPosition
...
q=interp_trace(old,s)
```

iii. Step 13: "The paper's motion-energy loader aligns camera samples using per-trial frame timestamps, subtracts a 0.5 s video offset and the selected event time, interpolates to the common time axis, fills edge NaNs by nearest values ... This is essential to reproduce." The 0.5 s value comes from `funcs/fig3/loadMotionEnergy_Behav.m` (the behaviour-only variant) and from the `catch` branch of `DataLoadingScripts/loadMotionEnergy.m`, which uses `frameTimes = (1:nframes)/400` when real frame times are missing. The agent had listed `funcs/findVideoOffset.m` in a `cat` batch at step 8 but its output was scrolled off the terminal; at step 26 it planned to "print go-cue ranges and trajectory frame-time basis to confirm the paper's 0.5 s video offset" and never did.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` tracking: every feature whose name contains `paw`, which on the bottom camera is `top_paw` and `bottom_paw` (the side camera has none, so only the bottom view contributes). Same `ts`, `frameTimes` and `goCue` inputs as the tongue.

ii.
```python
paws=[j for j,n in enumerate(names) if 'paw' in n]
s,_=landmark_speed(ts,paws)
if s is not None:
    pv=interp_trace(old,s); components.append(pv)
```

iii. Step 27: "Paw may not be among Figure 8's selected kinematic features, but the decoder specifically requests it, so all DLC features must be considered." Step 28 confirmed the bottom camera carries the paw landmarks.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to the tongue: likelihood ≥ 0.9 mask, `nanmean` centroid of `top_paw` and `bottom_paw`, forward-difference speed × 400, NaN where neither paw is visible, linear interpolation onto the 550-bin grid (again filling gaps inside the covered span). No smoothing, no normalisation, no per-paw treatment — the two forepaws are merged into one centroid, so whichever is visible drives the trace. The paw is `not visible` in 11.8% of bins overall, nearly all of it the uncovered head/tail of the window.

ii.
```python
xy=ts[indices,:2,:]; lk=ts[indices,2,:]; vis=lk>=0.9
xy=np.where(vis[:,None,:],xy,np.nan)
cx=np.nanmean(xy[:,0,:],axis=0); cy=np.nanmean(xy[:,1,:],axis=0)
speed=np.r_[np.nan, np.hypot(np.diff(cx),np.diff(cy))*400.]
speed[~visible]=np.nan
```

iii. Step 28's plan treats tongue and paw with the same routine ("velocity from DLC x/y derivatives with visibility from likelihood"); no separate discussion of which paw landmark is reliable appears in the trajectory.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Exactly as the tongue: `discretize_session` with the session-wide 50th percentile of finite values; 0 below, 1 at or above, 2 where the value is NaN.

ii.
```python
disc=[]; thresholds=[]
for x in rawvid:
    d,th=discretize_session(x); disc.append(d); thresholds.append(th)
```

iii. As 7-c: the prompt's rule.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Through the same `old = frameTimes − 0.5 − goCue[trial]` mapping and the same `np.interp` onto the shared grid, using the bottom camera's own `frameTimes`. It therefore inherits the same 0.49 s misalignment on the four JEB19 sessions, which is what makes the paw appear untracked before ≈ −1.98 s in those sessions.

ii.
```python
for cami,tr in enumerate(traj):
    ts,names,ft=camera_trial(f,tr,ti)
    if ts is None:continue
    old=ft-0.5-go[ti] # exact offset used by paper loadMotionEnergy/getPosition
```

iii. Same as 7-d.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Not from the authors' motion energy at all. The `motionEnergy_<anm>_<date>.mat` file that accompanies every one of the 12 sessions in `/app/data/Ephys_Behavior` is never opened, and neither is `obj.me`. Instead motion energy is a proxy built from the same DLC tracking: for each camera, the tongue-centroid speed, the paw-centroid speed and the speed of the centroid of *all* tracked landmarks are stacked and averaged bin-by-bin with `nanmean`.

ii.
```python
# Aggregate all confidently visible landmarks as image-motion proxy.
s,_=landmark_speed(ts,list(range(len(names))))
if s is not None: components.append(interp_trace(old,s))
...
if components:
    with np.errstate(invalid='ignore'): me=np.nanmean(np.stack(components),axis=0).astype(np.float32)
else: me=np.full(TIME.size,np.nan,np.float32)
```
```python
'video_processing':'... Motion energy is an aggregate landmark-motion proxy because released files contain no raw video/pixel-motion stream.'
```

iii. Step 24: "The supplied data has no separate motion-energy directory, so motion energy must either be represented in trajectory data or derivable from video/DLC coordinates." Step 28: "No raw video or precomputed pixel motion-energy files are supplied, so true paper motion energy cannot be reconstructed. A justified substitute is aggregate visible-landmark motion from both cameras, while reserving class 2 for trials with no usable video." The premise is false: the agent did issue `find /app/data -type f | grep -iE 'motion|energy|ME_|video'` twice (steps 18 and 19), but in both batches the terminal screen it read back had been overwritten by later commands, so it never saw the 25 `motionEnergy_*.mat` files that are in the same folder as the session files it was loading.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per camera and trial: mask landmarks with likelihood < 0.9, take the `nanmean` centroid across *all* landmarks of that camera (tongue, jaw, nose, trident, lickport, paws, nostrils), forward-difference it to a speed × 400, interpolate to the 550-bin grid; then average that trace together with the camera's tongue trace and paw trace (up to five traces across the two cameras) with `nanmean`. A bin is NaN only if no camera contributed anything. There is no pixel-difference computation, no 99th-percentile-across-pixels reduction, and no per-camera scaling before averaging, so the average mixes two cameras with different pixel scales. The resulting classes duplicate the paw classes on 69% of bins (65% among bins where both are visible).

ii.
```python
for cami,tr in enumerate(traj):
    ...
    s,_=landmark_speed(ts,tong);  components.append(interp_trace(old,s))
    s,_=landmark_speed(ts,paws);  components.append(pv)
    s,_=landmark_speed(ts,list(range(len(names)))); components.append(interp_trace(old,s))
me=np.nanmean(np.stack(components),axis=0).astype(np.float32)
```

iii. As 9-a — the proxy is presented as the best available substitute for an absent stream.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretize_session` call: per-session 50th percentile of the finite proxy values, 0 below / 1 at or above, and 2 where the proxy is NaN. The third class is named `no video` in `output_values`, consistent with the prompt.

ii.
```python
rawvid=session_video(f,tids,go)
disc=[]; thresholds=[]
for x in rawvid:
    d,th=discretize_session(x); disc.append(d); thresholds.append(th)
...
['below session median','at or above session median','no video']
```

iii. The prompt's rule; the agent's step-28 plan says "threshold each valid continuous signal at its session median; assign visibility/no-video class 2".

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It inherits the alignment of the DLC traces it is built from: `frameTimes − 0.5 − goCue[trial]`, linearly interpolated onto the shared 550-bin grid. So it is offset by 0.49 s on the four JEB19 sessions, as in 7-d, and is exact to within one bin elsewhere.

ii.
```python
old=ft-0.5-go[ti] # exact offset used by paper loadMotionEnergy/getPosition
...
components.append(interp_trace(old,s))
```

iii. Step 13 (the 0.5 s offset read out of the motion-energy loader) is the stated source.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Everything is handled by producing NaN and letting the discretiser turn it into class 2; nothing is dropped and nothing is nearest-filled. Specifically: trials whose go cue is not finite are excluded by the trial mask; a camera whose `ts` cannot be read, is not 3-D, or is not `(feature, 3, frame)` is skipped with a bare `except Exception` and contributes nothing; a trace with fewer than two finite samples yields an all-NaN row; bins outside the span of finite frame times stay NaN; landmarks below the likelihood cut are NaN'd and, if all landmarks of a group are invisible in a frame, that frame is NaN. `np.errstate` and `nanmean` warnings for all-NaN slices are tolerated (they are printed once per session as RuntimeWarnings). Units are not checked for empty spike trains. The one place where missing data is *not* preserved is inside `interp_trace`: NaN gaps between the first and last finite sample are silently interpolated across, so untracked frames within the covered span become ordinary 0/1 classes rather than `not visible`.

ii.
```python
def camera_trial(f,traj,ti):
    try:
        ts=arr(f,refs(traj['ts'])[ti]).astype(float)
        ft=arr(f,refs(traj['frameTimes'])[ti]).astype(float)
    except Exception:return None,None,None
    if ts.ndim!=3:return None,None,None
    if ts.shape[1]!=3:return None,None,None
```
```python
if good.sum()>=2:
    out[:]=np.interp(TIME,old_t[good],val[good]).astype(np.float32)
    out[TIME < old_t[good].min()]=np.nan; out[TIME > old_t[good].max()]=np.nan
```
```python
keep=(label['hit']|label['miss']|label['no']) & np.isfinite(go)
```

iii. Step 29: "The warnings are expected at frames where no selected DLC landmark is visible; these locations are intentionally preserved as NaN and later assigned class 2." Step 28's plan states the same policy of reserving class 2 for missing video rather than dropping trials.

## 11-a. What are the most time-consuming steps of the code?

i. Reading the HDF5 cell arrays. Profiling one session (`JGR3_2021-11-18`, 261 trials, 67 clusters, 3.35 s total) puts 2.72 s in `session_video`, of which 2.28 s is `camera_trial` — almost entirely `h5py` `read_direct` of the per-trial `ts` matrices (7,695 dataset reads, 1.33 s) plus 4,437 calls to `matlab_text` decoding the feature-name strings again for every trial and camera. The spike side is cheap for small sessions (7,308 histogram calls, 0.19 s; 7,308 `causal_smooth` calls, 0.21 s) but grows as units × trials: `JEB19_2023-04-18` (6.1 s total) has 436 clusters read twice each and 133 × 301 ≈ 40,000 histogram-plus-convolution calls, which cost 0.93 s and 1.11 s respectively against 3.2 s for its video. Writing the 665 MB pickle at the end is also non-trivial.

ii.
```python
def camera_trial(f,traj,ti):
    ts=arr(f,refs(traj['ts'])[ti]).astype(float)
    ...
    names=[matlab_text(f[r]) for r in refs(names_obj)]
```
```python
for ni,ui in enumerate(good):
    tr=arr(f,trial_cells[ui]).astype(int)-1; rel=arr(f,tm_cells[ui]).astype(float)
    for ti,oj in tid_to_out.items():
        sp=rel[tr==ti]-go[ti]
        counts=np.histogram(sp,bins=EDGES)[0].astype(np.float32)/DT
        neural_trials[oj][ni]=causal_smooth(counts)
```

iii. Not discussed in the trajectory; the agent only polled for completion and never profiled or optimised.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three. (1) The unit × trial double loop for spike binning: every cluster's spikes are re-scanned once per trial with a boolean mask `rel[tr==ti]`, which is O(units × trials × spikes); a single `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, EDGES])` per unit — or per session — would replace it, and the mask alone makes this loop quadratic in trial count. (2) `causal_smooth` is called once per (unit, trial) on a 550-sample vector; it could be applied once to the whole `(units, trials, 550)` array along the last axis. (3) The firing-rate pre-pass loops over all clusters doing the same `rel − go[tr]` subtraction that the binning loop then repeats. The per-trial camera loop is harder to vectorise because trials have different frame counts, but the feature-name lookup inside it is loop-invariant.

ii.
```python
for ti,oj in tid_to_out.items():
    sp=rel[tr==ti]-go[ti]                     # re-scans the whole spike train per trial
    counts=np.histogram(sp,bins=EDGES)[0].astype(np.float32)/DT
    neural_trials[oj][ni]=causal_smooth(counts)
```

iii. Not discussed in the trajectory.

## 11-c. What processing does the code repeat multiple times?

i. Four repetitions. (1) Each cluster's `trial` and `trialtm` cells are read from disk and go-cue-subtracted in the rate filter, then read and subtracted again in the binning loop. (2) `featNames` is dereferenced and decoded character-by-character for every trial of every camera, although it is constant within a camera. (3) `landmark_speed` is run three times per camera per trial — tongue, paws, all landmarks — and the all-landmark pass recomputes what the tongue and paw passes already did. (4) The bottom camera's tongue trace is always computed and interpolated even though it is used only if the side camera produced nothing. (5) Per-trial constants are materialised as 550-column arrays with `np.repeat` for every trial.

ii.
```python
for ui in range(len(alltm)):
    tr_all=arr(f,trial_cells[ui]).astype(int)-1
    rel_all=arr(f,tm_cells[ui]).astype(float)
...
for ni,ui in enumerate(good):
    tr=arr(f,trial_cells[ui]).astype(int)-1; rel=arr(f,tm_cells[ui]).astype(float)
```
```python
names=[matlab_text(f[r]) for r in refs(names_obj)]   # inside the per-trial loop
```

iii. Not discussed in the trajectory.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items. The `visible` mask returned by `landmark_speed` is discarded at every call site (`s,_=...`). The bottom camera's tongue interpolation is discarded whenever the side camera has any finite sample. `n_source_trials`, `source_trial_indices`, `n_units_probe` and the per-session thresholds are computed for `session_info` and are not used downstream (they are useful provenance, but they are not read by the decoder). Constant per-trial labels are expanded to 550 identical columns, which inflates the pickle; together with storing all six outputs as `int16` rather than `int8`, this accounts for a large part of the 665 MB file. Conversely, the largest genuinely unnecessary work is upstream: the code reads and processes every cluster on the probe, including the hundreds labelled `garbage`, only to store many of them as neurons.

ii.
```python
s,_=landmark_speed(ts,tong)         # `visible` never used
...
if cami==0 or np.all(~np.isfinite(tv)): tv=q   # cam1 tongue trace usually thrown away
```
```python
const=np.repeat(const,TIME.size,axis=1)
outputs.append(np.vstack([const,disc[0][j],disc[1][j],disc[2][j]]).astype(np.int16))
```

iii. Not discussed in the trajectory; the agent's final summary (step 42) reports only that the artefacts exist and that the verifier passed.
