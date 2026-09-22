# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes 12 two-context sessions from `/app/data/Ephys_Behavior`, including a selected probe for each, and opens each v7.3 file directly with `h5py`. Motion energy is loaded separately with SciPy. It does not load the randomized-delay sessions or all 44 sessions used by the human solution.

ii.
```python
ROOT=Path('/app/data/Ephys_Behavior')
SESSIONS=[('JEB6','2021-04-18',2), ... ,('JEB19','2023-04-21',1)]
...
with h5py.File(path,'r') as f:
...
m=loadmat(p,squeeze_me=True,struct_as_record=False)['me']
```

iii. The AI says the required WC/DR output makes the paper's 12-session Figure 8 two-context subset the relevant cohort; it excluded behavior-only and randomized-delay cohorts and used targeted HDF5 traversal for speed.

## 1-b. How are the data split into subjects?

i. Subject IDs are the hard-coded animal strings in `SESSIONS`. During assembly, subjects are added in first-appearance order and each session receives the corresponding index. This yields seven released IDs.

ii.
```python
if sub not in subjects: subjects.append(sub)
sidx.append(subjects.index(sub))
```

iii. The AI notes that the 12 selected sessions contain seven released IDs although the manuscript says six mice, and treats this as a source discrepancy rather than dropping an ID.

## 1-c. How are the data split into sessions?

i. Each tuple in `SESSIONS` is one session and maps to one `data_structure_<subject>_<date>.mat`; the resulting neural/input/output entry is one session. Only 12 fixed-delay/two-context sessions are included.

ii.
```python
for i,(sub,date,probe) in enumerate(sessions):
    n,x,y,info,cont=convert_session(sub,date,probe,args.show_processing)
    neural.append(n); inputs.append(x); outputs.append(y)
```

iii. The AI chose sessions identified from Figure 8 loaders, asserting that these match the paper's 12-session two-context analysis.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines the native trial count. Per-trial behavioral arrays, spike `trial` indices, trajectory reference arrays, and motion-energy cells are indexed by that trial number. Retained trial indices become the per-session trial list.

ii.
```python
ntr=int(np.asarray(b['Ntrials'][()]).squeeze())
keep_trials=np.flatnonzero((~stim)&np.isfinite(go))
...
for j,tr in enumerate(keep_trials):
```

iii. The AI states that native trial correspondence is required and trials should not be inferred or invented.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained if photostimulation is disabled and go-cue time is finite. Early-lick and ignore trials are retained; the code does not remove trials after the ephys recording ends.

ii.
```python
keep_trials=np.flatnonzero((~stim)&np.isfinite(go))
```

iii. The AI says Figure 8 conditions require non-stimulation trials, while the requested ignore outcome justifies retaining ignore trials. It also explicitly chose to retain early trials, mapping otherwise-unclassified responses to ignore.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the selected probe's cluster `trial` and `trialtm` arrays, cluster `quality`, behavioral `bp.ev.goCue`, and `Ntrials`.

ii.
```python
trialrefs=cg['trial'][()].ravel(order='F')
tmrefs=cg['trialtm'][()].ravel(order='F')
qualities=np.array([chars(f,r).lower() for r in cg['quality'][()].ravel(order='F')])
go=vec(b['ev/goCue'],float)
```

iii. The AI identified the data as extracellular spike times and followed the reference operation `trialtm - goCue`, rather than applying imaging transforms.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned, histogrammed in 10 ms bins over -2.5 to +2.5 s, divided by 0.01 to obtain Hz, stacked as trial × neuron × time, and smoothed with a 15-bin centered uniform filter using nearest-edge padding.

ii.
```python
mats[remap[old]]=np.histogram(z,EDGES)[0]/DT
neural=np.stack(unit_counts,axis=1).astype(np.float32)
neural=uniform_filter1d(neural,size=15,axis=2,mode='nearest').astype(np.float32)
```

iii. The AI interpreted Figure 8a-c population parameters as 10 ms bins and smoothing width 15, and described `mySmooth` as a centered boxcar.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units with lower-cased quality in `{'garbage','','noisy','real?'}` are removed. Remaining units must have mean rate strictly above 1 Hz, calculated from spikes in the five-second aligned window across all native trials.

ii.
```python
ARTIFACT={'garbage','','noisy','real?'}
rate=np.sum((aligned_all>=TMIN)&(aligned_all<TMAX))/(ntr*(TMAX-TMIN))
if q not in ARTIFACT and rate>1:
```

iii. The AI says population analyses retain curated single and multiunits above 1 Hz, and excludes only clear artifacts/questionable or blank labels. It intentionally retains `poor`, unlike the human reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each spike, the trial's `goCue` is subtracted from within-trial spike time before histogramming.

ii.
```python
z=tm[(tr==old)]-go[old]
mats[remap[old]]=np.histogram(z,EDGES)[0]/DT
```

iii. The AI cites the reference `alignSpikes` operation and says no time warping is appropriate for direct go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses 10 ms bins (500 bins across five seconds). Raw spike events are histogrammed directly to this grid; video streams are interpolated to its centers.

ii.
```python
DT=.01; TMIN=-2.5; TMAX=2.5
EDGES=np.arange(TMIN,TMAX+DT/2,DT)
TIME=(EDGES[:-1]+EDGES[1:])/2
```

iii. The AI selected the 10 ms Figure 8a-c population setting rather than the 5 ms setting it associated with Figure 8d.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived from the configured -2.5 to +2.5 s analysis window and 10 ms bin edges; no raw trial value beyond the alignment definition is used in the stored input.

ii.
```python
TIME=(EDGES[:-1]+EDGES[1:])/2
```

iii. The AI planned the analytical bin centers as the continuous decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Adjacent bin-edge midpoints are computed, cast to float32, given shape `(1, 500)`, and copied for every trial.

ii.
```python
[TIME[None,:].astype(np.float32).copy() for _ in keep_trials]
```

iii. The AI says the vector should be `-2.495, ..., 2.495` seconds and verified it analytically.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. `TIME` contains the centers of the exact `EDGES` used to histogram go-cue-relative spikes, so corresponding neural and input columns represent the same bins.

ii.
```python
EDGES=np.arange(TMIN,TMAX+DT/2,DT); TIME=(EDGES[:-1]+EDGES[1:])/2
```

iii. The AI describes the input as the common neural time grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from behavioral `R`, `hit`, `miss`, and `no` flags (`L` is loaded but not needed).

ii.
```python
R,L,hit,miss,no,early,aw=map(get,['R','L','hit','miss','no','early','autowater'])
```

iii. The AI correctly notes that instructed side alone is not actual lick direction; misses must invert it and no-response trials need a separate class.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit maps to the instructed side, a miss to the opposite side, and `no` or any unclassified trial to none. The scalar class is repeated across all time bins.

ii.
```python
if no[tr]:lick=2
elif hit[tr]:lick=1 if R[tr] else 0
elif miss[tr]:lick=0 if R[tr] else 1
else:lick=2
```

iii. The AI says this represents actual choice and avoids confusing instructed side with behavior.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived directly from `bp.autowater`.

ii.
```python
aw=vec(b['autowater'],bool)
```

iii. The AI identified `autowater` as the WC/AW context indicator whose blocks match the paper.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `autowater=True` maps to WC code 0; false maps to DR code 1. It is repeated over time.

ii.
```python
context=0 if aw[tr] else 1
```

iii. The AI states that this mapping follows Figure 8 conditions.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses `hit` and `miss`; `no` and `early` are loaded but the code assigns ignore by falling through rather than explicitly using them.

ii.
```python
outcome=1 if hit[tr] else (0 if miss[tr] else 2)
```

iii. The AI planned correct, incorrect, and otherwise-ignore categories so required ignore trials could remain.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit becomes correct (1), miss incorrect (0), and every other case ignore (2), repeated across time.

ii.
```python
outcome=1 if hit[tr] else (0 if miss[tr] else 2)
```

iii. The AI treats retention of ignore as a decoder-task-required departure from paper analyses that omit them.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses side-view (`views[0]`) `traj` feature `tongue`: feature names, x/y coordinates, likelihood, frame times, plus the trial go cue. It does not combine both tongue camera views as the human solution does.

ii.
```python
ft,pos,lk=feature_trial(f,views[0],tr,'tongue')
tongue[j]=velocity_on_grid(ft,pos,lk,go[tr])
```

iii. The AI planned side-view DLC tongue landmark positions and said named features should be resolved rather than fixed columns.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Coordinates are linearly filled across finite samples, smoothed by a 21-frame uniform filter, finite-differenced and divided by timestamp differences, converted to Euclidean speed, invalidated where likelihood is not above 0.9, and linearly interpolated to the common grid. Nearest-frame visibility then restores gaps.

ii.
```python
pp[:,j]=uniform_filter1d(fill,size=21,mode='nearest')
speed=np.r_[np.nan,np.sqrt(np.sum(np.diff(pp,axis=0)**2,axis=1))/np.where(dt>0,dt,np.nan)]
speed[~visible]=np.nan
out[inside]=np.interp(TIME[inside],rel[good],speed[good])
```

iii. The AI says this follows reference position smoothing and finite-difference velocity while preserving invisibility rather than zero-filling it.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One median is computed per session over every finite aligned tongue sample. Finite values below it are 0, values equal to or above it are 1, and missing values remain 2.

ii.
```python
med=float(np.median(a[finite]))
out[finite]=(a[finite]>=med).astype(np.int64)
```

iii. The AI follows the prompt's per-session 50th-percentile rule and excludes missing samples from the threshold.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The code uses `frameTimes - goCue`, interpolating directly onto neural bin centers. It does not apply the required session video-to-behavior clock offset used in the human reference.

ii.
```python
rel=ft-go
out[inside]=np.interp(TIME[inside],rel[good],speed[good])
```

iii. The AI claimed native camera timestamps could be interpolated to the go-cue grid, but its notes do not justify omitting `findVideoOffset` clock correction.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity uses the second/bottom view's named `top_paw` x/y positions, likelihood and frame times, plus go cue.

ii.
```python
ft,pos,lk=feature_trial(f,views[1],tr,'top_paw')
paw[j]=velocity_on_grid(ft,pos,lk,go[tr])
```

iii. The AI selected `top_paw` as the relevant DLC feature and used named-feature lookup.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. It uses the same 21-frame fill/smooth, finite-difference Euclidean speed, likelihood >0.9 visibility, interpolation, and gap restoration as tongue velocity.

ii.
```python
paw[j]=velocity_on_grid(ft,pos,lk,go[tr])
```

iii. The AI planned the same reference-inspired velocity pipeline for tongue and paw.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A session-wide median over finite paw values defines below (0) and at/above (1); missing is 2.

ii.
```python
pd,pmed=disc(paw)
```

iii. The AI follows the required per-session median and preserves visibility missingness.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-view frame times have the trial go cue subtracted and are interpolated to `TIME`, but no session camera-clock offset is removed.

ii.
```python
rel=ft-go
```

iii. The AI says video outputs are interpolated to the neural grid without time warping; it did not document the omitted clock correction.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is read from the separate `motionEnergy_<subject>_<date>.mat` file's `me.data`, paired with side-view frame times and the trial go cue.

ii.
```python
m=loadmat(p,squeeze_me=True,struct_as_record=False)['me']
raw=np.asarray(m.data,dtype=object).ravel(order='F')
```

iii. The AI identified motion energy as a ragged per-trial stream corresponding one-to-one with camera frames and avoided re-reading files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Each trace is truncated with frame times to their shared minimum length, finite samples are linearly interpolated to the common bin centers, and the result is discretized. No additional spatial or temporal filtering is performed.

ii.
```python
n=min(len(y),len(ft)); y=y[:n]; rel=ft[:n]-go[tr]
me[j,inside]=np.interp(TIME[inside],rel[good],y[good])
```

iii. The AI says the source trace is already reduced to one value per frame and therefore only alignment/interpolation is needed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A session-wide median over finite values gives below (0) and at/above (1); missing/no-video bins are 2.

ii.
```python
md,mmed=disc(me)
```

iii. This implements the prompt's per-session 50th percentile and keeps absent video separate.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are paired with motion-energy samples, `goCue` is subtracted, and values are interpolated onto neural bin centers. The video clock offset is not corrected.

ii.
```python
ft=video_times[j]
rel=ft[:n]-go[tr]
me[j,inside]=np.interp(TIME[inside],rel[good],y[good])
```

iii. The AI states samples correspond one-to-one with camera frames, but does not justify bypassing the reference clock-offset calculation.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Feature/frame arrays are truncated to their common minimum length; absent/short tracks return all NaN; nonfinite or low-likelihood video stays missing and becomes category 2. Missing/unreadable motion files return `None`, making the session stream all class 2. Loader exceptions issue warnings. Trials with invalid go cues are dropped.

ii.
```python
n=min(len(ft),len(x))
if ft is None or len(ft)<2:return out
except Exception as e:
    warnings.warn(...);return None
```

iii. The AI emphasizes that missing video must not be silently treated as low velocity and that native trial correspondence should be preserved.

## 11-a. What are the most time-consuming steps of the code?

i. The AI identifies recursive full MAT loading as problematic and instead loads selected HDF5 references. In the final code, per-unit spike reading/histogramming and per-trial video interpolation are the principal repeated work; sample conversion was about three seconds per session.

ii.
```python
for q,rr,rt in zip(qualities,trialrefs,tmrefs):
...
for j,tr in enumerate(keep_trials):
```

iii. Its notes say initial `mat73` recursive loading stalled, while targeted access predicted the full 12-session run would finish in under a minute.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested unit/trial neural histogram loop could be replaced by a 2D histogram over trial and aligned time. Output assembly and some video operations could be batched, although ragged camera traces still require per-trial handling.

ii.
```python
for q,rr,rt in zip(...):
    ...
    for old in np.unique(tr[use]):
        mats[remap[old]]=np.histogram(z,EDGES)[0]/DT
```

iii. The AI claims histogram construction was vectorized “where possible,” but the final neural implementation still loops over retained trials for every unit; it justifies trial-wise video work by ragged traces.

## 11-c. What processing does the code repeat multiple times?

i. `feature_trial` repeatedly resolves feature names and dereferences trajectory/frame-time data for each trial and feature. Side frame times are loaded once for tongue and again for `video_times`; `velocity_on_grid` repeats identical smoothing/alignment logic for tongue and paw. A separate copy of `TIME` is also made for every trial.

ii.
```python
ft,pos,lk=feature_trial(...,'tongue')
ft,pos,lk=feature_trial(...,'top_paw')
video_times.append(deref_array(f,views[0]['frameTimes'],tr)...)
```

iii. The AI says it avoided redundant motion-file reads, but does not discuss these smaller repeated dereferences and copies.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `L` and `early` without using them directly, computes and returns continuous `(tongue, paw, me)` arrays even when plots are disabled, passes an unused `show` argument to `convert_session`, and stores per-unit mean rates only as metadata. The full conversion also returns `cont` to `main` although it is used only for optional plots of the first two sessions.

ii.
```python
R,L,hit,miss,no,early,aw=map(get,...)
return ..., info, (tongue,paw,me)
```

iii. The AI's notes focus on targeted loading and do not identify these final-code leftovers; they are relatively small compared with neural/output storage.
