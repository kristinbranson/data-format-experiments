# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent glob-loads only the 11 hard-coded Figure-8 sessions present in `/app/data/Ephys_Behavior`, using `mat73` for each `data_structure` file and SciPy for an optional motion-energy file. It does not load the randomized-delay folder or the other paper sessions.

ii. `for f in glob.glob(ROOT+'/data_structure_*.mat'):` and `if (mouse,day) in PROBES: files.append((mouse,day,f))`; later, `o=mat73.loadmat(f)['obj']`.

iii. The trajectory says Figure 8 explicitly uses six mice and concludes that its 11 released sessions are the “authoritative subset.”

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from filenames, constrained to six hard-coded mice, sorted, and indexed once per session.

ii. `mouse,day=base.split('_',1)`; `subjects=sorted(MICE)`; `subject_idx.append(subjects.index(mouse))`.

iii. The trajectory identifies the Figure-8 mice from the repository scripts.

## 1-c. How are the data split into sessions?

i. Each selected `data_structure_<mouse>_<date>.mat` file becomes one session-level element of `neural`, `input`, and `output`; files are sorted lexically.

ii. `files.sort()` and `for mouse,day,f in files:` followed by `neural.append(ns); inputs.append(ins); outputs.append(outs)`.

iii. The agent treated the 11 files selected by its `PROBES` map as the released two-context sessions.

## 1-d. How are the data split into trials?

i. Trials are the indices `0..Ntrials-1` in `obj.bp`; retained trial indices address behavior, spike, trajectory, and motion-energy arrays.

ii. `n=int(b['Ntrials'])`; `keep=np.flatnonzero(~early)`; `for j,t in enumerate(keep):`.

iii. The trajectory recognized the saved objects as holding per-trial behavior, spike-trial labels, trajectories, and motion-energy traces.

## 1-e. How are trials filtered based on quality controls?

i. Only early-lick trials are excluded. Ignore/no-response trials are deliberately retained. Photostimulation trials and trials beyond valid ephys recording are not filtered.

ii. `early=np.asarray(b['early']).astype(bool); keep=np.flatnonzero(~early)`.

iii. The agent cites the paper for omitting early licks and retains ignores because the requested outcome has an `ignore` class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the selected `obj.clu` probe(s), specifically each unit’s `trial` and `trialtm`, plus `bp.ev.goCue` for alignment.

ii. `for tr,tm in zip(c['trial'],c['trialtm']): units.append((np.asarray(tr,int).ravel()-1,np.asarray(tm,float).ravel()))`.

iii. The trajectory notes that `trialtm` is relative to trial start and must have the corresponding go cue subtracted.

## 2-b. How is the `neural` data processed?

i. Per-unit spikes are histogrammed into 5-ms bins, divided by bin width to obtain Hz, and smoothed separately per trial with a one-sided 15-bin Gaussian whose first seven weights are zeroed. Selected probes are concatenated; there is no normalization.

ii. `h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT`; in `smooth_rates`, `k[:7]=0; k/=k.sum()` and `np.convolve(z,k,mode='same')`.

iii. The trajectory explicitly chose repository-style “causal 15-bin Gaussian smoothing” and adjusted its standard deviation to the MATLAB `gausswin(15)` equivalent.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are retained only when their mean rate over all retained trials and time bins exceeds 1 Hz. Manual `quality` labels are ignored.

ii. `use=rates.mean(axis=(1,2))>1.0; rates=rates[use]`.

iii. The agent cites the paper’s >1-Hz inclusion rule and says repository filtering uses mean trial-averaged activity.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike’s trial-relative time has that trial’s go-cue time subtracted before histogramming.

ii. `np.histogram(tm[tr==t]-go[t],EDGES)`.

iii. The trajectory found that `trialtm` is relative to trial start and that repository alignment subtracts the selected event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Data use 1000 non-overlapping 5-ms bins from −2.5 to +2.5 seconds. Spikes are rebinned into this grid; camera streams are nearest-interpolated to its centers.

ii. `DT=.005; T0=-2.5; T1=2.5`; `EDGES=np.arange(T0,T1+DT/2,DT); TIME=EDGES[:-1]+DT/2`.

iii. The agent says these values reproduce the repository parameters.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a generated grid of bin centers relative to the go cue, rather than a raw per-trial measurement.

ii. `TIME=EDGES[:-1]+DT/2` and `ins.append(TIME[None,:].astype(np.float32))`.

iii. The common go-cue-centered grid was chosen to match the paper and neural bins.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Consecutive 5-ms edge pairs are converted to their centers; the same row is copied into every retained trial.

ii. `TIME=EDGES[:-1]+DT/2`.

iii. No additional justification was given beyond use of the repository’s grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. `TIME` contains the centers of the exact `EDGES` used to bin go-cue-relative spikes.

ii. `h=np.histogram(...,EDGES)[0]` and `ins.append(TIME[None,:])`.

iii. The trajectory planned a single common go-cue-centered grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.L`, `bp.R`, `bp.hit`, `bp.miss`, and `bp.no` (with the neither-hit-nor-miss condition also treated as no lick).

ii. `L=np.asarray(b['L']).astype(bool); R=np.asarray(b['R']).astype(bool); hit=...; miss=...; no=...`.

iii. The agent reasoned that hits use the instructed side, misses use the opposite side, and no-response trials need a third class.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Codes are left 0, right 1, none 2. A hit maps to the instructed side, a miss to its opposite, and all other/no trials to none; the scalar class is repeated across time.

ii. `lick=2 if no[t] or (not hit[t] and not miss[t]) else (0 if (hit[t] and L[t]) or (miss[t] and R[t]) else 1)`.

iii. The trajectory describes this as “actual lick” inferred from instruction and outcome.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived directly from `bp.autowater`.

ii. `aw=np.asarray(b['autowater']).astype(bool)`.

iii. Repository Figure-8 code was found to use `autowater` directly as the WC/DR label.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater/WC is encoded 0 and its complement/DR is 1, then repeated over all bins.

ii. `np.full(TIME.size,int(not aw[t]))` with `output_values` equal to `['WC','DR']`.

iii. The trajectory caught and corrected an initial reversed encoding so numeric codes match the declared labels.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses `bp.hit`, `bp.miss`, and `bp.no`, with neither hit nor miss also classed as ignore.

ii. `hit=np.asarray(b['hit']).astype(bool); miss=...; no=...`.

iii. Ignore trials were retained specifically because the decoder specification requests this outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss is incorrect 0, hit is correct 1, and no/other is ignore 2; the code repeats the value through time.

ii. `outcome=2 if no[t] or (not hit[t] and not miss[t]) else (1 if hit[t] else 0)`.

iii. This directly follows the requested class semantics.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses `obj.traj[1]`, its `ts` coordinates/likelihoods and `frameTimes`, and assumes landmark indices 0–3 are tongue landmarks. `bp.ev.goCue` supplies alignment.

ii. `cam=traj[1]; ts=np.asarray(cam['ts'][trial],float); ft=np.asarray(cam['frameTimes'][trial],float).ravel()-go`; `tongue,tvis=speed([0,1,2,3])`.

iii. The trajectory chose camera-1 landmarks after inspecting feature layouts, aiming to preserve an explicit invisible state.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Coordinates below 0.9 likelihood are made NaN; visible landmark coordinates are averaged into a centroid; x/y gradients per frame form speed magnitude; speed and visibility are nearest-interpolated to the 5-ms grid. No position smoothing, real-time derivative, second view, or view normalization is used.

ii. `pos=np.nanmean(xy,axis=2)`; `vx=np.gradient(pos[:,0]); vy=np.gradient(pos[:,1]); sp=np.hypot(vx,vy)`; `interp_nearest(ft,sp,TIME)`.

iii. The trajectory states that MATLAB `findVelocity` uses per-frame gradients and says invisibility is preserved rather than zero-filled.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median over bins marked visible is used: below median 0, at/above median 1, invisible 2.

ii. `th=np.nanpercentile(x[vis],50)` and `z[vis]=(x[vis]>=th).astype(np.int8)`.

iii. The 50th-percentile session threshold and third visibility class come directly from the task.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Raw `frameTimes` have the trial’s go cue directly subtracted and are nearest-interpolated to `TIME`. No video-to-behavior clock offset is applied.

ii. `ft=np.asarray(cam['frameTimes'][trial],float).ravel()-go`; `interp_nearest(ft,sp,TIME)`.

iii. The trajectory assumed the saved frame times could simply be shifted by go cue; it did not discuss the required bitcode offset.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses the same `obj.traj[1]` stream, with landmark indices 4 and 5 treated jointly as paw landmarks, plus frame times and go cue.

ii. `paw,pvis=speed([4,5])`.

iii. The trajectory says it selected camera-1 landmarks based on the inspected coordinate layout.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same likelihood masking, multi-landmark centroid, per-frame gradient speed, and nearest interpolation used for tongue are applied.

ii. Inside `speed(ids)`: `xy=np.where((lk>=.9)[:,None,:],xy,np.nan)`, `pos=np.nanmean(...)`, and `np.gradient`.

iii. The agent intended a consistent visible/not-visible velocity pipeline.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A session median over visible interpolated values defines below 0 versus at/above 1; invisible is 2.

ii. `pc,pth=classes(paw,pvis)`.

iii. This implements the requested per-session 50th-percentile split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw frame times have go cue subtracted and are nearest-interpolated to the neural bin centers, without clock-offset correction.

ii. `return interp_nearest(ft,sp,TIME), ...` where `ft=...-go`.

iii. The agent’s plan was to align all video-derived signals through the same camera frame times.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It reads `me.data` from a separate `motionEnergy_<mouse>_<date>.mat` and pairs each trace with the selected camera’s frame times.

ii. `me=loadmat(mef,... )['me'].data`; `mv=np.asarray(me[t],float).ravel()`.

iii. The trajectory identified motion energy as one variable-length trace per trial with a matching trial count.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is trimmed to the shorter of trace and frame-time lengths, then nearest-interpolated onto `TIME`; no smoothing or spatial processing is performed.

ii. `m=min(len(mv),len(ft)); z=interp_nearest(ft[:m],mv[:m],TIME)`.

iii. The agent regarded the raw file as already containing the reduced per-frame motion-energy signal.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The median of finite/visible session samples defines below 0 versus at/above 1; missing/no-video bins are 2.

ii. `mc,mth=classes(mes,mvis)`.

iii. This follows the requested per-session median and no-video class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It reuses the camera `ft` already shifted only by go cue and nearest-interpolates motion energy to neural bin centers. It does not correct the video clock offset.

ii. `z=interp_nearest(ft[:m],mv[:m],TIME)`.

iii. The trajectory planned to align motion energy using camera frame times.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/invalid tracking is represented as class 2. Too few finite points yields an all-missing stream; missing motion files or mismatched trial/frame lengths are guarded. Interior gaps are nevertheless nearest-interpolated, and nonfinite go cues cause neural bins to remain zero.

ii. `if ok.sum()<2:return np.full(target.shape,fill,float)`; `m=min(len(mv),len(ft))`; otherwise `mes.append(np.full(TIME.size,np.nan))`.

iii. The agent explicitly wanted an invisible class rather than zero-filling and described the centroid warning as benign.

## 11-a. What are the most time-consuming steps of the code?

i. Full `mat73` deserialization and the nested unit/trial spike histogram-and-smoothing loops are the principal costs; video processing also loops through every retained trial.

ii. `o=mat73.loadmat(f)['obj']`; nested `for ui,...`, `for t in np.unique(tr):`, and `for t in keep:` loops.

iii. During development the trajectory observed that eagerly loading full objects was slow and that complete conversion took repeated multi-minute passes.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike counting loops over every unit and its unique trials, smoothing calls `apply_along_axis`, and output assembly loops over trials. Spike counts could be computed per unit across trials with a 2-D histogram; smoothing and constant-output construction could be batched.

ii. `for ui,(tr,tm) in enumerate(units): for t in np.unique(tr):`; `np.apply_along_axis(...)`; `for j,t in enumerate(keep):`.

iii. The trajectory prioritized direct implementation and manageable memory, but did not justify these loops as necessary.

## 11-c. What processing does the code repeat multiple times?

i. `smooth_rates` rebuilds and normalizes the same Gaussian kernel for every unit-trial histogram. Camera lookup, interpolation sorting, and array conversion are repeated per trial and feature.

ii. `rates[ui,j]=smooth_rates(h[None,:])[0]`; every `smooth_rates` call executes `k=gaussian(...); k[:7]=0; k/=k.sum()`.

iii. No explicit justification was provided for rebuilding invariant objects.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `mat73` loads the entire MATLAB object although only selected behavior, cluster, and trajectory fields are used. It computes and retains visibility arrays only until class creation, and computes threshold floats solely for metadata. Loaded unused probes and fields are discarded.

ii. `o=mat73.loadmat(f)['obj']`; later `del o,rates`.

iii. The trajectory itself abandoned an all-session eager scan because full-object loading unnecessarily deserialized large spike/video arrays, but retained the same loader in the converter for simplicity.
