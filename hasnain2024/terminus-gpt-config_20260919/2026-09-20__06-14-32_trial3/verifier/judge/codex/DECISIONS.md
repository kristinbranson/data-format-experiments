# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI parses uncommented session/probe entries from the authors' MATLAB loader scripts, intersects them with `data_structure_*.mat` files in both ephys folders, and loads v7.3 files with `h5py` or older files with `scipy.io.loadmat`. Companion motion-energy files are loaded separately.

ii. `for f in sorted(DATA.glob('*Ephys_Behavior/data_structure_*.mat')):` … `if key in active and active[key]: out.append((key,f,active[key]))`; and `load_h5(f,probes) if is_h5 else load_v5(f,probes)`.

iii. The notes say this reproduces the 44 active, available author sessions (25 fixed-delay and 19 randomized-delay) and their selected probes while excluding commented, unavailable, and clusterless recordings.

## 1-b. How are the data split into subjects?

i. Subject IDs are taken from the filename/session key before the first underscore. Unique IDs are sorted, and each session receives an index into that list.

ii. `animals.append(key.split('_')[0])`; `subjects=sorted(set(animals)); subject_idx=np.asarray([subjects.index(a) for a in animals],dtype=np.int64)`.

iii. The notes report 14 subjects and treat the filename animal ID as the stable source identifier.

## 1-c. How are the data split into sessions?

i. Each active `data_structure_<animal>_<date>.mat` is one session and one element of the top-level neural/input/output lists. Sessions with fewer than two converted trials would be skipped.

ii. `for i,(key,f,probes) in enumerate(sessions): ...`; `if len(n)<2: ... continue`; `neural.append(n); inputs.append(x); outputs.append(y)`.

iii. The AI justifies the selection by the authors' active loader entries and reports all 44 sessions survived.

## 1-d. How are the data split into trials?

i. Per-trial behavioral arrays define the trial count and zero-based trial IDs. Spikes carry one-based trial IDs, which are converted to zero-based. Each retained trial becomes one matrix in each session list.

ii. `B['n']=len(B['go'])`; `tr=np.asarray(...,int).ravel()-1`; `for j,tid in enumerate(tids): trialsN.append(neural[j])`.

iii. The notes explicitly check the MATLAB-to-Python index conversion and report 13,762 retained trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have native ephys, a finite positive go cue, no early lick, and no stimulation. After neural construction, trials with zero activity in every retained neuron and bin are also removed to handle erroneous trailing validity flags.

ii. `valid=B['haveE']&np.isfinite(B['go'])&(B['go']>0)&(~B['early'])&(~B['stim'])`; `neural_valid=np.any(neural!=0,axis=(1,2))`.

iii. The AI says early/stim removal follows canonical analyses, ignores are retained because requested as an output, and 61 trailing JEB24 trials were removed because recording had ended despite native flags.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from selected `obj.clu` probes: each cluster's `quality`, `trial`, and `trialtm`, plus trial-specific `bp.ev.goCue`.

ii. `q=...quality...`; `tr=np.asarray(...['trial']...)-1`; `tm=np.asarray(...['trialtm']...)`; `al=tm[ok]-B['go'][tr]`.

iii. The notes identify these as the reference spike fields and follow probe choices from the author loaders.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to go cue, histogrammed into 5-ms bins, divided by 0.005 to produce spikes/s, and convolved with a 15-point one-sided Gaussian-window kernel. No normalization or z-scoring is applied.

ii. `mat[oi]=np.histogram(vals,EDGES)[0].astype(np.float32)/DT`; `kern[:N//2]=0; kern/=kern.sum()`; `np.convolve(row,kern,mode='same')`.

iii. The AI claims this is the exact reference `mySmooth` causal `gausswin(15)` convention, after correcting an earlier symmetric implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `garbage` and misspelled `gabrga` labels are excluded; all other curated labels are retained. Units must have a raw-window mean firing rate strictly above 0.5 Hz.

ii. `BADQ={'garbage','gabrga'}`; `if q in BADQ: continue`; `if rate<=0.5: continue`.

iii. The notes prefer executable-code defaults and the near-match of 2,498 units to the paper's 2,496, despite paper prose specifying above 1 Hz.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time has that trial's go-cue time subtracted before binning.

ii. `al=tm[ok]-B['go'][tr]`.

iii. The notes say this exactly follows `alignSpikes` and requires no interpolation because both values share the behavioral clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data use 1000 non-overlapping 5-ms bins over [-2.5, 2.5) seconds. Raw spikes are rebinned into this grid; smoothed values remain at 5-ms resolution.

ii. `DT=0.005; TMIN=-2.5; TMAX=2.5`; `EDGES=np.arange(TMIN,TMAX+DT/2,DT)`.

iii. The AI ties this to the reference 200-Hz object axis and documents bin centers from -2.4975 to 2.4975 seconds.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a constructed common time axis based on the requested go-cue alignment and the reference window/bin size, rather than a changing raw trial variable.

ii. `TIME=((EDGES[:-1]+EDGES[1:])/2).astype(np.float32)`.

iii. The notes justify it as the common signed-seconds axis required by the decoder.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Adjacent bin edges are averaged to obtain 1000 centers, then the same 1×1000 float32 row is copied for every trial.

ii. `trialsI.append(TIME[None,:].copy())`.

iii. The AI notes this avoids edge/center ambiguity and matches histogram bins.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of the exact edges used to histogram go-cue-aligned spikes.

ii. `np.histogram(vals,EDGES)` and `TIME=((EDGES[:-1]+EDGES[1:])/2)`.

iii. Independent checks in the notes report exact equality to a separately generated time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from `bp.L`, `bp.R`, `bp.hit`, and `bp.miss`; neither outcome produces the none class.

ii. `lick=0 if (B['L'][tid] and B['hit'][tid]) or (B['R'][tid] and B['miss'][tid]) else 1 if ... else 2`.

iii. The notes explain that hits follow the instructed side, misses imply the opposite side, and nonresponses are none.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Actual left is encoded 0, actual right 1, and no lick 2, then repeated across all time bins of the trial.

ii. `np.full(len(TIME),lick,np.int8)`; output values are `['left','right','none']`.

iii. The AI says this matches reference choice-condition definitions and verified raw examples.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is taken from the per-trial `bp.autowater` flag.

ii. `B={k:... for k in (...,'autowater')}`.

iii. The notes identify autowater as the direct WC-versus-DR indicator.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The boolean is cast directly to integer, encoding DR=false as 0 and WC=true as 1, and repeated over time.

ii. `np.full(len(TIME),int(B['autowater'][tid]),np.int8)`; output values are `['DR','WC']`.

iii. The AI's mapping plan explicitly states `0=DR`, `1=WC`; metadata is consistent with that choice.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses `bp.hit` and `bp.miss`; all other retained trials are treated as ignore (although `bp.no` is loaded).

ii. `outcome=1 if B['hit'][tid] else 0 if B['miss'][tid] else 2`.

iii. The notes state the requested ignore category requires retaining nonresponse trials omitted from some paper analyses.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss is incorrect 0, hit is correct 1, and neither is ignore 2; the scalar is repeated through the trial.

ii. `np.full(len(TIME),outcome,np.int8)`; output values are `['incorrect','correct','ignore']`.

iii. This is presented as a direct relabeling of mutually exclusive behavior flags.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses the side-camera `tongue` feature from `obj.traj`: frame times and tracked x/y coordinates (whose NaNs represent invisibility), plus `bp.ev.goCue` and `haveVid`. It does not use the bottom-camera `top_tongue` feature.

ii. `tongue=feature_speed(traj,tids,0,'tongue',B)`; inside, `ft,ts,names=z` and `xy=a[:m,:2,fi]`.

iii. The notes planned a consistently named tongue point and say visibility must come from tracking NaNs, not zero velocity.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. X/y differences are computed with `np.gradient` per frame, combined by Euclidean magnitude at visible frames, linearly interpolated to bin centers, and masked using a linearly interpolated raw visibility indicator. There is no position smoothing, division by frame-time intervals, second camera, or view normalization.

ii. `dx=np.gradient(xy[:,0]); dy=np.gradient(xy[:,1]); vel[vis]=np.hypot(dx[vis],dy[vis])`; `speed=interp_visible(rel,vel,TIME)`.

iii. The notes describe deriving speed magnitude, aligning/interpolating it, and preserving invisibility; they do not acknowledge the omitted second view or time derivative.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One session-wide `nanmedian` over all retained trial/time values is used: finite values below it are 0, values at or above it 1, and NaNs 2 (not visible).

ii. `med=float(np.nanmedian(x))`; `y[ok]=(x[ok]>=med).astype(np.int8)`.

iii. This follows the prompt's per-session 50th-percentile split and preserves missingness as its own class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Raw frame times have the trial go cue subtracted and the resulting samples are interpolated onto neural bin centers. No session video-to-behavior clock offset is computed or subtracted.

ii. `rel=ft-B['go'][tid]`; `speed=interp_visible(rel,vel,TIME)`.

iii. The notes claim video features are put on the common 5-ms go-cue axis, but do not justify omission of the reference bitcode clock correction.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera `top_paw` tracking: frame times, x/y coordinates/NaNs, go cue, and `haveVid`.

ii. `paw=feature_speed(traj,tids,1,'top_paw',B)`.

iii. The mapping plan chooses the bottom-view top paw as a consistent tracked feature.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. It uses the same frame-index gradients, magnitude, interpolation, and visibility masking as tongue, without smoothing or division by elapsed time.

ii. `dx=np.gradient(xy[:,0]); dy=np.gradient(xy[:,1])`; `speed=interp_visible(rel,vel,TIME)`.

iii. The notes characterize this as speed magnitude aligned/interpolated to 5 ms while preserving missingness.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A session-wide median of finite values defines class 0 below and class 1 at/above; NaNs are class 2.

ii. `pd,pmed=discretize(paw)`.

iii. The AI cites the mandated session median and reports near-balanced visible classes.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are shifted only by trial go cue and interpolated onto `TIME`; the session video-clock offset is omitted.

ii. `rel=ft-B['go'][tid]`; `speed=interp_visible(rel,vel,TIME)`.

iii. The notes assert common-axis alignment but provide no bitcode-offset calculation.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses per-trial values from the companion `motionEnergy_*.mat` file and side-camera frame times, plus go cue and video availability.

ii. `mf=f.with_name(f.name.replace('data_structure_','motionEnergy_'))`; `raw=me.data`; `ft=traj[0][tid][0]`.

iii. The notes say every selected session has a companion file and document unwrapping legacy nested `data` fields.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is linearly interpolated to 5-ms centers and then every remaining internal or exterior NaN is nearest-filled. It is not smoothed or recomputed from pixels.

ii. `z=interp_visible(rel,y[:m],TIME); z=nearest_fill(z).astype(np.float32)`.

iii. The AI says this follows the reference interpolation/nearest-fill behavior and reserves class 2 for absent video rather than ordinary sampling gaps.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. One session-wide median over finite aligned values gives below=0 and at/above=1; NaNs become 2 (`no_video`).

ii. `md,mmed=discretize(motion)`.

iii. The task-mandated median overrides the paper's manually selected movement threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are shifted by trial go cue and interpolated to neural bin centers, followed by nearest filling. The video/behavior clock offset is omitted.

ii. `rel=np.asarray(ft[:m])-B['go'][tid]`; `interp_visible(rel,y[:m],TIME)`.

iii. The notes claim frame-time/go-cue interpolation matches the reference common axis.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing trajectories/video become all-NaN continuous streams and class 2 after discretization; visibility masks prevent interpolation across invisible intervals. Motion gaps are nearest-filled when any samples exist. Missing motion files yield class 2. All-zero neural trials are dropped as invalid trailing recording periods.

ii. `if z is None or not B['haveV'][tid]: out.append(np.full(...,np.nan))`; `speed[~vi]=np.nan`; `z=nearest_fill(z)`; `neural_valid=np.any(neural!=0,axis=(1,2))`.

iii. The notes emphasize categorical missingness, identify one missing-video/motion trial, and document the 61 erroneous trailing ephys trials.

## 11-a. What are the most time-consuming steps of the code?

i. The AI identifies large raw trajectory reads and per-trial MATLAB-reference traversal as the main costs, with spike histogram/convolution loops also contributing.

ii. `for ref in h['obj/traj'][:,0]: ... for j in range(B['n']):`; and nested unit/trial work in `neural_arrays`.

iii. The notes report roughly 3–6 minutes estimated for full conversion and say lazy HDF5/session-wise processing reduced cost.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested per-unit/per-trial spike histogram loop and per-row convolution could be vectorized across trials (as in a 2-D histogram). Ragged trajectory decoding and per-trial feature processing are less directly vectorizable.

ii. `for q,tr,tm in units:` then `for oi,tid in enumerate(trial_ids):`; `np.stack([np.convolve(row,kern,mode='same') for row in mat])`.

iii. The notes acknowledge that single-trial spike histograms require grouping, while claiming vectorized alignment and compact processing as speedups.

## 11-c. What processing does the code repeat multiple times?

i. `feature_speed` repeats trajectory layout normalization, gradient/interpolation, and visibility-mask interpolation separately for tongue and paw. Neural smoothing repeats convolution per trial row, and the output loop repeatedly allocates constant time series.

ii. `tongue=feature_speed(...)`; `paw=feature_speed(...)`; `np.full(len(TIME),lick,...)`, `np.full(...context...)`, `np.full(...outcome...)`.

iii. The notes do not explicitly identify these repetitions; they frame the common feature function and shared global grid as implementation reuse.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads all tracked features and all selected trials' full trajectory arrays even though only `tongue` and `top_paw` are used. It also computes and stores unit mean rates and extensive session metadata not consumed by decoder training, and duplicates per-trial time/constant label rows.

ii. `ts=np.asarray(...); names=[...]` for every trajectory; `info={...,'mean_rates_hz':rates,...}`; `TIME[None,:].copy()`.

iii. The notes mainly justify metadata for sanity checks and compact dtypes; they do not discuss these values being unused by downstream training.
