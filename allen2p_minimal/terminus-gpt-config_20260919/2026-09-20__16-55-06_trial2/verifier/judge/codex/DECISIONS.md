# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers NWB files, opens them directly with `h5py`, and retains files whose `session_description` is one of four active-task session types. It does not use the AllenSDK cache or combine planes.

ii. `glob.glob(DATA_GLOB,recursive=True)` and `if text(h['session_description']) in ACTIVE: files.append(f)`; each retained file is later opened with `h5py.File(f,'r')`.

iii. The trajectory says direct HDF5 was chosen because AllenSDK eagerly loaded large dF/F arrays and was slow. Active OPHYS 1/3/4/6 sessions were included; passive sessions were excluded because they lack active Go/Catch behavior.

## 1-b. How are the data split into subjects?

i. Subjects are unique `general/subject/subject_id` strings across retained files, sorted globally and mapped to indices.

ii. `subjects=sorted({text(h5py.File(f,'r')['general/subject/subject_id']) for f in files})` and `subject_idx.append(subjmap[mouse])`.

iii. The trajectory identifies these as the NWB mouse identifiers and reports 38 subjects.

## 1-c. How are the data split into sessions?

i. Every retained NWB/ophys experiment (one imaging plane) becomes an independent output session; simultaneous planes are not grouped by `ophys_session_id`.

ii. `for fi,f in enumerate(files): ... neural.append(sn)` and session metadata stores `ophys_experiment_id`.

iii. The agent reasoned that each NWB had its own neuronal population and timestamps and was valid as a decoder session, even after noticing simultaneous multiplane files with identical trials.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials`. Each retained row is represented by a fixed window from 3.0 s before through 4.2 s after `change_time`, divided into 72 100-ms bins.

ii. `keep=np.flatnonzero((go|catch)&~abort&~auto&np.isfinite(change))` and `edges=change[j]+OFF0+np.arange(NB+1)*DT`.

iii. The trajectory found that trial starts vary but valid trials share approximately -3.0 to +4.2 s around change, so it selected that common fixed interval.

## 1-e. How are trials filtered based on quality controls?

i. Only Go or Catch trials are retained; aborted, auto-rewarded, and nonfinite-change-time trials are removed. Files yielding fewer than two trials are omitted.

ii. `keep=np.flatnonzero((go|catch)&~abort&~auto&np.isfinite(change))` and `if len(sn)>=2:`.

iii. The trajectory explicitly ties this to the instruction to include Go/Catch and exclude aborted/auto-rewarded trials; finite change time enables alignment.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from detected calcium events and their timestamps.

ii. `ev=h['processing/ophys/event_detection/data']` and `ot=np.asarray(h['processing/ophys/event_detection/timestamps'],dtype=float)`.

iii. The agent cited the paper methods as favoring detected events rather than dF/F.

## 2-b. How is the `neural` data processed?

i. For each trial, event frames are assigned to 100-ms bins and averaged cell-wise; empty bins remain zero. Only the trial's contiguous event slab is read.

ii. `bi=np.floor((tt-edges[0])/DT).astype(int)` followed by `z=slab[bi==k]` and `mat[:,k]=z.mean(axis=0)`.

iii. The trajectory says 100-ms binning makes the full dataset tractable and is suitable for relatively slow calcium dynamics; slab reads avoid full-recording loads.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional cell-level filtering is performed; all cells in the NWB event matrix are used.

ii. `ncell=ev.shape[1]` and every column of `ev[a:b,:]` is retained.

iii. The trajectory calls them curated cells and relies on released NWB processing; it documents no extra QC threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to scheduled image change time (real Go change or Catch sham change), from -3.0 to +4.2 seconds.

ii. `edges=change[j]+OFF0+np.arange(NB+1)*DT`, with metadata `temporal_alignment_event` set to scheduled image change.

iii. The common peri-change interval was selected after inspecting valid-trial relative boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 100 ms. Native event samples are rebinned by their mean into 72 bins.

ii. `DT=0.100`, `NB=int(round((OFF1-OFF0)/DT))`, and the per-bin mean loop.

iii. The agent justified this as a tractability/dynamics tradeoff; it is deliberate temporal rebinning.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from the first stimulus-presentation TimeSeries' `timestamps` and integer `data`, plus the A/B session type.

ii. `pg=h['stimulus/presentation/'+preskeys[0]]`, `stim_t=np.asarray(pg['timestamps'])`, and `stim_i=np.asarray(pg['data'])`.

iii. The trajectory inspected raw HDF5 paths and identified these as presentation onset/index data.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Gray is code 0. Valid indices 0–7 are offset to distinct A-image codes 1–8 or B-image codes 9–16. A bin is an image only during the 250-ms display after onset; omissions/out-of-range values are gray.

ii. `shown=ok&(centers<stim_t[pp]+.25)&(stim_i[pp]<8)` and `img[shown]=(setoff+stim_i[pp[shown]].astype(int))`.

iii. The agent reasoned that familiar and novel sets need separate global classes and that the screen is gray outside each 250-ms presentation.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at the same 100-ms bin centers used for the neural trial window.

ii. `centers=(edges[:-1]+edges[1:])/2` and `pos=np.searchsorted(stim_t,centers,side='right')-1`.

iii. The trajectory states all streams are aligned by absolute synchronized timestamps.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived solely from each retained trial's `change_time`.

ii. `change=np.asarray(tr['change_time'],dtype=float)` and `ch[np.argmin(abs(centers-change[j]))]=1`.

iii. The agent treated `change_time` as the scheduled Go change/Catch sham-change alignment event.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created and exactly one nearest 100-ms bin is marked one for every retained Go and Catch trial.

ii. `ch=np.zeros(NB,dtype=np.int16); ch[np.argmin(abs(centers-change[j]))]=1`.

iii. The trajectory planned a one-frame pulse at change onset.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: 0 is “no change” and 1 is “image change”; no continuous threshold is used.

ii. `output_values` supplies `['no change','image change']`.

iii. No separate justification was recorded beyond the requested binary output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The nearest neural/behavior bin center to `change_time` is marked, which is the central alignment bin.

ii. `np.argmin(abs(centers-change[j]))`.

iii. The agent intended all labels to use the same peri-change bins.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `processing/running/speed/data` and its timestamps.

ii. `rt=np.asarray(h['processing/running/speed/timestamps'])` and `rv=np.asarray(h['processing/running/speed/data'])`.

iii. The trajectory identifies this as the synchronized native running stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite samples are linearly interpolated to bin centers, then categorized using quintile cut points computed from the full raw running stream of that NWB file.

ii. `run=interp_clean(rt,rv,centers)`, `rq=quintile_reference(rv)`, and `runbin=np.digitize(run,rq)`.

iii. The agent planned session-wide percentile bins so each recording has five behavioral levels.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 20th, 40th, 60th, and 80th within-file quantiles form five codes 0–4.

ii. `np.quantile(x,[.2,.4,.6,.8])` and `np.digitize(run,rq)`.

iii. This directly implements the requested five equal-percentile bins, interpreted within session/file.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated at the same trial-bin centers as the neural bins.

ii. `run=interp_clean(rt,rv,centers)`.

iii. The agent relied on the common synchronized ophys clock.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking timestamps and `pupil_tracking/width`.

ii. `pt=np.asarray(h['acquisition/EyeTracking/eye_tracking/timestamps'])` and `pv=np.asarray(h[pbase+'/width'])`.

iii. The trajectory explicitly identified tracked pupil width as the available diameter measure.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Finite width values are linearly interpolated to bin centers and digitized using within-file raw-width quintiles. If eye tracking is absent, zero arrays are produced.

ii. `pup=interp_clean(pt,pv,centers)`, `pq=quintile_reference(pv)`, and `pupbin=np.digitize(pup,pq)`.

iii. The agent planned the same session-wide percentile approach as running; no blink removal was discussed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Raw pupil-width quantiles at 20/40/60/80% define codes 0–4.

ii. `pq=quintile_reference(pv)` and `np.digitize(pup,pq)`.

iii. This is the agent's interpretation of five equal-percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Width is interpolated at the identical 100-ms centers used for neural data.

ii. `pup=interp_clean(pt,pv,centers)`.

iii. The agent relied on synchronized NWB clocks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It comes from trial-table `hit`, `miss`, `false_alarm`, and `correct_reject` flags.

ii. `hit=np.asarray(tr['hit']); miss=...; fa=...; cr=...`.

iii. These were recognized as the canonical four Go/Catch outcomes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Priority logic maps hit/miss/false alarm/correct reject to 0/1/2/3, then repeats the static code over all 72 bins.

ii. `outcome=0 if hit[j] else 1 if miss[j] else 2 if fa[j] else 3` and `np.full(NB,outcome,dtype=np.int16)`.

iii. The trajectory says repetition was used because the validator expects a rectangular time-varying output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Nonfinite behavioral samples are removed before interpolation/quantiles; fewer than two valid samples yield zeros; absent pupil tracking yields empty inputs and therefore zeros; nonfinite change times are excluded; empty quantile sources return four `-inf` edges. There is no per-file exception recovery.

ii. `good=np.isfinite(t)&np.isfinite(x)`, `if good.sum()<2: return np.zeros(...)`, and `else: pt=np.array([]); pv=np.array([])`.

iii. The trajectory aimed for robust conversion of missing pupil data and valid alignment, but did not discuss malformed files or blink flags.

## 9-a. What are the most time-consuming steps of the code?

i. Reading large event slabs from 202 NWBs and repeatedly selecting/averaging event frames for every trial and time bin dominate runtime; serialization of the large pickle is also material.

ii. `slab=np.asarray(ev[a:b,:],dtype=np.float32)` inside the trial loop and the nested `for k in range(NB)`.

iii. The trajectory observed that SDK full-array loading was prohibitively slow and monitored the direct-HDF5 conversion for several minutes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested file, trial, and 72-bin loops—especially `slab[bi==k].mean(axis=0)`—could be replaced with grouped sums/counts (`np.add.at` or equivalent). Image and outcome assembly are already vectorized per trial.

ii. `for j in keep:` containing `for k in range(NB):`.

iii. The agent prioritized contiguous slab reads and tractable memory, but recorded no attempt to vectorize the bin aggregation.

## 9-c. What processing does the code repeat multiple times?

i. It rereads identical behavior/trial/stimulus streams and recomputes quintiles for every plane-level NWB, including simultaneous planes from the same behavioral session. Trial labels and interpolated behavior are also recomputed per plane.

ii. All behavior extraction and `rq=quintile_reference(rv); pq=quintile_reference(pv)` occur inside `for fi,f in enumerate(files)`.

iii. The trajectory noticed identical trials across simultaneous planes but retained each file as its own session, accepting this repetition.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `cr` but never reads it in outcome assignment (correct reject is merely the fallback); it also constructs bin `edges` although only centers and endpoints are ultimately needed. More importantly, duplicated behavior processing across planes produces repeated labels rather than reusable session-level results.

ii. `cr=np.asarray(tr['correct_reject'])` is unused, and `outcome=... else 3` does not test `cr[j]`.

iii. The trajectory does not identify these as unnecessary; it emphasizes direct HDF5 efficiency and successful validation.
