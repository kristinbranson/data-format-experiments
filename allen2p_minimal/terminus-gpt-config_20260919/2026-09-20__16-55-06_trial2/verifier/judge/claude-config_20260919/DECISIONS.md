# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI bypassed the AllenSDK entirely and read the released NWB (HDF5) files directly with `h5py`. All 284 `*.nwb` files under `/app/data` are globbed, each is opened once to read its `session_description`, and only the four **active** behavior stages (`OPHYS_1_images_A`, `OPHYS_3_images_A`, `OPHYS_4_images_B`, `OPHYS_6_images_B`) are kept — 202 of the 284 supplied experiments. The two passive replay stages (`OPHYS_2_images_A_passive`, `OPHYS_5_images_B_passive`, 82 files) are dropped. No filtering by `project_code` is done, so the 34 active `VisualBehaviorMultiscope` plane-files (all from mouse 457841) are included alongside the 168 active single-plane `VisualBehavior` experiments. Within each file, only the datasets actually needed are read (`event_detection/data` slabs, `intervals/trials` columns, running speed, eye tracking, stimulus presentation), never the full dF/F arrays.

ii.
```python
DATA_GLOB='/app/data/**/*.nwb'
ACTIVE={'OPHYS_1_images_A','OPHYS_3_images_A','OPHYS_4_images_B','OPHYS_6_images_B'}
...
    files=[]
    for f in sorted(glob.glob(DATA_GLOB,recursive=True)):
        with h5py.File(f,'r') as h:
            if text(h['session_description']) in ACTIVE: files.append(f)
```
```python
    for fi,f in enumerate(files):
      with h5py.File(f,'r') as h:
        stype=text(h['session_description']); mouse=text(h['general/subject/subject_id'])
        expid=int(text(h['identifier'])); setoff=1 if 'images_A' in stype else 9
        ev=h['processing/ophys/event_detection/data']
        ot=np.asarray(h['processing/ophys/event_detection/timestamps'],dtype=float)
```

iii. From the trajectory: the AI first tried the SDK (`BehaviorOphysExperiment.from_nwb_path`) for an inventory of all 284 files and abandoned it — "The SDK inventory is far too slow (only 20/284 after another 30 seconds)… `from_nwb_path` eagerly reads full dF/F arrays even though only metadata were requested. Raw h5py access is therefore essential both for inventory and for an efficient final converter." Passive sessions were excluded because "passive sessions cannot supply valid Go/Catch trials anyway" and "they are not active Visual Behavior trials"; all four active stages were kept because "Using only OPHYS_1 would omit many supplied active task recordings".

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the unique values of the NWB field `general/subject/subject_id`, collected over the retained (active) files and sorted. This yields 38 mice. Each session's `subject_idx` is the index of its file's `subject_id` in that sorted list.

ii.
```python
    subjects=sorted({text(h5py.File(f,'r')['general/subject/subject_id']) for f in files})
    subjmap={v:i for i,v in enumerate(subjects)}
    ...
        subject_idx.append(subjmap[mouse])
```

iii. The trajectory reports the inventory result — "284 NWBs, 38 mice, and 42,147 curated cells across six standardized session types" — and the mouse id is taken as the canonical animal identifier. No explicit justification beyond using the NWB's own subject field.

## 1-c. How are the data split into sessions?

i. **Every NWB file (i.e. every ophys *experiment* = one imaging plane) is treated as one "session"** in the output. There is no grouping by `ophys_session_id`. For the 168 single-plane `VisualBehavior` experiments this is equivalent to a session; for the 34 active multiscope plane-files (8 multiscope sessions, of which 6 are active, all from mouse 457841) it splits simultaneously recorded planes into 3–7 separate output "sessions" that share the identical behavioral trial set, running trace, pupil trace and stimulus stream. Sessions are emitted in sorted-filename order, and a session is written out only if it yields ≥2 trials. Result: 202 sessions.

ii.
```python
    for fi,f in enumerate(files):
      with h5py.File(f,'r') as h:
        ...
        ncell=ev.shape[1]
        ...
        if len(sn)>=2:
            neural.append(sn); inputs.append(si); outputs.append(so)
            subject_idx.append(subjmap[mouse]); region_idx.append(np.full(ncell,regionmap[loc],dtype=np.int32))
            session_info.append({'ophys_experiment_id':expid,'mouse_id':mouse,'session_type':stype,
                                 'n_cells':ncell,'n_trials':len(sn),'source_file':os.path.basename(f)})
```

iii. The AI initially planned to group planes ("group all planes of each ophys session"), then reversed: "Most files are distinct plane-level experiments, and exact start time is not a reliable ophys-session grouping key… Treat each NWB experiment as a recording session, as it has its own neural population and timestamps." Late in the run it noticed the consequence and dismissed it: "The late experiment IDs reveal several simultaneous multiplane recordings with identical trial counts, but each NWB remains a distinct ophys experiment with its own neuronal population and is valid as a decoder session." The grouping key it needed (`ophys_session_id`) is present in `/app/data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv`, which the AI never consulted.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table (the AllenSDK-derived change-detection trial table). Kept trials are `(go | catch) & ~aborted & ~auto_rewarded & isfinite(change_time)`. Instead of the trial's own `start_time`→`stop_time` span, each trial is re-cut as a **fixed peri-change window from −3.0 s to +4.2 s around `change_time`**, binned at 100 ms → exactly 72 bins for every trial in every session.

ii.
```python
DT=0.100
OFF0,OFF1=-3.0,4.2
NB=int(round((OFF1-OFF0)/DT))   # 72
...
        tr=h['intervals/trials']
        go=np.asarray(tr['go']); catch=np.asarray(tr['catch'])
        abort=np.asarray(tr['aborted']); auto=np.asarray(tr['auto_rewarded'])
        change=np.asarray(tr['change_time'],dtype=float)
        keep=np.flatnonzero((go|catch)&~abort&~auto&np.isfinite(change))
...
        for j in keep:
            edges=change[j]+OFF0+np.arange(NB+1)*DT
            centers=(edges[:-1]+edges[1:])/2
```

iii. "Trial timing is now clear: valid Go/Catch trials all extend roughly 4.23 s after change, while their starts vary from about −3.02 to −8.28 s due to the variable pre-change repeats. Thus a fixed peri-change window of −3.0 to +4.2 s preserves the common portion of every valid trial and provides consistent dimensions." The instruction's requirement that "time bins should be the same size for all trials and sessions" and the validator's rectangular-array checks motivated a fixed-length representation.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level QC is exactly the mask above: aborted trials (early lick, no change shown) and auto-rewarded trials are dropped, and trials whose `change_time` is NaN are dropped. Sessions yielding fewer than 2 trials are not written. There is no additional QC: no clipping/skipping of windows that fall off the ends of the recording (verified unnecessary — 0/5292 sampled trials have a window outside the ophys recording), no rejection of trials whose neural window is entirely empty, and no rejection of sessions missing a behavioral stream. 51,992 trials survive across 202 sessions.

ii.
```python
        keep=np.flatnonzero((go|catch)&~abort&~auto&np.isfinite(change))
...
        if len(sn)>=2:
            neural.append(sn); ...
```

iii. Directly follows the instruction ("Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials"). The trajectory adds that a very low-performing Sst session "should not be arbitrarily excluded because supplied data and paper cohort already reflect curation," i.e. the AI deliberately relied on the Allen pipeline's own curation rather than adding its own thresholds.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/event_detection/data` (detected calcium event magnitudes, stored time × cells) together with `processing/ophys/event_detection/timestamps` (the ophys clock). dF/F, corrected fluorescence, demixed and neuropil traces are all ignored. The event-detection ROI set is identical to the dF/F ROI set (verified: both 13 × for experiment 1007107386), i.e. the Allen-curated valid cells.

ii.
```python
        ev=h['processing/ophys/event_detection/data']
        ot=np.asarray(h['processing/ophys/event_detection/timestamps'],dtype=float)
        ncell=ev.shape[1]
```
and in the module docstring: "detected calcium Events (rather than dF/F) are the neural signal".

iii. "The methods establish the key neural-processing choice: use detected calcium `events` (not dF/F), matching the paper." This is a direct quote of `/app/methods.txt` l.208: "For all analysis of neural data we used the detected calcium events as described in Garrett et al.… " and l.179: "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f."

## 2-b. How is the `neural` data processed?

i. For each trial, the contiguous frame slab covering the trial window is read once from HDF5, frames are assigned to 100 ms bins by `floor((t − window_start)/0.1)`, and the **mean event magnitude across the frames in each bin** is taken per neuron. Bins containing no frame are left at 0. No normalization, smoothing, z-scoring, neuron selection, or cross-plane concatenation is performed (each file is its own session, so there is nothing to stack). Output dtype float32, shape (n_cells, 72).

ii.
```python
            a=np.searchsorted(ot,edges[0]); b=np.searchsorted(ot,edges[-1])
            slab=np.asarray(ev[a:b,:],dtype=np.float32)
            tt=ot[a:b]; mat=np.zeros((ncell,NB),dtype=np.float32)
            bi=np.floor((tt-edges[0])/DT).astype(int)
            for k in range(NB):
                z=slab[bi==k]
                if len(z): mat[:,k]=z.mean(axis=0)
```

iii. "use paper-standard detected calcium events at ophys samples"; "Average event magnitude in each temporal bin. Reading the common contiguous frame slab once per trial avoids loading full recordings" (code comment). The AI relied on the Allen pipeline's own processing (motion correction, demixing, neuropil correction, event extraction) and added nothing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level QC at all: every ROI present in `event_detection` is kept (4–666 cells per session, 29,444 total). No minimum-cell threshold, no activity threshold, no removal of trials/neurons whose event trace is identically zero. A consequence: 2,995 of the 51,992 trials (5.8%, spread over 94 sessions, concentrated in low-cell-count sessions) have an **entirely zero** neural matrix — the validator emits a warning for each. Spot-checking experiment 958741234 (4 cells, 10.7 Hz) reproduces this: 19/80 trials all-zero, with 0% empty bins, i.e. the zeros come from genuine event sparsity, not from a binning gap.

ii. No QC code; the only implicit filter is the ROI set stored in the NWB:
```python
        ncell=ev.shape[1]
```

iii. Implicit in "the supplied data and paper cohort already reflect curation" — the event-detection table only contains cells that passed the Allen segmentation/validation pipeline, so the AI added no further filtering. The all-zero-trial warnings were seen in the validator output but were characterized as "the validator's per-session statistics rather than an error" and not acted on.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to each trial's **`change_time`** (the actual image change for Go trials, the sham change for Catch trials), on the ophys clock: the window is `change_time + [−3.0 s, +4.2 s)` and frames are selected by `searchsorted` on `event_detection/timestamps`, which is the ophys timestamp vector. All other streams are resampled onto the same bin centers, so the neural, stimulus and behavioral rows of a trial share one time axis. `metadata['temporal_alignment_event'] = 'scheduled image change time (Go change or Catch sham-change)'`, `off_start=-3.0`, `off_end=4.2`.

ii.
```python
            edges=change[j]+OFF0+np.arange(NB+1)*DT
            centers=(edges[:-1]+edges[1:])/2
            a=np.searchsorted(ot,edges[0]); b=np.searchsorted(ot,edges[-1])
            slab=np.asarray(ev[a:b,:],dtype=np.float32)
```
```python
          'metadata':{...,'temporal_alignment_event':'scheduled image change time (Go change or Catch sham-change)',
                      'off_start':OFF0,'off_end':OFF1,...}
```

iii. "The likely appropriate alignment is each trial's change_time, using a fixed peri-change window that fits Go and Catch trials"; the instruction "Temporally align based on ophys timestamp" is satisfied by doing all binning and interpolation in the ophys clock of `event_detection/timestamps`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — the native ophys sampling is rebinned to a uniform **100 ms** bin (`metadata['time_bin_size'] = 100.0`), giving 72 bins per trial for all 51,992 trials. This is a ~3× downsampling for the single-plane experiments (30.95 Hz) and a ~1.07× rebin for the multiscope plane-files (10.7 Hz); the uniform grid is what makes those two populations dimensionally compatible. Rebinning is by mean over the frames falling in each bin.

ii.
```python
DT=0.100
NB=int(round((OFF1-OFF0)/DT))
...
            bi=np.floor((tt-edges[0])/DT).astype(int)
            for k in range(NB):
                z=slab[bi==k]
                if len(z): mat[:,k]=z.mean(axis=0)
...
          'metadata':{...,'time_bin_size':DT*1000,...}
```

iii. "Native-frame data would create a very large pickle; 100 ms bins preserve calcium-event and behavior dynamics while making the complete dataset tractable" and "To keep the complete active-task dataset tractable while respecting temporal alignment, 100 ms bins are appropriate for the relatively slow calcium-event signal." (The resulting pickle is still 2.3 GB.)

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the stimulus presentation TimeSeries `stimulus/presentation/<Natural_Images_...>`: its `data` (per-flash image index 0–7) and `timestamps` (per-flash onset times), combined with the session's image set read from `session_description` (`images_A` → codes 1–8, `images_B` → codes 9–16). It does **not** use the trials table's `initial_image_name`/`change_image_name`. Class 0 is an explicit "gray" (inter-flash blank) category, giving 17 categories.

ii.
```python
    image_values=['gray']+[f'{s}_image_{i}' for s in ('A','B') for i in range(8)]
...
        setoff=1 if 'images_A' in stype else 9
...
        preskeys=list(h['stimulus/presentation'].keys())
        pg=h['stimulus/presentation/'+preskeys[0]]
        stim_t=np.asarray(pg['timestamps'],dtype=float); stim_i=np.asarray(pg['data'])
```

iii. "Image identity will include gray plus all natural images"; code comment: "Global categorical image names are index-defined separately for familiar and novel sets, hence prefix by set A/B. Gray is class zero." The instruction's phrasing "Image identity (of the image presented during the non-grey screen)" was read as requiring the actual on-screen stimulus, hence flash-resolved identity with a distinct gray state. (Verified correct: the index→name mapping is constant across all 150 A-set and all 134 B-set files, so the A/B-prefixed codes are globally consistent and correspond 1:1 to the 16 image names.)

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each 100 ms bin center, the most recent flash onset is found; the bin is labelled with that flash's image (offset by the set code) only if the center falls within 250 ms of the onset — otherwise it is labelled gray (0). Indices ≥ 8 (the "omitted" code) are also forced to gray; in these files the presentation TimeSeries already excludes omitted flashes (4610 entries vs 4806 table rows), so omissions fall out as gray automatically because the gap to the previous onset is 1.5 s. The result is a 17-class time series in which each 750 ms flash cycle appears as 2–3 image bins followed by 5 gray bins (≈68% of all bins are gray).

ii.
```python
            # Screen is gray except during [onset,onset+.25). Omitted indices are gray.
            pos=np.searchsorted(stim_t,centers,side='right')-1
            img=np.zeros(NB,dtype=np.int16)
            ok=(pos>=0)
            pp=np.clip(pos,0,len(stim_t)-1)
            shown=ok&(centers<stim_t[pp]+.25)&(stim_i[pp]<8)
            img[shown]=(setoff+stim_i[pp[shown]].astype(int)).astype(np.int16)
```

iii. Derived from the whitepaper's stimulus structure (250 ms image, 500 ms gray, 750 ms cycle): "derive image timing from stimulus presentation TimeSeries" and "Image identity will include gray plus all natural images".

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated on exactly the same bin-center vector used for the neural bins of that trial (`centers`), so row 0 of `output` is bin-for-bin aligned to the neural matrix. Verified in the saved pickle: session 0 / trial 0 image row is `[4 4 0 0 0 0 0 4 4 4 0 0 0 ... 0 7 7 0 ...]`, switching from the pre-change image to the post-change image at bin 30 (the first bin after `change_time`, which sits on edge 30).

ii.
```python
            centers=(edges[:-1]+edges[1:])/2
            ...
            pos=np.searchsorted(stim_t,centers,side='right')-1
            ...
            out=np.vstack([img,ch,runbin,pupbin,np.full(NB,outcome,dtype=np.int16)])
```

iii. Implicit: the AI aligned every stream "by absolute timestamps" onto the common ophys-clock bin grid ("align all labels/behavior by absolute timestamps").

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Only `intervals/trials['change_time']`. The `go`/`catch`/`is_change` columns are **not** used for this output, so Catch trials (sham change, where the image is provably identical before and after — verified `initial_image_name == change_image_name` and `is_change == False` for 100% of catch trials) are given the same change marker as Go trials.

ii.
```python
        change=np.asarray(tr['change_time'],dtype=float)
...
            ch=np.zeros(NB,dtype=np.int16); ch[np.argmin(abs(centers-change[j]))]=1
```

iii. "image-change will be a one-frame pulse at change onset"; `metadata['temporal_alignment_event']` calls it the "scheduled image change time (Go change or Catch sham-change)", i.e. the AI knowingly treated the sham-change time as a change event. No justification is given in the trajectory for marking catch trials as changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A single bin is set to 1 — the bin whose center is closest to `change_time` — and all other 71 bins are 0. Because every trial is aligned to `change_time` on the same grid, that bin is **always index 29** (verified analytically over 2000 random change times and directly in the saved pickle: the set of change-flag indices over sessions 0–4 is exactly `{29}`). Bin 29 spans `[change_time − 0.1 s, change_time)`, i.e. the pulse lands in the bin immediately **before** the change, one bin earlier than the image-identity switch at bin 30 (the tie between bins 29 and 30, both 50 ms from the change, is broken toward the earlier one by `argmin`).

ii.
```python
            ch=np.zeros(NB,dtype=np.int16); ch[np.argmin(abs(centers-change[j]))]=1
```

iii. Intent stated in the trajectory: "pulse image change in the alignment bin" / "a one-frame pulse at change onset". The instruction is "Have value of 1 right after a change in image identity, otherwise 0."

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — it is constructed directly as a binary 0/1 series, declared as `output_values[1] = ['no change','image change']`. Realized distribution: 1/72 = 1.39% ones.

ii.
```python
          'output_names':['image identity','image change',...],
          'output_values':[image_values,['no change','image change'], ...]
```

iii. Follows the instruction that image change is a binary variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It uses the same `centers` grid as the neural matrix, so it is mechanically aligned; but because the alignment event *is* `change_time` and the pulse is a single nearest-center bin, the flag is at a constant index (29) in every one of the 51,992 trials, and that bin precedes the change rather than following it.

ii.
```python
            centers=(edges[:-1]+edges[1:])/2
            ch=np.zeros(NB,dtype=np.int16); ch[np.argmin(abs(centers-change[j]))]=1
            out=np.vstack([img,ch,runbin,pupbin,np.full(NB,outcome,dtype=np.int16)])
```

iii. Same as 4-b; the AI did not comment on the fact that change-aligned trials make this output constant across trials.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` with `processing/running/speed/timestamps` — the AllenSDK filtered running speed in cm/s (the same stream exposed as `dataset.running_speed`; the unfiltered variant `speed_unfiltered` is not used).

ii.
```python
        rt=np.asarray(h['processing/running/speed/timestamps']); rv=np.asarray(h['processing/running/speed/data'])
```

iii. "interpolate running and pupil width onto ophys timestamps" — the AI located the canonical running interface in the NWB after inspecting the HDF5 paths.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Non-finite samples are masked out, then the speed is **linearly interpolated onto the 100 ms bin centers** of each trial (`np.interp`, which clamps rather than extrapolates outside the recorded span). No smoothing, rectification or absolute value; negative speeds are retained and simply fall into the lowest quintile. The interpolated value is then quintile-binned (see 5-c).

ii.
```python
def interp_clean(t,x,q):
    t=np.asarray(t); x=np.asarray(x,dtype=float)
    good=np.isfinite(t)&np.isfinite(x)
    if good.sum()<2: return np.zeros(len(q),dtype=np.float32)
    return np.interp(q,t[good],x[good]).astype(np.float32)
...
            run=interp_clean(rt,rv,centers)
```

iii. "Running and pupil diameter will be interpolated and session-wide percentile-binned"; the running encoder samples at ~60 Hz, far faster than the 10 Hz bin grid, so interpolation at bin centers is a faithful resampling.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Into five quintile bins, with the quintile edges (20/40/60/80th percentiles) computed **per session from the entire session's running trace** (not from the trial-restricted values, and not pooled across sessions), then applied with `np.digitize`. Realized global distribution is 0.197 / 0.207 / 0.207 / 0.204 / 0.185 — close to but not exactly 20% each, because edges are derived from whole-session data and applied only to the trial windows.

ii.
```python
def quintile_reference(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if len(x)==0: return np.array([-np.inf]*4)
    return np.quantile(x,[.2,.4,.6,.8])
...
        rq=quintile_reference(rv)
...
            runbin=np.digitize(run,rq).astype(np.int16)
```
```python
          'metadata':{...,'behavior_binning':'within-session quintiles; pupil diameter is tracked pupil width',...}
```

iii. "quintile-bin continuous behavior per session"; documented in metadata as "within-session quintiles". Per-session edges normalize away between-session/between-animal differences in locomotion level so that each session contributes balanced classes.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated directly at the trial's neural bin centers on the ophys clock, so row 2 of `output` is bin-for-bin aligned with the neural matrix by construction.

ii.
```python
            run=interp_clean(rt,rv,centers)
            runbin=np.digitize(run,rq).astype(np.int16)
            out=np.vstack([img,ch,runbin,pupbin,np.full(NB,outcome,dtype=np.int16)])
```

iii. Code docstring: "Streams are aligned in the common ophys clock"; the AI treated the NWB timestamps of all streams as hardware-synchronized to a single clock.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/width` (the ellipse-fit pupil width from the DeepLabCut-based eye-tracking pipeline), timestamped with `acquisition/EyeTracking/eye_tracking/timestamps`. The `likely_blink` flag is not read explicitly, but in these NWBs blink frames are already stored as NaN in `pupil_tracking/width` (verified: NaN fraction = blink fraction = 0.0919 exactly for experiment 1007107386), and the AI's interpolation helper drops all non-finite samples — so blinks are in fact excluded before interpolation. If the eye-tracking group is absent the pupil stream is set to empty arrays.

ii.
```python
        pbase='acquisition/EyeTracking/pupil_tracking'
        if pbase in h:
            pt=np.asarray(h['acquisition/EyeTracking/eye_tracking/timestamps'])
            pv=np.asarray(h[pbase+'/width'])
        else: pt=np.array([]); pv=np.array([])
```

iii. "pupil diameter is available as `acquisition/EyeTracking/pupil_tracking/width`" (found by searching HDF5 paths for pupil/eye keywords); documented in metadata as "pupil diameter is tracked pupil width".

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Same helper as running speed: drop non-finite samples (which removes blink frames), linearly interpolate across the resulting gaps onto the trial's bin centers, then quintile-bin. **Missing-stream handling is silent and wrong**: for the 3 active experiments that have no `EyeTracking` group at all (795953296, 806456687, 833631914), `interp_clean` returns all zeros and `quintile_reference` returns `[-inf]*4`, so `np.digitize` labels every bin as quintile **4** (the *highest* pupil bin). Confirmed in the saved pickle: sessions 45, 51 and 72 have `unique(pupil) == [4]`.

ii.
```python
            pup=interp_clean(pt,pv,centers)
            pupbin=np.digitize(pup,pq).astype(np.int16)
```
```python
def quintile_reference(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if len(x)==0: return np.array([-np.inf]*4)
    return np.quantile(x,[.2,.4,.6,.8])
```

iii. "Running and pupil diameter will be interpolated and session-wide percentile-binned." No justification is offered for the empty-stream fallback; the `-inf` edges and the zero-fill are defensive defaults that were never checked against the three sessions that actually trigger them.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five within-session quintiles of pupil width, identical machinery to running speed (`np.quantile([.2,.4,.6,.8])` over the whole session's finite pupil samples, applied with `np.digitize`). Realized distribution 0.173 / 0.207 / 0.215 / 0.217 / 0.188, the skew being partly caused by the three all-"4" sessions.

ii.
```python
        pq=quintile_reference(pv)
...
            pupbin=np.digitize(pup,pq).astype(np.int16)
...
          'output_values':[..., ['0-20%','20-40%','40-60%','60-80%','80-100%'], ...]
```

iii. Per-session quintiles are the AI's stated policy for all continuous behavior; for pupil this is additionally defensible because pupil width is measured in camera pixels and its scale depends on rig/animal/eye position, so cross-session pooling would confound.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated at the same bin centers as the neural matrix on the ophys clock; row 3 of `output` is therefore bin-for-bin aligned.

ii.
```python
            pup=interp_clean(pt,pv,centers)
            pupbin=np.digitize(pup,pq).astype(np.int16)
            out=np.vstack([img,ch,runbin,pupbin,np.full(NB,outcome,dtype=np.int16)])
```

iii. Same as running speed — "Streams are aligned in the common ophys clock."

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, mapped to codes 0/1/2/3 with `output_values[4] = ['hit','miss','false alarm','correct reject']`.

ii.
```python
        hit=np.asarray(tr['hit']); miss=np.asarray(tr['miss'])
        fa=np.asarray(tr['false_alarm']); cr=np.asarray(tr['correct_reject'])
...
            outcome=0 if hit[j] else 1 if miss[j] else 2 if fa[j] else 3
```

iii. "Trial outcome will be hit/miss/false alarm/correct reject" — the canonical change-detection outcome labels for Go (hit/miss) and Catch (false alarm/correct reject) trials, which is exactly the set that remains after aborted/auto-rewarded trials are removed.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The scalar code is broadcast across all 72 bins of the trial so that `output` is a rectangular (5, 72) integer matrix; the metadata documents it as static per trial. Note the chained conditional makes `correct_reject` the fall-through default — if none of the four flags were set the trial would be silently labelled "correct reject"; this never fires in practice (verified: 0 of 5292 sampled kept trials have other than exactly one outcome flag set). Realized distribution: hit 0.307, miss 0.568, false alarm 0.018, correct reject 0.107.

ii.
```python
            outcome=0 if hit[j] else 1 if miss[j] else 2 if fa[j] else 3
            out=np.vstack([img,ch,runbin,pupbin,np.full(NB,outcome,dtype=np.int16)])
```

iii. "Trial outcome will be repeated over time if the validator requires a rectangular output matrix, while metadata will document that it is static." The AI confirmed from `/app/decoder.py` that "Outputs should likewise be a rectangular (5, n_timepoints) matrix; repeating the static outcome across time is compatible."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled: (a) NaN/inf samples in running speed and pupil width are masked out before interpolation, which also removes blink frames; (b) trials with NaN `change_time` are dropped; (c) sessions with <2 valid trials are not emitted; (d) a missing `EyeTracking` group is tolerated (no crash); (e) the brain-region lookup falls back to the string `'visual cortex'` if no imaging-plane `location` attribute is found; (f) `stim_i` values ≥8 (omitted flashes) are forced to gray, and bins before the first flash are gray. Not handled: (g) the missing-pupil fallback mislabels 3 whole sessions as pupil quintile 4 instead of flagging them or dropping them; (h) bins containing no ophys frame are silently set to 0 (indistinguishable from "no events"), and 2,995 trials end up with an all-zero neural matrix without any flag or exclusion; (i) there is no `try/except` around per-file processing, so one unreadable NWB would abort the whole 202-file run; (j) no clipping logic for trial windows that run past the recording (harmless here — verified 0 occurrences).

ii.
```python
def interp_clean(t,x,q):
    good=np.isfinite(t)&np.isfinite(x)
    if good.sum()<2: return np.zeros(len(q),dtype=np.float32)
    return np.interp(q,t[good],x[good]).astype(np.float32)

def quintile_reference(x):
    x=x[np.isfinite(x)]
    if len(x)==0: return np.array([-np.inf]*4)
```
```python
        loc='visual cortex'
        for _,g in plane.items():
            if isinstance(g,h5py.Group) and 'location' in g:
                loc=text(g['location']); break
...
        keep=np.flatnonzero((go|catch)&~abort&~auto&np.isfinite(change))
...
        if len(sn)>=2:
```

iii. The guards are written as defensive one-liners; the trajectory never discusses missing behavioral streams or the all-zero neural trials (the 2,995 validator warnings were read as "the validator's per-session statistics rather than an error").

## 9-a. What are the most time-consuming steps of the code?

i. (1) The per-trial random-access reads of the event-detection dataset — 51,992 HDF5 slab reads of shape (~75–225 frames × n_cells) out of files whose event arrays are up to 140,204 × 666; this dominates and is I/O bound. (2) Reading each NWB three separate times (once in the session-type filter pass, once in the `subjects` set comprehension, once in the main loop), i.e. 284 + 202 + 202 file opens. (3) Reading the entire running-speed (~270k samples) and pupil (~136k samples) arrays per session into memory to compute quintile edges. (4) The pure-Python inner loop over the 72 bins for every trial (3.7 M mask-and-mean operations overall). (5) Pickling and writing the 2.3 GB output. The run took many minutes of wall clock (the agent polled progress across seven turns, from 34/202 to 202/202).

ii.
```python
    for f in sorted(glob.glob(DATA_GLOB,recursive=True)):
        with h5py.File(f,'r') as h:
            if text(h['session_description']) in ACTIVE: files.append(f)
    ...
    subjects=sorted({text(h5py.File(f,'r')['general/subject/subject_id']) for f in files})
```
```python
            a=np.searchsorted(ot,edges[0]); b=np.searchsorted(ot,edges[-1])
            slab=np.asarray(ev[a:b,:],dtype=np.float32)
```

iii. The AI deliberately optimized the biggest cost it had identified — "Raw h5py access is therefore essential" instead of the SDK, and the code comment "Reading the common contiguous frame slab once per trial avoids loading full recordings" — and accepted 100 ms binning specifically to keep the output "tractable".

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. (1) The `for k in range(NB)` bin loop, which builds a boolean mask over the slab 72 times per trial; it could be replaced by a single `np.add.reduceat` / `np.bincount`-weighted reduction over the frame-to-bin index, or by `(counts, sums) = np.bincount(bi, ...)`. (2) The outer `for j in keep` trial loop: `edges`/`centers` for all trials could be computed as one (n_trials, 73) array, and `interp_clean`, `np.digitize`, the stimulus `searchsorted`, and the change-flag assignment could all be done once per session on flattened center arrays instead of per trial. (3) `np.argmin(abs(centers-change[j]))` recomputes a constant (always 29) 51,992 times. (4) The `subjects` set comprehension re-opens every file instead of reusing the metadata already read in the filter pass.

ii.
```python
            mat=np.zeros((ncell,NB),dtype=np.float32)
            bi=np.floor((tt-edges[0])/DT).astype(int)
            for k in range(NB):
                z=slab[bi==k]
                if len(z): mat[:,k]=z.mean(axis=0)
```
```python
        for j in keep:
            edges=change[j]+OFF0+np.arange(NB+1)*DT
            centers=(edges[:-1]+edges[1:])/2
```

iii. No justification is given; the AI never profiled the inner loop and its efficiency reasoning stopped at avoiding the SDK's eager loading.

## 9-c. What processing does the code repeat multiple times?

i. (1) Every NWB is opened and its scalar metadata decoded up to three times (session-type filter, subject-id comprehension, main loop) — and the comprehension's file handles are never closed. (2) `np.arange(NB+1)*DT` and the bin-center arithmetic are rebuilt for each of the 51,992 trials although only the offset changes. (3) `np.argmin(abs(centers-change[j]))` recomputes the same constant index every trial. (4) For multiscope sessions, the *same* behavioral and stimulus processing (trial table, running interpolation, pupil interpolation, quintile edges, image identity, outcomes) is redone independently for each of the 3–7 plane files of the same physical session, and the identical resulting label matrices are stored 3–7 times in the output. (5) `np.clip(pos,0,len(stim_t)-1)` and the `stim_i[pp]<8` test are evaluated per trial though the presentation arrays are session-constant.

ii.
```python
    subjects=sorted({text(h5py.File(f,'r')['general/subject/subject_id']) for f in files})
```
```python
            edges=change[j]+OFF0+np.arange(NB+1)*DT
            ...
            ch=np.zeros(NB,dtype=np.int16); ch[np.argmin(abs(centers-change[j]))]=1
```

iii. Not discussed in the trajectory; the duplicated multiscope work is a direct consequence of the "each NWB is a session" decision (1-c), which the AI reaffirmed after noticing the "identical trial counts".

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) An empty `(0, 72)` float32 array is allocated and stored for every one of the 51,992 trials as the decoder input, although the task has no inputs. (2) The static trial outcome is materialized as 72 identical values per trial. (3) The `stim_i[pp]<8` guard is dead code — the presentation TimeSeries in these files only contains indices 0–7 (omitted flashes are absent from it). (4) The full pupil stream is read, interpolated and quintiled for the three sessions where the result is a constant, meaningless label. (5) `image_values` always carries all 17 A+B labels even though a given session can only produce 9 of them. (6) The neural matrices are stored as full float32 for a signal that is >90% exact zeros, producing a 2.3 GB pickle where a sparse or float16 representation would be far smaller; likewise 5.8% of trials are stored despite carrying an identically zero neural matrix. (7) `noise_stds`, `lambdas`, ROI tables and the dF/F traces are correctly never read — this is the main thing the code avoids doing.

ii.
```python
            sn.append(mat); si.append(np.empty((0,NB),dtype=np.float32)); so.append(out)
...
            out=np.vstack([img,ch,runbin,pupbin,np.full(NB,outcome,dtype=np.int16)])
...
            shown=ok&(centers<stim_t[pp]+.25)&(stim_i[pp]<8)
```

iii. The empty-input arrays and the replicated outcome row were deliberate: "The validator requires every input trial to be a 2D array and checks its first dimension consistently, so the no-input task should be represented as shape `(0, n_timepoints)`" and "repeating the static outcome across time is compatible" with the rectangular output check. The remaining items are unjustified leftovers.
