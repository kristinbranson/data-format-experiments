# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI bypassed the AllenSDK loading API and read the released NWB (HDF5) files directly with `h5py`. It globbed every `*.nwb` file in `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments` (284 files, all experiments present in the local release, with no filtering on `project_code`, `session_type`, or the project metadata tables) and processed them one at a time in sorted filename order. It used the released, pre-processed data products only: `processing/ophys/dff/traces/{data,timestamps}` (neural), `intervals/trials` (trial table), `stimulus/presentation/<series>` + `stimulus/templates/<series>/control_description` (images), `processing/running/speed` (running), `acquisition/EyeTracking/pupil_tracking` (pupil), `general/subject/subject_id` and `general/optophysiology/*/location` (metadata). A cheap pre-pass over all 284 files first builds the global vocabularies (subjects, brain regions, image names); a second pass streams each file and writes the per-trial slices into memory before one final pickle dump.

ii.
```python
ROOT='/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments'
OUT='/app/converted_data.pkl'
FILES=sorted(glob.glob(os.path.join(ROOT,'*.nwb')))
...
subjects=[]; regions=[]; images=set()
for p in FILES:
    with h5py.File(p,'r') as f:
        sid=dec(f['general/subject/subject_id'][()])
        if sid not in subjects: subjects.append(sid)
        plane=next(iter(f['general/optophysiology'].values()))
        reg=dec(plane['location'][()])
        if reg not in regions: regions.append(reg)
        tr=f['intervals/trials']
        for col in ('initial_image_name','change_image_name'):
            images.update(dec(v) for v in tr[col][:] if dec(v) not in ('','nan','None'))
...
for fi,p in enumerate(FILES):
    with h5py.File(p,'r') as f:
        tr=f['intervals/trials']
        ot=find_dset(f,['processing/ophys/dff/traces/timestamps'])[:]
        dff=find_dset(f,['processing/ophys/dff/traces/data'])
```

iii. From the trajectory, the AI first tried the SDK (`BehaviorOphysExperiment.from_nwb`) and inspected the SDK's tables to learn the canonical semantics, then deliberately switched to direct HDF5 reads: "Loading all 284 files through pynwb/AllenSDK would be unnecessarily slow and memory-heavy, so the converter should read the equivalent released datasets directly with h5py while preserving the SDK-defined trial flags and processed streams" (step 13), and "There are 284 experiment NWBs totaling 247 GB, so conversion must stream one file at a time" (step 4). It verified in advance that the raw HDF5 paths correspond to the SDK-reconstructed tables (steps 12, 14, 15).

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the unique values of the NWB field `general/subject/subject_id`, collected over all files in the pre-pass and sorted. This gives 38 subjects. Each converted session's `subject_idx` is the index of its file's `subject_id` in that sorted list.

ii.
```python
sid=dec(f['general/subject/subject_id'][()])
if sid not in subjects: subjects.append(sid)
...
subjects=sorted(subjects)
...
subject_idx.append(subjects.index(sid))
```

iii. The trajectory (step 8) reports the pre-scan result — "284 plane-level experiments from 38 mice and 42,147 curated cells" — and the subject id stored in the NWB is the same `mouse_id` used by the SDK metadata tables, so no separate metadata table lookup was considered necessary.

## 1-c. How are the data split into sessions?

i. **One NWB experiment file = one session.** The AI did not group experiments by `ophys_session_id`, and did not use the `ophys_experiment_table.csv` at all. For the 239 single-plane `VisualBehavior` experiments in the local release this is equivalent to one session each. For the 45 `VisualBehaviorMultiscope` experiments (1 mouse, 8 real imaging sessions, ~5–8 simultaneously recorded planes each) this splits each real session into up to 8 "sessions" that share the same behavior, the same trial table and the same trials, but carry only that plane's 4–18 neurons. The resulting dataset has 281 sessions, one mouse contributing 45 of them, and 84,313 trials versus ~72,159 for a project-filtered, session-grouped conversion.

ii.
```python
FILES=sorted(glob.glob(os.path.join(ROOT,'*.nwb')))
for fi,p in enumerate(FILES):
    with h5py.File(p,'r') as f:
        ...
        neural.append(ns); inputs.append(ins); outputs.append(outs)
        subject_idx.append(subjects.index(sid))
        brain_region_idx.append(np.full(ncell,regions.index(reg),dtype=np.int16))
        session_info.append({'ophys_experiment_id':eid,'mouse_id':sid,
          'targeted_structure':reg,'n_cells':ncell,'n_trials':len(ns),
          'ophys_frame_rate_hz':float(plane['imaging_rate'][()])})
```

iii. The AI raised the question early — "284 plane-level experiments may represent fewer multi-plane sessions, but the target permits each recording experiment as a session and each NWB has plane-specific neural traces" (step 6) — and resolved it at step 8: "Session IDs and structures are not at the guessed raw paths... Treating each NWB experiment as one decoder session is likely appropriate because each has its own neuron population and timestamps." It never went back to the project metadata tables to check whether files share an `ophys_session_id`, so the premise ("its own ... timestamps") is false for the 45 Multiscope files.

## 1-d. How are the data split into trials?

i. Trials come from the released `intervals/trials` table. Retained trials are `(go OR catch) AND NOT aborted AND NOT auto_rewarded`. Each retained trial spans the ophys frames from the trial's `start_time` to its `stop_time` (`searchsorted` left/right), giving variable-length trials (~77–391 frames; ~8.7 s at 31 Hz). Trials whose window is empty are dropped, and the window is clipped to the end of the recording.

ii.
```python
keep=(tr['go'][:].astype(bool)|tr['catch'][:].astype(bool))
keep &= ~tr['aborted'][:].astype(bool)
keep &= ~tr['auto_rewarded'][:].astype(bool)
tids=np.flatnonzero(keep)
...
starts=tr['start_time'][:]; stops=tr['stop_time'][:]
for j in tids:
    a=np.searchsorted(ot,starts[j],side='left')
    b=np.searchsorted(ot,stops[j],side='right')
    a=max(0,a); b=min(ot.size,b)
    if b>a: retained[a:b]=True
    bounds.append((j,a,b))
```

iii. Step 9: "The representative experiment has 498 total trials: 323 go, 42 catch, 133 aborted, and 5 auto-rewarded. Thus the requested filter should retain go/catch trials while explicitly removing auto-rewarded ones." The full `start_time`→`stop_time` window was chosen (step 12) so that each trial contains the pre-change flashes and the post-change response window, which is what makes the time-varying outputs meaningful. The methods text defines aborted trials as trials in which the animal responded before the change, i.e. no change was shown.

## 1-e. How are trials filtered based on quality controls?

i. Besides the go/catch, non-aborted, non-auto-rewarded filter: empty windows (`b<=a`) are dropped; windows are clipped to the recording length; sessions with fewer than 2 retained trials are dropped both before and after slicing; whole sessions are dropped if the release has no eye-tracking stream (3 files), fewer than 2 finite pupil samples, or no stimulus presentations; a trial that matches none of hit/miss/false_alarm/correct_reject raises an error rather than being silently coded.

ii.
```python
if 'acquisition/EyeTracking/pupil_tracking' not in f:
    print(f'{fi+1}/{len(FILES)} skip: no pupil tracking {os.path.basename(p)}', flush=True); continue
if 'stimulus/presentation' not in f or len(f['stimulus/presentation']) == 0:
    print(f'{fi+1}/{len(FILES)} skip: no stimulus presentations {os.path.basename(p)}', flush=True); continue
...
if len(tids)<2: print('skip <2 trials',p,flush=True); continue
if np.isfinite(pa).sum() < 2:
    print(f'{fi+1}/{len(FILES)} skip: insufficient pupil samples {os.path.basename(p)}', flush=True); continue
...
for j,a,b in bounds:
    if b<=a: continue
...
if len(ns)<2: continue
```

iii. Step 19: "Conversion stopped at file 63 because at least one experiment lacks EyeTracking data... Since pupil diameter is a required output, fabricating values would be inappropriate; sessions without pupil tracking should be excluded as a documented session-level curation decision." The ≥2-trial rule follows the format requirement that each session must have at least two trials for the decoder to be evaluated.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/dff/traces/data`, i.e. the Allen-released, detrended dF/F traces of the curated ROIs for the one imaging plane in that NWB, with `processing/ophys/dff/traces/timestamps` as the common clock. No other neural product (raw fluorescence, neuropil-corrected traces, or the `events` / inferred-spike product) is used.

ii.
```python
ot=find_dset(f,['processing/ophys/dff/traces/timestamps'])[:]
dff=find_dset(f,['processing/ophys/dff/traces/data'])
ncell=dff.shape[1]
...
x=np.asarray(dff[a:b,:],dtype=np.float32).T
```

iii. Module docstring: "use released, ROI-curated dF/F (not recomputed fluorescence or inferred events)". The methods text describes the full Allen dF/F pipeline (median-filter baseline, noise normalisation, detrending), so the released trace is the paper's neural signal.

## 2-b. How is the `neural` data processed?

i. Essentially none beyond formatting: per-trial frame slices are read from the HDF5 dataset, transposed from (time, cells) to (cells, time), cast to float32, and any non-finite dF/F values are replaced with 0. No smoothing, z-scoring, normalisation, neuropil correction or rebinning is applied, and — following the one-file-per-session decision — no merging of neurons across simultaneously recorded planes.

ii.
```python
# h5py reads time x cells; target is cells x time.
x=np.asarray(dff[a:b,:],dtype=np.float32).T
# Very rare nonfinite dF/F values are invalid for the decoder.
if not np.isfinite(x).all():
    x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
```

iii. Step 12: DFF "data are stored time-by-cell and must be transposed"; the AI treats the released dF/F as already fully processed by the standardized Allen pipeline and only guards against non-finite values that would break the decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering is applied. The AI first checked whether ROI curation was needed and found that the released NWBs contain only ROIs flagged `valid_roi == True` (confirmed here: 405/405 valid in a representative file), so all traces present in `dff` are kept. Filtering happens only at the session level (see 1-e).

ii. No code — absence of filtering is the decision; the relevant supporting field is `processing/ophys/image_segmentation/cell_specimen_table/valid_roi`, which the AI inspected but did not need to apply.

iii. Step 4: "The 13 ROIs in the first file also indicate valid_roi filtering may materially affect neuron counts", then step 8: "Every stored ROI is already valid, indicating the released NWBs contain only curated ROIs." The docstring records the result: "Allen released detrended dF/F from valid curated ROIs".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the ophys timestamps, exactly as instructed: the ophys frame grid is the master clock and every other stream is interpolated/indexed onto it. Each trial is the contiguous block of ophys frames `[a, b)` with `a` = first frame at/after `start_time` and `b` = last frame at/before `stop_time`; trials are therefore aligned to trial start, not to `change_time`, and are variable length. `metadata['temporal_alignment_event']` records this, and `off_start`/`off_end` are `None`.

ii.
```python
a=np.searchsorted(ot,starts[j],side='left')
b=np.searchsorted(ot,stops[j],side='right')
a=max(0,a); b=min(ot.size,b)
...
x=np.asarray(dff[a:b,:],dtype=np.float32).T
...
'temporal_alignment_event':'Native ophys timestamps; each trial spans SDK trial start_time through stop_time.',
'off_start':None,'off_end':None,
```

iii. Step 12/15: the plan was to "slice each trial from SDK start_time through stop_time on ophys timestamps" and to align every other stream onto the same frame grid, so no cross-stream resampling error is introduced into the neural data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning: the native ophys frame grid is used as-is. For the 236 retained single-plane (`VisualBehavior`) sessions the frame period is 32.32 ms (~30.9 Hz); for the 45 Multiscope sessions it is 93.2 ms (~10.7 Hz). The dataset therefore contains **two different bin sizes**, while `metadata['time_bin_size']` is hard-coded to `1000/31 = 32.26 ms` for all of them.

ii.
```python
'time_bin_size':1000.0/31.0,
```
(There is no resampling code; the only sampling-rate value read from the file is recorded per session as documentation:)
```python
session_info.append({... 'ophys_frame_rate_hz':float(plane['imaging_rate'][()])})
```

iii. Docstring: "use native ophys frames (~31 Hz) as the common clock and trial bins". Step 9: "Ophys sampling is about 31 Hz." Step 25 shows the AI was aware of the issue and planned to "potentially correct any metadata time-bin issue caused by small frame-rate differences across sessions", but no correction was made — and the Multiscope difference is a factor of ~2.9, not a small difference.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the stimulus presentation series itself, not from the trials table: `stimulus/presentation/<series>/timestamps` (flash onset times) and `.../data` (template index per flash), decoded to image names through `stimulus/templates/<series>/control_description`. The set of allowed image names (the global vocabulary) is taken from the trials table columns `initial_image_name` / `change_image_name` across all files, so passive/movie template names could not leak in.

ii.
```python
pres=f['stimulus/presentation']
series=next(iter(pres.values()))
pon=series['timestamps'][:]; pind=series['data'][:].astype(int)
templ=next(iter(f['stimulus/templates'].values()))
if 'control_description' in templ:
    labels=[dec(x) for x in templ['control_description'][:]]
else:
    labels=images
```

iii. Step 14: "The raw stimulus series contains image indices and onset timestamps at the 750 ms flash cadence. Image names can be decoded from the template's control descriptions." Step 15: "Image identity must account for the experiment's 250 ms image flashes separated by 500 ms gray periods rather than holding each image for the full 750 ms cadence." Vocabulary comment in the code: "Global vocabularies are based on actual task trial labels, avoiding passive movie/template names unrelated to the Visual Behavior task."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A session-long integer label vector on the ophys grid is built, initialised to `0 = 'gray'`. For each flash onset, the frames in `[onset, onset+250 ms)` are set to that image's global code. Frames outside a flash — the 500 ms inter-stimulus gray and the 5% omitted flashes (which are simply absent from the presentation series) — remain `gray`. The global code map is `['gray'] + sorted(image names)` = 17 categories (16 images from image sets A and B, plus gray). In the final dataset gray occupies 67% of timepoints and each image ~2%.

ii.
```python
img=np.zeros(ot.size,dtype=np.int16)
for onset,ii in zip(pon,pind):
    if ii>=len(labels): continue
    label=labels[ii]
    if label not in image_code: continue
    a=np.searchsorted(ot,onset,'left'); b=np.searchsorted(ot,onset+0.250,'left')
    img[a:b]=image_code[label]
...
image_values=['gray']+images
image_code={v:i for i,v in enumerate(image_values)}
```

iii. Docstring: "image flashes last 250 ms in this task and are followed by 500 ms gray. Omitted flashes therefore remain gray." The methods text confirms 250 ms stimulus / 500 ms inter-stimulus gray and 5% omissions. The AI treated "image identity of the image presented during the non-grey screen" as requiring an explicit gray state whenever no image is on the monitor (step 10: "A key remaining decision is categorical coding during gray intervals"; step 12: "map each ophys frame to the active image or gray (with omitted presentations treated as gray)").

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed on the full-session ophys timestamp vector `ot` and then sliced with exactly the same `[a, b)` indices used for the dF/F, so it is aligned frame-for-frame by construction. Flash onsets are mapped to frames with `searchsorted(..., 'left')`, i.e. the first ophys frame at or after the monitor onset time (the stored times are already display-lag-corrected by the SDK).

ii.
```python
a=np.searchsorted(ot,onset,'left'); b=np.searchsorted(ot,onset+0.250,'left')
img[a:b]=image_code[label]
...
y=np.vstack((img[a:b],change,runbin[a:b],pupbin[a:b],
             np.full(T,outcome,dtype=np.int8))).astype(np.int16)
```

iii. Consistent with the stated design of using the ophys frame grid as the single common clock for all streams (docstring, step 12), which makes alignment exact and removes any need for per-stream offsets.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The trials table columns `change_time` and `go` (a boolean flagging trials with a real image change, as opposed to `catch` trials, which carry a sham `change_time`).

ii.
```python
ct=float(tr['change_time'][j])
# Catch trials have a sham change_time but no identity change.
if np.isfinite(ct) and bool(tr['go'][j]):
    k=np.searchsorted(ot[a:b],ct,'left')
    if k<T: change[k]=1
```

iii. Step 28: "the image-change impulse count currently equals the total trial count because catch-trial sham change times were marked. The specification requires an impulse only after an actual image identity change, so catch trials must remain zero for this output." The AI re-ran the whole conversion to fix this and verified 73,733 impulses == 73,733 go trials.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A single-frame impulse: a zero vector of length T with exactly one `1` at the first ophys frame at/after `change_time`, and only on go trials. No temporal widening to the flash duration or the response window is applied. The resulting positive rate is 0.4% of all timepoints (1 frame out of ~250 per trial, and zero for all catch trials).

ii.
```python
change=np.zeros(T,dtype=np.int8)
ct=float(tr['change_time'][j])
if np.isfinite(ct) and bool(tr['go'][j]):
    k=np.searchsorted(ot[a:b],ct,'left')
    if k<T: change[k]=1
```

iii. Docstring: "An image-change impulse is placed at the first ophys frame at/after the SDK display-lag-corrected trial change_time." Step 12: "mark image change at the first ophys frame at/after each true change." The AI read the instruction "have value of 1 right after a change in image identity" literally as a point event, and applied the same reasoning it used elsewhere for representing event times as binary time series.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Binary, with `output_values` `['no change','change']`. There is no continuous quantity to threshold; the only categorisation rule is go-trial change → 1 at one frame, everything else (all catch-trial frames, all pre/post-change frames, all gray/omission frames) → 0.

ii.
```python
'output_names':['image identity','image change','running speed quintile','pupil diameter quintile','trial outcome'],
'output_values':[image_values,['no change','change'], ...]
```

iii. As in 4-a/4-b: catch trials are explicitly excluded from the positive class because their `change_time` is a sham; go trials are the only trials with an actual identity change.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The impulse index is computed by `searchsorted` **within the trial's own slice of the ophys timestamps** (`ot[a:b]`), so it is in trial-local frame coordinates and lines up exactly with column `k` of the trial's neural matrix. A guard drops impulses that would fall past the end of the trial window.

ii.
```python
k=np.searchsorted(ot[a:b],ct,'left')
if k<T: change[k]=1
```

iii. Same rationale as 2-d/3-c: everything is expressed on the ophys frame grid of the trial, so no separate alignment step is needed.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/{data,timestamps}` — the Allen-processed (wrap-corrected, transient-removed, 10 Hz low-pass filtered) linear running speed in cm/s, which is what the SDK exposes as `running_speed`. The unfiltered `running_speed_raw` variant is not used.

ii.
```python
# Processed running speed supplied by AllenSDK/NWB.
rt=f['processing/running/speed/timestamps'][:]
rv=f['processing/running/speed/data'][:]
run=interp_finite(ot,rt,rv)
```

iii. Docstring: "use released filtered running speed". Step 5: "The methods confirm official SDK-filtered running speed"; the methods text describes the encoder processing pipeline and states that the filtered version is the `running_speed` attribute.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation from the running-encoder timestamps onto the ophys timestamp grid (over the finite samples only), then discretisation into quintiles (see 5-c). If a session had no finite samples the stream would be filled with zeros, and with a single finite sample it would be held constant. No additional smoothing, absolute value, or clipping is applied.

ii.
```python
def interp_finite(tnew,t,x):
    t=np.asarray(t,float); x=np.asarray(x,float)
    ok=np.isfinite(t)&np.isfinite(x)
    if ok.sum()==0: return np.zeros(len(tnew),dtype=np.float32)
    if ok.sum()==1: return np.full(len(tnew),x[ok][0],dtype=np.float32)
    return np.interp(tnew,t[ok],x[ok]).astype(np.float32)
```

iii. Step 12/15: "interpolate official filtered running speed and pupil diameter to ophys time". The running stream is sampled at ~60 Hz (stimulus frame rate), well above the ophys rate, so linear interpolation onto the slower ophys grid is a faithful resample; `np.interp` clamps at the edges rather than producing NaNs, avoiding missing labels at session boundaries.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-count bins (quintiles), computed **per session** from the finite running values at the frames belonging to retained trials only, and applied to the whole session's vector with `searchsorted(..., 'right')` → codes 0–4 (`Q1`…`Q5`). Edges are stored per session in `metadata['percentile_edges']`. Because bins are per session, the marginal distribution is exactly 20% per bin in the final dataset.

ii.
```python
def quintile(x, mask):
    """Codes 0..4; ties are handled deterministically by percentile edges."""
    vals=x[mask & np.isfinite(x)]
    if vals.size==0: return np.zeros(x.size,dtype=np.int8), [np.nan]*4
    edges=np.quantile(vals,[.2,.4,.6,.8])
    return np.searchsorted(edges,x,side='right').astype(np.int8), edges.tolist()
...
retained=np.zeros(ot.size,bool); bounds=[]
...
runbin,redges=quintile(run,retained)
```

iii. Docstring: "discretize running and pupil separately within each recording into quintiles using all retained trial frames (the requested equal percentile bins)". Step 12: "compute five percentile bins per experiment from finite retained-trial values". Restricting the percentile base to retained trial frames keeps the bins matched to the data actually exported; the decoder is trained and evaluated within each session, so per-session bins guarantee balanced classes for every session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated onto the full-session ophys timestamp vector before trial segmentation, then sliced with the same `[a, b)` indices as the neural data, so alignment is exact frame-for-frame.

ii.
```python
run=interp_finite(ot,rt,rv)
...
runbin,redges=quintile(run,retained)
...
y=np.vstack((img[a:b],change,runbin[a:b],pupbin[a:b], np.full(T,outcome,dtype=np.int8)))
```

iii. Same single-clock rationale as 2-d: all streams in the NWB share the session's synchronised clock, so interpolation onto `ot` is valid and removes any residual offset.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (the filtered pupil-ellipse area, in which likely blinks and outliers are already NaN — verified here: NaNs coincide exactly with `EyeTracking/likely_blink`) with the matching `pupil_tracking/timestamps`. Diameter is derived as the equivalent-circle diameter `2*sqrt(area/pi)`. Sessions without this group are dropped entirely.

ii.
```python
pg=f['acquisition/EyeTracking/pupil_tracking']
pt=pg['timestamps'][:] if 'timestamps' in pg else f['acquisition/EyeTracking/eye_tracking/timestamps'][:]
pa=pg['area'][:]
if np.isfinite(pa).sum() < 2:
    print(f'{fi+1}/{len(FILES)} skip: insufficient pupil samples {os.path.basename(p)}', flush=True)
    continue
diam=interp_finite(ot,pt,2*np.sqrt(np.maximum(pa,0)/np.pi))
```

iii. Step 12: "Canonical eye tracking provides filtered `pupil_area` with blink/outlier periods represented as NaN, which should be converted to diameter as `2*sqrt(area/pi)`." Step 15: "Processed pupil area is in `acquisition/EyeTracking/pupil_tracking/area`; likely blinks and outliers have already been set to NaN, matching the SDK's `pupil_area`." Using area rather than a single ellipse axis folds in both axes of the fitted ellipse.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area → equivalent diameter (with a `max(area, 0)` guard), then linear interpolation over the finite (non-blink, non-outlier) samples onto the ophys grid — which implicitly bridges blink gaps rather than leaving NaNs — then per-session quintile discretisation. No temporal smoothing or additional outlier rejection is added on top of the released filtering.

ii.
```python
diam=interp_finite(ot,pt,2*np.sqrt(np.maximum(pa,0)/np.pi))
...
pupbin,pedges=quintile(diam,retained)
```

iii. Docstring: "Pupil diameter is 2*sqrt(area/pi). Missing blink/outlier samples are linearly interpolated." Interpolating across NaN-marked blinks (instead of emitting a missing-data category) keeps the output categorical and complete at every timepoint, which the target format requires.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identically to running speed: five equal-count bins from `np.quantile(..., [.2,.4,.6,.8])` computed per session over finite pupil values at retained-trial frames, applied with `searchsorted(..., 'right')` → codes 0–4 (`Q1`…`Q5`), edges stored in `metadata['percentile_edges']`.

ii.
```python
pupbin,pedges=quintile(diam,retained)
...
percentile_edges.append({'running_speed_cm_per_s':redges,'pupil_diameter_pixels':pedges})
```

iii. Same justification as 5-c — per-recording quintiles, because absolute pupil size in camera pixels is not comparable across sessions (camera position, lighting, eye size), so a global threshold would confound session identity with pupil state.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the full-session ophys timestamps before segmentation and sliced with the same `[a, b)` indices as the neural data — exact frame-for-frame alignment.

ii.
```python
diam=interp_finite(ot,pt,2*np.sqrt(np.maximum(pa,0)/np.pi))
...
y=np.vstack((img[a:b],change,runbin[a:b],pupbin[a:b], np.full(T,outcome,dtype=np.int8)))
```

iii. Same single-clock rationale as the other streams; the eye-tracking timestamps are in the same synchronised session clock as the ophys timestamps.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that order.

ii.
```python
if bool(tr['hit'][j]): outcome=0
elif bool(tr['miss'][j]): outcome=1
elif bool(tr['false_alarm'][j]): outcome=2
elif bool(tr['correct_reject'][j]): outcome=3
else: raise ValueError(f'unclassified retained trial {j} in {p}')
```

iii. Step 12: the AI checked "whether trial outcomes can map cleanly to hit/miss/false alarm/correct reject" and confirmed that "Trials have all required classification flags" for the retained go/catch set. The hard failure on an unclassified trial was chosen so a silent mislabel cannot slip through (no such error occurred over all 284 files).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Mapped to a fixed integer code 0–3 with `output_values` `['hit','miss','false alarm','correct reject']`, then broadcast constant across all T frames of the trial so that the static per-trial label is a time-varying row like the other outputs.

ii.
```python
y=np.vstack((img[a:b],change,runbin[a:b],pupbin[a:b],
             np.full(T,outcome,dtype=np.int8))).astype(np.int16)
```

iii. Step 10: "The decoder requires each output trial to have shape (d_output, T), so even the static trial outcome should be repeated across time." No further processing (e.g. splitting passive-session trials, which are all miss/correct_reject) was applied; the resulting marginal is hit 18.5%, miss 68.9%, false alarm 1.1%, correct reject 11.5%.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Layered handling:
- **Missing eye tracking / stimulus presentations / <2 finite pupil samples** → the whole session is skipped with a printed reason (3 sessions skipped for missing pupil tracking); nothing is fabricated.
- **Blink/outlier NaNs in pupil, NaNs in running** → excluded from the interpolation basis and bridged by linear interpolation; edge extrapolation is clamped by `np.interp` rather than producing NaNs; all-NaN → zeros, single sample → constant.
- **Non-finite dF/F** → replaced with 0 per trial.
- **Trial windows running past the recording / empty windows** → clipped to `ot.size`, and empty windows dropped.
- **Missing template `control_description`** → falls back to the global trial-derived image list.
- **Unknown image labels** → left as gray rather than guessed.
- **NaN quantile inputs** → excluded from the edge computation via the `isfinite` mask.
- **Sessions with <2 retained trials** → dropped.
- **Unclassifiable trial outcome** → raises, stopping the run rather than emitting a bogus label.

ii.
```python
def interp_finite(tnew,t,x):
    ok=np.isfinite(t)&np.isfinite(x)
    if ok.sum()==0: return np.zeros(len(tnew),dtype=np.float32)
    if ok.sum()==1: return np.full(len(tnew),x[ok][0],dtype=np.float32)
    return np.interp(tnew,t[ok],x[ok]).astype(np.float32)
...
if not np.isfinite(x).all():
    x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
...
a=max(0,a); b=min(ot.size,b)
...
vals=x[mask & np.isfinite(x)]
if vals.size==0: return np.zeros(x.size,dtype=np.int8), [np.nan]*4
...
with open(OUT+'.tmp','wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
os.replace(OUT+'.tmp',OUT)
```

iii. Step 19 states the governing principle for missing behavioural streams: "Since pupil diameter is a required output, fabricating values would be inappropriate; sessions without pupil tracking should be excluded as a documented session-level curation decision", recorded in `metadata['trial_filter']`. The atomic `.tmp` + `os.replace` write avoids leaving a half-written pickle if the run dies. The one place the AI chose to fail loudly rather than degrade gracefully is the trial-outcome classification.

## 9-a. What are the most time-consuming steps of the code?

i. (1) Reading dF/F out of 284 NWBs totalling ~247 GB — this dominates and is I/O/decompression bound; it is done as ~84k separate `dff[a:b,:]` slice reads rather than one contiguous read per file. (2) The whole-file pre-pass, which opens all 284 files a second time just to build the subject/region/image vocabularies. (3) Serialising the ~14 GB result with `pickle.dump`, plus holding all of it in RAM first. (4) Within a session, the Python loop over ~4,600 stimulus onsets and the loop over ~400 trials.

ii.
```python
for p in FILES:            # pre-pass over all 284 files
    with h5py.File(p,'r') as f: ...
...
x=np.asarray(dff[a:b,:],dtype=np.float32).T   # per-trial HDF5 read, ~84k times
...
with open(OUT+'.tmp','wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 4: "There are 284 experiment NWBs totaling 247 GB, so conversion must stream one file at a time and avoid retaining unnecessary precision." Step 13 gives the explicit reason for the h5py route over the SDK: "Loading all 284 files through pynwb/AllenSDK would be unnecessarily slow and memory-heavy." Steps 17–24 and 29–34 show the AI polling through multi-minute runs, i.e. it observed the I/O cost directly (and paid it three times, because the pipeline was re-run end-to-end after each fix).

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:
- The stimulus-onset loop (`for onset,ii in zip(pon,pind)`), ~4,600 iterations per session, each doing two `searchsorted` calls — this is fully vectorizable with one `np.searchsorted(ot, pon)` / `np.searchsorted(ot, pon+0.25)` pair plus a label-code lookup array, filling `img` via `np.repeat`/index ranges.
- The trial-bounds loop (`for j in tids`) — `starts`/`stops` could be converted to `a`/`b` arrays in two vectorized `searchsorted` calls.
- The per-trial extraction loop — the `change`, `outcome` and slicing bookkeeping could be computed for all trials at once; only the ragged dF/F copy genuinely needs a loop.
- The vocabulary pre-pass uses `if sid not in subjects` / `subjects.index(sid)` linear scans instead of dicts (negligible at this size).

ii.
```python
for onset,ii in zip(pon,pind):
    if ii>=len(labels): continue
    label=labels[ii]
    if label not in image_code: continue
    a=np.searchsorted(ot,onset,'left'); b=np.searchsorted(ot,onset+0.250,'left')
    img[a:b]=image_code[label]
...
for j in tids:
    a=np.searchsorted(ot,starts[j],side='left')
    b=np.searchsorted(ot,stops[j],side='right')
```

iii. The AI never comments on this; its stated efficiency concern was file I/O and memory (steps 4, 13, 15), and relative to a 247 GB read these Python loops are indeed second-order.

## 9-c. What processing does the code repeat multiple times?

i. - Every NWB is opened and parsed **twice**: once in the vocabulary pre-pass (subject id, plane location, both image-name columns of the trials table) and once in the main loop, which re-reads the same trials table.
- dF/F is read per trial, so HDF5 chunks spanning adjacent trials are decompressed more than once; one contiguous per-session read (or per-chunk streaming) would touch each byte once.
- `subjects.index(sid)` / `regions.index(reg)` re-scan the vocabulary lists for every session.
- `tr['go'][:]`, `tr['hit'][j]`, etc. are re-indexed from the HDF5 datasets inside the per-trial loop rather than being materialised once as arrays.
- At the pipeline level, the entire 284-file conversion was executed three times (initial run, missing-pupil fix, catch-trial fix) rather than caching intermediate per-session results.

ii.
```python
for p in FILES:                      # pass 1
    with h5py.File(p,'r') as f:
        tr=f['intervals/trials']
        for col in ('initial_image_name','change_image_name'): ...
...
for fi,p in enumerate(FILES):        # pass 2, same files, same trials table
    with h5py.File(p,'r') as f:
        tr=f['intervals/trials']
...
    if bool(tr['hit'][j]): outcome=0   # per-trial HDF5 scalar reads
```

iii. The pre-pass exists for a stated reason — the code comment "Global vocabularies are based on actual task trial labels" and step 13's plan for "a small pre-scan for global image labels/regions/subjects" — so that image/region/subject codes are consistent across sessions. That requires seeing all files before emitting any, so some repetition is intrinsic; the AI deliberately kept the pre-pass cheap (metadata columns only, no dF/F).

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. - **The area→diameter conversion is a no-op for the output.** `2*sqrt(area/pi)` is strictly monotonic in `area`, and the only use of `diam` is a quantile binning, so the quintile labels are bit-for-bit identical to those from the raw area. The conversion buys interpretability of `percentile_edges`, nothing more.
- **Full-session computation of streams that are only used inside trials.** `run`, `diam`, `img` and the quintile codes are computed for every frame of the session (~140k frames), but only the frames inside retained trials (roughly half to two-thirds) ever reach the output.
- **The `retained` boolean mask** is built solely to select the quantile base; the same values could be gathered from the `bounds` list already being built.
- **Empty input arrays** `np.empty((0,T))` are allocated and pickled once per trial (84,313 objects) although `input_names` is empty and the decoder has no inputs.
- **Redundant precision/volume in the output**: keeping every retained trial's full dF/F at native resolution produces a 14 GB pickle; a large part of it is the 45 Multiscope plane-"sessions", which duplicate the same 8 sessions' behavioural outputs and trial windows across planes.
- `np.isfinite(x).all()` is evaluated for every trial's dF/F block even though non-finite values are described as "very rare".

ii.
```python
diam=interp_finite(ot,pt,2*np.sqrt(np.maximum(pa,0)/np.pi))   # monotonic -> same quintiles as area
...
retained=np.zeros(ot.size,bool); bounds=[]
...
runbin,redges=quintile(run,retained)     # computed for all session frames
pupbin,pedges=quintile(diam,retained)
...
ns.append(x); ins.append(np.empty((0,T),dtype=np.float32)); outs.append(y)
```

iii. The AI's stated aims were fidelity to the released products and controlling memory/size — "process one experiment at a time and retain only requested go/catch trial slices" (step 14) and "avoid retaining unnecessary precision" (step 4, implemented as float32 neural and int8/int16 outputs). The diameter conversion is justified in the docstring on semantic grounds ("Pupil diameter is 2*sqrt(area/pi)") rather than on any effect on the output, and the empty-input arrays are there because the verifier was checked to accept "2D empty inputs ... as long as their time dimension matches neural data" (step 11).
