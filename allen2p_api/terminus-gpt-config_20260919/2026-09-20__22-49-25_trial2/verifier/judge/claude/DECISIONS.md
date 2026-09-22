# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All data are read through the AllenSDK `VisualBehaviorOphysProjectCache`, instantiated with `from_local_cache('/app/data')` (no S3 access; the cache is a local copy of Visual Behavior Ophys release 1.1.0). The AI enumerates the experiments that are actually present on disk by globbing the NWB filenames in `behavior_ophys_experiments/` (filenames only — the files are never opened directly), intersects those ids with `get_ophys_experiment_table()`, and then loads each experiment object with `cache.get_behavior_ophys_experiment(id)`. It does **not** filter on `project_code`, so both `VisualBehavior` (168 active experiments) and `VisualBehaviorMultiscope` (34 active experiments) are included; it *does* drop every `session_type` containing "passive" and three hard-coded experiment ids that have no eye-tracking rows. All per-experiment content (`ophys_timestamps`, `events.filtered_events`, `trials`, `stimulus_presentations`, `running_speed`, `eye_tracking`) is pulled from the single loaded experiment object in one pass.

ii.
```python
CACHE_DIR = Path('/app/data')
MISSING_EYE_IDS = {795953296, 833631914, 806456687}

def local_ids():
    root=CACHE_DIR/'visual-behavior-ophys-1.1.0'/'behavior_ophys_experiments'
    return sorted(int(re.search(r'_(\d+)\.nwb$',p.name).group(1)) for p in root.glob('behavior_ophys_experiment_*.nwb'))

cache=VisualBehaviorOphysProjectCache.from_local_cache(CACHE_DIR)
table=cache.get_ophys_experiment_table(); ids=table.index.intersection(local_ids())
tab=table.loc[ids].sort_index()
tab=tab[~tab.session_type.str.contains('passive',case=False,na=False)]
tab=tab[~tab.index.isin(MISSING_EYE_IDS)]
...
for j,(xid,row) in enumerate(tab.iterrows(),1):
    exp=cache.get_behavior_ophys_experiment(int(xid))
    n,i,o,info,raw=convert_experiment(exp,row,IMAGE_TO_ID)
```

iii. From CONVERSION_NOTES Step 1/2: the SDK documentation "explicitly recommends loading and interacting with NWB-backed data through AllenSDK; conversion will never open NWB files directly." Only 284 of the release's 1,936 experiments are present locally, so "conversion can only process local files"; the AI therefore distinguishes release metadata (1,936 experiments / 107 mice) from the provided subset (284 experiments / 247 sessions / 38 mice) and converts the provided subset. Passive sessions are excluded because "the paper explicitly excludes passive viewing from its analyses" and SDK-generated passive trial rows "receive artificial no-response outcomes." Each experiment is loaded exactly once "and all source tables/streams are reused" for efficiency.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the retained experiment table, cast to `str` and sorted. `subject_idx[session]` is the position of that experiment's mouse in the sorted list. 38 mice result.

ii.
```python
subjects=sorted(tab.mouse_id.astype(str).unique())
...
subject_idx.append(subjects.index(str(row.mouse_id)))
...
'subjects':subjects,'subject_idx':np.asarray(subject_idx,np.int64),
```

iii. `mouse_id` is the SDK's canonical animal identifier in the experiment metadata table (CONVERSION_NOTES Step 5 mapping: "`mouse_id` → `subjects`, `subject_idx`; sorted unique strings and per-experiment lookup"). The count (38 mice in the local active subset) was cross-checked against the SDK tables in Step 2/Step 9.

## 1-c. How are the data split into sessions?

i. A "session" in the converted file is **one ophys experiment = one imaging plane**, not one behavioural/ophys session. Each row of the filtered experiment table becomes one entry of `neural`/`input`/`output`/`subject_idx`/`brain_region_idx`. For multiscope recordings (34 of 199 retained experiments), the 3–7 simultaneously-recorded planes of one `ophys_session_id` become that many separate "sessions", each carrying its own neurons but the *same* behavioural trials/labels. 199 decoder sessions come from 171 unique ophys sessions. `brain_region_idx` is therefore constant within a session (one `targeted_structure` per plane).

ii.
```python
tab=table.loc[ids].sort_index()          # one row per ophys_experiment_id (plane)
...
for j,(xid,row) in enumerate(tab.iterrows(),1):
    ...
    region_idx.append(np.full(info['n_neurons'],regions.index(row.targeted_structure),np.int64))
    ...
'metadata':{... 'session_unit':'ophys experiment (imaging plane)', ...}
```

iii. CONVERSION_NOTES Step 4 discrepancy table: "Planes in a multiscope session share identical trial times and ophys timestamps but contain different cells/areas… Paper samples and reports decoding over imaging planes → Treat each `ophys_experiment_id` as one decoder session; do not merge planes or duplicate cells across areas." Step 5 Key Decision 2: "Use one imaging plane per decoder session: matches paper decoding and preserves homogeneous neuron populations; multiscope planes legitimately share behavioral labels." Step 9 notes that converted plane-trials (51,075) intentionally exceed unique behavioural-session trials (44,892).

## 1-d. How are the data split into trials?

i. Trials come from the SDK `exp.trials` table. Every row with `go == True` or `catch == True` is kept. The trial window is the full native `start_time` → `stop_time` interval, so trials are variable length (71–126 bins ≈ 7–12.6 s, mean 85 bins). Within a trial, bins are laid out as `ceil((stop-start)/0.1)` consecutive 100 ms bins anchored at `start_time`, with the final edge clipped to `stop_time`.

ii.
```python
trials=exp.trials
trials=trials[trials.go.astype(bool)|trials.catch.astype(bool)].copy()
if len(trials)<2: raise ValueError('fewer than two valid trials')
...
for ti,(_,tr) in enumerate(trials.iterrows()):
    start=float(tr.start_time); stop=float(tr.stop_time)
    n=max(1,int(np.ceil((stop-start)/BIN_S)))
    edges=start+np.arange(n+1)*BIN_S; edges[-1]=stop
    centers=(edges[:-1]+edges[1:])/2
```

iii. CONVERSION_NOTES Step 1: "SDK trial logic makes go/catch mutually exclusive. Aborted trials have neither flag; auto-rewarded trials are excluded from go." Step 5 Key Decision 4: "Retain SDK go/catch trial boundaries… Trial matrices may have variable timepoint counts, while all use the same 100 ms bins." Key Decision 6: "No alignment-event cropping: trials are segmented at native `start_time` and `stop_time`; the required segmentation is trial-based, not a fixed event-centered analysis." Key Decision 10: "`ceil((stop-start)/0.1)` … so no tail is dropped."

## 1-e. How are trials filtered based on quality controls?

i. Four filters: (1) session-level — any `session_type` containing "passive" is dropped (34 single-plane + 11 multiscope experiments), because passive replay produces trial rows with artificial no-response outcomes; (2) session-level — three experiments with an empty `eye_tracking` table are dropped because pupil diameter is a mandatory output; (3) trial-level — only `go | catch` rows are kept, which by SDK construction excludes aborted and auto-rewarded attempts (74,476 of 148,231 raw trial attempts survive at source level; 51,075 plane-trials after the session exclusions); (4) an experiment with fewer than two valid trials, an empty eye-tracking table, or non-exclusive outcome flags raises and would abort the run (no such case occurred; the minimum retained session has 39 trials). Outcome exclusivity is asserted per trial.

ii.
```python
tab=tab[~tab.session_type.str.contains('passive',case=False,na=False)]
tab=tab[~tab.index.isin(MISSING_EYE_IDS)]
...
trials=trials[trials.go.astype(bool)|trials.catch.astype(bool)].copy()
if len(trials)<2: raise ValueError('fewer than two valid trials')
eye=exp.eye_tracking
if len(eye)==0: raise ValueError('missing eye tracking')
...
def outcome_value(row):
    vals=[bool(row[k]) for k in OUTCOMES]
    if sum(vals)!=1: raise ValueError(f'non-exclusive outcome flags {vals}')
    return int(np.flatnonzero(vals)[0])
```

iii. Step 3 curation rules: "Include go and catch only; exclude aborted trials (lick before change) and auto-rewarded trials. Valid outcome classes are hit, miss, false alarm, and correct reject." Step 4: passive sessions excluded because "the paper explicitly says passive viewing was not analyzed" and their outcomes are meaningless. Step 5 Key Decision 9: the three eye-less experiments are excluded "because an entire required output cannot be imputed responsibly." Step 10 edge-case review confirms "Every retained session has >=39 trials and >=4 cells."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `exp.events.filtered_events` — the AllenSDK L0-detected, SDK-filtered calcium **event magnitude** traces, one value per ophys frame per valid cell — together with `exp.ophys_timestamps`. dF/F is explicitly *not* used.

ii.
```python
ot=np.asarray(exp.ophys_timestamps,float)
ev=exp.events
traces=np.stack(ev.filtered_events.to_numpy()).astype(np.float32)
if traces.shape[1]!=len(ot): raise ValueError('event/timestamp length mismatch')
if not np.isfinite(traces).all(): raise ValueError('nonfinite filtered events')
```

iii. Step 4 discrepancy table: "Paper says all neural analyses use detected calcium events → Use SDK `filtered_events`, preserving event magnitude and avoiding slow fluorescence decay." Step 3 quotes the paper directly: "For all analysis of neural data we used the detected calcium events" and "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of GCaMP6f."

## 2-b. How is the `neural` data processed?

i. Event magnitudes are **summed** into consecutive 100 ms bins. To avoid re-scanning the trace per trial, a per-experiment cumulative sum over time is computed once (`(N, T+1)`), and each bin's value is the difference of the cumsum at the frame indices bracketing the bin, found with `searchsorted(..., side='left')` (half-open `[edge_i, edge_{i+1})` frame assignment). No normalisation by the number of frames per bin, no smoothing, no z-scoring, no cross-plane merging (each plane is its own session). Output dtype float32.

ii.
```python
event_cumsum=np.concatenate([np.zeros((traces.shape[0],1),np.float32),
                             np.cumsum(traces,dtype=np.float32,axis=1)],axis=1)

def aggregate_events(event_cumsum, timestamps, edges):
    """Sum event magnitude by time bin from one experiment-level cumulative sum."""
    lo=np.searchsorted(timestamps,edges[:-1],side='left')
    hi=np.searchsorted(timestamps,edges[1:],side='left')
    return (event_cumsum[:,hi]-event_cumsum[:,lo]).astype(np.float32)
...
neu=aggregate_events(event_cumsum,ot,edges)
```

iii. Step 5 mapping: "Sum detected event magnitudes in consecutive 100 ms physical-time bins within each trial… Preserves event magnitude; common physical bin across 10.7/30.9 Hz rigs." Key Decision 5: "Event values are summed (not averaged) so each bin represents detected event magnitude in that duration." Step 6: "Event aggregation uses timestamp `searchsorted` plus cumulative sums across all cells" (this was the fix for an initial bottleneck where the cumsum was recomputed per trial).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is performed. The AI relies on the AllenSDK pipeline, which exposes only ROIs with `roi_valid=True` in `cell_specimen_table` / `events`, plus Allen Institute session-level QC. Cell counts per retained plane range 4–666 (median 64); 29,168 cells total. The only neural-side assertions are length and finiteness checks.

ii.
```python
traces=np.stack(ev.filtered_events.to_numpy()).astype(np.float32)   # valid ROIs only
if traces.shape[1]!=len(ot): raise ValueError('event/timestamp length mismatch')
if not np.isfinite(traces).all(): raise ValueError('nonfinite filtered events')
```
(No neuron-level exclusion code exists.)

iii. Step 1: "`cell_specimen_table` is already curated to valid ROIs (`roi_valid=True`), so this SDK-level curation must be retained"; "No electrophysiology quality filtering applies because this is two-photon calcium imaging." Step 3 neuron curation rules: "ROIs that are unions, duplicates/ghosts, non-cell objects, or otherwise invalid are excluded [by the Allen pipeline]." Step 4: "Conversion should not duplicate these native processing steps." The 2,421 all-zero-neural trials flagged by the validator (4.74%) were deliberately retained: "Removing them would apply unsupported activity-dependent selection; adding values would fabricate neural activity."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the SDK trial `start_time`: bin 0 of every trial begins exactly at `start_time`, and bin edges advance in absolute time by 100 ms (`off_start = 0.0`, `off_end = None` because trials are variable length). Ophys frames are assigned to bins by `searchsorted` on the true `ophys_timestamps`, so no frame is double-counted or dropped, and all streams (neural, image, change, running, pupil) share the identical bin grid of each trial.

ii.
```python
edges=start+np.arange(n+1)*BIN_S; edges[-1]=stop
centers=(edges[:-1]+edges[1:])/2
neu=aggregate_events(event_cumsum,ot,edges)
...
'temporal_alignment_event':'SDK trial start_time; bins follow absolute ophys time',
'off_start':0.0,'off_end':None,'trial_window':'variable SDK start_time to stop_time',
```

iii. Step 5 Key Decision 6: "No alignment-event cropping: trials are segmented at native `start_time` and `stop_time`… Metadata alignment is trial start (`off_start=0`, variable `off_end=None`)." Step 10 edge-case check: "`searchsorted(..., side='left')` assigns frames to half-open bins without double counting… Trials use `ceil(duration/bin)` so no tail is dropped." Step 1: "All streams will be sampled/aligned explicitly to microscope (`ophys_timestamps`) without assuming equal sampling rates."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 100 ms (`metadata['time_bin_size'] = 100.0`), uniform for every trial and every session. Yes — the native data are **rebinned**: single-plane experiments are acquired at ~30.94 Hz (3–4 frames per bin) and multiscope planes at ~10.73 Hz (1–2 frames per bin), and both are aggregated onto the common 100 ms physical-time grid. There is no interpolation of the neural signal; frames are summed into their containing bin.

ii.
```python
BIN_S = 0.1
...
n=max(1,int(np.ceil((stop-start)/BIN_S)))
edges=start+np.arange(n+1)*BIN_S; edges[-1]=stop
...
'time_bin_size':100.0,
'neural_signal':'AllenSDK filtered_events summed in 100 ms bins',
```

iii. Step 4: "Single-plane approximately 31 Hz; multiscope approximately 11 Hz → Resample/bin by physical time onto one common bin width; never equate frame indices across rigs." Step 5 Key Decision 5: "This common physical bin has roughly 1 sample for multiscope and 3 for single-plane data, provides enough temporal resolution for 250 ms images and the one-bin change pulse, and greatly limits memory." The target format also requires "Time bins should be the same size for all trials and sessions."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `exp.stimulus_presentations`, restricted to the rows whose `stimulus_block_name == 'change_detection_behavior'` (the task block), using the `image_name`, `start_time` and `end_time` columns. Presentations whose `image_name` is not one of the 16 real natural-image names (i.e. `omitted` flashes) are dropped, so they fall back to the gray class. The trials table's `initial_image_name`/`change_image_name` are *not* used.

ii.
```python
IMAGE_NAMES = ['im000','im031','im035','im045','im054','im061','im062','im063',
               'im065','im066','im069','im073','im075','im077','im085','im106']
IMAGE_TO_ID = {x:i+1 for i,x in enumerate(IMAGE_NAMES)}

def task_stimuli(exp):
    sp=exp.stimulus_presentations
    return sp[sp.stimulus_block_name.eq('change_detection_behavior')].copy()
...
sp=task_stimuli(exp)
real=sp.image_name.isin(image_to_id)
presentations=sp[real][['start_time','end_time','image_name','is_change']]
```

iii. Step 1: "Newer releases contain multiple stimulus blocks. The cache warns that legacy task stimuli are the rows whose `stimulus_block_name` contains `change_detection`; non-task blocks must not be used to construct image outputs." Step 5 mapping: "`omitted` is gray/no image, not an image identity." Step 4 resolution: "Keep active task sessions only and use `change_detection_behavior` rows."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A 17-class categorical time series per trial: 0 = gray screen / omitted flash (the initialisation value), 1–16 = the globally fixed, alphabetically ordered list of the 16 natural images used across image sets A and B. For each stimulus presentation overlapping the trial, every bin whose **center** falls inside the half-open interval `[start_time, end_time)` of that flash (≈250 ms, i.e. 2–3 bins) is set to that image's code. Because the 500 ms inter-stimulus gray is not labelled, class 0 occupies 66.9% of all timepoints, and each image occupies ~1.9–2.2%.

ii.
```python
image=np.zeros(n,np.int64); change=np.zeros(n,np.int64)
cand=presentations[(presentations.end_time>start)&(presentations.start_time<stop)]
for _,pr in cand.iterrows():
    m=(centers>=float(pr.start_time))&(centers<float(pr.end_time))
    image[m]=image_to_id[str(pr.image_name)]
...
'output_values':[['gray']+IMAGE_NAMES, ...]
```

iii. Step 5 Key Decision 7: "Image gray class: The required identity is the image presented during non-gray screen. Gray periods and omitted flashes are class 0. Sixteen image names across sets A/B are globally stable classes." Key Decision 10: "Nearest stream interpolation is not used for categorical stimuli; interval membership prevents onset smearing." Step 7 sanity check: "gray occupies ~67% (consistent with 250 ms image/500 ms gray plus trial margins)."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image labels are written into the *same* per-trial 100 ms bin grid used for the neural matrix, decided by whether the bin center lies inside a flash interval. There is no latency correction: the label is on only during the physical 250 ms of the flash, whereas the paper's own convention assigns activity to the 750 ms interval beginning at each flash and its response window is 50–800 ms after onset (both facts are recorded in the AI's Step 3 notes but not applied here).

ii.
```python
centers=(edges[:-1]+edges[1:])/2
...
m=(centers>=float(pr.start_time))&(centers<float(pr.end_time))
image[m]=image_to_id[str(pr.image_name)]
```

iii. Step 10 sanity check 4: image interval labels were "reconstructed… directly from freshly loaded SDK tables without calling conversion functions. Every comparison passed `np.allclose()` for both rig types." Step 12: an alignment plot "overlays mean neural event activity, image identity, image-change pulse, running bin, and pupil bin on the common trial-start time axis. It shows no temporal shift or edge artifact."

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` boolean and `start_time` of the same task-block `stimulus_presentations` rows. Because `is_change` is only True for a genuine image-identity change, catch (sham-change) trials never receive a pulse, and omitted flashes cannot produce one (they were filtered out of `presentations`).

ii.
```python
presentations=sp[real][['start_time','end_time','image_name','is_change']]
...
if bool(pr.is_change) and start<=float(pr.start_time)<stop:
    bi=min(n-1,max(0,int(np.floor((float(pr.start_time)-start)/BIN_S))))
    change[bi]=1
```

iii. Step 5 mapping: "`stimulus_presentations.is_change/start_time` → `output[1]` image change; 1 in the first 100 ms bin containing each real change onset, otherwise 0… A one-bin event pulse at identity change onset." Step 9 consistency table records "Change pulses … go outcomes = 44,672 → 44,672 → Exact", i.e. exactly one pulse per go trial.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time series initialised to 0, with a **single** bin set to 1 — the bin containing the change flash onset, indexed by `floor((change_onset - trial_start)/0.1)` and clamped into `[0, n-1]`. No widening to the flash duration or the 750 ms presentation interval, and no latency shift. Positives are therefore only 1.0% of all timepoints (44,672 of 4,374,171).

ii.
```python
change=np.zeros(n,np.int64)
...
if bool(pr.is_change) and start<=float(pr.start_time)<stop:
    bi=min(n-1,max(0,int(np.floor((float(pr.start_time)-start)/BIN_S))))
    change[bi]=1
```

iii. Step 12: "Image change is positive at only 44,672 of 4,374,171 timepoints (1.02%)… a single 100 ms pulse is intrinsically harder than the paper's balanced 400 ms image snippets. Expanding the pulse would violate the instruction to use value 1 'right after' a change and would distort timing." The AI investigated the resulting 1.13× chance accuracy and deliberately kept the one-bin representation.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Two categories, `output_values[1] = ['no_change','change']`: 1 exactly where the SDK's `is_change` flag marks a real image-identity change, 0 everywhere else (including every bin of catch trials, all gray/omission bins, and all repeats). No threshold on a continuous quantity is involved — the categorisation is inherited from the SDK's boolean.

ii.
```python
'output_names':['image_identity','image_change','running_speed_quintile','pupil_diameter_quintile','trial_outcome'],
'output_values':[['gray']+IMAGE_NAMES,['no_change','change'], ...]
...
out[1]=change
```

iii. Step 10 reference-comparison table: "Change output | One-bin pulse at source `is_change` onset | SDK presentation change flag | Exact." Step 9 verified the total pulse count equals the number of go trials.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same trial-start-anchored 100 ms grid as the neural data; the pulse is placed in the bin that physically contains the change onset time (floor indexing from `start_time`, guarded so a change exactly at a boundary maps to the bin beginning at that boundary). Changes outside `[start, stop)` are ignored.

ii.
```python
bi=min(n-1,max(0,int(np.floor((float(pr.start_time)-start)/BIN_S))))
change[bi]=1
```

iii. Step 10 edge-case review: "Change at an exact boundary maps by floor to the bin immediately after onset." Step 10 sanity check 4 independently reconstructed "change-onset pulses… directly from freshly loaded SDK tables" and passed `np.allclose()`.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `exp.running_speed`, using its `speed` and `timestamps` columns (the SDK's encoder-derived running speed on its own clock, ~60 Hz, ~288k samples per session).

ii.
```python
run=exp.running_speed
rt=run.timestamps.to_numpy(float); rv=run.speed.to_numpy(float)
```

iii. Step 1: "`BehaviorSession.running_speed` — Return running speed with timestamps." Step 5 mapping: "`running_speed.timestamps/speed` → `output[2]` running speed." Step 3 notes "Running is open-loop and does not control task progression."

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation of the continuous speed onto the trial's 100 ms bin **centers**, then quintile discretisation. `interp_valid` first drops any non-finite timestamp/value pairs, sorts by time, requires ≥2 valid samples, and clamps rather than extrapolates outside the sampled range (`left=v[0]`, `right=v[-1]`). No smoothing or absolute-value transform is applied.

ii.
```python
def interp_valid(t, v, q):
    t=np.asarray(t,float); v=np.asarray(v,float)
    good=np.isfinite(t)&np.isfinite(v)
    if good.sum()<2: raise ValueError('fewer than two valid samples')
    t=t[good]; v=v[good]; order=np.argsort(t); t=t[order]; v=v[order]
    return np.interp(q,t,v,left=v[0],right=v[-1])
...
run_cont=interp_valid(rt,rv,centers)
out[2]=discretize(run_cont,run_edges)
```

iii. Step 5 mapping: "Linear interpolation to bin centers, then discretize using experiment-wide quintile edges." Step 6: "Continuous streams and percentile edges are prepared once per experiment." Step 10 sanity check 3/4: interpolated running values and the resulting bins were re-derived from freshly loaded SDK streams and matched under `np.allclose()`.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five bins (labels 0–4, `['Q1_slowest','Q2','Q3','Q4','Q5_fastest']`) defined by the 20/40/60/80th percentiles of that **experiment's entire running-speed stream** (all valid samples of the whole recording, not only the retained-trial samples, and not pooled across sessions). Assignment uses `searchsorted(edges, v, side='right')` clipped to 0–4. Because the edges are fitted on the full stream, the realised class fractions inside the converted trials are approximately but not exactly 20% each (0.190 / 0.202 / 0.206 / 0.207 / 0.195 over the full dataset).

ii.
```python
def quintile_edges(v):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    if len(v)<5: raise ValueError('insufficient samples for quintiles')
    return np.quantile(v,[.2,.4,.6,.8])

def discretize(v, edges):
    return np.clip(np.searchsorted(edges,v,side='right'),0,4).astype(np.int64)
...
run_edges=quintile_edges(rv)
```

iii. Step 5 Key Decision 8: "Behavior quintiles are fitted per experiment: This yields equal percentile bins within each recording and avoids cross-rig calibration/units differences. Repeated quantile edges are handled monotonically; digitized labels are clipped to 0-4." Step 7 notes the trial-subset fractions are "not exactly 20% because edges are fitted over each full recording, which is intentional." Recorded in metadata as `percentile_scope: 'within ophys experiment over full valid continuous stream'`.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is sampled at the centers of exactly the same 100 ms bins that define the neural matrix for that trial, so `output[2]` is element-wise aligned to `neural` by construction; per-trial assertions require equal timepoint counts.

ii.
```python
centers=(edges[:-1]+edges[1:])/2
neu=aggregate_events(event_cumsum,ot,edges)
run_cont=interp_valid(rt,rv,centers)
...
assert neu.shape[1]==inp.shape[1]==out.shape[1]
```

iii. Step 1: "Neural and behavioral streams expose their own timestamps. All streams will be sampled/aligned explicitly to microscope (`ophys_timestamps`) without assuming equal sampling rates." Step 10: "Temporal alignment | Source stream timestamps mapped to absolute 100 ms trial-start grid | Consistent; physical resampling required by mixed rig rates."

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `exp.eye_tracking`, using the processed `pupil_area` column and its `timestamps`. Diameter is reconstructed as `2*sqrt(pupil_area/pi)`. The SDK has already set `pupil_area` to NaN on `likely_blink` frames, and those NaNs are dropped in `interp_valid`, so blink frames never enter the interpolation. Experiments with an empty eye-tracking table are excluded outright.

ii.
```python
eye=exp.eye_tracking
if len(eye)==0: raise ValueError('missing eye tracking')
pupil=2.0*np.sqrt(eye.pupil_area.to_numpy(float)/np.pi)
pt=eye.timestamps.to_numpy(float)
```

iii. Step 4 discrepancy table: "SDK exposes processed area and ellipse dimensions, not a column named diameter… Whitepaper defines processed area from a circle whose diameter is ellipse major axis → Derive diameter as `2*sqrt(pupil_area/pi)` and preserve NaN until interpolation/validity handling." Step 3: "Frames marked likely blink (fit missing or area z-score >3, expanded by two adjacent frames) are represented as NaN." Step 5 Key Decision 9: "Interpolate only over SDK-valid points; do not use raw blink-contaminated values."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Identical pipeline to running speed: area→diameter conversion, removal of non-finite (blink/outlier) samples, sort, linear interpolation to the trial's bin centers with endpoint clamping, then quintile discretisation. Typical within-session missingness is 1–3%.

ii.
```python
pupil=2.0*np.sqrt(eye.pupil_area.to_numpy(float)/np.pi)
pupil_edges=quintile_edges(pupil)
...
pupil_cont=interp_valid(pt,pupil,centers)
out[3]=discretize(pupil_cont,pupil_edges)
```

iii. Step 5 mapping: "`2*sqrt(pupil_area/pi)`; interpolate over valid nonblink samples to bin centers; experiment-wide quintile edges." Step 10 sanity check 4 independently reproduced the interpolated pupil quintiles from freshly loaded SDK tables (`np.allclose()` passed).

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five bins (0–4, `['Q1_smallest','Q2','Q3','Q4','Q5_largest']`) from the 20/40/60/80th percentiles of that experiment's full valid pupil-diameter stream — again per-experiment, not global, and fitted over the whole recording rather than only the retained trials. Realised full-dataset fractions are 0.173 / 0.209 / 0.219 / 0.223 / 0.176.

ii.
```python
pupil_edges=quintile_edges(pupil)
out[3]=discretize(pupil_cont,pupil_edges)
```

iii. Step 5 Key Decision 8 (same rationale as running): per-experiment percentile fitting "avoids cross-rig calibration/units differences" — relevant here because `pupil_area` is in camera pixel units that depend on the eye-camera setup of each session.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Sampled at the same 100 ms bin centers as the neural matrix, so it is element-wise aligned by construction; equal-length assertions are made per trial.

ii.
```python
pupil_cont=interp_valid(pt,pupil,centers)
out[3]=discretize(pupil_cont,pupil_edges)
assert neu.shape[1]==inp.shape[1]==out.shape[1]
```

iii. Same rationale as running speed (Step 5 / Step 10): all streams are placed on one absolute-time trial grid derived from the ophys clock, which the whitepaper states is hardware-synchronised with the eye-tracking camera clock.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the SDK trials table — `hit`, `miss`, `false_alarm`, `correct_reject` — in that fixed order. The code asserts that exactly one is True for every retained trial.

ii.
```python
OUTCOMES = ['hit','miss','false_alarm','correct_reject']

def outcome_value(row):
    vals=[bool(row[k]) for k in OUTCOMES]
    if sum(vals)!=1: raise ValueError(f'non-exclusive outcome flags {vals}')
    return int(np.flatnonzero(vals)[0])
```

iii. Step 1: "SDK trial logic makes go/catch mutually exclusive… Valid requested outcomes are hit, miss, false alarm, and correct reject." Step 3 curation rules repeat this. The exclusivity check is one of the Step 5 planned sanity checks ("outcomes sum to trial count"), verified in Step 9 (15,682 hit / 28,990 miss / 920 false alarm / 5,483 correct reject = 51,075 trials).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is mapped to an integer 0–3 and **broadcast across all timepoints of the trial**, so `output[4]` is a constant row of the `(5, n_timepoints)` matrix. The value is static per trial semantically; the repetition exists only so that all five outputs can share one 2-D array.

ii.
```python
out=np.empty((5,n),np.int64)
...
# Validator/model accepts static output as (doutput,), but outputs are
# jointly represented; repeat static outcome over time for a uniform
# (5,T) output and true static semantics.
out[4]=outcome_value(tr)
```

iii. Step 10 comparison table: "Outcome | Repeat one SDK static class across T | SDK mutually exclusive trial outcome flags; task says static per trial | Semantically exact and compatible with joint time-varying output matrix." README: "This static value is repeated across time for a uniform output matrix." (Note: the Step 5 mapping table still describes it as a "static integer vector shape `(1,)`", which is stale relative to the implemented and later-documented choice.)

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is a mixture of exclusion, interpolation-level cleaning, and fail-fast assertions:
- **Missing eye tracking (whole session)**: three experiments with zero eye-tracking rows are excluded by a hard-coded id set, discovered by an exhaustive pre-scan of all 202 active experiments; a defensive `if len(eye)==0: raise` also exists.
- **Blink / outlier pupil samples and any NaN behavioural samples**: dropped before interpolation (`np.isfinite` mask) rather than imputed; the remaining valid samples are interpolated across the gap.
- **Extrapolation beyond a stream's sampled range**: clamped to the first/last valid value instead of producing NaN, so no NaN ever reaches the outputs (unlike a NaN→bin-0 fallback).
- **Unsorted timestamps**: sorted defensively inside `interp_valid`.
- **Structural anomalies** (event/timestamp length mismatch, non-finite events, non-exclusive outcome flags, <2 valid trials, <5 behaviour samples): raise immediately; there is no per-session `try/except`, so any such case would abort the whole run rather than skip that session.
- **Whole-dataset invariants**: asserted after the loop (equal list lengths, ≥2 trials per session, matching trial counts across neural/input/output), plus per-trial finiteness and shape assertions.
- Not guarded: a trial whose `stop_time` exceeded the last ophys timestamp would silently yield zero-valued trailing bins rather than being clipped; I verified on three experiments that trials end ~600 s before the recording ends, so this case does not arise in this dataset.

ii.
```python
MISSING_EYE_IDS = {795953296, 833631914, 806456687}
tab=tab[~tab.index.isin(MISSING_EYE_IDS)]
...
if len(eye)==0: raise ValueError('missing eye tracking')
good=np.isfinite(t)&np.isfinite(v)
if good.sum()<2: raise ValueError('fewer than two valid samples')
t=t[good]; v=v[good]; order=np.argsort(t); t=t[order]; v=v[order]
return np.interp(q,t,v,left=v[0],right=v[-1])
...
assert neu.shape[1]==inp.shape[1]==out.shape[1]
assert np.isfinite(neu).all() and np.isfinite(out).all()
...
assert all(len(s)>=2 for s in neural)
```

iii. Step 5 Key Decision 9: the eye-less experiments are excluded "because an entire required output cannot be imputed responsibly." Step 10: "Three active files had no eye data: Excluded because pupil diameter is mandatory and whole-session imputation is unjustified." Step 3: blink frames "are represented as NaN" by the SDK, so the AI interpolates "only over SDK-valid points." The all-zero-neural trials were explicitly judged to be genuine sparsity, not missing data, and retained.

## 9-a. What are the most time-consuming steps of the code?

i. (1) `cache.get_behavior_ophys_experiment()` plus the lazy NWB reads it triggers — I/O bound and by far the largest share (3.6–6.5 s of the ~5.6 s mean per experiment; total full run 1,109 s / 18.5 min for 199 experiments). (2) Materialising the event matrix (`np.stack` of up to 666 × 150,000 floats) and its full-session cumulative sum. (3) The per-trial Python loop, dominated by the two `interp_valid` calls and the pandas boolean mask that selects candidate stimulus presentations, executed ~250 times per experiment. (4) Pickling the 2.9 GB output. An earlier version recomputed the cumulative sum inside the trial loop, which made the run quadratic in trial count (20–33 s/session, projected >60 min); the AI detected this during the full run, killed it, hoisted the cumsum, and re-ran at 4–10 s/session.

ii.
```python
exp=cache.get_behavior_ophys_experiment(int(xid))            # dominant cost
traces=np.stack(ev.filtered_events.to_numpy()).astype(np.float32)
event_cumsum=np.concatenate([np.zeros((traces.shape[0],1),np.float32),
                             np.cumsum(traces,dtype=np.float32,axis=1)],axis=1)
```

iii. Step 6: "AllenSDK object creation dominates runtime (~15 minutes for a prior exhaustive 202-file scan)"; speedups were "Each experiment is loaded exactly once and all source tables/streams are reused" and "Event aggregation uses timestamp `searchsorted` plus cumulative sums across all cells." Step 9: "Full conversion initially projected too slowly because a session-wide event cumulative sum was recomputed per trial… Runtime improved from 20-33 s to typically 4-10 s/session; final runtime was 1,109.2 s (18.5 min)."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain:
- The outer per-trial loop `for ti,(_,tr) in enumerate(trials.iterrows())`. All bin edges could be generated once for all trials and `aggregate_events` called a single time on the concatenated edge array, then split per trial.
- The inner `for _,pr in cand.iterrows()` over stimulus presentations of a trial. Image labels could be assigned in one shot with `np.searchsorted` of the bin centers into the presentation start/end arrays, and all change pulses placed with one vectorised index computation.
- The per-trial interpolation calls: `np.interp` could be called once per experiment on a concatenated center array instead of ~250 times.
The neural aggregation itself is already fully vectorised across neurons and bins by the cumulative-sum trick, which is the step that mattered most.

ii.
```python
for ti,(_,tr) in enumerate(trials.iterrows()):
    ...
    cand=presentations[(presentations.end_time>start)&(presentations.start_time<stop)]
    for _,pr in cand.iterrows():
        m=(centers>=float(pr.start_time))&(centers<float(pr.end_time))
        image[m]=image_to_id[str(pr.image_name)]
    ...
    run_cont=interp_valid(rt,rv,centers)
    pupil_cont=interp_valid(pt,pupil,centers)
```

iii. The AI's stated plan (Step 5 Efficiency Plan) was to "Use `searchsorted`, `reduceat`/bin indices, and interval slicing rather than per-timepoint Python loops" and to iterate "Only overlapping stimulus rows… per trial" (Step 6). It vectorised across neurons/timepoints but consciously kept per-trial and per-presentation loops, on the grounds that SDK loading dominates runtime and the achieved 18.5 min full run was within the 15–22 min budget it had estimated.

## 9-c. What processing does the code repeat multiple times?

i. Two genuine redundancies remain inside `convert_experiment`:
- `interp_valid(rt, rv, centers)` and `interp_valid(pt, pupil, centers)` are called once per trial, and each call re-runs `np.isfinite` masking, boolean indexing, and `np.argsort` on the **entire** session-long running (~288k samples) / pupil (~145k samples) arrays. Only the final `np.interp` query depends on the trial. Benchmarked at ~0.6 s per 250 calls per stream, i.e. roughly 1 s per experiment (~20% of per-experiment compute, ~3–4 min of the full run).
- `cand=presentations[(...)&(...)]` re-filters the full task-stimulus DataFrame (~4–5k rows) for every trial, where one `searchsorted` into sorted start times would do.
- Minor: `quintile_edges` is correctly hoisted out of the trial loop, and the previously per-trial `event_cumsum` was hoisted after the AI caught it, so the large repetitions were eliminated.

ii.
```python
for ti,(_,tr) in enumerate(trials.iterrows()):
    ...
    cand=presentations[(presentations.end_time>start)&(presentations.start_time<stop)]   # repeated full-table mask
    run_cont=interp_valid(rt,rv,centers)      # re-sorts/re-filters 288k samples every trial
    pupil_cont=interp_valid(pt,pupil,centers) # re-sorts/re-filters 145k samples every trial
```

iii. The AI documented the intent not to repeat work — Step 6: "Each experiment is loaded exactly once and all source tables/streams are reused"; "Continuous streams and percentile edges are prepared once per experiment" (true for the quantile edges, not for the interpolation preprocessing). Step 9/10 record the one repetition it did find and fix: "A session-wide event cumulative sum was mistakenly recomputed once per trial… It was moved outside the trial loop."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little, and all of it cheap:
- `raw_plot` — the plotting tuple (bin centers, up to 8 neurons, image, change, continuous and discretised behaviour) is assembled for the first trial of **every** experiment even when `--show-processing` is not passed, then discarded.
- `info` fields that no downstream step consumes: `trial_duration_min_s`/`max_s`, `ophys_frame_rate_measured`, and the per-experiment quintile edges (kept in `metadata['session_info']` for provenance only).
- The full-array validation passes `np.isfinite(traces).all()` over the whole (N × ~150k) event matrix and the per-trial `np.isfinite(neu).all()`/`np.isfinite(out).all()` assertions.
- `event_cumsum` is built over the entire recording, including the ~20 min of gray-screen/movie blocks and aborted-trial time that no trial window ever indexes (though computing it once is still cheaper than any per-trial alternative).
- `gc.collect()` after every experiment.
- Discretised outputs are stored as `int64`, twice the width needed for 17 classes (the reference uses `int8`), which is the main driver of the 2.9 GB pickle alongside the float32 neural data.

ii.
```python
if raw_plot is None:
    raw_plot=(centers,neu[:min(8,len(neu))],image,change,run_cont,pupil_cont,out[2],out[3])
...
if not np.isfinite(traces).all(): raise ValueError('nonfinite filtered events')
assert np.isfinite(neu).all() and np.isfinite(out).all()
...
out=np.empty((5,n),np.int64)
...
del exp,n,i,o,raw; gc.collect()
```

iii. Step 6: "Implementation includes explicit assertions for event/timestamp lengths, finite values, exclusive outcomes, trial counts, and matching per-trial dimensions" — the redundant checks are deliberate validation rather than oversight. Step 5 Key Decision 11: "Storage types: Neural float32, empty input float32, categorical outputs int64, indices int64. This is validator-compatible and memory efficient." Step 6 lists releasing SDK references per experiment as a deliberate memory speedup.
