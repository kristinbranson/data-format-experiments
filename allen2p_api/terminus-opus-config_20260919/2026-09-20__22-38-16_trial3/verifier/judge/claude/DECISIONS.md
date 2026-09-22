# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads everything exclusively through the AllenSDK `VisualBehaviorOphysProjectCache`, opened as a **local** cache on `/app/data` (`from_local_cache(cache_dir='/app/data', use_static_cache=False)`; it notes `use_static_cache=True` fails because that layout expects `visual-behavior-ophys/manifests`). The canonical `get_ophys_experiment_table()` gives one row per imaging plane (`ophys_experiment_id`). Because the local cache holds only 284 of the 1936 released experiments, the AI restricts the table to the experiment IDs whose NWB files are actually present, by globbing the cache's `behavior_ophys_experiments/*.nwb` filenames (filenames only — the NWB files are never opened directly). It then drops **passive** experiments (`passive == True`, i.e. OPHYS_2/5), leaving 202 active experiments → 174 ophys sessions → 38 mice → 29,444 cells. Both project codes present in the cache are kept (`VisualBehavior` 239 experiments, single-plane ~31 Hz; `VisualBehaviorMultiscope` 45 experiments, up to 7 planes, ~10.7 Hz). Sessions are then processed in parallel (24 worker processes), each worker calling `cache.get_behavior_ophys_experiment(eid)` for every plane of its session.

ii.
```python
def get_cache():
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    return VisualBehaviorOphysProjectCache.from_local_cache(
        cache_dir=CACHE_DIR, use_static_cache=False)


def downloaded_experiment_table(cache):
    """Experiment-table rows for the NWB files that are actually in the local cache."""
    et = cache.get_ophys_experiment_table()
    ids = sorted(int(os.path.basename(f).split('_')[-1].split('.')[0])
                 for f in glob.glob(NWB_GLOB))
    return et.loc[et.index.isin(ids)].copy()
```
```python
    cache = get_cache()
    et = downloaded_experiment_table(cache)
    act = et[~et['passive'].astype(bool)].copy()
    print(f'downloaded experiments: {len(et)}; active (non-passive): {len(act)}')
    groups = act.groupby('ophys_session_id')
    session_ids = sorted(groups.groups.keys())
```
```python
    for eid in experiment_ids:
        try:
            exps.append((eid, cache.get_behavior_ophys_experiment(eid)))
        except Exception as e:
            print(f'  [session {session_id}] could not load experiment {eid}: {e}')
```

iii. From CONVERSION_NOTES Steps 1/2/4/5: "`VisualBehaviorOphysProjectCache` … Entry point to the released dataset" is the only sanctioned API, and the tutorials use exactly this pattern; the AI explicitly records "no direct NWB/h5py access". The glob-restriction exists because "284 NWB files are actually present locally (a subset of the 1936-experiment release)" and calling `get_behavior_ophys_experiment` on an absent id would fail. Passive sessions are excluded because "Passive sessions (OPHYS_2/5) *do* have a trials table (go/catch) but **0 licks and 0 rewards** (the lick spout is retracted), so trial outcome / behaviour cannot be decoded there". Both rigs are kept because "The paper's restriction served its scientific question (novelty, cell-class comparison). For decoder training we keep all active (non-passive) behaviour sessions from both rigs and all experience levels, which is the 'Visual Behavior' task data."

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values. The AI reads `mouse_id` from each loaded experiment's `metadata` (stored in the per-session `info` dict) rather than from the experiment table, and builds `subjects` as the list of mouse ids in order of first appearance while iterating sessions; `subject_idx[session]` is the index of that session's mouse in `subjects`. Result: 38 mice, 2–9 sessions each.

ii.
```python
    info = dict(
        ...
        mouse_id=str(meta['mouse_id']),
        ...
    )
```
```python
        mouse = info['mouse_id']
        if mouse not in subjects:
            subjects.append(mouse)
        data['subject_idx'].append(subjects.index(mouse))
    ...
    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5: "`subjects` = `mouse_id` strings; `subject_idx` = index per session." `mouse_id` is the SDK's unique animal identifier; the count (38) was cross-checked against the experiment table for the 284 locally present experiments (Step 2 table, Step 9 consistency table "Subjects … 38 mice in the local cache | 38 | YES"). Sanity check in Step 10 Check 2: "`subjects[subject_idx[session]]` vs. `metadata[mouse_id]` → match".

## 1-c. How are the data split into sessions?

i. One output "session" = one `ophys_session_id`. All imaging planes (`ophys_experiment_id`s) recorded *simultaneously* in that session are concatenated along the neuron axis into a single population. The experiment table is grouped with `act.groupby('ophys_session_id')` and sessions are processed in ascending `ophys_session_id` order (no per-mouse chronological sorting). Behavioural streams (`trials`, `stimulus_presentations`, `running_speed`, `eye_tracking`) are taken from the first plane only, since they are identical across simultaneously recorded planes. 174 active sessions were attempted, 171 kept.

ii.
```python
    groups = act.groupby('ophys_session_id')
    session_ids = sorted(groups.groups.keys())
    ...
    for i, sid in enumerate(session_ids):
        exp_ids = list(groups.get_group(sid).index.values)
        jobs.append((sid, exp_ids, args.neural, show, path, args.zscore))
```
```python
    ref = exps[0][1]           # behaviour streams are identical across planes
    meta = ref.metadata
```

iii. CONVERSION_NOTES Step 5: "A **session** in the target format = one **ophys session** (`ophys_session_id`), i.e. all imaging planes recorded **simultaneously** in that session are concatenated along the neuron axis. This is the physically correct notion of a simultaneous population recording (single-plane sessions have 1 plane, Multiscope up to 7)." Keeping planes together also gives the `brain_regions` axis meaning (VISp and VISl neurons can co-occur in one session).

## 1-d. How are the data split into trials?

i. Trials are the rows of the SDK `trials` table for which `go | catch` is True (this is exactly `trial_masks.contingent_trials`, and it automatically excludes aborted and auto-rewarded trials, whose `go` and `catch` flags are both False). Rows with a non-finite `change_time` are dropped. Each surviving trial is then represented **not** by the experiment's `start_time`→`stop_time` interval but by a **fixed window of [−3.0 s, +3.0 s] around `change_time`**, tiled into **24 bins of 250 ms**. Bin edges are built in absolute sync-clock seconds, identically for neural and behavioural streams.

ii.
```python
BIN_SIZE = 0.25           # s, = image presentation duration, 1/3 of the 750 ms flash cycle
OFF_START = -3.0          # s relative to the change (4 flash cycles before)
OFF_END = 3.0             # s relative to the change (4 flash cycles after)
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 24
```
```python
    trials = ref.trials
    keep = (trials['go'].astype(bool) | trials['catch'].astype(bool))
    tr = trials[keep].copy()
    tr = tr[np.isfinite(tr['change_time'].values)]
    if len(tr) < 2:
        return None
    ...
    change_times = tr['change_time'].values.astype(np.float64)
    edges = change_times[:, None] + (OFF_START + BIN_SIZE * np.arange(NBINS + 1))[None, :]
    centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
```

iii. CONVERSION_NOTES Step 5 "Trials": "keep rows with `go | catch` (== `trial_masks.contingent_trials`), dropping `aborted` and `auto_rewarded` (per the decoder-task instruction)." For the window: "every go/catch trial has ≥ 3.02 s between trial start and the change and ≥ 4.20 s between the change and trial stop, so the window lies inside the trial; the minimum interval between successive go/catch changes is 7.51 s, so windows never overlap. 6 s = exactly **8 flash cycles** (4 before, 4 after the change)." All three of those numbers were measured from the data in Step 4. The AI explicitly considered using the trial table's own `start_time`/`stop_time` (trajectory step 56: "'trial' should reflect the experimental definition … the format allows variable n_timepoints per trial") and chose the fixed change-locked window instead so that every trial has an identical, stimulus-aligned bin grid across two rigs with different frame rates.

## 1-e. How are trials filtered based on quality controls?

i. Five filters, in order: (1) `go | catch` only (drops aborted and auto-rewarded); (2) `np.isfinite(change_time)`; (3) the ±3 s window must lie entirely inside the coverage of *all* streams — ophys timestamps of every plane, running timestamps, eye-tracking timestamps, and the `change_detection_behavior` stimulus block (counted as `n_trials_dropped_window`); (4) the trial must have exactly one of `hit`/`miss`/`false_alarm`/`correct_reject` set, otherwise it is dropped (`n_trials_dropped_outcome`); (5) sessions with fewer than 2 surviving trials are dropped entirely. In the full run, filters 3 and 4 dropped 0 trials and filter 5 dropped 0 sessions. 43,975 trials were kept over 171 sessions (go fraction 0.8746 vs. the whitepaper's 0.875).

ii.
```python
    keep = (trials['go'].astype(bool) | trials['catch'].astype(bool))
    tr = trials[keep].copy()
    tr = tr[np.isfinite(tr['change_time'].values)]
    if len(tr) < 2:
        return None
```
```python
    t_lo = max([t[0] for t in ophys_t] + [run_t[0], eye_t[0], pres_start[0]])
    t_hi = min([t[-1] for t in ophys_t] + [run_t[-1], eye_t[-1],
                                           pres_start[-1] + FLASH_DURATION])
    valid = ((change_times + OFF_START) >= t_lo) & ((change_times + OFF_END) <= t_hi)
    n_dropped_window = int((~valid).sum())
    tr = tr[valid]
    change_times = change_times[valid]
    if len(tr) < 2:
        return None
```
```python
    ok = outcome >= 0
    n_dropped_outcome = int((~ok).sum())
    if n_dropped_outcome:
        neural = neural[:, ok, :]
        ...
    if ntrials < 2:
        return None
```

iii. CONVERSION_NOTES Step 3 "Curation Steps": "`trial_masks.contingent_trials` → keep **go** and **catch** trials, drop **aborted** and **auto_rewarded** (matches the decoder-task instruction exactly)." Step 10 Check 5 "Edge cases" lists each remaining filter and its measured count: "Trials whose ±3 s window would fall outside the ophys/running/eye/stimulus coverage are dropped …; the full run dropped **0**, confirming the window choice is safe. Trials with a non-finite `change_time` are dropped before anything else. Trials that are go|catch but have no outcome flag set are dropped (0 in the full run). Sessions with fewer than 2 usable trials would be dropped (none were)." The go-fraction check (0.8746 ≈ 0.875) was the planned sanity check that the trial selection reproduces the experiment's matrix-sampling schedule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `BehaviorOphysExperiment.dff_traces['dff']` — the Allen-pipeline ΔF/F traces of every valid ROI — together with `ophys_timestamps` for alignment. `--neural events` and `--neural filtered_events` (the L0-deconvolved calcium events the reference paper used) remain available as options, but the default and the value used for the delivered `converted_data.pkl` is `dff`.

ii.
```python
    ap.add_argument('--neural', default='dff',
                    choices=['events', 'filtered_events', 'dff'])
```
```python
    for eid, e in exps:
        tbl = e.events if neural_source in ('events', 'filtered_events') else e.dff_traces
        col = {'events': 'events', 'filtered_events': 'filtered_events',
               'dff': 'dff'}[neural_source]
        if len(tbl) == 0:
            continue
        traces = np.vstack(tbl[col].values)                 # (ncells, T)
```
(The full run log confirms: `processing 174 sessions with neural source "dff"`.)

iii. This is a deliberate, measured deviation from the reference paper. CONVERSION_NOTES Step 3: the paper states "all analyses on discrete calcium events that were regressed from the raw fluorescence traces". Step 7 "Choice of neural signal (measured, not assumed)": "Raw L0 events are nonzero in only ~0.3% of frames, so most bins are exactly 0 (the verifier warned that all neural data is zero for many trials). I therefore measured all three Allen pipeline signals on the same sessions (everything else identical)" — a 6-session controlled comparison gave dF/F 0.432/0.610/0.235/0.287/0.303 vs. events 0.272/0.554/0.225/0.243/0.244 validation balanced accuracy. "**Decision**: use **dF/F** … This is the deviation from the reference paper explicitly allowed by the task …: the extreme sparsity of the event trains makes an instantaneous per-bin decoder fail, and dF/F integrates the same calcium transients over the bin." Step 1 also records that ΔF/F does not need to be recomputed — the SDK supplies it already neuropil-corrected and baseline-normalised.

## 2-b. How is the `neural` data processed?

i. Two operations only: (1) **temporal bin-averaging** — for each neuron, the mean ΔF/F over the ophys frames whose timestamps fall in each 250 ms bin, computed with a vectorised cumulative-sum + `searchsorted` over the whole (ntrials × 25) edge grid at once; bins containing no frame are set to 0 and counted (`n_empty_neural_bins`, 0 in the full run). (2) **plane concatenation** — the per-plane binned arrays are concatenated along the neuron axis, each plane contributing its `targeted_structure` to the per-neuron region list. No z-scoring, no normalisation, no smoothing is applied by default (`--zscore` exists but was not used). Output per trial: `(n_neurons, 24)` float32.

ii.
```python
def bin_means(values, timestamps, edges):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    ntrials, nedge = edges.shape
    idx = np.searchsorted(timestamps, edges.ravel(), side='left').reshape(ntrials, nedge)
    counts = np.diff(idx, axis=1)                      # (ntrials, nbins)
    csum = np.concatenate([np.zeros((values.shape[0], 1)), np.cumsum(values, axis=1)],
                          axis=1)
    sums = csum[:, idx[:, 1:]] - csum[:, idx[:, :-1]]  # (n_signals, ntrials, nbins)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = sums / counts[None, :, :]
    out[np.broadcast_to(counts[None, :, :] == 0, out.shape)] = 0.0
    return out, counts
```
```python
        ts = e.ophys_timestamps
        binned, counts = bin_means(traces, ts, edges)       # (ncells, ntrials, nbins)
        zero_bins += int((counts == 0).sum())
        neural_planes.append(binned.astype(np.float32))
        region_names += [e.metadata['targeted_structure']] * traces.shape[0]
    neural = np.concatenate(neural_planes, axis=0)          # (nneurons, ntrials, nbins)
```

iii. CONVERSION_NOTES Step 1: "dF/F does not need to be computed: `dff_traces` is provided by the Allen pipeline." Step 5 Key Decision 2: "simultaneously recorded planes form one population; gives the `brain_regions` axis a meaning (VISp/VISl in the same session)." Step 12 justifies not z-scoring: "z-scoring changes nothing (the decoder starts from an SVD of the session data, so it is scale-tolerant) — raw dF/F is kept because it is the unmodified pipeline output." Step 6 records the efficiency motivation for `bin_means`: "vectorised cumsum/searchsorted binning of all neurons × trials at once → O(T) instead of a Python loop over trials/bins (~10× vs. per-trial slicing)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level filtering is applied. The AI relies entirely on the Allen pipeline's own ROI curation, which the SDK applies by default (`exclude_invalid_rois=True`, so `dff_traces`/`events`/`cell_specimen_table` contain only `valid_roi == True` cells), plus the release-level session QC (d′ ≥ 1, z-drift, motion, photobleaching). Session-level exclusions exist but are not neural-quality based: passive sessions (no behaviour) and 3 sessions with an empty eye-tracking table are dropped. Empty planes (`len(tbl) == 0`) are skipped, and a session with no usable plane returns `None`.

ii.
```python
        if len(tbl) == 0:
            continue
        ...
    if not neural_planes:
        return None
```
```python
        neuron_selection=('all cells released by the AllenSDK, i.e. valid ROIs only '
                          '(exclude_invalid_rois=True is the SDK default)'),
```

iii. CONVERSION_NOTES Step 3 "Neuron curation rules": "the Allen pipeline already excludes non-somatic/duplicate/union/edge/dim ROIs via the ROI-filtering classifier; the SDK exposes only `valid_roi == True` cells … Session-level QC (saturation, photobleaching, z-drift > 10 µm, motion, interictal events, d-prime ≥ 1) has already been applied to the released data." Step 4 discrepancy table: "Cell filtering … No extra filtering needed; use SDK defaults." Step 10 Check 3 confirms this matches the reference: "(b) Neuron filtering … rely on the SDK defaults, no extra filtering | YES".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is **`trials.change_time`** — the real image change on go trials and the sham change on catch trials — which the AI verified always coincides exactly with a stimulus-flash onset (100% of change times appear in `stimulus_presentations.start_time`). Alignment is done in absolute sync-clock seconds: the 25 bin edges of trial *i* are `change_time_i + (−3.0 + 0.25·k)`, and each plane's frames are assigned to bins via `searchsorted` on that plane's own `ophys_timestamps`. This means each imaging plane is aligned with its own timestamps (important because Multiscope planes are scanned at different times), and all data streams share the identical edge grid. `metadata['temporal_alignment_event']` documents this.

ii.
```python
    edges = change_times[:, None] + (OFF_START + BIN_SIZE * np.arange(NBINS + 1))[None, :]
    centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
    ...
        ts = e.ophys_timestamps
        binned, counts = bin_means(traces, ts, edges)
```
```python
        temporal_alignment_event=('time of the image change (`trials.change_time`; the sham '
                                  'change time on catch trials), which always coincides with '
                                  'a stimulus flash onset'),
        off_start=OFF_START,
        off_end=OFF_END,
```

iii. CONVERSION_NOTES Step 1: "Alignment event for go/catch trials is `change_time` … which is the canonical alignment used throughout the SDK/paper analyses" (the reference `swdb/save_trial_response_df.py` aligns to `trials.change_time` with a [−4, +8] s window). Step 4 verified "Change times fall on flash onsets … **100%** of change_times are in `stimulus_presentations.start_time`". Step 10 Check 3(c): "align to `change_time`, window [−3, +3] s, bins defined in absolute sync time and filled with the ophys frames that fall inside | YES (we use exact timestamps instead of a fixed frame count, which is more accurate for the two different frame rates)." Step 10 Check 2 spot-checked 9 (trial, neuron, bin) triples against an independent naive recomputation from `dff_traces` and got `np.allclose` matches.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **250 ms bins, 24 bins per trial, identical for every trial and session** (`metadata['time_bin_size'] = 250.0` ms). Yes — explicit rebinning is applied: the native ophys sampling (30.94 Hz ≈ 32 ms for single-plane VisualBehavior sessions, 10.73 Hz ≈ 93 ms for Multiscope planes) is averaged into the common 250 ms grid, and the behavioural streams (60 Hz running, ~60 Hz eye tracking) are averaged into the same grid. A `--bin` CLI option allows other sizes; 250 ms was used throughout.

ii.
```python
BIN_SIZE = 0.25           # s, = image presentation duration, 1/3 of the 750 ms flash cycle
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 24
...
        time_bin_size=BIN_SIZE * 1000.0,
```
```python
    assert all(t.shape[1] == NBINS for s in data['neural'] for t in s)
    assert all(o.shape == (len(OUTPUT_NAMES), NBINS) for s in data['output'] for o in s)
```

iii. CONVERSION_NOTES Step 5: "**Bin size: 250 ms** = the image presentation duration and exactly 1/3 of the 750 ms flash cycle, so bin edges align with image onsets/offsets for every trial (the window is aligned to a flash onset). 6 s / 0.25 s = **24 bins per trial**. It is also ≥ 2 ophys frames even on the 11 Hz Multiscope rig, so every bin contains data from both rigs, which is required because the format demands one common bin size." Step 2 flags the driving constraint: "Ophys frame rate differs between rigs: 30.9 Hz (VisualBehavior, single plane) vs 10.7 Hz (VisualBehaviorMultiscope). The target format requires one common time-bin size for all trials/sessions." Step 3 links it to the paper: the paper assigned events to 750 ms image-presentation intervals and decoded the first 400 ms, so 250 ms bins are "compatible; finer because the provided decoder is per-bin". Step 12 reports a 500 ms-bin control run and keeps 250 ms because the larger bin "would no longer tile the 750 ms flash cycle exactly".

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `stimulus_presentations`, restricted to the `change_detection_behavior` stimulus block: the `start_time` and `image_name` columns (sorted by onset). Not the trial-table fields `initial_image_name`/`change_image_name`. Because `image_name` takes the value `'omitted'` on omitted flashes, the category set is the 16 natural images of image sets A and B plus `'omitted'` → 17 classes.

ii.
```python
IMAGE_NAMES = ['im000', 'im031', 'im035', 'im045', 'im054', 'im061', 'im062', 'im063',
               'im065', 'im066', 'im069', 'im073', 'im075', 'im077', 'im085', 'im106']
IMAGE_VALUES = IMAGE_NAMES + ['omitted']
IMAGE_TO_IDX = {n: i for i, n in enumerate(IMAGE_VALUES)}
```
```python
    sp = ref.stimulus_presentations
    beh = sp[sp['stimulus_block_name'] == BEHAVIOR_BLOCK].copy()
    pres_start = beh['start_time'].values.astype(np.float64)
    pres_image = beh['image_name'].values.astype(object)
    pres_change = beh['is_change'].astype(bool).values
    order = np.argsort(pres_start)
    pres_start, pres_image, pres_change = pres_start[order], pres_image[order], pres_change[order]
```

iii. CONVERSION_NOTES Step 5 Variable Mapping: "`stimulus_presentations.image_name` → `output[0]` 'image_identity' … classes = 16 images (sets A+B) + `omitted`", following the paper's convention of "assigning behavioural events to each image presentation interval … 750 ms interval beginning with each image presentation". Step 10 Check 5: "`stimulus_presentations` is restricted to the `change_detection_behavior` block, so the 5 min grey screens and the natural-movie block can never leak into a trial." Step 4 verified 8 images per session and the two image sets (A: im061/062/063/065/066/069/077/085; B: im000/031/035/045/054/073/075/106).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each of the 24 bin centres is mapped to the stimulus presentation whose onset most recently preceded it (`searchsorted(..., side='right') - 1`, clipped to the valid range), and that presentation's `image_name` is looked up in the fixed global `IMAGE_TO_IDX` mapping to give an int64 code 0–16. The label is therefore held constant through the 500 ms grey ISI that follows each 250 ms image, i.e. it labels the whole 750 ms *image-presentation interval*. The mapping is global (hard-coded, identical for all sessions), and `output_values[0] = IMAGE_VALUES` records the names.

ii.
```python
    pidx = np.searchsorted(pres_start, centers.ravel(), side='right') - 1
    pidx = np.clip(pidx, 0, len(pres_start) - 1)
    img_names = pres_image[pidx]
    img_identity = np.array([IMAGE_TO_IDX[n] for n in img_names],
                            dtype=np.int64).reshape(ntrials, NBINS)
```

iii. CONVERSION_NOTES Step 5 Key Decision 8: "**Image identity during the gray ISI**: labelled with the image of the current 750 ms presentation interval (the paper's image-interval convention), so every bin has a label; omission intervals get the separate class `omitted`." A global mapping across all sessions is needed because different sessions use image set A or B; the measured distribution (Step 9) is "16 images each 0.058–0.063, `omitted` 0.031", i.e. uniform within an image set, matching the matrix-sampling schedule.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed on exactly the same `(ntrials, 24)` bin grid as the neural data — the identical `edges`/`centers` arrays derived from `change_time`. No separate resampling step exists, so alignment is guaranteed by construction. The AI verified independently that on go trials the bins covering t ∈ [0, 0.75) carry the change image and the preceding bins carry the initial image.

ii.
```python
    edges = change_times[:, None] + (OFF_START + BIN_SIZE * np.arange(NBINS + 1))[None, :]
    centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
    ...
    pidx = np.searchsorted(pres_start, centers.ravel(), side='right') - 1
```
```python
    output_list = [np.stack([img_identity[i], img_change[i], run_q[i], pupil_q[i],
                             outcome_bins[i]], axis=0) for i in range(ntrials)]
```

iii. CONVERSION_NOTES Step 5: bin edges are aligned to a flash onset because `change_time` is always a flash onset and 250 ms tiles the 750 ms cycle exactly, "so bin edges align with image onsets/offsets for every trial". Step 10 Check 2 sanity check: "**Output image identity / change** | for 3 random (trial, bin) per session: last `stimulus_presentations` row whose `start_time <= bin centre` | exact match for all 9". Step 5 planned check: "The image label at t in [0, 0.75) equals `trials.change_image_name` on go trials, and `initial_image_name` just before t=0."

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `stimulus_presentations['is_change']` (from the `change_detection_behavior` block), evaluated at the same bin centres as image identity. `trials.change_time` enters only indirectly, as the anchor of the bin grid.

ii.
```python
    pres_change = beh['is_change'].astype(bool).values
    ...
    img_change = pres_change[pidx].astype(np.int64).reshape(ntrials, NBINS)
```

iii. CONVERSION_NOTES Step 5 Variable Mapping: "`stimulus_presentations.is_change` / `trials.change_time` → `output[1]` 'image_change' … binary; only go trials contain 1s (catch = sham change, no identity change)". This mirrors the paper's change-vs-repeat decoder, which classifies each image presentation as a change or a repeat.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The same `pidx` lookup used for image identity indexes `pres_change`, so every bin inherits the `is_change` flag of its 750 ms image-presentation interval. Because `change_time` is exactly a flash onset and bins are 250 ms, this sets the three bins covering [change_time, change_time + 0.75 s) — bins 12, 13, 14 — to 1 on go trials, and nothing to 1 on catch trials (a sham change has `is_change == False`). No separate go/catch branch is coded; the SDK's `is_change` already encodes it. Measured fraction of 1s: 0.1093 = (3/24) × 0.8746, exactly as predicted.

ii.
```python
    pidx = np.searchsorted(pres_start, centers.ravel(), side='right') - 1
    pidx = np.clip(pidx, 0, len(pres_start) - 1)
    img_change = pres_change[pidx].astype(np.int64).reshape(ntrials, NBINS)
```

iii. CONVERSION_NOTES Step 5: "1 for the bins inside the 750 ms presentation interval of a changed image, 0 otherwise". Step 9 consistency table: "image_change fraction … 0.1093 = (3/24 bins) × 0.8746 go trials = 0.1093 | exact". Step 10 Check 2: "Change structure | go trials must have `image_change == 1` in bins 12–14 (t = 0–0.75 s) and catch trials never | True in all 3 sessions." The 750 ms extent (rather than a single bin) is justified as the image-presentation interval convention used by the paper.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — `is_change` is already boolean. It is cast to int64 giving 2 classes, named `['no_change', 'change']` in `output_values[1]`.

ii.
```python
    img_change = pres_change[pidx].astype(np.int64).reshape(ntrials, NBINS)
```
```python
                output_values=[IMAGE_VALUES,
                               ['no_change', 'change'],
                               ...
```

iii. The decoder-task instruction specifies "Image change, binary variable. Have value of 1 right after a change in image identity, otherwise 0." The AI's Step 5 mapping matches this literally, extending the 1 over the full 750 ms interval of the changed flash rather than a single bin so the label covers the evoked response.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: same `centers` array, same `pidx`, same `(ntrials, 24)` grid, stacked into the same output matrix as the neural trial arrays.

ii.
```python
    pidx = np.searchsorted(pres_start, centers.ravel(), side='right') - 1
    img_identity = ...reshape(ntrials, NBINS)
    img_change = pres_change[pidx].astype(np.int64).reshape(ntrials, NBINS)
```

iii. CONVERSION_NOTES Step 7 "Processing Plots Review": "bin edges line up with the flash onsets, the red change line sits exactly at a flash onset, the image_change output is 1 for exactly the three bins of the changed flash". Step 10 Check 2 confirms bins 12–14 on go trials and never on catch trials.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `BehaviorOphysExperiment.running_speed`, columns `timestamps` and `speed` (cm/s, 60 Hz, already low-pass/median filtered by the SDK's `running_processing`). Taken from the first plane of the session (behaviour streams are shared).

ii.
```python
    run = ref.running_speed
    run_t = run['timestamps'].values.astype(np.float64)
    run_v = interpolate_nans(run['speed'].values.astype(np.float64), run_t)
    if run_v is None:
        return None
```

iii. CONVERSION_NOTES Step 1: "`BehaviorSession.running_speed` … DataFrame `timestamps`, `speed` (cm/s), low-pass filtered + median filtered (`running_processing.py`), 60 Hz". Step 3: "**Running speed**: use the SDK `running_speed` (filtered) attribute." Step 4 verified the 60 Hz sampling (dt = 16.7 ms) against the whitepaper.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. (1) Any NaN samples are linearly interpolated over time (edges filled with the nearest valid value); a stream with fewer than 2 finite samples causes the session to be dropped. (2) The signal is averaged within each 250 ms bin using the same vectorised `bin_means` routine as the neural data, giving a continuous `(ntrials, 24)` array. (3) That array is discretised into quintiles (see 5-c). The continuous per-session min/max is retained in `metadata.session_info[i].running_speed_range` for diagnostics.

ii.
```python
def interpolate_nans(x, t):
    """Linearly interpolate NaNs of x over t (edges use nearest valid value)."""
    x = np.asarray(x, dtype=np.float64).copy()
    good = np.isfinite(x)
    if good.sum() < 2:
        return None
    x[~good] = np.interp(t[~good], t[good], x[good])
    return x
```
```python
    run_binned = bin_means(run_v, run_t, edges)[0][0]        # (ntrials, nbins)
    run_q = quantile_bin(run_binned)
```

iii. CONVERSION_NOTES Step 5 Variable Mapping: "`exp.running_speed` (`speed`, cm/s, 10 Hz-filtered) → `output[2]` 'running_speed' | mean within each 250 ms bin, then discretised into 5 equal-percentile (quintile) bins". Averaging (rather than point-sampling) is the natural counterpart of the neural bin-averaging and uses all ~15 running samples per bin. `metadata['alignment_note']` states "behaviour streams are averaged within the same 250 ms bins as the neural data".

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins (quintiles). The 20/40/60/80th percentiles are computed **within each session**, from that session's own binned running values, and applied with `np.digitize`. Labels 0–4, named `quintile_0 … quintile_4`. The resulting global distribution is [0.200, 0.200, 0.200, 0.200, 0.2001].

ii.
```python
def quantile_bin(values, nq=NQUANTILES):
    """Discretise into `nq` equal-percentile bins. Returns int labels 0..nq-1."""
    flat = values.ravel()
    edges = np.percentile(flat, np.linspace(0, 100, nq + 1)[1:-1])
    return np.digitize(values, edges, right=False).astype(np.int64)
```
```python
    run_q = quantile_bin(run_binned)
```
```python
        discretization=('running speed and pupil diameter are averaged in each bin and then '
                        'discretised into 5 equal-percentile (quintile) bins computed within '
                        'each session'),
```

iii. CONVERSION_NOTES Step 5 Key Decision 7: "**Quintiles computed per session** for running speed and pupil diameter: pupil area is in camera pixels and depends on the rig/geometry of each session, and running propensity varies greatly between mice/sessions; per-session percentile bins make the variable a well-defined within-session quantity (exactly 20% of bins per class in every session) and avoid the decoder simply reading out session identity."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is binned onto the very same `edges` grid as the neural data (the same function call signature, only the timestamps differ), so the alignment is exact by construction — no interpolation onto a separate timebase and no resampling step that could introduce a shift.

ii.
```python
    run_binned = bin_means(run_v, run_t, edges)[0][0]        # (ntrials, nbins)
```
```python
        alignment_note=('all data streams (2P frames, running, eye tracking, stimulus, '
                        'trials) are on the common sync clock exposed by the AllenSDK; '
                        'behaviour streams are averaged within the same 250 ms bins as the '
                        'neural data'),
```

iii. CONVERSION_NOTES Step 1: "everything (ophys frames, running, eye tracking, stimulus, trials, licks, rewards) is expressed in the same sync-clock seconds". Step 10 Check 2 verified it exactly: "**Output running speed** | recompute binned means for **all** trials and re-derive quintiles | exact label match, 0/4512, 0/6720, 0/3312 mismatched bins". Step 7 plot review: "binned running/pupil traces follow the raw 60 Hz traces, and the quintile traces track the binned values."

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `BehaviorOphysExperiment.eye_tracking`, columns `timestamps` and **`pupil_area`**, converted to an equivalent circular diameter `d = 2·sqrt(area/π)` (pixels). Blink handling is implicit: the SDK's `filter_on_blinks` already writes NaN into `pupil_area` wherever `likely_blink` is True, and those NaNs are interpolated. The blink/NaN fraction per session is recorded as `frac_blink`. Sessions whose eye-tracking table is missing or empty are dropped from the dataset (3 sessions).

ii.
```python
    eye = ref.eye_tracking
    if eye is None or len(eye) == 0:
        return None
    eye_t = eye['timestamps'].values.astype(np.float64)
    # pupil_area = pi * semi_major * semi_minor -> equivalent circular diameter
    pupil_d = 2.0 * np.sqrt(eye['pupil_area'].values.astype(np.float64) / np.pi)
    frac_blink = float(np.mean(~np.isfinite(pupil_d)))
    pupil_d = interpolate_nans(pupil_d, eye_t)
    if pupil_d is None:
        return None
```

iii. CONVERSION_NOTES Step 1: "`BehaviorSession.eye_tracking` … Ellipse fits at ~60 Hz: `pupil_area`, `pupil_width`, `pupil_height`, `pupil_area_raw`, `likely_blink`. Area/width/height are **NaN where `likely_blink == True`**." Step 5 mapping: "`exp.eye_tracking` `pupil_area` → diameter `2*sqrt(area/pi)` (px) … area = pi*w*h verified against `pupil_width`/`pupil_height`." Step 4 measured "Blink/NaN pupil frames … 1.4–2.9% of frames | consistent". Step 5 Key Decision 9: "sessions with no eye-tracking table are dropped from the dataset (a `pupil_diameter` output could not be defined); short blink gaps are interpolated."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Identical pipeline to running speed: area → equivalent diameter, linear interpolation across NaN (blink) samples, bin-averaging onto the shared 250 ms grid, then quintile discretisation. Continuous per-session range kept in `metadata.session_info[i].pupil_diameter_range`.

ii.
```python
    pupil_d = 2.0 * np.sqrt(eye['pupil_area'].values.astype(np.float64) / np.pi)
    pupil_d = interpolate_nans(pupil_d, eye_t)
    ...
    pupil_binned = bin_means(pupil_d, eye_t, edges)[0][0]
    pupil_q = quantile_bin(pupil_binned)
```

iii. CONVERSION_NOTES Step 5 mapping: "blink frames (NaN) linearly interpolated over time, mean within each bin, then 5 equal-percentile bins". Using the ellipse *area* rather than a single axis is justified in the notes as the SDK's primary pupil measure, with `area = π·w·h` verified against the width/height columns; converting to an equivalent diameter keeps the units interpretable as a diameter as the instruction asks.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five equal-percentile bins computed **per session** from that session's binned pupil values, using the same `quantile_bin` helper as running speed. Labels 0–4, `quintile_0 … quintile_4`; measured global distribution [0.200, 0.200, 0.200, 0.200, 0.2001].

ii.
```python
    pupil_q = quantile_bin(pupil_binned)
```
```python
def quantile_bin(values, nq=NQUANTILES):
    flat = values.ravel()
    edges = np.percentile(flat, np.linspace(0, 100, nq + 1)[1:-1])
    return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. CONVERSION_NOTES Step 5 Key Decision 7 (quoted in 5-c) gives pupil as the primary motivation: "pupil area is in camera pixels and depends on the rig/geometry of each session … per-session percentile bins make the variable a well-defined within-session quantity (exactly 20% of bins per class in every session) and avoid the decoder simply reading out session identity."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Bin-averaged onto the identical `edges` grid as the neural data, using each stream's own timestamps on the shared sync clock. No independent resampling stage.

ii.
```python
    pupil_binned = bin_means(pupil_d, eye_t, edges)[0][0]
```

iii. Same justification as running speed: `metadata['alignment_note']` — all streams are on the AllenSDK sync clock and all are averaged within the same 250 ms bins. Step 10 Check 2 verified exactness: "**Output pupil diameter** | same, with `2*sqrt(pupil_area/pi)` and NaN interpolation | exact label match, 0 mismatched bins."

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, mapped to codes 0/1/2/3 respectively (`output_values[4] = ['hit', 'miss', 'false_alarm', 'correct_reject']`).

ii.
```python
OUTCOME_VALUES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
    outcome = np.full(ntrials, -1, dtype=np.int64)
    outcome[tr['hit'].astype(bool).values] = 0
    outcome[tr['miss'].astype(bool).values] = 1
    outcome[tr['false_alarm'].astype(bool).values] = 2
    outcome[tr['correct_reject'].astype(bool).values] = 3
```

iii. CONVERSION_NOTES Step 1: "`Trial._get_trial_data()` … outcomes hit/miss/false_alarm/correct_reject (auto_rewarded trials have all four False)". Step 3: "Outcomes: HIT / MISS (go), FALSE ALARM / CORRECT REJECT (catch)." The AI also uses this as a structural sanity check: hit+miss must equal the number of go trials and FA+CR the number of catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The per-trial integer code is **broadcast across all 24 bins** so that the output is time-varying in shape (the instruction asks for time-varying outputs "if at all possible"), while being constant within a trial. Trials whose outcome remained −1 (no flag set) are dropped rather than being given a sentinel class; 0 such trials occurred in the full run. Final distribution: hit 0.317, miss 0.558, false alarm 0.019, correct reject 0.106 (13,940 / 24,520 / 834 / 4,681 trials).

ii.
```python
    ok = outcome >= 0
    n_dropped_outcome = int((~ok).sum())
    if n_dropped_outcome:
        ...
        outcome = outcome[ok]
        ...
    outcome_bins = np.repeat(outcome[:, None], NBINS, axis=1)
```
```python
    output_list = [np.stack([img_identity[i], img_change[i], run_q[i], pupil_q[i],
                             outcome_bins[i]], axis=0) for i in range(ntrials)]
```

iii. CONVERSION_NOTES Step 5 mapping: "`trials.hit/miss/false_alarm/correct_reject` → `output[4]` 'trial_outcome' | static per trial, broadcast over the 24 bins | `Trial._get_trial_data` | 4 categories". Step 10 Check 2 verified "Trial outcome | compare the whole sequence with the trials table, and constancy within the trial | True", and the conversion log asserts "hit+miss == go: True; fa+cr == catch: True". Step 12 Check 3 discusses the consequence of the broadcast: "all 24 bins of a trial share the same value, so the effective sample size is the number of trials rather than the number of bins", which explains the 1.62 train/val ratio for this output.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Seven distinct cases, all documented and counted:
- **Unloadable plane**: `get_behavior_ophys_experiment` failures are caught per plane and reported; a session with no loadable plane returns `None`. Whole-session failures are caught in `_worker` with a traceback and the session is marked DROPPED rather than crashing the run.
- **Empty plane** (`len(tbl) == 0`) is skipped.
- **Missing eye tracking** (`eye is None or len(eye) == 0`) → session dropped. 3 sessions (795625712, 805989030, 832881662) were dropped this way and verified individually.
- **Blink / NaN pupil samples and NaN running samples** → linearly interpolated over time (edge NaNs take the nearest valid value); `frac_blink` recorded per session. If fewer than 2 finite samples exist, the session is dropped.
- **Trials whose ±3 s window exceeds any stream's coverage** → dropped and counted (`n_trials_dropped_window`; 0 in the full run).
- **Trials with no outcome flag** → dropped and counted (`n_trials_dropped_outcome`; 0 in the full run).
- **Bins containing no sample** → set to 0 and counted (`n_empty_neural_bins`; 0 in the full run).
- **Non-finite `change_time`** → trial dropped. **Sessions with < 2 usable trials** → dropped.

ii.
```python
    for eid in experiment_ids:
        try:
            exps.append((eid, cache.get_behavior_ophys_experiment(eid)))
        except Exception as e:
            print(f'  [session {session_id}] could not load experiment {eid}: {e}')
    if not exps:
        return None
```
```python
    eye = ref.eye_tracking
    if eye is None or len(eye) == 0:
        return None
```
```python
    out[np.broadcast_to(counts[None, :, :] == 0, out.shape)] = 0.0
```
```python
def _worker(args):
    session_id, exp_ids, neural_source, show, path, zscore = args
    try:
        return session_id, process_session(session_id, exp_ids, neural_source, show, path, zscore)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'  [session {session_id}] FAILED: {e}')
        return session_id, None
```

iii. CONVERSION_NOTES Step 10 Check 5 "Edge cases" enumerates every one of these with its measured count, and Step 9 documents the 3 dropped sessions: "all three have an **empty eye-tracking table** (`len(exp.eye_tracking) == 0`), so the pupil-diameter output cannot be defined. They were verified individually (`/app/cache/check_dropped.py`)." Step 5 Key Decision 9 gives the rationale for dropping rather than imputing: a `pupil_diameter` output could not be defined at all for those sessions. Interpolating short blink gaps rather than dropping them is justified as preserving the continuity of a slow signal; the counters exist so that the user can confirm none of the fallbacks were ever exercised in practice.

## 9-a. What are the most time-consuming steps of the code?

i. Loading the NWB data through `cache.get_behavior_ophys_experiment(eid)` dominates — it is I/O bound and scales with the number of imaging planes in the session. The conversion log shows single-plane sessions at ~3.4–4 s and 7-plane Multiscope sessions at ~21 s, i.e. roughly 3 s per experiment; `load_time_s` is recorded separately from `total_time_s` in each session's `info` for exactly this diagnosis. Everything after loading (binning, discretisation, output construction) is vectorised and negligible. Total: 174 sessions in 52.5 s wall clock with 24 worker processes.

ii.
```python
    t0 = time.time()
    cache = get_cache()
    ...
    t_load = time.time() - t0
    ...
    info = dict(
        ...
        load_time_s=t_load, total_time_s=time.time() - t0,
    )
```
```python
            print(f'[{n}/{len(jobs)}] session {sid}: {i["n_neurons"]} neurons, '
                  f'{i["n_trials"]} trials (go {i["n_go"]}, catch {i["n_catch"]}), '
                  f'{i["total_time_s"]:.1f}s '
                  f'[{n / (time.time() - t0):.2f} sessions/s]', flush=True)
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: naive per-trial slicing of the 140k-frame traces, and re-opening the NWB file per plane. Code speedups added: vectorised cumsum/searchsorted binning of all neurons × trials at once; one cache/NWB read per experiment; multiprocessing over sessions." Step 7 Run Time Estimates: single-plane 3.6 s/session, Multiscope 21 s/session, "total (24 workers) … **~1–3 min wall clock** (well under 15 min)" — the actual full run took 55 s.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorised the loop that mattered: binning is done for all neurons × all trials × all bins in one `cumsum` + `searchsorted` (`bin_means`), replacing a per-trial/per-bin Python loop. Remaining Python-level loops, none of which are on the critical path:
- the per-plane loop in `process_session` (irreducible — one NWB read per plane);
- `[IMAGE_TO_IDX[n] for n in img_names]`, a per-bin dict lookup over ~1M entries, which could be a `np.unique`/`searchsorted` map or a `pd.Series.map`;
- the two per-trial list comprehensions building `neural_list` and `output_list` (these mostly just create views/copies required by the output format);
- in `main`, `regions.index(r)` is called once per neuron inside a list comprehension, and `r not in regions` once per neuron — an O(n_neurons × n_regions) list scan that should be a dict lookup (harmless here because there are only 2 regions and 29k neurons, but it is the one genuinely gratuitous quadratic pattern);
- `subjects.index(mouse)` per session, same pattern at trivial scale.

ii.
```python
        for r in res['region_names']:
            if r not in regions:
                regions.append(r)
        data['brain_region_idx'].append(
            np.array([regions.index(r) for r in res['region_names']], dtype=np.int64))
```
```python
    img_identity = np.array([IMAGE_TO_IDX[n] for n in img_names],
                            dtype=np.int64).reshape(ntrials, NBINS)
```
```python
    neural_list = [np.ascontiguousarray(neural[:, i, :]) for i in range(ntrials)]
    output_list = [np.stack([img_identity[i], img_change[i], run_q[i], pupil_q[i],
                             outcome_bins[i]], axis=0) for i in range(ntrials)]
```

iii. CONVERSION_NOTES Step 6: "`bin_means()` bins any (n_signals, T) stream onto the (ntrials, nbins+1) edge grid with a single cumsum + searchsorted → O(T) instead of a Python loop over trials/bins", credited with "~10× vs. per-trial slicing". The AI's own notes do not identify the residual loops listed above; they were not pursued because loading dominates (Step 7 timing table) and the 15-minute budget was met by a factor of ~15.

## 9-c. What processing does the code repeat multiple times?

i. Little is repeated, and the AI explicitly designed against the obvious repeats ("one NWB read per experiment"). What does repeat:
- `get_cache()` is called once inside every `process_session` call, i.e. 174 times, re-instantiating the `VisualBehaviorOphysProjectCache` and re-reading the project manifest in each worker for each session (it could be created once per worker process).
- `downloaded_experiment_table()` globs the 284-file NWB directory once in the parent — cheap, but it duplicates information already in the experiment table.
- In `--show-processing` mode, `make_processing_plot` re-reads and re-`vstack`s plane 0's full trace matrix that `process_session` already loaded and binned.
- `bin_means` is called three times per session (neural, running, pupil), each recomputing `searchsorted` on the same `edges` array against a different timebase — necessary, since the timestamps differ.

ii.
```python
def process_session(session_id, experiment_ids, ...):
    t0 = time.time()
    cache = get_cache()            # re-created for every session
```
```python
    tbl = e0.events if neural_source in ('events', 'filtered_events') else e0.dff_traces
    col = {'events': 'events', 'filtered_events': 'filtered_events', 'dff': 'dff'}[neural_source]
    traces0 = np.vstack(tbl[col].values)      # already loaded/binned in process_session
```

iii. CONVERSION_NOTES Step 6 claims the repeats that mattered were removed: "one cache/NWB read per experiment; multiprocessing over sessions". The per-session `get_cache()` and the plotting re-read are not mentioned in the notes; both are small relative to the ~3 s/plane NWB read (manifest parsing is cached on disk, and plotting runs for at most 2 sessions).

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A small amount of diagnostic/optional work that never reaches the output arrays:
- **`cell_ids`** is accumulated for every neuron of every plane and returned from `process_session`, but `main` never reads `res['cell_ids']` — it is pure dead work and dead memory.
- The continuous `run_binned` / `pupil_binned` arrays survive only to produce `running_speed_range` / `pupil_diameter_range` in `info` (and the plots); only the quintised versions are saved.
- `frac_blink`, `n_empty_neural_bins`, `zero_bins`, `load_time_s`, and the full per-session `info` dict are computed and stored in `metadata['session_info']` for 171 sessions; useful as provenance, unused by the decoder.
- The `--zscore` branch and the `events`/`filtered_events` branches exist but are not exercised in the delivered run.
- `centers` is computed for all trials even when `--show-processing` is off (it is needed for the image outputs, so only the plotting-specific parts are extra).

ii.
```python
        cell_ids += list(tbl.index.values)
    ...
    return dict(neural=neural_list, output=output_list, region_names=region_names,
                cell_ids=cell_ids, info=info)
```
```python
        running_speed_range=[float(run_binned.min()), float(run_binned.max())],
        pupil_diameter_range=[float(pupil_binned.min()), float(pupil_binned.max())],
        load_time_s=t_load, total_time_s=time.time() - t0,
```

iii. The AI does not flag any of this in CONVERSION_NOTES; the diagnostics are deliberate and are used throughout Steps 9/10/12 to justify the conversion (`n_trials_dropped_window`, `n_empty_neural_bins`, `frac_blink` are all quoted in the Step 10 edge-case audit), and `--neural`/`--zscore` were retained on purpose so the paper's event-based choice can be reproduced ("`--neural events` / `--neural filtered_events` remain available in the script for reproducing the paper choice"). Only `cell_ids` appears to be genuinely vestigial.
