# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read through the AllenSDK `VisualBehaviorOphysProjectCache`, opened on the **local**
cache directory `/app/data` (`from_local_cache(..., use_static_cache=False)`); no `.nwb` file is ever
opened directly with `h5py`/`pynwb`. The set of candidate experiments is obtained by intersecting
`cache.get_ophys_experiment_table()` with the NWB files that actually exist on disk
(`locally_available_experiment_ids()`, 284 files). Two selection filters are then applied:
`project_code == 'VisualBehavior'` (the single-plane Scientifica variant; excludes the 45
`VisualBehaviorMultiscope` experiments) and `passive == False` (excludes the 71 passive
OPHYS_2/OPHYS_5 replay sessions). This yields 168 experiments = 168 sessions from 37 mice. Each
selected experiment is then loaded with `cache.get_behavior_ophys_experiment(eid)` inside a
`multiprocessing.Pool` (24–32 workers), and each worker opens its own cache handle. Three sessions
fail at load time (empty `eye_tracking` table) and are reported and skipped, leaving 165 sessions /
28 821 neurons / 42 470 trials.

ii.
```python
def get_cache():
    """Open the local AllenSDK cache (no network access)."""
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    return VisualBehaviorOphysProjectCache.from_local_cache(
        cache_dir=CACHE_DIR, use_static_cache=False)


def locally_available_experiment_ids():
    """ophys_experiment_ids whose NWB file is present in the local cache."""
    files = os.listdir(NWB_DIR)
    return sorted(int(re.findall(r'(\d+)', f)[0]) for f in files
                  if f.endswith('.nwb'))


def select_experiments(cache):
    et = cache.get_ophys_experiment_table()
    sub = et.loc[locally_available_experiment_ids()]
    # D1: single-plane "VisualBehavior" project variant only ...
    # D2: active behaviour only ...
    sel = sub[(~sub.passive) & (sub.project_code == 'VisualBehavior')]
    return sel.sort_index()
```
```python
def convert_experiment(ophys_experiment_id, collect_debug=False):
    cache = get_cache()
    ds = cache.get_behavior_ophys_experiment(ophys_experiment_id)
    ts = np.asarray(ds.ophys_timestamps, dtype=float)
...
    with Pool(min(args.workers, max(1, len(jobs)))) as pool:
        for i, res in enumerate(pool.imap(_worker, jobs, chunksize=1)):
```

iii. From CONVERSION_NOTES Step 5: **D1** — the target format requires "time bins ... the same size for
all trials and sessions" and the task requires alignment "based on ophys timestamp". All single-plane
rigs sample at 30.95 Hz (dt = 32.310–32.330 ms across all 168 sessions), so the native ophys grid is
already a constant bin; mixing in the Multiscope sessions would mix 30.95 Hz with 10.7 Hz data, and the
locally available Multiscope data is only 6 sessions from one mouse (347 cells, 1.2 %).
`VisualBehavior` is also the literal name of this dataset variant in the SDK. **D2** — passive sessions
replay the stimulus with the lick spout retracted to a sated mouse, so there is no Go/Catch/Aborted/
Auto-rewarded response structure and no hit/miss/false-alarm/correct-rejection outcome to decode; they
fall outside "the Visual Behavior task". **D10** — sessions with an empty `eye_tracking` table are
dropped rather than filled with fabricated pupil values (cost: 3 sessions, 917 trials, 276 cells).

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values taken from each loaded experiment's
`BehaviorOphysExperiment.metadata`. The `subjects` list is built in order of first appearance over
the (experiment-id-sorted) session list, and `subject_idx[session]` is the index of that session's
mouse in the list. Result: 37 mice, 1–9 sessions each.

ii.
```python
    info = {... 'mouse_id': str(md['mouse_id']), ...}
    result = {... 'mouse_id': str(md['mouse_id']), ...}
...
    subjects = []
    for r in ok:
        if r['mouse_id'] not in subjects:
            subjects.append(r['mouse_id'])
    data = {...
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['mouse_id']) for r in ok],
                                dtype=np.int64),
    }
```

iii. `mouse_id` is the SDK's unique animal identifier. The notes record 37 mice in the selected subset
and the verification log prints the per-subject session counts (1–9, mean 4.5), which the notes
cross-check against the whitepaper's description of 3–11 sessions per imaging container.

## 1-c. How are the data split into sessions?

i. One converted session = one `ophys_experiment_id`. The AI verified that for the `VisualBehavior`
(single-plane) project code there is exactly one imaging plane per session, so experiment and session
are 1:1 (239 experiments ↔ 239 `ophys_session_id`s locally; 168 after the passive filter). Sessions are
processed in ascending experiment-id order; `ophys_session_id` and `behavior_session_id` are recorded in
`metadata['session_info']` for traceability. No merging of planes is performed because none is needed.

ii.
```python
    sel = sub[(~sub.passive) & (sub.project_code == 'VisualBehavior')]
    return sel.sort_index()
...
    jobs = [(eid, i < n_debug) for i, eid in enumerate(eids)]
...
    info = {
        'ophys_experiment_id': int(ophys_experiment_id),
        'ophys_session_id': int(md['ophys_session_id']),
        'behavior_session_id': int(md['behavior_session_id']),
        ...
    }
```

iii. CONVERSION_NOTES Step 2/Step 5-D1: "one imaging plane (= one experiment) per session" for the
single-plane CAM2P rigs; the hierarchy mouse → container → session → experiment → cells is documented in
Step 2, and the multi-plane variant (where one session contains several experiments that would have to
be merged) is excluded by D1.

## 1-d. How are the data split into trials?

i. Trials come from the SDK's own `BehaviorOphysExperiment.trials` table. The rows kept are those with
`go == True` or `catch == True`, sorted by `start_time`. Each trial owns the half-open window of ophys
frames `[trials.start_time, trials.stop_time)`, located with `np.searchsorted(..., side='left')`, so T
is variable (217–389 frames, mean 262, ≈ 7.0–12.6 s) and adjacent trials can never share a frame. All
five output variables and the neural matrix are sliced with the same `[a, b)` index range.

ii.
```python
    trials = ds.trials
    # `go`, `catch`, `aborted` and `auto_rewarded` are mutually exclusive in the SDK ...
    sel = trials[trials['go'].to_numpy().astype(bool)
                 | trials['catch'].to_numpy().astype(bool)].sort_values('start_time')
...
    # A trial owns every ophys frame with start_time <= t < stop_time.
    a_idx = np.searchsorted(ts, sel['start_time'].to_numpy(dtype=float), side='left')
    b_idx = np.searchsorted(ts, sel['stop_time'].to_numpy(dtype=float), side='left')
...
    for i, (a, b) in enumerate(zip(a_idx, b_idx)):
        T = b - a
        neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. The instruction is "Segment each recording session into individual trials based on how they are
defined in the experiment. Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and
'Auto-rewarded' trials" (D4). The AI read `allensdk/.../trials/trial.py` L182-213 and documented that
`go`, `catch`, `aborted` and `auto_rewarded` are mutually exclusive, so `go | catch` is *exactly* the
requested set; it verified empirically that `go+catch+aborted+auto_rewarded == n_trials` in all 168
sessions. The full `[start_time, stop_time)` window (rather than a fixed window around `change_time`)
is used because the alignment event is the ophys timestamp grid and the format permits variable T;
`off_start = 0.0`, `off_end = None`.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level filters: (1) keep only `go | catch` (drops aborted and auto-rewarded trials — ~418
aborted and ~3.8 auto-rewarded per session); (2) drop any trial whose ophys window contains fewer than
2 frames (`b_idx - a_idx >= 2`), counted in `info['n_trials_dropped_short']` — this never fires (0 in
all 165 sessions); (3) an assertion that every kept trial carries exactly one of
hit/miss/false_alarm/correct_reject, otherwise the session is aborted with an error. Session-level
filters that indirectly remove trials: passive sessions and sessions with no eye tracking (3 sessions,
917 trials). No further per-trial quality metric is applied — sessions with as few as 39 trials or 6
neurons are kept, since the format only requires ≥ 2 trials.

ii.
```python
    sel = trials[trials['go'].to_numpy().astype(bool)
                 | trials['catch'].to_numpy().astype(bool)].sort_values('start_time')
    if len(sel) == 0:
        raise ValueError(f'{ophys_experiment_id}: no go/catch trials')

    outcome = np.full(len(sel), -1, dtype=np.int64)
    for j, name in enumerate(OUTCOME_NAMES):
        outcome[sel[name].to_numpy().astype(bool)] = j
    if np.any(outcome < 0):
        raise ValueError(f'{ophys_experiment_id}: unclassified go/catch trial')
...
    keep = (b_idx - a_idx) >= 2
    n_dropped_short = int((~keep).sum())
    a_idx, b_idx = a_idx[keep], b_idx[keep]
    outcome = outcome[keep]
```

iii. CONVERSION_NOTES Step 3 "Trial curation rules": keep `go|catch`, drop `aborted` and
`auto_rewarded` (explicit task instruction + the SDK's mutually exclusive trial classes); drop trials
containing no ophys frame or with missing eye/running coverage. Step 10 Check 5 documents the edge cases
checked: trial windows never overlap (`start_time[i+1] >= stop_time[i]` for all trials), off-by-one
bracketing of `start_time`/`stop_time` verified with `np.allclose`-style assertions, no trial shorter
than 2 frames, and sessions where one outcome class never occurs are kept because "class coverage is
global, not per-session".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `BehaviorOphysExperiment.dff_traces['dff']` — the Allen pipeline's detrended ΔF/F traces, one row
per valid cell ROI, sampled on `ophys_timestamps`. The SDK's `events` / `filtered_events` (the signal
used by the Neuron reference paper) were explicitly considered and rejected (decision D3).

ii.
```python
    dff = ds.dff_traces
    cell_ids = dff.index.to_numpy()
    traces = np.vstack(dff['dff'].to_numpy()).astype(np.float32)   # (N, F)
    if traces.shape[1] != n_frames:
        raise ValueError(f'{ophys_experiment_id}: dff has {traces.shape[1]} samples '
                         f'but there are {n_frames} ophys timestamps')
```

iii. D3: the paper uses detected calcium events, and that was the AI's first choice, but at 30.95 Hz the
event train is non-zero for only **0.23 %** of (cell, frame) pairs, so 99.77 % of the samples the
per-timepoint decoder must label carry no signal. A controlled pilot on 6 sessions with identical
decoder settings gave validation balanced accuracies of 0.121/0.549/0.209/0.214/0.248 for `events`,
0.169/0.578/0.222/0.246/0.251 for `filtered_events` and 0.386/0.658/0.269/0.252/0.249 for `dff_traces`.
dF/F comes from the same Allen pipeline (`methods.txt` §DF/F CALCULATION), is the stream used in the
SDK tutorials, and is defined at every ophys frame; the instructions permit departing from the reference
processing where "training a neural decoder requires otherwise".

## 2-b. How is the `neural` data processed?

i. Essentially none beyond stacking: the per-cell dF/F arrays are stacked into an `(N_neurons, F)`
`float32` matrix, the ROI set is intersected with `cell_specimen_table.valid_roi`, and per-trial
contiguous slices `traces[:, a:b]` are taken. No normalisation, z-scoring, smoothing, baselining or
rebinning is applied. Brain region is taken from `metadata['targeted_structure']` (VISp for all 165
sessions) and broadcast to every neuron in `brain_region_idx`.

ii.
```python
    traces = np.vstack(dff['dff'].to_numpy()).astype(np.float32)   # (N, F)
...
    cst = ds.cell_specimen_table
    if not bool(cst['valid_roi'].all()):
        keep = cst['valid_roi'].to_numpy().astype(bool)
        traces = traces[keep]
        cell_ids = cell_ids[keep]
        n_neurons = traces.shape[0]
...
        neural.append(np.ascontiguousarray(traces[:, a:b]))
...
        'brain_region_idx': [
            np.full(r['n_neurons'], brain_regions.index(r['brain_region']),
                    dtype=np.int64) for r in ok],
```

iii. The release already ships fully processed dF/F (dewarping → motion correction → segmentation → ROI
filtering → demixing → neuropil subtraction → dF/F → detrending), so no further processing is
warranted. Step 12 explicitly tested two alternatives on 20 random sessions — per-neuron z-scoring and
per-session global rescaling — and found the gains small and inconsistent (z-scoring *hurt*
`trial_outcome` by 0.056), so "the unmodified pipeline dF/F was kept — it is the more faithful
representation of the reference processing".

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level filtering. The converter defensively re-applies the
`cell_specimen_table.valid_roi` mask, but the AI verified that `valid_roi` is `True` for 29 097/29 097
cells in the selected sessions, so the mask is a no-op. Neuron count per session ranges 6–666 (mean
174.7); low-cell-count sessions are kept.

ii.
```python
    cst = ds.cell_specimen_table
    if not bool(cst['valid_roi'].all()):
        keep = cst['valid_roi'].to_numpy().astype(bool)
        traces = traces[keep]
```

iii. D11 / Step 1 notes: the release already applied the ROI filters described in `methods.txt`
§ROI FILTERING (union ROIs, duplicates, motion-border ROIs, apical dendrites, too small/narrow/dim
ROIs, non-positive demixed traces) plus session- and container-level QC (saturation, >20 %
photobleaching, >10 µm z-drift, peak d′ < 1, interictal events). "No additional neuron filtering is
required, and the whitepaper describes no further criterion that is exposed in the NWB files."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instruction is "Temporally align based on ophys timestamp". Every stream is placed on the native
`ophys_timestamps` grid of its own session: dF/F is used as recorded (no resampling), the 60 Hz running
trace and ~30 Hz pupil trace are linearly interpolated onto `ts`, and stimulus/trial variables are
evaluated at those same frame times. A trial is the half-open frame range
`[searchsorted(ts, start_time), searchsorted(ts, stop_time))`, so the neural matrix and all five output
rows share one index range by construction. `metadata['temporal_alignment_event']` describes the ophys
frame grid; `off_start = 0.0` (trial start), `off_end = None` (variable trial length).

ii.
```python
    ts = np.asarray(ds.ophys_timestamps, dtype=float)      # (F,) session clock, s
...
    run_frames = np.interp(ts, run_t[run_ok], run_v[run_ok])
    pupil_frames = np.interp(ts, eye_t[eye_ok], eye_d[eye_ok])
    image_idx, is_change, interval = build_stimulus_series(ds.stimulus_presentations, ts)
...
    a_idx = np.searchsorted(ts, sel['start_time'].to_numpy(dtype=float), side='left')
    b_idx = np.searchsorted(ts, sel['stop_time'].to_numpy(dtype=float), side='left')
...
        neural.append(np.ascontiguousarray(traces[:, a:b]))
        out[0] = image_idx[a:b]; out[1] = is_change[a:b]
        out[2] = run_bin[a:b];   out[3] = pupil_bin[a:b]; out[4] = outcome[i]
```

iii. Step 3: all NWB streams are already on one common session clock (100 kHz sync board), so "no extra
alignment step is needed — only resampling of the 60 Hz running trace and the 30 Hz eye trace onto the
30.95 Hz ophys grid". The half-open window is chosen "so that adjacent trials cannot share a frame".
Alignment was verified three ways (Step 10 Check 2 and the `--show-processing` figure): off-by-one
bracketing assertions (`ts[a] >= start_time and ts[a-1] < start_time`; `ts[b-1] < stop_time and
ts[b] >= stop_time`), spot checks of `neural[s][trial][neuron, t]` against `dff_traces` at
independently derived frame indices, and a change-triggered average dF/F that is flat before t = 0 and
peaks ~0.3 s after it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **No rebinning.** The data stay on the native single-plane 2p frame grid: one time bin = one ophys
frame = 32.3193 ms on average (30.95 Hz), sd 0.0055 ms, range 32.310–32.330 ms across all 165 sessions.
`metadata['time_bin_size']` is the mean over sessions of each session's median inter-frame interval, and
the spread is also recorded (`time_bin_size_std_ms`, `time_bin_size_range_ms`). The requirement that
bins be equal across sessions is what motivated restricting to the single-plane project code (D1).

ii.
```python
    md_dt = float(np.median(np.diff(ts)))     # info['ophys_frame_interval_s']
...
    dts = np.array([r['info']['ophys_frame_interval_s'] for r in ok])
    data['metadata'] = {
        ...
        'time_bin_size': float(np.mean(dts) * 1000.0),
        'time_bin_size_std_ms': float(np.std(dts) * 1000.0),
        'time_bin_size_range_ms': [float(dts.min() * 1000.0),
                                   float(dts.max() * 1000.0)],
```

iii. Step 3/Step 10 Check 3(d): "the neural data live on the native 2p frame grid"; the paper's 750 ms
presentation interval and 400 ms post-flash window are *analysis* units, not a neural binning choice, so
they are used only as the unit for `image_identity` / `image_change`. Rebinning would destroy the
one-sample-per-bin correspondence and, given the whitepaper's 31 Hz nominal rate, buy nothing. The
sync-derived 30.95 Hz vs. the metadata's nominal 31.0 Hz discrepancy was checked and documented as
consistent (Step 4).

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `stimulus_presentations` (restricted to `stimulus_block_name == 'change_detection_behavior'`), using
the `start_time` and `image_name` columns, plus `ophys_timestamps`. Each ophys frame is mapped to the
presentation interval `[start_time_i, start_time_{i+1})` that contains it, and takes that flash's image
identity. Omitted flashes (`image_name == 'omitted'`) are forward-filled with the previous real image.
The trials table's `initial_image_name` / `change_image_name` are *not* used.

ii.
```python
    sp = stimulus_presentations
    sp = sp[sp['stimulus_block_name'] == STIM_BLOCK].sort_values('start_time')
    start = sp['start_time'].to_numpy(dtype=float)
    names = sp['image_name'].to_numpy()
    raw = np.array([IMAGE_TO_IDX.get(n, -1) if isinstance(n, str) else -1
                    for n in names], dtype=np.int64)
    # forward-fill the omitted (-1) entries with the previous image identity
    valid_pos = np.where(raw >= 0)[0]
    fill_src = valid_pos[np.searchsorted(valid_pos,
                                         np.arange(len(raw)), side='right') - 1]
    filled = raw[fill_src]
    k = np.searchsorted(start, ophys_timestamps, side='right') - 1
    k = np.clip(k, 0, len(start) - 1)
    return filled[k], change[k].astype(np.int64), k
```

iii. D5: intervals are defined as `[flash start_time, next flash start_time)` — "the same 750 ms
'image-presentation interval' the reference paper uses for behavioural events" ("By image presentation
interval we refer to the 750 ms interval beginning with each image presentation"). The identity is held
through the gray screen and through omissions; the alternatives considered and rejected were (a) a
separate "gray" category ("would make ~67 % of all samples a single class that is not an image identity
at all") and (b) a separate "omitted" category ("the *absence* of a stimulus rather than an identity").
Holding the current image "is the only option that keeps the variable to the 16 real image identities,
matching the instruction 'image identity of the image presented during the non-grey screen'".

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to a **global, hard-coded, alphabetically sorted list of the 16 natural images**
used by the single-plane variant (image set A ∪ image set B), so the same integer always means the same
image in every session. Each session contributes only its own 8 images. `output_values[0]` stores the 16
names. Values are stored as `int64` in row 0 of the `(5, T)` output array. The resulting marginal
distribution is 0.060–0.065 per image (uniform to ±4 %).

ii.
```python
IMAGE_NAMES = ['im000', 'im031', 'im035', 'im045', 'im054', 'im061', 'im062', 'im063',
               'im065', 'im066', 'im069', 'im073', 'im075', 'im077', 'im085', 'im106']
IMAGE_TO_IDX = {n: i for i, n in enumerate(IMAGE_NAMES)}
...
OUTPUT_VALUES = [IMAGE_NAMES, CHANGE_NAMES, QUINTILE_NAMES, QUINTILE_NAMES, OUTCOME_NAMES]
...
        out = np.empty((len(OUTPUT_NAMES), T), dtype=np.int64)
        out[0] = image_idx[a:b]
```

iii. D6: "The single-plane variant uses image set A (88 sessions) and image set B (80 sessions).
`output_values` is a single global list, so per-session indices 0–7 would give the same label to
different images in different sessions. Using the 16 real image names keeps the labels meaningful
(chance = 1/16)." The AI enumerated the images across all sessions before hard-coding the list
(trajectory step 42) and verified every session has exactly 8 distinct identities.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed for **every** ophys frame of the session on the same `ophys_timestamps` grid as the
neural data, then sliced with the identical `[a, b)` trial index range, so alignment is exact by
construction. Frames that precede the first flash of the change-detection block (at most one frame per
session) are clamped onto interval 0.

ii.
```python
    k = np.searchsorted(start, ophys_timestamps, side='right') - 1
    k = np.clip(k, 0, len(start) - 1)
...
        out[0] = image_idx[a:b]
        neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. Step 10 Check 2: `image_identity` was recomputed by an independent explicit python loop over
`stimulus_presentations` and compared (12 random spot checks per session × 5 sessions, 0 failures). Step
10 Check 5 item 1 documents the clamping fix: in session 792815735 the first go/catch trial starts
0.021 s before the first flash, which previously produced `image_identity = -1` and crashed the decoder;
the affected frame is now assigned the identity of the flash that is about to appear.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `stimulus_presentations['is_change']` (within the `change_detection_behavior` block), combined with
the same per-frame presentation-interval index `k`. The SDK sets `is_change` only for real image
changes; sham changes on catch trials are flagged `is_sham_change` and therefore produce no change
frames.

ii.
```python
    change = sp['is_change'].to_numpy().astype(bool)
...
    return filled[k], change[k].astype(np.int64), k
...
        out[1] = is_change[a:b]
```

iii. Step 5 mapping table: `stimulus_presentations.is_change` → `output[1] image_change`, "1 for every
frame of the presentation interval that begins with a change". Using the stimulus table rather than
`trials.change_time` means the change interval boundaries are read from the data instead of assumed
(Step 10 Check 5 item 2: inter-flash intervals are actually 733–801 ms, not exactly 750 ms, because the
stimulus computer occasionally drops a frame).

## 4-b. What processing is involved in computing `output` *Image change*?

i. None beyond the vectorised interval lookup: the boolean `is_change` per flash is gathered by frame
(`change[k]`) and cast to `int64`. Two categories, `['no_change', 'change']`. The resulting global
distribution is 0.9229 / 0.0771, consistent with 0.75 s × ~230 changes per 8.47 s × ~258 trials.

ii. See 4-a.

iii. N/A — no thresholding or smoothing is applied; the variable is already binary in the source.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary by construction: 1 for every ophys frame that falls inside the presentation interval
beginning with a change flash (i.e. the 250 ms changed image plus the following ~500 ms gray, ≈ 23
frames), 0 everywhere else — including all of every catch trial. The interval end is the *next* flash
onset read from the data, not a fixed 750 ms offset.

ii.
```python
    # interval i spans [start_time_i, start_time_{i+1})
    k = np.searchsorted(start, ophys_timestamps, side='right') - 1
    ...
    return filled[k], change[k].astype(np.int64), k
```

iii. D7: "'Value of 1 right after a change in image identity'. The change flash is 250 ms and the
calcium response to it peaks 200–500 ms later, so the unit is the 750 ms interval beginning at the
change (again the reference paper's unit). Catch (sham-change) trials therefore contain no change
frames — verified." Step 10 Check 2 verifies that every change-marked frame lies in
`[change_time, next flash onset)` and that every GO trial contains exactly one change interval while no
CATCH trial contains any.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identical mechanism to image identity: computed for every ophys frame of the session and sliced with
the same `[a, b)` range as the neural matrix.

ii.
```python
        out[1] = is_change[a:b]
```

iii. Same justification as 3-c; additionally, the change-triggered average dF/F (panel 9 of
`processing_*.png`) is flat before t = 0 and peaks ~0.3 s after, and the AI's re-implementation of the
paper's change-vs-repeat random-forest decoder on the converted data reaches 74.0 % ± 0.9 % (paper's
range ≈ 65–85 %), which it argues "a temporal misalignment of even one flash (750 ms) would collapse
towards 50 %".

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `BehaviorOphysExperiment.running_speed`, columns `timestamps` and `speed` — the SDK's 60 Hz linear
running speed in cm/s (already de-wrapped, transient-corrected and 10 Hz low-pass filtered by
`running_processing.py`). `raw_running_speed` is not used.

ii.
```python
    rs = ds.running_speed
    run_t = rs['timestamps'].to_numpy(dtype=float)
    run_v = rs['speed'].to_numpy(dtype=float)
    run_ok = np.isfinite(run_v)
```

iii. Step 1/Step 10 Check 3: the SDK's `running_speed` is the processed stream described in the
whitepaper ("analog encoder, converted to cm/s, 10 Hz low-pass filtered"); it is "used as provided …
no additional filtering". Observed range −24 … +100 cm/s matches the survey of the raw files.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Two steps: (1) linear resampling onto the ophys frame times with `np.interp` (non-finite samples
dropped first; `np.interp` clamps at the ends rather than producing NaN); (2) discretisation into five
equal-percentile bins. No smoothing, no rectification of negative speeds, no absolute value.

ii.
```python
    run_frames = np.interp(ts, run_t[run_ok], run_v[run_ok])
...
    run_edges = np.percentile(run_frames[frame_mask], QUANTILE_PCTS)
    run_bin = np.searchsorted(run_edges, run_frames, side='right').astype(np.int64)
...
        out[2] = run_bin[a:b]
```

iii. Step 3/Step 5: all streams are on the common sync clock, so only resampling onto the 30.95 Hz ophys
grid is needed; the decoder output spec requires "running speed, discretized into five equal percentile
bins".

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Into quintiles, with the 20/40/60/80th-percentile edges computed **per session**, and over exactly
the frames that end up in the converted data (the union of the kept trial windows), not the whole
session. `np.searchsorted(edges, x, side='right')` yields bins 0–4. Each session therefore contributes
exactly 20.0 % of its samples to each bin (max deviation from 0.200 over all 165 sessions: 6.3e-5). The
per-session edges are stored in `metadata['session_info'][i]['running_speed_quintile_edges']`.

ii.
```python
    # Quintile edges are computed per session over exactly the frames that end up in
    # the converted data, so each session contributes ~20 % of its samples to each bin.
    frame_mask = np.zeros(n_frames, dtype=bool)
    for a, b in zip(a_idx, b_idx):
        frame_mask[a:b] = True
    run_edges = np.percentile(run_frames[frame_mask], QUANTILE_PCTS)
    pupil_edges = np.percentile(pupil_frames[frame_mask], QUANTILE_PCTS)
```

iii. D8: the primary argument is about pupil diameter — it is measured in camera pixels and its
across-session spread (sd of session medians 18.4 px, range 48–166 px) exceeds the typical within-session
inter-quintile spread (~15 px), so "global quintiles would mostly encode which session a sample came
from — a per-session constant that the session-specific projection of the decoder can read off
trivially, inflating accuracy without decoding anything about the pupil". Per-session edges also give
every session an exactly balanced 5-class problem, "which is what the balanced-accuracy metric assumes".
"The same convention is used for running speed so that the two discretisations are comparable." The
acknowledged cost: in 18/165 sessions the mouse essentially never ran (80th percentile < 1 cm/s) so its
running bins are close to sensor noise — the AI argues global edges "would not help those sessions
either; they would simply collapse them into one class".

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated onto the full-session `ophys_timestamps` vector before trial segmentation, binned on the
same vector, then sliced with the identical `[a, b)` trial index range as the neural matrix.

ii.
```python
    run_frames = np.interp(ts, run_t[run_ok], run_v[run_ok])
    run_bin = np.searchsorted(run_edges, run_frames, side='right').astype(np.int64)
...
        out[2] = run_bin[a:b]
        neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. All timestamps come from the same 100 kHz sync board, so linear interpolation from 60 Hz onto
30.95 Hz is a pure resampling. Panel 2 of `processing_*.png` overlays the raw 60 Hz trace and the
resampled trace to show there is no lag; Step 10 Check 2 recomputes the quintile edges and spot-checks
10 `running_speed_bin` values per session from the raw `running_speed` table (all PASS).

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `BehaviorOphysExperiment.eye_tracking`, columns `pupil_width`, `pupil_height` and `timestamps`
(~30 Hz DeepLabCut ellipse fits). Pupil **diameter** is defined as `2 · max(pupil_width, pupil_height)`.
Frames flagged `likely_blink` by the SDK are NaN in these columns and are dropped.

ii.
```python
def pupil_diameter_series(eye_tracking):
    """... AllenSDK computes `pupil_area = pi * max(pupil_width, pupil_height)**2`
    (`eye_tracking_processing.compute_circular_area`), i.e. it treats
    `max(width, height)` as the pupil *radius*.  The matching diameter is therefore
    `2 * max(width, height)`. ..."""
    diam = 2.0 * np.maximum(eye_tracking['pupil_width'].to_numpy(dtype=float),
                            eye_tracking['pupil_height'].to_numpy(dtype=float))
    t = eye_tracking['timestamps'].to_numpy(dtype=float)
    good = np.isfinite(diam)
    return t, diam, good
```

iii. Step 1/Step 4: the AI read `eye_tracking_processing.compute_circular_area` (L80-100), which
computes `π · max(w,h)²` — i.e. the SDK treats `max(width, height)` as the *radius* — and concluded
"pupil diameter = 2·max(width,height) = 2·√(area/π)". The whitepaper does not itself define a
"diameter", so the SDK's own geometric convention was adopted.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) Blink/outlier frames removed (they are already NaN in the SDK output; `np.maximum` propagates the
NaN and `np.isfinite` drops them) — 3.7 % of frames on average, up to 29.6 % in one session; (2) linear
interpolation across the removed frames and onto the ophys grid in a single `np.interp` call;
(3) quintile discretisation. Sessions with an empty eye-tracking table, or with fewer than 100 valid
pupil frames, raise and are skipped.

ii.
```python
    eye = ds.eye_tracking
    if eye is None or len(eye) == 0:
        raise ValueError(f'{ophys_experiment_id}: no eye tracking data')
    eye_t, eye_d, eye_ok = pupil_diameter_series(eye)
    if eye_ok.sum() < 100:
        raise ValueError(f'{ophys_experiment_id}: too few valid pupil frames')
    pupil_frames = np.interp(ts, eye_t[eye_ok], eye_d[eye_ok])
...
    pupil_edges = np.percentile(pupil_frames[frame_mask], QUANTILE_PCTS)
    pupil_bin = np.searchsorted(pupil_edges, pupil_frames, side='right').astype(np.int64)
```

iii. D9: "`eye_tracking` sets `pupil_width/height/area` to NaN on `likely_blink` frames (blinks plus
|z| > 3 area outliers, dilated by ±2 frames) … Those samples are dropped and the trace is linearly
interpolated across them when it is resampled onto the ophys grid. A NaN in the output is a hard error
for the validator, and a separate 'blink' category would not be a percentile bin of diameter."
`pupil_blink_fraction` is recorded per session in the metadata.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: per-session 20/40/60/80th-percentile edges computed over the frames inside the
kept trial windows, giving exactly 20.0 % of samples per bin in every session; edges stored per session
in `metadata['session_info'][i]['pupil_diameter_quintile_edges']`.

ii.
```python
    pupil_edges = np.percentile(pupil_frames[frame_mask], QUANTILE_PCTS)
    pupil_bin = np.searchsorted(pupil_edges, pupil_frames, side='right').astype(np.int64)
...
        out[3] = pupil_bin[a:b]
```

iii. D8 (quoted in 5-c): pupil diameter is in camera pixels and is not comparable across sessions/rigs
(session medians range 48–166 px, sd 18.4 px, larger than the within-session inter-quintile spread of
~15 px), so global quintiles "would mostly encode which session a sample came from", inflating decoding
accuracy without decoding anything about the pupil.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed: resampled onto the session's `ophys_timestamps`, binned on that grid, then
sliced with the identical `[a, b)` trial index range.

ii.
```python
    pupil_frames = np.interp(ts, eye_t[eye_ok], eye_d[eye_ok])
...
        out[3] = pupil_bin[a:b]
```

iii. Step 3: eye-tracking timestamps are on the same sync clock as the ophys frames, so interpolation is
a pure resampling. Panel 8 of `processing_*.png` shows a 60 s excerpt with the raw samples, the dropped
blink frames and the interpolated ophys-grid trace; Step 10 Check 2 recomputes the pupil quintile edges
and spot-checks bin values from the raw `eye_tracking` table (all PASS).

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table — `hit`, `miss`, `false_alarm`,
`correct_reject` — in that fixed order, for the kept `go | catch` trials.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
    outcome = np.full(len(sel), -1, dtype=np.int64)
    for j, name in enumerate(OUTCOME_NAMES):
        outcome[sel[name].to_numpy().astype(bool)] = j
    if np.any(outcome < 0):
        raise ValueError(f'{ophys_experiment_id}: unclassified go/catch trial')
```

iii. Step 1: `trial.py` L182-213 shows `correct_reject = catch and not false_alarm` and that
auto-rewarded trials have all four flags forced to False, so "every such trial carries exactly one of
hit / miss / false_alarm / correct_reject (verified empirically: 0 unclassified trials in all 168
sessions)". The `raise` makes that invariant a hard check rather than an assumption.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The per-trial integer code (0–3) is broadcast across all T frames of the trial as row 4 of the
`(5, T)` output array, i.e. it is stored as a constant time series rather than a 1-D per-trial value.
`output_values[4] = ['hit','miss','false_alarm','correct_reject']`. Resulting distribution over all
frames: hit 0.3157, miss 0.5590, false_alarm 0.0184, correct_reject 0.1068.

ii.
```python
        out = np.empty((len(OUTPUT_NAMES), T), dtype=np.int64)
        ...
        out[4] = outcome[i]
```

iii. D12: "All trials must share one `d_output`, so the per-trial label is broadcast across the trial's
frames. This is numerically identical to supplying it as a 1-D per-trial value
(`decoder.SessionData.__getitem__` broadcasts 1-D outputs across the trial's rows in exactly the same
way)." Step 10 Check 2 verified the label against the trials table for *every* trial of 5 sessions, and
that GO → hit/miss and CATCH → false_alarm/correct_reject.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Session-level failures** are caught per worker, reported to stdout and stored in
  `metadata['skipped_sessions']`, without killing the run. Three sessions (795953296, 806456687,
  833631914) have an empty `eye_tracking` table and are skipped this way; a session with < 100 valid
  pupil frames would be too.
- **Blink / outlier pupil frames** (NaN in the SDK): dropped and linearly interpolated across.
- **Non-finite running samples**: dropped before interpolation.
- **Trials starting before the first flash** (≤ 1 frame, session 792815735): presentation index clamped
  to `[0, n_flashes-1]` instead of producing an invalid label of −1.
- **Dropped stimulus frames**: interval boundaries are always read from the data
  (`[start_time_i, start_time_{i+1})`), never assumed to be 750 ms.
- **Omitted flashes**: forward-filled with the previous image identity, no spurious category.
- **Trials with < 2 ophys frames** and **sessions with no go/catch trials**: guarded (never fire /
  raise).
- **Consistency assertions**: dF/F sample count must equal the number of ophys timestamps; every kept
  trial must have exactly one outcome label.
- Interpolation uses `np.interp`, which clamps at the ends, so no NaN can leak into the outputs; the
  verification log confirms no NaN/Inf anywhere.

ii.
```python
def _worker(args):
    eid, collect_debug = args
    try:
        return convert_experiment(eid, collect_debug=collect_debug)
    except Exception as exc:                                   # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}', 'eid': int(eid)}
```
```python
    if traces.shape[1] != n_frames:
        raise ValueError(f'{ophys_experiment_id}: dff has {traces.shape[1]} samples ...')
    if eye is None or len(eye) == 0:
        raise ValueError(f'{ophys_experiment_id}: no eye tracking data')
    if eye_ok.sum() < 100:
        raise ValueError(f'{ophys_experiment_id}: too few valid pupil frames')
    run_ok = np.isfinite(run_v)
    good = np.isfinite(diam)
    k = np.clip(k, 0, len(start) - 1)
    keep = (b_idx - a_idx) >= 2
```

iii. Step 10 Check 5 enumerates nine edge cases and their handling. D10: sessions without eye tracking
"are excluded rather than filled with fabricated values" because pupil diameter is a required output.
D9: dropping + interpolating blink frames is preferred because "a NaN in the output is a hard error for
the validator, and a separate 'blink' category would not be a percentile bin of diameter". The
skipped-session list is printed in `conversion_full_out.txt` and kept in the pickle so nothing is lost
silently.

## 9-a. What are the most time-consuming steps of the code?

i. `cache.get_behavior_ophys_experiment(eid)` — reading the NWB file through the SDK — dominates
completely at 3.4–5.0 s per experiment (the per-session `convert_seconds` printed in
`conversion_full_out.txt` is essentially all load time). Everything after the load is vectorised and
costs ~0.05 s per session. The other measurable cost is pickling the 8.73 GB output (13.5 s). Total wall
clock for 168 sessions: 49.5 s, of which 34 s is the parallel load phase. The AI mitigated the
bottleneck with a `multiprocessing.Pool` over experiments and printed per-session timings.

ii.
```python
def convert_experiment(ophys_experiment_id, collect_debug=False):
    t_start = time.time()
    cache = get_cache()
    ds = cache.get_behavior_ophys_experiment(ophys_experiment_id)
...
    info = {..., 'convert_seconds': time.time() - t_start}
...
    with Pool(min(args.workers, max(1, len(jobs)))) as pool:
        for i, res in enumerate(pool.imap(_worker, jobs, chunksize=1)):
            ...
    t_load = time.time() - t0
    print(f'conversion of {len(jobs)} sessions took {t_load:.1f}s '
          f'({t_load / max(1, len(jobs)):.2f}s/session wall clock)')
```

iii. Step 6: "NWB load dominates (~3.5–5 s/experiment, single threaded); a naive serial loop over 168
experiments would take ~11 min." Speed-ups recorded: multiprocessing (11 min → 34 s), vectorised
per-frame mappings (~1.5 s → ~0.05 s per session), float32 neural data, and collecting the heavy debug
arrays only for the ≤ 2 plotted sessions.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Very little is left. The per-frame stimulus mapping, the omitted-flash forward fill, the behavioural
resampling, the trial-window lookup and the discretisation are all vectorised
(`np.searchsorted`, `np.interp`, `np.percentile`). The remaining python loops are all O(n_trials) or
O(n_flashes) and negligible:
- `for i, (a, b) in enumerate(zip(a_idx, b_idx))` — the per-trial assembly loop; unavoidable because
  trials have different T and must be separate array objects in the output format (the slicing copy
  itself dominates and is done in C).
- `for a, b in zip(a_idx, b_idx): frame_mask[a:b] = True` — could be replaced by a cumulative-sum /
  `np.add.at` difference trick, but costs microseconds.
- `for j, name in enumerate(OUTCOME_NAMES)` — 4 iterations.
- The list comprehension `[IMAGE_TO_IDX.get(n, -1) ... for n in names]` over ~4800 flashes per session
  — could be a `pandas.Series.map`, again negligible.

ii.
```python
    # vectorised: forward-fill of omitted flashes and frame→interval lookup
    valid_pos = np.where(raw >= 0)[0]
    fill_src = valid_pos[np.searchsorted(valid_pos, np.arange(len(raw)), side='right') - 1]
    k = np.searchsorted(start, ophys_timestamps, side='right') - 1
...
    # remaining small loops
    for a, b in zip(a_idx, b_idx):
        frame_mask[a:b] = True
    for i, (a, b) in enumerate(zip(a_idx, b_idx)):
        neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. Step 6: "all per-frame mappings vectorised with `np.searchsorted`/`np.interp`; the omitted-flash
forward fill done with a `searchsorted` over the indices of valid flashes instead of a python loop".
The AI's stated position is that after vectorisation the residual loops are irrelevant next to the I/O
bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. Two genuine repetitions, both small:
1. **`get_cache()` is called once per experiment** (168 times), inside `convert_experiment`, re-reading
   the project manifest and the metadata CSVs (including the 133 066-row cells table) every time.
   Because the workers are separate processes the cache object cannot be shared, but it could have been
   built once per worker with a `Pool(initializer=...)` instead of once per task
   (`chunksize=1` guarantees one construction per experiment). `main()` opens a third copy for
   `select_experiments`.
2. **`ds.stimulus_presentations` is accessed three times** in the debug path (once in
   `build_stimulus_series` and twice in the `result['debug']` dict literal), and `ds.trials` is
   re-derived for the debug slice — only for the ≤ 2 sessions that are plotted.

Otherwise each session is loaded exactly once and every derived quantity (stimulus series, resampled
behaviour, quintile edges, trial windows) is computed once and reused for all trials; nothing is
recomputed at assembly time.

ii.
```python
def convert_experiment(ophys_experiment_id, collect_debug=False):
    cache = get_cache()                      # re-opened for every experiment
    ds = cache.get_behavior_ophys_experiment(ophys_experiment_id)
...
            'stim': ds.stimulus_presentations[
                ds.stimulus_presentations['stimulus_block_name'] == STIM_BLOCK][...]
```

iii. The AI does not discuss this in CONVERSION_NOTES; its efficiency discussion (Step 6) focuses on the
NWB load and on vectorisation. The repeated cache construction is masked by the 32-way parallelism and
by the fact that the NWB read is ~10× more expensive.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. All of it is cheap, but it exists:
- The stimulus series, running/pupil resampling and quintile binning are computed for **every ophys
  frame of the session** (~140 000 frames, ≈ 75 min) even though only the frames inside go/catch trial
  windows (roughly half) are ever sliced out. `frame_mask` is already computed, so the percentile step
  at least uses only the kept frames.
- `build_stimulus_series` returns the interval index `interval` (`k`), which is never used by the
  caller.
- `cell_specimen_ids` is carried in every session's result dict and then dropped when the final
  dictionary is assembled — it never reaches the pickle.
- Per-session summary statistics (`frac_hit`, `frac_miss`, `frac_false_alarm`, `frac_correct_reject`,
  `pupil_blink_fraction`, quintile edges) are computed for every session and stored in
  `metadata['session_info']`; useful for auditing, unused by the decoder.
- Outputs are stored as `int64` although every value fits in `int8` (5 categories max), which inflates
  the output arrays ~8× (≈ 450 MB instead of ~56 MB inside the 8.73 GB pickle).
- An empty `(0, T)` float32 input array is allocated per trial (42 470 allocations) although the task
  specifies no decoder inputs and `input_names` is empty.
- `n_trials_dropped_short` is tracked but is 0 in every session.

ii.
```python
    return filled[k], change[k].astype(np.int64), k      # `k` never used by the caller
...
        out = np.empty((len(OUTPUT_NAMES), T), dtype=np.int64)   # int8 would suffice
        inputs.append(np.zeros((0, T), dtype=np.float32))        # no decoder inputs
...
    result = {..., 'cell_specimen_ids': cell_ids, ...}           # dropped at assembly
...
        'frac_hit': float(np.mean(outcome == 0)), ...            # metadata only
```

iii. Not discussed as waste in CONVERSION_NOTES; the whole-session computation is the natural
consequence of the design choice to put every stream on the full-session ophys grid before slicing
trials (which is what makes the alignment trivially correct), and the metadata fields were added
deliberately for the Step 9/10 consistency checks. The total conversion still runs in 50 s, so none of
this was a practical problem.
