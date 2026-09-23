# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the AllenSDK project-cache object. Instead it reads the release's own metadata CSV (`project_metadata/ophys_experiment_table.csv`) straight off disk, intersects it with the set of NWB files that actually exist in `behavior_ophys_experiments/` (parsed from the filenames), and then opens each surviving NWB file directly with the SDK class `BehaviorOphysExperiment.from_nwb_path()`. Every experiment is converted in a separate worker process (`ProcessPoolExecutor`, 24 workers by default) and its extracted per-trial content is written to a per-experiment pickle in `/app/_conversion_cache/`; a second pass (`assemble`) reads those caches back and builds the final dictionary. From each loaded experiment object it pulls `ophys_timestamps`, `dff_traces`, `trials`, `stimulus_presentations`, `running_speed`, `eye_tracking` and `metadata`.

ii.
```python
def select_experiments():
    """Return the metadata rows of the experiments that will be converted."""
    exp = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    available = set()
    for fname in os.listdir(NWB_DIR):
        m = re.match(r'behavior_ophys_experiment_(\d+)\.nwb$', fname)
        if m:
            available.add(int(m.group(1)))
    exp = exp[exp.ophys_experiment_id.isin(available)]
    ...
```
```python
    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
        BehaviorOphysExperiment,
    )

    nwb_path = os.path.join(
        NWB_DIR, f'behavior_ophys_experiment_{ophys_experiment_id}.nwb')
    ds = BehaviorOphysExperiment.from_nwb_path(nwb_path)
    md = ds.metadata
```
```python
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, e) for e in eids]
        for i, fut in enumerate(as_completed(futs)):
            eid, path, err = fut.result()
```

iii. From the trajectory: the AI first inspected the on-disk release and found only a subset of the full release is present locally (284 NWB files vs. the full manifest). Intersecting the experiment table with the files on disk is what lets it enumerate "all the data" without attempting network fetches for absent experiments. It chose the per-experiment process pool plus disk cache explicitly for throughput (247 GB of NWB on disk) and restartability — the cache check at the top of `convert_experiment` makes the script resumable.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the unique `mouse_id` strings taken from the metadata of each loaded experiment, sorted, and each session is given the index of its mouse. 37 mice result.

ii.
```python
        'mouse_id': str(md['mouse_id']),
```
```python
    subjects = sorted({s['mouse_id'] for s in sessions})
    subject_to_idx = {m: i for i, m in enumerate(subjects)}
    ...
        subject_idx.append(subject_to_idx[s['mouse_id']])
```

iii. No separate justification is given in the trajectory beyond the implicit one: `mouse_id` is the SDK's canonical per-animal identifier and is carried in both the experiment table and the per-experiment metadata. The AI cross-checked the count against the metadata table (37 mice for `project_code == 'VisualBehavior'`).

## 1-c. How are the data split into sessions?

i. One converted "session" == one ophys experiment. The AI restricts to `project_code == 'VisualBehavior'` (the single-plane Scientifica rigs), where each ophys session contains exactly one imaging plane, so experiment and session are 1:1 (it verified 239 experiments = 239 sessions). It additionally drops all *passive* sessions (`passive == True`), keeps both familiar and novel image sets, and drops the 3 experiments that have no eye-tracking data. Sessions are ordered by `ophys_experiment_id`. Result: 165 sessions.

ii.
```python
    # single-plane "VisualBehavior" variant -> one common ophys frame rate and
    # one imaging plane per session
    exp = exp[exp.project_code == 'VisualBehavior']
    # active behavior only (passive sessions have no behavioral report)
    exp = exp[~exp.passive.astype(bool)]
    return exp.sort_values('ophys_experiment_id').reset_index(drop=True)
```
```python
    # ---- pupil ------------------------------------------------------------ #
    pt, pv = _pupil_diameter(ds.eye_tracking)
    if pt is None:
        # without pupil data the session cannot supply one of the required
        # outputs, so it is dropped
        return None
```

iii. Docstring and final summary: the Multiscope (multi-plane) variant is excluded because it is acquired at 10.7 Hz and "cannot share a common time bin with the single-plane data", while the format requires one bin size everywhere. Passive sessions are excluded because "the lick spout is retracted, so there is no behavioral report and the trial-outcome output would be undefined (every trial is a miss / correct-reject by construction)" — the AI verified this empirically on experiment 877696762 (`OPHYS_2_images_A_passive`: 354 go / 52 catch trials, 0 hits, 0 false alarms). Novel image sets are kept because the reference paper's familiar-only restriction "served its novelty question, not this decoding task". The three eye-tracking-less experiments (795953296, 806456687, 833631914) were diagnosed individually (`eye rows 0`) before being dropped.

## 1-d. How are the data split into trials?

i. Trials come from the SDK's `BehaviorOphysExperiment.trials` table. A trial spans `start_time` → `stop_time`; the corresponding ophys frame indices are found with `np.searchsorted` (side='left'), so trial length varies (217–389 bins, mean 262 ≈ 8.5 s). Only Go and Catch trials are retained; Aborted and Auto-rewarded are dropped.

ii.
```python
    trials = ds.trials
    keep = ((trials['go'].astype(bool) | trials['catch'].astype(bool))
            & ~trials['aborted'].astype(bool)
            & ~trials['auto_rewarded'].astype(bool))
    trials = trials[keep]
```
```python
    starts = np.searchsorted(ts, trials['start_time'].values, side='left')
    stops = np.searchsorted(ts, trials['stop_time'].values, side='left')
    ...
    for k in range(len(trials)):
        i0, i1 = int(starts[k]), int(stops[k])
```

iii. The instructions explicitly say to "segment each recording session into individual trials based on how they are defined in the experiment", to include Go and Catch and exclude Aborted and Auto-rewarded, which is exactly the boolean mask used. The AI verified in the trajectory that `go`/`catch` and `auto_rewarded` are disjoint in this release, and that the resulting trial count matches the SDK selection exactly (`ntrials cache 39 sel 39` on experiment 775614751). It used the full `start_time`→`stop_time` window rather than a fixed window so that pre-change flashes and the post-change response window are both inside the trial, which is what makes the time-varying image-identity/change outputs meaningful.

## 1-e. How are trials filtered based on quality controls?

i. Beyond the Go/Catch selection: (a) trials shorter than 2 ophys frames are skipped; (b) trials containing any frame that precedes the first stimulus flash are skipped (their image identity would be undefined); (c) trials whose row is not exactly one of hit/miss/false_alarm/correct_reject are dropped; (d) an experiment producing fewer than 2 usable trials is dropped entirely; (e) experiments with no usable eye-tracking are dropped entirely. No engagement/lick-rate/performance-based trial filtering is applied.

ii.
```python
    outcome = np.full(len(trials), -1, dtype=np.int64)
    for i, name in enumerate(OUTCOME_NAMES):
        outcome[trials[name].astype(bool).values] = i
    if np.any(outcome < 0):
        # every go/catch trial is exactly one of hit/miss/FA/CR
        ok = outcome >= 0
        trials = trials[ok]
        outcome = outcome[ok]
        if len(trials) < 2:
            return None
```
```python
        i0, i1 = int(starts[k]), int(stops[k])
        if i1 - i0 < 2:
            continue
        if np.any(image_name[i0:i1] == None):  # noqa: E711 - before 1st flash
            continue
    ...
    if len(neural_trials) < 2:
        return None
```

iii. The AI's stated rationale is that the Allen pipeline plus the instruction-mandated Go/Catch selection is the curation; the extra guards are defensive. The ≥2-trial rule comes directly from the format requirement ("There needs to be at least two trials within each session in order to evaluate the decoder performance"). The "before first flash" guard exists because `_flash_labels` returns `None` for frames preceding the first stimulus presentation, which would otherwise have no image label.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `dff_traces` — the Allen pipeline's detrended ΔF/F trace for every released ROI, accessed as `ds.dff_traces['dff']` and stacked into an (n_neurons, T) matrix. The script keeps an environment-variable switch (`VB_TRACE=events` / `filtered_events`) to select the deconvolved event traces instead, but the default and the shipped dataset use `dff`.

ii.
```python
    if TRACE == 'dff':
        src = ds.dff_traces
        col = 'dff'
    else:
        src = ds.events
        col = TRACE
    cell_ids = src.index.values
    traces = np.vstack([np.asarray(v, dtype=np.float32)
                        for v in src[col].values])
    assert traces.shape[1] == ts.shape[0]
```

iii. This is an explicitly flagged departure from the reference paper, which analyses deconvolved calcium `events`. The AI first built the whole dataset with `filtered_events`, ran the decoder, and compared: with events, 1594 of 42467 trials (3.8 %) are all-zero (the format checker emits 159+ "all neural data is zero" warnings) and validation balanced accuracy drops on every output (image identity 0.40→0.21, image change 0.61→0.55, outcome 0.30→0.28). Its argument is that the paper works at 750 ms image-presentation granularity where sparse events are still informative, whereas this decoder classifies every 32 ms frame; the task instructions permit discrepancies "required by ... the task of training a neural decoder". It documented the switch so the event-based variant is reproducible.

## 2-b. How is the `neural` data processed?

i. None beyond what the Allen pipeline already did (motion correction → segmentation → demixing → neuropil subtraction → ΔF/F → detrending). The AI casts to float32, vertically stacks the per-ROI traces, asserts the trace length equals the number of ophys timestamps, and replaces any non-finite value with 0. No normalisation, smoothing, z-scoring, or PCA. Because only single-plane experiments are used, there is no cross-plane merging to do.

ii.
```python
    traces = np.vstack([np.asarray(v, dtype=np.float32)
                        for v in src[col].values])
    assert traces.shape[1] == ts.shape[0]
    # the released traces are finite, but guard against the occasional gap
    np.nan_to_num(traces, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The docstring states the Allen pipeline's processing is the processing documented in the technical whitepaper (reproduced in `methods.txt`), so no further step is warranted. The `nan_to_num` call is described as a guard, not as an expected correction.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering at all. Every ROI present in the NWB `dff_traces` table is kept (28,821 neurons over 165 sessions; 6–666 per session).

ii. No filtering code exists; the relevant statement is the docstring:
```
  * All ROIs released in the NWB file are kept: the Allen pipeline has already
    applied its ROI filtering / demixing / neuropil-correction QC (every ROI in
    the release has `valid_roi == True`), and the reference paper applies no
    further neuron-level curation.
```

iii. The AI checked `cell_specimen_table.valid_roi` on sample experiments and found `ncells == valid` in every case (e.g. `ncells 478 valid 478`, `ncells 30 valid 30`), i.e. the released NWB files already contain only ROIs that passed the pipeline's multi-label ROI classifier described in `methods.txt`. It therefore treats further curation as redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The ophys timestamps are the master clock, as the instructions demand. Each trial is the contiguous block of ophys frames from the first frame at or after `trials.start_time` up to (exclusive) the first frame at or after `trials.stop_time`; the neural matrix is simply that column slice. `metadata['temporal_alignment_event']` is the trial `start_time`, with `off_start = 0.0` and `off_end = None` (variable-length trials).

ii.
```python
    ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
    ...
    starts = np.searchsorted(ts, trials['start_time'].values, side='left')
    stops = np.searchsorted(ts, trials['stop_time'].values, side='left')
    ...
        neural_trials.append(np.ascontiguousarray(traces[:, i0:i1]))
```
```python
            'temporal_alignment_event': (
                'trial start_time (onset of the change-detection trial, i.e. '
                'the first ophys frame at or after trials.start_time); all data '
                'streams are sampled on the ophys frame clock'),
            'off_start': 0.0,
            'off_end': None,
```

iii. "Temporally align based on ophys timestamp" is an explicit instruction, and the AI makes the ophys frame grid the common index for every stream so that no re-alignment is needed at trial-slicing time. It verified alignment frame-by-frame against the SDK on experiment 775614751: cached trial length equals `searchsorted(stop) - searchsorted(start)` exactly, and the change label lands at `change_time - start_time` to within one frame (6.060 s vs. 6.059 s).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling of the neural data. The time bin *is* the native ophys frame, ~32.32 ms (31 Hz). `metadata['time_bin_size']` is the median across sessions of each session's median inter-frame interval, in ms (32.32). Restricting to the single-plane `VisualBehavior` project is what makes a single common bin size valid. All other streams are brought onto this grid rather than the neural data being moved.

ii.
```python
        'dt': float(np.median(np.diff(ts))),
```
```python
    dts = np.array([s['dt'] for s in sessions])
    time_bin_size = float(np.median(dts) * 1000.0)
```

iii. Docstring: "Time bins are the ophys frames themselves (~32.32 ms); the stimulus, running and pupil streams are aligned onto those frame times, so no resampling of the neural data is needed." The AI rejected the Multiscope experiments precisely because their 10.7 Hz rate would have broken the format's "time bins should be the same size for all trials and sessions" requirement.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From `ds.stimulus_presentations` — specifically the `start_time` and `image_name` columns of the change-detection stimulus block — rather than from the trials table. Every ophys frame is labelled with the `image_name` of the most recent flash onset at or before it.

ii.
```python
    stim = ds.stimulus_presentations
    if 'stimulus_block_name' in stim.columns:
        stim = stim[stim['stimulus_block_name'].astype(str)
                    .str.contains('change_detection')]
    else:  # pragma: no cover - fallback for older files
        stim = stim[stim['active'].astype(bool)]
    stim = stim.sort_values('start_time')
    image_name, is_change = _flash_labels(stim, ts)
```
```python
def _flash_labels(stim, ts):
    starts = stim['start_time'].values
    # index of the most recent flash onset at or before each ophys frame
    idx = np.searchsorted(starts, ts, side='right') - 1
    valid = idx >= 0
    idx_clipped = np.where(valid, idx, 0)

    names = stim['image_name'].values.astype(object)
    changes = stim['is_change'].values.astype(bool)

    image_name = np.where(valid, names[idx_clipped], None)
    is_change = np.where(valid, changes[idx_clipped], False)
    return image_name, is_change
```

iii. The docstring cites the reference paper's own definition: "By image presentation interval we refer to the 750 ms interval beginning with each image presentation. For image omissions we used the 750 ms following the time of the omission." Labelling by the most recent flash onset reproduces that interval exactly and, unlike a trials-table-derived label, automatically represents the 5 %-probability omitted flashes (which carry `image_name == 'omitted'` in the SDK) as their own category.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The per-frame image names are mapped to integer codes with a single global vocabulary built across all sessions: the real image names sorted alphabetically, with `'omitted'` appended last. 17 categories result (16 images from set A + set B, plus `omitted`). The code is held constant across the whole 750 ms flash cycle (image + following grey screen).

ii.
```python
    images = sorted({name for s in sessions for tr in s['image_name']
                     for name in np.unique(tr)} - {'omitted'})
    image_values = images + ['omitted']
    image_to_idx = {name: i for i, name in enumerate(image_values)}
```
```python
            img = np.array([image_to_idx[n] for n in s['image_name'][k]],
                           dtype=np.int64)
```

iii. A single global, deterministically sorted mapping keeps codes comparable across sessions; `omitted` is placed last so that the 16 real images occupy contiguous codes 0–15. Holding the label through the grey period follows the paper's image-presentation-interval convention (the alternative — labelling grey frames as a separate "no image" class — would make ~2/3 of all bins that class). The resulting distribution is near-uniform: each image ≈ 0.058–0.062 of bins, `omitted` 0.034.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The label array is computed once per session on the full ophys timestamp vector, then sliced with the *same* `[i0:i1]` indices as the neural matrix, so alignment is exact by construction.

ii.
```python
    image_name, is_change = _flash_labels(stim, ts)
    ...
        neural_trials.append(np.ascontiguousarray(traces[:, i0:i1]))
        image_trials.append(image_name[i0:i1].copy())
```

iii. Computing every stream on the ophys frame grid *before* trial segmentation and then using one index slice for all of them is the AI's stated alignment strategy throughout. It verified the transitions by printing them relative to trial start for a real trial (im062 at t=0.019 s, im069 at t=6.060 s, with `change_time - start_time = 6.059 s`).

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the `is_change` boolean column of `ds.stimulus_presentations` (not from `trials.change_time`/`trials.go`), carried per flash via the same `_flash_labels` lookup.

ii.
```python
    changes = stim['is_change'].values.astype(bool)
    is_change = np.where(valid, changes[idx_clipped], False)
```

iii. `stimulus_presentations` separates genuine changes (`is_change`) from sham changes (`is_sham_change`), so using `is_change` automatically gives 1s only on Go trials and none on Catch trials — which the docstring states explicitly ("Catch trials contain a sham change and therefore carry no 1s"). Deriving it from the same flash table as image identity keeps the two outputs mutually consistent.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The boolean is held high for the entire 750 ms presentation interval of the changed flash (i.e. from the change flash onset until the next flash onset), then cast to int64. Everything else is 0.

ii.
```python
            chg = s['is_change'][k].astype(np.int64)
```
(the per-frame value itself comes from `_flash_labels`, quoted in 4-a)

iii. The instruction is "Have value of 1 right after a change in image identity, otherwise 0". Marking the whole flash cycle rather than a single 32 ms bin gives the decoder a target that is actually learnable at 31 Hz; the AI verified the marked window is exactly one cycle (`change window t rel: 6.060 → 6.803`, i.e. 743 ms) and that the resulting global fraction of 1s is 0.077.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — the variable is already binary. It is emitted as {0, 1} with value names `['no_change', 'change']`.

ii.
```python
        'output_values': [
            image_values,
            ['no_change', 'change'],
            ...
```

iii. Implicit: the instruction defines it as a binary variable, so the flash-level boolean is used directly.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: computed on the full ophys timestamp grid, sliced with the same `[i0:i1]` trial indices.

ii.
```python
        change_trials.append(is_change[i0:i1].astype(np.int64))
```

iii. Same rationale as 3-c; verified frame-by-frame against `trials.change_time` on a sample trial.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `ds.running_speed`, using its `timestamps` and `speed` columns (the running-wheel encoder signal, ~60 Hz).

ii.
```python
    run = ds.running_speed
    run_t = np.asarray(run['timestamps'].values, dtype=np.float64)
    run_v = np.asarray(run['speed'].values, dtype=np.float64)
```

iii. `running_speed` is the SDK's standard locomotion interface; the AI checked its sampling rate and time range against the ophys clock before using it.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Non-finite samples are dropped, the signal is linearly interpolated onto the ophys frame times with `np.interp`, and later discretised into quintiles. `np.interp` clamps (holds the endpoint value) outside the encoder's time range rather than producing NaN, which matters because the running stream ends ~16 s before the last ophys frame.

ii.
```python
    good = np.isfinite(run_t) & np.isfinite(run_v)
    running = np.interp(ts, run_t[good], run_v[good])
```

iii. Linear interpolation onto the ophys clock is the AI's uniform strategy for every non-ophys stream so that one slice index serves all of them. It verified the cached values reproduce `np.interp` on the raw SDK arrays exactly.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Into five quintile bins. The 20/40/60/80th percentiles are computed **once, globally**, over the pooled distribution of every converted time bin in the whole dataset (not per session, not per trial), and applied with `np.searchsorted(..., side='right')`. Edges came out at [0.043, 4.403, 22.636, 36.295] cm/s, and the realised distribution is exactly 0.200 per bin.

ii.
```python
    run_all = np.concatenate([np.concatenate(s['running_speed']) for s in sessions])
    qs = np.linspace(0, 100, NBINS + 1)[1:-1]
    run_edges = np.percentile(run_all, qs)
```
```python
            run = np.searchsorted(run_edges, s['running_speed'][k],
                                  side='right').astype(np.int64)
```

iii. Docstring: "The bin edges are the 20/40/60/80th percentiles of the pooled distribution over every time bin of the whole converted dataset, so a bin index means the same absolute range everywhere." The AI noted in its final report that it read "five equal percentile bins" as dataset-wide rather than per-session, and flagged the consequence (some sessions sit almost entirely in one bin) for a human to review.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated onto the ophys frame grid before trial segmentation, then sliced with the same `[i0:i1]` indices as the neural matrix.

ii.
```python
    running = np.interp(ts, run_t[good], run_v[good])
    ...
        running_trials.append(running[i0:i1].astype(np.float64))
        neural_trials.append(np.ascontiguousarray(traces[:, i0:i1]))
```

iii. Same single-grid strategy; verified numerically (`running cache first5` == `np.interp(...)` on the raw stream).

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. From `ds.eye_tracking`, using the `pupil_area` column (the fitted pupil-ellipse area) and its `timestamps`. Diameter is derived as the equivalent-circle diameter, `2*sqrt(area/pi)`, in pixels.

ii.
```python
def _pupil_diameter(eye_tracking):
    if eye_tracking is None or len(eye_tracking) == 0:
        return None, None
    t = eye_tracking['timestamps'].values
    area = eye_tracking['pupil_area'].values.astype(float)
    good = np.isfinite(area) & np.isfinite(t) & (area > 0)
    if good.sum() < 100:
        return None, None
    return t[good], 2.0 * np.sqrt(area[good] / np.pi)
```

iii. Docstring: "Diameter is derived from the fitted pupil ellipse area as 2*sqrt(area/pi), which is more robust than either single ellipse axis." Using the area rather than one axis avoids sensitivity to which axis the ellipse fit happened to assign as major/minor.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are removed, then the signal is linearly interpolated onto the ophys frame times and later quintile-binned. Blink removal is implicit: the AI relies on the SDK already writing NaN into `pupil_area` wherever `likely_blink` is True, and simply keeps only finite, positive-area samples. A session with fewer than 100 usable pupil samples (in practice, zero eye-tracking rows) returns `None` and the whole experiment is dropped.

ii.
```python
    pt, pv = _pupil_diameter(ds.eye_tracking)
    if pt is None:
        # without pupil data the session cannot supply one of the required
        # outputs, so it is dropped
        return None
    pupil = np.interp(ts, pt, pv)
```

iii. Docstring: "The SDK already sets `pupil_area` to NaN on frames flagged `likely_blink`; those frames are dropped here and the signal is linearly interpolated onto the ophys clock by the caller." The AI verified this empirically before relying on it — across sample experiments the NaN fraction of `pupil_area` equals the `likely_blink` fraction exactly (0.008/0.008, 0.035/0.035, 0.018/0.018).

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Exactly as running speed: five global quintile bins from the pooled distribution over every converted time bin. Edges came out at [75.42, 84.94, 94.18, 107.17] pixels, realised distribution 0.200 per bin.

ii.
```python
    pup_all = np.concatenate([np.concatenate(s['pupil_diameter']) for s in sessions])
    pup_edges = np.percentile(pup_all, qs)
    ...
            pup = np.searchsorted(pup_edges, s['pupil_diameter'][k],
                                  side='right').astype(np.int64)
```

iii. Same reasoning as 5-c, including the explicit caveat the AI raised: "the global pupil quintiles are strongly session-dependent, so some sessions sit almost entirely in one bin; per-session quintiles would force within-session decoding instead, but 'five equal percentile bins' read to me as dataset-wide."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the ophys frame grid before trial segmentation, sliced with the same `[i0:i1]` indices.

ii.
```python
    pupil = np.interp(ts, pt, pv)
    ...
        pupil_trials.append(pupil[i0:i1].astype(np.float64))
```

iii. Same single-grid strategy as every other stream; the eye and ophys clocks are hardware-synced in this release (the AI checked that the eye-tracking time range covers the ophys time range).

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. From the four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, in that fixed order.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
    outcome = np.full(len(trials), -1, dtype=np.int64)
    for i, name in enumerate(OUTCOME_NAMES):
        outcome[trials[name].astype(bool).values] = i
```

iii. These are the SDK's canonical outcome labels for the change-detection task. The AI verified on sample sessions that every Go/Catch non-auto-rewarded trial is labelled by exactly one of the four (`outcome sum {1: N}`), which is why the `-1` sentinel path is only a guard.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The per-trial integer code (0–3) is broadcast across all of the trial's time bins, so the output row is constant within a trial but still time-varying in shape, matching the other four outputs. Realised distribution: hit 0.316, miss 0.559, false_alarm 0.018, correct_reject 0.107.

ii.
```python
            out = np.full(T, s['trial_outcome'][k], dtype=np.int64)
            ...
            sess_output.append(np.stack([img, chg, run, pup, out], axis=0))
```

iii. Docstring: "Held across the trial's time bins rather than stored as a 1-D per-trial value, since the format asks for time-varying outputs where possible." This also keeps all five outputs in one (5, T) array, which the format requires.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Layered handling: (a) every experiment is converted inside a try/except in the worker, so one bad file prints a traceback and is dropped without killing the run; (b) non-finite values in the ΔF/F matrix are zeroed (`nan_to_num`); (c) non-finite running samples are dropped before interpolation, and `np.interp` clamps rather than extrapolating past the encoder's time range; (d) blink/NaN/zero-area pupil samples are dropped before interpolation; (e) an experiment with no usable pupil signal is dropped whole (3 experiments); (f) degenerate trials (<2 frames, frames before the first flash, unlabelled outcome) are skipped; (g) sessions left with <2 trials are dropped; (h) the per-experiment cache is written to a `.tmp` file and atomically `os.replace`d, so an interrupted run cannot leave a half-written cache.

ii.
```python
def _worker(eid):
    try:
        return eid, convert_experiment(eid), None
    except Exception:  # pragma: no cover
        import traceback
        return eid, None, traceback.format_exc()
```
```python
    np.nan_to_num(traces, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    ...
    good = np.isfinite(run_t) & np.isfinite(run_v)
    ...
    good = np.isfinite(area) & np.isfinite(t) & (area > 0)
```
```python
    tmp = cache_path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    os.replace(tmp, cache_path)
```

iii. The AI's stated position is that the released data are already clean (it checked: `run nan 0.0`, all ROIs valid, outcomes exhaustive) and that these paths are guards rather than expected corrections; the one case that actually fires is the missing eye-tracking, which it diagnosed explicitly rather than silently absorbing. The final validator run reports "Data format is valid, no errors or warnings."

## 9-a. What are the most time-consuming steps of the code?

i. (1) Reading and decoding the NWB files — `BehaviorOphysExperiment.from_nwb_path` plus materialising `dff_traces`, `events`, `eye_tracking` (135k–271k rows) and `running_speed` for 168 experiments out of a 247 GB on-disk release. This dominates and is I/O bound. (2) Serialising and deserialising the per-experiment cache — ~8 GB written then read back. (3) Writing the final 8.73 GB pickle. The AI mitigates (1) with a 24-process pool, and (2)/(3) are the price of that design.

ii.
```python
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, e) for e in eids]
```
```python
    cache_path = os.path.join(CACHE_DIR, f'{ophys_experiment_id}.pkl')
    if os.path.exists(cache_path):
        return cache_path
```

iii. The AI never states this in prose, but the design is unambiguous about where it believed the cost was: the only two optimisations in the script (process pool, resumable disk cache) both target NWB loading, and the `--limit`/`--workers` flags exist to control it. In the trajectory it timed a single load at 4.8 s and ran the full conversion under `timeout 3000`.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three candidates, all in the second pass: (1) the per-frame Python dict lookup that maps image names to codes — `[image_to_idx[n] for n in s['image_name'][k]]` runs once per time bin, ~11 M iterations in total, and could be done with `np.unique(..., return_inverse=True)` or a `searchsorted` on a sorted name array; (2) the vocabulary build, which calls `np.unique` on every trial of every session inside a set comprehension; (3) the per-trial slicing loop in `convert_experiment` — the `searchsorted` calls are already vectorised outside the loop, so only the slicing/copying remains, and that cannot be usefully vectorised because trials have different lengths. The frame-labelling itself (`_flash_labels`) is already fully vectorised.

ii.
```python
            img = np.array([image_to_idx[n] for n in s['image_name'][k]],
                           dtype=np.int64)
```
```python
    images = sorted({name for s in sessions for tr in s['image_name']
                     for name in np.unique(tr)} - {'omitted'})
```

iii. No justification is offered in the trajectory; the AI vectorised the parts it touched directly (flash labelling, interpolation, trial boundary lookup) and left the assembly-stage loops in their readable form, where they are small relative to I/O.

## 9-c. What processing does the code repeat multiple times?

i. (1) The neural data is serialised twice and copied at least three times: `np.ascontiguousarray(traces[:, i0:i1])` in pass 1 → pickled to the per-experiment cache → unpickled in `assemble` → `np.asarray(s['neural'][k], dtype=np.float32)` (a no-op cast that still walks the array) → pickled again into the 8.73 GB output. (2) `np.unique` is called on every trial's image-name array purely to build the vocabulary, and then every frame is re-visited to map names to codes. (3) During development the AI converted the entire dataset twice, once with `filtered_events` and once with `dff`, and trained the decoder on both; the shipped script keeps that as an env-var switch so it is not repeated at runtime.

ii.
```python
        neural_trials.append(np.ascontiguousarray(traces[:, i0:i1]))
    ...
    with open(tmp, 'wb') as f:
        pickle.dump(out, f, protocol=4)
```
```python
    for p in cache_paths:
        with open(p, 'rb') as f:
            sessions.append(pickle.load(f))
    ...
            sess_neural.append(np.asarray(s['neural'][k], dtype=np.float32))
```

iii. The round-trip through the cache is a deliberate trade: it is what allows the 24-way process pool (worker results have to cross a process boundary anyway) and makes the multi-hour run resumable, which the AI exercised — it re-ran `assemble` alone at the end to fix metadata without re-reading any NWB file.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `cell_specimen_ids` is extracted and cached for every experiment but never written into the output dictionary. (2) Continuous float64 running-speed and pupil-diameter traces are cached per trial at full precision even though only their quintile index survives; `assemble` then materialises two full pooled copies (`run_all`, `pup_all`, ~11 M floats each) purely to call `np.percentile`. (3) All five output rows are stored as int64 when the value ranges are 0–16 at most — int8 would cut ~0.4 GB from the pickle. (4) `input` is materialised as a real `(0, T)` float32 array per trial, one object per trial, rather than a single shared empty array. (5) Per-session `dt` is stored for 165 sessions but only its median is used; `behavior_session_id`, `imaging_depth`, `equipment_name` etc. are carried into `session_info` but not consumed by the decoder.

ii.
```python
    cell_ids = src.index.values
    ...
        'cell_specimen_ids': np.asarray(cell_ids),
```
```python
    run_all = np.concatenate([np.concatenate(s['running_speed']) for s in sessions])
    pup_all = np.concatenate([np.concatenate(s['pupil_diameter']) for s in sessions])
```
```python
            sess_input.append(np.zeros((0, T), dtype=np.float32))
            sess_output.append(np.stack([img, chg, run, pup, out], axis=0))
```

iii. Most of these are provenance rather than waste by intent — the instructions ask for a `session_info` metadata field and the AI populated it generously. The continuous running/pupil values genuinely have to be retained until the end, because the quintile edges are global and cannot be computed before every session has been converted; `del run_all, pup_all` immediately after the percentile call shows the AI was aware of the memory cost.
