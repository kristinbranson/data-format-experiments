# Decisions

Documentation of the decisions made by the agentic AI system (`claude-code` / `claude-opus-5`) when
converting the Allen Brain Observatory Visual Behavior 2P dataset into the decoder format.
Sources: `/app/convert_data.py`, `/app/CONVERSION_NOTES.md`, `/app/README.md`, `/logs/agent/trajectory.json`.

---

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. **Decisions**

- All data are read through the AllenSDK cloud-cache API: `VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir='/app/data')`, which works offline against the already-downloaded manifest. No `.nwb` file is opened directly with `h5py`/`pynwb`.
- The experiment table (`cache.get_ophys_experiment_table()`, 1936 released experiments) is the master list. It is intersected with the set of NWB files actually present on disk (obtained by listing the cache directory, not by opening the files), which gives 284 local experiments.
- From those, only **active-behaviour** experiments are kept (`~et.passive`): 202 experiments in 174 ophys sessions from 38 mice. **Both** project codes are kept — `VisualBehavior` (single plane, 31 Hz, 168 sessions) and `VisualBehaviorMultiscope` (up to 8 simultaneous planes, 11 Hz, 6 sessions).
- Experiments are grouped by `ophys_session_id`; each group is converted in one worker process, which loads every plane with `cache.get_behavior_ophys_experiment(oeid)`. Sessions are processed in parallel (`multiprocessing.Pool`), full conversion = 69 s.
- Per session, everything is taken from the `BehaviorOphysExperiment` object: `dff_traces`, `ophys_timestamps`, `trials`, `stimulus_presentations`, `running_speed`, `eye_tracking`, `metadata`.

ii. **Code snippets**

```python
CACHE_DIR = '/app/data'

def get_cache():
    """Open the local AllenSDK cache (offline)."""
    return VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=CACHE_DIR)

def local_experiment_ids():
    """ophys_experiment_ids whose NWB file is present in the local cache."""
    d = os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0', 'behavior_ophys_experiments')
    return sorted(int(re.findall(r'(\d+)', f)[0]) for f in os.listdir(d))

def select_sessions(cache):
    et = cache.get_ophys_experiment_table()
    ids = local_experiment_ids()
    sel = et[et.index.isin(ids) & (~et.passive)].copy()
    sel = sel.sort_values(['ophys_session_id', 'ophys_experiment_id'])
    session_ids = sorted(sel.ophys_session_id.unique().tolist())
    return sel, session_ids
```

```python
def convert_session(args):
    session_id, oeids, show_processing, signal_name, neural_lag = args
    cache = get_cache()
    datasets = [cache.get_behavior_ophys_experiment(int(o)) for o in oeids]
    ds0 = datasets[0]
```

```python
    for i, sid in enumerate(session_ids):
        oeids = exps[exps.ophys_session_id == sid].index.tolist()
        jobs.append((int(sid), oeids, show and i < 2, args.signal, args.neural_lag))
    with Pool(args.nproc) as pool:
        for k, r in enumerate(pool.imap_unordered(convert_session, jobs, chunksize=1)):
```

iii. **Justification (CONVERSION_NOTES Steps 1, 2, 4, 5)**

- "`from_local_cache` fails here because the cache was built by `from_s3_cache` and has no `manifests/` subfolder"; `from_s3_cache` reads the already-downloaded manifest and works fully offline.
- The project metadata CSVs "describe the **whole release**; only a subset of NWB files is present locally", so the experiment table is intersected with the files on disk to avoid attempting S3 fetches.
- Passive sessions are excluded because "in passive sessions the lick spout is retracted and no choices exist", so trial outcome would be undefined; "Both project codes (single-plane + Multiscope) are kept: both run the identical change-detection task."
- Loading is the only real cost ("~3.5 s / experiment, 98 % of per-session time"), hence multiprocessing over sessions.

---

## 1-b. How are the data split into subjects?

i. **Decisions**

- The subject of a session is `ds0.metadata['mouse_id']` (string), read from the loaded experiment rather than from the table.
- `subjects` is the sorted list of unique mouse ids across the converted sessions (38 mice); `subject_idx` is the index of each session's mouse in that list.

ii. **Code snippets**

```python
    meta = ds0.metadata
    result = {..., 'mouse_id': str(meta['mouse_id']), ...}
```

```python
    subjects = sorted({r['mouse_id'] for r in results})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    data = {...
        'subjects': subjects,
        'subject_idx': np.array([subject_to_idx[r['mouse_id']] for r in results], dtype=np.int64),
    ...}
```

iii. **Justification**

`mouse_id` is the SDK's unique animal identifier. The notes cross-check the resulting count against the data ("38 mice with local active NWBs" from the experiment table vs. 38 in the converted data) and against the release-level number in the paper (82 mice in the whole release, of which only a subset is downloaded here). Sorting makes the mapping deterministic.

---

## 1-c. How are the data split into sessions?

i. **Decisions**

- A "session" in the output format = one **`ophys_session_id`** (one continuous recording), not one experiment/imaging plane. All experiments (planes) sharing an `ophys_session_id` are loaded together and **concatenated along the neuron axis**, with each neuron tagged by its plane's `targeted_structure` for `brain_region_idx`.
- Only active-behaviour sessions are included (passive replay sessions dropped); sessions are ordered by `ophys_session_id`.
- 174 candidate sessions → 171 converted (3 dropped for completely missing eye tracking).

ii. **Code snippets**

```python
    sel = sel.sort_values(['ophys_session_id', 'ophys_experiment_id'])
    session_ids = sorted(sel.ophys_session_id.unique().tolist())
```

```python
    neural_blocks, region_per_neuron, cell_ids = [], [], []
    for ds in datasets:
        ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
        traces = np.vstack(ds.dff_traces.dff.values).astype(np.float64)
        means, counts = bin_mean(traces, ts - neural_lag, edges_flat)
        block = means[:, keep].reshape(traces.shape[0], len(sel), NBINS)
        neural_blocks.append(block)
        region_per_neuron += [ds.metadata['targeted_structure']] * traces.shape[0]
    neural = np.concatenate(neural_blocks, axis=0)   # (n_neurons, n_trials, NBINS)
```

iii. **Justification (Step 4 discrepancy table, Step 5 decision 1)**

"Use **ophys_session** as the 'session' of the output format: planes recorded simultaneously are concatenated along the neuron axis. This is the only definition under which all neurons of a session are simultaneously recorded, and it is what lets `brain_region_idx` carry VISp/VISl within a session." The notes verify that the planes of a Multiscope session share one `trials` table and one behaviour stream before merging. Passive sessions are excluded because "trial outcome (hit/miss/FA/CR) is undefined/degenerate without licking" and the paper's behavioural analyses use "all active behavioral sessions".

---

## 1-d. How are the data split into trials?

i. **Decisions**

- Trials come from the SDK `trials` table of the session's first plane (verified identical across planes).
- Selected trials = `(go | catch) & ~aborted & ~auto_rewarded & change_time.notna()`.
- Each trial is a **fixed window of [-2.25 s, +3.0 s] around `trials.change_time`**, cut into **21 bins of 250 ms** (7 complete 750 ms flash cycles: 3 pre-change flashes, the change flash, 3 post-change flashes). Trial length is therefore identical (21 bins) for every trial and session.
- Bin edges for all trials of a session are built as one flattened, strictly-increasing array; the code asserts strict monotonicity (i.e. non-overlapping trial windows) and skips the session otherwise.

ii. **Code snippets**

```python
BIN_SIZE = 0.25          # s, 1/3 of the 750 ms flash cycle, >= 2 ophys frames even at 11 Hz
OFF_START = -2.25        # s relative to the change: 3 flash cycles before the change
OFF_END = 3.00           # s relative to the change: the change flash + 3 flash cycles after
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 21

def select_trials(trials):
    m = (trials.go | trials.catch) & (~trials.aborted) & (~trials.auto_rewarded)
    m &= trials.change_time.notna()
    return trials[m]
```

```python
    sel = select_trials(trials)
    if len(sel) < 2:
        return {'session_id': session_id, 'skip': f'only {len(sel)} usable trials'}
    change_times = sel.change_time.values.astype(np.float64)
    edges = bin_edges_for_trials(change_times)
    edges_flat = edges.ravel()
    if np.any(np.diff(edges_flat) <= 0):
        return {'session_id': session_id, 'skip': 'overlapping trial windows'}
```

iii. **Justification (Step 5 decisions 2–4, Step 10 Check 5)**

Trial selection follows the decoder-task specification ("Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials"); the notes verify that this reproduces the whitepaper's 87.5 %/12.5 % go/catch split (measured 87.46/12.54). The fixed window is justified by a scan of the whole dataset: "the window [-2.25, +3.0] s is inside every trial's own `start_time`/`stop_time` (dataset minima 2.79 s before and 4.20 s after the change), so no trial is truncated and no neighbouring trial leaks in", and it aligns the 750 ms flash grid to t = 0 for every trial.

---

## 1-e. How are trials filtered based on quality controls?

i. **Decisions**

Trial/session-level curation, in order:

1. Sessions with a completely empty `eye_tracking` table are dropped (3 sessions, 917 trials) because pupil diameter is a required output.
2. Sessions whose running-speed or pupil stream is entirely NaN are dropped.
3. A trial is dropped if any of its 21 neural bins is empty (`~np.isfinite(neural).all()`), i.e. the 2p frames do not cover the whole window.
4. A trial is dropped if a behavioural sensor dropout longer than `MAX_SENSOR_GAP = 2 s` overlaps its window (9 trials in the whole dataset); shorter gaps are interpolated instead of discarding the trial.
5. A trial is dropped if any bin centre cannot be assigned to a stimulus flash (`image_names == 'none'`).
6. Sessions left with < 2 usable trials are skipped (format requirement).
7. Guard rails: a session is skipped if any binned stream leaves the min/max range of the raw stream it was averaged from (`within_range`), or if any selected trial lacks a hit/miss/FA/CR label.

No neuron-level filtering is applied.

ii. **Code snippets**

```python
    et = ds0.eye_tracking
    if len(et) == 0:
        return {'session_id': session_id, 'skip': 'no eye tracking'}
```

```python
    trial_ok = np.isfinite(neural).all(axis=(0, 2))
    ...
    run_binned, ok = fill_short_gaps(run_binned, centers, MAX_SENSOR_GAP)
    trial_ok &= ok
    ...
    pupil_binned, ok = fill_short_gaps(pupil_binned, centers, MAX_SENSOR_GAP)
    trial_ok &= ok
    ...
    trial_ok &= ~(image_names == 'none').any(axis=1)

    n_dropped = int((~trial_ok).sum())
    if trial_ok.sum() < 2:
        return {'session_id': session_id,
                'skip': f'only {int(trial_ok.sum())} trials with complete data'}
    if n_dropped:
        sel = sel[trial_ok]; neural = neural[:, trial_ok, :]
        run_binned = run_binned[trial_ok]; pupil_binned = pupil_binned[trial_ok]
        image_names = image_names[trial_ok]; image_change = image_change[trial_ok]
```

```python
    outcome = outcome_labels(sel)
    if (outcome < 0).any():
        return {'session_id': session_id, 'skip': 'trial without an outcome label'}
```

iii. **Justification (Step 7 anomalies, Step 9 accounting, Step 10 Check 5)**

The 2 s gap rule replaced an earlier rule that threw away whole sessions: "14 sessions were being dropped for 'empty pupil bins'. … the eye-tracking camera drops a handful of frames in many sessions (1–4 bins per session, longest gap 0.76 s). Dropping whole sessions for a sub-second sensor dropout is wrong". Dropped data are accounted for exactly: 29,444 − 276 = 29,168 neurons, 43,975 − 917 − 9 = 43,966 trials. Trial-outcome counts per session were cross-checked against the independent `behavior_session_table` counts (identical in 169/171 sessions; the 2 differences are exactly the 9 dropped trials).

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. **Decisions**

`ds.dff_traces.dff` (one dF/F trace per cell, produced by the Allen pipeline) together with `ds.ophys_timestamps` for the time base. The Allen pipeline's `events` / `filtered_events` (the signal used by the Vip–Sst reference paper) remain selectable via `--signal`, but the default and the delivered dataset use `dff`.

ii. **Code snippets**

```python
NEURAL_SIGNAL = 'dff'  # 'dff' (default, see CONVERSION_NOTES Step 5), 'events' or 'filtered_events'
...
        if signal_name == 'dff':
            traces = np.vstack(ds.dff_traces.dff.values).astype(np.float64)
            index = ds.dff_traces.index.values
        else:
            traces = np.vstack(ds.events[signal_name].values).astype(np.float64)
            index = ds.events.index.values
        assert traces.shape[1] == len(ts), 'neural trace length != ophys_timestamps'
```

iii. **Justification (Step 1 notes, Step 5 decision 7, Step 10 Check 3)**

dF/F does not have to be computed: "the NWB files ship `dff_traces` and `events` already produced by the Allen pipeline (dewarping → motion correction → segmentation → ROI filtering → demixing → neuropil subtraction → dF/F → event detection)". The switch from the paper's `events` to `dff` is documented as a deliberate, measured deviation: at 250 ms resolution events are "84–99 % of frames exactly zero", and a 14-session head-to-head gave image-identity balanced accuracy 0.258 (events) vs. 0.511 (dF/F). "dF/F is the primary neural product of the same Allen pipeline (whitepaper 'DF/F CALCULATION') … `--signal events` reproduces the paper's choice exactly."

---

## 2-b. How is the `neural` data processed?

i. **Decisions**

- Per plane: the per-cell dF/F traces are stacked into an `(n_cells, T)` matrix and **averaged within each 250 ms trial bin**, with each ophys frame assigned to a bin by its `ophys_timestamps` value. The binning is fully vectorised (one `searchsorted` + one `add.reduceat` for all trials × bins of a session).
- Planes of the same session are concatenated along the neuron axis in `ophys_experiment_id` order; the per-neuron `targeted_structure` is recorded in parallel.
- Output is cast to `float32`; no z-scoring, smoothing, baseline subtraction or normalisation is applied (z-scoring was tested and rejected).

ii. **Code snippets**

```python
def bin_mean(values, timestamps, edges_flat):
    one_d = values.ndim == 1
    v = values[None, :] if one_d else values
    idx = np.searchsorted(timestamps, edges_flat, side='left')
    counts = np.diff(idx)
    # np.add.reduceat's last segment always runs to the end of the array, so the array is truncated
    # at the final bin edge; that makes the last segment stop at that edge like every other one.
    stop = max(int(idx[-1]), 1)
    starts = np.minimum(idx[:-1], stop - 1)
    sums = np.add.reduceat(v[:, :stop], starts, axis=1)
    means = sums / np.maximum(counts, 1)[None, :]
    empty = counts <= 0
    if empty.any():
        means[:, empty] = np.nan
    return (means[0] if one_d else means), counts
```

```python
        means, counts = bin_mean(traces, ts - neural_lag, edges_flat)
        block = means[:, keep].reshape(traces.shape[0], len(sel), NBINS)
        if not within_range(block, traces):
            return {'session_id': session_id, 'skip': 'binned neural outside raw trace range'}
        neural_blocks.append(block)
    neural = np.concatenate(neural_blocks, axis=0)
    ...
    'neural': neural.astype(np.float32),
```

iii. **Justification (Step 5 decisions 5–6, Step 6, Step 12 Check 2)**

Bin-averaging is the minimal operation that puts the 31 Hz and the 11 Hz rigs on one common bin size, which the target format requires ("Time bins should be the same size for all trials and sessions"). No further processing is applied because "the dF/F traces are already processed by the Allen pipeline"; a z-scored variant was measured and rejected ("no gain, less faithful"). The `within_range()` guard ("a bin mean can never leave the range of the samples it averages") was added after a `reduceat` bug corrupted the last bin of each session, and it is now run on every stream of every session.

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. **Decisions** — No neuron-level filtering at all. Every cell in the released NWB is kept (29,444 cells across 202 active experiments; 29,168 in the converted data after the 3 dropped sessions). Sessions with as few as 6 neurons are kept. The only neural-related exclusion is at the trial level (a trial whose window is not fully covered by 2p frames is dropped) and the `within_range` session guard.

ii. **Code snippets**

```python
        'neuron_selection': (
            'all cells in the released NWB files (cell_specimen_table.valid_roi is True for every cell; '
            'ROI filtering, demixing and neuropil correction were applied by the Allen pipeline)'),
```

```python
    trial_ok = np.isfinite(neural).all(axis=(0, 2))
```

iii. **Justification (Step 1, Step 3 curation, Step 10 Check 5)**

"ROI filtering (motion border, union/duplicate ROIs, apical dendrites, too small / narrow / dim) was applied by the Allen pipeline before release; the NWB files only contain valid cells (`valid_roi` True for all 29,444 cells). Neither reference paper applies further neuron filtering, so I apply none." The agent verified `valid_roi` is True for every cell in all 202 active experiments. "Sessions with very few neurons (min 6) are kept: the reference papers apply no cell-count criterion and the decoder handles small populations."

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. **Decisions**

- The alignment event is **`trials.change_time`** — the onset of the changed image on go trials and of the sham change on catch trials. `metadata['temporal_alignment_event']` documents this, with `off_start = -2.25` and `off_end = 3.0`.
- The bin grid for a trial is `change_time + (-2.25 + 0.25·k)`, k = 0…21, on the shared session clock; every 2p frame is assigned to a bin by its `ophys_timestamps` value, and every other stream is averaged over the *same* absolute-time edges, so all streams are aligned by construction.
- Bin 9 (centre +0.125 s) and bin 10 (centre +0.375 s) cover the changed image flash.

ii. **Code snippets**

```python
def bin_edges_for_trials(change_times):
    offsets = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    edges = change_times[:, None] + offsets[None, :]
    return edges
...
    change_times = sel.change_time.values.astype(np.float64)
    edges = bin_edges_for_trials(change_times)
    edges_flat = edges.ravel()
    centers_flat = (edges_flat[:-1] + edges_flat[1:]) / 2.0
    keep = keep_index(len(sel))
    centers = centers_flat[keep].reshape(len(sel), NBINS)
```

```python
        'temporal_alignment_event': (
            'stimulus change time (trials.change_time): the onset of the changed image on go trials and '
            'of the sham change on catch trials; equal to the start_time of that stimulus flash'),
        'off_start': OFF_START,
        'off_end': OFF_END,
```

iii. **Justification (Step 4, Step 5 decisions 3 and 6, Step 10 Check 2)**

"Alignment event = `change_time` … This is the only event common to every included trial, it is exactly a flash onset, and the reference paper aligns its analyses to image presentations/changes." The agent verified on all experiments that `change_time` is "**exactly equal** to the `start_time` of the corresponding `stimulus_presentations` flash (max abs difference 0.0 s)", so the 750 ms flash grid is phase-locked to t = 0. On "temporally align based on ophys timestamp": "each ophys frame is assigned to a bin by its `ophys_timestamps` value, and every other stream … is resampled onto those same absolute-time bins." A brute-force re-computation of binned dF/F from `dff_traces` + `ophys_timestamps` (`cache/sanity_checks.py`) passed `np.allclose`, and grid shifts of +125/+250 ms were tested and gave no improvement (evidence against a residual misalignment).

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Decisions**

- **Yes, explicit rebinning**: the native ophys sampling (32.3 ms at 31 Hz on Scientifica rigs, 93.2 ms at 11 Hz on the Multiscope) is re-binned to a uniform **250 ms** bin, 21 bins per trial, `metadata['time_bin_size'] = 250.0`.
- All other streams (running 60 Hz, eye 30 Hz, stimulus flashes) are averaged/assigned onto the same 250 ms grid.

ii. **Code snippets**

```python
BIN_SIZE = 0.25          # s, 1/3 of the 750 ms flash cycle, >= 2 ophys frames even at 11 Hz
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 21
...
        'time_bin_size': BIN_SIZE * 1000.0,
        'n_timepoints_per_trial': NBINS,
```

iii. **Justification (Step 5 decision 5, Step 10 Check 3)**

"**Bin size = 250 ms** (21 bins/trial). It divides the 750 ms flash cycle exactly into 3 bins (image / gray / gray) and is a whole multiple of the 250 ms image duration; and it is ≥ 2 ophys frames even at the 11 Hz Multiscope rate (93.23 ms), so no bin is ever empty for any rig. A bin size equal to the native ophys frame period cannot be used because the format requires one bin size for all sessions while the rigs sample at 31 Hz and 11 Hz." Against the references: "the paper bins neural activity by image presentation interval (750 ms) and uses the first 400 ms after each presentation for decoding. There is no canonical fixed-width bin in the references"; 250 ms is described as "the largest bin that still resolves image vs. gray within a flash cycle and the smallest that is never empty at the 11 Hz Multiscope frame rate".

---

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. **Decisions**

`ds.stimulus_presentations`, restricted to `stimulus_block_name == 'change_detection_behavior'`, using the `start_time` and `image_name` of every flash (including rows whose `image_name` is `'omitted'`). The trials table's `initial_image_name` / `change_image_name` are **not** used.

ii. **Code snippets**

```python
STIM_BLOCK = 'change_detection_behavior'
...
    sp = ds0.stimulus_presentations
    spa = sp[sp.stimulus_block_name == STIM_BLOCK]
    flash_start = spa.start_time.values.astype(np.float64)
    flash_name = spa.image_name.values.astype(object)
```

iii. **Justification (Step 1, Step 5 mapping table and decision 8)**

`stimulus_presentations` is "one row per flash: `start_time`, `end_time`, `image_name`, `omitted`, `is_change`, `is_sham_change`, `stimulus_block_name`", and "Tutorials filter to `stimulus_block_name == 'change_detection_behavior'`". Using the actual flash table (rather than the two per-trial image-name columns) is what makes omitted flashes and the true flash sequence representable.

---

## 3-b. What processing is involved in computing `output` *Image identity*?

i. **Decisions**

- Every bin centre is assigned the image of the **750 ms flash interval** that contains it: the last flash that started before the bin centre, provided the centre is within 750 ms (+50 ms tolerance) of that flash start. Bin centres that fall outside any flash interval are labelled `'none'` and cause the trial to be dropped.
- The grey inter-stimulus period therefore inherits the identity of the flash that opened its interval.
- Labels are mapped to a **global** integer code built from the union of image names over all sessions: 16 natural images (image sets A and B) sorted alphabetically, with `'omitted'` appended as a 17th category.

ii. **Code snippets**

```python
    # the flash interval is [start_time, start_time + 750 ms): assign every bin centre to the last
    # flash that started before it (the paper's "image presentation interval" convention).
    j = np.searchsorted(flash_start, centers_flat, side='right') - 1
    valid = (j >= 0) & (j < len(flash_start))
    j_clipped = np.clip(j, 0, len(flash_start) - 1)
    within = valid & (centers_flat - flash_start[j_clipped] < FLASH_CYCLE + 0.05)
    image_name_flat = np.where(within, flash_name[j_clipped], 'none')
    image_names = image_name_flat[keep].reshape(len(sel), NBINS)
    trial_ok &= ~(image_names == 'none').any(axis=1)
```

```python
    image_values = sorted({n for r in results for n in np.unique(r['image_names']) if n != 'omitted'})
    image_values = image_values + ['omitted']
    image_to_idx = {n: i for i, n in enumerate(image_values)}
    ...
        img = np.vectorize(image_to_idx.get)(r['image_names']).astype(np.int64)
```

iii. **Justification (Step 5 decisions 8 and 9, Step 10 Check 2)**

"**Image identity is labelled per 750 ms flash interval**, exactly the paper's 'image presentation interval' convention ('the 750 ms interval beginning with each image presentation; for omissions the 750 ms following the time of the omission'). The gray period therefore carries the identity of the image that was flashed at the start of its interval, which also matches the ~100–300 ms lag of calcium events. `omitted` is kept as its own category, since an omission is a distinct stimulus condition with a known neural signature in this dataset." Global codes are used "so that a label means the same physical image in every session". Verified: identity is constant within every 750 ms interval in 100 % of trials, differs across t = 0 on 100 % of go trials and 0 % of catch trials, and `omitted` never appears at the change flash or the flash before it (reproducing a whitepaper claim). A 16-class variant folding `omitted` into the repeated image was tested and rejected as "less faithful".

---

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. **Decisions** — Image identity is computed on exactly the same bin-centre array (`centers_flat`, derived from the same `edges_flat`) that defines the neural bins, and indexed with the same `keep` index and the same `trial_ok` mask, so the image label of bin *k* describes the same 250 ms window as the neural activity in bin *k*. No additional shift is applied.

ii. **Code snippets**

```python
    centers_flat = (edges_flat[:-1] + edges_flat[1:]) / 2.0
    keep = keep_index(len(sel))
    centers = centers_flat[keep].reshape(len(sel), NBINS)
    ...
    j = np.searchsorted(flash_start, centers_flat, side='right') - 1
    image_names = image_name_flat[keep].reshape(len(sel), NBINS)
```

iii. **Justification** — "every stream is averaged over the same absolute-time bins, anchored to `trials.change_time`; 2p frames assigned by `ophys_timestamps`" (Step 10 Check 3). Alignment was verified both by the `--show-processing` panel "image identity per bin (line) vs. actual flashes (shaded, labelled)" and by the independent check "`image_identity` vs. the last `stimulus_presentations` row starting before each bin centre — PASS".

---

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. **Decisions** — `trials.change_time` (the alignment event) and `trials.is_change` (True on go trials, False on catch/sham-change trials), both from the SDK trials table.

ii. **Code snippets**

```python
    is_change = sel.is_change.values.astype(bool)
    rel_centers = centers - change_times[:, None]
    image_change = ((rel_centers >= 0) & (rel_centers < CHANGE_WINDOW)
                    & is_change[:, None]).astype(np.int64)
```

iii. **Justification (Step 4, Step 5 mapping table)** — The agent verified that `change_time` "is never NaN for go/catch trials; it is exactly equal to the `start_time` of the corresponding `stimulus_presentations` flash …, which is `is_change` for go trials and `is_sham_change` for catch trials". Catch trials are sham changes where "the image identity does not change", so `is_change` is the correct gate.

---

## 4-b. What processing is involved in computing `output` *Image change*?

i. **Decisions** — A binary time series per trial: 1 in the bins whose centre lies in `[change_time, change_time + 0.5 s)` **on change (go) trials only**, 0 everywhere else and 0 for the whole trial on catch trials. Given the 21-bin grid this is always exactly bins 9 and 10. No smoothing or other processing.

ii. **Code snippets**

```python
CHANGE_WINDOW = 0.5      # s after the change that image_change marks as 1 (see notes Step 5/12)
...
    image_change = ((rel_centers >= 0) & (rel_centers < CHANGE_WINDOW)
                    & is_change[:, None]).astype(np.int64)
```

iii. **Justification (Step 5 decision 8b)** — "'Right after a change' has to be given a width; 500 ms is the representable window closest to the 400 ms post-presentation window the reference paper decodes in, it sits inside the behavioural response window (150–750 ms) … Catch trials have no change of identity, so they are 0 throughout." The notes also state it was chosen among three tested widths by decoding performance (Step 12: 250 ms → 0.704, 500 ms → 0.729, 750 ms → 0.682 validation balanced accuracy on a 14-session subset).

---

## 4-c. How is `output` *Image change* thresholded into categories?

i. **Decisions** — The variable is already binary; the only "threshold" is the width of the post-change window (`CHANGE_WINDOW = 0.5 s`, i.e. 2 of 21 bins) and the `is_change` gate. `output_values[1] = ['no_change', 'change']`. The resulting marginal is 8.3 % change / 91.7 % no-change (= go fraction 0.8746 × 2/21).

ii. **Code snippets**

```python
        'output_values': [image_values, ['no_change', 'change'],
                          QUANTILE_NAMES, QUANTILE_NAMES, OUTCOMES],
...
        'image_change_definition': (
            f'1 in the {CHANGE_WINDOW * 1000:.0f} ms following the change of image identity on go '
            'trials (the two bins covering the presentation of the new image and the start of the '
            'following gray period), 0 elsewhere and everywhere on catch trials (sham change, the '
            'image identity does not change)'),
```

iii. **Justification (Step 10 Check 2)** — Verified whole-dataset: "`image_change` is 1 only in bins 9–10 and only on go trials; its mean over trials is 0.8746 = the go fraction". The window width choice is justified in 4-b.

---

## 4-d. How is `output` *Image change* aligned with the neural data?

i. **Decisions** — Computed directly from the per-trial bin centres relative to `change_time` (`rel_centers = centers - change_times[:, None]`), i.e. the same grid used for the neural bins, then masked by the same `trial_ok`. Because trials are aligned to the change, the indicator is at a fixed position (bins 9–10) in every trial.

ii. **Code snippets**

```python
    rel_centers = centers - change_times[:, None]
    image_change = ((rel_centers >= 0) & (rel_centers < CHANGE_WINDOW)
                    & is_change[:, None]).astype(np.int64)
    ...
        image_change = image_change[trial_ok]
```

iii. **Justification** — Same shared-bin-grid argument as 2-d/3-c; independently re-derived from `trials.is_change` and `change_time` in `cache/sanity_checks.py` ("PASS") and plotted per trial in the `--show-processing` figures (panel 7, with the change time and the 500 ms boundary marked).

---

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. **Decisions** — `ds0.running_speed`, using its `timestamps` and `speed` columns (the SDK's 60 Hz, 10 Hz-low-pass-filtered wheel speed in cm/s, which may be negative). The running stream of the first plane is used for the whole session (all planes of a session share one behaviour stream).

ii. **Code snippets**

```python
    rs = ds0.running_speed
    run_ts = rs.timestamps.values.astype(np.float64)
    run_v = interpolate_nans(rs.speed.values)
    if run_v is None:
        return {'session_id': session_id, 'skip': 'no running speed'}
```

iii. **Justification (Step 1, Step 3)** — "`ds.running_speed` (`timestamps`, `speed`): Filtered running speed (cm/s) at 60 Hz (whitepaper: 10 Hz low-pass Butterworth of the wrap/transient-corrected encoder signal)"; the agent's scan found `running_speed` contains no NaNs in any of the 202 active experiments.

---

## 5-b. What processing is involved in computing `output` *Running speed*?

i. **Decisions**

1. Any NaN in the raw speed is linearly interpolated (defensive; none occur in practice).
2. The speed is **averaged within each 250 ms trial bin** using the same vectorised `bin_mean` as the neural data.
3. Empty bins (camera/encoder dropouts) are filled by linear interpolation in time across the session's bin centres, unless the gap exceeds 2 s, in which case the trial is dropped.
4. A `within_range` guard asserts the binned values stay within the raw range.
5. The binned continuous values are then discretized (see 5-c).

ii. **Code snippets**

```python
    run_binned, _ = bin_mean(run_v, run_ts, edges_flat)
    run_binned = run_binned[keep].reshape(len(sel), NBINS)
    run_binned, ok = fill_short_gaps(run_binned, centers, MAX_SENSOR_GAP)
    trial_ok &= ok
    if not within_range(run_binned, run_v):
        return {'session_id': session_id, 'skip': 'binned running speed outside raw range'}
```

```python
def fill_short_gaps(binned, centers, max_gap):
    flat = binned.ravel().astype(np.float64); ct = centers.ravel()
    good = np.isfinite(flat)
    ...
    filled = np.interp(ct, ct[good], flat[good])
    prev_t = np.maximum.accumulate(np.where(good, ct, -np.inf))
    next_t = np.minimum.accumulate(np.where(good, ct, np.inf)[::-1])[::-1]
    ok = good | ((ct - prev_t <= max_gap) & (next_t - ct <= max_gap))
    trial_ok = ok.reshape(binned.shape).all(axis=1)
    return filled.reshape(binned.shape), trial_ok
```

iii. **Justification (Step 5, Step 7 anomalies)** — Bin-averaging is the same operation applied to every stream so that all streams live on one grid; the interpolation rule for short gaps was introduced because "the eye-tracking camera drops a handful of frames in many sessions … Dropping whole sessions for a sub-second sensor dropout is wrong", with the 2 s cap so that "interpolation would [not] be fiction rather than a fix". `within_range` is the permanent guard added after the `reduceat` bug.

---

## 5-c. How is `output` *Running speed* thresholded into categories?

i. **Decisions** — Five equal-percentile bins (quintiles), with edges at the 20/40/60/80th percentiles computed **per session**, over exactly the timepoints that are exported (after bad trials are removed). Labels 0–4, named `['0-20%', …, '80-100%']`; the per-session edges in cm/s are stored in `metadata['session_info'][i]['running_speed_quintile_edges_cm_s']`.

ii. **Code snippets**

```python
def quantile_bins(values, nq=NQUANTILES):
    """Discretize into `nq` equal-percentile bins. Returns (labels, edges)."""
    edges = np.percentile(values, np.linspace(0, 100, nq + 1)[1:-1])
    labels = np.searchsorted(edges, values, side='right').astype(np.int64)
    return np.clip(labels, 0, nq - 1), edges
```

```python
    # ---- discretize running / pupil (per session, over exactly the exported timepoints) --------
    run_bin, run_edges = quantile_bins(run_binned.ravel())
    pupil_bin, pupil_edges = quantile_bins(pupil_binned.ravel())
    run_bin = run_bin.reshape(len(sel), NBINS)
```

iii. **Justification (Step 5 decision 10)** — "**Quintile bins for running speed and pupil are computed per session** over exactly the timepoints that are written out. Pupil area is in camera pixels² and is not comparable across sessions/mice (eye size, camera distance), and running speed distributions differ strongly across mice, so per-session percentiles are the only way the 5 labels carry the same meaning in every session; it also guarantees the 20/20/20/20/20 class balance that 'five equal percentile bins' asks for." Verified: every session's quintiles contain 20.0 % of timepoints.

---

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. **Decisions** — The 60 Hz speed samples are averaged over exactly the same absolute-time bin edges (`edges_flat`) as the neural frames, and the same `keep` index and `trial_ok` mask are applied, so bin *k* of the running output covers the same 250 ms of session time as bin *k* of `neural`.

ii. **Code snippets**

```python
    run_binned, _ = bin_mean(run_v, run_ts, edges_flat)
    run_binned = run_binned[keep].reshape(len(sel), NBINS)
    ...
        run_binned = run_binned[trial_ok]
```

iii. **Justification (Step 1, Step 10 Check 3)** — "All data streams (2p, stimulus, running, eye) are already synchronized onto one session clock by the SDK (whitepaper 'DATA SYNCHRONIZATION'), so alignment only requires resampling onto a common time base", and the SDK tutorials align every stream by `timestamps` on that clock. Checked independently ("`running_speed_bin` vs. brute-force binned `running_speed` + the stored quintile edges — PASS") and visually (raw 60 Hz trace vs. binned step plot with the bin edges drawn).

---

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. **Decisions** — `ds0.eye_tracking`, using `timestamps` and `pupil_area`. Pupil **diameter** is derived as `2·sqrt(pupil_area/π)`. `pupil_width`/`pupil_height` are deliberately not used. `likely_blink` is used implicitly: the SDK already sets `pupil_area` to NaN on blink/outlier frames.

ii. **Code snippets**

```python
    et = ds0.eye_tracking
    if len(et) == 0:
        return {'session_id': session_id, 'skip': 'no eye tracking'}
    eye_ts = et.timestamps.values.astype(np.float64)
    # whitepaper: pupil_area is the area of a circle whose diameter is the ellipse major axis, and is
    # NaN on likely_blink frames -> diameter in the same pixel units, blinks interpolated over.
    pupil_area = et.pupil_area.values.astype(np.float64)
    pupil_diam_raw = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. **Justification (Step 4 discrepancy table, Step 5 mapping)** — "whitepaper: pupil area = area of circle whose diameter is the ellipse major axis … diameter = 2*sqrt(pupil_area/pi) — a monotone function of `pupil_area`, so percentile bins are identical either way, and it uses the blink-cleaned column." (`pupil_width`/`pupil_height` are the semi-axes the SDK squares to build `pupil_area = π·max(w,h)²`, so this expression recovers the actual diameter `2·max(w,h)`.) The agent measured that `pupil_area` is NaN on `likely_blink` frames in "mean 3.5 % of frames, max 29.6 %", and that 3 experiments have an entirely empty eye-tracking table.

---

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. **Decisions** — Identical pipeline to running speed: (1) blink/outlier NaNs linearly interpolated in time within the session (edges held constant); (2) bin-averaged onto the 250 ms trial grid; (3) empty bins filled for gaps ≤ 2 s, trials with longer gaps dropped; (4) `within_range` guard; (5) per-session quintile discretization.

ii. **Code snippets**

```python
def interpolate_nans(x):
    """Linear interpolation over NaNs in a 1-D array; edges are held constant. Returns None if all NaN."""
    x = np.asarray(x, dtype=np.float64)
    good = np.isfinite(x)
    if not good.any(): return None
    if good.all(): return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[good], x[good])
```

```python
    pupil_diam = interpolate_nans(pupil_diam_raw)
    if pupil_diam is None:
        return {'session_id': session_id, 'skip': 'pupil all NaN'}
    pupil_binned, _ = bin_mean(pupil_diam, eye_ts, edges_flat)
    pupil_binned = pupil_binned[keep].reshape(len(sel), NBINS)
    pupil_binned, ok = fill_short_gaps(pupil_binned, centers, MAX_SENSOR_GAP)
    trial_ok &= ok
    if not within_range(pupil_binned, pupil_diam):
        return {'session_id': session_id, 'skip': 'binned pupil outside raw range'}
```

iii. **Justification (Step 5 decision 11)** — "**Blink NaNs in pupil are linearly interpolated** over time within the session (edges held constant) before binning, rather than dropping trials: blinks are short (3.5 % of frames) and dropping them would bias the trial set toward calm periods." The `--show-processing` panel plots raw (with blink NaNs), interpolated and binned pupil together to make the interpolation auditable.

---

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. **Decisions** — Same as running speed: five equal-percentile bins with 20/40/60/80th-percentile edges computed **per session** over exactly the exported timepoints; labels 0–4; edges stored per session in `metadata['session_info'][i]['pupil_diameter_quintile_edges_px']`. Realised marginal is 0.200 in every class.

ii. **Code snippets**

```python
    pupil_bin, pupil_edges = quantile_bins(pupil_binned.ravel())
    pupil_bin = pupil_bin.reshape(len(sel), NBINS)
...
        'output_discretization': (
            'running speed and pupil diameter are discretized into five equal-percentile bins computed '
            'per session over exactly the exported timepoints; pupil diameter = 2*sqrt(pupil_area/pi) '
            'with blink NaNs linearly interpolated over time'),
```

iii. **Justification (Step 5 decision 10)** — "Pupil area is in camera pixels² and is not comparable across sessions/mice (eye size, camera distance) … per-session percentiles are the only way the 5 labels carry the same meaning in every session". The notes also address the leakage question raised by per-session edges (Step 12 Check 3): the edges are a property of the labels shared by a session's train and test trials and "cannot leak neural information".

---

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. **Decisions** — The 30 Hz eye-tracking samples are averaged over the same `edges_flat` bin edges as the neural data, with the same `keep` index and `trial_ok` mask; no shift is applied.

ii. **Code snippets**

```python
    pupil_binned, _ = bin_mean(pupil_diam, eye_ts, edges_flat)
    pupil_binned = pupil_binned[keep].reshape(len(sel), NBINS)
    ...
        pupil_binned = pupil_binned[trial_ok]
```

iii. **Justification** — Same shared-clock/shared-bin-grid argument as running speed; verified independently ("`pupil_diameter_bin` vs. brute-force binned 2*sqrt(pupil_area/pi) + stored edges — PASS") and visually in `processing_<session>.png`.

---

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. **Decisions** — The four mutually exclusive boolean columns of the SDK trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, in that fixed order. A trial with none of the four set gets label −1, which makes the converter skip the whole session (never triggered).

ii. **Code snippets**

```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']

def outcome_labels(sel):
    """Integer trial-outcome label (index into OUTCOMES) for each selected trial."""
    lab = np.full(len(sel), -1, dtype=np.int64)
    for i, name in enumerate(OUTCOMES):
        lab[sel[name].values.astype(bool)] = i
    return lab
```

iii. **Justification (Step 2 data-quality facts, Step 3)** — The agent verified across all 202 active experiments that "`go`, `catch`, `aborted` and `auto_rewarded` are mutually exclusive … and hit+miss+false_alarm+correct_reject == n(go|catch) exactly", i.e. these four labels partition exactly the selected trial set. They are the whitepaper's canonical outcomes for the change-detection task (response window 150–750 ms after the change).

---

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. **Decisions** — One integer label (0–3) per trial, **broadcast over all 21 time bins** so that the output array is `(5, 21)` like the other outputs; `output_values[4] = ['hit','miss','false_alarm','correct_reject']`. Resulting distribution: hit 0.317, miss 0.558, false_alarm 0.019, correct_reject 0.106.

ii. **Code snippets**

```python
            out = np.stack([
                img[t],
                r['image_change'][t],
                r['run_bin'][t],
                r['pupil_bin'][t],
                np.full(NBINS, r['outcome'][t], dtype=np.int64),
            ], axis=0)
            output_trials.append(out)
```

iii. **Justification (Step 5 decision 12, Step 9)** — "**Trial outcome is broadcast over time** because the format requires a single (d_output, T) array per trial" (the decoder task lists it as "static per-trial"). The per-session hit/miss/FA/CR counts were cross-checked against the independent Allen `behavior_session_table` counts and match in 169/171 sessions, the two exceptions being exactly the 9 trials dropped for sensor dropout. Step 12 notes that the 9 pre-change bins carry a label that is not yet determined at that time, and shows per-timepoint accuracy rising from ~0.35 before the change to ~0.42 after it.

---

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. **Decisions**

| Problem | Handling |
|---|---|
| Eye-tracking table entirely empty (3 sessions) | session dropped (pupil is a required output) |
| Running or pupil stream all-NaN | session dropped |
| Blink/outlier NaNs in `pupil_area` (3.5 % of frames, up to 29.6 %) | linearly interpolated in time before binning |
| Short behavioural sensor dropouts (empty bins, ≤ 2 s) | filled by linear interpolation across bin centres |
| Dropouts > 2 s inside a trial window | that trial dropped (9 trials dataset-wide) |
| Bin with no 2p frame | trial dropped (`np.isfinite(neural).all`) |
| Bin centre not inside any flash interval | labelled `'none'` → trial dropped |
| Session left with < 2 usable trials | session skipped |
| Binned value outside the raw stream's range | session skipped (`within_range` guard — catches binning/alignment bugs) |
| Overlapping trial windows | session skipped |
| Plotting failure | caught, conversion continues |
| Trial with no outcome flag | session skipped |

All drops are counted and printed (`n_dropped_trials`, `skipped sessions`), and the notes reconcile the totals exactly (29,444 − 276 = 29,168 neurons; 43,975 − 917 − 9 = 43,966 trials). No NaN reaches the output.

ii. **Code snippets**

```python
MAX_SENSOR_GAP = 2.0      # s; behavioural bins with no sample are interpolated across gaps up to
                          # this length, trials with longer sensor dropouts are dropped
```

```python
def within_range(binned, raw, tol=1e-6):
    """A bin mean can never leave the range of the samples it averages — cheap alignment/binning check."""
    return (np.nanmin(binned) >= np.nanmin(raw) - tol) and (np.nanmax(binned) <= np.nanmax(raw) + tol)
```

```python
    if show_processing:
        try:
            plot_processing(...)
        except Exception as e:      # plotting must never break the conversion
            print(f'  [warn] plotting failed for session {session_id}: {e!r}')
```

iii. **Justification (Step 7 anomaly 2, Step 10 Check 5)** — The 2 s rule is a deliberate compromise found by investigating an earlier over-aggressive rule: "14 sessions were being dropped for 'empty pupil bins'. Investigation showed the eye-tracking camera drops a handful of frames in many sessions (1–4 bins per session, longest gap 0.76 s). Dropping whole sessions for a sub-second sensor dropout is wrong, so empty behavioural bins are now filled by linear interpolation in time and only trials with a dropout longer than 2 s are dropped (9 trials in the entire dataset)." Interpolating blinks rather than dropping trials is justified because dropping "would bias the trial set toward calm periods".

---

## 9-a. What are the most time-consuming steps of the code?

i. **Decisions / findings**

- The agent instrumented the conversion (`timing['load']`, `timing['neural']`, `timing['behavior']`, `timing['stimulus']`, per-session `total_time`, running ETA) and found that **reading the NWB via `cache.get_behavior_ophys_experiment()` dominates**: ~3.0–3.5 s per experiment out of ~3.8–4.1 s per session, i.e. ≈ 98 % of the per-session cost; binning is ~0.05–0.1 s.
- Remedy: sessions are converted in parallel worker processes (`--nproc`, default 8), taking the full conversion from ~700 s serial to **69 s** wall clock — far inside the 15-minute budget.
- Downstream, decoder training (~3 min on GPU) and writing the 707 MB pickle are the other notable costs.

ii. **Code snippets**

```python
    t0 = time.time()
    datasets = [cache.get_behavior_ophys_experiment(int(o)) for o in oeids]
    timing['load'] = time.time() - t0
```

```python
                print(f'[{k + 1}/{len(jobs)}] session {r["session_id"]}: '
                      + (... f'{r["total_time"]:.1f}s (load {r["timing"]["load"]:.1f}s, '
                         f'neural {r["timing"]["neural"]:.1f}s)')
                      + f' | elapsed {el:.0f}s, eta {el / (k + 1) * (len(jobs) - k - 1):.0f}s', flush=True)
```

iii. **Justification (Step 6, Step 7 run-time table)** — "Loading an NWB file is ~3.5 s and is the only real cost (98 % of the per-session time)"; measured speed-ups: "vectorised binning … 0.05 s/session vs. ~1 s/session looped" and "16-way multiprocessing over sessions: 700 s → 69 s wall clock". (The notes say 16 workers while the script's `--nproc` default is 8; the logged run took 69 s either way.)

---

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. **Decisions / findings**

- The agent deliberately vectorised what would otherwise be the expensive loops: there is **no Python loop over trials, bins or neurons** in the binning path — one `np.searchsorted` of all trials' edges into a stream's timestamps plus one `np.add.reduceat` produces every bin of every trial at once, for all neurons simultaneously.
- Image identity, image change, quantile labelling and the trial-validity masks are likewise computed as whole-array operations over all trials.
- The loops that remain are small or unavoidable: over planes of a session (1–8), over sessions (parallelised), and the per-trial loop in `assemble()` that slices `neural[:, t, :]` and stacks the five output rows for ~44 k trials — this one is required by the target format (a list of per-trial arrays) but could be made cheaper (e.g. `np.moveaxis` + `list(...)` instead of `ascontiguousarray` per trial). `np.vectorize(image_to_idx.get)` over the `(n_trials, 21)` string array is a Python-level loop in disguise and could be replaced by `np.searchsorted` on a sorted name array.

ii. **Code snippets**

```python
    idx = np.searchsorted(timestamps, edges_flat, side='left')
    ...
    sums = np.add.reduceat(v[:, :stop], starts, axis=1)
```

```python
        for t in range(n_trials):                      # per-trial loop kept: format requires a list
            neural_trials.append(np.ascontiguousarray(r['neural'][:, t, :]))
            input_trials.append(np.zeros((0, NBINS), dtype=np.float32))
            out = np.stack([...], axis=0)
```

```python
        img = np.vectorize(image_to_idx.get)(r['image_names']).astype(np.int64)   # Python-level loop
```

iii. **Justification (Step 6)** — "Code inefficiencies identified: Naively looping over trials (×257) and neurons (×666) per session would dominate the runtime … Code speedups added: One `searchsorted` + one `reduceat` per data stream per session (no Python loop over trials or bins)." Since loading dominates, the remaining small loops were left alone.

---

## 9-c. What processing does the code repeat multiple times?

i. **Decisions / findings**

- By design, each experiment's NWB is read exactly once, and per-session products are computed once and reused for discretization, assembly and metadata.
- Genuine repetitions are minor: (1) `get_cache()` is re-opened inside **every** worker job (once per session, 174 times) rather than once per worker process; (2) for multi-plane Multiscope sessions all planes are loaded although only plane 0's behaviour/trials/stimulus tables are used (the neural traces of the other planes are needed, so the load itself is not wasted); (3) `within_range()` makes an extra full pass (`nanmin`/`nanmax`) over each raw trace; (4) the sample and full conversions were re-run several times during the iteration protocol (by instruction).

ii. **Code snippets**

```python
def convert_session(args):
    session_id, oeids, show_processing, signal_name, neural_lag = args
    t_start = time.time()
    cache = get_cache()          # re-opened for every session job
```

```python
    trials = ds0.trials          # only plane 0's behaviour tables are used
    rs = ds0.running_speed
    et = ds0.eye_tracking
    sp = ds0.stimulus_presentations
```

iii. **Justification (Step 6, Step 10)** — The notes state the design intent ("Binning is done directly on the trace instead of a cumulative sum, so memory stays at one experiment per worker") and verify that "the 7 planes of a session share one `trials` table and one behaviour stream; verified that the trials tables are identical across planes before merging", which is why only plane 0's tables are read. The repeated cache opening is not discussed in the notes; it is cheap relative to the 3.5 s NWB read.

---

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. **Decisions / findings**

- `cell_ids` are collected for every neuron of every session and never used or written to the output.
- `bin_mean` always returns per-bin sample `counts`; for the behavioural streams they are discarded (`run_binned, _ = ...`).
- The flattened edge construction deliberately creates `n_trials × (NBINS+1) − 1` segments and then throws away the inter-trial "gap" segment of every trial (`keep_index`) — a small amount of computation traded for full vectorisation.
- Continuous `run_binned` / `pupil_binned` arrays are kept in the per-session result (needed for the quantile edges and the plots) and pickled back from the worker, but are not part of the saved dataset; likewise `change_times`, `trial_ids`, `timing`, `imaging_depths`.
- The `--show-processing` path re-reads and re-stacks the dF/F traces to draw the raw-trace panel (only for 2 sessions, only when requested).
- More substantively, the trial window keeps only [-2.25, +3.0] s of each ~8 s trial, so the remaining in-trial ophys data (≈ 2.8 s before and ≥ 1.2 s after the window) is loaded but never binned; and the `--signal`, `--neural-lag`, `--nsessions` options exist only for the comparison experiments in Steps 7/12.
- Nothing computed is *wrong-but-kept*: the five outputs, `neural`, `brain_region_idx` and `subject_idx` are all consumed by `train_decoder.py`; the empty `(0, 21)` input arrays are required by the format.

ii. **Code snippets**

```python
        cell_ids += list(index)           # collected, never used downstream
    ...
        'cell_ids': cell_ids,
```

```python
    run_binned, _ = bin_mean(run_v, run_ts, edges_flat)     # counts discarded
```

```python
def keep_index(n_trials):
    """... per trial the first NBINS are real bins and the last is the gap to
    the next trial (dropped)."""
    base = np.arange(n_trials)[:, None] * (NBINS + 1)
    return (base + np.arange(NBINS)[None, :]).ravel()
```

iii. **Justification** — The notes do not flag these as waste; the vectorisation trade-off is documented in `keep_index`'s docstring, and the retained continuous arrays are explicitly used ("discretize running / pupil (per session, over exactly the exported timepoints)") and for the `--show-processing` figures, which the instructions require. The window truncation is justified in Step 5 decision 4 (a window guaranteed to lie inside every trial, with the flash grid phase-locked to the change).
