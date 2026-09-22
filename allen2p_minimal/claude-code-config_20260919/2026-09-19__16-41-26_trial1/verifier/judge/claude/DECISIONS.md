# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI works directly off the local release copy at `/app/data/visual-behavior-ophys-1.1.0`. It reads the SDK metadata CSV `project_metadata/ophys_experiment_table.csv`, intersects it with the set of `.nwb` files that are actually present in `behavior_ophys_experiments/` (284 files; 239 of them `VisualBehavior`, 45 `VisualBehaviorMultiscope`), and then keeps only `project_code == 'VisualBehavior'` and non-`passive` rows. Each surviving experiment is opened one at a time with `BehaviorOphysExperiment.from_nwb_path()`. Loading/extraction is run in a 12–16-worker `ProcessPoolExecutor`, with each session's trial-aligned result written to an `.npz` file under `/app/cache_sessions` so a re-run does not re-read the NWB. A second pass reads the cache back and assembles the final dictionary. Only two streams-worth of data per session (`events`, `stimulus_presentations`, `trials`, `running_speed`, `eye_tracking`, `ophys_timestamps`) are touched; `dff_traces` is never loaded in the final script.

ii.
```python
def session_table():
    """Table of the sessions to convert, one row per session."""
    et = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    available = set()
    for fn in os.listdir(EXPERIMENT_DIR):
        if fn.endswith('.nwb'):
            available.add(int(fn.split('_')[-1].split('.')[0]))
    et = et[et.ophys_experiment_id.isin(available)]
    et = et[(et.project_code == PROJECT_CODE) & (~et.passive.astype(bool))]
    # single plane imaging: one imaging plane (experiment) per session
    assert et.ophys_session_id.nunique() == len(et)
    et = et.sort_values('ophys_experiment_id').reset_index(drop=True)
    return et
```
```python
    path = os.path.join(EXPERIMENT_DIR,
                        f'behavior_ophys_experiment_{oeid}.nwb')
    ds = BehaviorOphysExperiment.from_nwb_path(path)
```
```python
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(extract_and_cache, oeids)):
            ...
    for oeid in oeids:
        with np.load(cache_path(oeid), allow_pickle=True) as f:
            if 'error' in f:
                skipped.append((oeid, str(f['error'])))
                continue
            sessions.append({k: f[k] for k in f.files})
```

iii. From the trajectory: the AI first enumerated the NWB files on disk and cross-tabulated them against the experiment table (steps 12/15), establishing that all 239 `VisualBehavior` experiments are present locally, so no network/S3 access is needed and the on-disk intersection cannot silently request a missing file. It timed a single session load (~9 s, dominated by opening the NWB) and measured peak RSS for the largest session (666 cells) before choosing the parallel + on-disk-cache design, so that the ~165-session run fits in memory and is restartable.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the filtered experiment table, sorted as strings. `subject_idx` is the index of each session's mouse in that sorted list. 37 mice result.

ii.
```python
    subjects = sorted(table.mouse_id.astype(str).unique().tolist())
    subject_idx = np.array([subjects.index(str(m)) for m in table.mouse_id],
                           dtype=np.int64)
```

iii. `mouse_id` is the SDK's canonical animal identifier in the experiment table; the AI verified in exploration (step 15) that the `VisualBehavior` project contains 37 mice and printed the per-mouse session counts, matching the project documentation.

## 1-c. How are the data split into sessions?

i. One session = one `ophys_experiment_id` (one NWB file). The AI asserts that for `project_code == 'VisualBehavior'` there is exactly one experiment (imaging plane) per `ophys_session_id`, so no merging across planes is needed. Two exclusions are applied at the session level:
- **Multiscope experiments excluded** by the `project_code == 'VisualBehavior'` filter (and, redundantly, by a 30–32 Hz frame-rate guard), because their per-plane frame rate is ~3× slower and would break the requirement of a single time-bin size.
- **Passive sessions excluded** (`~passive`, i.e. `OPHYS_2_images_A_passive` and `OPHYS_5_images_B_passive`), 71 of the 239.

Three further sessions (795953296, 806456687, 833631914) are dropped because their `eye_tracking` table is empty. Sessions yielding fewer than 2 usable trials would also be dropped. Final: **165 sessions, 37 mice, 28,821 neurons, 42,470 trials**.

ii.
```python
    et = et[(et.project_code == PROJECT_CODE) & (~et.passive.astype(bool))]
    assert et.ophys_session_id.nunique() == len(et)
```
```python
    frame_rate = 1.0 / dt
    if not (MIN_FRAME_RATE <= frame_rate <= MAX_FRAME_RATE):
        return {'oeid': oeid, 'error': f'frame rate {frame_rate:.2f} Hz'}
```
```python
    eye = ds.eye_tracking
    if len(eye) == 0:
        return {'oeid': oeid, 'error': 'no eye tracking'}
    ...
    if len(neural) < 2:
        return {'oeid': oeid, 'error': f'only {len(neural)} usable trials'}
```

iii. The AI's stated reasoning (script docstring and final message): passive sessions have the lick spout retracted and a satiated mouse, so the required `trial_outcome` output "is not defined behaviourally (every go trial is a miss by construction)", and the reference paper analyses only active behavior sessions. It also decided to *keep* both familiar and novel image sets, explicitly noting that the reference paper's familiar-only restriction was specific to its novelty question and that nothing in the decoding task requires it, while novel sessions roughly double the data and contribute the second 8-image set. Multiscope is excluded on the uniform-bin-size argument. The three eye-tracking-free sessions were confirmed empirically (step 85: `eye rows 0`) and dropped because pupil diameter is a required output.

## 1-d. How are the data split into trials?

i. Trials come from the experiment's own `BehaviorOphysExperiment.trials` table. Kept rows are `(go | catch) & ~aborted & ~auto_rewarded & change_time.notna()`. Rather than using the trials table's own `start_time`–`stop_time` extent, each trial is re-cut as a **fixed 4-second window, [-2 s, +2 s] around `change_time`**, i.e. `T = round(4 s / dt) = 124` ophys frames starting at the first frame at or after `change_time - 2 s`. Every trial in the dataset therefore has exactly 124 bins.

ii.
```python
    trials = ds.trials
    keep = ((trials.go.astype(bool) | trials.catch.astype(bool))
            & ~trials.aborted.astype(bool)
            & ~trials.auto_rewarded.astype(bool)
            & trials.change_time.notna())
    trials = trials[keep]

    for _, tr in trials.iterrows():
        change_time = float(tr.change_time)
        i0 = int(np.searchsorted(ts, change_time + OFF_START, side='left'))
        if i0 + T > len(ts):
            n_dropped_edge += 1
            continue
        tt = ts[i0:i0 + T]
        if tt[0] < tr.start_time or tt[-1] > tr.stop_time:
            n_outside_trial += 1
```

iii. Docstring: "the experiment's own trial definition is used … Only 'go' and 'catch' trials are kept", and the fixed window is justified as safe because "Every go/catch trial is at least 3.0 s long before the change and 4.2 s long after it, so this window always lies inside the experiment-defined trial." The AI verified this empirically twice: in exploration (step 33) it printed `change_time - start_time` and `stop_time - change_time` distributions, and after the full run (step 79) it aggregated `n_dropped_edge = 0, n_outside_trial = 0, avail = used = 42470`, i.e. no trial was lost and no window fell outside its parent trial. The uniform window was chosen to make `T` identical across every trial and session.

## 1-e. How are trials filtered based on quality controls?

i. Filters, in order: (1) trial type — only `go` or `catch`, excluding `aborted` and `auto_rewarded`; (2) `change_time` must be non-NaN; (3) the 124-frame window must fit inside the recording (`i0 + T <= len(ts)`); (4) the image-identity lookup must find a preceding flash (`k >= 0`); (5) exactly one of `hit/miss/false_alarm/correct_reject` must be set, otherwise the trial is rolled back and dropped; (6) a session must yield ≥ 2 usable trials. In the actual run, filters 3–5 removed **zero** trials.

ii.
```python
    keep = ((trials.go.astype(bool) | trials.catch.astype(bool))
            & ~trials.aborted.astype(bool)
            & ~trials.auto_rewarded.astype(bool)
            & trials.change_time.notna())
```
```python
        k = np.searchsorted(flash_start, tt, side='right') - 1
        if np.any(k < 0):
            n_dropped_edge += 1
            continue
        ...
        else:
            n_dropped_edge += 1
            image_name.pop(); change.pop(); running.pop(); pupil.pop()
            continue
```
```python
    if len(neural) < 2:
        return {'oeid': oeid, 'error': f'only {len(neural)} usable trials'}
```

iii. The instructions explicitly ask for go and catch trials and for aborted/auto-rewarded trials to be excluded; the AI's docstring restates the rationale ("aborted trials (mouse licked before the change, no stimulus change happened)" and "auto_rewarded trials (free rewards at session start / after 10 consecutive misses)"). It checked during exploration (step 33) that `(go|catch)` trials always have a non-NaN `change_time` and that `hit|miss|false_alarm|correct_reject` covers all of them, so the extra guards are defensive only. The ≥ 2 trial rule comes from the format requirement that each session needs at least two trials for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The detected calcium events table, `BehaviorOphysExperiment.events`, column **`filtered_events`** — the L0-regularised deconvolution of the dF/F traces, in the SDK's half-Gaussian-smoothed form. One row per `cell_specimen_id`, each of length `len(ophys_timestamps)`. `dff_traces` is *not* used.

ii.
```python
    # ---- neural activity: detected calcium events ------------------------
    events = ds.events
    neural_all = np.stack(events['filtered_events'].values).astype(np.float32)
    if neural_all.shape[1] != len(ts):
        return {'oeid': oeid, 'error': 'events / timestamps length mismatch'}
```

iii. The AI read `/app/methods.txt` first, where the reference paper states "For all analysis of neural data we used the detected calcium events … thus removing the slow decay dynamics of the calcium indicator GCaMP6f". Its docstring cites exactly this: "the detected calcium events … as used for all neural analysis in the reference paper (Piet et al.)". It then ran an explicit head-to-head decodability probe (step 77) on two sessions comparing `events`, `filtered_events`, and `dff`:

```
=== 822024770  events   nonzero=0.001  img=0.247 ch250=0.573 out=0.297
               filtered nonzero=0.017  img=0.458 ch250=0.597 out=0.422
               dff      nonzero=0.990  img=0.744 ch250=0.631 out=0.615
```

and chose `filtered_events` over raw `events` because "the raw event train is nonzero in only ~0.1% of the 32 ms bins, which carries almost no signal at the single-bin resolution at which this decoder has to make a prediction", noting that filtering "roughly doubled image-identity decoding (0.25 → 0.46) without changing what the signal is". It explicitly saw that `dff` decoded better still but rejected it on the grounds of paper consistency.

## 2-b. How is the `neural` data processed?

i. Essentially no processing beyond what the SDK ships: the per-cell event vectors are stacked into an `(n_cells, n_frames)` `float32` matrix and sliced per trial. No z-scoring, no baseline subtraction, no normalisation, no additional smoothing, no cross-plane merging (single-plane data, one plane per session). Values stay in raw event-magnitude (dF/F) units; `metadata['neural_units']` records this.

ii.
```python
    neural_all = np.stack(events['filtered_events'].values).astype(np.float32)
    ...
        neural.append(neural_all[:, i0:i0 + T])
    ...
        neural.append([np.ascontiguousarray(s['neural'][i]) for i in range(ntrials)])
```

iii. The half-Gaussian filtering is already applied by the SDK, so the AI treated `filtered_events` as the final signal. Its metadata documents it as "detected calcium events (L0-regularized deconvolution of the dF/F traces, half-gaussian filtered: AllenSDK events['filtered_events']), sampled at the ophys frame times". It verified the result was not degenerate (step 61: nonzero fraction ≈ 2–6% per session, no all-zero sessions) and inspected the change-triggered population average (step 75), which shows a clean transient peaking ~0.4 s after the change bin.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level curation at all. Every cell in the released NWB's `events` table is kept (6–666 cells per session, mean 175). The only neural-level guards are structural: the event matrix length must equal the number of ophys timestamps, and a session with zero cells is dropped.

ii.
```python
    if neural_all.shape[1] != len(ts):
        return {'oeid': oeid, 'error': 'events / timestamps length mismatch'}
    if neural_all.shape[0] == 0:
        return {'oeid': oeid, 'error': 'no cells'}
```

iii. The docstring states: "for each session all cells contained in the released NWB file are used. The published files only contain ROIs that passed the Allen ROI-filtering / QC pipeline (`valid_roi == True`), so no further neuron curation is applied." The AI confirmed this empirically in step 21 by printing `cell_specimen_table.valid_roi.value_counts()` → all `True`. It also checked cell counts per session against `ophys_cells_table.csv` (step 31) and chose not to drop low-cell-count sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the **change time** — `trials.change_time`, which is the onset of the changed image flash on go trials and the onset of the sham-change flash on catch trials. The window is `[-2 s, +2 s]`, taken as the 124 consecutive ophys frames beginning with the first frame at or after `change_time - 2 s` (`searchsorted`, `side='left'`). Because the window is snapped to whole ophys frames, the change falls in bin 61 or 62 (a sub-frame, ≤ 32 ms, jitter). `metadata['temporal_alignment_event']`, `off_start = -2.0`, `off_end = +2.0` record this. All other streams are evaluated at exactly those same `tt` frame times, so every row of every trial shares one time base.

ii.
```python
        i0 = int(np.searchsorted(ts, change_time + OFF_START, side='left'))
        if i0 + T > len(ts):
            n_dropped_edge += 1
            continue
        tt = ts[i0:i0 + T]
        ...
        neural.append(neural_all[:, i0:i0 + T])
```
```python
            'temporal_alignment_event':
                'image change time (onset of the changed image flash on go '
                'trials, onset of the sham-change flash on catch trials)',
            'off_start': OFF_START,
            'off_end': OFF_END,
```

iii. The AI verified in step 47 that `change_time` coincides *exactly* with a stimulus-presentation onset: "go change_time − nearest flash start: min 0.0000 max 0.0000", with `frac is_change = 1.0` for go trials and `frac is_sham = 1.0` for catch trials. It then verified after conversion (step 81) that the change label always begins at bin 61 or 62 across sessions. The instruction "temporally align based on ophys timestamp" is satisfied by snapping the window to ophys frames and resampling every other stream onto those frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native ophys frame period is used: **32.31 ms (≈ 31 Hz)**, one bin per two-photon frame. **No rebinning, downsampling, or smoothing across time is applied to the neural data.** `T = round(4 s / dt) = 124` bins per trial for every trial in every session. The script asserts that `T` is identical across sessions and reports the spread of per-session `dt`; `metadata['time_bin_size']` is the mean `dt` in ms across sessions.

ii.
```python
    ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
    dt = float(np.median(np.diff(ts)))
    T = int(round((OFF_END - OFF_START) / dt))
```
```python
    Ts = np.array([int(s['T']) for s in sessions])
    dts = np.array([float(s['dt']) for s in sessions])
    assert np.all(Ts == Ts[0]), f'inconsistent number of time bins: {set(Ts)}'
    T = int(Ts[0])
    print(f'T = {T} bins of {np.mean(dts) * 1000:.3f} ms '
          f'(range {dts.min() * 1000:.3f}-{dts.max() * 1000:.3f} ms)')
```

iii. The docstring: "the ophys frames are the time base ('temporally align based on ophys timestamp') … All single-plane sessions run at 31 Hz (bin = 32.3 ms), so T and the bin size are identical for every trial of every session. All other data streams are resampled onto these frame times." The 30–32 Hz guard plus the `T` assertion enforce this rather than assuming it. The AI's verification run reported `T_mean = T_median = T_min = T_max = 124` and `time_bin_size = 32.31 ms`.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From `BehaviorOphysExperiment.stimulus_presentations`, restricted to the `change_detection_behavior` block: the `start_time`, `image_name`, and `omitted` columns. Omitted flashes are removed from the lookup table so that the image held during an omission is the image of the preceding real flash. The trials table's `initial_image_name`/`change_image_name` are *not* used.

ii.
```python
    sp = ds.stimulus_presentations
    sp = sp[sp.stimulus_block_name == 'change_detection_behavior']
    sp = sp.sort_values('start_time')
    shown = sp[~sp.omitted.astype(bool)]
    flash_start = shown.start_time.values.astype(np.float64)
    flash_image = shown.image_name.values.astype(str)
```

iii. Docstring: image identity is "which of the 16 natural images (8 in set A, 8 in set B) is currently being shown. Held over the whole 750 ms image-presentation cycle, i.e. also over the gray period that follows the 250 ms flash and over omitted flashes (5% of flashes are omitted; an omission is a continuation of the gray screen and is never a change or a pre-change flash, so the withheld image is always the current image)." This mirrors the reference paper's "image presentation interval" convention (methods.txt: "By image presentation interval we refer to the 750 ms interval beginning with each image presentation. For image omissions we used the 750 ms following the time of the omission"). In step 47 the AI confirmed the flash grid (inter-flash interval 0.7506 s), that omitted rows carry `image_name == 'omitted'`, and the image-name↔index map.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Per-frame step-hold lookup followed by a global integer encoding. For each frame time `tt`, `searchsorted(flash_start, tt, side='right') - 1` gives the most recent non-omitted flash, whose `image_name` is the label. The union of image names over all sessions is sorted to a 16-entry list (both image set A and set B), and names are mapped to integer codes with that global mapping; `output_values[0]` is the same sorted list.

ii.
```python
        k = np.searchsorted(flash_start, tt, side='right') - 1
        if np.any(k < 0):
            n_dropped_edge += 1
            continue
        image_name.append(flash_image[k])
```
```python
    images = sorted({str(im) for s in sessions
                     for im in np.unique(s['image_name'])})
    image_to_idx = {im: i for i, im in enumerate(images)}
    ...
        img = np.vectorize(image_to_idx.__getitem__)(s['image_name'])
```

iii. A single global, sorted mapping is used so a given code means the same image in every session — necessary because the decoder shares one output head across sessions and because image sets A and B both appear. The AI validated the result against the source (step 81): at the bin just after the change the decoded image name equals `trials.change_image_name` for 100% of trials, and just before the change it equals `initial_image_name` for ~89–92% of trials (the remainder being the trials where the change lands in bin 61 rather than 62).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated at exactly the same 124 ophys frame times `tt = ts[i0:i0+T]` used to slice the neural matrix, so the image label in bin *b* and the neural activity in bin *b* are the same two-photon frame by construction.

ii.
```python
        tt = ts[i0:i0 + T]
        ...
        k = np.searchsorted(flash_start, tt, side='right') - 1
        image_name.append(flash_image[k])
        ...
        neural.append(neural_all[:, i0:i0 + T])
```

iii. The stimulus, behavior and ophys clocks are hardware-synchronised in this dataset (methods.txt, "Temporal synchronization of all data-streams … recording all experimental clocks on a single NI PCI-6612 digital IO board at 100 kHz"), so taking each stream at the ophys frame times is a valid common alignment. The AI's design statement is "All other data streams are resampled onto these frame times."

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From `stimulus_presentations.is_change` (within the `change_detection_behavior` block) and the corresponding `start_time`s. Sham changes on catch trials carry `is_sham_change`, not `is_change`, so they are automatically excluded. The trials table's `go` flag is not used for this output.

ii.
```python
    change_start = sp[sp.is_change.astype(bool)].start_time.values.astype(np.float64)
```

iii. Step 47 established that `is_change` flags exactly the real image changes (`frac is_change = 1.0` for the flash nearest each go trial's `change_time`) and `is_sham_change` the catch-trial sham changes, and that omitted flashes are never `is_change`. Docstring: "Sham changes on catch trials are 0 (the image does not change)."

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary step function: for each frame time, find the most recent real change onset and set the label to 1 if the frame is within 750 ms of it (one image-presentation cycle: 250 ms image + 500 ms gray), else 0. Frames before the first change of the session get 0.

ii.
```python
        k = np.searchsorted(change_start, tt, side='right') - 1
        ch = np.zeros(T, dtype=np.int16)
        valid = k >= 0
        ch[valid] = (tt[valid] - change_start[k[valid]] < IMAGE_CYCLE)
        change.append(ch)
```
with `IMAGE_CYCLE = 0.75  # duration of one image presentation cycle (250 ms image + 500 ms gray)`.

iii. Docstring: "1 during the 750 ms image-presentation cycle that starts with a real image change, 0 elsewhere." This matches both the instruction ("value of 1 right after a change in image identity") and the paper's 750 ms image-presentation-interval convention. The AI verified after conversion (step 81) that catch trials have zero change-labelled bins (0/24, 0/30, 0/28 across three sessions) and that on go trials the run of 1s always starts at bin 61 or 62 and is 23–24 bins long (≈ 0.75 s).

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary — no thresholding of a continuous quantity is involved. `output_values[1] = ['no_change', 'change']`. In the delivered dataset the class balance is 83.6% / 16.4% (higher than a full-trial window would give, because the 0.75 s positive interval sits inside a 4 s trial).

ii.
```python
        'output_values': [
            images,
            ['no_change', 'change'],
            ...
```

iii. The binarisation is defined by the 750 ms window rather than by a threshold; the AI's justification is the image-presentation-interval convention described above.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity: evaluated at the identical 124 ophys frame times `tt`, so the change flag and the neural activity share bin indices.

ii.
```python
        tt = ts[i0:i0 + T]
        k = np.searchsorted(change_start, tt, side='right') - 1
        ch[valid] = (tt[valid] - change_start[k[valid]] < IMAGE_CYCLE)
```

iii. Because the window is aligned to `change_time`, which the AI verified is exactly a flash onset, the positive interval is anchored at bin 61/62 in every trial — confirmed empirically in step 81.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `BehaviorOphysExperiment.running_speed`, columns `timestamps` and `speed` (cm/s, the SDK's 10 Hz-lowpass-filtered running speed, sampled at ~60 Hz). Non-finite samples are discarded before use.

ii.
```python
    run = ds.running_speed
    run_t = run.timestamps.values.astype(np.float64)
    run_v = run.speed.values.astype(np.float64)
    good = np.isfinite(run_t) & np.isfinite(run_v)
    run_t, run_v = run_t[good], run_v[good]
    if len(run_t) == 0:
        return {'oeid': oeid, 'error': 'no running data'}
```

iii. `running_speed` is the SDK's standard, wrap/transient-corrected and lowpass-filtered locomotion signal (methods.txt describes the correction and the 10 Hz Butterworth filter, with `running_speed` being the filtered attribute). The AI inspected it in step 21 (`(270240, 2)`, `dt ≈ 0.0167 s`) and its per-session percentiles in step 49.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Two steps. (1) Linear interpolation onto the trial's ophys frame times with `np.interp` (which clamps rather than extrapolating at the edges). (2) Discretisation into 5 equal-count percentile bins whose edges — the 20th/40th/60th/80th percentiles — are computed **once, pooled over every frame of every trial of every session**, then applied with `searchsorted(..., side='right')`.

ii.
```python
        running.append(np.interp(tt, run_t, run_v))
```
```python
def percentile_bins(values, nbins):
    """Equal-count percentile bin edges (interior edges only)."""
    qs = np.linspace(0, 100, nbins + 1)[1:-1]
    return np.percentile(values, qs)
...
    all_running = np.concatenate([s['running'].ravel() for s in sessions])
    run_edges = percentile_bins(all_running, NBINS_BEHAVIOR)
...
        run_bin = np.searchsorted(run_edges, s['running'], side='right')
```

iii. Docstring: "running speed (cm/s) resampled to the ophys frame times and discretised into 5 equal-count percentile bins", and "The percentile bin edges for running speed and pupil diameter are computed once over all trials of all sessions, so that a given bin index means the same physical value in every session (the decoder shares one output head across sessions)." The edges are printed and stored in `metadata['running_speed_bin_edges']`. The resulting marginal is exactly 20% per bin in the verification output.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile (equal-count) bins, as the instructions require. Interior edges at the 20/40/60/80th percentiles of the pooled distribution; bin index = number of edges the value exceeds, giving codes 0–4 with names `running_speed_bin0 … bin4`.

ii.
```python
NBINS_BEHAVIOR = 5
...
    qs = np.linspace(0, 100, nbins + 1)[1:-1]
    return np.percentile(values, qs)
...
        run_bin = np.searchsorted(run_edges, s['running'], side='right')
```

iii. Directly follows the instruction "Running speed, discretized into five equal percentile bins". Global (not per-session) edges are used so the categories are comparable across sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated directly onto the trial's ophys frame times `tt`, the same times used to slice the neural matrix, so both share bin indices.

ii.
```python
        tt = ts[i0:i0 + T]
        running.append(np.interp(tt, run_t, run_v))
        ...
        neural.append(neural_all[:, i0:i0 + T])
```

iii. Same synchronisation argument as for the stimulus streams — all clocks are hardware-synced, so evaluating the behavioral stream at the ophys frame times is a valid common alignment ("All other data streams are resampled onto these frame times").

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `BehaviorOphysExperiment.eye_tracking`, columns `timestamps`, `pupil_area` and `likely_blink`. Diameter is computed as the equivalent-disc diameter of the fitted pupil ellipse area, `2*sqrt(pupil_area/pi)`, in eye-camera pixels. Samples flagged `likely_blink`, and any non-finite sample, are dropped. Sessions with an empty eye-tracking table, or with no valid pupil sample at all, are dropped entirely.

ii.
```python
    eye = ds.eye_tracking
    if len(eye) == 0:
        return {'oeid': oeid, 'error': 'no eye tracking'}
    pupil_t = eye.timestamps.values.astype(np.float64)
    # diameter of a disc with the fitted pupil area; blinks are already NaN in
    # pupil_area, drop them (and any other non-finite sample) and interpolate.
    pupil_d = 2.0 * np.sqrt(eye.pupil_area.values.astype(np.float64) / np.pi)
    good = (np.isfinite(pupil_t) & np.isfinite(pupil_d)
            & ~eye.likely_blink.values.astype(bool))
    pupil_t, pupil_d = pupil_t[good], pupil_d[good]
    if len(pupil_t) == 0:
        return {'oeid': oeid, 'error': 'no valid pupil data'}
```

iii. Docstring: "pupil diameter (2*sqrt(pupil_area/pi), in camera pixels) from the eye-tracking ellipse fit, blinks removed and linearly interpolated". The AI examined the eye-tracking table columns in step 21 and sanity-checked per-session pupil-diameter percentiles across 10 random sessions in step 49 (values in a plausible, consistent range), and confirmed in step 85 that exactly three sessions have zero eye-tracking rows.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area → equivalent-disc diameter conversion; blink and non-finite sample removal; linear interpolation (`np.interp`) across the resulting gaps onto the trial's ophys frame times; then 5 equal-count percentile bins with globally pooled edges, exactly as for running speed. No per-session normalisation of pupil size is applied.

ii.
```python
    pupil_d = 2.0 * np.sqrt(eye.pupil_area.values.astype(np.float64) / np.pi)
    ...
        pupil.append(np.interp(tt, pupil_t, pupil_d))
    ...
    all_pupil = np.concatenate([s['pupil'].ravel() for s in sessions])
    pupil_edges = percentile_bins(all_pupil, NBINS_BEHAVIOR)
    ...
        pupil_bin = np.searchsorted(pupil_edges, s['pupil'], side='right')
```

iii. Blinks are removed *before* interpolation so blink artifacts are not smeared into neighbouring frames; the same global-percentile argument as for running speed applies ("so that a given bin index means the same physical value in every session"). Edges are recorded in `metadata['pupil_diameter_bin_edges']`.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five equal-count percentile bins (interior edges at the 20/40/60/80th percentiles of the pooled across-session distribution), codes 0–4, names `pupil_diameter_bin0 … bin4`. The verification run shows an essentially exact 20% per bin marginal.

ii.
```python
    pupil_edges = percentile_bins(all_pupil, NBINS_BEHAVIOR)
    print('pupil diameter bin edges (px):', np.round(pupil_edges, 3))
    ...
        pupil_bin = np.searchsorted(pupil_edges, s['pupil'], side='right')
```

iii. Directly follows the instruction "Pupil diameter, discretized into five equal percentile bins", with the same global-edge rationale as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the same 124 ophys frame times `tt` used for the neural slice, so bin indices coincide.

ii.
```python
        tt = ts[i0:i0 + T]
        pupil.append(np.interp(tt, pupil_t, pupil_d))
```

iii. Same hardware-synchronisation / common-time-base argument as the other streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table — `hit`, `miss`, `false_alarm`, `correct_reject` — tested in that fixed order and mapped to 0/1/2/3. A trial matching none of them is dropped (never occurred).

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
        if tr.hit:
            o = 0
        elif tr.miss:
            o = 1
        elif tr.false_alarm:
            o = 2
        elif tr.correct_reject:
            o = 3
        else:
            n_dropped_edge += 1
            image_name.pop(); change.pop(); running.pop(); pupil.pop()
            continue
        outcome.append(o)
```

iii. These are the SDK's canonical change-detection outcome labels; the AI checked in step 33 that `(hit|miss|false_alarm|correct_reject).all()` is true for the go/catch subset, so the four-way encoding is exhaustive and unambiguous. The ordering is recorded in `output_values[4]`. This categorisation also underpins the decision to drop passive sessions, where every go trial would be a miss.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The integer code is broadcast to a constant `T`-length row so that the static per-trial variable is represented as a time series, as the format guidance prefers ("If at all possible, make it time-varying").

ii.
```python
            out.append(np.stack([
                img[i].astype(np.int16),
                s['change'][i].astype(np.int16),
                run_bin[i].astype(np.int16),
                pupil_bin[i].astype(np.int16),
                np.full(T, s['outcome'][i], dtype=np.int16),
            ]))
```

iii. Keeping all five output rows the same `(5, T)` shape lets the decoder treat every output uniformly; the trial-outcome row is simply constant in time. The AI verified against the source trials table (step 81, "outcome match: 1.0") on three sessions.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is mostly "detect and exclude", at several granularities:
- **Whole session, hard errors**: any exception while extracting a session is caught, the traceback is stored as that session's `error`, and the session is skipped at assembly time — one bad file cannot kill the run.
- **Whole session, missing streams**: empty `eye_tracking` (3 sessions), no valid pupil samples, no running samples, zero cells, an events/timestamps length mismatch, a frame rate outside 30–32 Hz, or fewer than 2 usable trials all cause the session to be skipped with a printed reason.
- **Within-session missing samples**: non-finite running or pupil samples and `likely_blink` frames are removed and then bridged by linear interpolation; `np.interp` clamps at the ends, so no NaN ever reaches the discretisation step (no artificial "bin 0" for missing data).
- **Trial level**: trials whose 4 s window would run off the end of the recording, trials with no preceding image flash, and trials with no outcome flag are dropped, with the partially appended rows rolled back so the per-trial lists stay in lockstep.
- **Reporting**: `n_dropped_edge`, `n_outside_trial`, `n_trials_available` are carried out of every session and printed; the AI aggregated them (step 79) to confirm `edge = 0, out = 0, avail = used = 42470`.

ii.
```python
    try:
        res = extract_session(oeid)
    except Exception:
        res = {'oeid': oeid, 'error': traceback.format_exc(limit=3)}
```
```python
    for oeid in oeids:
        with np.load(cache_path(oeid), allow_pickle=True) as f:
            if 'error' in f:
                skipped.append((oeid, str(f['error'])))
                continue
            sessions.append({k: f[k] for k in f.files})
    print(f'{len(sessions)} sessions kept, {len(skipped)} skipped')
    for oeid, err in skipped:
        print(f'  skipped {oeid}: {err}')
```
```python
    good = (np.isfinite(pupil_t) & np.isfinite(pupil_d)
            & ~eye.likely_blink.values.astype(bool))
    pupil_t, pupil_d = pupil_t[good], pupil_d[good]
```
```python
        else:
            n_dropped_edge += 1
            image_name.pop(); change.pop(); running.pop(); pupil.pop()
            continue
```

iii. The AI's stated position is that pupil diameter is a *required* output, so a session with no eye tracking at all cannot contribute and should be dropped rather than filled with a sentinel; within a session, gaps are short (blinks) and interpolation is the natural fix. The run log surfaces every skipped session and its reason, and the AI followed up on all three skipped sessions to confirm the cause (step 85: `eye rows 0`, shape `(0, 23)`).

## 9-a. What are the most time-consuming steps of the code?

i. (1) Opening each NWB file — the AI measured ~7.3 s of the ~9 s per-session cost in `BehaviorOphysExperiment.from_nwb_path` alone, versus 0.09 s to stack the event matrix and 0.003 s for the remaining tables; this is pure I/O/HDF5 parsing and dominates. (2) Writing and re-reading the 165 per-session `.npz` cache files (a full extra round-trip of ~3.7 GB through disk). (3) Pickling the 3.94 GB output file. The AI mitigated (1) with a 16-process pool, which brought the whole conversion to a few minutes.

ii.
```python
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(extract_and_cache, oeids)):
```
```python
def extract_and_cache(oeid):
    """Worker: extract one session and write it to the cache directory."""
    out = cache_path(oeid)
    if os.path.exists(out):
        with np.load(out, allow_pickle=True) as f:
            return {'oeid': oeid, 'cached': True, ...}
```

iii. The AI profiled a single session before writing the script (step 45: `open 7.26 s`, `events tbl 0.002 s`, `stack 0.087 s`, `rest 0.003 s`) and checked `nproc`/`free -g` (step 44) plus peak RSS on the largest session (step 51) to size the worker pool. The on-disk cache was added so that a crash or a change in the assembly stage does not require re-reading 165 NWB files.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops are per-trial Python loops that could in principle be vectorised:
- The main `for _, tr in trials.iterrows()` loop in `extract_session` — `iterrows()` is the slow pandas access pattern; the `searchsorted` calls for `i0`, image identity and change, and the two `np.interp` calls could all be done once for all trials of a session by building a `(ntrials, T)` index matrix.
- The assembly loop `for i in range(ntrials): out.append(np.stack([...]))`, which builds each trial's `(5, T)` output row-by-row when the four time-varying rows are already available as whole `(ntrials, T)` arrays.
- `np.vectorize(image_to_idx.__getitem__)` is a Python-level loop in disguise; a `np.searchsorted` on the sorted image list, or a `pd.Categorical`, would be far faster.

None of these matter much in practice because NWB reading dominates, and the AI parallelised at the session level instead.

ii.
```python
    for _, tr in trials.iterrows():
        change_time = float(tr.change_time)
        i0 = int(np.searchsorted(ts, change_time + OFF_START, side='left'))
        ...
        running.append(np.interp(tt, run_t, run_v))
        pupil.append(np.interp(tt, pupil_t, pupil_d))
```
```python
        img = np.vectorize(image_to_idx.__getitem__)(s['image_name'])
        ...
        for i in range(ntrials):
            out.append(np.stack([...]))
```

iii. The AI never states a vectorisation rationale; its efficiency effort went into process-level parallelism and caching, which targets the actual (I/O) bottleneck it had measured.

## 9-c. What processing does the code repeat multiple times?

i. Genuine repetition in the code:
- **Per-trial re-interpolation of whole-session behavioral streams.** `np.interp(tt, run_t, run_v)` is called once per trial with the full ~270,000-sample running array (and ~136,000-sample pupil array) as the source. Interpolating each stream once onto the full `ophys_timestamps` vector and then slicing — as the reference solution does — would do the same work once per session instead of ~250 times.
- **A full serialise/deserialise round-trip through the `.npz` cache.** Every session's arrays are written to disk and immediately read back in the same run; on a cold cache this is pure overhead (it only pays off on a re-run).
- **Image names are carried as string arrays** (`np.stack(image_name)`, dtype `<U…`) through the cache and are only converted to integer codes at the very end, so the string form is built, written, read and then re-mapped.
- Minor: `searchsorted` over `ts` is done per trial; the experiment table is re-indexed and re-iterated in the assembly stage.

ii.
```python
        running.append(np.interp(tt, run_t, run_v))
        pupil.append(np.interp(tt, pupil_t, pupil_d))
```
```python
    np.savez(tmp, **res)
    os.replace(tmp, out)
    ...
        with np.load(cache_path(oeid), allow_pickle=True) as f:
            sessions.append({k: f[k] for k in f.files})
```
```python
        'image_name': np.stack(image_name),              # (ntrials, T) str
```

iii. The cache round-trip is a deliberate trade-off the AI made for restartability and for keeping worker memory bounded (it is the mechanism that lets the parallel pool return small results instead of gigabytes through IPC). The per-trial interpolation is not justified anywhere; it is simply the straightforward way to write the loop, and is cheap relative to NWB reading.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts:
- `cell_specimen_ids` is computed and cached for every session but is never written into the output dictionary or used anywhere.
- `n_outside_trial` is computed per trial (`tt[0] < tr.start_time or tt[-1] > tr.stop_time`) purely as a diagnostic counter; the code explicitly notes it "never happens" and takes no action on it. `n_trials_available` and `n_dropped_edge` are likewise diagnostics only.
- The 30–32 Hz frame-rate check is redundant with the `project_code == 'VisualBehavior'` filter, which already removes every Multiscope (slower) experiment.
- The whole `.npz` cache (~3.7 GB) is dead weight after the single run that produced the pickle.
- The string image-name arrays are an intermediate representation that is discarded after integer encoding.
- All five output rows are stored as `int16` when `int8` would suffice for every one of them (max value 15).

Notably, the code does *not* waste effort loading `dff_traces` in the final script — that comparison was done in a separate throwaway probe.

ii.
```python
    cell_specimen_ids = np.asarray(events.index.values)
```
```python
        if tt[0] < tr.start_time or tt[-1] > tr.stop_time:
            # window would leave the experiment-defined trial; never happens
            # with the [-2, 2] s window, counted for the report
            n_outside_trial += 1
```
```python
    if not (MIN_FRAME_RATE <= frame_rate <= MAX_FRAME_RATE):
        return {'oeid': oeid, 'error': f'frame rate {frame_rate:.2f} Hz'}
```

iii. The AI framed the diagnostic counters as verification rather than waste — it used exactly these counters in step 79 to prove that no trial was lost and that every fixed window lay inside its experiment-defined trial, which is the evidence backing its trial-windowing decision. The redundant frame-rate guard is defensive: it makes the "one bin size everywhere" invariant fail loudly rather than silently if the session selection were ever widened.
