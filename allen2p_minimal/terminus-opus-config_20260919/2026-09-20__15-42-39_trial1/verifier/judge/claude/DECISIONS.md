# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the AllenSDK `VisualBehaviorOphysProjectCache.from_local_cache(cache_dir='/app/data')` to load experiments. It first reads the project metadata table `project_metadata/ophys_experiment_table.csv` directly with pandas, and intersects it with the set of NWB files actually present on disk (`behavior_ophys_experiments/*.nwb`, 284 files), so only locally available experiments are ever requested. From that table it selects experiments with `behavior_type == 'active_behavior'` and `experience_level == 'Familiar'` (110 experiments → 92 ophys sessions → 38 mice; 88 single-plane `VisualBehavior` sessions plus 4 multi-plane `VisualBehaviorMultiscope` sessions). Each selected experiment is loaded with `cache.get_behavior_ophys_experiment(oeid)`, and from each dataset object the AI pulls `events`, `ophys_timestamps`, `running_speed`, `eye_tracking`, `stimulus_presentations`, `trials` and `metadata`. Conversion is done session-by-session in a 12-process `multiprocessing.Pool`. Passive sessions and Novel/Novel>1 sessions are never loaded.

ii.
```python
def get_experiment_table():
    tbl = pd.read_csv(os.path.join(
        CACHE_DIR, 'visual-behavior-ophys-1.1.0', 'project_metadata',
        'ophys_experiment_table.csv'))
    have = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in glob.glob(
        os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0',
                     'behavior_ophys_experiments', '*.nwb'))]
    return tbl[tbl.ophys_experiment_id.isin(have)]


def select_sessions():
    """Return {ophys_session_id: [ophys_experiment_id, ...]} for the sessions used."""
    tbl = get_experiment_table()
    sel = tbl[(tbl.behavior_type == 'active_behavior') &      # mouse doing the task
              (tbl.experience_level == 'Familiar')]           # familiar image set
    sel = sel.sort_values(['ophys_session_id', 'ophys_experiment_id'])
    return sel.groupby('ophys_session_id').ophys_experiment_id.apply(list).to_dict()
```
```python
    cache = VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)
    planes = []
    for oeid in oeids:
        ds = cache.get_behavior_ophys_experiment(int(oeid))
```
```python
    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        results = pool.map(_worker, items)
```

iii. From the trajectory: the AI explored the cache layout, found that `from_local_cache` (not `from_s3_cache`/static cache) matches the on-disk structure, and verified that a single experiment loads in ~2.7 s. It restricted the selection on two grounds, both taken from the paper's methods (`/app/methods.txt`: *"For neural analysis we used neurons recorded during familiar image set presentations on the multi-plane imaging rig"*): (1) **active behavior only**, because "in passive sessions the lick spout is retracted, so there are no choices, rewards or trial outcomes to decode" — trial outcome is a required decoder output; (2) **Familiar only**, following the paper, with the added practical benefit that "every familiar active session in this release used image set A, so the eight image identities are the same in every session and share one set of output labels." It deliberately re-added the Multiscope sessions late in the run (step 64) after realising the paper's neural analysis used the multi-plane rig, once the fixed 250 ms grid made the differing rig frame rates irrelevant.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values taken from the per-session SDK metadata (`ds.metadata['mouse_id']`) of the sessions that survive conversion. `subjects` is the sorted list of those ids (38 mice), and `subject_idx` is the index of each converted session's mouse into that list.

ii.
```python
    info = {
        ...
        'mouse_id': str(md['mouse_id']),
```
```python
    subjects = sorted({r['info']['mouse_id'] for r in results})
    ...
        'subject_idx': np.array([subjects.index(r['info']['mouse_id'])
                                 for r in results], dtype=np.int64),
```

iii. Not discussed at length; `mouse_id` is the SDK's canonical animal identifier. The AI cross-checked in its survey that the 284 locally available experiments belong to 38 mice and reported "38 mice" in its final sanity check, confirming that every mouse contributes at least one familiar/active session.

## 1-c. How are the data split into sessions?

i. One converted session per `ophys_session_id`. `select_sessions()` groups the selected experiments by `ophys_session_id`, so the several simultaneously imaged planes of a Multiscope session (3–7 planes) are merged into a single converted session: their neurons are stacked along the neuron axis and each neuron keeps the `targeted_structure` of its own plane (VISp or VISl). Behavioural streams (running, eye tracking, trials, stimulus presentations) are taken from the first plane's dataset, since they are shared across planes of a session. Sessions are ordered by `ophys_session_id`. Result: 92 candidate sessions, 91 kept.

ii.
```python
    return sel.groupby('ophys_session_id').ophys_experiment_id.apply(list).to_dict()
```
```python
    ds = planes[0]['ds']     # behavior streams are shared by all planes of a session
```
```python
    regions = sum([[p['region']] * p['ncells'] for p in planes], [])
```
```python
        'brain_region_idx': [np.array([brain_regions.index(reg) for reg in r['regions']],
                                      dtype=np.int64) for r in results],
```

iii. "For Multiscope (multi-plane) sessions the simultaneously imaged planes are merged into a single session, with each neuron labeled by the brain region of its plane (VISp / VISl)." The AI's stated reason for merging is that planes imaged simultaneously in one behavioural session share one behaviour/stimulus stream, so they are one recording; keeping them separate would duplicate the same trials. Because each plane is binned from *its own* ophys timestamps onto the shared 250 ms grid, the different per-plane timestamp grids of the Mesoscope are handled correctly.

## 1-d. How are the data split into trials?

i. Trials are rows of the SDK `trials` table with `go == True` or `catch == True` and a non-NaN `change_time`; this excludes aborted and auto-rewarded trials by construction. Each trial spans its own `[start_time, stop_time)` window, but is represented by the whole 250 ms bins that fit inside that window, with a bin edge placed exactly at the trial's `change_time` (sham change time on catch trials). Trials therefore have variable length (median 31–33 bins ≈ 8 s: roughly 3–7 s before the change and 4.25 s after).

ii.
```python
    # ---- trials: go and catch only (excludes aborted and auto-rewarded) ----
    tr = ds.trials
    tr = tr[(tr.go | tr.catch) & tr.change_time.notna()]
```
```python
    for _, trial in tr.iterrows():
        oc = [k for k in OUTCOME_NAMES if bool(trial[k])]
        if len(oc) != 1:
            continue
        # fixed 250 ms bins with an edge exactly at the (sham) change time
        k0 = int(np.ceil((trial.start_time - trial.change_time) / BIN_SIZE))
        k1 = int(np.floor((trial.stop_time - trial.change_time) / BIN_SIZE))
        edges = trial.change_time + BIN_SIZE * np.arange(k0, k1 + 1)
```

iii. The instructions explicitly asked for Go and Catch trials only; the AI noted that selecting `go | catch` "excludes the aborted and auto-rewarded trials by construction" (rather than negating the `aborted`/`auto_rewarded` flags). It kept the full trials-table window instead of a fixed symmetric window around the change (step 40: "segment trials using the trials-table start_time/stop_time (the experiment's own trial definition)") so that each trial contains several pre-change image flashes, which is what makes image identity and image change genuinely time-varying within a trial.

## 1-e. How are trials filtered based on quality controls?

i. Beyond the go/catch + valid-`change_time` selection, a trial is dropped when:
- it does not have exactly one of the four outcome flags set (`len(oc) != 1`);
- it yields fewer than 2 complete bins (`len(edges) < 3`);
- its window is not fully covered by *all* data streams (ophys timestamps of every plane, running timestamps, eye-tracking timestamps): `edges[0] < tmin or edges[-1] > tmax`;
- any bin contains no ophys frame of some plane, or no running sample, or no eye-tracking sample (`np.any(nfr < 1)`, `np.any(rn < 1)`, `np.any(pn < 1)`);
- any bin center falls before the first image presentation (`np.any(j < 0)`).

Whole sessions are dropped when there is no eye tracking or no running data, when no plane has events, or when fewer than 2 usable trials remain. Any session that raises an exception is caught in `_worker`, reported, and skipped.

ii.
```python
    tmin = max([p['ts'][0] for p in planes] + [run_t[0], eye_t[0]])
    tmax = min([p['ts'][-1] for p in planes] + [run_t[-1], eye_t[-1]])
    ...
        if len(edges) < 3 or edges[0] < tmin or edges[-1] > tmax:
            continue             # trial not fully covered by all data streams
```
```python
        for p in planes:
            vals, nfr = bin_sum(p['cum'], p['ts'], edges)
            if np.any(nfr < 1):
                ok = False
                break
        ...
        rsum, rn = bin_sum(run_cum, run_t, edges)
        psum, pn = bin_sum(pupil_cum, eye_t, edges)
        if np.any(rn < 1) or np.any(pn < 1):
            continue
```
```python
    if len(trials) < 2:
        return None
```
```python
    if eye is None or len(eye) == 0:
        return None              # no eye tracking -> pupil output cannot be made
```

iii. The AI's rule is "no partially observed trials": since every output stream must be defined in every bin, a trial is kept only if all four streams (ophys, running, eye, stimulus) cover the whole trial window with at least one sample per bin. This avoids having to impute values in empty bins. The ≥2-trial rule follows the explicit format requirement ("There needs to be at least two trials within each session in order to evaluate the decoder performance"). One session was dropped for missing eye tracking (found in the AI's pre-conversion survey of all 110 experiments), leaving 91 sessions / 22,752 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is the SDK's **detected calcium events** table, `ds.events['events']` (one row per cell, one column per ophys frame), together with `ds.ophys_timestamps` for the times of those frames. `filtered_events` is exposed as a CLI option (`--neural filtered_events`) but the default, and what was used for `/app/converted_data.pkl`, is the raw detected `events`.

ii.
```python
        ev = ds.events
        if len(ev) == 0:
            continue
        act = np.vstack(ev[neural_key].values).astype(np.float64)
        assert act.shape[1] == len(ts)
```
```python
    ap.add_argument('--neural', default='events', choices=['events', 'filtered_events'])
```

iii. Directly from the paper's methods, which the AI quoted in its design notes: *"We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f."* The AI's docstring says "detected calcium events ... as used in the reference paper". Notably the AI ran a pilot comparison and found `filtered_events` (half-Gaussian smoothed) decoded *better* (0.332 vs 0.300 balanced accuracy for image identity) yet still chose raw `events` for consistency with the paper, compensating for their sparsity with larger time bins instead.

## 2-b. How is the `neural` data processed?

i. Two operations only: (1) events are **summed within each 250 ms bin** of the trial's grid, computed efficiently from a per-session cumulative sum evaluated at the bin edges; (2) the per-plane binned matrices are **vertically stacked** into one (n_neurons, n_bins) matrix per trial, cast to float32. No normalization, z-scoring, smoothing, or baseline subtraction is applied, and no neuron-level averaging.

ii.
```python
        planes.append({
            'oeid': int(oeid), 'ds': ds, 'ts': ts,
            'cum': np.concatenate([np.zeros((act.shape[0], 1)),
                                   np.cumsum(act, axis=1)], axis=1),
```
```python
def bin_sum(values_cumsum, timestamps, edges):
    """Sum of values within each bin of `edges`, plus the sample count per bin."""
    idx = np.searchsorted(timestamps, edges)
    if values_cumsum.ndim == 1:
        return values_cumsum[idx[1:]] - values_cumsum[idx[:-1]], np.diff(idx)
    return (values_cumsum[:, idx[1:]] - values_cumsum[:, idx[:-1]]), np.diff(idx)
```
```python
        neural_trial = np.vstack(neural_parts).astype(np.float32)
```

iii. Summing (rather than averaging) event magnitudes within a bin preserves total event "mass", which is the natural aggregation for discrete events. The AI justified binning empirically with pilot decoder runs: at the native 32 ms frame period the raw events (only ~0.3 % of samples non-zero) decoded barely above chance (image identity 0.170), whereas summing into ~250 ms bins raised validation accuracy to 0.296–0.300. Stacking planes preserves all simultaneously recorded neurons while each neuron keeps its own plane's region label. A consequence the AI noticed and accepted: in sessions with very few cells (some Sst sessions have 7–27 neurons) a number of trials contain no event at all — the format checker emitted 1,027 "all neural data is zero" warnings — which the AI described as "inherent to the data, not an error".

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering is applied: every ROI in `ds.events` is kept. Experiments whose `events` table is empty are skipped; sessions where no plane has events are dropped entirely.

ii.
```python
        ev = ds.events
        if len(ev) == 0:
            continue
    ...
    if not planes:
        return None
```
```python
        'selection_criteria': (... 'all ROIs in the released data already passed the '
                               'Allen Institute cell segmentation and ROI-filtering QC'),
```

iii. The AI checked this explicitly: in its survey over all 110 candidate experiments it verified that the `valid_roi` flag is True for every released ROI ("all valid ROIs"), i.e. the Allen pipeline's segmentation/ROI QC has already been applied to the released data, so no further cell filtering is warranted. The whitepaper describes this QC (cell segmentation, crosstalk and neuropil correction, demixing) as part of the released pipeline.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the trial's **image change time** (`trials.change_time`, the sham change time on catch trials). The 250 ms bin grid of a trial is constructed so that one bin edge lies exactly at `change_time`: bin edges are `change_time + 0.25 * k` for the integer k that keep the bins inside `[start_time, stop_time)`. Neural data are then binned from the native `ophys_timestamps` of each plane onto that grid, so alignment is on the ophys clock, with a residual quantization of one 250 ms bin. `metadata['temporal_alignment_event']` documents this, and `off_start`/`off_end` are given as the typical (median) values −3.75 s and +4.0 s with an explicit range field, because the pre-change interval varies from trial to trial.

ii.
```python
        k0 = int(np.ceil((trial.start_time - trial.change_time) / BIN_SIZE))
        k1 = int(np.floor((trial.stop_time - trial.change_time) / BIN_SIZE))
        edges = trial.change_time + BIN_SIZE * np.arange(k0, k1 + 1)
```
```python
            'temporal_alignment_event': (
                'image change time of the trial (sham change time on catch trials): the '
                '250 ms time bins are laid out with an edge exactly at the change time, '
                'and every data stream (ophys calcium events, stimulus, running, pupil) is '
                'binned from its own native timestamps onto that common grid'),
            'off_start': -3.75,   # typical (median) trial start relative to the change
            'off_end': 4.0,       # typical (median) trial end relative to the change
            'off_start_range': [-11.0, -3.0],
            'off_end_range': [3.75, 4.25],
```

iii. "trials are aligned to the change time (the sham change time on catch trials) and binned on a fixed 250 ms grid (the duration of one image flash) with a bin edge exactly at the change time, so bin edges are locked to the stimulus flash cycle." The change is the behaviourally and visually relevant event of the change-detection task, and anchoring the grid there also makes every bin fall entirely inside either an image flash or a gray period, which keeps the image-identity and image-change labels unambiguous within a bin. At step 71–72 the AI measured the actual distribution of `start_time - change_time` (median −3.77 s, range −3.0 to −6.8 s) and `stop_time - change_time` (≈ +4.23 s) before filling in the metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — explicit rebinning. All data are resampled onto a **fixed 250 ms grid** (`BIN_SIZE = 0.25 s`, `metadata['time_bin_size'] = 250.0` ms), identical for every trial, session and rig. This is a ~8× coarsening for single-plane experiments (native 30.94 Hz ≈ 32 ms frames) and ~2.7× for Multiscope planes (~10.7 Hz ≈ 93 ms). Each stream is binned from its own native timestamps: neural events are summed per bin, running speed and pupil diameter averaged per bin, stimulus labels evaluated at the bin centers. Median trial length is 33 bins (~8.25 s).

ii.
```python
BIN_SIZE = 0.25          # s, duration of one image flash
FLASH_INTERVAL = 0.75    # s, image presentation interval (250 ms image + 500 ms gray)
```
```python
        centers = 0.5 * (edges[:-1] + edges[1:])
        ...
        run_trial = rsum / rn
        pupil_trial = psum / pn
```
```python
            'time_bin_size': BIN_SIZE * 1000.0,
```

iii. Three reasons, all stated by the AI: (1) 250 ms is the image flash duration, so the grid is locked to the 750 ms flash cycle; (2) a fixed bin "makes single-plane (~31 Hz) and Multiscope (~11 Hz) recordings commensurate", which is required by the format ("Time bins should be the same size for all trials and sessions") once both rigs are included; (3) empirically, sparse detected events are nearly undecodable at 32 ms resolution and much more informative when summed over ~250 ms — the AI ran a bin-size sweep on a pilot (k = 1, 3, 6, 8 frames) and observed image-identity validation accuracy rising from 0.170 to 0.300.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the `stimulus_presentations` table of the change-detection block: `image_name`, `start_time`, and `omitted`. Presentations are restricted to `stimulus_block_name == 'change_detection_behavior'`, omitted flashes are removed, and the remaining flashes are sorted by `start_time`. The set of image labels is `sorted(unique(image_name) - {'omitted'})` (8 images of image set A in every kept session). The trials table's `initial_image_name`/`change_image_name` are *not* used.

ii.
```python
    sp = ds.stimulus_presentations
    sp = sp[sp.stimulus_block_name == 'change_detection_behavior']
    image_names = sorted(set(sp.image_name.unique()) - {'omitted'})
    shown = sp[(~sp.omitted) & (sp.image_name != 'omitted')].sort_values('start_time')
    pres_start = shown.start_time.values
    pres_img = np.array([image_names.index(n) for n in shown.image_name.values])
```

iii. The stimulus-presentation table is the ground truth for what was actually on the screen at each moment, flash by flash, rather than the two-image-per-trial summary in the trials table. Filtering to the change-detection block excludes the fingerprint/gratings blocks. Omitted flashes (5 % of repeats, per the paper) are removed from the lookup so that the identity of the last actually-shown image is carried through an omission, which is the paper's convention of assigning data to image presentation intervals.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes 0–7 by their position in the per-session sorted image list; the AI asserts in `main()` that all sessions have identical image name lists, so the codes are globally consistent. For each 250 ms bin, the code is that of the most recent preceding flash (`searchsorted(..., side='right') - 1`), i.e. identity is held across the 500 ms gray screen and across omitted flashes. The result is an int64 time series of length n_bins per trial, stored as row 0 of the `output` matrix with `output_values[0] = ['im061', 'im062', ...]`.

ii.
```python
        j = np.searchsorted(pres_start, centers, side='right') - 1
        if np.any(j < 0):
            continue
        image_trial = pres_img[j].astype(np.int64)
```
```python
    image_names = results[0]['image_names']
    assert all(r['image_names'] == image_names for r in results), 'image sets differ'
```

iii. "Stimulus variables follow the reference paper's convention of assigning data to the 750 ms image presentation interval (250 ms flash + 500 ms gray screen), so the image identity is held across the gray screen and across omitted flashes." Holding the label avoids an artificial "gray" class and matches the standard analysis convention for this dataset, where responses are aligned to and attributed to the image presentation that started them. Restricting the dataset to the familiar image set A guarantees a single shared 8-class label set, which the code asserts rather than assumes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated at the **centers of exactly the same 250 ms bins** used for the neural data, so the image row has the same length as the neural matrix by construction and both are anchored to the same change-time-locked grid. Because bin edges coincide with the flash cycle, a bin never straddles two different images.

ii.
```python
        centers = 0.5 * (edges[:-1] + edges[1:])
        ...
        j = np.searchsorted(pres_start, centers, side='right') - 1
        image_trial = pres_img[j].astype(np.int64)
        ...
        output.append(np.stack([
            t['image'],
            ...
```

iii. The AI's stated principle: one shared grid, each stream binned from its own native timestamps, so all streams are aligned to the ophys clock without resampling one stream through another. Its final sanity check confirmed that all 19,888 image-change onsets coincide with an image-identity transition, and that catch trials never carry a change — an end-to-end check that the identity and change series are correctly aligned with each other and with the trial structure.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the same filtered `stimulus_presentations` table: the `is_change` boolean column combined with `start_time` of each presentation. The start times of all presentations flagged `is_change` are extracted as `change_start`.

ii.
```python
    change_start = pres_start[shown.is_change.values]
```

iii. `is_change` is the SDK's per-flash flag marking a flash whose image differs from the preceding one, i.e. the actual visual change event. Using it (rather than the trials table's `change_time` + `go`) keeps image change defined by what was displayed, consistent with how image identity was derived, and automatically gives 0 on catch trials, where no real change occurs.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each bin center the AI finds the most recent `is_change` flash and sets the indicator to 1 if that center is less than 750 ms after that flash's onset, else 0. Bins before the first change of the session get 0. The result is a binary int64 time series per trial (row 1 of `output`), with `output_values[1] = ['no_change', 'change']`. Empirically 7.65 % of bins are 1 (reference: 7.05 %).

ii.
```python
        # image change: 1 during the presentation interval of the changed image
        jc = np.searchsorted(change_start, centers, side='right') - 1
        change_trial = np.zeros(len(centers), dtype=np.int64)
        good = jc >= 0
        change_trial[good] = (
            (centers[good] - change_start[jc[good]]) < FLASH_INTERVAL).astype(np.int64)
```

iii. The instruction is "value of 1 right after a change in image identity, otherwise 0". The AI implements "right after" as the changed image's full 750 ms presentation interval (one 250 ms flash + the following 500 ms gray), the same interval convention used for image identity, giving 3 consecutive bins of 1 per change. This both matches the flash cycle and keeps the positive class large enough (~7.6 %) to be learnable.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: 1 during the 750 ms presentation interval of a changed image, 0 everywhere else. No continuous quantity is thresholded. Catch (sham change) trials are 0 throughout, since their sham change is not an `is_change` presentation.

ii.
```python
FLASH_INTERVAL = 0.75    # s, image presentation interval (250 ms image + 500 ms gray)
...
        change_trial[good] = (
            (centers[good] - change_start[jc[good]]) < FLASH_INTERVAL).astype(np.int64)
```
```python
            ['no_change', 'change'],
```

iii. The AI verified the semantics in its final check: "19888 change onsets, 0 of which occur without an image identity transition; change count exactly equals hit+miss (go trials), and catch trials correctly have no change." The 750 ms width is justified as one image presentation interval, i.e. the window during which the changed image is the current stimulus.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity — evaluated at the centers of the shared 250 ms bins, so it has exactly the same length and timebase as the neural matrix, with the change falling at a bin boundary because the grid is anchored to `change_time`.

ii.
```python
        jc = np.searchsorted(change_start, centers, side='right') - 1
        ...
        output.append(np.stack([
            t['image'],
            t['change'],
            ...
```

iii. Anchoring bin edges at the trial's change time means the first post-change bin starts exactly at the change, so no bin mixes pre- and post-change neural activity — this was one of the AI's explicit reasons for choosing change-time-anchored bins.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ds.running_speed`, using its `speed` column (cm/s from the running-disk encoder, processed by the SDK's `running_processing` module) and its own `timestamps` column (behaviour-camera/stimulus clock, ~60 Hz), not the ophys timestamps.

ii.
```python
    run = ds.running_speed
    run_t = run.timestamps.values
    run_v = interp_nans(run.speed.values)
    if run_v is None:
        return None
    run_cum = np.concatenate([[0.0], np.cumsum(run_v)])
```

iii. `running_speed` is the SDK's standard locomotion product; the whitepaper section in `methods.txt` describes the unwrapping/filtering that produces it, so no further preprocessing of the raw encoder voltage is needed. Keeping the native timestamps lets the AI bin the stream directly onto the common grid rather than interpolating it onto ophys frames.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. (1) Any non-finite samples are linearly interpolated (by sample index); (2) the speed trace is averaged within each 250 ms bin, implemented as the cumulative sum difference divided by the number of samples in the bin; (3) the resulting per-bin means are pooled over all kept trials of the session and discretized into five equal-percentile bins. Bins containing no running sample cause the trial to be dropped, so no imputation is needed at this stage.

ii.
```python
def interp_nans(x):
    """Linearly interpolate NaNs (blinks / lost tracking) in a 1d time series."""
    x = np.asarray(x, dtype=float)
    bad = ~np.isfinite(x)
    if bad.all():
        return None
    if bad.any():
        idx = np.arange(len(x))
        x = x.copy()
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x
```
```python
        rsum, rn = bin_sum(run_cum, run_t, edges)
        ...
        run_trial = rsum / rn
```
```python
    run_code = quantile_bins(np.concatenate([t['run'] for t in trials]))
```

iii. Averaging (not summing) is the correct aggregation for an intensive quantity like speed. Because the running stream is sampled ~2× faster than the 250 ms bin at minimum, each bin gets several samples, so the bin mean is a faithful low-pass estimate; the AI preferred this to interpolating the stream onto a sparser neural timebase.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Into five **equal-percentile (quintile) bins computed per session**, over all time bins of all kept trials in that session. `np.quantile` gives the four interior edges and `np.searchsorted(..., side='right')` assigns codes 0–4. Because the edges are session-specific, each session has almost exactly 20 % of its bins in each class (measured global fractions 0.2000 each).

ii.
```python
def quantile_bins(values, nbins=NQUANTILES):
    """Discretize into nbins equal-percentile bins; returns int codes 0..nbins-1."""
    edges = np.quantile(values, np.linspace(0, 1, nbins + 1)[1:-1])
    return np.searchsorted(edges, values, side='right').astype(np.int64)
```
```python
    # Running speed and pupil diameter are discretized into five equal-percentile
    # bins computed within each session, over the time bins included in the trials.
    # Mice differ strongly in running and pupil statistics and pupil size is measured
    # in camera pixels, so its scale is session specific; a session-wise quantization
    # is therefore the meaningful one.
    run_code = quantile_bins(np.concatenate([t['run'] for t in trials]))
```

iii. Quoted directly from the code comment above: mice differ strongly in their running statistics, so session-wise quantiles keep the five classes balanced and interpretable within each session (the unit in which the decoder is evaluated), rather than having whole sessions collapse into one or two global bins. The instruction only asked for "five equal percentile bins" without specifying the pooling scope.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. By construction: the running trace is binned from its own native timestamps onto the same change-time-anchored 250 ms edges used for the neural data, giving one value per neural time bin. Trials are only kept if the running stream covers the whole trial window with ≥1 sample in every bin, so there are no gaps or extrapolated values.

ii.
```python
    tmin = max([p['ts'][0] for p in planes] + [run_t[0], eye_t[0]])
    tmax = min([p['ts'][-1] for p in planes] + [run_t[-1], eye_t[-1]])
    ...
        rsum, rn = bin_sum(run_cum, run_t, edges)
        psum, pn = bin_sum(pupil_cum, eye_t, edges)
        if np.any(rn < 1) or np.any(pn < 1):
            continue
```

iii. "The grid is shared by all data streams; each stream is binned from its own native timestamps (ophys frame times for the neural data, stimulus times for the images, behavior/eye camera times for running and pupil)." The SDK timestamps of all streams are already on the synchronised experiment clock, so binning each stream onto a common grid aligns them without any interpolation artefacts.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. From `ds.eye_tracking`: the `pupil_area` column (ellipse-fit pupil area in camera pixels²) and the table's `timestamps`. The AllenSDK already sets `pupil_area` to NaN on frames flagged `likely_blink`, so blink frames enter as NaN and are interpolated.

ii.
```python
    try:
        eye = ds.eye_tracking
    except Exception:
        eye = None
    if eye is None or len(eye) == 0:
        return None              # no eye tracking -> pupil output cannot be made
    pupil_area = interp_nans(eye.pupil_area.values)     # NaN on likely-blink frames
    if pupil_area is None:
        return None
    pupil_v = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)   # diameter, pixels
    eye_t = eye.timestamps.values
```

iii. The whitepaper describes the DeepLabCut-based ellipse fits and the blink detection; the AI's comment shows it knew that the SDK nullifies pupil measurements on likely-blink frames (which is exactly what `eye_tracking_processing.py` does). It chose `pupil_area` and converted it to an equivalent-circle diameter so the reported quantity is literally "pupil diameter" as the instructions ask, in camera pixels (no eye-to-camera calibration is applied).

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) NaNs (blinks / lost tracking) are linearly interpolated by sample index; (2) area is converted to diameter, `d = 2*sqrt(area/pi)`, with negatives clipped to 0; (3) the trace is averaged within each 250 ms bin from its native ~60 Hz timestamps; (4) per-bin means are pooled across the session's kept trials and discretized into five equal-percentile bins. Sessions with no eye tracking, or with an all-NaN pupil trace, are dropped entirely (one session).

ii.
```python
    pupil_area = interp_nans(eye.pupil_area.values)     # NaN on likely-blink frames
    pupil_v = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)   # diameter, pixels
    eye_t = eye.timestamps.values
    pupil_cum = np.concatenate([[0.0], np.cumsum(pupil_v)])
```
```python
        psum, pn = bin_sum(pupil_cum, eye_t, edges)
        ...
        pupil_trial = psum / pn
```
```python
    pupil_code = quantile_bins(np.concatenate([t['pupil'] for t in trials]))
```

iii. Interpolating rather than dropping blink frames keeps the pupil time series continuous so every bin has a defined value; the AI documents this in the metadata as "blinks linearly interpolated". The area→diameter conversion makes the variable match the requested "pupil diameter"; note that since the final output is a quantile code, this monotone transform has no effect on the discretized result.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identically to running speed: five equal-percentile bins with edges computed **per session** from all kept time bins of that session, giving codes 0–4 (`pupil_quintile_1..5`) and ~20 % of bins per class.

ii.
```python
    pupil_code = quantile_bins(np.concatenate([t['pupil'] for t in trials]))
```
```python
            ['pupil_quintile_%d' % (i + 1) for i in range(NQUANTILES)],
```

iii. The AI's comment is strongest for pupil: "pupil size is measured in camera pixels, so its scale is session specific; a session-wise quantization is therefore the meaningful one." Camera position, zoom and eye size differ between sessions, so absolute pixel thresholds are not comparable across sessions, while within-session quintiles measure relative arousal state.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running: binned from the eye-tracking camera's native timestamps onto the same shared 250 ms grid, with the trial dropped if the eye stream does not fully cover the window or if any bin lacks an eye-tracking sample. The result is one value per neural time bin.

ii.
```python
    tmin = max([p['ts'][0] for p in planes] + [run_t[0], eye_t[0]])
    tmax = min([p['ts'][-1] for p in planes] + [run_t[-1], eye_t[-1]])
    ...
        psum, pn = bin_sum(pupil_cum, eye_t, edges)
        if np.any(rn < 1) or np.any(pn < 1):
            continue
```

iii. Same justification as running speed — all SDK streams live on the synchronised experiment clock, so binning each onto the shared grid aligns them with the neural data without resampling through an intermediate timebase.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. From the four mutually exclusive boolean columns of the SDK trials table: `hit`, `miss`, `false_alarm`, `correct_reject`.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
        oc = [k for k in OUTCOME_NAMES if bool(trial[k])]
        if len(oc) != 1:
            continue
```

iii. These are the canonical change-detection outcome labels: hit/miss for go trials (real change, licked or not) and false alarm/correct reject for catch trials (sham change, licked or not). Since only go and catch trials are kept, exactly one of the four is true for every valid trial, which the AI enforces rather than assumes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome name is mapped to its index in `OUTCOME_NAMES` (0 = hit, 1 = miss, 2 = false_alarm, 3 = correct_reject) and, since the format prefers time-varying outputs, the constant code is **tiled across all time bins of the trial** as row 4 of the `output` matrix. Trials where the four flags do not select exactly one outcome are dropped instead of being given a fallback code.

ii.
```python
        trials.append({..., 'outcome': OUTCOME_NAMES.index(oc[0])})
```
```python
        output.append(np.stack([
            t['image'],
            t['change'],
            run_code[pos:pos + T],
            pupil_code[pos:pos + T],
            np.full(T, t['outcome'], dtype=np.int64),
        ]).astype(np.int64))
```
```python
            list(OUTCOME_NAMES),
```

iii. The instruction lists trial outcome as "Static per-trial", but the format spec says "If at all possible, make it time-varying"; tiling satisfies both and keeps all five outputs in one (5, n_bins) array as the verifier requires. The resulting distribution (hit 26 %, miss 49 %, false alarm 5 %, correct reject 19 % in the AI's data) reflects the ~4:1 go:catch design of active sessions.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. A layered strategy:
- **Per-session exceptions**: each session is converted inside `_worker`, which catches any exception, prints `FAILED session <id>: ...`, and returns `None` so the rest of the dataset still converts.
- **Missing eye tracking / all-NaN pupil / missing running**: the session is dropped (`return None`); one session was lost this way.
- **Experiments without events**: skipped; if no plane of a session has events, the session is dropped.
- **NaNs inside behavioural traces** (blinks, lost tracking): linearly interpolated by `interp_nans` before binning; an all-NaN trace returns `None`.
- **Incomplete coverage**: trials whose window is not fully inside all streams' time ranges, or that contain a bin with no sample from some stream, are skipped rather than padded or imputed.
- **Ambiguous outcome flags**: trials without exactly one outcome flag are skipped.
- **Degenerate sessions**: sessions with fewer than 2 usable trials are dropped.
- **Cross-session consistency**: `main()` asserts that every kept session has the identical image-name list, so the shared `output_values` mapping cannot be silently wrong.

ii.
```python
def _worker(args):
    osid, oeids, neural_key = args
    try:
        return convert_session(osid, oeids, neural_key)
    except Exception as exc:
        print('FAILED session %s: %s: %s' % (osid, type(exc).__name__, exc), flush=True)
        return None
```
```python
    bad = ~np.isfinite(x)
    if bad.all():
        return None
    if bad.any():
        ...
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
```
```python
    assert all(r['image_names'] == image_names for r in results), 'image sets differ'
```

iii. The AI's guiding principle is to never fabricate a label: where a value can be recovered by short-range interpolation (blinks) it interpolates; where it cannot (a bin with no sample, a stream that does not cover the trial, an undefined outcome) it removes the affected trial or session, so no missing-data sentinel ever reaches the decoder. It accepted one known artefact — trials in low-cell-count sessions whose event matrix is entirely zero — as a genuine property of sparse detected events rather than a defect.

## 9-a. What are the most time-consuming steps of the code?

i. Loading the NWB files through the SDK dominates: `cache.get_behavior_ophys_experiment()` takes ~2.7 s per experiment (the AI timed it) for 110 experiments, and each call also parses running, eye-tracking, stimulus and trials tables. Next are the per-session cumulative sums over the full events matrix (n_neurons × n_frames, float64), the Python `for _, trial in tr.iterrows()` loop over ~250 trials per session, and finally pickling the 526 MB result. The AI mitigated the I/O cost with a 12-process pool, bringing the full conversion down to a few minutes.

ii.
```python
    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        results = pool.map(_worker, items)
```
```python
            'cum': np.concatenate([np.zeros((act.shape[0], 1)),
                                   np.cumsum(act, axis=1)], axis=1),
```

iii. The AI measured single-experiment load time during exploration (step 25: "SDK loads fine (2.7 s/experiment)") and designed around it: a pilot mode (`--limit`) for iteration and multiprocessing for the full run. It never re-loads an experiment: everything needed from a session is extracted in a single pass.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The binning itself is already vectorized — `bin_sum` turns per-bin aggregation into two `searchsorted` calls and a cumulative-sum difference, and the stimulus labels are computed with `searchsorted` over all bin centers at once. What remains scalar is the `tr.iterrows()` trial loop (one iteration per trial, each doing a handful of small array ops) and the `for p in planes` loop inside it; the per-trial edge computation and `bin_sum` calls could in principle be done for all trials of a session at once by concatenating their edge arrays into a single `searchsorted`. The final assembly loops (`for t in trials`, list comprehensions over sessions) are also per-trial but trivial.

ii.
```python
    for _, trial in tr.iterrows():
        ...
        for p in planes:
            vals, nfr = bin_sum(p['cum'], p['ts'], edges)
```
```python
    idx = np.searchsorted(timestamps, edges)
    return (values_cumsum[:, idx[1:]] - values_cumsum[:, idx[:-1]]), np.diff(idx)
```

iii. The AI does not comment on this explicitly, but the design shows the intent: the expensive inner operation (summing events over arbitrary bin boundaries for hundreds of neurons) was deliberately replaced by the cumsum + searchsorted trick, so the remaining Python loop runs a fixed small number of vectorized calls per trial and is negligible next to NWB loading.

## 9-c. What processing does the code repeat multiple times?

i. A few genuine repetitions:
- `VisualBehaviorOphysProjectCache.from_local_cache(...)` is re-instantiated inside `convert_session` for every session (92 times), re-reading the project manifest each time; it is also implicitly duplicated across the 12 worker processes.
- `get_experiment_table()` reads and globs the cache once in `select_sessions()`, in addition to the manifest reads the SDK does.
- For multi-plane Mesoscope sessions, every plane's NWB is fully parsed, which re-parses the *same* behaviour, eye-tracking, stimulus and trials tables once per plane, although only `planes[0]`'s copies are used.
- `np.searchsorted(pres_start, ...)` / `np.searchsorted(change_start, ...)` are recomputed per trial rather than once per session.

ii.
```python
def convert_session(osid, oeids, neural_key='events'):
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    cache = VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)
```
```python
    ds = planes[0]['ds']     # behavior streams are shared by all planes of a session
```

iii. Re-creating the cache per session is a consequence of the multiprocessing design (the cache object is not picklable across processes, so it must be built inside the worker); the AI accepted the manifest re-read as cheap relative to NWB parsing. Nothing substantive (loading an experiment, binning a stream) is done twice for the same data within a worker.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items:
- The area→diameter conversion `2*sqrt(area/pi)` is monotone and therefore has no effect after quantile discretization; the quintile codes would be identical from the raw area.
- Similarly, interpolating NaNs by *index* rather than by timestamp, and computing the exact per-bin means, is finer than the 5-level discretization ultimately needs.
- Cumulative sums are built over the **entire session** for every stream (events for all neurons over all frames, running, pupil), while only the bins inside kept trials are read out; for events this is the largest intermediate array in the program and is cast to float64.
- Each plane's full `BehaviorOphysExperiment` object is kept alive in the `planes` dict (`'ds': ds`) for the whole session conversion even though only its metadata is needed after extraction.
- Metadata that is collected per session but not used by the decoder: `cell_specimen_ids`, `imaging_depths`, `ophys_frame_rates`, `behavior_session_id`, `cre_line`, `equipment_name`, `project_code`.
- A zero-width input array `np.zeros((0, T))` is materialized for every trial, since the task specifies no decoder inputs.
- `--neural filtered_events` support and the whole pilot/limit machinery remain in the shipped script but are not exercised by the final run.

ii.
```python
    pupil_v = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)   # diameter, pixels
```
```python
            'cum': np.concatenate([np.zeros((act.shape[0], 1)),
                                   np.cumsum(act, axis=1)], axis=1),
```
```python
        'input': [[np.zeros((0, t.shape[1]), dtype=np.float32) for t in r['neural']]
                  for r in results],
```
```python
        'cell_specimen_ids': sum([p['cell_ids'] for p in planes], []),
```

iii. Most of these are deliberate: the extra metadata is provenance the format explicitly invites (`session_info`), the empty input arrays satisfy the required `input` key with `d_input = 0`, and the diameter conversion exists so the saved variable means what its name says even though the discretization erases the transform. The session-wide cumulative sums are the price of the vectorized binning scheme — they make per-trial extraction O(n_bins) instead of O(n_frames), at the cost of one full-session pass and its memory.
