# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All access goes through the AllenSDK `VisualBehaviorOphysProjectCache`, opened as a **local** cache
(`from_local_cache(cache_dir='/app/data', use_static_cache=False)`; the S3/static variants are unusable offline).
`cache.get_ophys_experiment_table()` gives the canonical table of all released imaging planes, but the local cache only
holds 284 of the 1,936 released NWB files, so the table is intersected with the experiment ids actually present on disk
(`available_experiment_table`). The table is then filtered to **active-behavior experiments only** (`~passive`, i.e.
OPHYS_1/3/4/6; the passive OPHYS_2/OPHYS_5 sessions are dropped), leaving 202 experiments / 174 ophys sessions / 38 mice.
No `project_code` filter is applied, so both `VisualBehavior` (single-plane, 168 active sessions) and
`VisualBehaviorMultiscope` (6 active sessions, 3–7 planes each) are included. Experiments are grouped by
`ophys_session_id` and each group is handed to a worker process that calls
`cache.get_behavior_ophys_experiment(experiment_id)` for every plane of that session; sessions are processed in parallel
with a 16-way `multiprocessing.Pool`. No `.nwb` file is ever opened directly.

ii.
```python
def get_cache():
    return VisualBehaviorOphysProjectCache.from_local_cache(
        cache_dir=CACHE_DIR, use_static_cache=False)


def available_experiment_table(cache):
    '''Experiment table restricted to experiments whose NWB file is present locally.'''
    et = cache.get_ophys_experiment_table()
    avail = sorted(int(os.path.basename(f).split('_')[-1].split('.')[0])
                   for f in glob.glob(NWB_GLOB))
    sub = et.loc[et.index.isin(avail)].copy()
    return sub
```
```python
    cache = get_cache()
    et = available_experiment_table(cache)
    print(f'experiments with local NWB files: {len(et)}')
    active = et[~et['passive']].copy()
    ...
    groups = active.groupby('ophys_session_id')
    session_ids = sorted(groups.groups.keys())
    ...
    with Pool(min(args.nproc, len(jobs))) as pool:
        results = pool.map(process_session, jobs)
```
```python
    session_id, eids, region_map, want_debug, signal = args
    cache = get_cache()
    datasets = [cache.get_behavior_ophys_experiment(int(e)) for e in eids]
```

iii. From CONVERSION_NOTES.md Steps 1–2 and 4: the experiment table is "the SDK's canonical listing"; the metadata tables
"describe the **entire** published VBO dataset (1936 experiments …) but only **284 experiments** have NWB files present
locally -> the usable dataset is this subset", so the table must be restricted "BEFORE anything is loaded". Passive
sessions are excluded because they "have no behavioral responses/trial outcomes", and the paper itself states that
"Imaging was also performed during passive viewing of the same stimulus, which was not analyzed here". Decision 8 states
that "All active sessions of all cre lines, both image sets, both areas and all experience levels are kept … for building
a decoding dataset, more sessions is strictly better and the task says to convert data collected under the Visual Behavior
task", which is the stated reason no `project_code` restriction is applied. Parallelism and one cache handle per worker
are documented as the main speed-ups (90 s for the full conversion).

## 1-b. How are the data split into subjects?

i. Subjects are mice, identified by the string `mouse_id`. The id is read from the loaded experiment's metadata
(`ds0.metadata['mouse_id']`) rather than the experiment table, and subjects are registered in the order sessions are
emitted; `subject_idx` is the index of that session's mouse in the `subjects` list. 38 mice result.

ii.
```python
            mouse_id=str(ds0.metadata['mouse_id']),
```
```python
        if r['mouse_id'] not in subjects:
            subjects.append(r['mouse_id'])
        data['subject_idx'].append(subjects.index(r['mouse_id']))
    ...
    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. `mouse_id` is the SDK's unique animal identifier (CONVERSION_NOTES Step 1 table lists it as a column of the
experiment table and of `experiment.metadata`). The notes cross-check the count: "Subjects (mice) | 38" from the data
scan, versus 38 in the converted file (Step 9 consistency table), and note that the paper's 82 mice cannot be matched
because the local cache is a subset of the release.

## 1-c. How are the data split into sessions?

i. A session is one `ophys_session_id`. All `ophys_experiment_id`s (imaging planes) sharing that session id are loaded
together and their neurons **concatenated along the neuron axis**, because they are simultaneously recorded populations
that share one behavior session. The trials/stimulus/running/eye streams are taken from the first plane (`ds0`), after
verifying that all planes of a session have identical trial tables. Only active-behavior sessions enter the list
(174 sessions); 3 are later dropped for having no eye tracking, leaving 171 sessions.

ii.
```python
    groups = active.groupby('ophys_session_id')
    session_ids = sorted(groups.groups.keys())
    ...
    jobs = [(int(s), list(groups.get_group(s).index.values), region_map, i < n_debug,
             args.neural_signal)
            for i, s in enumerate(session_ids)]
```
```python
        datasets = [cache.get_behavior_ophys_experiment(int(e)) for e in eids]
        ds0 = datasets[0]
        ...
        for ds, eid in zip(datasets, eids):
            ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
            ...
            neural_planes.append(binned)
            region_per_neuron += [region_map[int(eid)]] * ev.shape[0]
        neural = np.concatenate(neural_planes, axis=0)  # (n_neurons, n_trials, N)
```

iii. Step 4/Step 5: "**Session** = one `ophys_session_id` (a single continuous recording). For Multiscope sessions the
3-7 simultaneously recorded imaging planes … are **concatenated along the neuron axis** into one session, because they are
simultaneously recorded neurons sharing one behavior session (verified: all planes of a session have identical trial
tables … 0/174 sessions disagree)". "Only **active behavior** sessions are used (passive OPHYS_2/OPHYS_5 have no
choices/outcomes)." Merging planes is also said to match "the whitepaper definition of a *session* vs an *experiment*".

## 1-d. How are the data split into trials?

i. Trials come from the SDK `trials` table of the session. A trial is kept if `(go | catch) & ~aborted & ~auto_rewarded`
and it has a non-NaN `change_time`. Each kept trial is cut as a **fixed window of [-2.25 s, +3.75 s] around
`trials.change_time`**, divided into **24 bins of 250 ms** — i.e. 8 flash cycles, 3 before and 5 after the change. All
trials in the dataset therefore have exactly the same length (24 timepoints). Sessions with fewer than 2 usable trials are
dropped. 41,904 trials from 171 sessions result.

ii.
```python
OFF_START = -2.25       # s relative to change time
OFF_END = 3.75          # s relative to change time
BIN_SIZE = 0.25         # s
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 24
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2.0
```
```python
        trials = ds0.trials
        sel = ((trials['go'] | trials['catch']) & (~trials['aborted'])
               & (~trials['auto_rewarded']))
        trials = trials[sel]
        trials = trials[~trials['change_time'].isna()]
        if len(trials) < 2:
            out['error'] = 'fewer than 2 go/catch trials'
            return out
        change_times = trials['change_time'].values.astype(np.float64)
        edges_abs = change_times[:, None] + BIN_EDGES[None, :]      # (n_trials, N+1)
        centers_abs = change_times[:, None] + BIN_CENTERS[None, :]  # (n_trials, N)
```

iii. Step 5: "**Trial** = a `trials` row with `(go | catch) & ~aborted & ~auto_rewarded`, per the Decoder Task spec."
"**Alignment event** = `trials.change_time` … This is the reference event used by the Allen SWDB analysis code
(`save_trial_response_df.py`)." The window is justified empirically: "over all 202 active experiments
min(change_time - start_time) = 2.79 s and min(stop_time - change_time) = 4.20 s, so this window is **always inside the
trial** (no bleed into neighbouring trials, no truncation, identical length for every trial) and always inside
`ophys_timestamps`" (verified by `/app/cache/check_windows.py`). The 250 ms bin is chosen because "(a) it equals the image
presentation duration and divides the 750 ms flash cycle exactly into 3, so image/change labels never straddle bins;
(b) it is larger than the slowest ophys frame interval (93 ms on the Mesoscope), so every bin of every session contains
>= 2 imaging frames, allowing one common bin size for all sessions as the format requires".

## 1-e. How are trials filtered based on quality controls?

i. Four filters: (1) aborted and auto-rewarded trials removed, NaN `change_time` removed (see 1-d); (2) a trial is dropped
if **any** of its 24 bins has no valid running-speed or pupil sample (i.e. a blink gap > 0.5 s covering a whole bin) —
2,071 trials (4.6 %); (3) a whole session is dropped if it has no eye-tracking table at all (3 sessions, 917 trials) or if
fewer than 2 trials survive; (4) an entire session is dropped with a recorded reason if any trial lacks an outcome label.
Trial accounting is reported and reconciles exactly: 41,904 + 2,071 + 917 = 44,892 go+catch trials in the SDK tables.

ii.
```python
        # ---------------- trial curation: require complete behaviour
        keep = (~np.isnan(pupil).any(axis=1)) & (~np.isnan(running).any(axis=1))
        n_dropped = int((~keep).sum())
        if keep.sum() < 2:
            out['error'] = 'fewer than 2 trials with complete behaviour'
            return out
```
```python
        eye = ds0.eye_tracking
        if len(eye) == 0:
            out['error'] = 'no eye tracking data'
            return out
```
```python
        if np.any(outcome < 0):
            out['error'] = 'trial without an outcome label'
            return out
```
```python
    good = [r for r in results if not r['error']]
    bad = [r for r in results if r['error']]
    for r in bad:
        print(f"  EXCLUDED session {r['session_id']}: {r['error'].splitlines()[0]}")
```

iii. Step 5 Decision 5: aborted/auto-rewarded exclusion is "per the Decoder Task spec; this also matches the SDK's own
convention that hit/miss/false-alarm/correct-reject are only defined for go/catch trials". Decision 7 on missing pupil:
"3 of 174 active sessions have *no* eye-tracking table at all -> those sessions are dropped (a required output cannot be
fabricated). Within the remaining sessions, blink gaps <= 0.5 s are linearly interpolated (blinks are short); trials that
still contain a bin with no valid pupil sample (~6% of trials) are dropped. **Fabricating pupil values for a *decoded*
variable would corrupt the evaluation.**" Excluded sessions are also written to `metadata['sessions_excluded']`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The dF/F traces computed by the Allen pipeline: `experiment.dff_traces['dff']`, together with
`experiment.ophys_timestamps` for the time base. The script keeps `events` (L0-detected calcium events) and
`filtered_events` as selectable alternatives via `--neural-signal`, but the **default and the signal used for
`converted_data.pkl` is `dff`**.

ii.
```python
    ap.add_argument('--neural-signal', type=str, default='dff',
                    choices=['events', 'dff', 'filtered_events'],
```
```python
            ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
            if signal == 'dff':
                ev = np.vstack(ds.dff_traces['dff'].values).astype(np.float64)
            elif signal == 'filtered_events':
                ev = np.vstack(ds.events['filtered_events'].values).astype(np.float64)
            else:
                ev = np.vstack(ds.events['events'].values).astype(np.float64)
            assert ev.shape[1] == ts.shape[0], 'trace/timestamps mismatch'
```

iii. Step 1: "dF/F does NOT need to be computed: the SDK returns pipeline-computed `dff_traces` for each valid ROI."
Step 5 originally chose `events` to follow the paper ("For all analysis of neural data we used the detected calcium
events"), but Step 8 ran a controlled comparison and reversed the choice: "**Decision**: use **dF/F** (`dff_traces`,
averaged within each 250 ms bin) … (i) it is the signal used by the Allen reference trial-analysis code
(`swdb/save_trial_response_df.py` builds its trial response dataframe from `dff_trace`) and by the SDK tutorials; (ii) the
paper's preference for detected events was motivated by removing GCaMP decay for *event-triggered averaging*, whereas for
a decoder the extra-sparse event trains (98 % empty 250 ms bins) discard amplitude information, and empirically decode
every variable *worse*; (iii) dF/F keeps the slow components that carry running/pupil state information." The measured
validation accuracies for all three signals are tabulated in Step 8.

## 2-b. How is the `neural` data processed?

i. Two operations only: (1) **temporal re-binning** — the dF/F of each neuron is *averaged* over the ophys frames whose
timestamps fall in each 250 ms bin (implemented as a cumulative-sum / `searchsorted` bin sum divided by the number of
frames in the bin); (2) **plane concatenation** — the per-plane (n_cells, n_trials, 24) arrays of a session are stacked
along the neuron axis, and a `targeted_structure` label (VISp/VISl) is recorded per neuron. Output is cast to float32.
No normalisation, smoothing, baseline subtraction or z-scoring is applied.

ii.
```python
def bin_sum(values, timestamps, edges_abs):
    csum = np.concatenate([np.zeros((values.shape[0], 1), dtype=np.float64),
                           np.cumsum(values, axis=1)], axis=1)
    idx = np.searchsorted(timestamps, edges_abs.ravel())          # (n_trials*(N+1),)
    idx = idx.reshape(edges_abs.shape)
    take = csum[:, idx]                                           # (n_signals, n_trials, N+1)
    return (take[:, :, 1:] - take[:, :, :-1]).astype(np.float32)
```
```python
            n_per_bin = bin_sum(np.ones((1, ts.shape[0])), ts, edges_abs)   # (1, n_trials, N)
            binned = bin_sum(ev, ts, edges_abs)        # (n_cells, n_trials, N)
            if signal == 'dff':
                # average dF/F within the bin (sum would scale with frame rate)
                binned = binned / np.maximum(n_per_bin, 1)
            neural_planes.append(binned)
            region_per_neuron += [region_map[int(eid)]] * ev.shape[0]
        neural = np.concatenate(neural_planes, axis=0)  # (n_neurons, n_trials, N)
```

iii. Averaging rather than summing is justified in the code comment itself ("sum would scale with frame rate") — essential
because the dataset mixes 31 Hz single-plane and 11 Hz Mesoscope sessions. Step 5 Decision 2: "dF/F is not recomputed -
the Allen pipeline already provides `dff_traces`, and the event detection was run on those traces." Decision 3: Mesoscope
planes are merged because they are "simultaneously recorded populations". Step 6 documents the vectorised cumsum binning
as a ~10× speed-up over per-trial slicing, and float32 storage as a memory optimisation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No filtering beyond what the SDK already performs. The AI verified that the SDK returns only ROIs with
`valid_roi == True` (`exclude_invalid_rois=True` is the default and the released NWBs contain only valid ROIs), and added
no activity-, SNR- or drift-based neuron rejection. All 29,168 neurons of the 171 kept sessions are exported.

ii. No filtering code exists; the closest thing is the shape assertion that ties traces to timestamps:
```python
            assert ev.shape[1] == ts.shape[0], 'trace/timestamps mismatch'
```

iii. Step 1 Notes: "**ROI curation is automatic**: `exclude_invalid_rois=True` is the default, so only ROIs that passed the
Allen segmentation/classification QC (`valid_roi`) are returned." Step 3 Curation: "ROI filtering already applied by the
Allen pipeline (multi-label classifier; unions/duplicates/edge/dendrite/small/dim ROIs marked `valid_roi=False`) … **No
further neuron filtering is described in the paper.**" Step 4 records the empirical check: "`cell_specimen_table.valid_roi`
is True for 100 % of returned ROIs". Session-level QC (z-drift, photobleaching, d′ ≥ 1) was applied by Allen before
release.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial is aligned to **`trials.change_time`** (the real change on go trials, the sham change on catch trials).
Absolute bin edges are formed as `change_time + BIN_EDGES`, and frames are assigned to bins by `np.searchsorted` on that
plane's own `ophys_timestamps`, so each imaging plane is binned on its own clock (important for Mesoscope, where planes
are offset in time) while all planes share the same absolute bin boundaries. The alignment event, `off_start = -2.25` and
`off_end = +3.75` are all recorded in metadata.

ii.
```python
        change_times = trials['change_time'].values.astype(np.float64)
        edges_abs = change_times[:, None] + BIN_EDGES[None, :]      # (n_trials, N+1)
```
```python
    idx = np.searchsorted(timestamps, edges_abs.ravel())
```
```python
        temporal_alignment_event=('stimulus change time of the trial (trials.change_time; the '
                                  'sham change time on catch trials), binned on the ophys '
                                  'timestamp clock'),
        off_start=OFF_START,
        off_end=OFF_END,
```

iii. Step 1/Step 5: "Reference trial alignment in Allen analysis code is to the stimulus **change time** of each trial with
a window of a few seconds before/after", citing `swdb/save_trial_response_df.py` which "aligns traces to
`trials['change_time']` with `get_trace_around_timepoint`" and `window_around_timepoint_seconds = [-4, 8]`. Step 10
Check 5 records the off-by-one verification: "change flash onset = left edge of bin 9; `image_change` occupies bins 9-11
only — verified for every trial of the sample and by sanity checks", and the independent sanity checks confirm
"`change_time` must equal the start time of an `is_change` flash (bin 9 left edge) — true for every go trial tested".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — explicit rebinning. The native ophys sampling (30.9 Hz single-plane ≈ 32 ms, 10.7 Hz Mesoscope ≈ 93 ms) is
re-binned onto a **uniform 250 ms grid**, 24 bins per trial, identical for every trial, session and rig.
`metadata['time_bin_size'] = 250.0` ms. Running speed (60 Hz) and pupil (30 Hz) are averaged onto the same grid, so all
streams share one time base.

ii.
```python
BIN_SIZE = 0.25         # s
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 24
```
```python
        time_bin_size=BIN_SIZE * 1000.0,
        ...
        n_time_bins=N_BINS,
```

iii. Step 5: 250 ms was chosen because "(a) it equals the image presentation duration and divides the 750 ms flash cycle
exactly into 3, so image/change labels never straddle bins; (b) it is larger than the slowest ophys frame interval (93 ms
on the Mesoscope), so every bin of every session contains >= 2 imaging frames, allowing one common bin size for all
sessions as the format requires". This was verified: "`check_bin_occupancy.py`: **min 2 frames/bin** (median 7) across all
171 sessions, 0 empty bins". Step 10 Check 3 adds: "a *common* bin size across 31 Hz and 11 Hz rigs is required by the
target format". The paper's own behavioural analyses assign events to the 750 ms presentation interval, which the 250 ms
grid subdivides exactly.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `experiment.stimulus_presentations`, restricted to the change-detection block
(`stimulus_block_name` containing `change_detection`) and to non-omitted flashes (`image_name != 'omitted'`), using the
columns `start_time` and `image_name`. It is *not* taken from the trials table's `initial_image_name` /
`change_image_name`.

ii.
```python
        sp = ds0.stimulus_presentations
        sp = sp[sp['stimulus_block_name'].astype(str).str.contains('change_detection')]
        shown = sp[sp['image_name'] != 'omitted']
        flash_start = shown['start_time'].values.astype(np.float64)
        flash_image = shown['image_name'].values.astype(str)
```

iii. Step 5 variable-mapping table: image identity comes from "`stimulus_presentations.image_name` (change_detection
block, non-omitted)"; the tutorials are cited for the
`stimulus_block_name.str.contains('change_detection')` idiom, which removes the gray-screen, natural-movie and
post-behaviour blocks. Dropping `omitted` rows implements the rule that "during a 5 % omission the identity of the ongoing
(repeating) image is unchanged, so it is held".

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each of the 24 bin centres, the most recent non-omitted flash onset at or before that centre is found with
`searchsorted(..., side='right') - 1`, and its `image_name` is used as the label for that bin — i.e. the image identity is
*held* through the 500 ms gray ISI and through omissions. Names are then mapped to integer codes through a **global**
sorted mapping of all 16 image names in the dataset (both image sets A and B), so codes are comparable across sessions.
A session is rejected if any bin precedes the first flash.

ii.
```python
        # index of the most recent shown-image onset at or before each bin centre
        j = np.searchsorted(flash_start, centers_abs, side='right') - 1
        if np.any(j < 0):
            out['error'] = 'bin before first flash'
            return out
        image_name = flash_image[j]                                   # (n_trials, N)
```
```python
    image_values = sorted({str(im) for r in good for im in np.unique(r['image_name'])})
    image_to_idx = {im: i for i, im in enumerate(image_values)}
```
```python
        img = np.vectorize(lambda x: image_to_idx[str(x)])(r['image_name']).astype(np.int64)
```

iii. Step 5: "per bin: image of the most recent non-omitted flash onset <= bin centre; 16 global classes (im000…im106) …
Holding the identity across the 500 ms gray ISI implements the paper's 'image presentation interval' (750 ms starting at
each flash)". Decision 9: "**Image identity uses the 16 global image names**, not the per-session 0-7 `image_index`,
because index *i* means different images in image sets A and B." The resulting marginal distribution (~6 % per image over
16 classes, 8 images per session) is checked against the whitepaper's "Each session included 8 images".

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated at the **centre of each of the same 24 bins** used for the neural data (`centers_abs =
change_time + BIN_CENTERS`), so the image label of bin *k* describes the same 250 ms interval as neural column *k*. No
separate alignment step is needed. Because the change flash starts exactly at the left edge of bin 9, the identity step
occurs between bins 8 and 9 on go trials.

ii.
```python
        centers_abs = change_times[:, None] + BIN_CENTERS[None, :]  # (n_trials, N)
        ...
        j = np.searchsorted(flash_start, centers_abs, side='right') - 1
        image_name = flash_image[j]
```

iii. Step 7 checks: "`image_identity` differs between bin 8 and bin 9 for every go trial and is identical across the sham
change for every catch trial." Step 10 independent sanity check: "Image identity — most recent non-omitted flash onset at
each bin centre, from `stimulus_presentations` — 24/24 exact match". The `--show-processing` figure overlays the flash
raster on the identity step function to show there is no temporal offset.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `stimulus_presentations.is_change` (of the same non-omitted change-detection flashes), together with the flash
`start_time`. Because `is_change` is False for the *sham* change of catch trials, catch trials automatically get an
all-zero image-change trace; the trials table's `go`/`catch` columns are not needed for this output.

ii.
```python
        flash_ischange = shown['is_change'].values.astype(bool)
```

iii. Step 5: "`stimulus_presentations.is_change` -> `output[1]` image_change … Catch trials have a *sham* change
(`is_change` False) so they stay 0 - exactly the change-vs-repeat discrimination of the paper's change decoder."

## 4-b. What processing is involved in computing `output` *Image change*?

i. A bin is labelled 1 if the most recent flash at its centre is a change flash **and** the centre lies within the 750 ms
presentation interval that starts at that flash. This makes bins 9, 10 and 11 equal to 1 on go trials (0–750 ms after the
change) and everything else 0. It is stored as a binary int.

ii.
```python
FLASH_CYCLE = 0.75      # s, image presentation interval (paper)
...
        since = centers_abs - flash_start[j]
        # image change: bins inside the 750 ms presentation interval of a change flash
        image_change = (flash_ischange[j] & (since < FLASH_CYCLE)).astype(np.int64)
```

iii. Step 5: "1 in the bins of the presentation interval of a change flash (3 bins = 750 ms starting at the change),
else 0". Step 3 motivates the 750 ms quantum: "behavioural events assigned to the 750 ms *image presentation interval*
beginning at each flash (also used for omissions). This motivates using the 750 ms flash cycle as the natural quantum of
time." The instruction's "value of 1 right after a change in image identity" is thus implemented as the one image
presentation interval that follows the change.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required — it is intrinsically binary, with `output_values = ['no_change', 'change']`. The only
implicit "threshold" is the 750 ms cut-off (`since < FLASH_CYCLE`) that decides how many bins after the change are
labelled 1, giving 3 of 24 bins on go trials. The resulting global class balance is 0.891 / 0.109.

ii.
```python
    output_values = [
        list(image_values),
        ['no_change', 'change'],
        ...
```
```python
        image_change = (flash_ischange[j] & (since < FLASH_CYCLE)).astype(np.int64)
```

iii. Step 7 arithmetic check: "Expected change fraction: 3 bins / 24 bins x (go fraction 0.87) = 0.109, observed 0.108";
full data 0.109. Step 7 also verifies "`image_change` is 1 in **exactly** bins 9-11 (= [0, 0.75) s after the change) for
all 213 go trials and 0 for all 33 catch trials".

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: it is evaluated on the same 24 bin centres relative to `change_time`, so it shares the
neural time base exactly. The change flash onset coincides with the left edge of bin 9 by construction.

ii.
```python
        since = centers_abs - flash_start[j]
        image_change = (flash_ischange[j] & (since < FLASH_CYCLE)).astype(np.int64)
```

iii. Step 10 sanity check: "Image change — `is_change` flash and bin centre within 750 ms of it — 24/24 exact match";
"Alignment — `change_time` must equal the start time of an `is_change` flash (bin 9 left edge) — true for every go trial
tested". The processing figure shades the 750 ms post-change interval and overlays the binary trace.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `experiment.running_speed`, columns `timestamps` and `speed` (cm/s, ~60 Hz), used as delivered by the SDK — i.e. the
stream that already has z-score>10 transient removal and the low-pass Butterworth filter applied by
`running_processing.get_running_df`.

ii.
```python
        run = ds0.running_speed
        run_t = run['timestamps'].values.astype(np.float64)
        run_v = run['speed'].values.astype(np.float64)
```

iii. Step 4: "Use SDK `running_speed` as-is (already the processed, filtered stream)", matching the whitepaper's
description ("Any additional transients with z-score of 10 or greater were removed … smoothing the raw running speed with
a 10 Hz lowpass Butterworth filter") and the units check ("running_speed_cm_per_sec = angular_speed * (2/3 *
wheel_radius)"). The data scan found "`running_speed` @59.95 Hz, no NaNs in any experiment".

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The 60 Hz speed trace is **averaged within each 250 ms bin** (NaN-aware cumulative-sum mean; a bin with no sample
becomes NaN, which causes the trial to be dropped). The binned means are then discretised into 5 equal-percentile
("quintile") classes using bin edges computed **globally over every bin of every kept session** (20/40/60/80th
percentiles), via `np.digitize`.

ii.
```python
def bin_mean_nan(values, timestamps, edges_abs):
    '''Mean of `values` (1d, may contain NaN) within each bin; NaN if no valid sample.'''
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(filled)])
    ccnt = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])
    idx = np.searchsorted(timestamps, edges_abs.ravel()).reshape(edges_abs.shape)
    s = csum[idx][:, 1:] - csum[idx][:, :-1]
    n = ccnt[idx][:, 1:] - ccnt[idx][:, :-1]
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out
```
```python
        running = bin_mean_nan(run_v, run_t, edges_abs)               # (n_trials, N)
```
```python
def quantile_bins(values, n_bins=N_QUANTILES):
    '''Global equal-percentile edges (interior) for a 1-d array of values.'''
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    return np.percentile(values, qs)
```

iii. Step 5 mapping: "mean speed per 250 ms bin -> discretised into 5 **global** equal-percentile bins (20/40/60/80th pct
of all bins in the dataset)"; Step 6 notes `bin_mean_nan` "propagates NaN only when a bin contains *no* valid sample".
Averaging (rather than point-sampling) is the natural downsampling from 60 Hz to 4 Hz.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five classes by global equal-percentile edges over all trials/sessions (measured edges 0.019, 2.517, 19.262,
33.851 cm/s), applied with `np.digitize`, giving exactly 20.0 % of bins in each class. The alternative (per-session
percentiles) is implemented behind `--quantile-scope session` but not used.

ii.
```python
    all_run = np.concatenate([r['running'].ravel() for r in good])
    run_edges = quantile_bins(all_run)
    ...
    if args.quantile_scope == 'session':
        for r in good:
            r['run_edges'] = quantile_bins(r['running'].ravel())
    else:
        for r in good:
            r['run_edges'] = run_edges
```
```python
def digitize_with(edges, values):
    return np.digitize(values, edges).astype(np.int64)
...
        run_q = digitize_with(r['run_edges'], r['running'])
```

iii. Step 5 Decision 6: "**Global (dataset-wide) quintile edges** for running speed and pupil diameter, so the 5 classes
are equally populated over the whole dataset as 'five equal percentile bins' requires, and so that one shared decoder head
sees a consistent labelling." Step 9 adds an empirical comparison on a 24-session subset showing global edges also decode
better (running 0.406 vs 0.329, pupil 0.404 vs 0.285). Edges are stored in
`metadata['running_speed_quintile_edges_cm_per_s']`.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is binned on exactly the same absolute bin edges (`edges_abs = change_time + BIN_EDGES`) as the neural data, using
the running stream's own sync-clock timestamps, so bin *k* of the running output covers the same 250 ms of wall-clock time
as bin *k* of the neural matrix.

ii.
```python
        running = bin_mean_nan(run_v, run_t, edges_abs)               # (n_trials, N)
```

iii. Step 1 Notes: "ophys frames, stimulus frames (60 Hz; running speed lives here), eye-tracking frames (~30 Hz), and
event times … are all on a common sync clock in seconds, so they can be aligned directly by interpolating/binning onto the
ophys timestamps." Step 10 sanity check: "Running quintile — independent per-bin mean of `running_speed.speed` +
`np.digitize` with the stored edges — 24/24 exact match".

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `experiment.eye_tracking`, columns `timestamps` and `pupil_area`. Diameter is derived as the equivalent-circle
diameter `2*sqrt(area/pi)`. Blink frames are already NaN in the SDK table (`EyeTrackingTable.from_nwb` recomputes
`likely_blink` with z=3, dilation 2 and calls `filter_on_blinks`), so no explicit `likely_blink` masking is needed.
Sessions with an empty eye-tracking table are excluded.

ii.
```python
        eye = ds0.eye_tracking
        if len(eye) == 0:
            out['error'] = 'no eye tracking data'
            return out
        eye_t = eye['timestamps'].values.astype(np.float64)
        pupil_area = eye['pupil_area'].values.astype(np.float64)
        pupil_diam = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. Step 5 mapping: "`eye_tracking.pupil_area` (30 Hz) -> diameter = 2*sqrt(area/pi) … **Diameter (not area) as
required**", with the note that `EyeTrackingTable.from_nwb` already NaNs blinks. Step 4 records the data check: "~3.5 %
NaN frames/experiment; **3 sessions have no eye-tracking table at all**". (In the SDK, `pupil_area = pi * max(width,
height)^2`, so this formula returns exactly the fitted pupil diameter in pixels.)

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) NaN runs (blinks) shorter than 0.5 s are linearly interpolated; longer runs, and runs touching the edges of the
recording, are left NaN. (2) The trace is averaged within each 250 ms bin with the same NaN-aware binning used for running
speed. (3) Any trial that still contains a NaN bin is dropped (see 1-e). (4) The binned diameters are discretised into 5
global equal-percentile classes.

ii.
```python
def interpolate_short_gaps(t, v, max_gap):
    '''Linearly interpolate NaN runs shorter than max_gap seconds; leave longer runs NaN.'''
    ...
    filled = np.interp(idx, idx[~isn], v[~isn])
    d = np.diff(np.concatenate(([0], isn.astype(np.int8), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    out = filled
    for s0, e0 in zip(starts, ends):
        t0 = t[s0 - 1] if s0 > 0 else t[0]
        t1 = t[e0] if e0 < len(t) else t[-1]
        if (t1 - t0) > max_gap or s0 == 0 or e0 >= len(t):
            out[s0:e0] = np.nan
    return out
```
```python
PUPIL_MAX_INTERP_GAP = 0.5   # s, blink gaps shorter than this are interpolated
...
        pupil_diam = interpolate_short_gaps(eye_t, pupil_diam, PUPIL_MAX_INTERP_GAP)
        pupil = bin_mean_nan(pupil_diam, eye_t, edges_abs)            # (n_trials, N)
```

iii. Step 5 Decision 7: "blink gaps <= 0.5 s are linearly interpolated (blinks are short); trials that still contain a bin
with no valid pupil sample (~6 % of trials) are dropped … Fabricating pupil values for a *decoded* variable would corrupt
the evaluation." The 0.5 s cap is the stated boundary between a blink (interpolatable) and genuine signal loss.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same scheme as running speed: five classes from global equal-percentile edges over all bins of all kept sessions
(measured edges 74.711, 84.174, 93.148, 105.825 px), applied with `np.digitize`; each class holds 20.0 % of bins.

ii.
```python
    all_pup = np.concatenate([r['pupil'].ravel() for r in good])
    pup_edges = quantile_bins(all_pup)
    ...
        pup_q = digitize_with(r['pup_edges'], r['pupil'])
```
```python
        pupil_diameter_quintile_edges_pixels=pup_edges.tolist(),
```

iii. Same as 5-c (Step 5 Decision 6 and the Step 9 quantile-scope experiment). The check "running/pupil quintiles each
~20 % of bins" was listed as a planned sanity check and reported as satisfied (0.200 × 5).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Binned on the same absolute edges `change_time + BIN_EDGES` as the neural data, on the eye camera's own sync-clock
timestamps, so pupil bin *k* covers the same 250 ms as neural bin *k*.

ii.
```python
        pupil = bin_mean_nan(pupil_diam, eye_t, edges_abs)            # (n_trials, N)
```

iii. Same justification as running speed (common sync clock, Step 1 Notes). Step 10 sanity check: "Pupil quintile —
independent blink-gap interpolation + per-bin mean of 2*sqrt(pupil_area/pi) — 24/24 exact match".

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the SDK trials table — `hit`, `miss`, `false_alarm`, `correct_reject` —
evaluated on the already-filtered go/catch trials.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
        outcome = np.full(n_trials, -1, dtype=np.int64)
        for k, name in enumerate(OUTCOMES):
            outcome[trials[name].values.astype(bool)] = k
        if np.any(outcome < 0):
            out['error'] = 'trial without an outcome label'
            return out
```

iii. Step 4: "`trial.py`: hit/miss/false_alarm/correct_reject defined only for go/catch — every go+catch trial has exactly
one of the four — Use the 4 outcomes as the static per-trial output." Step 1 documents `Trial._get_trial_data()` as the
definition source (auto-rewarded trials force all four to False, which is why they must be excluded first).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is turned into an integer code 0–3 in the fixed order `['hit','miss','false_alarm','correct_reject']` and,
because the target format wants time-varying rows where possible, the per-trial code is **broadcast to all 24 bins** of
that trial. A trial with no outcome label would abort the whole session (never triggered). The observed distribution is
hit 0.323 / miss 0.552 / false_alarm 0.019 / correct_reject 0.106.

ii.
```python
            out_trials.append(np.stack([
                img[t], r['image_change'][t], run_q[t], pup_q[t],
                np.full(N_BINS, r['outcome'][t], dtype=np.int64)]).astype(np.int64))
```
```python
    output_values = [
        ...
        list(OUTCOMES),
    ]
```

iii. Step 5 mapping: "static per trial, broadcast to all 24 bins; 4 classes". Step 12 explains the consequence honestly:
"the label is *static* per trial, so the decoder is asked to predict it from pre-change bins too, where the outcome is not
yet determined", and "the format requires a per-trial static output for this variable". The implied hit rate
(0.323/(0.323+0.552) = 0.369) and FA rate (0.019/(0.019+0.106) = 0.152) are cross-checked against the SDK tables
(0.350 / 0.143 before the pupil-completeness filter).

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is explicit and always *documented rather than silently patched*:
- **No eye-tracking table** (3 sessions) → session excluded, reason recorded in `metadata['sessions_excluded']`.
- **Blinks / NaN pupil** → gaps ≤ 0.5 s interpolated; longer gaps left NaN and the affected trial dropped (2,071 trials,
  4.6 %); trials with a NaN running bin likewise dropped.
- **Empty bins** (no sample in a 250 ms bin) → NaN by construction in `bin_mean_nan`, then dropped; for neural data the
  window/bin size were chosen so this cannot happen (verified min 2 frames/bin).
- **Missing `change_time`** → trial dropped.
- **Trial with no outcome label**, **bin before the first flash**, **trace/timestamp length mismatch** → the session is
  aborted with a recorded error rather than producing wrong data.
- **Any unexpected exception** → caught per session, the traceback tail is stored, the session is excluded and listed in
  the run log; the remaining sessions still convert.
- **Sessions with < 2 usable trials** → excluded (format requirement).
Trial accounting is printed so nothing is lost silently: 41,904 kept + 2,071 dropped + 917 in excluded sessions = 44,892
go+catch trials.

ii.
```python
    except Exception as exc:                                   # pragma: no cover
        import traceback
        out['error'] = f'{exc!r}\n{traceback.format_exc()[-800:]}'
    return out
```
```python
        keep = (~np.isnan(pupil).any(axis=1)) & (~np.isnan(running).any(axis=1))
        n_dropped = int((~keep).sum())
        if keep.sum() < 2:
            out['error'] = 'fewer than 2 trials with complete behaviour'
            return out
```
```python
        sessions_excluded=[dict(session_id=r['session_id'], reason=r['error'].splitlines()[0])
                           for r in bad],
```
```python
                                 n_trials_dropped_missing_behavior=int(r['n_trials_dropped'])))
```

iii. Step 5 Decision 7 and Step 10 Check 5 ("Edge cases" table) give the rationale: a decoded (output) variable must never
be fabricated, because an invented pupil value becomes an invented *label*; whole-session failures must not kill the run;
and every drop must be reconcilable against the SDK's own trial counts, which the Step 9 "Trial accounting (no data
silently lost)" table demonstrates ("Sum check: 41,904 + 2,071 + 917 = 44,892 exactly").

## 9-a. What are the most time-consuming steps of the code?

i. Loading the NWB files through the SDK (`get_behavior_ophys_experiment`) dominates: the script times it separately
(`t_load`) and reports ~10.3 s per session serially (~3 s per imaging plane), versus a negligible cost for binning.
Secondary costs are pickling the 774 MB output and the global percentile computation over ~1 M bins. With 16 worker
processes the whole 174-session conversion takes 90 s wall time (0.50 s/session).

ii.
```python
    session_id, eids, region_map, want_debug, signal = args
    t_start = time.time()
    cache = get_cache()
    ...
        datasets = [cache.get_behavior_ophys_experiment(int(e)) for e in eids]
        t_load = time.time() - t_start
    ...
            t_load=t_load,
            t_total=time.time() - t_start,
```
```python
    t_proc = time.time() - t0
    print(f'session loading/binning took {t_proc:.1f} s '
          f'({t_proc / max(len(jobs), 1):.2f} s/session)')
```

iii. Step 6/Step 7: "Re-opening the cache per experiment is expensive"; "load + bin one session (1-7 planes) | 10.3 s wall
per session (serial), ~3 s per plane | 174 sessions, 202 planes, 16 workers -> ~2-4 min". The realised time (90 s) beat
the estimate, so no further optimisation was pursued.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The expensive loops were already removed: binning of *all* trials × *all* neurons is done in one cumulative-sum +
`searchsorted` call, and image identity/change are computed for the whole (n_trials, 24) grid at once. The loops that
remain are cheap and were left alone: (a) the per-plane loop (1–7 iterations, each dominated by I/O); (b) the per-trial
assembly loop that slices `r['neural'][:, t, :]` and stacks the five output rows — this could be replaced by
`np.moveaxis`/list comprehension but only touches already-loaded arrays; (c) the per-neuron `regions.index(reg)` lookup
and `np.vectorize` image-code mapping, which are O(n_neurons) / O(n_trials·24) Python-level and are the only genuinely
vectorisable leftovers; (d) the gap loop in `interpolate_short_gaps`, which iterates over blink runs only.

ii.
```python
        for t in range(n_trials):
            neural_trials.append(np.ascontiguousarray(r['neural'][:, t, :]))
            out_trials.append(np.stack([
                img[t], r['image_change'][t], run_q[t], pup_q[t],
                np.full(N_BINS, r['outcome'][t], dtype=np.int64)]).astype(np.int64))
            in_trials.append(np.zeros((0, N_BINS), dtype=np.float32))
```
```python
        ridx = []
        for reg in [str(x) for x in r['regions']]:
            if reg not in regions:
                regions.append(reg)
            ridx.append(regions.index(reg))
```
```python
        img = np.vectorize(lambda x: image_to_idx[str(x)])(r['image_name']).astype(np.int64)
```

iii. Step 6: "A naive implementation would slice traces per trial in a Python loop (n_trials x n_cells slices)" →
"Cumulative-sum + `searchsorted` binning: all trials of a session are binned in one vectorised call" (~10× saving), plus
"16-way multiprocessing over sessions" (~16×). The remaining per-trial loop exists because the target format itself is a
*list of per-trial arrays*, so the loop is required to materialise the output structure.

## 9-c. What processing does the code repeat multiple times?

i. A few repetitions remain, all small relative to NWB I/O:
- `get_cache()` is called once **per session job** (174 times), so the project manifest/experiment table is parsed once per
  session instead of once per worker process.
- `n_per_bin` (frames per bin) is recomputed for every imaging plane; this is necessary for per-plane clocks, but it is
  computed even for the `events`/`filtered_events` modes where it is never used.
- Global quantile edges are computed even when `--quantile-scope session` is selected, in which case per-session edges are
  computed on top of them.
- Summary statistics re-concatenate all outputs and re-scan all neural arrays after the pickle has been assembled.
- In `--show-processing` mode the debug payload re-reads the `events` traces for a session whose neural data are dF/F.

ii.
```python
def process_session(args):
    session_id, eids, region_map, want_debug, signal = args
    t_start = time.time()
    cache = get_cache()
```
```python
            n_per_bin = bin_sum(np.ones((1, ts.shape[0])), ts, edges_abs)   # (1, n_trials, N)
            binned = bin_sum(ev, ts, edges_abs)        # (n_cells, n_trials, N)
            if signal == 'dff':
                binned = binned / np.maximum(n_per_bin, 1)
```
```python
    run_edges = quantile_bins(all_run)
    pup_edges = quantile_bins(all_pup)
    ...
    if args.quantile_scope == 'session':
        for r in good:
            r['run_edges'] = quantile_bins(r['running'].ravel())
```

iii. Step 6 lists the repetition that *was* eliminated ("One cache handle per worker process, sessions processed in
parallel"), and each session is loaded exactly once — the per-trial data extracted in the worker are reused for the global
percentile edges and for final assembly without any reloading. The residual repetitions are not discussed in the notes;
measured end-to-end time (90 s) made further tuning unnecessary.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts:
- `cell_ids` (cell_specimen_ids, read from `ds.events.index`) are collected per session and never written to the output
  dictionary — note also that these come from the `events` table while the neural data come from `dff_traces`.
- `change_times`, `go`, `catch`, `n_trials_total`, `t_load`/`t_total` are carried in the per-session result but are used
  only for plotting/printing, not for the converted data.
- `n_per_bin` is computed in the `events`/`filtered_events` modes where it is discarded.
- The `--show-processing` debug dict loads and returns full-session `events`, lick and reward arrays that are irrelevant
  to the dF/F pipeline.
- Whole passive experiments are listed by `available_experiment_table` before being filtered out (cheap, table-level).
- More fundamentally, every session's full-length dF/F traces (~140 k frames × n_neurons) are read in order to keep only
  ~250 trials × 24 bins; this is unavoidable given the NWB layout.

ii.
```python
            cell_ids += list(ds.events.index.values)
...
            cell_ids=np.array(cell_ids),
            change_times=change_times[keep],
            go=trials['go'].values.astype(bool)[keep],
            catch=trials['catch'].values.astype(bool)[keep],
            n_trials_dropped=n_dropped,
            n_trials_total=n_trials,
            t_load=t_load,
```
```python
        if want_debug:
            out['debug'] = dict(
                ophys_timestamps=np.asarray(datasets[0].ophys_timestamps, dtype=np.float64),
                events0=np.vstack(datasets[0].events['events'].values).astype(np.float32),
                ...
                lick_times=np.asarray(ds0.licks['timestamps'].values, dtype=np.float64),
                reward_times=np.asarray(ds0.rewards['timestamps'].values, dtype=np.float64),
            )
```

iii. The notes do not flag these as waste; the debug/licks/rewards extraction is deliberate and gated behind
`--show-processing` ("Plots for `--show-processing` mode should visually convince the user that every step of the
conversion is correct" — the figure overlays licks and rewards to verify outcome labels). The per-session bookkeeping
fields (`n_trials_dropped`, `n_trials_total`) feed the Step 9 trial-accounting table and `metadata['session_info']`, so
they are intentional even though the decoder ignores them.
