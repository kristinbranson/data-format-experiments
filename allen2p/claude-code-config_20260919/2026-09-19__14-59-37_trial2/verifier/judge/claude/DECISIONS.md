# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI works entirely from the local copy of the release (`/app/data/visual-behavior-ophys-1.1.0`) rather than from the SDK's S3 cache object. `select_sessions()` reads the project metadata table `project_metadata/ophys_experiment_table.csv` (1936 rows = full release), intersects it with the set of NWB files actually present on disk (284 of 1936), drops `passive` experiments, and groups the remainder by `ophys_session_id`. Each group (= one ophys session = 1 plane on the Scientifica rigs, up to 7 simultaneously recorded planes on the Multiscope) becomes one job. Each job loads every plane with the canonical AllenSDK loader `BehaviorOphysExperiment.from_nwb_path`, and all streams (trials, stimulus_presentations, running_speed, eye_tracking, dff_traces/events, ophys_timestamps, metadata) are taken from that single object. Jobs are fanned out over a `ProcessPoolExecutor` (16 workers). No project-code filter is applied, so both `VisualBehavior` (239 experiments, single plane, 31 Hz) and `VisualBehaviorMultiscope` (45 experiments, one mouse, 11 Hz) are included. Result: 174 candidate active sessions → 171 converted, 38 mice, 199 planes, 29 168 neurons, 43 975 trials.

ii.
```python
DATA_ROOT = '/app/data/visual-behavior-ophys-1.1.0'
EXP_DIR = os.path.join(DATA_ROOT, 'behavior_ophys_experiments')
META_DIR = os.path.join(DATA_ROOT, 'project_metadata')

def load_experiment(eid: int):
    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
        BehaviorOphysExperiment)
    return BehaviorOphysExperiment.from_nwb_path(
        os.path.join(EXP_DIR, 'behavior_ophys_experiment_%d.nwb' % eid))

def select_sessions():
    """Return a DataFrame of the downloaded, active-behavior experiments grouped by session."""
    exp = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    have = set()
    for f in os.listdir(EXP_DIR):
        if f.endswith('.nwb'):
            have.add(int(f.split('_')[-1].split('.')[0]))
    exp = exp[exp.ophys_experiment_id.isin(have)]
    exp = exp[~exp.passive]                      # active behavior sessions only
    exp = exp.sort_values(['ophys_session_id', 'ophys_experiment_id'])
    return exp
```
```python
        experiments = [load_experiment(e) for e in eids]
        ...
        ds0 = experiments[0]
        trials = ds0.trials
        stim = ds0.stimulus_presentations
        running = ds0.running_speed
        eye = ds0.eye_tracking
```
```python
    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        for i, r in enumerate(ex.map(convert_session, jobs)):
```

iii. From CONVERSION_NOTES Step 1 and Step 10 Check 3: the AI verified in the SDK source that `bc.get_behavior_ophys_experiment(eid)` is literally `BehaviorOphysExperiment.from_nwb_path(nwb)` (`behavior_project_cloud_api.py:181`), so reading the local NWB files directly is the identical loading path without needing network access; likewise `project_metadata/ophys_experiment_table.csv` is the offline equivalent of `get_ophys_experiment_table()`. The AI states that *all* processing (motion correction, segmentation, ROI QC, demixing, neuropil subtraction, dF/F, event detection, running-speed processing, eye tracking) is already applied upstream and materialised in the NWB, so the SDK merely exposes it. Parallelism was chosen because NWB reading (~250 MB/experiment) is the documented bottleneck.

## 1-b. How are the data split into subjects?

i. One subject per unique `mouse_id`. The mouse id is read from each session's SDK metadata (`ds0.metadata['mouse_id']`, cast to `str`); after conversion the global `subjects` list is the sorted set of mouse ids over the successfully converted sessions, and `subject_idx[s] = subjects.index(mouse_id_of_session_s)`. 38 mice result (37 from `VisualBehavior` plus the one Multiscope mouse 457841).

ii.
```python
        md0 = ds0.metadata
        info = dict(..., mouse_id=str(md0['mouse_id']), ...)
        result = dict(..., mouse_id=str(md0['mouse_id']), ...)
```
```python
    subjects = sorted({r['mouse_id'] for r in ok})
    ...
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['mouse_id']) for r in ok], dtype=np.int64),
```

iii. `mouse_id` is the SDK's unique animal identifier (CONVERSION_NOTES Step 5 variable-mapping table maps `ophys_experiment_table.mouse_id` → `subjects`/`subject_idx`). The count (38) was cross-checked against the metadata tables in Step 9/Step 10 Check 2 ("each session maps to one mouse; `subject_idx` correct — PASS"). The AI explicitly notes that keeping the Multiscope mouse adds "a 38th mouse" and the only VISl neurons.

## 1-c. How are the data split into sessions?

i. A session is one `ophys_session_id`, i.e. one continuous recording, not one NWB file. Experiments (imaging planes) sharing an `ophys_session_id` are grouped and their neurons concatenated along the neuron axis; `brain_region_idx` keeps track of which plane/area each neuron came from. Sessions are then curated: passive sessions (OPHYS_2/OPHYS_5) are excluded at table level; sessions with no eye-tracking rows (or with no finite `pupil_width`) are excluded; sessions with no running data are excluded; sessions yielding fewer than 2 usable trials are excluded. Sessions are emitted sorted by `ophys_session_id` (not by acquisition date). 174 active sessions → 171 kept (3 dropped for missing eye tracking: 795625712, 805989030, 832881662).

ii.
```python
    exp = select_sessions()
    groups = exp.groupby('ophys_session_id')
    session_ids = list(groups.groups.keys())
    ...
        jobs.append(dict(ophys_session_id=int(sid),
                         experiment_ids=g.ophys_experiment_id.tolist(), ...))
```
```python
        # consistency of behavior data across planes
        for ds in experiments[1:]:
            assert len(ds.trials) == len(trials), 'trial tables differ across planes'
            assert np.allclose(ds.trials.start_time.values, trials.start_time.values), \
                'trial start times differ across planes'
        ...
        if len(eye) == 0 or not np.any(np.isfinite(eye['pupil_width'].values)):
            return {'ophys_session_id': sid, 'error': 'no eye tracking data'}
        if len(running) == 0:
            return {'ophys_session_id': sid, 'error': 'no running data'}
        ...
        if len(neural_trials) < 2:
            return {'ophys_session_id': sid, 'error': 'fewer than 2 usable trials'}
```
```python
            neural = np.concatenate(mats, axis=0) if len(mats) > 1 else mats[0]
```

iii. CONVERSION_NOTES Step 4/Step 5 Key Decision 1: the whitepaper defines a session as "data collected in a single continuous recording", and the 6 active Multiscope sessions contain 3–7 simultaneously recorded planes sharing one behaviour/trial structure, so grouping by `ophys_session_id` is the whitepaper's own definition. Key Decision 2: only active behaviour sessions are used because "passive sessions have no licking/reward and hence no Go/Catch trial outcomes; Piet et al. also analyze only active sessions". Key Decision 7: sessions without eye tracking are dropped "because pupil diameter is a required output and cannot be fabricated". The cross-plane assertions were added so that merging planes can never mix mismatched behaviour tables.

## 1-d. How are the data split into trials?

i. Trials are the rows of the SDK `trials` table selected by `go | catch`, sorted by `start_time`. The trial window is the half-open interval `[start_time, stop_time)`, i.e. the full behavioural trial including the variable-length pre-change flash sequence and the ~4.2 s post-change response window (measured mean 8.47 s, range 7.0–12.6 s). Within a trial the window is divided into `nbins = floor((stop_time − start_time)/BIN_SIZE)` bins of 1/31 s, with bin *centres* at `start_time + (k+0.5)·BIN_SIZE`; the trailing partial bin is dropped so no bin extends past `stop_time`.

ii.
```python
        # go | catch  ==  exclude aborted and auto-rewarded, keep Go and Catch trials
        sel = trials[(trials.go | trials.catch)].sort_values('start_time')
        assert not sel.aborted.any() and not sel.auto_rewarded.any()
        assert sel.change_time.notna().all()
        ...
        for tid, tr in sel.iterrows():
            t_beg, t_stop = float(tr.start_time), float(tr.stop_time)
            nbins = int(np.floor((t_stop - t_beg) / BIN_SIZE))
            if nbins < 2:
                continue
            # bin centres (the sample time of each bin)
            tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
```

iii. Step 5 Key Decision 3: "Trials = `go | catch` from the SDK trials table, window `[start_time, stop_time)`. This is exactly the task's requirement (Go and Catch, no Aborted, no Auto-rewarded); `go`/`catch` are False on aborted and auto-rewarded trials, so the boolean mask is sufficient. Trials are variable length (7.0–12.6 s) because that is how the experiment defines them; the format allows variable T." Step 12 note 3 explicitly rejects using a short window around the change: "it would throw away ~60 % of the recorded task time". The assertions on `aborted`/`auto_rewarded`/`change_time` are self-checks that the `go|catch` mask really is equivalent to the exclusion rule.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level filters, in order: (1) `go | catch` (removes Aborted and Auto-rewarded); (2) `nbins < 2` → skip; (3) the trial's bin centres must lie entirely inside every plane's `ophys_timestamps` range, otherwise skip; (4) the trial must carry one of the four outcome flags, otherwise skip. Session-level: <2 usable trials → session dropped. The AI verified (Step 10 Check 5) that filters 2–4 never actually fire on this dataset — per-session trial counts exactly equal `go_trial_count + catch_trial_count` from `behavior_session_table.csv` for all 171 sessions — so they are purely defensive guards.

ii.
```python
            nbins = int(np.floor((t_stop - t_beg) / BIN_SIZE))
            if nbins < 2:
                continue
            # --- require full ophys coverage of the trial ---
            if any(tc[0] < ts[0] or tc[-1] > ts[-1] for ts in plane_ts):
                continue
```
```python
            if bool(tr.hit):
                outcome = 0
            elif bool(tr.miss):
                outcome = 1
            elif bool(tr.false_alarm):
                outcome = 2
            elif bool(tr.correct_reject):
                outcome = 3
            else:
                continue  # go/catch trial with no outcome flag: should not happen
```
```python
        if len(neural_trials) < 2:
            return {'ophys_session_id': sid, 'error': 'fewer than 2 usable trials'}
```

iii. Step 3 "Trial curation rules": "keep `go | catch` (this exactly excludes `aborted` and `auto_rewarded`, as required by the task); require the trial window to be covered by the ophys timestamps." Step 2 survey finding 6 established that "every go/catch trial has a non-NaN `change_time`, and every go/catch trial lies entirely inside the ophys timestamp range (0 exceptions in 202 experiments)", which is why the coverage filter is a no-op in practice. Step 10 Check 5: "No trial is silently dropped: trials/session equals `go_trial_count+catch_trial_count` for all 171 sessions, so none of the defensive filters ever triggered. The guards remain for robustness."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `dff_traces.dff` — the per-cell dF/F trace produced by the Allen pipeline — together with `ophys_timestamps` for its time base, for every plane of the session. `events.events` and `events.filtered_events` are selectable via `--trace` but `dff` is the default and is what produced `/app/converted_data.pkl` (`metadata['neural_data_type'] == 'dff'`).

ii.
```python
        for ds, eid in zip(experiments, eids):
            if trace_name == 'dff':
                tbl = ds.dff_traces
                mat = np.stack(tbl['dff'].values).astype(np.float32)
            else:
                tbl = ds.events
                mat = np.stack(tbl[trace_name].values).astype(np.float32)
            ts = np.asarray(ds.ophys_timestamps, dtype=float)
            assert mat.shape[1] == len(ts), 'trace length != n ophys timestamps'
```

iii. Two competing references were weighed (Step 4 discrepancy table, Step 6, Step 7). Piet et al. state "For all analysis of neural data we used the detected calcium events", but the AI measured that raw `events` are non-zero on only 0.41 % of frames, "which makes a per-timepoint decoder nearly blind". It then ran a controlled comparison of `dff` vs `filtered_events` vs `events` on the same 6 sessions with everything else identical: dF/F won on every output (image identity 0.575 vs 0.326 vs 0.210 validation balanced accuracy; change 0.651/0.600/0.582; running 0.307/0.249/0.229; pupil 0.305/0.242/0.226; outcome 0.345/0.257/0.260). It also argued that the whitepaper (`methods.txt`, "DF/F CALCULATION") documents dF/F as *the* processed neural signal of this dataset and contains no event-detection section, so dF/F is faithful to the dataset reference, and the deviation from Piet et al. is "required by the decoding task".

## 2-b. How is the `neural` data processed?

i. Almost none beyond what the Allen pipeline already did. Per plane the dF/F table is stacked into an `(n_cells, n_frames)` `float32` matrix; per trial each plane's matrix is sampled at the ophys frame nearest each bin centre; the per-plane trial matrices are then concatenated along the neuron axis in plane (sorted `ophys_experiment_id`) order. No z-scoring, no smoothing, no normalisation, no baseline subtraction is applied. The corresponding brain region label per neuron is built with the same plane ordering.

ii.
```python
            mats = [mat[:, nearest_index(ts, tc)]
                    for mat, ts in zip(plane_traces, plane_ts)]
            neural = np.concatenate(mats, axis=0) if len(mats) > 1 else mats[0]
            ...
            neural_trials.append(np.ascontiguousarray(neural, dtype=np.float32))
```
```python
        brain_region_per_neuron = np.concatenate(
            [[reg] * mat.shape[0] for reg, mat in zip(plane_region, plane_traces)])
```

iii. Step 1 notes: "dF/F does not need to be computed by us — it is precomputed in the NWB (`dff_traces`)... Neuropil subtraction, demixing and detrending are already applied." Step 12 Check 1(c) reports that per-neuron z-scoring of dF/F was tried as a controlled variant and gave no material improvement (0.573/0.649/0.310/0.304/0.286 vs the chosen 0.575/0.651/0.307/0.305/0.345), so the raw pipeline output was kept. `float32` was chosen to halve RAM/IPC cost.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional filtering. Every ROI returned by the SDK (`dff_traces` / `cell_specimen_table`) is kept, in all 199 planes — 29 168 neurons. Sessions with as few as 6 neurons are retained.

ii. No filtering code exists; the relevant documentation is in the metadata:
```python
            'curation':
                'Kept: active-behavior ophys sessions (OPHYS_1/3/4/6) present in the local data '
                'directory, with eye tracking. ... All ROIs returned by '
                'the SDK are used: ROI quality control (motion border, unions, duplicates, '
                'dendrites, low SNR) is already applied by the Allen pipeline.',
```

iii. Step 1/Step 3: "Neuron ('cell') quality filtering is also already applied upstream: `cell_specimen_table` contains only ROIs that passed the multi-label ROI classifier (`valid_roi == True` for every row in all 199 surveyed experiments; `len(cell_specimen_table) == len(events) == len(dff_traces)` everywhere), and every `cell_specimen_id` is a valid positive id". Step 3 lists the upstream rules (motion border, unions, duplicates, apical dendrites, too small/narrow/dim, non-positive demixed traces). Step 10 Check 5 justifies keeping tiny populations: "Sessions with tiny populations (6–20 neurons, Sst/Vip) are kept: the reference paper analyses exactly these small inhibitory populations; the decoder handles `n_neurons < npcs` by padding the projection with orthogonal rows." Step 10 Check 2 verified total neuron count against `ophys_cells_table.csv` (29 168, PASS).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is **trial start** (`trials.start_time`); `metadata['temporal_alignment_event']` documents this, with `off_start = 0.0` and `off_end = None` (variable trial length). Bin *k* of a trial is sampled at `start_time + (k+0.5)·BIN_SIZE`, and its neural value is the ophys frame whose timestamp is *nearest* that bin centre (per plane, using each plane's own `ophys_timestamps`). All other streams (stimulus, running, pupil) are sampled at exactly the same bin centres, so every stream shares one time base — the ophys timestamps — as the task requires ("Temporally align based on ophys timestamp").

ii.
```python
def nearest_index(sorted_times: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of the element of `sorted_times` nearest to each element of `query`."""
    idx = np.searchsorted(sorted_times, query)
    idx = np.clip(idx, 1, len(sorted_times) - 1)
    left = sorted_times[idx - 1]
    right = sorted_times[idx]
    idx = np.where(query - left <= right - query, idx - 1, idx)
    return np.clip(idx, 0, len(sorted_times) - 1)
```
```python
            tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
            ...
            mats = [mat[:, nearest_index(ts, tc)]
                    for mat, ts in zip(plane_traces, plane_ts)]
```
```python
            'temporal_alignment_event':
                'Trial start (trials.start_time of the AllenSDK trials table). Time bins run '
                'forward from trial start; neural data are the ophys (2-photon) frames nearest '
                'each bin centre, and all behavioural/stimulus streams are sampled at the same '
                'bin centres, i.e. everything is aligned on the ophys timestamps.',
            'off_start': 0.0,
            'off_end': None,
```

iii. Step 3: "Temporal synchronization of all streams is already done upstream on a single 100 kHz sync board; every stream in the NWB carries timestamps on that common clock. Aligning streams therefore only requires sampling them at a common set of times — here the ophys frame times." Step 5 Key Decision 4 defines bin *k* as "the ophys frame nearest to `start_time + k·dt`". The alignment was validated two ways: a per-bin spot check against independently loaded NWB data (Step 10 Check 2, PASS), and a change-triggered population average (`cache/alignment_check.py`) in which z-scored dF/F "is at its minimum exactly at the change bin, rises immediately after it and peaks at +0.29 s (GCaMP6f kinetics)... Activity never precedes the stimulus → no temporal misalignment and no sign/offset error."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. A single uniform bin size of `BIN_SIZE = 1/31 s = 32.258 ms` is used for every trial in every session (`metadata['time_bin_size'] = 32.258`). Yes, rebinning is applied, but it is a resampling rather than an averaging: each bin takes the value of the single nearest ophys frame. For the 165 single-plane experiments (native frame period 32.31–32.33 ms) this is effectively a 1:1 relabelling of the native frames with no interpolation; for the 34 Multiscope planes (93.23 ms, 11 Hz) it is a zero-order hold that upsamples ~3×, which is also what makes it possible to merge planes of a Multiscope session onto one grid. Behavioural streams are *linearly interpolated* (not held) onto the same bin centres. Converted trial lengths are 217–389 bins (mean 261.9 ≈ 8.45 s).

ii.
```python
BIN_SIZE = 1.0 / 31.0          # s; nominal single-plane 2P frame period (metadata ophys_frame_rate)
...
            nbins = int(np.floor((t_stop - t_beg) / BIN_SIZE))
            tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
...
            'time_bin_size': 1000.0 * BIN_SIZE,
```

iii. Step 4 discrepancy table: "Target format requires one common bin size for all sessions → resample onto a common 1/31 s grid (identity for 97 % of sessions, zero-order hold for the 6 Multiscope sessions)". Step 5 Key Decision 4 repeats this and stresses that for single-plane data "this is a 1:1 relabelling of the native ophys frames with no interpolation of neural data". Step 10 Check 3 lists it as a deliberate difference from the reference convention: "a uniform 1/31 s bin instead of each rig's native period — the target format demands one bin size for all sessions." Step 10 Check 5 verified `T == floor((stop_time − start_time)/dt)` for every trial and noted that bin centres at `+0.5·dt` can never coincide exactly with a flash boundary since 750 ms and 32.258 ms are incommensurate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The `stimulus_presentations` table restricted to the `change_detection_behavior` stimulus block: columns `start_time`, `end_time`, `image_name`, `omitted`. It is *not* derived from the trials table's `initial_image_name` / `change_image_name`.

ii.
```python
BEHAVIOR_BLOCK = 'change_detection_behavior'
...
        beh = stim[stim.stimulus_block_name == BEHAVIOR_BLOCK].sort_values('start_time')
        flash_start = beh.start_time.values.astype(float)
        flash_end = beh.end_time.values.astype(float)
        flash_img = beh.image_name.values.astype(str)
        flash_omitted = beh.omitted.values.astype(bool)
        # omitted "flashes" are gray screen; make sure they never match a bin
        flash_end = np.where(flash_omitted | ~np.isfinite(flash_end), -np.inf, flash_end)
```

iii. Step 1: "Tutorials restrict the stimulus table to `stimulus_block_name.str.contains('change_detection')`, i.e. the behavior block, excluding the 5-min gray screens and the 5-min natural movie." Step 5 variable-mapping table cites the tutorial `plot_stimuli` as the reference function. Using the flash table rather than the trials table gives the actual physical screen content at every bin, including the 500 ms inter-flash blanks and the 5 % omitted flashes, which the trials-table columns cannot express.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each bin centre the code finds the last flash that started at or before it (`searchsorted(..., 'right') − 1`) and checks whether the bin centre is still inside that flash (`start ≤ tc < end`). If yes, the bin gets that flash's image code; if no (inter-flash grey period) or if the flash was omitted (its `end_time` was set to `−inf`), the bin gets code 0 = `gray`. Codes are first assigned from the *session's own* sorted image list (1…8), then remapped in `main()` to a *global* sorted list of the 16 unique image names across sets A and B, giving 17 classes overall (`['gray'] + 16 images`). Measured distribution over the full dataset: gray 0.670, each image 0.020–0.022.

ii.
```python
        image_names = sorted(set(flash_img[~flash_omitted]) - {'omitted'})
        img_lookup = {name: i + 1 for i, name in enumerate(image_names)}  # 0 = gray
        ...
            # --- image identity (0 = gray / omitted / blank) ---
            j = np.searchsorted(flash_start, tc, side='right') - 1
            j = np.clip(j, 0, len(flash_start) - 1)
            on_screen = (tc >= flash_start[j]) & (tc < flash_end[j])
            img = np.zeros(nbins, dtype=np.int64)
            if np.any(on_screen):
                img[on_screen] = [img_lookup[n] for n in flash_img[j[on_screen]]]
```
```python
    # remap per-session image labels (which used the session's own image list) to global indices
    global_img = {name: i + 1 for i, name in enumerate(image_names)}
    for r in ok:
        remap = np.zeros(len(r['image_names']) + 1, dtype=np.int64)
        for local, name in enumerate(r['image_names']):
            remap[local + 1] = global_img[name]
        for out in r['output']:
            out[0] = remap[out[0]]
```

iii. Step 5 Key Decision 9: "Image identity uses the union of the 16 image names across image sets A and B plus a 'gray' class for the inter-flash blanks and omitted flashes (17 categories). The image sets are disjoint, so a session only ever expresses 8 of the 16 image classes — that is a property of the experiment, not of the conversion." Step 12 Interpretation note 1 records that the alternative reading (carry the last presented image through the blank) was implemented and measured (+0.019 balanced accuracy) and deliberately rejected "as less literal: with it, the screen content and the label disagree for two thirds of the time." The gray fraction was sanity-checked against theory: 0.670 vs the (750−250)/750 = 0.667 expected from the 250 ms flash / 750 ms cycle, slightly higher because ~4–5 % of flashes are omitted.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed at exactly the same bin centres `tc` as the neural data, from the flash `start_time`/`end_time` on the shared sync clock — so it is aligned by construction, frame for frame, with no separate resampling step.

ii.
```python
            tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
            ...
            mats = [mat[:, nearest_index(ts, tc)] for mat, ts in zip(plane_traces, plane_ts)]
            ...
            j = np.searchsorted(flash_start, tc, side='right') - 1
            on_screen = (tc >= flash_start[j]) & (tc < flash_end[j])
```

iii. Step 3: all streams carry timestamps on one 100 kHz sync clock, so "aligning streams therefore only requires sampling them at a common set of times". The alignment was verified independently (Step 10 Check 2): image identity was re-derived with "a slow, independent 'which flash contains this bin centre' implementation" for 5 random trials per session (PASS), and for every trial the last image before `change_time` was checked to equal `initial_image_name` and the first image at/after `change_time` to equal `change_image_name` (PASS).

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `trials.go` and `trials.change_time`, combined with the `stimulus_presentations` flash table (`start_time`, `end_time`, `is_change`) of the behaviour block.

ii.
```python
        flash_is_change = beh.is_change.values.astype(bool)
        ...
            chg = np.zeros(nbins, dtype=np.int64)
            if bool(tr.go):
                k = int(np.searchsorted(flash_start, float(tr.change_time), side='right') - 1)
                assert flash_is_change[k], 'change_time does not match an is_change flash'
                chg[(tc >= flash_start[k]) & (tc < flash_end[k])] = 1
```

iii. Step 2 exploration findings 8–10: "`trials.change_time` is exactly equal (diff = 0.0 s) to the `start_time` of the `stimulus_presentations` row with `is_change` (go) or `is_sham_change` (catch) for that trial"; "n(`is_change` flashes) == n(go)+n(auto_rewarded) and n(`is_sham_change`) == n(catch)"; "On catch trials `initial_image_name == change_image_name` (sham change: image does NOT change); on go trials they always differ." Hence `go` is the correct gate and the `is_change` flash is the correct window; the `assert` is a run-time consistency check of that finding.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary per-bin indicator: zeros everywhere, set to 1 for the bins whose centre falls inside the 250 ms presentation of the changed image on Go trials. Catch trials (sham change, image does not actually change) are all zeros. There is no smoothing or other transform. Result: ~7–8 bins per go trial; measured global distribution 0.975 / 0.025.

ii. See 4-a — the whole computation is those four lines.

iii. Step 5 Key Decision 10: "Image change is marked over the 250 ms of the changed-image flash (≈8 bins) rather than a single bin, so that the label covers the stimulus event itself; a 1-bin delta would be 0.4 % of the data and almost unlearnable. Catch (sham change) trials contain no image change → all zeros."

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding of a continuous quantity is involved — the variable is intrinsically binary, with 2 classes named `['no_change', 'change']`. The only "threshold" is the choice of time window: the 250 ms changed-image flash, i.e. `[flash_start[k], flash_end[k])`. The alternative 750 ms image-presentation interval (flash + following grey) used by Piet et al. was implemented and measured as a controlled variant.

ii.
```python
                chg[(tc >= flash_start[k]) & (tc < flash_end[k])] = 1
...
    output_values = [
        [GRAY_LABEL] + image_names,
        ['no_change', 'change'],
        ...
```

iii. Step 12 Interpretation note 2: "Image change window. 1 during the 250 ms presentation of the changed image ('right after a change in image identity'). The 750 ms image-presentation-interval convention of Piet et al. was measured and is no better." The measurement (Step 12 Check 1c) gave image_change validation balanced accuracy 0.6512 for the 250 ms window vs 0.6442 for the 750 ms window, so the more literal reading of the instruction was kept. Sanity checks in Step 10 Check 2 confirmed "exactly one contiguous change epoch per go trial" and "all-zero on every catch trial".

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same bin centres `tc` as the neural data; the change window is delimited by the flash `start_time`/`end_time` from the same sync clock.

ii. See 4-a/4-c.

iii. Identical reasoning to 3-c. Verified in Step 10 Check 2 ("change == bins inside the changed flash on every go trial — PASS") and in the `--show-processing` figures, which overlay `change_time` on the converted outputs and show the label rising exactly at the change.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `dataset.running_speed` — the SDK's filtered linear running speed (cm/s, ~60 Hz) and its `timestamps`. Not `raw_running_speed`.

ii.
```python
        running = ds0.running_speed
        for ds in experiments[1:]:
            if len(running) == 0:
                running = ds.running_speed
        ...
        run_t = running['timestamps'].values.astype(float)
        run_v = running['speed'].values.astype(float)
```

iii. Step 4 discrepancy table: "`running_speed` (10 Hz low-pass) vs `raw_running_speed` — both present; whitepaper describes both; filtered is the default → Use `running_speed` (filtered), which is what the tutorials plot." Step 1 records that this stream is "already unwrapped, transient-corrected and 10 Hz low-pass filtered". Step 2 survey: "Running speed has no NaNs; range over the dataset −24.1 … 99.9 cm/s."

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation of speed onto the trial's bin centres (`np.interp`, ignoring any non-finite samples), then per-session quintile discretisation into 5 classes. Because `np.interp` clamps outside the sample range, no NaNs can be introduced at trial edges; the code nevertheless asserts that all interpolated values are finite and drops the session otherwise.

ii.
```python
def interp_nonan(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Linear interpolation of y(x) at xq, ignoring NaN samples of y (e.g. blinks)."""
    ok = np.isfinite(y) & np.isfinite(x)
    if not np.any(ok):
        return np.full(xq.shape, np.nan)
    return np.interp(xq, x[ok], y[ok])
...
            run_trials.append(interp_nonan(run_t, run_v, tc))
...
        run_all = np.concatenate(run_trials)
        if not np.all(np.isfinite(run_all)):
            return {'ophys_session_id': sid, 'error': 'non-finite running speed'}
        run_lab, run_edges = quantile_bin(run_all)
```

iii. Step 5 variable-mapping table: "linear interpolation onto bin times, then per-session quintile bins (0–4)", with `running_processing.py` cited as the reference code. The AI's rationale for interpolation rather than nearest-sample is that the 60 Hz running stream is faster than the 31 Hz bins and already low-pass filtered, so linear interpolation is the natural resampling; and Step 3 records that all streams live on one sync clock so interpolation across streams is legitimate.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins whose edges are the 20th/40th/60th/80th percentiles **computed separately within each session**, over all bins of all included trials of that session. Labels are 0–4, named `run_q0 … run_q4`. Edges are stored per session in `metadata['session_info'][i]['running_quantile_edges']`. Every session therefore has exactly 20 % of its bins in each class (measured global fractions 0.2000 × 5).

ii.
```python
def quantile_bin(values: np.ndarray, nbins: int = N_QUANTILE_BINS):
    """Discretise into `nbins` equal-percentile bins. Returns (labels, edges)."""
    edges = np.quantile(values, np.arange(1, nbins) / nbins)
    labels = np.searchsorted(edges, values, side='right').astype(np.int64)
    return np.clip(labels, 0, nbins - 1), edges
...
        run_lab, run_edges = quantile_bin(run_all)
```

iii. Step 5 Key Decision 8: "Running speed / pupil quintiles are computed per session, over all timepoints of that session's included trials. Pupil width is measured in camera pixels and is not comparable across sessions/rigs (different zoom and eye position), and running distributions differ strongly across mice; per-session quantization yields the 'five equal percentile bins' the task asks for in every session and a well-posed balanced 5-class problem. (Pooled-dataset quintiles would give the same ≈20 % marginal but degenerate per-session distributions.)" Step 12 Check 3 addresses the leakage question: "every quantity (including the running/pupil quantile edges) is computed per session from all of that session's trials, i.e. identically for train and test trials (the edges are unsupervised statistics of the behavioural variable, not of the labels being predicted)."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated directly onto the same bin centres `tc` used to sample the neural data, so alignment is by construction.

ii.
```python
            tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
            mats = [mat[:, nearest_index(ts, tc)] for mat, ts in zip(plane_traces, plane_ts)]
            ...
            run_trials.append(interp_nonan(run_t, run_v, tc))
```

iii. Step 3/Step 10 Check 3: all streams are hardware-synced on one clock, so sampling them at a common set of times is sufficient. The `--show-processing` figure includes a dedicated panel overlaying the raw 60 Hz running trace and the interpolated bin samples for one trial, plus the quintile edges; Step 7 reports "converted bins lie exactly on the raw traces". Step 10 Check 2 re-derived the quintile labels from raw `running_speed` independently of the conversion code (PASS).

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `dataset.eye_tracking`, column `pupil_width` (with `timestamps`). The SDK already sets `pupil_width` to NaN on `likely_blink` frames, and those NaNs are simply skipped by the interpolator. If plane 0 has an empty eye-tracking table, the code falls back to other planes; if no plane has usable data, the session is dropped.

ii.
```python
        eye = ds0.eye_tracking
        for ds in experiments[1:]:
            if len(eye) == 0:
                eye = ds.eye_tracking
        if len(eye) == 0 or not np.any(np.isfinite(eye['pupil_width'].values)):
            return {'ophys_session_id': sid, 'error': 'no eye tracking data'}
        ...
        eye_t = eye['timestamps'].values.astype(float)
        eye_v = eye['pupil_width'].values.astype(float)
```

iii. Step 4 discrepancy table: "`eye_tracking` has `pupil_width`, `pupil_height`, `pupil_area`... tutorial comment 'function to plot pupil diameter' uses `pupil_width` → Use `pupil_width` as pupil diameter; linearly interpolate across blink/outlier NaNs." Step 1 documents that `pupil_width`/`pupil_height`/`pupil_area` "are set to NaN on `likely_blink` frames (missing fit or |z|>3 outlier, dilated by 2 frames)", so excluding NaNs is exactly equivalent to excluding `likely_blink` rows. Step 2 survey: 3 experiments have zero eye-tracking rows; in the rest the NaN fraction is median 2.9 %, max 29.6 %.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Identical machinery to running speed: NaN (blink/outlier) samples are dropped, the remaining `pupil_width` samples are linearly interpolated onto the trial's bin centres — which implicitly interpolates *across* blink gaps — and the result is discretised into per-session quintiles. A finiteness check drops the session if anything non-finite survives.

ii.
```python
            pup_trials.append(interp_nonan(eye_t, eye_v, tc))
...
        pup_all = np.concatenate(pup_trials)
        if not np.all(np.isfinite(pup_all)):
            return {'ophys_session_id': sid, 'error': 'non-finite pupil after interpolation'}
        pup_lab, pup_edges = quantile_bin(pup_all)
```

iii. Step 5 variable-mapping table: "interpolate over blink NaNs, linear interpolation onto bin times, then per-session quintile bins", citing `eye_tracking_processing.py`. Step 10 Check 5: "in the rest, blink/outlier NaNs (median 2.9 %, max 29.6 % of frames) are linearly interpolated over before sampling." The AI's reasoning is that pupil size is a slow signal, so bridging a short blink by linear interpolation is preferable to injecting a missing-data class.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five equal-percentile bins with edges computed **per session** from all bins of all that session's included trials; labels 0–4, named `pupil_q0 … pupil_q4`; edges stored in `metadata['session_info'][i]['pupil_quantile_edges']`. Measured global fractions 0.2000 × 5.

ii.
```python
        pup_lab, pup_edges = quantile_bin(pup_all)
```
(with `quantile_bin` as quoted in 5-c)

iii. Step 5 Key Decision 8, quoted in 5-c, argues the case most strongly for pupil: "Pupil width is measured in camera pixels and is not comparable across sessions/rigs (different zoom and eye position)... Pooled-dataset quintiles would give the same ≈20 % marginal but degenerate per-session distributions." Step 10 Check 2 independently recomputed the quintile labels from the raw `eye_tracking.pupil_width` and confirmed fractions of 0.2 ± 0.01 (PASS).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the same bin centres `tc` as the neural data; no separate alignment step.

ii.
```python
            tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE
            ...
            pup_trials.append(interp_nonan(eye_t, eye_v, tc))
```

iii. Same justification as running speed (shared 100 kHz sync clock; sampling all streams at a common set of times). The `--show-processing` figure contains a pupil alignment panel overlaying the raw ~30 Hz trace with the interpolated bin samples and the quintile edges; Step 7 reports no anomalies.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the SDK trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, tested in that order.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
            if bool(tr.hit):
                outcome = 0
            elif bool(tr.miss):
                outcome = 1
            elif bool(tr.false_alarm):
                outcome = 2
            elif bool(tr.correct_reject):
                outcome = 3
            else:
                continue  # go/catch trial with no outcome flag: should not happen
```

iii. Step 5 variable-mapping table maps `trials.hit/miss/false_alarm/correct_reject` → `output[4]`, citing the tutorials' trial-type queries (`trials.query('hit')` etc.) as the reference usage. Hit/miss apply to Go trials, false alarm/correct rejection to Catch trials, so the four labels exactly partition the `go|catch` set — which the AI confirmed by checking total counts against `behavior_session_table.csv` (hit 13 940 / miss 24 520 / FA 834 / CR 4 681, exact match, Step 10 Check 2). This is also the stated reason passive sessions were excluded: they have 0 hits and 0 false alarms by construction.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The trial's outcome is mapped to an integer 0–3 and then broadcast as a constant over all bins of that trial, occupying row 4 of the `(5, T)` output array. `output_values[4] = ['hit', 'miss', 'false_alarm', 'correct_reject']`. The resulting per-bin distribution is hit 0.313 / miss 0.562 / FA 0.018 / CR 0.107 (per-trial: 0.317/0.558/0.019/0.106 — the small difference is the trial-length weighting).

ii.
```python
        outputs = []
        pos = 0
        for i, n in enumerate([len(x) for x in run_trials]):
            out = np.empty((5, n), dtype=np.int64)
            out[0] = img_trials[i]
            out[1] = chg_trials[i]
            out[2] = run_lab[pos:pos + n]
            out[3] = pup_lab[pos:pos + n]
            out[4] = outcome_trials[i]
            outputs.append(out)
            pos += n
        assert pos == len(run_all)
```

iii. Step 5 Key Decision 11: "Outputs are all time-varying except trial outcome, which is static per trial and broadcast over the trial's bins (the target format requires one array per trial, so all five outputs share the (5, T) array)." Step 12 Check 3 flags this as the source of the largest train/validation gap (1.48×) and explains it: "within a training trial the target is constant for ~260 bins, so the model can partly memorise the trial's overall neural state." Step 10 Check 2 verified that the outcome is constant within each trial and equals the trials-table flags (PASS).

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. A layered set of guards, each logged:
- **Whole-session failure**: `convert_session` wraps everything in `try/except` and returns an `{'error': ...}` dict with a traceback instead of crashing the pool; the reason is recorded in `metadata['excluded_sessions']`.
- **Missing eye tracking**: if plane 0's table is empty, fall back to the other planes; if no plane has finite `pupil_width`, drop the session (3 sessions: 795625712, 805989030, 832881662). Same fallback for running speed.
- **Blinks / outlier eye frames**: NaN `pupil_width` samples are skipped and bridged by linear interpolation.
- **Non-finite behaviour after interpolation**: session dropped with an explicit error.
- **Omitted flashes**: their `end_time` (and any non-finite `end_time`) is forced to `−inf` so they can never match a bin and always fall through to `gray`.
- **Trials at the edge of the recording**: trials whose bin centres are not fully covered by every plane's ophys timestamps are skipped; trials shorter than 2 bins are skipped; the trailing partial bin of every trial is dropped.
- **Trials with no outcome flag**: skipped.
- **Degenerate sessions**: fewer than 2 usable trials → session dropped.
- **Multi-plane inconsistency**: assertions that the trials table (length and all start times) is identical across planes before merging.
- **Plotting**: wrapped in `try/except` so a plotting bug can never cause loss of a session's data.

ii.
```python
    except Exception as exc:  # pragma: no cover - defensive
        import traceback
        return {'ophys_session_id': sid, 'error': repr(exc),
                'traceback': traceback.format_exc()}
```
```python
        for ds in experiments[1:]:
            if len(eye) == 0:
                eye = ds.eye_tracking
            if len(running) == 0:
                running = ds.running_speed
        if len(eye) == 0 or not np.any(np.isfinite(eye['pupil_width'].values)):
            return {'ophys_session_id': sid, 'error': 'no eye tracking data'}
```
```python
        flash_end = np.where(flash_omitted | ~np.isfinite(flash_end), -np.inf, flash_end)
```
```python
            if nbins < 2:
                continue
            if any(tc[0] < ts[0] or tc[-1] > ts[-1] for ts in plane_ts):
                continue
```
```python
        if job.get('show_processing'):
            # plotting must never lose a session's data
            try:
                _plot_processing(...)
            except Exception as exc:
                ...
```
```python
        'excluded_sessions': [{'ophys_session_id': r['ophys_session_id'],
                               'reason': r['error']} for r in bad],
```

iii. Step 6: "Robustness/edge cases handled: sessions/planes with missing eye tracking or running data (session dropped with a logged reason), trials shorter than 2 bins, trials not fully covered by the ophys timestamps, `end_time` NaN on omitted flashes, go/catch trials without an outcome flag, plotting failures (never lose data), non-finite interpolated behaviour." Step 5 Key Decision 7 justifies dropping eye-less sessions rather than imputing: "pupil diameter is a required output and cannot be fabricated." Step 10 Check 5 confirms that apart from the 3 eye-tracking sessions, none of the guards ever fired: "trials/session equals `go_trial_count+catch_trial_count` for all 171 sessions, so none of the defensive filters (T < 2 bins, ophys coverage, missing outcome flag) ever triggered. The guards remain for robustness."

## 9-a. What are the most time-consuming steps of the code?

i. Reading the NWB files (`BehaviorOphysExperiment.from_nwb_path`) dominates: ~4–5 s per single-plane experiment and ~17 s for a 7-plane Multiscope session, versus 1–3 s for trace extraction plus per-trial assembly. Pickling the 8.83 GB output is the second cost (~25 s). The script instruments this with a `timing` dict per session and prints a running ETA. Measured full run: 79 s wall clock with 16 worker processes.

ii.
```python
        t0 = time.time()
        experiments = [load_experiment(e) for e in eids]
        timing['load_nwb'] = time.time() - t0
        ...
        timing['load_traces'] = time.time() - t0
        ...
        timing['trials'] = time.time() - t0
```
```python
        with ProcessPoolExecutor(max_workers=nworkers) as ex:
            for i, r in enumerate(ex.map(convert_session, jobs)):
                ...
                print('[%3d/%3d] session %d  %s  (elapsed %.0fs, %.1fs/session, eta %.0fs)' % ...)
```

iii. Step 6: "Bottleneck is NWB reading (~250 MB/experiment). `ProcessPoolExecutor` with 16 workers over sessions; each NWB is opened exactly once and all five streams are taken from that one object." Step 7's timing table estimates ≈1100 s of single-worker work → ≈2 min at 16 workers plus ~2 min pickling, and Step 9 reports the actual 79 s.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The heavy inner work is already vectorised (`nearest_index` is one `searchsorted` per trial per plane; `np.interp` for behaviour; one `searchsorted` for the whole quantile assignment of a session). What remains as Python-level loops:
- `for tid, tr in sel.iterrows()` — row-wise pandas iteration over ~260 trials per session; `.itertuples()` or pulling the needed columns into NumPy arrays once would be markedly faster. Trial-level `searchsorted`/slicing could in principle be done for all trials at once.
- `img[on_screen] = [img_lookup[n] for n in flash_img[j[on_screen]]]` — a per-bin Python dict lookup inside every trial; precomputing an integer `flash_code` array once per session would make this a pure fancy-index.
- The per-session load of Multiscope planes (`[load_experiment(e) for e in eids]`) is serial inside a worker.
- In `main()`, `brain_regions.index(x)` is called once per neuron (linear search over the region list), and `subjects.index(...)` once per session; a dict would be O(1). The final output-assembly and image-remap loops are also per-trial Python loops.

ii.
```python
        for tid, tr in sel.iterrows():
            ...
            img[on_screen] = [img_lookup[n] for n in flash_img[j[on_screen]]]
```
```python
        'brain_region_idx': [np.array([brain_regions.index(x)
                                       for x in r['brain_region_per_neuron']], dtype=np.int64)
                             for r in ok],
```

iii. Step 6: "Nearest-frame lookup is a vectorised `np.searchsorted` per trial, not a Python loop over bins... Behaviour streams are interpolated with `np.interp` (vectorised) on the trial's bin centres. Per-session quantile edges are computed once on the concatenated session values, then applied to all trials with one `np.searchsorted`." Step 7 credits these with "~50× on the per-trial step". The one remaining inefficiency the AI names explicitly is the serial plane load: "loading the planes of a Multiscope session is serial inside a worker (7 planes ≈ 17 s); this affects 6 of 171 sessions and is not worth extra process nesting." Given the 79 s total runtime, none of the above is worth optimising further.

## 9-c. What processing does the code repeat multiple times?

i. Very little, and by design. Each NWB is opened exactly once and all five streams are read from that one object. The repetitions that do exist are:
- For multi-plane (Multiscope) sessions, `ds.trials` is materialised for *every* plane in order to run the cross-plane consistency assertions, and `ds.eye_tracking` / `ds.running_speed` are touched on other planes in the fallback path — i.e. the behaviour tables are reconstructed up to 7 times for 6 of the 171 sessions.
- `np.concatenate(run_trials)` / `np.concatenate(pup_trials)` are recomputed inside `_plot_processing` for the histogram panels (only under `--show-processing`).
- Image identity is coded twice: once with the session-local lookup inside the worker, then remapped to global codes in `main()`.

ii.
```python
        for ds in experiments[1:]:
            assert len(ds.trials) == len(trials), 'trial tables differ across planes'
            assert np.allclose(ds.trials.start_time.values, trials.start_time.values), \
                'trial start times differ across planes'
```
```python
    for r in ok:
        remap = np.zeros(len(r['image_names']) + 1, dtype=np.int64)
        for local, name in enumerate(r['image_names']):
            remap[local + 1] = global_img[name]
        for out in r['output']:
            out[0] = remap[out[0]]
```

iii. Step 6: "each NWB is opened exactly once and all five streams are taken from that one object" (credited with "~2× vs re-opening per stream"). The repeated per-plane `trials` access is a deliberate correctness check — Step 10 Check 5: "Multi-plane sessions: behavior tables are asserted identical across planes (trial count and all start times) before merging". The two-stage image coding is a consequence of the worker processes not knowing the global image list until all sessions are done.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A modest amount, essentially all of it diagnostic/provenance rather than compute-heavy:
- `cell_specimen_ids` is concatenated and returned per session but never written into the final `data` dict — it is computed and then discarded.
- A full `trial_info` dict per trial (~44 000 dicts) is built and used only for plotting and the summary; it is not saved.
- A large `info` block per session (equipment name, frame rate, imaging depths, table-level go/catch/aborted/auto counts, quantile edges, …) is stored in `metadata['session_info']` but is not consumed by the decoder.
- The cross-plane `ds.trials` assertions rebuild behaviour tables that are then thrown away.
- `inputs` are materialised as one `(0, T)` `float32` array per trial even though the task specifies no decoder inputs; the decoder only needs the shape.
- The two-stage local→global image remap rewrites row 0 of every output array a second time.
- Whole-session dF/F matrices are read for every plane even though only the `[start_time, stop_time)` windows of go/catch trials are kept (~60 % of session time is retained, so some is genuinely discarded) — this is unavoidable given the NWB layout.

ii.
```python
        result = dict(
            ...
            cell_specimen_ids=np.concatenate(plane_cellid),
            info=info,
            trial_info=trial_info,
            timing=timing,
        )
```
```python
            trial_info.append(dict(trials_id=int(tid), start_time=t_beg, stop_time=t_stop,
                                   change_time=float(tr.change_time), nbins=nbins,
                                   go=bool(tr.go), catch=bool(tr.catch), outcome=outcome))
```
```python
        inputs = [np.zeros((0, x.shape[1]), dtype=np.float32) for x in neural_trials]
```

iii. The AI does not flag any of this as waste; it presents the extra metadata as deliberate provenance. Step 13's file inventory and the `metadata` block show the intent: `session_info`, `excluded_sessions`, `output_descriptions`, `curation` and `source` are included so that "an expert" can audit the conversion, and Step 10 Check 2 uses exactly these per-session records (plane order, quantile edges, trial counts) to run the independent sanity checks. The cost is negligible relative to the 8.83 GB of neural data and the 79 s runtime.
