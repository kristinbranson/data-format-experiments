# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI works directly off the local Allen "visual-behavior-ophys-1.1.0" cache instead of the AllenSDK project cache object (it first tried `VisualBehaviorOphysProjectCache.from_local_cache`, found the on-disk layout incompatible, and fell back to reading the NWB files directly). The experiment list comes from `project_metadata/ophys_experiment_table.csv`, intersected with the NWB files actually present on disk (284 files). Three metadata filters are then applied: `behavior_type == 'active_behavior'` (drop passive-viewing sessions), `experience_level == 'Familiar'` (drop Novel 1 / Novel >1), and `project_code == 'VisualBehavior'` (single-plane rig only, dropping the Multiscope experiments). That leaves 88 experiments = 88 sessions = 37 mice. Each selected experiment is then loaded in a worker process with `BehaviorOphysExperiment.from_nwb_path`, and all streams (`events`, `stimulus_presentations`, `running_speed`, `eye_tracking`, `trials`, `metadata`) are read from that one object. Loading is parallelised with a 16-worker (22 at run time) `multiprocessing.Pool`.

ii.
```python
DATA_DIR = '/app/data/visual-behavior-ophys-1.1.0'
EXP_DIR = os.path.join(DATA_DIR, 'behavior_ophys_experiments')
META = os.path.join(DATA_DIR, 'project_metadata', 'ophys_experiment_table.csv')

def select_experiments():
    meta = pd.read_csv(META)
    have = set()
    for f in os.listdir(EXP_DIR):
        if f.endswith('.nwb'):
            have.add(int(f.split('_')[-1].split('.')[0]))
    sel = meta[meta.ophys_experiment_id.isin(have)
               & (meta.behavior_type == 'active_behavior')
               & (meta.experience_level == 'Familiar')
               & (meta.project_code == 'VisualBehavior')].copy()
    sel = sel.sort_values('ophys_experiment_id').reset_index(drop=True)
    return sel
```
```python
    exp = BehaviorOphysExperiment.from_nwb_path(
        os.path.join(EXP_DIR, f'behavior_ophys_experiment_{eid}.nwb'))
```
```python
    with Pool(args.workers) as pool:
        results = pool.map(process_experiment, [(e, args.signal) for e in eids])
```

iii. From the module docstring and trajectory: passive-viewing sessions are excluded because the mouse makes no behavioural report (the AI describes them as containing "no trials/behavioral report"); familiar-only follows the paper's data-selection statement ("Except where noted we used familiar image set sessions"); the single-plane `VisualBehavior` project code is kept because those experiments are all imaged at ~31 Hz, so every session shares one time-bin size, whereas the few locally available Multiscope sessions (22 planes from 4 sessions of a single mouse) run at 11 Hz and would break the "same bin size for all trials and sessions" requirement. Direct NWB reading was adopted after the SDK's local-cache loader rejected the directory layout; the AI verified that all 110 familiar+active NWBs load successfully before committing.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values (taken from each experiment's NWB `metadata`, stored as strings), sorted; `subject_idx` for each session is the index of that session's mouse in the sorted list. 37 mice result.

ii.
```python
        mouse_id=str(md['mouse_id']), structure=str(md['targeted_structure']),
```
```python
    subjects = sorted({r['mouse_id'] for r in results})
...
        data['subject_idx'].append(subjects.index(r['mouse_id']))
...
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. No explicit discussion beyond using the SDK's canonical animal identifier; the AI cross-checked the resulting counts (37 mice, 1–7 sessions each) against the metadata table in the trajectory.

## 1-c. How are the data split into sessions?

i. One session = one processed experiment (one NWB file). Because the selection is restricted to the single-plane `VisualBehavior` project, each `ophys_session_id` contains exactly one imaging plane, so experiment and session are 1:1 (the AI verified 88 experiments ↔ 88 `ophys_session_id`s). The `ophys_session_id` is nevertheless recorded per session in `metadata['session_info']`. Sessions are ordered by `ophys_experiment_id`, not by acquisition date. A session is kept only if it yields ≥2 usable trials and has eye-tracking data.

ii.
```python
    eids = [int(x) for x in sel.ophys_experiment_id.values]
    ...
    results = [r for r in results if r.get('skip') is None and len(r['trials']) >= 2]
```
```python
        ophys_session_id=int(md['ophys_session_id']),
```
```python
        session_info.append(dict(
            ophys_experiment_id=r['eid'], ophys_session_id=r['ophys_session_id'],
            mouse_id=r['mouse_id'], cre_line=r['cre_line'],
            session_type=r['session_type'], targeted_structure=r['structure'], ...))
```

iii. The AI established in the trajectory that "single-plane familiar+active: 88 experiments = 88 sessions, 37 mice, 14663 neurons, all VISp, uniform 31 Hz", i.e. that treating an experiment as a session is exact for this project code, and that the multi-plane project (where the mapping would be many-to-one) was being excluded anyway for bin-size reasons.

## 1-d. How are the data split into trials?

i. Trials come from the SDK `trials` table. The AI selects the rows flagged `go` or `catch`, which by construction excludes aborted and auto-rewarded trials. Each trial spans `start_time` → `stop_time` (the full trial window: variable number of pre-change flashes plus the ~4.2 s post-change response window), converted to ophys-frame indices with `np.searchsorted`. Trial lengths in the output range 217–389 frames (~7–12.5 s), mean 263 frames. Trials shorter than 2 frames are dropped.

ii.
```python
    tr = exp.trials
    gc = tr[(tr.go | tr.catch)]
    trials = []
    for tid, row in gc.iterrows():
        a = int(np.searchsorted(ts, row.start_time, side='left'))
        b = int(np.searchsorted(ts, row.stop_time, side='left'))
        if b - a < 2:
            continue
```
```python
        trials.append(dict(
            neural=neural[:, a:b].copy(),
            image_id=image_id[a:b].copy(),
            change=change[a:b].copy(), ...))
```

iii. Docstring: 'Trials: the "go" and "catch" trials defined by the Allen trials table, i.e. aborted and auto-rewarded trials are excluded' — exactly what the instructions ask for. In the trajectory the AI checked the trials table columns and confirmed the go/catch/aborted/auto_rewarded flags are mutually exclusive and that every go/catch trial has exactly one outcome label and a non-NaN `change_time`. It also considered a fixed `[-3, +4.2] s` window around `change_time` (verifying min pre-change = 2.79 s, min post-change = 4.23 s, min inter-change gap = 7.49 s) but settled on the native `start_time`→`stop_time` trial definition, "segment each recording session into individual trials based on how they are defined in the experiment".

## 1-e. How are trials filtered based on quality controls?

i. Four filters, applied per trial: (1) trials with fewer than 2 ophys frames are dropped; (2) trials whose pupil signal is valid for <50% of frames (blink / lost eye) are dropped — 0.6% of go/catch trials, 21,624 of 21,756 kept; (3) trials whose outcome flags do not resolve to exactly one of hit/miss/false_alarm/correct_reject are dropped; (4) at the session level, an experiment with no eye-tracking rows or <100 finite pupil samples is skipped entirely (one experiment, 806456687), and sessions with <2 surviving trials are dropped. No filtering on neural criteria.

ii.
```python
    et = exp.eye_tracking
    if len(et) == 0:
        return dict(eid=eid, skip='no eye tracking')
    ...
    if good.sum() < 100:
        return dict(eid=eid, skip='no valid pupil samples')
```
```python
        if b - a < 2:
            continue
        if np.mean(pupil_valid[a:b]) < 0.5:
            continue  # pupil not tracked for most of this trial (blink / lost eye)
        oc = [o for o in OUTCOMES if bool(row[o])]
        if len(oc) != 1:
            continue
```
```python
    results = [r for r in results if r.get('skip') is None and len(r['trials']) >= 2]
```

iii. The AI scanned all 88 candidate sessions first, finding ~4% blink NaNs with occasional long gaps and one session with no pupil data at all; it then confirmed by direct inspection that experiment 806456687 has zero eye-tracking rows, so it cannot supply the required pupil output and is correctly excluded. The 50% coverage rule was chosen so that a trial's pupil quintiles are based mostly on real measurements rather than long interpolations across blinks; the AI verified that this costs only 0.6% of trials. The ≥2-trial rule follows the explicit format requirement ("at least two trials within each session").

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `exp.events`, i.e. the AllenSDK detected-calcium-event table, using the `filtered_events` column by default (`--signal` can select the raw `events` column instead). Each row is one cell (`cell_specimen_id`), each column entry a full-session trace already on the ophys frame clock. dF/F traces are *not* used.

ii.
```python
    ev_table = exp.events
    cell_ids = list(ev_table.index.values)
    neural = np.stack([np.asarray(x, dtype=np.float32)
                       for x in ev_table[signal].values])  # (ncells, nframes)
    assert neural.shape[1] == nframes
```
```python
    ap.add_argument('--signal', default='filtered_events',
                    choices=['events', 'filtered_events'])
```

iii. Docstring: "Neural data: detected calcium events (the paper: 'For all analysis of neural data we used the detected calcium events'). We use the AllenSDK 'filtered_events' trace, which is exactly those detected events convolved with a causal half-normal kernel: the raw event trace is nonzero in only ~0.3% of the 32 ms frames, which carries almost no information in a single time bin, while the filtered trace keeps the event times but spreads each event over the following few frames." The AI also ran both variants through the reference decoder on an 8-session pilot: filtered_events beat raw events on every output (image identity 0.263 vs 0.202, change 0.636 vs 0.605, running 0.243 vs 0.220).

## 2-b. How is the `neural` data processed?

i. Only one processing step beyond reading the trace: each cell is divided by the standard deviation of its own trace over the whole session (zero-SD cells divided by 1). No detrending, smoothing, mean subtraction, or rebinning. Traces are stored as float32 and sliced per trial.

ii.
```python
    # Scale each cell by the standard deviation of its trace over the whole session.
    # Event magnitudes are in units of dF/F and vary by orders of magnitude between
    # cells (and between cre lines / imaging depths); without this, the per-session
    # projection of the decoder is dominated by a few high-variance cells.
    sd = neural.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    neural = neural / sd
```

iii. Docstring as quoted above. In the trajectory the AI first read the decoder code, noted that the per-session projection is SVD/PCA-initialised on the raw traces with a fixed learning rate so trace scale matters, and then measured the effect on the pilot: per-neuron SD scaling raised validation balanced accuracy for every output (image identity 0.284 vs 0.263, change 0.649 vs 0.636, outcome 0.290 vs 0.255). It describes the step as "standard, well-justified … prevents high-variance cells dominating the per-session PCA/projection".

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neural quality control is applied by the AI. Every cell in `exp.events` (i.e. every ROI that passed the Allen segmentation/QC pipeline) is kept: 14,437 neurons over 87 sessions, 7–666 per session. The AI noticed that the sparse event traces make some trials entirely zero in the smallest sessions and deliberately kept them.

ii. No filtering code; the full event table is used:
```python
    ev_table = exp.events
    cell_ids = list(ev_table.index.values)
```
```python
        data['brain_region_idx'].append(
            np.full(len(r['cell_ids']), regions.index(r['structure']), dtype=np.int64))
```

iii. Trajectory (step 45): "Warnings are about trials with all-zero neural data, which occur in sessions with very few cells (7-13 Sst/Vip neurons) where the sparse detected events can be entirely absent for an 8-second trial. These are genuine data, not errors." The AI also confirmed earlier that `cell_specimen_table.valid_roi` is all True in the released NWBs, i.e. the SDK has already removed invalid ROIs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Everything is placed on the ophys frame clock (`exp.ophys_timestamps`), as the instructions require ("temporally align based on ophys timestamp"). Since the event traces are natively sampled at those frame times, no resampling of the neural data is needed. Each trial is the contiguous slice of frames between the frame index of `trials.start_time` and that of `trials.stop_time` (`side='left'` searchsorted, i.e. the first frame at or after each boundary). `metadata['temporal_alignment_event']` records trial start, with `off_start = 0.0` and `off_end = None` (variable-length trials).

ii.
```python
    ts = np.asarray(exp.ophys_timestamps, dtype=np.float64)
    nframes = len(ts)
    ...
        a = int(np.searchsorted(ts, row.start_time, side='left'))
        b = int(np.searchsorted(ts, row.stop_time, side='left'))
        ...
            neural=neural[:, a:b].copy(),
```
```python
        temporal_alignment_event=(
            'trial start (start_time of the go/catch trial in the Allen trials '
            'table); all data streams are resampled onto the ophys frame '
            'timestamps of the imaging plane'),
        off_start=0.0,
        off_end=None,
```

iii. Docstring: "All data streams are put on one common clock: the ophys frame timestamps of the imaging plane … The stimulus table, running speed (60 Hz) and pupil (30 Hz eye-tracking camera) are resampled onto those frame times, so one time bin = one ophys frame (~32.3 ms)." The AI spot-checked one converted trial against the raw session and confirmed that the change flag lands exactly on the frame given by `searchsorted(change_time)` and that flashes occupy ~8 frames (250 ms at 31 Hz).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One time bin = one ophys frame. No rebinning, downsampling, or smoothing is applied to any stream; the other streams are interpolated *onto* the frame times instead. The reported `time_bin_size` is the mean over sessions of the median inter-frame interval: 32.3194 ms (~31 Hz). Uniformity across sessions is guaranteed by restricting to the single-plane `VisualBehavior` rig.

ii.
```python
        dt=float(np.median(np.diff(ts))),
...
    dt = float(np.mean([r['dt'] for r in results]))
    data['metadata'] = dict(..., time_bin_size=dt * 1000.0, ...)
```
```python
  * Only the single-plane VisualBehavior project code. All those experiments are
    imaged at ~31 Hz, so that every session shares the same time-bin size (as
    required by the target format). The few Multiscope (11 Hz) sessions available
    locally (4 sessions from a single mouse) would break that requirement.
```

iii. Docstring as quoted. The AI verified across all 88 candidate sessions that the single-plane rig gives a uniform `dt = 0.03232 s`, whereas Multiscope gives ~11 Hz, and used that as the deciding argument for excluding Multiscope (the paper's own rig) in favour of the format requirement that "time bins should be the same size for all trials and sessions".

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From `exp.stimulus_presentations`, restricted to `active == True` (the change-detection block, excluding the 5-min gray-screen blocks and the natural-movie block) and `omitted == False`: the columns `image_name`, `start_time`, `end_time`. It is **not** derived from the trials table's `initial_image_name` / `change_image_name`. This gives a genuinely per-frame variable: the image code during each 250 ms flash and a dedicated `gray_screen` category (0) during the 500 ms inter-flash gray periods and during omitted flashes.

ii.
```python
    sp = exp.stimulus_presentations
    flashes = sp[(sp.active == True) & (sp.omitted == False)]  # noqa: E712
    image_names = sorted(flashes.image_name.unique())
    image_id = np.zeros(nframes, dtype=np.int64)   # 0 == gray screen / omitted
    ...
    starts = np.searchsorted(ts, flashes.start_time.values, side='left')
    stops = np.searchsorted(ts, flashes.end_time.values, side='left')
    idx = np.array([image_names.index(n) + 1 for n in flashes.image_name.values])
```

iii. The instruction asks for "image identity (of the image presented during the non-grey screen)"; the AI reads this as the identity of the image actually on the screen at each moment, with gray as its own level, and takes it from the stimulus table, which is the only source that has per-flash times. Its metadata spells out the stimulus structure ("250 ms flashes separated by 500 ms of gray screen, 5% of flashes omitted"). Omitted flashes are treated as gray because no image was presented.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Two-stage integer coding. Per session, the session's own sorted image names are coded 1..n (0 reserved for gray). At assembly time a global sorted list of image names across all sessions is built and the per-session codes are remapped into the global codes with a small lookup array. All sessions turn out to use the same 8-image familiar set A, so the global list is `['im061','im062','im063','im065','im066','im069','im077','im085']` and `output_values[0] = ['gray_screen', ...those 8]`. In the converted data, gray occupies 66.9% of time bins and each image ~4.1%.

ii.
```python
    image_names = sorted({n for r in results for n in r['image_names']})
...
        img_map = np.array([0] + [image_names.index(n) + 1
                                  for n in r['image_names']])
...
            out[0] = img_map[t['image_id']]
```
```python
                output_values=[
                    ['gray_screen'] + image_names, ...
```

iii. The AI verified in its scan that every familiar+active session uses image set A with the same 8 images, so a single global code list is consistent across sessions; the mapping is recorded in `output_values` so the labels are recoverable.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The image-code time series is built for the whole session on the ophys frame grid (`searchsorted` of each flash's `start_time`/`end_time`), and then sliced with exactly the same `[a:b]` frame indices as the neural data, so the two are aligned frame-by-frame by construction. A 250 ms flash occupies ~8 frames.

ii.
```python
    for a, b, i, c in zip(starts, stops, idx, is_change):
        image_id[a:b] = i
        if c:
            change[a:b] = 1
...
        trials.append(dict(
            neural=neural[:, a:b].copy(),
            image_id=image_id[a:b].copy(), ...))
```

iii. Same one-clock argument as 2-d. The AI's spot-check against the raw session confirmed that the converted image-identity codes match the flash times and the change image for the trial.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the same `stimulus_presentations` table, using the `is_change` flag of each active, non-omitted flash (together with that flash's `start_time`/`end_time`). `is_change` is True only for a flash whose image differs from the preceding one, so catch (sham-change) trials never get a 1.

ii.
```python
    is_change = flashes.is_change.values.astype(bool)
    for a, b, i, c in zip(starts, stops, idx, is_change):
        image_id[a:b] = i
        if c:
            change[a:b] = 1
```

iii. The instruction says the variable should "have value of 1 right after a change in image identity, otherwise 0". Using the stimulus table's own `is_change` flag makes the change indicator consistent with the image-identity trace derived from the same table (the 1 marks exactly the flash at which the identity trace changes value).

## 4-b. What processing is involved in computing `output` *Image change*?

i. None beyond the binary indicator: `change` is 1 for the frames of the change flash itself (~8 frames ≈ 250 ms) and 0 everywhere else, including the gray period after the change flash. Across the dataset 2.6% of time bins are labelled `change`. `output_values[1] = ['no_change','change']`.

ii. See 4-a; at assembly:
```python
            out[1] = t['change']
```

iii. The AI's description is that the flag marks "whether an image change just occurred"; it is tied to the duration of the changed image presentation rather than to the full 750 ms image-presentation interval used in the paper's behavioural analysis. The AI verified in its spot-check that "the change flag lands exactly on the change flash (frame 117 = searchsorted of change_time), flashes last ~8 frames (250 ms at 31 Hz)".

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — the variable is natively binary (0/1, int64), produced directly from the `is_change` boolean.

ii.
```python
        if c:
            change[a:b] = 1
```
```python
                output_values=[..., ['no_change', 'change'], ...]
```

iii. N/A — a binary flag as specified by the instructions.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: built on the full-session ophys frame grid from flash start/end times, then sliced with the same trial frame indices as the neural data.

ii.
```python
            change=change[a:b].copy(),
```

iii. Same single-clock argument as 2-d/3-c; verified in the spot-check that the change frame index equals `searchsorted(ts, change_time)`.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `exp.running_speed`, using its `timestamps` and `speed` columns (the running-disc encoder signal, ~60 Hz, in cm/s).

ii.
```python
    rs = exp.running_speed
    running = np.interp(ts, rs.timestamps.values, rs.speed.values)
```

iii. The SDK's standard locomotion interface; the AI checked its sampling rate (dt ≈ 0.0167 s) and time coverage against the ophys timestamps during exploration.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation onto the ophys frame timestamps (`np.interp`, which clamps to the first/last sample outside the encoder's time range rather than producing NaN), no smoothing or filtering, followed by quintile discretization (see 5-c). The continuous interpolated values are carried per trial and only converted to bins at assembly time.

ii.
```python
    running = np.interp(ts, rs.timestamps.values, rs.speed.values)
...
            running=running[a:b].copy(),
...
            out[2] = np.searchsorted(run_edges, t['running'], side='right')
```

iii. Docstring: "The stimulus table, running speed (60 Hz) and pupil (30 Hz eye-tracking camera) are resampled onto those frame times, so one time bin = one ophys frame (~32.3 ms)." The SDK already hardware-syncs all streams to a common clock, so interpolation onto the ophys timestamps is the only step needed.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-occupancy (quintile) bins, with the bin edges computed **per session** from all retained time bins of that session (an option `--binning global` computes one set of edges pooled across all sessions, but the default and the shipped dataset use per-session edges). Bins are assigned with `searchsorted(edges, x, side='right')`, giving labels 0–4; the edges are recorded per session in `metadata['session_info']`. The resulting marginal distribution is exactly 20% per bin.

ii.
```python
    qs = np.linspace(0, 100, NBINS + 1)[1:-1]
    if args.binning == 'global':
        run_all = np.concatenate([t['running'] for r in results for t in r['trials']])
        ...
    else:
        edges = {}
        for r in results:
            run_all = np.concatenate([t['running'] for t in r['trials']])
            pup_all = np.concatenate([t['pupil'] for t in r['trials']])
            edges[r['eid']] = (np.percentile(run_all, qs), np.percentile(pup_all, qs))
```
```python
            out[2] = np.searchsorted(run_edges, t['running'], side='right')
```

iii. Code comment: "Pupil size is measured in camera pixels and its absolute value differs by up to ~3x between sessions/mice (eye size, camera position), and mice also differ strongly in how much they run. The decoder has a separate projection per session, so only within-session variation can be decoded. Bin edges are therefore computed per session (equal-occupancy quintiles within a session)." In the pilot the AI observed that globally pooled edges left some sessions with 81% of samples in a single bin, i.e. an almost constant target within those sessions, and switched to per-session quintiles for that reason.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The interpolation target is the ophys timestamp vector itself, so the running series is already on the neural time grid for the whole session; the trial slice uses the same `[a:b]` indices as the neural data.

ii.
```python
    ts = np.asarray(exp.ophys_timestamps, dtype=np.float64)
    running = np.interp(ts, rs.timestamps.values, rs.speed.values)
...
            neural=neural[:, a:b].copy(),
            running=running[a:b].copy(),
```

iii. Same argument as 2-d: one common clock for all streams, so the alignment is guaranteed by construction.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `exp.eye_tracking`, using `pupil_area`, `likely_blink` and `timestamps` (30 Hz eye camera). The diameter is computed from the area rather than taken from `pupil_width`; frames flagged `likely_blink` are set to NaN before interpolation. Sessions with an empty eye-tracking table or fewer than 100 finite samples are dropped.

ii.
```python
    et = exp.eye_tracking
    if len(et) == 0:
        return dict(eid=eid, skip='no eye tracking')
    area = et.pupil_area.values.astype(np.float64).copy()
    area[et.likely_blink.values.astype(bool)] = np.nan  # blinks are not measurements
    diam = 2.0 * np.sqrt(area / np.pi)
    good = np.isfinite(diam)
    if good.sum() < 100:
        return dict(eid=eid, skip='no valid pupil samples')
```

iii. Metadata: `pupil_measure='diameter = 2*sqrt(pupil_area/pi) in camera pixels, blinks removed and interpolated'` — i.e. the equivalent-circle diameter of the fitted pupil ellipse, which uses both ellipse axes instead of one. The AI measured the blink fraction (~2–4% of frames, with occasional long gaps) across all candidate sessions before deciding how to handle them.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames removed → linear interpolation of the remaining samples onto the ophys frame times (`np.interp`, so blink gaps are bridged) → per-frame validity flag marking frames whose nearest *real* sample is more than 0.5 s away → quintile discretization (6-c). The validity flag is used only to reject trials (>50% invalid), not to alter the values.

ii.
```python
    et_ts = et.timestamps.values.astype(np.float64)
    # interpolate the (blink) gaps, on the eye-tracking clock, then onto ophys frames
    pupil = np.interp(ts, et_ts[good], diam[good])
    # frames whose nearest valid pupil sample is far away are flagged as missing
    nearest = np.abs(ts - et_ts[good][np.clip(
        np.searchsorted(et_ts[good], ts), 0, good.sum() - 1)])
    nearest_prev = np.abs(ts - et_ts[good][np.clip(
        np.searchsorted(et_ts[good], ts) - 1, 0, good.sum() - 1)])
    gap = np.minimum(nearest, nearest_prev)
    pupil_valid = gap < 0.5  # within half a second of a real measurement
```

iii. Interpolating across short blinks keeps the trial intact, while the 0.5 s gap criterion prevents long stretches of invented values from being fed to the decoder as if they were measurements; trials that are mostly invented are dropped instead (see 1-e). The AI chose the 50%/0.5 s thresholds after quantifying blink fractions and maximum NaN gaps per session.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Exactly as running speed: five equal-occupancy quintile bins with per-session edges (default) computed from all retained frames of that session, assigned with `searchsorted(..., side='right')` to labels 0–4, edges stored per session in `session_info`. The marginal distribution is 20% per bin.

ii.
```python
            edges[r['eid']] = (np.percentile(run_all, qs), np.percentile(pup_all, qs))
...
            out[3] = np.searchsorted(pup_edges, t['pupil'], side='right')
```

iii. The per-session rationale is stated primarily for pupil: absolute pupil size in camera pixels differs up to ~3× across sessions/mice because of eye size and camera placement, so globally pooled quintiles mostly encode which session you are in, which the per-session projection of the decoder cannot exploit; within-session quintiles encode arousal fluctuations, which it can.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed: interpolated onto the session's ophys timestamps and then sliced with the same trial frame indices as the neural data.

ii.
```python
    pupil = np.interp(ts, et_ts[good], diam[good])
...
            pupil=pupil[a:b].copy(),
            neural=neural[:, a:b].copy(),
```

iii. Single common clock (the ophys frame times), as stated in the docstring; eye-tracking timestamps are already synced to that clock by the SDK.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table — `hit`, `miss`, `false_alarm`, `correct_reject` — for the selected go/catch trial.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
        oc = [o for o in OUTCOMES if bool(row[o])]
        if len(oc) != 1:
            continue
        trials.append(dict(..., outcome=OUTCOMES.index(oc[0]), ...))
```

iii. These are the SDK's canonical change-detection outcome labels. The AI verified across all 88 candidate sessions that "every go/catch trial has exactly one outcome", and made the "exactly one" requirement an explicit guard rather than an assumption.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Mapped to a fixed integer code 0–3 in the order hit, miss, false_alarm, correct_reject, then broadcast to a constant time series over the trial's frames so that all five outputs share one `(5, T)` array. Observed distribution: hit 29.9%, miss 57.6%, false alarm 2.1%, correct reject 10.5%.

ii.
```python
            out[4] = t['outcome']
...
                output_values=[..., ['hit', 'miss', 'false_alarm', 'correct_reject']]
```

iii. Trajectory (step 53): "the task says … trial outcome is 'Static per-trial'; I made outcome a constant time series within the trial, which is acceptable (all outputs in one array must share the time dimension)."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing pupil is the main case and is handled at three levels: blink samples are removed (NaN) rather than used; short gaps are interpolated; frames more than 0.5 s from any real sample are flagged invalid; trials with >50% invalid frames are dropped; sessions with no eye tracking or <100 finite samples are dropped whole. Trials shorter than 2 frames, and trials whose outcome flags are ambiguous, are dropped. Running speed outside the encoder's coverage is clamped to the nearest sample by `np.interp` rather than becoming NaN. Neurons with a zero-SD trace are divided by 1 instead of 0. Behaviour-block boundaries are handled by taking only `active`, non-omitted flashes, with everything else labelled gray. All-zero neural trials in very-small-cell-count sessions are knowingly kept as real data. There is no `try/except` around per-experiment processing, so an unexpected read failure in one worker would abort the whole pool rather than skipping that session.

ii.
```python
    sd = neural.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
```
```python
    if len(et) == 0:
        return dict(eid=eid, skip='no eye tracking')
    ...
    if good.sum() < 100:
        return dict(eid=eid, skip='no valid pupil samples')
```
```python
        if b - a < 2:
            continue
        if np.mean(pupil_valid[a:b]) < 0.5:
            continue
        oc = [o for o in OUTCOMES if bool(row[o])]
        if len(oc) != 1:
            continue
```
```python
    results = [r for r in results if r.get('skip') is None and len(r['trials']) >= 2]
```

iii. The AI pre-scanned every candidate session for pupil NaN fraction, maximum NaN gap, outcome completeness, NaN change times and window in-bounds before writing the converter, then confirmed afterwards that only one experiment (806456687, zero eye-tracking rows) and 0.6% of trials were lost. The verify step reported no NaN/Inf and no format errors.

## 9-a. What are the most time-consuming steps of the code?

i. Reading and decoding the 88 NWB files with `BehaviorOphysExperiment.from_nwb_path` (~300 MB each), which materialises the event traces and the stimulus/trials/eye/running tables, is by far the dominant cost and is I/O plus deserialisation bound. It is amortised with a process pool (22 workers at run time): the full conversion took 31 s wall clock versus 5 m 37 s of user CPU time. The second cost is writing the 4.3 GB pickle (dense float32 neural arrays, 21,624 trials × up to 666 neurons × ~263 frames), plus the memory traffic of copying every trial slice twice (`.copy()` in the worker, then `.astype(np.float32)` at assembly).

ii.
```python
    with Pool(args.workers) as pool:
        results = pool.map(process_experiment, [(e, args.signal) for e in eids])
```
```python
    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=4)
```

iii. The AI measured load time per experiment during exploration, checked the machine (128 CPUs, 1 TB RAM) and chose a pool size accordingly; no further optimisation was pursued because the whole conversion finishes in about half a minute.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python-level loops remain: (1) the per-flash loop that paints image identity and the change flag into the session-length arrays — this could be done with index arithmetic (`np.repeat` over run lengths, or `np.add.at`/cumulative-sum tricks) since the flash intervals are disjoint and sorted; (2) the per-trial loop over `gc.iterrows()`, which is the slowest way to walk a DataFrame — the `searchsorted` calls for all trial start/stop times are already vectorisable in one call, and `itertuples` or column arrays would avoid building a Series per row; (3) the assembly loop over trials in `main`, where the five output rows are written one trial at a time. Also `image_names.index(n)` inside a list comprehension is a linear scan per flash where a dict lookup would be O(1), and `np.stack([np.asarray(x) for x in ev_table[signal].values])` builds the neural matrix row by row.

ii.
```python
    for a, b, i, c in zip(starts, stops, idx, is_change):
        image_id[a:b] = i
        if c:
            change[a:b] = 1
```
```python
    for tid, row in gc.iterrows():
        a = int(np.searchsorted(ts, row.start_time, side='left'))
        b = int(np.searchsorted(ts, row.stop_time, side='left'))
```
```python
    idx = np.array([image_names.index(n) + 1 for n in flashes.image_name.values])
```

iii. Not discussed in the trajectory; the AI's implicit position is that data loading dominates (31 s total), so these loops were left in their readable form.

## 9-c. What processing does the code repeat multiple times?

i. (1) Every trial's arrays are copied twice — `.copy()` when the trial is cut in the worker, then `.astype(np.float32)` at assembly even though the array is already float32 (a second full copy of the 4 GB of neural data). (2) The running and pupil traces are concatenated across trials once to compute percentile edges and then re-read trial by trial to apply them. (3) In `--binning global` mode the dict comprehension re-evaluates `np.percentile(run_all, qs)` and `np.percentile(pup_all, qs)` over the full pooled arrays once per session (87 identical computations) instead of once. (4) Image names are coded twice: local per-session codes first, then remapped to global codes through `img_map`. (5) Two `np.searchsorted` passes over the pupil timestamps to get the forward and backward nearest-sample gaps.

ii.
```python
        edges = {r['eid']: (np.percentile(run_all, qs), np.percentile(pup_all, qs))
                 for r in results}
```
```python
            neural_s.append(t['neural'].astype(np.float32))
```
```python
        img_map = np.array([0] + [image_names.index(n) + 1
                                  for n in r['image_names']])
```

iii. Not discussed. The two-stage image coding follows from processing sessions independently in worker processes before the global label set is known; the duplicate `astype` is defensive.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items never reach the decoder: (1) `change_time`, `start_time`, `stop_time` and `trial_id` are stored on every trial dict and never used afterwards; (2) `cell_specimen_ids` for all 14,437 cells are written into `metadata['session_info']`, as are per-session bin edges and frame rates — informative, but unused by the decoder; (3) the continuous running and pupil traces are carried per trial through the whole pipeline and discarded once the quintile labels are computed; (4) image identity, change, running and pupil are computed for *all* frames of the session, including the 5-minute gray-screen blocks, the natural-movie block and the inter-trial gaps, although only the frames inside go/catch trials are kept; (5) per-cell SD normalisation is computed over the full session for the same reason; (6) the `pupil_valid` gap arrays are computed for every frame but used only as a per-trial average; (7) `exp.events` loads both the `events` and `filtered_events` columns while only one is used; (8) the `input` entries are genuine `(0, T)` empty arrays created per trial, as the task specifies no inputs. Finally, the neural arrays are ~98% zeros but are stored densely, which is what makes the output file 4.3 GB.

ii.
```python
            outcome=OUTCOMES.index(oc[0]),
            trial_id=int(tid),
            change_time=float(row.change_time),
            start_time=float(row.start_time),
            stop_time=float(row.stop_time),
```
```python
            cell_specimen_ids=[int(c) for c in r['cell_ids']]))
```
```python
            input_s.append(np.zeros((0, T), dtype=np.float32))
```

iii. Not discussed explicitly; the extra per-trial fields and `session_info` entries were kept for provenance and for the AI's own verification passes (it used `session_info` afterwards to check kept-trial fractions and to identify the dropped experiment), and the dense format is imposed by the target data specification.
