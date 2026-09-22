# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All data are read exclusively through the AllenSDK `VisualBehaviorOphysProjectCache`, instantiated with `from_local_cache(cache_dir='/app/data')` (no S3 traffic; the agent checked that `use_static_cache=True` expects a different layout and fails). `select_sessions()` pulls `bc.get_ophys_experiment_table()`, intersects its index with the ophys-experiment ids that actually exist as `.nwb` files in the local cache (284 of the 1936 released experiments = the complete set of experiments for 38 mice), drops `passive` experiments, and groups the remaining 202 experiments by `ophys_session_id` into 174 sessions. Every kept experiment is then opened once with `bc.get_behavior_ophys_experiment(exp_id)` inside a per-session worker process (16-way `multiprocessing.Pool`). Both project codes are kept (`VisualBehavior` single-plane/Scientifica, 31 Hz, and `VisualBehaviorMultiscope`/Mesoscope, 11 Hz per plane). No `.nwb` file is opened with `h5py`/`pynwb`.

ii.
```python
def get_cache():
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    return VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)


def select_sessions(sample=False):
    """Active (non-passive) ophys sessions and their experiment ids."""
    bc = get_cache()
    et = bc.get_ophys_experiment_table()
    local_ids = set(int(f.split('_')[-1].split('.')[0]) for f in os.listdir(
        os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0',
                     'behavior_ophys_experiments')))
    et = et.loc[sorted(set(et.index).intersection(local_ids))]
    et = et[~et.passive]                      # active behaviour sessions only
    groups = []
    for sid, sub in et.groupby('ophys_session_id'):
        groups.append((int(sid), sorted(int(i) for i in sub.index)))
    groups.sort()
```
```python
    for k, eid in enumerate(experiment_ids):
        ds = bc.get_behavior_ophys_experiment(int(eid))
```
```python
    if args.workers > 1 and len(jobs) > 1:
        with Pool(min(args.workers, len(jobs))) as p:
            results = p.map(process_session, jobs)
```

iii. From CONVERSION_NOTES Step 1/2: the experiment table is the SDK's canonical listing, and `get_behavior_ophys_experiment` is the documented accessor used by all `/app/tutorials`. The local-file intersection is needed because "the AllenSDK metadata is unmodified and still lists every released experiment" while only 284 NWB files are present; without it, loading would try to reach S3. Passive sessions (OPHYS_2/5) are dropped because the agent verified they contain "0 licks / 0 rewards, all trials miss/CR", so the trial-outcome output would be degenerate and the animal is not performing the task — consistent with the paper ("passive viewing ... was not analyzed here"). Both rigs are kept because restricting to the paper's multi-plane + familiar-image subset would leave "6 sessions from **1** mouse", which "cannot support a multi-subject decoder benchmark".

## 1-b. How are the data split into subjects?

i. One subject per unique `mouse_id`. Each session's mouse id is read from the first loaded experiment's `dataset.metadata['mouse_id']`; the global `subjects` list is the sorted set of those ids, and `subject_idx[i] = subjects.index(mouse_id_of_session_i)`. Result: 38 mice, 2–9 sessions each.

ii.
```python
        mouse_id=str(meta['mouse_id']),
```
```python
    subjects = sorted(set(r['mouse_id'] for r in results))
    ...
        data['subject_idx'].append(subjects.index(r['mouse_id']))
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. `mouse_id` is the SDK's unique animal identifier (documented in the Step 1 function table as a field of `dataset.metadata` and of the experiment table). The agent cross-checked the count against its raw scan of the cache (38 mice, both from the experiment table and from the per-session metadata) and a sanity check verified "subject id matches `metadata['mouse_id']`".

## 1-c. How are the data split into sessions?

i. A session = one `ophys_session_id`. All *active* imaging planes (ophys experiments) belonging to that session id are concatenated along the neuron axis into a single session entry: single-plane sessions contribute 1 plane, Multiscope sessions 3–7 planes. Each plane is binned on its **own** `ophys_timestamps` before concatenation, because the agent measured that Multiscope planes are offset by ~23 ms per plane group. Sessions are emitted in sorted `ophys_session_id` order (not chronological per mouse). 174 active sessions are found; 171 survive the eye-tracking filter.

ii.
```python
    for sid, sub in et.groupby('ophys_session_id'):
        groups.append((int(sid), sorted(int(i) for i in sub.index)))
```
```python
    for k, eid in enumerate(experiment_ids):
        ds = bc.get_behavior_ophys_experiment(int(eid))
        ...
        ots = np.asarray(ds.ophys_timestamps, dtype=float)
        binned = bin_sum(E, ots, edges)                # (ncells, ntrials, NBINS)
        ...
        neural_planes.append(binned)
        region_per_neuron += [ds.metadata['targeted_structure']] * E.shape[0]
    ...
    neural = np.concatenate(neural_planes, axis=0)     # (nneurons, ntrials, NBINS)
```

iii. Notes Step 5: "**Session** = one `ophys_session_id`; all *active* imaging planes (experiments) of that session are concatenated along the neuron axis". This is the natural unit because all planes of one session share the same behaviour, the same trials table and the same stimulus, so they form one simultaneously-recorded population. Per-plane timestamps are used because of the measured inter-plane offsets (Step 2 "Multiscope planes of the same session are offset by ~23 ms per plane group"). A dedicated sanity script (`sanity_multiplane.py`) verified on all 6 multi-plane sessions that "each plane's neurons sit at the right row offset, per-plane timestamps used".

## 1-d. How are the data split into trials?

i. A trial is one row of the SDK `dataset.trials` table with `go == True` or `catch == True`. The neural/behavioural window for that trial is **not** the table's `start_time`→`stop_time`; it is a fixed window of **[-2.0 s, +3.0 s] around `trials.change_time`**, divided into 20 bins of 250 ms. Trials are taken from the first loaded plane's trials table (identical across planes of a session).

ii.
```python
BIN_SIZE = 0.25          # s, = image duration, 1/3 of the 750 ms flash cycle
OFF_START = -2.0         # s relative to the change (start of trial window)
OFF_END = 3.0            # s relative to the change (end of trial window)
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 20
```
```python
def bin_edges_for_trials(change_times):
    """(ntrials, NBINS+1) array of bin edge times (s) for every trial."""
    offsets = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    return change_times[:, None] + offsets[None, :]
```
```python
            trials = ds.trials
            keep = trials[(trials.go | trials.catch)]
            if len(keep) < 2:
                return None
            change_times = keep.change_time.values.astype(float)
            edges = bin_edges_for_trials(change_times)
```

iii. The task text prescribes the trial set ("Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials"), and the agent verified in `trial.py` that `go = not catch and not auto_rewarded` for non-aborted trials, so `go | catch` is exactly the required subset. The fixed change-aligned window is justified in Step 5: the target format demands equal-size time bins and `off_start`/`off_end` relative to an alignment event; the agent measured that `change_time - start_time >= 3.02 s`, `stop_time - change_time = 4.23 s` (essentially constant) and the next trial's change is ≥ 4.48 s away, so "[-2, +3] s ... never leaves its own trial nor contains a second change". A window-length ablation ([-2,+3] vs [-2,+4] vs [-1,+3]) changed accuracy only within noise.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level: only `go | catch` rows are kept (aborted and auto-rewarded dropped); the agent verified separately that no go/catch trial has a missing `change_time`. Session-level: passive sessions are dropped; sessions with fewer than 2 go/catch trials are dropped (none occurred, min 39); the 3 sessions with **no** eye-tracking table at all (795625712, 805989030, 832881662) are dropped because the pupil output cannot be defined. Neuron-level: none beyond the SDK's own `exclude_invalid_rois=True`. Net: 174 → 171 sessions, 44,892 → 43,975 trials.

ii.
```python
            keep = trials[(trials.go | trials.catch)]
            if len(keep) < 2:
                return None
```
```python
    if pupil_binned is None:
        # a session without any usable eye tracking cannot supply the pupil output
        return dict(skip=True, ophys_session_id=int(ophys_session_id),
                    reason='no eye tracking')
```
```python
    et = et[~et.passive]                      # active behaviour sessions only
```
```python
    skipped = [r for r in results if r is None or r.get('skip')]
    results = [r for r in results if r is not None and not r.get('skip')]
```

iii. Notes Step 5, Key Decisions 3 and 5: "Active sessions only: passive sessions have no licking, so trial outcome is degenerate and the animal is not performing the task"; "Exclude the 3 sessions with no eye-tracking ... because the pupil output cannot be defined; all other sessions keep blinks interpolated (median 2.9 % of frames)". Step 10 Check 2 confirmed "no aborted / auto-rewarded trial kept", "hit+miss == n go trials; FA+CR == n catch trials", and that the resulting catch fraction (0.1254) matches the whitepaper's ~12.5 %.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `dataset.dff_traces['dff']` (the pipeline-computed ΔF/F trace per valid ROI) together with `dataset.ophys_timestamps`. `--neural-signal events|filtered_events` switches to `dataset.events['events']` / `['filtered_events']` (the L0 detected calcium events used by the reference paper), but the delivered dataset uses **dff**.

ii.
```python
        if signal == 'dff':
            ev = ds.dff_traces
            E = np.vstack(ev['dff'].values).astype(np.float32)
        else:
            ev = ds.events
            E = np.vstack(ev[signal].values).astype(np.float32)   # (ncells, nframes)
        ots = np.asarray(ds.ophys_timestamps, dtype=float)
```

iii. This is an explicitly documented *deviation* from the reference paper, which states "For all analysis of neural data we used the detected calcium events". The agent ran a controlled comparison on an identical 20-session subset (Notes Step 8b / Step 10 Check 3): validation balanced accuracy for `events`(sum) 0.282/0.588/0.267/0.250/0.318, `filtered_events`(sum) 0.327/0.606/0.281/0.258/0.327, `dff`(mean) 0.408/0.655/0.306/0.280/0.320 — dF/F wins on 4 of 5 outputs. It also removed 2,859 all-zero-trial warnings that arose in very sparse Vip/Sst planes with 7–29 cells. The agent argues dF/F "is not a different curation of the data — it is the same traces before L0 deconvolution", is described in the whitepaper ("DF/F CALCULATION"), and the paper's choice remains reproducible via a flag.

## 2-b. How is the `neural` data processed?

i. For each plane, the ΔF/F of every valid ROI is **averaged over the ophys frames falling inside each 250 ms bin** (sum of samples divided by the sample count per bin), producing `(ncells, ntrials, 20)`. Binning is vectorised with a cumulative sum + `searchsorted`, in chunks of 256 neurons. Planes of a session are then concatenated along the neuron axis, stored as `float32`, and each neuron is tagged with its plane's `targeted_structure` (VISp / VISl) for `brain_region_idx`. No z-scoring, smoothing, baseline subtraction or trial-averaging is applied. (For the `events` signal the bin **sum** is used instead of the mean.)

ii.
```python
def bin_sum(values_2d, sample_times, edges, chunk=256):
    idx = np.searchsorted(sample_times, edges.ravel(), side='left')
    out = np.empty((nrows, edges.shape[0], NBINS), dtype=np.float64)
    for a in range(0, nrows, chunk):
        b = min(a + chunk, nrows)
        c = np.concatenate(
            [np.zeros((b - a, 1), dtype=np.float64),
             np.cumsum(values_2d[a:b].astype(np.float64), axis=1)], axis=1)
        v = c[:, idx].reshape(b - a, edges.shape[0], NBINS + 1)
        out[a:b] = np.diff(v, axis=2)
    return out
```
```python
        binned = bin_sum(E, ots, edges)                            # (ncells, ntrials, NBINS)
        if signal == 'dff':
            # dF/F is a *rate-like* quantity: the correct bin statistic is the mean,
            # otherwise bins that happen to contain 7 vs 8 ophys frames (31 Hz) or
            # 2 vs 3 frames (11 Hz) would differ by a spurious scale factor.
            counts = bin_sum(np.ones((1, E.shape[1]), dtype=np.float32), ots, edges)
            binned = binned / np.maximum(counts, 1.0)
        neural_planes.append(binned)
        region_per_neuron += [ds.metadata['targeted_structure']] * E.shape[0]
```

iii. Key Decision 2: "dF/F is rate-like and the number of ophys frames per 250 ms bin varies (7–8 at 31 Hz, 2–3 at 11 Hz), so summing would impose a spurious rig-dependent scale. (For the event signals the bin **sum** is used, which is the correct analogue of a spike count.)" The cumulative-sum implementation was chosen so that "a bin with no samples yields exactly 0" and out-of-range edges are handled gracefully. Step 10 Check 2 recomputed the bin means directly from `dff_traces` + `ophys_timestamps` for 25 random (trial, neuron, bin) samples in 5 sessions: PASS at `np.allclose` 1e-5.

*Note for the reader*: the saved `metadata['neural_signal']` string is stale — it reads "sum of L0-detected calcium event magnitudes per 250 ms bin (dataset.events[\"dff\"])" and `task_description` says "Decoded from 2-photon calcium events", although the delivered array is the **mean dF/F**. The numerical processing is what the notes describe; only these two metadata strings were not updated.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No filtering is applied beyond what the AllenSDK already does. The agent verified in `cell_specimens.py` that `CellSpecimens(..., exclude_invalid_rois=True)` is the default, so `dff_traces`/`events` contain only ROIs that passed the pipeline ROI classifier (union / duplicate / motion-border / dendrite / too small / too dim), and that session-level QC (z-drift < 10 µm, peak d' ≥ 1.0, sync verified) was applied before release. All remaining neurons of every kept session enter the dataset (29,168 neurons, min 6 / max 666 per session).

ii. No code — absence of a filter. The neuron count is only asserted to be consistent:
```python
    nneurons_total = sum(len(b) for b in data['brain_region_idx'])
    print(f"\nSANITY: {len(data['neural'])} sessions, {len(subjects)} mice, "
          f'{nneurons_total} neurons, {ntrials_total} trials', flush=True)
```

iii. Notes Step 3 "Curation Steps — Neuron curation rules: only valid ROIs (already enforced by the SDK). No further per-cell filtering is described in the paper." Step 10 Check 3 row (b): "we take the tables as delivered, i.e. exactly the SDK's valid ROIs; no extra filtering (the paper adds none)". A sanity check confirmed "n neurons == sum over planes of len(dff_traces)".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial is aligned to `trials.change_time` — the onset of the changed image on go trials and of the sham change on catch trials. Bin edges are `change_time + [-2.0, -1.75, …, +3.0]`, so bin 8 is exactly [0, 250) ms after the change. Each plane's samples are assigned to bins using that plane's own `ophys_timestamps`, so all streams share one common clock (the SDK's sync clock). `metadata['temporal_alignment_event']` records this, with `off_start = -2.0`, `off_end = 3.0`.

ii.
```python
    offsets = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    return change_times[:, None] + offsets[None, :]
```
```python
        temporal_alignment_event=('stimulus change time (trials.change_time): the onset of the '
                                  'changed image on go trials, of the sham change on catch trials'),
        off_start=OFF_START,
        off_end=OFF_END,
```

iii. Step 2/Step 4: the agent empirically verified that `change_time` is **exactly** the `start_time` of the `is_change` flash (go) or the sham-change flash (catch) — "verified difference = 0.0 s", later re-checked at `atol 1e-9`. That makes it the natural, stimulus-locked alignment event and it exists for both trial types. An independent post-hoc check on the pickle showed "z-scored population activity averaged over change trials peaks in **bin 8** (0–0.25 s after the change) and is flat on catch trials", confirming there is no temporal shift.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — the native ophys sampling (31 Hz single-plane, ~11 Hz per Multiscope plane) is **rebinned to a uniform 250 ms**, giving 20 bins per trial for every trial in every session. `metadata['time_bin_size'] = 250.0` ms. Behavioural streams (60 Hz running, 30 Hz eye tracking) are rebinned onto the same edges.

ii.
```python
BIN_SIZE = 0.25          # s, = image duration, 1/3 of the 750 ms flash cycle
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 20
...
        time_bin_size=BIN_SIZE * 1000.0,
```

iii. Step 5: "**Time bin** = 250 ms = the image duration and exactly 1/3 of the 750 ms flash cycle → 20 bins per trial, bin edges phase-locked to the flash cycle. 250 ms also guarantees ≥ 2 ophys frames per bin even at the 11 Hz Multiscope rate (no empty bins), while 31 Hz sessions get ~7.75 frames/bin." Rebinning is *required* by the agent's session selection (mixing 31 Hz and 11 Hz rigs) and by the format requirement that "Time bins should be the same size for all trials and sessions". Step 10 Check 3 row (d) notes this is "compatible, finer" than the paper's 400 ms / 750 ms analysis windows.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `dataset.stimulus_presentations`, restricted to the `change_detection` stimulus block: the `start_time` and `image_name` columns (omitted flashes, whose `image_name` is `'omitted'`, are replaced by NaN and forward-filled so they inherit the preceding image). The trials table's `initial_image_name`/`change_image_name` are *not* used.

ii.
```python
def flash_table(dataset):
    """Change-detection flashes with a forward-filled image name."""
    sp = dataset.stimulus_presentations
    sp = sp[sp.stimulus_block_name.str.contains('change_detection')]
    sp = sp.sort_values('start_time')
    names = sp.image_name.astype(str).replace('omitted', np.nan).ffill().bfill().values
    return (sp.start_time.values.astype(float), names,
            sp.is_change.values.astype(bool), sp)
```

iii. Step 5 mapping table: source = `stimulus_presentations.image_name`, transform = "identity of the image whose 750 ms presentation interval contains the bin; omitted intervals inherit the preceding image". The 750 ms "image presentation interval" is the paper's own unit of behavioural analysis ("By image presentation interval we refer to the 750 ms interval beginning with each image presentation"). Step 4: "For the image-identity output the omitted 750 ms interval keeps the identity of the image that should have been shown, exactly as the paper treats omissions."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For every bin, the bin **centre** time is used to look up the last flash that started at or before it (`searchsorted(..., 'right') - 1`), and that flash's (forward-filled) image name is assigned. Names are then mapped to integer codes via a single global, sorted vocabulary built from all sessions — 16 images (sets A and B pooled); each session exercises only its own 8. `output_values[0]` stores the names.

ii.
```python
    centres = edges[:, :-1] + BIN_SIZE / 2.0                       # (ntrials, NBINS)
    fidx = np.searchsorted(starts, centres.ravel(), side='right') - 1
    fidx = np.clip(fidx, 0, len(starts) - 1)
    image_name = names[fidx].reshape(ntrials, NBINS)
```
```python
    images = sorted(set(np.concatenate([r['image_name'].ravel() for r in results])))
    img_index = {n: i for i, n in enumerate(images)}
    ...
        img_idx = np.vectorize(img_index.get)(r['image_name']).astype(np.int64)
```

iii. Step 10 Check 5: "the 16 images of sets A and B are pooled into one global vocabulary so that class indices mean the same thing in every session". Using the bin centre means every bin of a 750 ms interval (image + grey) carries that interval's image, which the agent justifies with the paper's presentation-interval convention. The delivered distribution is near-uniform (each image 5.9–6.5 % of bins). Sanity check: "OUTPUT image_identity: matches the forward-filled `stimulus_presentations.image_name` at the bin centre — PASS".

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Time-varying, one value per 250 ms bin, computed from exactly the same `edges` array that bins the neural data, so it is aligned by construction. Because the bin edges are phase-locked to `change_time` and `change_time` equals the change flash onset, the image identity switches exactly at bin 8 on change trials.

ii.
```python
    centres = edges[:, :-1] + BIN_SIZE / 2.0
    fidx = np.searchsorted(starts, centres.ravel(), side='right') - 1
    image_name = names[fidx].reshape(ntrials, NBINS)
```
```python
            out = np.stack([img_idx[t], r['image_change'][t], run_q[i][t], pup_q[i][t],
                            np.full(T, r['outcome'][t], dtype=np.int64)], axis=0)
```

iii. Step 7 "Independent alignment check (on the pickle)": "image identity switches exactly at bin 8 on 100 % of change trials and is constant within each 750 ms interval." The `--show-processing` figure overlays the flash raster (grey = image, red = change) with the image-identity trace to make the alignment visually verifiable.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `stimulus_presentations.is_change` (evaluated at the same per-bin flash index `fidx` used for image identity), plus the bin/flash start times. `is_change` is the SDK's `is_change_event`: the first presentation of a *new* image name, excluding omitted flashes and the first flash of the session. It is False for catch (sham) trials, so catch trials get all zeros.

ii.
```python
    starts, names, is_change, sp = flash_table(ds)
    ...
        change_flag = np.zeros((ntrials, NBINS), dtype=np.int64)
        flash_start = starts[fidx].reshape(ntrials, NBINS)
        first_bin = (centres - flash_start) < BIN_SIZE
        change_flag[is_change[fidx].reshape(ntrials, NBINS) & first_bin] = 1
```

iii. Step 1 documents `is_change_event()` in `stimulus_processing.py` as the reference implementation of "image change". Step 5 maps `stimulus_presentations.is_change` / `trials.change_time` → `output[1]`. Catch trials correctly carry no change because a sham change does not alter the image.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary `(ntrials, 20)` array. For each bin, the flash containing the bin centre is found; the bin is set to 1 only if that flash `is_change` **and** the bin is the *first* 250 ms bin of that flash's presentation interval. Because bin edges are phase-locked to `change_time`, this is exactly bin 8 (t ∈ [0, 250) ms) of every go trial. The alternative (`--change-window interval`, flagging all 3 bins of the 750 ms interval) is implemented but not used.

ii.
```python
    if change_window == 'interval':
        # 1 for every bin of the 750 ms presentation interval that starts with a change
        change_flag = is_change[fidx].reshape(ntrials, NBINS).astype(np.int64)
    else:
        # 1 only for the single 250 ms bin that starts at the change
        change_flag = np.zeros((ntrials, NBINS), dtype=np.int64)
        flash_start = starts[fidx].reshape(ntrials, NBINS)
        first_bin = (centres - flash_start) < BIN_SIZE
        change_flag[is_change[fidx].reshape(ntrials, NBINS) & first_bin] = 1
```

iii. Notes Step 8c: "whole 750 ms presentation interval (3 bins) 0.655 vs **single 250 ms bin at the change 0.689**. The single bin is adopted — it is also the literal reading of the task ('value of 1 right after a change')." Issue 4 in Step 12 records the change from the interval encoding to the single-bin encoding.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary — no thresholding of a continuous quantity is involved. `output_values[1] = ['no_change', 'change']`. The resulting marginal is 0.956 / 0.044 (1 of 20 bins on the 87.5 % of trials that are go).

ii.
```python
                output_values=[images, ['no_change', 'change'],
                               [f'q{i+1}' for i in range(NQUANTILES)],
                               [f'q{i+1}' for i in range(NQUANTILES)],
                               OUTCOMES])
```

iii. The task specifies a binary variable; the only free choice was the *width* of the "1" window, handled in 4-b. The imbalance is acceptable because `train_decoder.py` uses a balanced loss and reports balanced accuracy (the agent checked `decoder.py` for this in Step 6).

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Time-varying on the same 250 ms bin grid as the neural data, derived from the same `edges`/`centres` arrays; the 1 falls in the bin that starts at the alignment event itself.

ii.
```python
    centres = edges[:, :-1] + BIN_SIZE / 2.0
    fidx = np.searchsorted(starts, centres.ravel(), side='right') - 1
```
plus the `change_flag` assignment above and the shared `np.stack` into `output`.

iii. Verified by an in-script sanity check that runs on every conversion: on go trials the change pattern must equal the expected one-hot pattern at bin 8, and catch trials must never have a change flag — "SANITY: go trials with unexpected change pattern: 0; catch trials with a change flag: 0". Independently, "OUTPUT image_change: matches `stimulus_presentations.is_change` at the bin centre — PASS".

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `dataset.running_speed`, i.e. the `speed` (cm/s) and `timestamps` columns of the SDK's 60 Hz running table (already 10 Hz low-pass filtered with transients removed by the Allen pipeline).

ii.
```python
    rs = ds.running_speed
    speed = np.asarray(rs['speed'].values, dtype=float)
    speed_t = np.asarray(rs['timestamps'].values, dtype=float)
```

iii. Step 1/Step 2 identify `dataset.running_speed` as the SDK's standard locomotion stream ("60 Hz stimulus frame clock; low-pass 10 Hz Butterworth filtered, transients removed (see whitepaper)"), with "no NaNs" observed in the data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Any NaNs are linearly interpolated (a no-op in practice), then the speed is **averaged over the 60 Hz samples inside each 250 ms bin** (≈15 samples/bin) using the same cumulative-sum binning as the neural data. Bins containing no sample fall back to linear interpolation at the bin centre (does not occur at 60 Hz). The resulting `(ntrials, 20)` bin means are then discretised (5-c).

ii.
```python
def bin_mean_1d(values, sample_times, edges):
    v = values[None, :]
    ones = np.ones_like(v)
    s = bin_sum(v, sample_times, edges)[0]
    n = bin_sum(ones, sample_times, edges)[0]
    with np.errstate(invalid='ignore', divide='ignore'):
        m = s / n
    if np.any(n == 0):
        centres = edges[:, :-1] + BIN_SIZE / 2.0
        fill = np.interp(centres, sample_times, values)
        m = np.where(n == 0, fill, m)
    return m
```
```python
    speed = interpolate_nans(speed)
    running_binned = bin_mean_1d(speed, speed_t, edges)            # (ntrials, NBINS)
```

iii. Averaging (rather than sampling/interpolating at a point) is the natural bin statistic for a continuous behavioural signal sampled 15× faster than the bin, and it uses exactly the same binning machinery as the neural data so the two are guaranteed to share edges. The empty-bin fallback is documented in Step 10 Check 5 ("Empty bins ... at 250 ms with ≥ 11 Hz imaging and 30/60 Hz behaviour no empty bins occur in practice").

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five **equal-percentile (quintile) bins computed separately for each session** (`--quantile-scope session`, the default): for each session the 0/20/40/60/80/100th percentiles of all its bin means are taken and `np.digitize` assigns classes 0–4. Every session therefore contributes exactly 20 % of its bins to each class. A global-scope alternative exists (`--quantile-scope global`) and the global edges are still computed and stored in metadata (−19.7, 0.017, 2.10, 18.1, 32.9, 99.5 cm/s).

ii.
```python
def discretize(values_list, nq=NQUANTILES, scope='session'):
    allv = np.concatenate([v.ravel() for v in values_list])
    global_qs = np.percentile(allv, np.linspace(0, 100, nq + 1))
    if scope == 'global':
        out = [np.digitize(v, global_qs[1:-1], right=False).astype(np.int64)
               for v in values_list]
    else:
        out = []
        for v in values_list:
            qs = np.percentile(v.ravel(), np.linspace(0, 100, nq + 1))
            out.append(np.digitize(v, qs[1:-1], right=False).astype(np.int64))
    return out, global_qs
```
```python
    run_q, run_edges = discretize([r['running'] for r in results], scope=args.quantile_scope)
```

iii. Key Decision 7: "*revised from global; see Step 8a*: the pupil diameter is in camera pixels (rig/session dependent) and each mouse has its own running baseline, so global cuts partly encode session identity; per-session equal-percentile bins give exactly 20 % per class in every session and decode better." The measured effect on the 2-session sample was running 0.262 (per-session) vs 0.244 (global). The docstring of `discretize` repeats the argument. Delivered marginal: exactly 0.200 per class.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is binned onto the identical `edges` array used for the neural data (change-aligned, 250 ms), so alignment is exact by construction; no separate resampling onto the ophys timestamps is needed.

ii.
```python
    running_binned = bin_mean_1d(speed, speed_t, edges)
```
```python
            out = np.stack([img_idx[t], r['image_change'][t], run_q[i][t], pup_q[i][t],
                            np.full(T, r['outcome'][t], dtype=np.int64)], axis=0)
```

iii. All streams are placed on the SDK sync clock, which the agent confirmed is common to ophys, stimulus, running and eye-tracking timestamps; binning every stream with the same `edges` therefore removes any interpolation-induced shift. Sanity check: "OUTPUT running/pupil quintiles: recomputed from `running_speed` / `eye_tracking` — PASS (exact)". The `--show-processing` figure overlays the raw 60 Hz trace, the bin means and the quintile codes for visual confirmation.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `dataset.eye_tracking`: the `pupil_width`, `pupil_height` and `timestamps` columns of the 30 Hz eye-tracking table. Pupil diameter is defined as `2 * max(pupil_width, pupil_height)`. Blink frames are already NaN in the SDK table (`filter_on_blinks` sets them so) and are linearly interpolated rather than dropped.

ii.
```python
    et = ds.eye_tracking
    pupil_binned = None
    if len(et) > 0:
        diam = 2.0 * np.nanmax(
            np.vstack([et['pupil_width'].values, et['pupil_height'].values]), axis=0)
        diam = interpolate_nans(diam)
```
```python
def interpolate_nans(values):
    """Linearly interpolate NaNs (blinks); edges are filled with nearest value."""
    v = np.asarray(values, dtype=float).copy()
    good = np.isfinite(v)
    if not np.any(good):
        return None
    if not np.all(good):
        x = np.arange(len(v))
        v[~good] = np.interp(x[~good], x[good], v[good])
    return v
```

iii. Step 4 discrepancy table: "`pupil_width/height` are ellipse half-axes; `pupil_area = pi*max(w,h)^2` … whitepaper: 'major axis reflects the pupil diameter' → Pupil diameter = 2*max(pupil_width, pupil_height); blinks interpolated." The agent read `compute_circular_area` in the SDK to establish that the SDK itself treats `max(w, h)` as the radius, which makes `2*max(w,h)` the diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. After blink interpolation, the diameter trace is averaged over the 30 Hz samples inside each 250 ms bin (≈7.5 samples/bin) with the same `bin_mean_1d` routine as running speed, then discretised into per-session quintiles. If the eye-tracking table is empty or entirely NaN, `interpolate_nans` returns `None` and the whole session is skipped.

ii.
```python
        if diam is not None:
            pupil_binned = bin_mean_1d(diam, np.asarray(et['timestamps'].values, float),
                                       edges)
    if pupil_binned is None:
        return dict(skip=True, ophys_session_id=int(ophys_session_id),
                    reason='no eye tracking')
```
```python
    pup_q, pup_edges = discretize([r['pupil'] for r in results], scope=args.quantile_scope)
```

iii. Same rationale as running speed: bin-mean averaging matches the neural binning, and interpolating the ~2.9 % blink frames (rather than dropping them) keeps the output NaN-free, which `verify_data_format` requires ("no NaN/Inf allowed anywhere"). The three sessions with no eye tracking at all are "dropped rather than filled" (Step 10 Check 5).

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: five equal-percentile bins computed **per session** (default), with the global edges (19.6, 74.8, 84.4, 93.6, 106.6, 362.1 px) also computed and stored in metadata. Delivered marginal: exactly 0.200 per class.

ii.
```python
    pup_q, pup_edges = discretize([r['pupil'] for r in results], scope=args.quantile_scope)
    ...
        pupil_diameter_quintile_edges_px=pup_edges.tolist(),
```
(plus the `discretize` body quoted in 5-c)

iii. Key Decision 7 and the `discretize` docstring: "both measurements are only comparable *within* a session: the pupil diameter is in camera pixels (the eye-camera zoom and distance differ between rigs and sessions)". Measured on the sample: pupil 0.271 per-session vs 0.260 global. Issue 3 in Step 12: "Global quintiles mixed session identity into the running/pupil labels → switched to per-session percentile bins (also required by 'five equal percentile bins')."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Binned onto the same change-aligned 250 ms `edges` as the neural data; stacked into the same `(5, 20)` output matrix per trial.

ii.
```python
        pupil_binned = bin_mean_1d(diam, np.asarray(et['timestamps'].values, float), edges)
```

iii. Same justification as 5-d — one shared bin-edge array for all streams on the SDK sync clock. The `--show-processing` pupil panel plots the raw trace with its blink NaNs, the interpolated trace, the bin means and the quintile codes together.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually-exclusive boolean columns of `dataset.trials`: `hit`, `miss`, `false_alarm`, `correct_reject`, in that fixed order.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
    outcome = np.full(ntrials, -1, dtype=np.int64)
    for i, name in enumerate(OUTCOMES):
        outcome[keep[name].values.astype(bool)] = i
    assert np.all(outcome >= 0), 'trial with no hit/miss/FA/CR outcome'
```

iii. Step 1/Step 5: these are the SDK's canonical outcome labels and, once aborted and auto-rewarded trials are removed, "exactly the outcomes that exist for go and catch trials" (Key Decision 9). The assertion enforces that every kept trial has exactly one of the four. The delivered distribution (hit 0.317 / miss 0.558 / FA 0.019 / CR 0.106) matches the raw-data scan and implies a 36.2 % hit rate on go trials and 15.0 % FA rate on catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Mapped to integer codes 0–3 and, although the task calls it "static per-trial", it is **broadcast across all 20 time bins** so that every output row is time-varying and the output matrix is a uniform `(5, 20)`. `output_values[4] = OUTCOMES` records the names.

ii.
```python
            out = np.stack([img_idx[t], r['image_change'][t], run_q[i][t], pup_q[i][t],
                            np.full(T, r['outcome'][t], dtype=np.int64)], axis=0)
            output_trials.append(out.astype(np.int64))
```

iii. The target-format spec says outputs "Can be time-varying or discrete values per trial. If at all possible, make it time-varying", and `verify_data_format` requires consistent output dimensions across trials; broadcasting the static label satisfies both. Sanity check: "OUTPUT trial_outcome: matches the `trials` table; constant within a trial — PASS", and "hit+miss == n go trials; FA+CR == n catch trials — PASS".

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled explicitly:
- **Blink NaNs in the pupil trace** (median 2.9 % of frames): linearly interpolated, edges filled with the nearest valid value.
- **Sessions with no eye tracking at all** (3 sessions): `interpolate_nans` returns `None` → the session is skipped with a printed reason.
- **NaNs in running speed**: same interpolation path (none occur in practice).
- **Bins with no samples**: `bin_sum`'s cumulative-sum formulation returns exactly 0; for behavioural streams `bin_mean_1d` falls back to linear interpolation at the bin centre.
- **Bin edges outside the sampled interval**: `searchsorted` clamps to the array ends, so a window overhanging the recording degrades gracefully instead of erroring.
- **Omitted stimulus flashes**: image name forward-filled (and back-filled for a leading omission) so the identity is never `'omitted'`.
- **Sessions with < 2 valid trials**: dropped (none occurred).
- **Degenerate outcomes**: passive sessions excluded up front.
- **Assertion** that every kept trial has exactly one of the four outcomes.
There is **no** `try/except` around `process_session`, so a session that raised an unexpected exception would abort the whole `Pool.map` run (it did not happen on this dataset).

ii.
```python
def interpolate_nans(values):
    v = np.asarray(values, dtype=float).copy()
    good = np.isfinite(v)
    if not np.any(good):
        return None
    if not np.all(good):
        x = np.arange(len(v))
        v[~good] = np.interp(x[~good], x[good], v[good])
    return v
```
```python
    if np.any(n == 0):
        centres = edges[:, :-1] + BIN_SIZE / 2.0
        fill = np.interp(centres, sample_times, values)
        m = np.where(n == 0, fill, m)
```
```python
    if pupil_binned is None:
        return dict(skip=True, ophys_session_id=int(ophys_session_id),
                    reason='no eye tracking')
...
    for r in skipped:
        if r is not None:
            print(f"  SKIPPED session {r['ophys_session_id']}: {r['reason']}", flush=True)
```
```python
    names = sp.image_name.astype(str).replace('omitted', np.nan).ffill().bfill().values
```
```python
    assert np.all(outcome >= 0), 'trial with no hit/miss/FA/CR outcome'
```

iii. Step 10 Check 5 ("Check for edge cases") enumerates each of these: trial-window overrun (measured to be impossible given the 3.02 s / 4.23 s margins and 100 % ophys coverage), empty bins, blinks, omitted flashes, the first flash of a session, and sessions with < 2 trials. The design goal was that `verify_data_format` — which "allows no NaN/Inf anywhere" — passes with "no errors or warnings", which the full run achieves.

## 9-a. What are the most time-consuming steps of the code?

i. Reading the NWB files through `bc.get_behavior_ophys_experiment()` dominates completely: the per-session timing dictionaries printed during the run show `load` = 3.0–8.0 s for a single plane and up to 21 s for a 7-plane Multiscope session, versus `neural` (binning) 0.02–0.84 s and `stimulus` + `behaviour` ≈ 0.012 s. The agent instrumented this with a `timings` dict per session and mitigated it with a 16-process pool, bringing the full 174-session conversion to 77.5 s of session processing (81–87 s total).

ii.
```python
    for k, eid in enumerate(experiment_ids):
        t0 = time.time()
        ds = bc.get_behavior_ophys_experiment(int(eid))
        timings['load'] = timings.get('load', 0) + time.time() - t0
    ...
    if verbose:
        print(f"  session {ophys_session_id}: {neural.shape[0]} neurons, "
              f"{ntrials} trials, {len(experiment_ids)} plane(s), "
              f"{result['elapsed']:.1f}s {timings}", flush=True)
```
```python
    if args.workers > 1 and len(jobs) > 1:
        with Pool(min(args.workers, len(jobs))) as p:
            results = p.map(process_session, jobs)
    t_sessions = time.time() - t0
    print(f'session processing took {t_sessions:.1f}s '
          f'({t_sessions / max(len(jobs), 1):.1f}s per session)', flush=True)
```

iii. Step 6: "Code inefficiencies identified: the NWB file load (3-4 s/plane) dominates; binning and behaviour processing take <0.2 s/session. Code speedups added: multiprocessing over sessions (16 workers), cumulative-sum binning instead of per-trial loops, a single pass over each plane, and float32 storage." Step 7 estimates "202 planes → ~800 s serial" against a "measured full run with 16 workers: 83 s", i.e. a ~10× wall-clock saving, comfortably inside the 15-minute budget.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The hot paths are already vectorised (all trials × all bins are binned in one `searchsorted` + `cumsum` per plane; no per-trial Python loop touches the neural data). The remaining Python loops are in the cheap assembly phase and could still be vectorised:
- the `for t in range(ntrials)` loop in `main()` that builds one `np.stack` per trial (≈44,000 iterations) — the whole `(5, ntrials, 20)` output could be built once per session and then sliced;
- `np.vectorize(img_index.get)` for the image→code mapping, which is a disguised Python loop over `ntrials × 20` strings (a `pd.Series.map`, or codes carried through from `flash_table`, would be array-level);
- `[regions.index(x) for x in r['regions']]`, an O(n·k) list scan per neuron;
- the `for a in range(0, nrows, chunk)` chunk loop in `bin_sum` (deliberate, to bound memory);
- the double loop in the final `bad_go`/`bad_catch` sanity check, which iterates over every trial of every session in Python.
None of these matter next to the 3–20 s NWB load, and the `--sample`/full timings confirm binning + outputs stay under 0.2 s/session.

ii.
```python
        for t in range(ntrials):
            neural_trials.append(np.ascontiguousarray(r['neural'][:, t, :], dtype=np.float32))
            input_trials.append(np.zeros((0, T), dtype=np.float32))
            out = np.stack([img_idx[t], r['image_change'][t], run_q[i][t], pup_q[i][t],
                            np.full(T, r['outcome'][t], dtype=np.int64)], axis=0)
            output_trials.append(out.astype(np.int64))
```
```python
        img_idx = np.vectorize(img_index.get)(r['image_name']).astype(np.int64)
        ...
        data['brain_region_idx'].append(
            np.array([regions.index(x) for x in r['regions']], dtype=np.int64))
```
```python
    for i, r in enumerate(results):
        for t in range(r['image_change'].shape[0]):
            if r['go'][t] and not np.array_equal(r['image_change'][t], expect):
                bad_go += 1
```

iii. The agent's stated principle (Step 6) was to vectorise the parts that scale with neurons × trials × bins — "cumulative-sum vectorised binning" replaced per-trial slicing — and to parallelise the I/O-bound part, leaving the per-trial packing loop because the target format itself is a list-of-lists of per-trial arrays, so some per-trial Python work is unavoidable.

## 9-c. What processing does the code repeat multiple times?

i. A few genuine repetitions, all cheap:
- `bin_sum` is called a second time on an array of ones, per plane, purely to obtain the per-bin sample counts for the dF/F mean; the count depends only on the plane's timestamps, so it is recomputed for each plane (necessary across planes, but it duplicates work with the trace binning itself). The same pattern is repeated inside `bin_mean_1d`, once for running and once for pupil.
- `discretize` always computes the global percentiles (`global_qs`) even when `scope='session'`, where they are used only for the metadata printout.
- `make_plots` re-instantiates the cache and **re-loads** the first experiment of the session from disk (and additionally loads `ds.events`, `ds.licks`, `ds.rewards`), duplicating a 3–8 s load that was already done in the worker.
- `flash_table` is re-run inside `make_plots` for the same session.
- Across the *project* (not within one run), the agent re-ran the full conversion several times to compare `events` / `filtered_events` / `dff`, the change-window encoding, the quantile scope and the trial window — deliberate ablations, each logged.
Within a single conversion run each session is loaded exactly once, and the per-plane load is a single pass that feeds neural, stimulus and behaviour extraction.

ii.
```python
            counts = bin_sum(np.ones((1, E.shape[1]), dtype=np.float32), ots, edges)
            binned = binned / np.maximum(counts, 1.0)
```
```python
    s = bin_sum(v, sample_times, edges)[0]
    n = bin_sum(ones, sample_times, edges)[0]
```
```python
    allv = np.concatenate([v.ravel() for v in values_list])
    global_qs = np.percentile(allv, np.linspace(0, 100, nq + 1))
    if scope == 'global':
        ...
```
```python
def make_plots(res, images, run_q, pupil_q, outpath):
    bc = get_cache()
    ds = bc.get_behavior_ophys_experiment(res['experiment_ids'][0])
    starts, names, is_change, sp = flash_table(ds)
```

iii. Step 6 claims "a single pass over each plane"; the counts recomputation is the price of reusing one generic `bin_sum` helper for sums, means and sample counts, and it is O(nsamples) for a single row. The `make_plots` reload is confined to the optional `--show-processing` path (2 sessions) and was accepted because the plotting code is meant to be an *independent* re-derivation from the raw SDK objects, which is exactly what makes the figures a meaningful check.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts of work whose results are never consumed by the decoder:
- `flash_table` returns the full `sp` DataFrame as a fourth value that `process_session` never uses.
- `select_sessions` returns the experiment table `et`, which `main()` binds but never reads.
- `discretize` computes `global_qs` even in the default per-session mode (used only for a printed line and a metadata field).
- Running speed and pupil bin means are carried at `float64` and stored in the per-session result dicts even though only their integer quintile codes reach the output; likewise `change_times`, `go`, `catch`, `frame_rate`, `nplanes` and the timing dicts are kept per session (they feed the sanity checks and `session_info` metadata, not the decoder).
- An empty `np.zeros((0, T), dtype=np.float32)` input array is allocated for **every** trial (~44,000 allocations) although `input_names` is empty and the decoder receives zero input features.
- In `--show-processing` mode, `make_plots` loads `ds.events` (the L0 event traces) for the raw-trace panel even though the delivered neural signal is dF/F, so panel 1 is labelled "raw calcium events" while the binned panel below it shows dF/F — that panel is decorative and inconsistent with the delivered data.
- The end-of-run `bad_go`/`bad_catch` verification loop over all 43,975 trials is pure checking, not conversion.
- Across the project, the ablation conversions (events / filtered_events / alternative windows / interval change-encoding) produced pickles that were discarded after the comparison.

ii.
```python
    return (sp.start_time.values.astype(float), names,
            sp.is_change.values.astype(bool), sp)
...
    starts, names, is_change, sp = flash_table(ds)     # `sp` unused
```
```python
    groups, et = select_sessions(sample=args.sample)   # `et` unused afterwards
```
```python
    allv = np.concatenate([v.ravel() for v in values_list])
    global_qs = np.percentile(allv, np.linspace(0, 100, nq + 1))
```
```python
            input_trials.append(np.zeros((0, T), dtype=np.float32))
```
```python
        running=running_binned.astype(np.float64),
        pupil=pupil_binned.astype(np.float64),
```
```python
    ev = ds.events
    E = np.vstack(ev['events'].values).astype(np.float32)   # in make_plots only
```

iii. The agent's efficiency notes (Step 6) focus on the dominant cost (NWB I/O) and explicitly accept the remaining overhead as negligible; the retained per-session fields are justified because they back the in-script sanity checks and the `metadata['session_info']` table, which the agent added so that "any subset can be re-derived" (e.g. restricting post-hoc to the paper's multi-plane, familiar-image subset). The empty per-trial input array is required by the target format, which expects an `input` entry per trial even when `d_input = 0`; the agent verified in Step 6 that `d_input = 0` passes verification and training end-to-end.
