# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI works directly from the locally-downloaded release rather than the S3 cache object.
`get_session_table()` reads the project metadata table `project_metadata/ophys_experiment_table.csv`,
intersects it with the set of `behavior_ophys_experiment_<id>.nwb` files actually present in
`behavior_ophys_experiments/` (284 of the 1936 released experiments are on disk), drops passive
experiments, and groups the remaining rows by `ophys_session_id`, so that a "decoder session" is one
recording session and a Multiscope session's several imaging planes are loaded together. Each plane is
then loaded with `BehaviorOphysExperiment.from_nwb_path()` — the same call that the SDK's
`VisualBehaviorOphysProjectCache.get_behavior_ophys_experiment()` makes internally after it resolves a
file id to a local path (the AI documents tracing this through
`BehaviorProjectCloudApi.get_behavior_ophys_experiment`, CONVERSION_NOTES Step 1). Sessions are
processed independently in a `multiprocessing.Pool` of 24 workers; the full run loaded 202 planes /
174 sessions in ~50 s. No `project_code` filter is applied, so both `VisualBehavior` (single-plane,
31 Hz) and `VisualBehaviorMultiscope` (up to 7 planes, 11 Hz) sessions are included.

ii.
```python
DATA_DIR = '/app/data/visual-behavior-ophys-1.1.0/'
NWB_FMT = DATA_DIR + 'behavior_ophys_experiments/behavior_ophys_experiment_%d.nwb'
META_DIR = DATA_DIR + 'project_metadata/'

def get_session_table(sample=False):
    """Active experiments that are actually present on disk, grouped by ophys_session_id."""
    et = pd.read_csv(META_DIR + 'ophys_experiment_table.csv')
    have = set(int(f.split('_')[-1].split('.')[0])
               for f in os.listdir(DATA_DIR + 'behavior_ophys_experiments'))
    et = et[et.ophys_experiment_id.isin(have)]
    et = et[~et.passive]                       # active behavior sessions only
    et = et.sort_values(['ophys_session_id', 'ophys_experiment_id'])
    sessions = [(int(sid), list(map(int, g.ophys_experiment_id.values)))
                for sid, g in et.groupby('ophys_session_id')]
```
```python
    for eid in eids:
        planes.append((eid, BehaviorOphysExperiment.from_nwb_path(NWB_FMT % eid)))
    ref = planes[0][1]          # behavior streams are identical across planes
```
```python
    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(process_session, jobs)
```

iii. From CONVERSION_NOTES Step 1: the AI traced the cache API to
`BehaviorOphysExperiment.from_nwb_path(str(data_path))` and states "I replicate this by calling
`from_nwb_path` directly on the local files (no S3 access needed)". Intersecting with the files on
disk is justified because "Only **284 of the 1936** experiments in the manifest are actually present
on disk; the conversion therefore works from the intersection of the experiment table with the files
on disk" — i.e. it avoids attempting to fetch absent experiments. Grouping by `ophys_session_id` is
justified from the whitepaper definition ("data collected in a single continuous recording is defined
as a session") plus a direct check that Multiscope planes in a session share identical trials tables
and timestamps within 0.07 s. Including Multiscope is a documented, deliberate difference from the
paper's own restriction: "The paper restriction served its question (cell classes vs strategy). For
training a decoder we keep all active sessions from both rigs and all experience levels: more data,
and the Decoder Task does not ask for the novelty/strategy contrast."

## 1-b. How are the data split into subjects (mice)?

i. A subject is a `mouse_id`. The AI takes `mouse_id` from each loaded experiment's `metadata` dict
(rather than from the experiment table), stores it as a string on each session result, and then builds
the global `subjects` list as the sorted set of mouse ids over the sessions that survived conversion.
`subject_idx` is the position of each session's mouse in that sorted list. The full conversion yields
38 mice.

ii.
```python
    md = ref.metadata
    ...
    result = dict(session_id=sid, experiment_ids=eids, mouse=str(md['mouse_id']), ...)
```
```python
    subjects = sorted(set(r['mouse'] for r in results))
    ...
    data['subject_idx'].append(subjects.index(r['mouse']))
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. `mouse_id` is the SDK's unique animal identifier; the AI cross-checked the count against the raw
experiment table ("38 mice, 247 sessions" available on disk; 38 mice in the converted data) and against
the paper's statement of 82 mice in the full release, concluding the difference is explained by the
release subset actually present on disk (CONVERSION_NOTES Steps 2 and 9 consistency table: "38 mice
with ophys files on disk | 38 | yes").

## 1-c. How are the data split into sessions?

i. One converted session = one `ophys_session_id`. All imaging planes (experiments) sharing that id
are loaded together and their neurons are concatenated along the neuron axis into a single population;
behaviour streams (trials, stimulus presentations, running, eye tracking) are read only from the first
plane because they were verified to be identical across planes. Sessions are enumerated in
`ophys_session_id` order (pandas `groupby` sorts). Three exclusions are applied at session level:
(1) passive sessions (`OPHYS_2_images_A_passive`, `OPHYS_5_images_B_passive`) are removed via the
`~et.passive` filter; (2) sessions whose eye-tracking table is empty or all-NaN are dropped; (3)
sessions with fewer than 2 usable trials are dropped. 174 active sessions → 171 converted sessions
(3 lost to missing eye tracking).

ii.
```python
    et = et[~et.passive]                       # active behavior sessions only
    sessions = [(int(sid), list(map(int, g.ophys_experiment_id.values)))
                for sid, g in et.groupby('ophys_session_id')]
```
```python
    for eid, ds in planes:
        ev = np.vstack(ds.events.events.values).astype(np.float64)   # (n_cells, n_frames)
        ts = ds.ophys_timestamps.astype(float)
        s, _ = binned_sum(ev, ts, flat_t0, flat_t1)
        neural_planes.append(s.reshape(ev.shape[0], n_trials, N_BINS))
        struct = ds.metadata['targeted_structure']
        region_idx += [struct] * ev.shape[0]
    neural = np.concatenate(neural_planes, axis=0).astype(np.float32)   # (n_cells, n_tr, T)
```
```python
    eye = ref.eye_tracking
    if len(eye) == 0:
        print(f'  session {sid}: DROPPED (no eye tracking data)', flush=True)
        return None
```

iii. Merging planes: CONVERSION_NOTES Key Decision 3 — "simultaneously recorded from one animal on one
clock, so they form one population"; verified empirically that "Multiscope sessions have 3-7 planes
with identical trials and timestamps agreeing within 0.07 s". Note each plane is still binned on its
*own* `ophys_timestamps`, so the sub-frame offsets between planes are respected.
Excluding passive sessions: Key Decision 4 — "0 licks / 0 rewards makes trial outcome degenerate". The
AI explicitly checked this rather than assuming: passive sessions *do* contain go/catch trials (e.g.
406 trials, 356 go / 50 catch) but 0 licks and 0 rewards, so "every outcome is miss/correct-reject by
construction and trial outcome carries no behavioral information".
Dropping eye-tracking-free sessions: Key Decision 8 — "Drop the 3 sessions lacking eye tracking rather
than emit NaN, which verify_data_format rejects."

## 1-d. How are the data split into trials?

i. Trials are the rows of the SDK `trials` table for which `go` or `catch` is True — i.e. the
change-detection trials, one per (real or sham) image change. The *window* extracted for each trial,
however, is **not** the table's `start_time`→`stop_time` span: it is a fixed 8-bin window locked to the
change flash, spanning the 3 image-presentation intervals before the change flash, the change flash
interval itself, and the 4 intervals after it (`N_PRE=3`, `N_POST=4`, `N_BINS=8`, 750 ms per bin =
−2.25 s to +3.75 s relative to `change_time`). Bin boundaries are the *actual* `start_time`s of
consecutive flashes in the `change_detection` stimulus block, so the bins follow the real, slightly
jittered (0.73–0.77 s) flash cadence rather than an idealised grid. Every trial therefore has exactly
T = 8 time bins, in every session and on both rigs.

ii.
```python
BIN_SIZE = 0.75          # s, one image presentation interval (250 ms image + 500 ms gray)
N_PRE = 3                # flashes before the change flash
N_POST = 4               # flashes after the change flash
N_BINS = N_PRE + 1 + N_POST   # = 8
```
```python
    sp = ref.stimulus_presentations
    sp = sp[sp.stimulus_block_name.str.contains('change_detection', na=False)]
    sp = sp.sort_values('start_time')
    flash_start = sp.start_time.values.astype(float)
    ...
    trials = ref.trials
    keep = (trials.go.values.astype(bool) | trials['catch'].values.astype(bool))
    trials = trials[keep]
    if len(trials) < 2:
        return None
```
```python
    change_times = trials.change_time.values.astype(float)
    j = np.searchsorted(flash_start, change_times - 1e-6, side='left')
    ...
    bin_flash = j[:, None] + np.arange(-N_PRE, N_POST + 1)[None, :]
    bt0 = flash_start[bin_flash]                 # bin start times
    bt1 = bt0 + BIN_SIZE                         # bin end times
```

iii. `go | catch` is the SDK's own exhaustive complement of aborted + auto-rewarded trials
(`trial.py`: if aborted then `go = catch = auto_rewarded = False`; otherwise `go = not catch and not
auto_rewarded`), so this implements the instruction "Include both the 'Go' and 'Catch' trials, but
exclude the 'Aborted' and 'Auto-rewarded' trials". The flash-locked fixed window is Key Decision 1:
"matches the paper image-presentation interval and guarantees identical T across trials and across both
rigs (31 Hz and 11 Hz), which a fixed grid in seconds could not do cleanly", and Step 5: "Bin edges
come from the actual start_time of consecutive flashes in the change_detection stimulus block, so bins
track the real (slightly jittered, 0.73-0.77 s) cadence rather than an idealized grid." The AI verified
globally that the window never leaves the experimentally-defined trial: "Verified over all 51,992
go+catch trials that this window lies inside [start_time, stop_time]", supported by the measured
`min(change_time − start_time) = 2.79 s` (> 2.25 s needed) and `min(stop_time − change_time) = 4.20 s`
(> 3.75 s needed).

## 1-e. How are trials filtered based on quality controls?

i. A trial is kept only if all of the following hold: (a) it is `go` or `catch` (excludes aborted and
auto-rewarded); (b) `change_time` is finite; (c) `change_time` coincides with a flash onset to within
1e-4 s (the change is a real stimulus-presentation onset); (d) the full ±window of flash indices lies
inside the `change_detection` stimulus block (`j - N_PRE >= 0` and `j + N_POST < len(flash_start)`);
(e) exactly one of `hit/miss/false_alarm/correct_reject` is True. Sessions are dropped if fewer than 2
trials survive. In the full run, criteria (b)–(e) dropped **0** trials from any session, so the
effective filter is exactly "go or catch". The kept-trial count is 43,975 over 171 sessions.

ii.
```python
    j = np.searchsorted(flash_start, change_times - 1e-6, side='left')
    good = np.ones(len(trials), dtype=bool)
    good &= np.isfinite(change_times)
    j_clipped = np.clip(j, 0, len(flash_start) - 1)
    good &= np.abs(flash_start[j_clipped] - change_times) < 1e-4   # change is a flash onset
    good &= (j - N_PRE >= 0) & (j + N_POST < len(flash_start))     # window inside the block

    # outcome must be one of the four mutually exclusive categories
    oc = np.full(len(trials), -1, dtype=np.int64)
    for k, name in enumerate(OUTCOMES):
        oc[trials[name].values.astype(bool)] = k
    good &= oc >= 0

    idx = np.nonzero(good)[0]
    if len(idx) < 2:
        return None
```

iii. CONVERSION_NOTES Step 3 "Trial curation rules": "Keep go and catch trials; exclude aborted and
auto_rewarded, per the Decoder Task and consistent with the whitepaper which excludes aborted trials
from performance metrics and states auto-rewarded trials should not be categorized as hit/miss." The
extra guards are defensive: the AI verified beforehand that `change_time` is never NaN on go/catch
trials and coincides with a flash onset to 0.000000 s in 100% of 51,992 candidate trials, and that the
8-flash window always fits; the checks are retained so that any violation would be caught rather than
silently mis-binned. The ≥2-trial requirement matches the target format's "There needs to be at least
two trials within each session". Step 7 records that the AI verified the one very low-trial session
(39 trials) against the raw trials table rather than assuming a bug: "that session really has 1,078
aborted trials out of 1,117 (an unusually impulsive mouse)".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `events` table of each `BehaviorOphysExperiment` — specifically the `events` column, which is
the **detected calcium event magnitude** per cell per 2-photon frame — together with that plane's
`ophys_timestamps`. dF/F (`dff_traces`) and the smoothed `filtered_events` are explicitly *not* used.
Only valid ROIs are included, which is the SDK default (`exclude_invalid_rois=True`).

ii.
```python
    for eid, ds in planes:
        ev = np.vstack(ds.events.events.values).astype(np.float64)   # (n_cells, n_frames)
        ts = ds.ophys_timestamps.astype(float)
```

iii. CONVERSION_NOTES Step 3 quotes the reference paper directly: "For all analysis of neural data we
used the detected calcium events". Step 4's discrepancy table resolves the dff/events/filtered_events
choice in favour of `events`, and Step 10 Check 3 records the deliberate difference from
`filtered_events`: "the paper says 'detected calcium events'; `filtered_events` applies an extra causal
half-normal smoothing that would blur activity across my 750 ms bin boundaries." Step 1 also notes the
NWB already contains pipeline dF/F and detected events, so "dF/F does not need to be computed".

## 2-b. How is the `neural` data processed?

i. For each plane, the per-cell event trace is stacked into an `(n_cells, n_frames)` array and the
event magnitudes are **summed** inside each 750 ms bin, using a cumulative sum plus `searchsorted`
lookups so each trace is traversed once regardless of how many bins are requested. A bin covers frames
whose timestamp lies in `[t0, t1)`. The result is reshaped to `(n_cells, n_trials, 8)`, planes are
concatenated along the neuron axis, and the array is cast to `float32`. No normalisation, smoothing,
z-scoring, or baseline subtraction is applied. Each plane is binned against its own
`ophys_timestamps`, which matters for Multiscope sessions where planes are offset by tens of ms.

ii.
```python
def binned_sum(values, timestamps, t0, t1):
    """Sum `values` (n_signals, n_samples) over each [t0[i], t1[i]) window."""
    csum = np.concatenate([np.zeros((values.shape[0], 1)), np.cumsum(values, axis=1)], axis=1)
    i0 = np.searchsorted(timestamps, t0, side='left')
    i1 = np.searchsorted(timestamps, t1, side='left')
    return csum[:, i1] - csum[:, i0], (i1 - i0)
```
```python
        s, _ = binned_sum(ev, ts, flat_t0, flat_t1)
        neural_planes.append(s.reshape(ev.shape[0], n_trials, N_BINS))
    neural = np.concatenate(neural_planes, axis=0).astype(np.float32)   # (n_cells, n_tr, T)
```

iii. Key Decision 2: "the paper explicitly uses detected calcium events; summing is the natural per-bin
aggregate and keeps rigs with different frame rates comparable." Step 4 elaborates: "Summing is the
integral of event magnitude over the bin, avoids sparsity of instantaneous sampling, and needs no
arbitrary smoothing kernel." The cumsum implementation is justified on efficiency grounds in Step 6:
"Naively looping over bins and re-scanning the event array per bin would be O(n_bins x n_frames).
Replaced with one cumulative sum per plane plus `searchsorted` lookups." `float32` was chosen "to halve
memory".

## 2-c. How is the `neural` data filtered based on quality controls?

i. No filtering beyond what the SDK already applies. `from_nwb_path` defaults to
`exclude_invalid_rois=True`, so ROIs failing the whitepaper's ROI-filtering criteria (unions,
duplicates, motion-border, likely dendrites, too small/dim) never enter the data; the AI verified
`cell_specimen_table.valid_roi` is True for 100% of returned cells. No per-neuron activity threshold,
no SNR/event-rate cut, and — after an explicit investigation — **no minimum-neurons-per-session
filter**, even though 65 of 171 sessions have ≤50 neurons and produce 2,370 all-zero trials.

ii. There is no filtering code; quality control is inherited from the loader default:
```python
        planes.append((eid, BehaviorOphysExperiment.from_nwb_path(NWB_FMT % eid)))
```
with metadata recording the inherited rule:
```python
        neuron_curation='valid ROIs only (allensdk exclude_invalid_rois=True)',
```

iii. Step 3 "Neuron curation rules": "Only valid ROIs. `from_nwb_path` applies
`exclude_invalid_rois=True` by default ... No further per-neuron filtering is described for this
dataset; unlike ephys there are no spike-sorting quality metrics." Step 12 documents the decision not
to add a neuron-count filter and shows the AI knew it would raise its scores: "Restricting to N>=20
would keep 98.5% of neurons while dropping ~19% of trials, and would very likely raise every reported
accuracy. **Decision: I did not apply a neuron-count filter.** No reference text specifies a
minimum-neuron criterion for this dataset ... Removing ~20% of real trials *because they decode poorly*
would optimise the reported score rather than the fidelity of the conversion." Step 10 Check 1
similarly explains the all-zero-trial warnings as a real property of sparse detected events, verified
by recomputing the same trials from the raw NWB.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to `trials.change_time` — the real image change on go trials and the sham change on
catch trials. Because `change_time` was verified to coincide *exactly* with a stimulus-presentation
onset, the AI converts it to a flash index `j` and builds the trial's 8 bins from flash onsets
`j−3 … j+4`; the change flash is always bin index 3. Neural data are placed in those bins using each
plane's own `ophys_timestamps` via `searchsorted`, so the ophys frames assigned to a bin are exactly
those sampled within `[flash_start, flash_start + 0.75)`. All output streams use the identical
`flat_t0/flat_t1` window arrays, so alignment between neural and outputs is exact by construction.

ii.
```python
    change_times = trials.change_time.values.astype(float)
    j = np.searchsorted(flash_start, change_times - 1e-6, side='left')
    good &= np.abs(flash_start[j_clipped] - change_times) < 1e-4   # change is a flash onset
    ...
    bin_flash = j[:, None] + np.arange(-N_PRE, N_POST + 1)[None, :]
    bt0 = flash_start[bin_flash]
    bt1 = bt0 + BIN_SIZE
    flat_t0 = bt0.ravel(); flat_t1 = bt1.ravel()
```
```python
        s, _ = binned_sum(ev, ts, flat_t0, flat_t1)   # ts = this plane's ophys_timestamps
```
```python
        temporal_alignment_event=('image change (go trials) or sham image change (catch '
                                  'trials), i.e. trials.change_time, which coincides exactly '
                                  'with an image flash onset'),
        off_start=OFF_START,          # -2.25
        off_end=N_POST * BIN_SIZE,    # +3.00
```

iii. Step 4 resolves alignment: "change_time coincides exactly (0.000000 s) with a flash onset on every
go and catch trial ... change / sham change is the event the animal reports → Align trials to
change_time." Step 1 cites `Trial.calculate_change_frame` / `add_change_time` as the definition and
notes that `change_time` "is the natural per-trial alignment event and is defined for every go and
catch trial". The AI also reasoned in Step 4 that all NWB streams are already rebased onto one common
session clock, so "alignment means resampling onto a chosen grid, not applying offsets"; each plane is
nevertheless binned on its own timestamps to respect Multiscope inter-plane offsets (up to 0.07 s).
Note: the metadata `off_end` is reported as +3.00 s (the *start* of the last bin); the last bin
actually ends at +3.75 s, which the code defines as the unused constant `OFF_END`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — the data are aggressively rebinned. The native ophys sampling is 31 Hz (single-plane rigs,
~32 ms) or 11 Hz (Multiscope, ~91 ms); the converted data use **750 ms bins**, one per image
presentation interval (250 ms image + 500 ms grey), with exactly 8 bins per trial. `time_bin_size` is
reported as 750.0 ms. Neural data are summed within bins; running and pupil are averaged within bins.
Bin edges are the true consecutive flash onsets (median spacing 0.7506 s, range 0.73–0.77 s), so the
nominal 750 ms label is an approximation of a slightly jittered real cadence.

ii.
```python
BIN_SIZE = 0.75          # s, one image presentation interval (250 ms image + 500 ms gray)
...
def bin_edges_from_flashes(flash_starts):
    """Bin k spans [flash_start[k], flash_start[k] + BIN_SIZE)."""
    return flash_starts, flash_starts + BIN_SIZE
```
```python
    data['metadata'] = dict(
        ...
        time_bin_size=BIN_SIZE * 1000.0,
        n_time_bins=N_BINS,
```

iii. Key Decision 1: "Bin 750 ms, 8 bins/trial aligned to change_time: matches the paper image
presentation interval and guarantees identical T across trials and across both rigs (31 Hz and 11 Hz),
which a fixed grid in seconds could not do cleanly." Step 3 grounds the 750 ms choice in the paper:
"By image presentation interval we refer to the 750 ms interval beginning with each image
presentation", and Step 5 adds "Fixed T = 8 for every trial in every session satisfies the requirement
that time bins be the same size for all trials and sessions." Step 10 Check 3(d) records this as
matching the reference: "paper: 'the 750 ms interval beginning with each image presentation' | bins are
the actual consecutive flash `start_time`s, width 750 ms | yes".

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `stimulus_presentations.image_name`, taken from the `change_detection` stimulus block only (the
5-minute grey-screen and natural-movie blocks are excluded), together with the `omitted` column. The
identity for a bin is the `image_name` of the flash that defines that bin, indexed via the same
`bin_flash` index matrix used for everything else. Omitted flashes are forced to the literal category
`'omitted'`. The trials-table columns `initial_image_name` / `change_image_name` are used only as
cross-checks, not as the source.

ii.
```python
    sp = ref.stimulus_presentations
    sp = sp[sp.stimulus_block_name.str.contains('change_detection', na=False)]
    sp = sp.sort_values('start_time')
    flash_image = sp.image_name.values.astype(object)
    flash_omitted = sp.omitted.values.astype(bool)
    # omitted flashes carry image_name 'omitted' already in this release; enforce it
    flash_image = np.where(flash_omitted, 'omitted', flash_image)
    ...
    img = flash_image[bin_flash]                                  # (n_trials, T) of str
```

iii. Step 1 documents the need for the block filter: `stimulus_presentations` "Must be filtered with
`stimulus_block_name.str.contains('change_detection')` (SDK >=2.16 warning) to get the task block,
excluding the 5-min gray screens and the natural movie block." Step 4 justifies the omitted category:
"omitted flashes always flanked by the same image, never adjacent to a change ... Give omissions their
own image-identity category rather than dropping or filling them, so no timepoint is fabricated."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A single global vocabulary is built from the union of image names over all converted sessions:
16 natural images (image sets A and B, 8 each) sorted alphabetically, with `'omitted'` appended last,
giving 17 classes (codes 0–16). Per-bin strings are mapped to those integer codes and stored as
`int64`, time-varying, shape `(n_trials, 8)`, as output row 0. The class names are exposed in
`output_values[0]`.

ii.
```python
    images = sorted(set(v for r in results for v in np.unique(r['image'])))
    images = [v for v in images if v != 'omitted'] + ['omitted']
    img_map = {v: i for i, v in enumerate(images)}
```
```python
        img_idx = np.vectorize(img_map.get)(r['image']).astype(np.int64)
        data['output'].append([
            np.stack([img_idx[t], r['change'][t], r['run_cls'][t], r['pupil_cls'][t],
                      np.full(N_BINS, r['outcome'][t], dtype=np.int64)]).astype(np.int64)
            for t in range(n_tr)])
```

iii. Step 5 variable mapping: "categorical index over a shared 17-value vocabulary (16 images of sets A
and B, plus omitted); time-varying ... shared vocabulary keeps the shared readout consistent across
sessions." The AI first checked the decoder architecture to confirm a shared readout over sessions
(trajectory step 34), which is why a single global mapping rather than a per-session one was used.
Sorting makes the mapping deterministic; pushing `'omitted'` to the end keeps the 16 real images in a
contiguous, interpretable block. The verification log shows the resulting distribution — each image
0.058–0.063, `omitted` 0.032 — consistent with the documented 5% omission rate applied to non-change,
non-pre-change flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Alignment is exact by construction rather than by resampling: the image identity of bin *k* is the
`image_name` of the same flash whose `start_time` defines bin *k*'s window, and the neural data for bin
*k* are the events falling in that same window. Two assertions run on every session verify the
alignment against the independent trials table: the image at bin 3 must equal
`trials.change_image_name`, and the image at bin 2 must equal `trials.initial_image_name` (guarded for
omissions, which in practice never occur at bin 2).

ii.
```python
    img = flash_image[bin_flash]
    ...
    assert np.array_equal(img[:, N_PRE], tr_sel.change_image_name.values.astype(object)), \
        'image at change bin != trials.change_image_name'
    pre = img[:, N_PRE - 1]
    init = tr_sel.initial_image_name.values.astype(object)
    notom = pre != 'omitted'
    assert np.all(pre[notom] == init[notom]), 'image before change != trials.initial_image_name'
```

iii. Step 5 planned this as a sanity check ("Image identity at bin 3 equals trials.change_image_name;
at bin 2 equals trials.initial_image_name") and Step 10 Check 2b reports the external verification:
bin 2 "also equals `trials.initial_image_name` for **100%** of trials" and is `omitted` in "**0 of
43,975 trials**, exactly reproducing" the paper's statement that "image changes as well as the image
immediately before the change were not omitted" — an independent confirmation that the window is
locked to the correct flash. An independent script (`cache/sanity_checks.py`) re-derived the image
identities from the raw NWB without importing the conversion code and reports an exact match.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `stimulus_presentations.is_change`, indexed at the same 8 flash positions as everything else. The
trials-table `go`/`catch` flags are used only in assertions. `is_change` marks only *real* image
changes, so catch (sham-change) trials are 0 in every bin.

ii.
```python
    flash_ischange = sp.is_change.values.astype(bool)
    ...
    chg = flash_ischange[bin_flash].astype(np.int64)
```

iii. Step 5 variable mapping: "1 in the bin whose flash is_change, else 0; time-varying ... value 1
right after a change in image identity = the bin beginning at the change flash". The AI notes in the
script itself: "`is_change` marks only REAL image changes. Catch trials are sham changes where the
image does not change, so they must be 0 everywhere — which is exactly the semantics required for the
image_change output ('1 right after a change in image identity')."

## 4-b. What processing is involved in computing `output` *Image change*?

i. None beyond the index-and-cast: the boolean flash flags are gathered with `bin_flash` and cast to
`int64`, giving a binary `(n_trials, 8)` row. On go trials there is exactly one 1, always at bin 3; on
catch trials the row is all zeros. Stored as output row 1 with `output_values[1] = ['no_change',
'change']`. Three assertions enforce this structure per session.

ii.
```python
    chg = flash_ischange[bin_flash].astype(np.int64)
    ...
    assert np.all(chg[is_go, N_PRE] == 1), 'go trial change bin is not flagged is_change'
    assert np.all(chg[~is_go, N_PRE] == 0), 'catch trial change bin is wrongly flagged'
    assert np.all(chg[:, :N_PRE] == 0), 'a change appears before the aligned change bin'
```

iii. Step 6 records a bug found and fixed here: the AI's first assertion assumed every aligned trial has
`is_change == 1` and failed; investigation showed "this was my error, not a data problem ... on
**catch** trials the change_time marks a *sham* change where the image does not actually change. This
is exactly the semantics the Decoder Task wants." Step 10 Check 5 confirms the final structure over the
whole dataset: "38,460 trials with exactly one 1, **all at bin 3**; 5,515 catch trials all-zero; 0
anomalies", and independently "Catch trials: image at change bin == preceding image — **True for all
5,515**".

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed or applied: the source variable `is_change` is already boolean, so the
output is a 2-class categorical (`no_change` = 0, `change` = 1) obtained by a direct cast. The only
"category" decision is the temporal extent of the 1: it occupies exactly one 750 ms bin — the image
presentation interval that begins at the change — which is 1 of 8 bins on go trials, giving an overall
distribution of [0.891, 0.109].

ii.
```python
                output_values=[images, ['no_change', 'change'], ...]
```
```python
    chg = flash_ischange[bin_flash].astype(np.int64)
```

iii. Step 5: "value 1 right after a change in image identity = the bin beginning at the change flash",
matching the instruction "Have value of 1 right after a change in image identity, otherwise 0". The
750 ms extent is the paper's image presentation interval (250 ms image + 500 ms grey), i.e. the
duration for which the changed image and its trailing grey period are the current stimulus event.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: the flag is read from the same flash that defines each bin, and the
neural data for that bin are the events in the same `[flash_start, flash_start + 0.75)` window. The
change therefore always lands at bin index 3 of every trial, which is also what the alignment event
(`change_time`) defines, and the `--show-processing` figures render the whole `image_change` matrix to
confirm it is a single clean column at bin 3.

ii.
```python
    bin_flash = j[:, None] + np.arange(-N_PRE, N_POST + 1)[None, :]
    chg = flash_ischange[bin_flash].astype(np.int64)
    ...
    a.imshow(res['change'], aspect='auto', interpolation='nearest')
    a.set_title('image_change output (should be a single column at bin %d)' % N_PRE)
```

iii. Step 7 processing-plot review: the figures show "the `image_change` output as an image (a single
clean column at bin 3) ... No temporal misalignment or discretisation anomaly is visible." Step 10
Check 5 verified the same property numerically across all 43,975 trials, and `cache/sanity_checks.py`
re-derived `is_change` at the 8 flash positions from the raw NWB with an exact match.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `ref.running_speed`, i.e. the SDK's running-wheel DataFrame with `timestamps` and `speed` (cm/s,
~60 Hz), taken from the first imaging plane of the session (behaviour streams were verified identical
across planes).

ii.
```python
    run = ref.running_speed
    run_t = run.timestamps.values.astype(float)
    run_v = run.speed.values.astype(float)
    order = np.argsort(run_t)
    run_t, run_v = run_t[order], run_v[order]
    run_v = interpolate_nans(run_v)
```

iii. Step 1 identifies `.running_speed` as the SDK's standard locomotion interface: "DataFrame
`timestamps`, `speed` (cm/s) at ~60 Hz". Sorting by timestamp and NaN-interpolation are defensive
pre-processing so that the later `searchsorted`-based binning is valid and can never produce a NaN
output (which `verify_data_format` rejects).

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is averaged within each 750 ms bin, using the same cumulative-sum/`searchsorted` machinery as
the neural data (sum of samples in `[t0, t1)` divided by the number of samples). If a bin happens to
contain no running sample, the value falls back to a linear interpolation of the raw trace at the bin
centre, so no NaN can appear. The binned continuous values `(n_trials, 8)` are then discretised (5-c).
Running speed is *not* resampled onto the ophys timebase; it is aggregated directly onto the shared
flash-defined bin windows.

ii.
```python
def binned_mean_1d(values, timestamps, t0, t1):
    """Mean of a 1-D signal in each window; empty windows fall back to linear interpolation
    at the window centre so that no NaN is produced."""
    s, n = binned_sum(values[None, :], timestamps, t0, t1)
    s = s[0]
    n = n.astype(float)
    out = np.empty_like(s)
    ok = n > 0
    out[ok] = s[ok] / n[ok]
    if np.any(~ok):
        centres = 0.5 * (t0[~ok] + t1[~ok])
        out[~ok] = np.interp(centres, timestamps, values)
    return out
```
```python
    run_binned = binned_mean_1d(run_v, run_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)
```

iii. Step 5 variable mapping: "mean speed within each bin, discretized into 5 equal-percentile bins".
The mean (rather than a point sample) is the natural aggregate when downsampling a 60 Hz signal to
750 ms bins — ~45 samples per bin. The interpolation fallback exists so that missing samples can never
produce NaN, which the format verifier rejects; the AI also asserts finiteness per session
(`assert np.isfinite(...run_binned).all()`).

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Into 5 equal-percentile classes (quintiles), with the percentile edges computed **within each
session** over all bins of all trials of that session, not globally across the dataset. Edges are the
20/40/60/80th percentiles of the session's binned speeds; assignment is `np.searchsorted(edges, x,
side='right')`, so ties on an edge go to the upper class and degenerate (repeated) edges simply collapse
one class rather than erroring. The verification log confirms the overall distribution is exactly
[0.200, 0.200, 0.200, 0.200, 0.200].

ii.
```python
def quantile_bin(x, n=N_QUANTILES):
    """Discretise into n equal-percentile bins. Returns integer labels in [0, n-1].
    Percentile edges are computed on the values being discretised, so each class holds ~1/n
    of the samples. Ties (e.g. a mouse that is stationary for >20% of the time) can make
    edges coincide; np.searchsorted then simply assigns fewer samples to the collapsed class."""
    edges = np.quantile(x, np.linspace(0, 1, n + 1)[1:-1])
    return np.searchsorted(edges, x, side='right').astype(np.int64), edges
```
```python
    run_cls, run_edges = quantile_bin(run_binned.ravel())
    run_cls = run_cls.reshape(n_trials, N_BINS)
```

iii. Key Decision 7: "pupil width is measured in camera pixels whose scale depends on the per-session
camera alignment, and running propensity differs enormously between animals, so a single global edge
set would mostly encode which session/mouse a trial came from rather than the animal state. Computing
the five equal-percentile edges within each session makes each class mean the same thing (relative
level for this animal on this day) everywhere, and yields exactly ~20% per class both per session and
overall (verified: 0.199-0.202)." The metadata records the choice explicitly:
`discretization='running speed and pupil diameter are binned into 5 equal-percentile classes computed
within each session'`.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. By sharing the bin windows: `binned_mean_1d` is called with the identical `flat_t0`/`flat_t1` arrays
used for the neural binning, so bin *k* of the running row covers exactly the same wall-clock interval
as bin *k* of the neural matrix. Because all NWB streams already live on one session clock, no offset
correction is required. The `--show-processing` figure overlays the 8 bin windows on the raw 60 Hz
trace so the alignment can be inspected visually.

ii.
```python
    run_binned = binned_mean_1d(run_v, run_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)
```
```python
    a.plot(run_t[(run_t >= w0) & (run_t <= w1)], run_v[(run_t >= w0) & (run_t <= w1)], 'k')
    a.axvline(ct, color='r', lw=2)
    for e0, e1 in zip(bt0[0], bt1[0]):
        a.axvspan(e0, e1, color='b', alpha=0.08)
    a.set_title('raw running speed with the 8 trial bins shaded')
```

iii. Step 3 processing details: "All NWB data streams (ophys, running, eye, stimulus, trials, licks,
rewards) are already rebased onto one common session clock, so alignment means resampling onto a chosen
grid, not applying offsets." Using one shared set of window boundaries for every stream makes
cross-stream alignment exact by construction. `cache/sanity_checks.py` independently re-binned and
re-quantised running speed from the raw NWB; the single disagreeing bin out of 2,136 was traced to a
floating-point tie exactly on a quantile edge in the *checking* script, with a direct comparison of the
cumsum-based mean against a naive `.mean()` giving max abs diff 1.3e-10.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `ref.eye_tracking`, using the `pupil_width` column as the diameter measure and its `timestamps`
(~30 Hz). Blink frames appear as NaN in this column — the AI verified that the NaN mask is *exactly*
the `likely_blink` flag — and are filled by interpolation rather than by dropping the row. Sessions
whose eye-tracking table is empty, or whose pupil trace is entirely NaN, are dropped.

ii.
```python
    eye = ref.eye_tracking
    if len(eye) == 0:
        print(f'  session {sid}: DROPPED (no eye tracking data)', flush=True)
        return None
    pupil = interpolate_nans(eye.pupil_width.values)
    if pupil is None:
        print(f'  session {sid}: DROPPED (pupil all NaN)', flush=True)
        return None
    eye_t = eye.timestamps.values.astype(float)
    order = np.argsort(eye_t)
    eye_t, pupil = eye_t[order], pupil[order]
```

iii. Step 4 discrepancy table: "SDK gives pupil_area, pupil_width, pupil_height | pupil_area is NaN
exactly where likely_blink is True | Use **pupil_width as diameter** (area is an ellipse area, not a
diameter; width correlates 0.986 with area). Interpolate across blinks within session before binning."
The 0.986 correlation between width and area was measured directly from the data
(`cache/check_assumptions.py`). Key Decision 8 justifies dropping eye-tracking-free sessions rather
than emitting NaN.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Three steps: (1) blink NaNs are linearly interpolated across sample index (edges filled with the
nearest valid value); (2) the cleaned trace is averaged within each 750 ms bin with the same
`binned_mean_1d` used for running speed, including the interpolate-at-bin-centre fallback for empty
bins; (3) the binned values are discretised into 5 within-session quintiles. No smoothing, no
z-scoring, no conversion to physical units.

ii.
```python
def interpolate_nans(x):
    """Linearly interpolate NaNs (blinks) in a 1-D array; edges are filled with nearest value."""
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return None
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x
```
```python
    pup_binned = binned_mean_1d(pupil, eye_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)
    pup_cls, pup_edges = quantile_bin(pup_binned.ravel())
    pup_cls = pup_cls.reshape(n_trials, N_BINS)
```

iii. Step 5 variable mapping: "blinks (NaN) linearly interpolated within session, mean within bin, then
5 equal-percentile bins". Interpolating rather than dropping blink samples keeps the trace on a regular
grid and prevents a blink-heavy bin from becoming empty; because blinks are ~2.6% of samples and
isolated, interpolation across them is a standard treatment. `cache/sanity_checks.py` re-derived the
pupil output from the raw NWB (re-interpolate blinks, re-bin, re-quantile) and reports an exact match.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Exactly as running speed: 5 equal-percentile classes whose edges are computed **within each session**
from that session's binned pupil values, assigned with `np.searchsorted(..., side='right')`. Overall
distribution in the converted data is [0.200, 0.200, 0.200, 0.200, 0.200].

ii.
```python
    pup_cls, pup_edges = quantile_bin(pup_binned.ravel())
```
```python
                output_values=[..., [f'pupil_q{k+1}' for k in range(N_QUANTILES)], ...]
```

iii. Key Decision 7 (quoted in 5-c) makes the pupil-specific argument the primary one: "pupil width is
measured in camera pixels whose scale depends on the per-session camera alignment ... so a single
global edge set would mostly encode which session/mouse a trial came from rather than the animal
state." Step 12 adds the consequence for interpretation: "pupil_diameter is an arousal variable only
indirectly reflected in V1/LM activity, and is quantised within session, so the classes are by
construction equally likely and cannot be won by a prior."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. By sharing the bin windows, exactly as for running speed: `binned_mean_1d(pupil, eye_t, flat_t0,
flat_t1)` uses the same window arrays as the neural binning, so pupil bin *k* and neural bin *k* cover
the same interval on the common session clock. The `--show-processing` figure plots the raw pupil trace
with the 8 bin windows shaded, next to the raw events and the change marker.

ii.
```python
    pup_binned = binned_mean_1d(pupil, eye_t, flat_t0, flat_t1).reshape(n_trials, N_BINS)
```
```python
    me = (eye_t >= w0) & (eye_t <= w1)
    a.plot(eye_t[me], pupil[me], 'k')
    a.axvline(ct, color='r', lw=2)
    for e0, e1 in zip(bt0[0], bt1[0]):
        a.axvspan(e0, e1, color='b', alpha=0.08)
    a.set_title('raw pupil width (blinks interpolated)')
```

iii. Same justification as running speed (Step 3: all streams share one session clock; one shared set of
window boundaries makes alignment exact by construction). The AI additionally asserts per session that
the binned pupil is finite, and the independent sanity-check script reproduced the binned/quantised
pupil values exactly from the raw NWB.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually-exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`,
`correct_reject`, in that fixed order (codes 0–3). Trials for which none of the four is True are
dropped rather than being given a fallback label (this never occurred in the full run).

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
    oc = np.full(len(trials), -1, dtype=np.int64)
    for k, name in enumerate(OUTCOMES):
        oc[trials[name].values.astype(bool)] = k
    good &= oc >= 0
```

iii. Step 1 cites `Trial._get_trial_data` as the definitive source: "`hit/miss/false_alarm/
correct_reject` are mutually exclusive outcomes", with go trials mapping to hit/miss and catch trials
to false_alarm/correct_reject (Step 5 mapping table). The `oc >= 0` guard ensures no trial ever carries
an invalid class label. Step 9's consistency table shows the converted outcome counts (13,940 / 24,520
/ 834 / 4,681) match the raw trials tables exactly, and the derived hit rate among go (0.3625) and FA
rate among catch (0.1512) reproduce the raw data.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is a single integer per trial, broadcast (constant) across all 8 time bins so that the
output block has uniform shape `(5, 8)`. It occupies output row 4, with
`output_values[4] = ['hit', 'miss', 'false_alarm', 'correct_reject']`. No other processing; no
re-derivation from licks or rewards.

ii.
```python
    outcome = oc[idx]
```
```python
        data['output'].append([
            np.stack([img_idx[t], r['change'][t], r['run_cls'][t], r['pupil_cls'][t],
                      np.full(N_BINS, r['outcome'][t], dtype=np.int64)]).astype(np.int64)
            for t in range(n_tr)])
```

iii. Key Decision 6: "All outputs time-varying: image identity, change, running and pupil vary within a
trial; trial outcome is constant per trial but broadcast across bins, per the instruction to make
outputs time-varying if at all possible." Step 10 Check 5 verified "Outcome constant within trial: 0
violations". Step 12 discusses the consequence for decodability honestly: "trial_outcome is constant
within a trial, so the decoder must infer it from 8 bins that include 3 pre-change bins carrying no
outcome information at all; and the classes are very unbalanced (FA is 1.9% of trials)."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI's approach is to guarantee that no NaN/Inf ever reaches the output, and to drop rather than
impute whenever an entire stream is unusable:
- **No eye tracking / all-NaN pupil** → the whole session is dropped with a printed reason (3 sessions,
  917 trials lost).
- **Blinks (NaN pupil samples)** → linearly interpolated within the session before binning.
- **NaN running samples** → the same interpolation.
- **Empty behaviour bins** (a bin with no running/eye sample) → filled by linear interpolation of the
  raw trace at the bin centre instead of emitting NaN.
- **Unsorted timestamps** → both behaviour streams are explicitly sorted by timestamp.
- **Missing / non-flash-aligned `change_time`, window running off the stimulus block, trial with no
  valid outcome** → the trial is excluded.
- **Session with <2 usable trials** → the session is dropped.
- **Sparse/all-zero neural trials** (2,370 of 43,975) → deliberately *kept*, after verifying against
  the raw NWB that they are genuinely zero.
- Finally, per-session assertions check that neural, running and pupil are all finite, so any unhandled
  data defect fails loudly instead of silently producing bad output. There is, however, no `try/except`
  around `process_session`, so an unanticipated failure in one session would abort the whole run.

ii.
```python
    if len(eye) == 0:
        print(f'  session {sid}: DROPPED (no eye tracking data)', flush=True)
        return None
    pupil = interpolate_nans(eye.pupil_width.values)
    if pupil is None:
        print(f'  session {sid}: DROPPED (pupil all NaN)', flush=True)
        return None
```
```python
    if np.any(~ok):
        centres = 0.5 * (t0[~ok] + t1[~ok])
        out[~ok] = np.interp(centres, timestamps, values)
```
```python
    good &= np.isfinite(change_times)
    good &= np.abs(flash_start[j_clipped] - change_times) < 1e-4
    good &= (j - N_PRE >= 0) & (j + N_POST < len(flash_start))
    good &= oc >= 0
    idx = np.nonzero(good)[0]
    if len(idx) < 2:
        return None
```
```python
    assert np.isfinite(neural).all() and np.isfinite(run_binned).all() and np.isfinite(pup_binned).all()
```

iii. Key Decision 8: "Drop the 3 sessions lacking eye tracking rather than emit NaN, which
verify_data_format rejects." Step 5 mapping: "3 experiments (3 sessions) have an empty eye table; those
sessions are dropped since outputs cannot be NaN." The loss is accounted for exactly in the Step 9
reconciliation table (44,892 − 917 = 43,975 trials; 29,444 − 276 = 29,168 neurons), so the AI can show
that no data was lost unintentionally. Step 10 Check 1 justifies *not* removing all-zero neural trials:
"it reflects a true property of the recordings, not a conversion error. The independent sanity check
below recomputed these same trials from the raw NWB and got identical all-zero matrices."

## 9-a. What are the most time-consuming steps of the code?

i. Reading the NWB files dominates. The script instruments each stage with a `timings` dict
(`load`, `align`, `neural`, `outputs`) and prints per-session totals. Measured: ~2.5–3 s to load one
imaging plane, so a 1-plane session takes ~2.7–7.7 s end to end and a 7-plane Multiscope session ~18.5 s
— essentially all of it in `BehaviorOphysExperiment.from_nwb_path`. Binning is negligible by
comparison: the cumsum over the whole session's event array plus `searchsorted` lookups is a single
pass. Wall-clock for the full 202-plane conversion was 49.9 s with 24 worker processes (~9 min serial).

ii.
```python
    t0 = time.time()
    planes = []
    for eid in eids:
        planes.append((eid, BehaviorOphysExperiment.from_nwb_path(NWB_FMT % eid)))
    timings['load'] = time.time() - t0
```
```python
    print(f'  session {sid}: {len(eids)} plane(s), {neural.shape[0]} neurons, '
          f'{n_trials} trials (dropped {n_dropped}), {time.time() - t_start:.1f}s', flush=True)
```

iii. Step 1: "Loading one experiment from the local NWB takes ~2.5-3 s, so the full dataset is cheap to
convert with multiprocessing." Step 7's run-time table extrapolates "202 planes total, ~2.6 s/plane =>
~9 min serial ... with 24 workers: **~1-2 min wall clock**", and concludes "Well under the 15 minute
budget, so no further optimisation was needed" — which the 51.2 s actual full run confirms.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner processing is already vectorised: binning is done for *all* trials × all 8 bins at once by
flattening the window arrays (`flat_t0`, `flat_t1`) and doing one cumsum + two `searchsorted` calls per
plane, and image/change lookups are fancy-indexing with the `(n_trials, 8)` `bin_flash` matrix. The
loops that remain and could in principle be vectorised are all cheap post-processing in `main()`:
- `np.vectorize(img_map.get)(r['image'])` — `np.vectorize` is a Python-level loop; a
  `np.searchsorted` into the sorted vocabulary, or `pd.Categorical`, would be genuinely vectorised.
- the per-trial list comprehensions that split `r['neural']` into `n_tr` contiguous arrays and
  `np.stack` five rows per trial — these run once per trial (43,975 times) and each allocates a small
  array; the split is forced by the target format, which requires a Python list of per-trial arrays.
- `regions.index(x)` per neuron and `subjects.index(...)` per session — linear scans inside a
  comprehension; a dict lookup would be O(1), though with 2 regions and 38 mice it is irrelevant.
- the per-plane loop in `process_session` (at most 7 iterations, dominated by I/O anyway).

ii.
```python
        img_idx = np.vectorize(img_map.get)(r['image']).astype(np.int64)
        data['neural'].append([np.ascontiguousarray(r['neural'][:, t, :]) for t in range(n_tr)])
        data['brain_region_idx'].append(np.array([regions.index(x) for x in r['regions']],
                                                 dtype=np.int64))
```

iii. The AI's stated efficiency reasoning (Step 6) targeted the parts that mattered: "Naively looping
over bins and re-scanning the event array per bin would be O(n_bins x n_frames). Replaced with one
cumulative sum per plane plus `searchsorted` lookups", plus `multiprocessing.Pool` over sessions
(sessions are independent). The remaining loops are outside the measured bottleneck (I/O), so
vectorising them would not change the ~50 s run time; the AI did not discuss them explicitly.

## 9-c. What processing does the code repeat multiple times?

i. Very little is repeated. Each NWB is opened exactly once, and behaviour streams (trials, stimulus
presentations, running, eye tracking) are read only from the first plane rather than from every plane of
a Multiscope session. The repetitions that do exist are minor:
- A `BehaviorOphysExperiment` object is still constructed for every plane, so each plane's copy of the
  shared behaviour tables is parsed even though only plane 0's copy is used (the SDK loads much of this
  lazily, so the cost is mostly the NWB open itself, which is unavoidable).
- In `--show-processing` mode, `np.vstack(ds.events.events.values)` is executed a second time inside
  `plot_processing` for plane 0, re-materialising an array that was already built during binning.
- `binned_sum` computes a cumulative sum over the *entire* session event trace (e.g. 666 × 48,316) even
  though only ~8 × n_trials windows are read from it; this is a deliberate trade (one pass instead of
  per-bin rescans) but does touch frames that no bin uses.
- Quantile edges are computed once per session and also returned in the result dict for the plots.

ii.
```python
    ref = planes[0][1]          # behavior streams are identical across planes
    sp = ref.stimulus_presentations
    ...
    trials = ref.trials
```
```python
    ds = planes[0][1]
    ts = ds.ophys_timestamps
    ev = np.vstack(ds.events.events.values)   # re-materialised for plotting only
```

iii. Step 6 "Code inefficiencies identified": "Loading each NWB more than once (e.g. once for behaviour
and once for neural) would double the IO. Each plane is loaded exactly once and the behaviour streams
are read only from the first plane." Reading behaviour from plane 0 only is justified by the verified
fact that the planes of a session share an identical trials table (`trials_equal True` for every plane
in the checked Multiscope session).

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Nothing substantial; a handful of small items are computed and then not written to the pickle:
- `run_cont` / `pupil_cont` (the continuous binned running and pupil arrays) and `run_edges` /
  `pupil_edges` are kept in each session's result dict but only ever used by `plot_processing`; in a
  `--full` run without `--show-processing` they are computed, pickled across the multiprocessing
  boundary, and dropped.
- `n_cells_plane`, `trial_change_times`, `is_go`, `timings`, `total_time` are likewise carried back from
  the workers and only `timings`-adjacent printing uses some of them; `n_cells_plane` is never used.
- `binned_sum` returns a per-window sample count that the neural path discards (`s, _ = ...`); it is
  needed only by `binned_mean_1d`.
- The full-session cumulative sum covers frames outside every trial window (the inter-trial and
  aborted-trial periods), so a large fraction of the accumulated array is never read.
- Per-trial empty input arrays `np.zeros((0, N_BINS))` are allocated for all 43,975 trials even though
  the task has no decoder inputs.
- `bin_edges_from_flashes()` and the constant `OFF_END` are defined but never called/used (the metadata
  uses `N_POST * BIN_SIZE` instead).
- The three per-session `assert` blocks re-derive `change_image_name` / `initial_image_name`
  comparisons on every run; these are verification, not conversion, work (deliberately retained).

ii.
```python
    result = dict(
        session_id=sid, experiment_ids=eids, mouse=str(md['mouse_id']),
        neural=neural, image=img, change=chg, run_cls=run_cls, pupil_cls=pup_cls,
        outcome=outcome, regions=region_idx,
        run_cont=run_binned, pupil_cont=pup_binned,          # plotting only
        run_edges=run_edges, pupil_edges=pup_edges,          # plotting only
        n_trials=n_trials, n_dropped=n_dropped, n_cells_plane=n_cells_plane,  # unused
        ...)
```
```python
OFF_END = (N_POST + 1) * BIN_SIZE    # +3.75 s is end of last bin; start of last bin = +3.0
...
def bin_edges_from_flashes(flash_starts):   # never called
```

iii. The AI does not discuss these items; its efficiency discussion (Step 6) focuses on the two things
that dominate cost — I/O and per-bin rescans — both of which it eliminated. The retained assertions are
justified by the Step 10 review philosophy of failing loudly on any data anomaly, and the per-trial
empty input arrays are required to satisfy the target format ("Verified the decoder trains with
dinput=0", Step 5).
