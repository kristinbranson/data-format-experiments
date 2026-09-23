# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All access goes through the AllenSDK `VisualBehaviorOphysProjectCache`, opened with
`from_local_cache(cache_dir='/app/data')` (the container has no network, so the S3 constructor is
not usable). `get_ophys_experiment_table()` is the canonical listing; it is intersected with the
`ophys_experiment_id`s whose `.nwb` file is actually present locally (globbed from the cache
directory), then filtered to `project_code == 'VisualBehavior'` and `passive == False`, and sorted
by experiment id. Each selected experiment is then loaded with
`get_behavior_ophys_experiment(oeid)` inside a 16-process `multiprocessing.Pool`, and from each
loaded object the AI pulls `ophys_timestamps`, `dff_traces`, `stimulus_presentations`,
`running_speed`, `eye_tracking`, `trials` and `metadata`. Result: 168 experiments selected, 165
kept, 37 mice, 28,821 neurons, 42,470 trials. NWB files are never opened directly.

ii.
```python
def get_cache():
    global _CACHE
    if _CACHE is None:
        _CACHE = bpc.VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)
    return _CACHE

def local_experiment_ids():
    files = glob(os.path.join(NWB_DIR, '*.nwb'))
    return sorted(int(re.search(r'(\d+)\.nwb', f).group(1)) for f in files)

def select_experiments():
    et = get_cache().get_ophys_experiment_table()
    ids = local_experiment_ids()
    sel = et[(et.index.isin(ids)) &
             (et.project_code == PROJECT_CODE) &
             (~et.passive)].copy()
    return sel.sort_index()

def extract_session(oeid):
    ds = get_cache().get_behavior_ophys_experiment(oeid)
    ophys_ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
    ev = ds.dff_traces if NEURAL_SIGNAL == 'dff' else ds.events
    ...
with Pool(min(args.workers, len(oeids))) as pool:
    for i, info in enumerate(pool.imap(extract_session_safe, oeids, chunksize=1)):
        results.append(info)
```

iii. From CONVERSION_NOTES Step 5 Key Decision 1: the task says to convert data "under the *Visual
Behavior* task"; `VisualBehavior` is a literal `project_code` and a named *dataset variant* in the
whitepaper, and it is "the only project whose experiments are **completely** present in the local
cache (239/239)". The 45 locally present `VisualBehaviorMultiscope` planes were excluded because
they "belong to a single mouse and would require merging 3–7 planes acquired on different frame
clocks into one 'session'". Passive sessions were excluded because "the lick spout is retracted, so
no go/catch trial outcomes exist". `from_local_cache` instead of `from_s3_cache` is documented in
Step 10 Check 3 as "identical calls … because there is no network".

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the unique `mouse_id` values (taken from each experiment's SDK `metadata`), sorted
as strings; `subject_idx` is the position of each session's mouse in that sorted list. 37 mice.

ii.
```python
'mouse_id': str(md['mouse_id']),
...
subjects = sorted({s['mouse_id'] for s in sessions})
subject_idx = np.array([subjects.index(s['mouse_id']) for s in sessions], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5 variable-mapping table: `experiment_table.mouse_id` → `subjects`,
`subject_idx`, "unique sorted list", "37 mice". `mouse_id` is the SDK's animal identifier; the count
was cross-checked against the project metadata tables in Step 2.

## 1-c. How are the data split into sessions?

i. One "session" in the output = one `BehaviorOphysExperiment` = one imaging plane = one NWB file.
The AI verified that for the `VisualBehavior` project there is exactly one imaging plane per
`ophys_session_id`, so experiment ≡ session here (I independently confirmed this: all 239 local
`VisualBehavior` experiments map 1:1 onto 239 `ophys_session_id`s). Sessions are ordered by
ophys_experiment_id. Both the `ophys_experiment_id` and the `ophys_session_id` are recorded per
session in `metadata['session_info']`.

ii.
```python
# CONVERSION_NOTES Step 5: "session = one BehaviorOphysExperiment (one imaging plane; for
# VisualBehavior there is exactly one plane per ophys session, so session == experiment == NWB)"
sel = select_experiments()
oeids = list(sel.index.values)
...
session_info.append({
    'ophys_experiment_id': int(s['oeid']),
    'ophys_session_id': s['ophys_session_id'],
    'behavior_session_id': s['behavior_session_id'],
    ...})
```

iii. Docstring of `convert_data.py`: "One session == one BehaviorOphysExperiment (a single imaging
plane; the VisualBehavior project is single-plane, so session == experiment == NWB file)". The
multi-plane (Multiscope) project, where this identity would break, was deliberately excluded
(Step 5 Key Decision 1) precisely because merging planes with different frame clocks into one
session is not supported by the target format.

## 1-d. How are the data split into trials?

i. A trial is one row of the SDK `ds.trials` table with `go == True or catch == True`, spanning the
half-open interval `[start_time, stop_time)` on the ophys clock. Trial frames are the ophys frames
`t` with `start_time <= t < stop_time`, found with `np.searchsorted(..., side='left')`. Trials are
therefore variable length (217–389 frames, mean 263.6 ≈ 8.5 s) with a constant bin size. The
selection automatically excludes aborted and auto-rewarded trials, which the code asserts rather
than assumes.

ii.
```python
def frame_slice(ophys_ts, t_start, t_stop):
    """Indices of ophys frames with t_start <= t < t_stop."""
    i0 = int(np.searchsorted(ophys_ts, t_start, side='left'))
    i1 = int(np.searchsorted(ophys_ts, t_stop, side='left'))
    return i0, i1

trials = ds.trials
sel = trials[(trials.go.astype(bool)) | (trials.catch.astype(bool))]
assert not sel.aborted.any(), f'{oeid}: aborted trial selected'
assert not sel.auto_rewarded.any(), f'{oeid}: auto-rewarded trial selected'
...
for k, (_, tr) in enumerate(sel.iterrows()):
    i0, i1 = frame_slice(ophys_ts, tr.start_time, tr.stop_time)
```

iii. Step 5 Key Decision 4: "Trial window = the experiment's own trial (`start_time`→`stop_time`)
rather than a hand-chosen window around the change, because the task says to segment 'based on how
they are defined in the experiment'. Consequence: `off_start = 0.0` … and `off_end = None`
(variable-length trials)." Step 3 curation notes: "the SDK's own `Trial` logic makes `go`/`catch`
already mutually exclusive with aborted/auto-rewarded, so `trials[trials.go | trials.catch]` is
exactly the requested set (verified: 0 of 43,387 selected trials are aborted or auto-rewarded)".

## 1-e. How are trials filtered based on quality controls?

i. Four filters, at three levels:
- **Trial level**: only `go | catch` rows (i.e. aborted and auto-rewarded dropped); a trial with
  fewer than 2 ophys frames of coverage is dropped (`if i1 - i0 < 2: continue`). In the full run
  this guard never fired — `n_trials_kept == n_trials_selected` for all 165 sessions.
- **Session level**: `passive == False` (passive sessions have a retracted lick spout, so every
  trial would be a miss/correct-reject); sessions whose eye-tracking table is empty or unusable are
  dropped (3 sessions: 795953296, 806456687, 833631914 — 917 trials, 276 cells); sessions with no
  cells are dropped; sessions with fewer than 2 usable trials are dropped (none occurred; min was
  39 trials).
- No engagement/reward-rate filtering and no experience-level (familiar/novel) restriction.

ii.
```python
sel = trials[(trials.go.astype(bool)) | (trials.catch.astype(bool))]
...
    if i1 - i0 < 2:                    # no ophys coverage -> unusable trial
        continue
...
if len(ev) == 0:
    return {'oeid': oeid, 'skip': 'no cells'}
...
pupil = resample_pupil(ds, ophys_ts)
if pupil is None:
    return {'oeid': oeid, 'skip': 'no eye tracking'}
...
sessions = [r for r in results if not r['skip'] and r['n_trials_kept'] >= 2]
dropped_few = [r for r in results if not r['skip'] and r['n_trials_kept'] < 2]
for r in dropped_few:
    print(f'  SKIPPED {r["oeid"]}: only {r["n_trials_kept"]} usable trials')
```

iii. Step 3 curation rules: "per the Decoder Task, keep `go | catch` and drop `aborted` and
`auto_rewarded`"; "active behaviour only (passive sessions have a retracted lick spout, hence no
hit/miss/FA/CR outcomes)"; "sessions with an empty eye-tracking table dropped (3 sessions) because
pupil diameter is a required decoder output". Step 5 Key Decision 10: the 3 eye-tracking drops were
predicted in advance from the Step-2 scan "so nothing was lost silently". Piet et al.'s
familiar-images/multi-plane restriction was explicitly *not* applied because "applying it to this
cache would leave 22 imaging planes from **one** mouse, which cannot support a cross-session
decoder" (Step 10 Check 3).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `ds.dff_traces['dff']` — the detrended ΔF/F trace per cell produced by the Allen two-photon
pipeline, one value per cell per two-photon frame. `ds.events['events']` and
`['filtered_events']` remain selectable through a `--neural-signal` debug flag but are not used for
the shipped dataset.

ii.
```python
NEURAL_SIGNAL = 'dff'                    # detrended dF/F (Allen pipeline output)
...
ev = ds.dff_traces if NEURAL_SIGNAL == 'dff' else ds.events
if len(ev) == 0:
    return {'oeid': oeid, 'skip': 'no cells'}
activity = np.vstack(ev[NEURAL_SIGNAL].values).astype(np.float32)
assert activity.shape[1] == len(ophys_ts), \
    f'{oeid}: {NEURAL_SIGNAL} length {activity.shape[1]} != {len(ophys_ts)} ophys frames'
```

iii. Step 6 "Neural-signal decision": dF/F is "the endpoint of the whitepaper's processing chain
(methods.txt §DATA PROCESSING) … already computed in the released data (nothing for us to
recompute)". Piet et al. used detected calcium events, but "at the 32 ms ophys bin required here,
`events` is 99.77 % zeros (≈0.6 non-zero samples per neuron per trial), which carries almost no
per-timepoint information; this is exactly the case the task description covers with 'except where
… training a neural decoder require otherwise'". The choice was also made empirically: the same 11
sessions were converted three times and decoded, and dF/F beat both event variants on all five
outputs (e.g. image_identity 0.347 vs 0.248/0.178).

## 2-b. How is the `neural` data processed?

i. Essentially no processing: the per-cell object column is stacked into an `(n_neurons, n_frames)`
matrix, cast to `float32`, sliced per trial and stored contiguous. No z-scoring, smoothing,
normalisation, detrending or baseline subtraction is added. Per-neuron z-scoring was tested and
explicitly rejected. The length of each trace is asserted to equal the number of ophys timestamps,
and every stored trial is asserted finite.

ii.
```python
activity = np.vstack(ev[NEURAL_SIGNAL].values).astype(np.float32)
...
neural.append(activity[:, i0:i1].copy())
...
sess_neural.append(np.ascontiguousarray(act, dtype=np.float32))
...
assert np.isfinite(data['neural'][s][k]).all()
```

iii. Step 12 "Additional experiment: per-neuron z-scoring (rejected)": z-scoring each neuron's dF/F
within a session before PCA was worse on 4 of 5 outputs "and it is a transform neither reference
describes. **Not applied**; the shipped data is the Allen pipeline's dF/F as-is (itself already
normalised by each cell's baseline fluorescence)". The full Allen chain (motion correction,
segmentation, ROI filtering, demixing, neuropil subtraction, ΔF/F with a 600 s median-filter
baseline, detrending) is documented in `metadata['neural_signal']` as already applied upstream.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering. Every cell in `dff_traces` becomes a row of the neural matrix
(28,821 cells over 165 sessions, 6–666 per session). Sessions with zero cells would be skipped, and
sessions with very few cells (minimum 6) were deliberately kept.

ii.
```python
# no cell-level filter anywhere; the only cell-count guard is:
if len(ev) == 0:
    return {'oeid': oeid, 'skip': 'no cells'}
cell_specimen_ids = np.asarray(ev.index.values)
```
and in the metadata:
```python
'curation': (... 'no additional neuron filtering (the Allen pipeline\'s ROI '
             'filtering is already applied upstream and cell_specimen_table.valid_roi is '
             'all True in the released data).'),
```

iii. Step 3: "ROI filtering (unions, duplicates, motion-border, dendrites, too small/narrow/dim) and
removal of ROIs with non-positive demixed traces are performed *upstream*; the released
`events`/`dff_traces` contain only valid cells → **no additional neuron filtering** (verified
`valid_roi` is all True)." Step 10 sanity check B2 confirmed neurons/session equals the row count of
`ophys_cells_table.csv` for all 165 sessions. Step 10 Check 5 kept the 6-neuron session because
"the decoder's projection layer handles `nneurons < npcs`".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Everything lives on the ophys clock: `ds.ophys_timestamps` is the common time base and the trial
is the set of ophys frames in `[trials.start_time, trials.stop_time)`, located with
`np.searchsorted(side='left')`. The alignment event recorded in the metadata is the trial start
(`off_start = 0.0`, `off_end = None` because trials are variable length). All other streams
(running, pupil, image identity, image change) are put on the same frame grid before slicing, so a
single `i0:i1` index pair aligns every stream with the neural data by construction.

ii.
```python
i0, i1 = frame_slice(ophys_ts, tr.start_time, tr.stop_time)
neural.append(activity[:, i0:i1].copy())
img_tr.append(img_local[i0:i1].copy())
chg_tr.append(is_change_ts[i0:i1].copy())
run_tr.append(running[i0:i1].copy())
pup_tr.append(pupil[i0:i1].copy())
...
'temporal_alignment_event': (
    'Trial start (allensdk trials.start_time) of each go/catch trial; all data '
    'streams are resampled onto the two-photon ophys frame timestamps '
    '(ds.ophys_timestamps), which is the common clock for this dataset.'),
'off_start': 0.0,
'off_end': None,
```

iii. Step 3 processing details: "All data streams are already on a **common clock** (the sync board,
whitepaper §DATA SYNCHRONIZATION); AllenSDK returns every stream with timestamps in that clock, so
alignment = resampling onto `ophys_timestamps`." The task itself requires "Temporally align based on
ophys timestamp". Step 10 Check 5 verified the half-open convention: "trial frames are
`start_time <= t < stop_time` (half-open, `searchsorted(..., 'left')`); consecutive trials never
share a frame", and sanity check A1 reproduced whole trials with `np.allclose` from a freshly
loaded NWB.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native two-photon frame interval — **32.319 ms** (≈30.94 Hz; `metadata['ophys_frame_rate']`
is 31.0 Hz for these single-plane experiments). **No rebinning, resampling or smoothing of the
neural data is applied.** `time_bin_size` is stored as the mean of the per-session median frame
intervals in ms, and the per-session spread (32.310–32.331 ms) is stored alongside it in
`time_bin_size_range_ms`. Trials have variable numbers of bins but identical bin size.

ii.
```python
'dt': float(np.median(np.diff(ophys_ts))),
...
dt_all = np.array([s['dt'] for s in sessions])
'time_bin_size': float(np.mean(dt_all) * 1000.0),   # ms
'time_bin_size_range_ms': [float(dt_all.min() * 1000), float(dt_all.max() * 1000)],
'sampling_rate_hz': float(1.0 / np.mean(dt_all)),
```

iii. Step 5 Key Decision 5: "Time base = ophys frames, no re-binning: the task says to align on
ophys timestamps. Bin size is the 2-photon frame interval, constant to ±0.01 ms across the 168
sessions." Step 4 reconciled the nominal 31 Hz metadata with the measured 32.319 ms median Δt
("Consistent; use the measured mean Δt for `time_bin_size`"). Step 10 Check 3 notes that Piet et al.
additionally aggregate into 750 ms image-presentation intervals for their per-flash analyses, which
was intentionally not done here because the task requires ophys-timestamp alignment.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `ds.stimulus_presentations`, restricted to the change-detection block
(`stimulus_block_name` contains `'change_detection'`, which drops the 5-min grey-screen blocks and
the natural-movie block) and to non-omitted flashes; the columns used are `image_name`,
`start_time`, `end_time` and `omitted`. It is **not** derived from the trials table's
`initial_image_name` / `change_image_name`. The variable has 17 categories: `grey` (code 0) plus the
16 distinct natural images across image sets A and B.

ii.
```python
block = stim[stim.stimulus_block_name.str.contains('change_detection', na=False)]
shown = block[~block.omitted.astype(bool)]          # omitted flash == grey screen

image_names = sorted(set(shown.image_name.unique()))
code = {name: i + 1 for i, name in enumerate(image_names)}

starts = shown.start_time.values
ends = shown.end_time.values
i0 = np.searchsorted(ophys_ts, starts, side='left')
i1 = np.searchsorted(ophys_ts, ends, side='left')
codes = shown.image_name.map(code).values.astype(np.int16)
for a, b, bc, c, ch in zip(i0, i1, i1c, codes, changes):
    img_local[a:b] = c
```

iii. Step 5 Key Decision 6: "**Image identity during grey screen = a separate `grey` class**, and
**omitted flashes count as grey** (the screen is literally grey during an omission). The task says
'image identity *of the image presented during the non-grey screen*'." Using the flash table rather
than the trial table means the label tracks the actual stimulus frame by frame (250 ms on, 500 ms
grey) instead of being held constant across a trial. The resulting grey fraction, 0.669, was
checked against the expected 500/750 ms = 0.667 (Step 9 consistency table).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Per session, a per-ophys-frame `int16` vector is built once: 0 everywhere, then set to the
session-local image code on the frames of each non-omitted flash. After all sessions are read, a
global vocabulary (`grey` + the 16 sorted image names) is built and each session's local codes are
remapped to global codes through a small lookup table, so a given integer means the same image in
every session. Stored as `output[...][0]`.

ii.
```python
image_names_global = sorted({n for s in sessions for n in s['image_names']})
image_values = [GREY_LABEL] + image_names_global
global_code = {n: i + 1 for i, n in enumerate(image_names_global)}
...
lut = np.zeros(len(s['image_names']) + 1, dtype=np.int64)   # 0 stays grey
for i, name in enumerate(s['image_names']):
    lut[i + 1] = global_code[name]
...
out[0] = lut[s['image_local'][k]]
```

iii. The global vocabulary is needed because the two image sets (A and B) differ across sessions:
Step 2 records "A = im061,062,063,065,066,069,077,085 (88 sessions); B = im000,031,035,045,054,073,
075,106 (80 sessions)", i.e. 8 images per session but 16 across the dataset. Step 9's consistency
table checks the resulting distribution: "grey 0.669 + 16 images at 0.020–0.021 each", consistent
with 8 images sharing ~1/3 of the timepoints in each of two half-datasets.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The image-identity vector is built on the full-session ophys-frame grid (flash boundaries mapped
with `np.searchsorted(ophys_ts, ..., side='left')`, so a flash owns frames with
`start_time <= t < end_time`), and is then sliced with the *same* `i0:i1` indices used for the
neural matrix. Alignment is therefore exact by construction, to within half an ophys frame (16 ms).

ii.
```python
i0 = np.searchsorted(ophys_ts, starts, side='left')
i1 = np.searchsorted(ophys_ts, ends, side='left')
...
img_local[a:b] = c
...
i0, i1 = frame_slice(ophys_ts, tr.start_time, tr.stop_time)
neural.append(activity[:, i0:i1].copy())
img_tr.append(img_local[i0:i1].copy())
```

iii. Step 7 plot review: "**Image identity** steps up exactly on the cyan flash spans drawn straight
from `ds.stimulus_presentations` and returns to 0 (grey) in the inter-stimulus intervals and on
omitted flashes — no lag, no off-by-one frame." Step 10 sanity check A3a rebuilt image identity
frame-by-frame from a freshly loaded NWB without importing the conversion code: "PASS (15/15, 100 %
of frames agree)".

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` column of `ds.stimulus_presentations` (change-detection block, non-omitted
flashes) together with that flash's `start_time`/`end_time`. It is *not* derived from
`trials.change_time` / `trials.go`. Because the SDK flags sham changes with a separate
`is_sham_change` column, catch trials automatically contain no change frames.

ii.
```python
changes = shown.is_change.values.astype(bool)
...
for a, b, bc, c, ch in zip(i0, i1, i1c, codes, changes):
    img_local[a:b] = c
    if ch:
        is_change[a:bc] = 1
...
out[1] = s['is_change'][k]
```

iii. Step 5 variable mapping: "same table, `is_change == True` → `output[s][k][1]` image change …
catch trials are all-zero (sham change = no image change)". Step 4 reconciled `is_change` counts
with `go` counts: "`is_change` also flags the change on **auto-rewarded** trials (3.8/session).
229.7 − 225.9 = 3.8 ✓ exactly", and those changes fall outside every go/catch trial window. Step 10
sanity checks C1/C2 verified "every go trial has exactly one `image_change` episode (37,143 go
trials, 0 anomalies)" and "no catch trial contains an image change (0 offenders)".

## 4-b. What processing is involved in computing `output` *Image change*?

i. None beyond the vectorised flash→frame mapping: a per-session `uint8` vector is set to 1 on the
ophys frames belonging to the changed flash, and is sliced per trial. Stored as `output[...][1]`
with value names `['no_change', 'change']`. The full-dataset rate is 0.0257 of timepoints.

ii.
```python
is_change = np.zeros(n, dtype=np.uint8)
...
    if ch:
        is_change[a:bc] = 1
...
'output_values': [ image_values, ['no_change', 'change'], ... ]
```

iii. Step 9 consistency check derives the expected rate independently: "≈0.87 go × 7.74/263 =
0.0256" vs measured 0.0257, and sanity check C3 verified "the change episode lasts 7–9 frames
(250 ms at 30.94 Hz); mean 7.74".

## 4-c. How is `output` *Image change* thresholded into categories?

i. Binary: 1 for exactly the ophys frames of the 250 ms presentation of the changed image
(`[flash.start_time, flash.end_time)`), 0 everywhere else, including the following grey ISI. A
750 ms alternative (Piet et al.'s "image presentation interval", i.e. flash + following grey) is
implemented behind `--change-window presentation_interval` and was tested but not shipped.

ii.
```python
CHANGE_WINDOW = 'flash'    # 'flash' = 250 ms image, 'presentation_interval' = 750 ms
...
if CHANGE_WINDOW == 'presentation_interval':
    all_starts = block.start_time.values
    nxt = np.searchsorted(all_starts, starts, side='right')
    chg_ends = np.where(nxt < len(all_starts),
                        all_starts[np.minimum(nxt, len(all_starts) - 1)], ends)
else:
    chg_ends = ends
i1c = np.searchsorted(ophys_ts, chg_ends, side='left')
...
    if ch:
        is_change[a:bc] = 1
```

iii. Step 5 Key Decision 7: "Image change = 1 for the 250 ms presentation of the changed image — the
window 'right after a change in image identity', consistent with how image identity is coded."
Step 12 Check 1 tested the alternative empirically: "Re-converted 40 sessions with the change
labelled over the full 750 ms image-presentation interval … Result: image_change **0.6174** vs
**0.6462** for the 250 ms flash window — the shipped definition is the better one, and it is also
the one consistent with how image identity is coded."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: built on the full-session ophys-frame grid with
`np.searchsorted` on the same flash boundaries, then sliced with the same trial `i0:i1` indices as
the neural matrix.

ii.
```python
i0 = np.searchsorted(ophys_ts, starts, side='left')
i1c = np.searchsorted(ophys_ts, chg_ends, side='left')
...
chg_tr.append(is_change_ts[i0:i1].copy())
```

iii. Step 7 plot review: "**Image change** is 1 exactly on the flash that contains
`trials.change_time` (green line) and 0 everywhere else; on the catch trial it is 0 for the whole
trial while the green sham-change line still falls on a flash of the *unchanged* image." Step 12
explicitly ruled out misalignment as the cause of the modest 0.62 decoding accuracy, citing the
frame-by-frame rebuild (100 % agreement) and the verification plots.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `ds.running_speed` — its `timestamps` and `speed` columns (cm/s, 60 Hz), which the Allen SDK has
already z>10-transient-cleaned and 10 Hz low-pass filtered in `running_processing.py`.

ii.
```python
def resample_running(ds, ophys_ts):
    """Running speed (cm/s) linearly interpolated onto the ophys timestamps."""
    rs = ds.running_speed
    t = rs.timestamps.values.astype(np.float64)
    v = rs.speed.values.astype(np.float64)
```

iii. Step 5 variable mapping: "`ds.running_speed.speed` (60 Hz) → `output[s][k][2]` running-speed
bin … `running_processing.py` (already 10 Hz low-pass filtered)". Step 10 Check 5 kept extreme
speeds (up to ~100 cm/s) because "the SDK already removes z>10 transients and 10 Hz low-passes
(whitepaper §BEHAVIOR)".

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Non-finite speed samples are dropped (none occur in this dataset), then the trace is linearly
interpolated onto the ophys timestamps with `np.interp` (edge values held constant outside the
recorded range, so no NaNs are introduced) and cast to `float32`. No smoothing or rectification is
added. Discretisation happens later, globally.

ii.
```python
    good = np.isfinite(v)
    if not good.all():                      # not observed in this dataset, but be safe
        t, v = t[good], v[good]
    return np.interp(ophys_ts, t, v).astype(np.float32)
```

iii. Step 2 recorded "running_speed: 270,257 rows @60 Hz, no NaNs", and that all streams fully cover
the change-detection block in every session ("no session has `stream_t0 > block_t0` or
`stream_t1 < block_t1`"), so edge-holding extrapolation is never exercised inside a trial. Step 6
lists `np.interp` instead of pandas resampling as one of the deliberate speed-ups.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile (quintile) bins whose four interior edges are computed **globally** over
every converted trial timepoint of all 165 sessions, then applied with `np.digitize`. Edges:
[0.043, 4.403, 22.636, 36.295] cm/s over a range of −23.98…99.92 cm/s; the resulting dataset-wide
distribution is exactly [0.2, 0.2, 0.2, 0.2, 0.2]. The edges are stored in the metadata and the code
asserts they are strictly increasing.

ii.
```python
def quantile_edges(values, nbins=N_QUANTILE_BINS):
    """Interior edges of `nbins` equal-percentile bins."""
    qs = np.arange(1, nbins) / nbins
    return np.quantile(values, qs)

def digitize(x, edges):
    return np.digitize(x, edges, right=False).astype(np.int64)
...
all_run = np.concatenate([np.concatenate(s['running']) for s in sessions])
run_edges = quantile_edges(all_run)
assert np.all(np.diff(run_edges) > 0), 'degenerate running-speed quantiles'
...
out[2] = digitize(s['running'][k], run_edges)
```

iii. Step 5 Key Decision 8: "Quintile bins for running/pupil are computed globally over all trial
timepoints of all included sessions, so that the class labels mean the same thing in every session
and the dataset-level distribution is exactly 20 % per bin." The task specifies "discretized into
five equal percentile bins". Step 7 noted the failure mode that motivates the global choice: in a
2-session sample containing one non-runner the edges collapse to ±0.2 cm/s, whereas "with the full
dataset the edges are set by the many running sessions".

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The speed trace is interpolated onto `ds.ophys_timestamps` for the whole session *before* trial
segmentation, and then sliced with the same `i0:i1` indices as the neural matrix, so both share the
frame grid exactly.

ii.
```python
running = resample_running(ds, ophys_ts)
...
i0, i1 = frame_slice(ophys_ts, tr.start_time, tr.stop_time)
neural.append(activity[:, i0:i1].copy())
run_tr.append(running[i0:i1].copy())
```

iii. Step 10 Check 3 (c): the whitepaper's §DATA SYNCHRONIZATION says all streams are recorded on a
single 100 kHz sync board and the SDK returns every stream on that clock, so resampling onto
`ophys_timestamps` is the alignment. Step 7 plot review: "the resampled ophys-rate trace lies on top
of the raw 60 Hz / 30 Hz trace with no visible shift." Sanity check A3c independently recomputed
`digitize(np.interp(t, raw_t, raw_v), edges)` from a freshly loaded NWB: PASS 15/15.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `ds.eye_tracking` — its `timestamps` and `pupil_area` columns (30 Hz). The AI verified in Step 2
that the SDK sets `pupil_area` to NaN exactly on `likely_blink` frames, so it filters on
`isfinite(area) & (area > 0)` rather than reading `likely_blink` directly. Diameter is derived as
`2*sqrt(area/pi)`. Sessions whose eye-tracking table is missing, empty, or has fewer than 2 valid
samples return `None` and the whole session is dropped.

ii.
```python
def resample_pupil(ds, ophys_ts):
    try:
        eye = ds.eye_tracking
    except Exception:
        return None
    if eye is None or len(eye) == 0:
        return None
    t = eye.timestamps.values.astype(np.float64)
    area = eye.pupil_area.values.astype(np.float64)
    good = np.isfinite(area) & (area > 0)
    if good.sum() < 2:
        return None
    diam = 2.0 * np.sqrt(area[good] / np.pi)
    return np.interp(ophys_ts, t[good], diam).astype(np.float32)
```

iii. Step 2: "`pupil_area` NaN exactly where `likely_blink` (mean 5.5 % of frames, median 3.0 %,
max 29.6 %). **3 of 168 sessions have an empty eye-tracking table.**" Step 1 recorded that the SDK's
non-`_raw` eye-tracking columns are NaN-masked at blinks. Step 5 mapping note: the area→diameter
conversion is a monotone transform, so "bins identical whether computed on area or diameter".

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples (NaN/non-positive area) are removed; area is converted to diameter
`d = 2*sqrt(A/pi)`; the remaining samples are linearly interpolated with `np.interp` onto the ophys
timestamps, which simultaneously bridges the blink gaps and resamples, holding the edge values
constant outside the recorded range (so every ophys frame receives a value and T matches the neural
data). Cast to `float32`. Discretised later, globally.

ii.
```python
good = np.isfinite(area) & (area > 0)
diam = 2.0 * np.sqrt(area[good] / np.pi)
return np.interp(ophys_ts, t[good], diam).astype(np.float32)
```

iii. Step 5 Key Decision 9: "`pupil_area` is NaN exactly where `likely_blink`; linearly interpolate
across those gaps (edges held constant) rather than dropping timepoints, so trials keep a common T
with the neural data." Step 10 Check 5 left extreme pupil values in: "dataset max 480 px vs typical
session max ~120–160 px; spot-checked 10 sessions: `>2×` the session median occurs in ≤0.03 % of
frames … percentile binning is insensitive to a handful of outliers, and the SDK provides no
additional validity flag beyond `likely_blink`."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Exactly the same scheme as running speed: four interior quintile edges computed globally over
every converted trial timepoint of all 165 sessions, applied with `np.digitize`. Edges
[75.42, 84.94, 94.18, 107.17] px over a 16.93…480.86 px range; dataset-wide distribution exactly
[0.2 ×5]. Strict monotonicity of the edges is asserted.

ii.
```python
all_pup = np.concatenate([np.concatenate(s['pupil']) for s in sessions])
pup_edges = quantile_edges(all_pup)
assert np.all(np.diff(pup_edges) > 0), 'degenerate pupil quantiles'
...
out[3] = digitize(s['pupil'][k], pup_edges)
...
'pupil_diameter_bin_edges_px': pup_edges.tolist(),
```

iii. Same justification as running speed (Step 5 Key Decision 8): global edges make a bin label mean
the same thing in every session and give exactly 20 % per bin dataset-wide, which is what "five
equal percentile bins" asks for. The `--show-processing` panels plot value→bin and show "a monotone
step function whose steps sit exactly on the printed quintile edges".

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same mechanism as running speed: interpolated onto `ds.ophys_timestamps` for the whole session
before segmentation, then sliced with the same `i0:i1` trial indices as the neural matrix.

ii.
```python
pupil = resample_pupil(ds, ophys_ts)
if pupil is None:
    return {'oeid': oeid, 'skip': 'no eye tracking'}
...
pup_tr.append(pupil[i0:i1].copy())
```

iii. Same sync-board justification as running speed (Step 10 Check 3c). Step 7 plot review: "blink
gaps (grey, NaN) are bridged smoothly" and the resampled trace overlays the raw 30 Hz trace with no
visible shift. Sanity check A3d recomputed the pupil bin from `pupil_area` + blink mask on freshly
loaded NWBs: PASS 15/15.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of `ds.trials`: `hit`, `miss`, `false_alarm`,
`correct_reject`, read for the selected go/catch rows only. The code asserts that exactly one is
True per selected trial, and takes `argmax` to get a code in 0–3.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome_flags = sel[OUTCOME_NAMES].values.astype(bool)
assert (outcome_flags.sum(axis=1) == 1).all(), \
    f'{oeid}: trials without exactly one outcome'
outcomes = np.argmax(outcome_flags, axis=1).astype(np.int64)
```

iii. Step 5 variable mapping: "`ds.trials.hit/miss/false_alarm/correct_reject` → `output[s][k][4]`
trial outcome … exactly one is True for every go/catch trial (verified)". Step 10 sanity check B3
compared per-session hit/miss/FA/CR counts against `behavior_session_table`: PASS 165/165
(13,569 / 23,574 / 814 / 4,513 trials overall).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The integer code (0–3, ordered `hit, miss, false_alarm, correct_reject`) is broadcast across all
T timepoints of the trial so it can be packed into the single `(5, T)` output array per trial; the
value is constant within a trial, i.e. a static per-trial label represented as a constant time
series. The kept-trial subsetting (`outcomes[keep]`) keeps outcomes aligned with the kept trials.

ii.
```python
'outcome': outcomes[keep] if len(keep) else np.zeros(0, dtype=np.int64),
...
out = np.empty((5, T), dtype=np.int64)
...
out[4] = s['outcome'][k]                    # static, broadcast over T
...
'output_values': [ ..., OUTCOME_NAMES ],
```

iii. Step 5 Key Decision 11: "Outputs are packed as a single `(5, T)` integer array per trial (the
static trial outcome is broadcast across the trial) because the format requires one array per
trial." Step 12 Check 3 discusses the consequence — trial_outcome has the largest train/val gap
(1.48×) because "every one of a trial's ~263 timepoints carries the same label" — and concludes it
is ordinary overfitting, not leakage, since the split is by trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several explicit policies:
- **Blinks** (NaN `pupil_area`, 5.5 % of eye frames on average): bridged by linear interpolation
  rather than dropped, so T stays aligned with the neural data.
- **Missing eye tracking entirely** (3 sessions): the session is skipped and printed, because pupil
  diameter is a required output and cannot be fabricated.
- **Sessions with no cells**: skipped.
- **Trials with <2 ophys frames of coverage**: skipped (never triggered).
- **Sessions with <2 usable trials**: skipped (never triggered).
- **Extrapolation beyond a behavioural stream's range**: `np.interp` holds the edge value rather
  than producing NaN, so no NaN ever reaches the discretiser.
- **Unexpected data**: assertions on trace length vs. timestamp count, on no aborted/auto-rewarded
  trial being selected, on exactly one outcome flag per trial, on non-degenerate quantile edges, and
  a final structural/finiteness pass in `run_summary`.
- **Unexpected exceptions**: caught per session in the worker (keeping the pool alive), reported
  with a full traceback at the end, and then the run **aborts** (`sys.exit(1)`) rather than silently
  shipping a partial dataset.

ii.
```python
def extract_session_safe(oeid):
    try:
        return extract_session(oeid)
    except Exception as exc:                # keep the pool alive, report at the end
        import traceback
        return {'oeid': oeid, 'skip': f'ERROR {exc!r}', 'traceback': traceback.format_exc()}
...
errors = [r for r in results if ... str(r.get('skip')).startswith('ERROR')]
for r in errors:
    print(f'ERROR on {r["oeid"]}:\n{r.get("traceback")}')
if errors:
    sys.exit(1)
skipped = [r for r in results if r['skip']]
for r in skipped:
    print(f'  SKIPPED {r["oeid"]}: {r["skip"]}')
...
assert np.isfinite(data['neural'][s][k]).all()
```

iii. Step 9: "No data was lost beyond the three documented eye-tracking drops: every selected
go/catch trial of every kept session produced a trial (`n_trials_kept == n_trials_selected` for all
165 sessions), and every cell in `dff_traces` became a row of the neural matrix." Step 10 Check 5
enumerates each edge case examined (trials with no ophys coverage, off-by-one at trial and flash
boundaries, `is_change` on auto-rewarded trials, omitted flashes, blinks, sessions without eye
tracking, extreme pupil/running values, 6-neuron sessions, sessions missing an outcome class) with
the finding and the action taken for each.

## 9-a. What are the most time-consuming steps of the code?

i. Reading/decompressing the NWB files through `get_behavior_ophys_experiment` dominates
(I/O + decompression bound); the AI measured 2.2 s/session serial and reduced it to 0.35 s/session
wall with a 16-process pool — 58.7 s of the 71.5 s total run. The second cost is writing the 8.73 GB
pickle (7.9 s). Binning + assembly is 1.2 s. Timing is printed at each phase, as the instructions
require.

ii.
```python
t0 = time.time()
ds = get_cache().get_behavior_ophys_experiment(oeid)
t_load = time.time() - t0
...
print(f'Phase 1 (read+extract): {t_read:.1f}s total, '
      f'{t_read / max(1, len(oeids)):.2f}s per session (wall, {args.workers} workers)')
...
print(f'Phase 2+3 (bin+assemble): {time.time() - t0:.1f}s')
print(f'Wrote {args.outfile} ({os.path.getsize(args.outfile) / 1e9:.2f} GB) in {time.time() - t0:.1f}s')
```

iii. Step 6: "`multiprocessing.Pool(16)` over sessions (NWB decompression is the bottleneck):
2.2 s/session serial → 0.55 s/session wall". Step 7's table projected ≈3–5 min for the full run;
Step 9 reports the actual 71.5 s, "well inside the Step-7 estimate … so no re-optimisation was
needed".

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python loops remain, all cheap relative to I/O:
- `build_stimulus_timeseries`: the flash-boundary lookup is fully vectorised
  (`np.searchsorted` over all ~4,800 flashes at once), but the *fill* is still a Python loop over
  flashes (`for a, b, bc, c, ch in zip(...)`) doing tiny slice assignments — ~790k iterations over
  the dataset. It could be replaced by a cumulative-sum / `np.repeat` construction.
- `extract_session`: `for k, (_, tr) in enumerate(sel.iterrows())` — pandas `iterrows()` per trial;
  the `start_time`/`stop_time` columns could be `searchsorted` once per session in one vectorised
  call, and `iterrows()` is the slowest pandas row iterator.
- The assembly loop in `main()` over sessions × trials, which allocates one `(5, T)` array per
  trial.
The AI explicitly identified and avoided the one loop that would have mattered asymptotically —
per-trial searching over the flash table, which would have been O(n_trials × n_flashes).

ii.
```python
# vectorised boundary lookup ...
i0 = np.searchsorted(ophys_ts, starts, side='left')
i1 = np.searchsorted(ophys_ts, ends, side='left')
# ... but a per-flash Python fill loop
for a, b, bc, c, ch in zip(i0, i1, i1c, codes, changes):
    img_local[a:b] = c
    if ch:
        is_change[a:bc] = 1
...
for k, (_, tr) in enumerate(sel.iterrows()):
    i0, i1 = frame_slice(ophys_ts, tr.start_time, tr.stop_time)
```

iii. Step 6 "Code inefficiencies identified": "Per-trial `np.searchsorted` over the flash table would
be O(n_trials · n_flashes); instead the per-frame image-identity / change vectors are built **once
per session** with a single vectorised `searchsorted` over all 4,800 flashes, then sliced per trial."
The residual per-flash fill loop is not called out in the notes, but at 71.5 s total runtime it is
not a bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. Little is repeated in the conversion path: each NWB is opened exactly once per run, all six
streams are read from that one object, and the extracted per-trial arrays are reused for the global
quantile computation and for final assembly without re-reading. Repetitions that do exist:
- `--show-processing` re-opens each plotted session through the SDK and re-reads `dff_traces`,
  `stimulus_presentations`, `running_speed`, `eye_tracking` and `trials` — deliberate, so the plot
  is an independent check rather than a replay of the converted arrays (2 sessions only).
- Every per-trial slice is `.copy()`-ed in the worker and then shipped back through the
  multiprocessing pipe, so the ~8 GB of trial data is pickled and unpickled once between processes;
  a shared-memory or return-the-slices-lazily design would avoid that round trip.
- `all_run` / `all_pup` are materialised as full concatenated copies (11.2 M floats each) to compute
  four quantile edges.
- `sel.loc[s['oeid'], ...]` is re-queried per session when building `session_info` (negligible).

ii.
```python
def show_processing(info, run_edges, pup_edges, image_names_global, outfile):
    ds = get_cache().get_behavior_ophys_experiment(oeid)      # re-opened on purpose
    ...
    pop = np.vstack(tbl[NEURAL_SIGNAL].values).astype(np.float32)[:, m].mean(axis=0)
...
neural.append(activity[:, i0:i1].copy())                      # copied, then pickled to parent
...
all_run = np.concatenate([np.concatenate(s['running']) for s in sessions])
all_pup = np.concatenate([np.concatenate(s['pupil']) for s in sessions])
```

iii. Step 6: "Re-reading the NWB for each stream — avoided; the `BehaviorOphysExperiment` object is
opened once", and "`show_processing()` re-opens the session through the SDK and overlays the
converted trial data on the raw SDK tables, so the plots are an independent check rather than a
replay of the conversion."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest amounts:
- The image-identity, image-change, running and pupil timeseries are built for the **entire session**
  (~140,000 ophys frames) although only the go/catch trial windows (~11.2 M of ~23 M frames
  dataset-wide, i.e. roughly half) are kept. This is the deliberate trade for vectorising the flash
  mapping.
- Outputs are stored as `int64` `(5, T)` arrays although every value fits in `int8`; this costs
  ~450 MB of the 8.73 GB pickle for no downstream benefit (the human reference uses `int8`).
- `input` is materialised as a real `(0, T)` `float32` array per trial rather than a single shared
  empty array — harmless but 42,470 unnecessary allocations for a decoder that has no inputs.
- Per-session bookkeeping that the decoder never reads: `cell_specimen_ids`, `n_go_trials`,
  `n_catch_trials`, `equipment_name`, `load_seconds`/`total_seconds`, and the whole
  `metadata['session_info']` list.
- `np.ascontiguousarray(act, dtype=np.float32)` at assembly time is a no-op on an array that is
  already contiguous float32.
- The unused `--change-window presentation_interval` branch computes `chg_ends`/`i1c` on every run
  even in the default `flash` mode (`i1c` is then identical to `i1`).
- `--show-processing` re-loads and re-stacks a full session's dF/F purely for plotting.

ii.
```python
out = np.empty((5, T), dtype=np.int64)        # int8 would suffice
sess_in.append(np.zeros((0, T), dtype=np.float32))
sess_neural.append(np.ascontiguousarray(act, dtype=np.float32))
...
i1c = np.searchsorted(ophys_ts, chg_ends, side='left')   # == i1 in default mode
...
'cell_specimen_ids': s['cell_specimen_ids'],
```

iii. The notes do not flag these as waste; they justify the extra metadata as documentation
(Step 13: `session_info` records experience level and image set per session so the novelty
restriction Piet et al. use "can be reapplied downstream"), and the whole-session timeseries
construction is justified in Step 6 as the vectorisation that avoids O(n_trials × n_flashes) work.
Total runtime (71.5 s) and file size (8.73 GB, dominated by the float32 neural data) were both
judged acceptable, so none of this was optimised further.
