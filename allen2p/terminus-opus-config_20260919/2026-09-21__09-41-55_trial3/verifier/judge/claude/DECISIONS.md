# Decisions

*Documentation of the decisions made by the agentic AI system (`terminus-2`, `claude-opus-5`) in
`/app/convert_data.py`, as justified in `/app/CONVERSION_NOTES.md`, `/app/README.md` and the agent
trajectory `/logs/agent/trajectory.json`.*

---

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. **Decisions**

- Data are loaded exclusively through the AllenSDK `VisualBehaviorOphysProjectCache`, opened in
  **offline mode** with `from_local_cache(cache_dir='/app/data')` (not `from_s3_cache`).
- The SDK metadata table lists 1936 experiments, but only 284 NWB files exist locally. The AI
  therefore **globs the local NWB directory**, parses the `ophys_experiment_id` out of each
  filename, and intersects that id list with `get_ophys_experiment_table()`. This makes the
  locally present file set the definition of "all the data" and guarantees no attempt is ever
  made to fetch an absent experiment.
- Two selection filters are then applied to that local experiment table:
  1. `~passive` — only active (behaving) sessions (284 → 202 experiments);
  2. `session_type ∈ {OPHYS_1_images_A, OPHYS_3_images_A}` — only the **familiar** image set
     (202 → 110 experiments).
- Experiments (imaging planes) are grouped into sessions by `ophys_session_id`, and each session
  is loaded with `bc.get_behavior_ophys_experiment(exp_id)`, one call per plane. Sessions are
  processed in parallel with an 8-process `multiprocessing.Pool`.
- Both project codes present locally are kept (`VisualBehavior`, 239 experiments, Scientifica
  single-plane 31 Hz; and `VisualBehaviorMultiscope`, 45 experiments, Mesoscope 11 Hz).
- Final funnel (documented in CONVERSION_NOTES Step 9): 284 local NWB files → 202 active → 110
  familiar-active (92 sessions, 38 mice) → 91 sessions (1 dropped for missing eye tracking) →
  22,179 trials, 14,669 neurons.

ii. **Code snippets**

```python
CACHE_DIR = '/app/data'
NWB_DIR = os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0', 'behavior_ophys_experiments')

def get_cache():
    import allensdk.brain_observatory.behavior.behavior_project_cache as bpc
    return bpc.VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)

def local_experiment_table():
    """Experiment (imaging-plane) metadata table restricted to the NWB files present locally."""
    ids = sorted(int(re.search(r'(\d+)\.nwb', f).group(1))
                 for f in glob.glob(os.path.join(NWB_DIR, '*.nwb')))
    et = get_cache().get_ophys_experiment_table()
    et = et.loc[et.index.intersection(ids)]
    return et

def select_experiments():
    et = local_experiment_table()
    et = et[~et.passive]                                     # active behavior only
    et = et[et.session_type.isin(FAMILIAR_SESSION_TYPES)]    # familiar image set only
    return et
```

```python
for eid in exp_ids:
    ds = bc.get_behavior_ophys_experiment(int(eid))
    ev = np.vstack(ds.events[signal].values).astype(np.float64)   # (ncells, nframes)
    ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
    ...
    if beh is None:
        # behavior streams are identical across simultaneously recorded planes
        beh = dict(trials=ds.trials.copy(), stim=ds.stimulus_presentations.copy(),
                   run=ds.running_speed.copy(),
                   eye=None if ds.eye_tracking is None else ds.eye_tracking.copy(),
                   behavior_session_id=ds.behavior_session_id)
```

iii. **Justification (from CONVERSION_NOTES / trajectory)**

- Step 1: `from_local_cache` is the offline equivalent of the `from_s3_cache` used by every
  tutorial; `from_local_cache(..., use_static_cache=True)` failed in this container, the
  non-static form works with the provided manifest.
- Step 2: "The project manifest lists 1936 experiments; **only 284 are present locally**, so the
  local file set defines the available dataset." Globbing the NWB directory is what makes the
  script robust to that.
- Step 5, data-selection rules: passive sessions (OPHYS_2/5) are excluded because "the lick spout
  is retracted, so there are no Go/Catch outcomes"; only familiar sessions are kept because
  (a) the paper states "we restricted our analysis to familiar stimuli"/"For neural analysis we
  used neurons recorded during familiar image set presentations", and (b) all familiar sessions
  use the *same* image set A, so `image_identity` is one consistent 8-way categorical output,
  whereas adding novel (image set B) sessions would create 8 extra mutually exclusive classes
  never seen in a familiar session.
- Step 4 discrepancy table: the paper's exact neural subset (Mesoscope + familiar, 9 mice) cannot
  be reproduced because only one Mesoscope mouse is present locally, so "we keep **all** locally
  available active familiar sessions (both rigs) and document the deviation."

---

## 1-b. How are the data split into subjects?

i. **Decisions** — Subjects are the unique `mouse_id` values of the selected experiment table,
carried through each session's metadata as a string and sorted to give a deterministic
`subjects` list; `subject_idx` indexes into it per session. 38 mice, 1–7 sessions each.

ii. **Code snippets**

```python
for sid, grp in et.groupby('ophys_session_id'):
    meta = grp.iloc[0][['mouse_id', 'cre_line', 'session_type', 'experience_level',
                        'equipment_name', 'project_code', 'image_set']].to_dict()
    meta['mouse_id'] = str(meta['mouse_id'])
```

```python
subjects = sorted({r['mouse_id'] for r in good})
subj_to_idx = {s: i for i, s in enumerate(subjects)}
...
subject_idx=np.array([subj_to_idx[r['mouse_id']] for r in good], dtype=np.int64),
```

iii. **Justification** — `mouse_id` is the SDK's canonical animal identifier (Step 1 table:
`experiment.metadata` provides `mouse_id`). Cast to `str` because the target format specifies
`subjects: list of str`. A per-session sanity check in `/app/cache/sanity_raw.py` re-reads
`metadata['mouse_id']` from the raw NWB and compares it to the stored subject (Step 10, Check 2:
"subject id … equal").

---

## 1-c. How are the data split into sessions?

i. **Decisions**

- One converted "session" = one `ophys_session_id`. The 3–7 imaging planes (separate NWB
  "experiments") of a Mesoscope session recorded **simultaneously** are merged into a single
  session by concatenating their neurons along the neuron axis.
- Behavioural streams (`trials`, `stimulus_presentations`, `running_speed`, `eye_tracking`) are
  read from the **first plane only**, since they are identical for all planes of a session
  (verified by the shared `behavior_session_id`).
- Each neuron keeps its own plane's `targeted_structure` for `brain_region_idx`, so one merged
  session can contribute both VISp and VISl neurons.
- Sessions are sorted by `ophys_session_id` (not by acquisition date).

ii. **Code snippets**

```python
sessions = []
for sid, grp in et.groupby('ophys_session_id'):
    meta = grp.iloc[0][[...]].to_dict()
    sessions.append((int(sid), list(grp.index.values), meta))
sessions.sort(key=lambda s: s[0])
```

```python
for p in planes:
    sums, counts = bin_sum_count(p['events'], p['ts'], edges)
    neural_blocks.append((sums / counts[None, :, :]).astype(np.float32))
    region_list += [p['structure']] * p['events'].shape[0]
neural = np.concatenate(neural_blocks, axis=0)       # (ncells, ntrials, NBINS)
```

iii. **Justification** — Step 5, rule 3: "For Mesoscope sessions the 3–7 simultaneously recorded
imaging planes … are **merged into one session**, since they are recorded simultaneously from the
same mouse and constitute one simultaneously recorded population." Step 10, Check 5 records the
edge case explicitly: "behaviour streams are read once (verified identical `behavior_session_id`),
and `brain_region_idx` keeps each plane's own structure (a 7-plane session contributes both VISp
and VISl neurons — verified in the sample)."

---

## 1-d. How are the data split into trials?

i. **Decisions**

- Trials come from the SDK's built-in `dataset.trials` table, selected as
  `(go | catch) & ~auto_rewarded` — i.e. Go and Catch trials only, aborted and auto-rewarded
  excluded, exactly as the task specification requires.
- Rather than using the trial's native `start_time → stop_time` span, each trial is cut as a
  **fixed window around `trials.change_time`**: `[-2.25 s, +3.75 s]`, divided into **8 bins of
  750 ms**. Every trial therefore has exactly T = 8 time bins.
- The window is asserted to lie strictly inside the trial's own `[start_time, stop_time]`, so
  trials never overlap and no window contains a second image change.

ii. **Code snippets**

```python
BIN_SIZE = 0.750          # s, time bin size = one image-presentation interval
OFF_START = -2.25         # s, relative to change_time (3 image intervals before the change)
OFF_END = 3.75            # s, relative to change_time (5 image intervals from the change)
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 8
```

```python
tr = beh['trials']
keep = (tr['go'].astype(bool) | tr['catch'].astype(bool)) & ~tr['auto_rewarded'].astype(bool)
tr = tr[keep]
assert tr['change_time'].notna().all(), 'missing change_time on a go/catch trial'
change_times = tr['change_time'].values.astype(np.float64)

# trial windows must lie inside the trial
assert np.all(change_times - tr['start_time'].values >= -OFF_START), 'window precedes trial start'
assert np.all(tr['stop_time'].values - change_times >= OFF_END), 'window exceeds trial stop'

offs = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
edges = change_times[:, None] + offs[None, :]         # (ntrials, NBINS+1)
centers = edges[:, :-1] + BIN_SIZE / 2.0
```

iii. **Justification** — Step 5: "Trials: `trials` rows with `(go | catch) & ~auto_rewarded` (per
the task specification; also matches the whitepaper/SDK practice of excluding aborted trials from
performance metrics and treating free-reward trials as non-task trials)." For the window: the AI
measured across all 202 active experiments that `change_time − start_time ∈ [2.79, 8.31] s` and
`stop_time − change_time ∈ [4.20, 4.35] s`, so `[-2.25, +3.75] s` "is the largest multiple of
750 ms that fits inside *every* trial". It explicitly rejected the reference SDK window
`[-4, +8] s` (`save_trial_response_df.py`) because "the reference window crosses trial boundaries
and contains other image changes, which would make trials overlap and would put several changes
inside one 'trial'" (Step 10, Check 3).

---

## 1-e. How are trials filtered based on quality controls?

i. **Decisions** — Four filters, in order:

1. **Trial type**: aborted and auto-rewarded trials excluded (`(go|catch) & ~auto_rewarded`);
   61.9 % of all raw trials are aborted, 0.6 % auto-rewarded.
2. **Whole sessions with no eye-tracking data are dropped** (1 familiar session, 805989030),
   because pupil diameter is a required output.
3. **Trials with any time bin containing no valid running-speed or pupil sample are dropped**
   (599 / 22,778 trials = 2.6 %). This is what removes long blink/tracking-loss periods, since
   pupil gaps > 1 s are deliberately left as NaN.
4. **Sessions left with fewer than 2 usable trials are skipped** (none occurred).

Plus hard assertions that fail loudly rather than silently corrupting data: `change_time` present
on every kept trial; the four outcome flags exactly partition the kept trials; the analysis window
lies inside the trial; every time bin contains ≥ 1 imaging frame.

ii. **Code snippets**

```python
if beh['eye'] is None or len(beh['eye']) == 0:
    return dict(session_id=session_id, skip='no eye tracking', timings=timings)
```

```python
outcome_mat = np.stack([tr[c].astype(bool).values for c in OUTCOME_NAMES], axis=1)
assert np.all(outcome_mat.sum(axis=1) == 1), 'trial outcomes do not partition trials'
```

```python
# ---------------- drop trials with missing behavior data ----------------
bad = (~np.isfinite(run_binned).all(axis=1)) | (~np.isfinite(pupil_binned).all(axis=1))
nbad = int(bad.sum())
good = ~bad
if good.sum() < 2:
    return dict(session_id=session_id, skip=f'<2 usable trials ({int(good.sum())})',
                timings=timings)
neural = neural[:, good, :]
image_identity = image_identity[good]
...
```

iii. **Justification** — Step 5, rule 6: "Sessions without eye tracking are dropped (1 familiar
session, 805989030), because pupil diameter is a required output." Step 10, Check 5: "pupil NaNs
(median 2.9 %, max 29.6 % of frames, runs up to ~14 s) are linearly interpolated only across gaps
≤ 1 s; longer gaps stay NaN and the affected trials are dropped (599 trials, 2.6 %). Trials are
also dropped if a bin contains no valid running or pupil sample at all." The `< 2 trials` rule
exists because the decoder requires at least two trials per session to evaluate performance.

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. **Decisions** — `neural` comes from the SDK's **detected calcium events**,
`BehaviorOphysExperiment.events['events']` (one event-magnitude value per ROI per ophys frame),
timestamped with `BehaviorOphysExperiment.ophys_timestamps`. dF/F (`dff_traces`) is **not** used,
and the smoothed `filtered_events` variant is not used by default (it is retained behind a
`--neural-signal filtered_events` flag).

ii. **Code snippets**

```python
ds = bc.get_behavior_ophys_experiment(int(eid))
ev = np.vstack(ds.events[signal].values).astype(np.float64)   # (ncells, nframes)
ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
assert ev.shape[1] == ts.size, 'events / timestamps length mismatch'
```

```python
ap.add_argument('--neural-signal', type=str, default='events',
                choices=['events', 'filtered_events'])
```

iii. **Justification** — Step 1 note: "Imaging data: **no dF/F computation needed** — the NWB
release contains detrended dF/F *and* detected calcium events. The analysis paper uses the
**detected calcium events** (`events`), so that is our neural stream." Step 3 quotes the paper
directly: "For all analysis of neural data we used the detected calcium events"; "analyses on
discrete calcium events that were regressed from the raw fluorescence traces". `filtered_events`
is rejected because the SDK documents it as smoothing applied "to smooth it for visualization".
The AI also ran an empirical comparison over 14 sessions (Step 9 table): accuracy differences
between `events` and `filtered_events` were "small and inconsistent in sign", so the
paper-consistent choice was kept.

---

## 2-b. How is the `neural` data processed?

i. **Decisions** — Two processing steps:

1. **Binning**: for every (neuron, trial, bin), the **mean** (not sum) of the per-frame event
   magnitude over all imaging frames whose timestamp falls inside the bin. Implemented
   vectorised with a cumulative sum plus `searchsorted`, computing all neurons × trials × bins in
   three array operations. Planes of a multi-plane session are binned separately (each with its
   own `ophys_timestamps`) and concatenated along the neuron axis.
2. **Per-neuron z-scoring within session** (`--normalize zscore`, the default): each neuron's
   binned values are centred and scaled by its own mean/SD computed over *all selected go/catch
   trials* of that session (i.e. before the behaviour-NaN trial drop).

ii. **Code snippets**

```python
def bin_sum_count(values, timestamps, edges_flat):
    v = np.atleast_2d(values).astype(np.float64)
    idx = np.searchsorted(timestamps, edges_flat)            # (ntrials, nbins+1)
    cs = np.concatenate([np.zeros((v.shape[0], 1)), np.cumsum(v, axis=1)], axis=1)
    sums = cs[:, idx[:, 1:]] - cs[:, idx[:, :-1]]            # (n, ntrials, nbins)
    counts = (idx[:, 1:] - idx[:, :-1]).astype(np.float64)   # (ntrials, nbins)
    return sums, counts
```

```python
for p in planes:
    sums, counts = bin_sum_count(p['events'], p['ts'], edges)
    assert counts.min() > 0, 'empty time bin (no imaging frame): bin size too small'
    neural_blocks.append((sums / counts[None, :, :]).astype(np.float32))
neural = np.concatenate(neural_blocks, axis=0)       # (ncells, ntrials, NBINS)

if normalize in ('std', 'zscore'):
    flat = neural.reshape(neural.shape[0], -1)
    mu = flat.mean(axis=1, keepdims=True)
    sd = flat.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    if normalize == 'zscore':
        flat = (flat - mu) / sd
    else:
        flat = flat / sd
    neural = flat.reshape(neural.shape).astype(np.float32)
```

iii. **Justification**

- *Mean, not sum*: Step 5, Key Decision 2 — "makes values comparable between 31 Hz and 11 Hz
  rigs" (the dataset mixes Scientifica single-plane and Mesoscope multi-plane recordings).
- *Z-scoring*: added as "revision 2" after the first full run (Step 9). In-code comment: "detected
  calcium events have cell-specific magnitudes spanning orders of magnitude, and the decoder
  initialises its projection from a raw SVD, so a handful of large-amplitude cells would otherwise
  dominate every principal component. Scaling is monotone per neuron, so it changes no event time
  or relative time course." Empirical ablation over 14 sessions
  (`none` → `std` → `zscore`): image_identity 0.427/0.564/0.556, image_change 0.650/0.671/0.696,
  running 0.286/0.399/0.443, pupil 0.300/0.378/0.389, outcome 0.264/0.290/0.317. Side benefit:
  the decoder's "all neural data is zero" warnings for 1,341 sparse trials disappear.
- Step 10, Check 3 flags this as a deliberate deviation from the paper ("the paper uses a random
  forest (scale-invariant), so it does not normalise … required because the provided decoder does
  a raw (un-centred) SVD + linear readout").

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. **Decisions** — **No additional neuron-level filtering.** Every ROI returned by
`experiment.events` is kept. The only neural-side quality assertions are structural: `events`
length must equal `ophys_timestamps` length, every bin must contain ≥ 1 imaging frame
(`counts.min() > 0`), and a post-hoc sanity check that all converted values are finite.

ii. **Code snippets**

```python
assert ev.shape[1] == ts.size, 'events / timestamps length mismatch'
...
assert counts.min() > 0, 'empty time bin (no imaging frame): bin size too small'
...
out['neural_finite'] = bool(np.isfinite(neural).all())
out['neural_nonneg'] = bool((neural >= 0).all())  # False by design if z-scored
```

iii. **Justification** — Step 1: `cell_specimen_table` "only contains `roi_valid = True` entries,
as invalid ROIs / non-cell segmented objects have been filtered out → ROI quality filtering is
already applied by the SDK". Step 4 and Step 10 Check 3: "QC-failed planes/sessions/containers
already removed from the release; cells matched across sessions but no additional filtering in the
paper. No extra filtering applied here" — verified empirically that all cells are `valid_roi ==
True`, with 0 all-zero cells and 0 NaNs across the 202 active experiments.

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. **Decisions**

- The alignment event is **`trials.change_time`** — the onset of the image-change flash on Go
  trials and of the sham change on Catch trials. `metadata['temporal_alignment_event']` records
  this, with `off_start = -2.25` and `off_end = +3.75`.
- Bin edges are built as `change_time + [-2.25, -1.5, …, +3.75]`, i.e. edges are placed relative
  to the alignment event in absolute sync-clock seconds, and each stream (events, running, pupil,
  stimulus) is binned onto **those same edges**, using each stream's own SDK-provided
  sync-clock timestamps. Alignment is therefore exact by construction and shared by all streams.
- Because `change_time` always coincides with a flash onset and 2.25 s = 3 × 750 ms, bin 3 is
  exactly the change interval and bins 0–2 are the three preceding image intervals.

ii. **Code snippets**

```python
offs = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
edges = change_times[:, None] + offs[None, :]         # (ntrials, NBINS+1)
centers = edges[:, :-1] + BIN_SIZE / 2.0
```

```python
# 1. every change time coincides with a flash onset (as asserted in the reference code)
out['change_times_are_flash_onsets'] = bool(
    np.all(np.min(np.abs(change_times[:, None] - fstart[None, :]), axis=1) < 1e-6))
```

```python
temporal_alignment_event=('trials.change_time: onset of the image-change flash on GO '
                          'trials, or of the sham change on CATCH trials (aligned to the '
                          'ophys/sync clock; always coincides with a flash onset)'),
off_start=OFF_START,
off_end=OFF_END,
```

iii. **Justification** — Step 1/Step 3: the reference SDK trial analysis
`save_trial_response_df.py` aligns trials to `trials.change_time`; Step 5: "Alignment event:
`trials.change_time` … This matches the reference `save_trial_response_df.py`, which aligns trials
to `change_time`." Step 3: "Temporal alignment of all streams is done upstream on a 100 kHz sync
board; the SDK returns every stream with sync-clock timestamps, so streams are directly comparable
in seconds." Alignment correctness was verified three ways (Step 7/10): change times are flash
onsets in 100 % of trials; the t = 0 bin carries `trials.change_image_name` and the t = −750 ms bin
carries `trials.initial_image_name` on 100 % of Go trials; and the population-average event
magnitude in `processing_*.png` rises only *after* t = 0 with the 750 ms flash cadence.

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Decisions**

- **Yes, the data are rebinned.** The native ophys frame rate (31 Hz / 32.3 ms for Scientifica
  single-plane, 11 Hz / 93.2 ms for Mesoscope planes) is discarded in favour of a **uniform
  750 ms bin**, giving 8 bins per trial for every trial of every session
  (`metadata['time_bin_size'] = 750.0`).
- 750 ms is exactly one image-presentation cycle (250 ms image + 500 ms grey), and the bin edges
  are phase-locked to flash onsets because `change_time` is always a flash onset.
- The bin size and window are exposed as `--bin-size` / `--off-start` / `--off-end` CLI options
  (defaults 0.750 / −2.25 / +3.75).
- Every other stream (running speed, pupil, stimulus labels) is placed on the same 750 ms grid.

ii. **Code snippets**

```python
BIN_SIZE = 0.750          # s, time bin size = one image-presentation interval
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 8
...
ap.add_argument('--bin-size', type=float, default=BIN_SIZE)
...
BIN_SIZE, OFF_START, OFF_END = args.bin_size, args.off_start, args.off_end
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

```python
data['metadata'] = dict(
    ...
    time_bin_size=BIN_SIZE * 1000.0,
    off_start=OFF_START,
    off_end=OFF_END,
    n_time_bins=NBINS,
    ...
)
```

iii. **Justification** — Step 9, "revision 1": the bin size started at 250 ms and was changed to
750 ms after empirical comparison. Reasons given: (a) the paper performs *all* of its analyses on
the "750 ms image-presentation interval" ("By image presentation interval we refer to the 750 ms
interval beginning with each image presentation"); (b) "Detected calcium events are extremely
sparse (0.02–0.15 events/s/cell), so a 250 ms bin usually contains no event at all; 750 ms bins
match the paper's unit of analysis and raise the per-bin SNR"; (c) the bin must be ≥ the slowest
frame period (93 ms) so that every bin of every session contains ≥ 1 imaging frame, which is
required because the dataset mixes 31 Hz and 11 Hz rigs; (d) empirically (14 sessions, validation
balanced accuracy) image_identity 0.334 → 0.427, image_change 0.577 → 0.650, pupil 0.271 → 0.300.

*Note:* the module docstring and `metadata['task_description']` are stale on this point — they
still say "250 ms bins" and "[-2.0, +4.0] s … 24 bins", contradicting the actual (and correctly
recorded) `time_bin_size = 750.0`, `off_start = -2.25`, `off_end = 3.75`, `n_time_bins = 8`.

---

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. **Decisions** — Image identity is derived from the **`stimulus_presentations` flash table**
(not from the trials table), restricted to the `change_detection_behavior` stimulus block, using
the columns `start_time`, `image_name` and `omitted`. The 8 image names of set A
(im061, im062, im063, im065, im066, im069, im077, im085) become the 8 output classes.

ii. **Code snippets**

```python
stim = beh['stim']
stim = stim[stim.stimulus_block_name.astype(str).str.contains('change_detection')]
stim = stim.sort_values('start_time')
fstart = stim['start_time'].values.astype(np.float64)
omitted = stim['omitted'].fillna(False).astype(bool).values
names = stim['image_name'].astype(str).values
```

iii. **Justification** — Step 1: `session.stimulus_presentations` "must be subset to the
`change_detection_behavior` block via `stimulus_block_name.str.contains('change_detection')`
(tutorials do exactly this)". Using the flash table rather than the trials table gives the true
image on screen at every moment of the window, including the pre-change repeats that fall inside
the `[-2.25, 0)` s portion of the trial, and it makes the label well defined for omitted flashes.

---

## 3-b. What processing is involved in computing `output` *Image identity*?

i. **Decisions**

1. **Omission handling**: omitted flashes (3.5 % of flashes) have `image_name == 'omitted'`; they
   are set to `None` and **forward-filled** (with a `bfill` guard for a leading omission) so that
   the omitted interval inherits the identity of the image that would have repeated.
2. **Label lookup per bin**: the image-presentation interval containing each bin **centre** is
   found with `np.searchsorted(fstart, centers, side='right') - 1`; the bin takes that interval's
   image index. The identity is thus held for the whole 750 ms interval (image + following grey),
   not only the 250 ms image-on period.
3. **Integer coding**: image names are sorted and mapped to 0–7; the script asserts that every
   session has an identical image set, so codes are globally consistent.

ii. **Code snippets**

```python
# forward-fill image identity across omissions (an omission replaces a *repeat*,
# so the ongoing image identity is the previous image)
names_ff = pd.Series(np.where(omitted, None, names)).ffill().bfill().values
image_names = sorted(set(names_ff[~pd.isna(names_ff)]) - {'omitted'})
name_to_idx = {n: i for i, n in enumerate(image_names)}
img_idx = np.array([name_to_idx[n] for n in names_ff], dtype=np.int64)

# index of the image-presentation interval (750 ms) containing each bin centre
j = np.searchsorted(fstart, centers, side='right') - 1
assert j.min() >= 0, 'bin centre precedes the first flash'
image_identity = img_idx[j]                                   # (ntrials, NBINS)
```

```python
image_names = good[0]['image_names']
for r in good:
    assert r['image_names'] == image_names, 'image sets differ across sessions'
...
output_values=[list(image_names), ...]
```

iii. **Justification** — Step 5, Key Decision 4: "Image identity is held for the full 750 ms
interval (not only the 250 ms image-on bin): this is the paper's 'image presentation interval'
convention, and calcium-event responses to a flash extend into the following grey period." Step 10,
Check 5 on omissions: "identity is forward-filled from the last non-omitted flash (an omission
replaces a *repeat*, so the ongoing identity is unchanged) … omissions never coincide with a change
by design." The global consistency of the 8-class label space is one of the stated reasons for
restricting to the familiar image set (Step 5, rule 2).

---

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. **Decisions** — Image identity is computed on the **same `edges`/`centers` arrays** used to bin
the neural data, so it is aligned by construction: bin *k* of the output row corresponds to exactly
the same absolute time interval as bin *k* of the neural matrix. The label of a bin is that of the
flash interval containing the bin centre.

ii. **Code snippets**

```python
centers = edges[:, :-1] + BIN_SIZE / 2.0       # same `edges` used for the neural binning
...
j = np.searchsorted(fstart, centers, side='right') - 1
image_identity = img_idx[j]                                   # (ntrials, NBINS)
```

Verification built into the script:

```python
bin0 = int(np.searchsorted(offs, 0.0, side='right') - 1)
ok_change_img = [img_names[ident[i, bin0]] == change_img[i] for i in range(len(tr)) if go[i]]
out['bin0_image_is_change_image_go'] = float(np.mean(ok_change_img)) if ok_change_img else None
ok_init_img = [img_names[ident[i, bin0 - 1]] == init_img[i] for i in range(len(tr)) if go[i]]
out['prebin_image_is_initial_image_go'] = float(np.mean(ok_init_img)) if ok_init_img else None
```

iii. **Justification** — Step 5: "All streams (events, running speed, pupil, stimulus, trials)
carry sync-clock timestamps from the SDK, so binning every stream on the same edges guarantees
alignment." The two per-session checks above passed at 1.000 across all 91 sessions
(`conversion_full_out.txt`: `bin0_image_is_change_image_go … mean=1.00000`,
`prebin_image_is_initial_image_go … mean=1.00000`), and an independent raw-NWB re-derivation
matched exactly for 12/12 spot-checked trials (Step 10, Check 2).

---

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. **Decisions** — From the `is_change` boolean column of the same `change_detection_behavior`
subset of `stimulus_presentations` (cross-checked against `trials.change_time` and `trials.go`).
`is_change` is True only on the flash whose image differs from the previous one, so it is True on
real (Go) changes and False on Catch (sham) changes.

ii. **Code snippets**

```python
is_change = stim['is_change'].fillna(False).astype(bool).values
```

iii. **Justification** — Step 1 lists `is_change` / `is_sham_change` as the flash-table columns
distinguishing real from sham changes; Step 5: "1 for the bins in the 750 ms
image-presentation interval that begins at the change (i.e. right after the change in image
identity), else 0. Catch (sham-change) trials therefore have all-zero change, as no image identity
change occurred." This matches the instruction wording "Have value of 1 right after a change in
image identity, otherwise 0."

---

## 4-b. What processing is involved in computing `output` *Image change*?

i. **Decisions** — Minimal: the pandas nullable `<NA>` values are filled with `False`, the column
is cast to `bool`, and the value is looked up per bin with the *same* interval index `j` used for
image identity, then cast to `int64`. Because bins are 750 ms and edges are phase-locked to flash
onsets, exactly one bin per Go trial (bin 3, the change interval) is set to 1; Catch trials are 0
throughout. No smoothing, no window extension, no thresholding.

ii. **Code snippets**

```python
is_change = stim['is_change'].fillna(False).astype(bool).values
...
j = np.searchsorted(fstart, centers, side='right') - 1
image_change = is_change[j].astype(np.int64)                  # (ntrials, NBINS)
```

```python
# 3. image_change is 1 exactly in the 750 ms interval starting at the change on GO trials,
#    and never on CATCH trials
expected = np.zeros(chg.shape[1], dtype=int)
expected[bin0:bin0 + int(round(0.75 / BIN_SIZE))] = 1
out['change_pattern_go'] = float(np.mean([np.array_equal(chg[i], expected) ...]))
out['change_zero_catch'] = float(np.mean([chg[i].sum() == 0 ...]))
```

iii. **Justification** — Step 10, Check 5: `is_change` is `<NA>` on omitted rows and is filled with
`False`; "omissions never coincide with a change by design." The expected class balance was
predicted analytically and confirmed: "`image_change` fraction expected = (1 bin / 8) × P(go) =
0.125 × 0.874 = 0.109 → measured 0.109". Both per-session checks returned 1.000 for all 91
sessions.

---

## 4-c. How is `output` *Image change* thresholded into categories?

i. **Decisions** — No thresholding is required or applied: the variable is natively binary
(`is_change` boolean → `int64` 0/1), with `output_values[1] = ['no_change', 'change']`. The only
"discretisation" choice is the *temporal* extent of the 1: it lasts exactly one 750 ms
image-presentation interval (one bin), rather than persisting for the rest of the trial.

ii. **Code snippets**

```python
image_change = is_change[j].astype(np.int64)
...
output_values=[list(image_names),
               ['no_change', 'change'],
               ...]
```

```python
image_change='1 for the 750 ms image-presentation interval starting at the image change '
             '(3 bins), 0 elsewhere; always 0 on CATCH (sham change) trials',
```

iii. **Justification** — The instruction defines the variable as binary and "1 right after a change
in image identity". Step 5 makes the temporal-extent choice explicit: the 1 covers the image
presentation interval beginning at the change. (The `output_notes` string "(3 bins)" is a stale
remnant of the 250 ms-bin version; with 750 ms bins it is 1 bin, as the `change_pattern_go` check
confirms.)

---

## 4-d. How is `output` *Image change* aligned with the neural data?

i. **Decisions** — Identical alignment to image identity: computed from the same `centers` array
derived from the same `edges` used to bin the neural data, so bin *k* of `image_change` covers the
same absolute time interval as bin *k* of the neural matrix. Bin 3 of every trial is the change
interval by construction (`OFF_START = -2.25 = -3 × 750 ms`).

ii. **Code snippets**

```python
j = np.searchsorted(fstart, centers, side='right') - 1
image_identity = img_idx[j]
image_change = is_change[j].astype(np.int64)
```

```python
out['change_times_are_flash_onsets'] = bool(
    np.all(np.min(np.abs(change_times[:, None] - fstart[None, :]), axis=1) < 1e-6))
```

iii. **Justification** — Same as 3-c. Additionally the AI reproduced the reference code's assertion
that change times coincide exactly with flash onsets (Step 1: "Reference code asserts
`np.all(np.isin(change_times, flash_times))` … I reproduced this check (100 % true)"), which is
what makes the bin grid phase-locked to the stimulus and guarantees the change occupies exactly one
bin. Panel 2 of `processing_<session>.png` overlays the binned change indicator on the raw flash
raster to show visually that there is no offset.

---

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. **Decisions** — From `BehaviorOphysExperiment.running_speed`, using its `timestamps` and
`speed` columns (the SDK's 60 Hz, 10 Hz-low-pass-filtered, transient/wrap-corrected speed in
cm/s). The unfiltered `raw_running_speed` is not used.

ii. **Code snippets**

```python
run = beh['run']
run_t = run['timestamps'].values.astype(np.float64)
run_v = run['speed'].values.astype(np.float64)
```

iii. **Justification** — Step 1: "`session.running_speed` — 60 Hz, `timestamps`, `speed` (cm/s),
10 Hz low-pass filtered + transient/wrap corrected (whitepaper). `raw_running_speed` is the
unfiltered version." Step 4 consistency table: "Running speed | `running_speed` = filtered, 60 Hz |
0 NaNs, −11 to +69 cm/s | 'running speed' in cm/s | Use `running_speed.speed`". The tutorials use
the same attribute.

---

## 5-b. What processing is involved in computing `output` *Running speed*?

i. **Decisions**

1. **Binning, not interpolation**: the mean of all 60 Hz speed samples whose timestamps fall in
   each 750 ms bin, computed NaN-aware (sum of valid samples ÷ count of valid samples), so a bin
   with no valid sample becomes NaN rather than silently 0.
2. Trials containing any NaN bin are dropped (see 1-e); the surviving values are all finite.
3. **Discretisation into quintiles** happens at the very end of `main()`, after all sessions have
   been converted, so the percentile edges can be computed over the whole dataset.

ii. **Code snippets**

```python
def bin_mean_1d(values, timestamps, edges, nan_aware=False):
    """Mean of a 1-D time series within bins. NaN-aware if requested."""
    if nan_aware:
        valid = np.isfinite(values)
        sums, _ = bin_sum_count(np.where(valid, values, 0.0), timestamps, edges)
        cnts, _ = bin_sum_count(valid.astype(np.float64), timestamps, edges)
        with np.errstate(invalid='ignore', divide='ignore'):
            out = sums[0] / cnts[0]
        out[cnts[0] == 0] = np.nan
        return out
    ...

run_binned = bin_mean_1d(run_v, run_t, edges, nan_aware=True)  # (ntrials, NBINS)
```

```python
# NOTE: running speed and pupil diameter are discretised later (in `finalize`), so that
# the percentile edges can be computed either within session or across the whole dataset.
```

iii. **Justification** — Step 5 variable-mapping table: "mean speed per 750 ms bin, then
discretised at the 20/40/60/80th percentiles". Binning by mean (rather than sampling/interpolating
a single value) is the natural reduction when the target resolution (750 ms) is far coarser than
the source (60 Hz, ~45 samples per bin), and keeps the running representation consistent with the
neural representation (both are bin means). The NaN-aware path exists so that "trials are dropped
if a bin contains no valid running or pupil sample at all" (Step 10, Check 5) rather than being
filled with a fabricated value.

---

## 5-c. How is `output` *Running speed* thresholded into categories?

i. **Decisions**

- Five bins defined by the **20th/40th/60th/80th percentiles of the binned running speed pooled
  across the entire dataset** (all trials of all 91 sessions) — "global" scope, the default;
  `--quantile-scope session` is retained as an alternative.
- Applied with `np.digitize`, yielding classes 0–4 with exactly 20.0 % of bins each.
- Measured global edges: 0.0164 / 3.6344 / 18.1401 / 32.5132 cm/s; stored in
  `metadata['session_info'][i]['running_quintile_edges']`.

ii. **Code snippets**

```python
def quantile_bin(values, nq=NQUANTILES):
    """Discretise into nq equal-percentile bins; returns labels (ints) and the edges used."""
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    qs = np.linspace(0, 100, nq + 1)[1:-1]
    edges = np.percentile(finite, qs)
    labels = np.digitize(v, edges, right=False)
    return labels.astype(np.int64), edges
```

```python
if args.quantile_scope == 'global':
    run_all = np.concatenate([r['run_binned'].ravel() for r in good])
    _, run_edges_g = quantile_bin(run_all)
for r in good:
    if args.quantile_scope == 'global':
        run_edges, pupil_edges = run_edges_g, pupil_edges_g
        run_cls = np.digitize(r['run_binned'], run_edges)
    ...
    for i, o in enumerate(r['output']):
        o[2] = run_cls[i]
```

iii. **Justification** — Step 9, "revision 3": the first version used per-session quintiles; this
was changed to global because "This is the literal reading of 'discretised into five equal
percentile bins', and it makes a class label mean the same physical speed/diameter in every
session — which matters because the decoder's readout layer is *shared* across sessions."
Empirically (14 sessions): running 0.257 → 0.266, pupil 0.231 → 0.271. The earlier per-session
argument (that global edges would partly encode session identity) is documented in Step 5, Key
Decision 3, and the option is kept in the CLI. Class fractions are verified per session
(`checks['running_class_fracs']`) and dataset-wide (exactly [0.2, 0.2, 0.2, 0.2, 0.2]).

*Note:* `metadata['task_description']` still says "within-session quintiles" although
`metadata['quantile_scope']` and `output_notes` correctly say `global`.

---

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. **Decisions** — Running speed is binned on the **same `edges` matrix** used for the neural
data, using its own SDK sync-clock timestamps; there is no separate resampling step and therefore
no opportunity for drift between the two streams. `run_binned` has shape `(ntrials, NBINS)`,
matching the neural `(ncells, ntrials, NBINS)` on its trial and time axes, and the same `good`
trial mask is applied to both.

ii. **Code snippets**

```python
run_binned = bin_mean_1d(run_v, run_t, edges, nan_aware=True)  # same `edges` as the neural binning
...
neural = neural[:, good, :]
run_binned = run_binned[good]
pupil_binned = pupil_binned[good]
```

iii. **Justification** — Step 3: "Temporal alignment of all streams is done upstream on a 100 kHz
sync board; the SDK returns every stream with sync-clock timestamps, so streams are directly
comparable in seconds." Step 5: "binning every stream on the same edges guarantees alignment."
Panel 3 of `processing_<session>.png` overlays the raw 60 Hz trace, the binned mean and the
resulting quintile class on the same time axis as the neural panel, and Step 10 Check 2
independently re-binned the raw 60 Hz `running_speed.speed` from the NWB files and reproduced the
stored quintile classes with "exact equality for **all** trials of each session (0 mismatched bins
out of 1,168 / 1,736 / 2,584 / 3,504)".

---

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. **Decisions** — From `BehaviorOphysExperiment.eye_tracking`, using the `timestamps` and
**`pupil_area`** columns, converted to a diameter as `2·sqrt(pupil_area/π)`. Blink/outlier frames
are already NaN in `pupil_area` (the SDK sets them from `likely_blink`), so the NaNs themselves
act as the blink mask. Sessions with no eye-tracking table at all are dropped.

ii. **Code snippets**

```python
eye = beh['eye']
eye_t = eye['timestamps'].values.astype(np.float64)
# pupil_area = pi * max(width, height)^2  (SDK compute_circular_area)
pupil_diam = 2.0 * np.sqrt(eye['pupil_area'].values.astype(np.float64) / np.pi)
```

```python
if beh['eye'] is None or len(beh['eye']) == 0:
    return dict(session_id=session_id, skip='no eye tracking', timings=timings)
```

iii. **Justification** — Step 1: the AI read `allensdk/brain_observatory/behavior/
eye_tracking_processing.py` and found `compute_circular_area` = `pi * max(width, height)^2`, i.e.
the SDK treats `max(width, height)` as a **radius**; therefore the diameter is
`2·sqrt(area/π) = 2·max(width, height)` px. It also "verified numerically that `pupil_area !=
pi*w*h`", confirming the circular (not elliptical) convention. Step 1 also notes
`determine_likely_blinks` is what produces the NaNs, so no separate `likely_blink` mask is needed.

---

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. **Decisions**

1. Convert area → diameter (above).
2. **Gap-limited interpolation**: NaN runs are linearly interpolated **only if the gap is ≤ 1 s**
   (`PUPIL_INTERP_MAX_GAP = 1.0`). Longer runs (tracking failures, up to ~14 s) are left NaN.
   Gaps touching the start or end of the recording are never interpolated (no two-sided support).
3. NaN-aware binning to the 750 ms grid (same `bin_mean_1d` as running speed).
4. Trials still containing a NaN bin are dropped.
5. Discretisation into 5 global percentile bins at the end of `main()`.

ii. **Code snippets**

```python
def interpolate_short_gaps(t, x, max_gap):
    """Linearly interpolate NaNs in x(t) across gaps shorter than max_gap seconds.
    Longer gaps (tracking failures) are left as NaN so that the affected trials can be dropped."""
    x = np.asarray(x, dtype=np.float64).copy()
    good = np.isfinite(x)
    if good.sum() < 2:
        return x
    xi = np.interp(t, t[good], x[good])
    bad = ~good
    if bad.any():
        d = np.diff(bad.astype(np.int8))
        starts = list(np.nonzero(d == 1)[0] + 1)
        ends = list(np.nonzero(d == -1)[0] + 1)
        if bad[0]:  starts = [0] + starts
        if bad[-1]: ends = ends + [len(bad)]
        for s, e in zip(starts, ends):
            if s == 0 or e == len(bad):
                continue      # edge gaps have no two-sided support: leave NaN
            if t[e - 1] - t[s] <= max_gap:
                x[s:e] = xi[s:e]
    return x
```

```python
pupil_diam = interpolate_short_gaps(eye_t, pupil_diam, PUPIL_INTERP_MAX_GAP)
pupil_binned = bin_mean_1d(pupil_diam, eye_t, edges, nan_aware=True)
```

iii. **Justification** — Step 10, Check 5: "pupil NaNs (median 2.9 %, max 29.6 % of frames, runs up
to ~14 s) are linearly interpolated only across gaps ≤ 1 s; longer gaps stay NaN and the affected
trials are dropped (599 trials, 2.6 %)." The rationale is that a blink lasts a fraction of a second
so interpolating across it recovers a plausible value, whereas interpolating across a multi-second
tracking failure would fabricate data; dropping those trials is preferred to fabricating or
assigning them an arbitrary class.

---

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. **Decisions** — Exactly as running speed: 5 classes from the 20/40/60/80th percentiles of the
binned pupil diameter **pooled over the whole dataset** (global scope by default), applied with
`np.digitize`, giving exactly 20 % of bins per class. Measured global edges:
74.983 / 84.046 / 93.425 / 108.913 px, stored per session in `metadata`.

ii. **Code snippets**

```python
pupil_all = np.concatenate([r['pupil_binned'].ravel() for r in good])
_, pupil_edges_g = quantile_bin(pupil_all)
...
pupil_cls = np.digitize(r['pupil_binned'], pupil_edges)
...
r['info']['pupil_quintile_edges'] = [float(x) for x in pupil_edges]
r['checks']['pupil_class_fracs'] = [float((pupil_cls == c).mean()) for c in range(NQUANTILES)]
```

iii. **Justification** — Same as 5-c (Step 9, revision 3): the global edges are "the literal
reading of 'discretised into five equal percentile bins'" and give a label the same physical
meaning in every session under the decoder's shared readout. The AI explicitly acknowledges the
trade-off it accepted: pupil diameter is in camera pixels and depends on rig/zoom, so with global
edges individual sessions can occupy only 1–2 quintiles (visible in
`verification_full_out.txt`, where several sessions have a pupil range such as `[3.0, 4.0]` or
`[4.0, 4.0]`), while the dataset-wide distribution is exactly uniform. Step 10 Check 2 verified the
stored global edges bracket the per-session raw percentiles recomputed from the NWB files.

---

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. **Decisions** — Same mechanism as running speed: the ~30 Hz eye-tracking samples, carrying
sync-clock timestamps, are binned onto the **same `edges`** as the neural data; the same `good`
trial mask is then applied. No separate resampling onto an intermediate time base is performed.

ii. **Code snippets**

```python
pupil_binned = bin_mean_1d(pupil_diam, eye_t, edges, nan_aware=True)
...
pupil_binned = pupil_binned[good]
```

iii. **Justification** — Step 3/Step 5: all streams are hardware-synced upstream and returned by
the SDK on the same clock, so binning on shared edges is sufficient and exact. Panel 4 of
`processing_<session>.png` shows the raw pupil trace, its binned mean and the resulting quintile
class on the same axis as the neural and stimulus panels.

---

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. **Decisions** — From the four mutually exclusive boolean columns of the trials table,
`hit`, `miss`, `false_alarm`, `correct_reject`, in that fixed order
(`OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']`). The script **asserts** that
these four flags exactly partition the selected go/catch non-auto-rewarded trials (exactly one
True per trial) rather than silently falling back to an "other" class.

ii. **Code snippets**

```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome_mat = np.stack([tr[c].astype(bool).values for c in OUTCOME_NAMES], axis=1)
assert np.all(outcome_mat.sum(axis=1) == 1), 'trial outcomes do not partition trials'
outcome = np.argmax(outcome_mat, axis=1).astype(np.int64)
```

iii. **Justification** — Step 1 lists these as the SDK trial-table outcome flags; Step 5 variable
mapping: "these four flags exactly partition the go+catch non-auto trials: verified sum == n
trials". The resulting dataset fractions (hit 0.3033 / miss 0.5714 / FA 0.0215 / CR 0.1038) match
an independent raw scan of all active experiments (0.3033/0.5710/0.0215/0.1043) to ~4 decimals
(Step 9 consistency table).

---

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. **Decisions** — The four booleans are collapsed by `argmax` into an integer code 0–3 and the
per-trial constant is **broadcast across all 8 time bins**, so `output[4]` is a constant row within
each trial. `output_values[4] = ['hit', 'miss', 'false_alarm', 'correct_reject']`. Outcome is kept
as one 4-class variable rather than split into two binary variables. The per-session counts of each
outcome are recorded in `metadata['session_info']`.

ii. **Code snippets**

```python
output_trials.append(np.stack([
    image_identity[i],
    image_change[i],
    np.zeros(NBINS, dtype=np.int64),   # placeholder: running quintile
    np.zeros(NBINS, dtype=np.int64),   # placeholder: pupil quintile
    np.full(NBINS, outcome[i], dtype=np.int64),
]).astype(np.int64))
```

```python
n_hit=int((outcome == 0).sum()), n_miss=int((outcome == 1).sum()),
n_false_alarm=int((outcome == 2).sum()), n_correct_reject=int((outcome == 3).sum()),
...
trial_outcome='static per trial (broadcast over time bins)'),
```

iii. **Justification** — Step 5, Key Decision 5: "Trial outcome as 4 classes (hit/miss/false alarm/
correct reject) rather than 2 × 2, keeping the natural experimental categories; it is constant
within a trial (static per-trial), broadcast over time as required by the (d_output, T) format."
In Step 12 the AI re-examined this in light of the low 4-class accuracy (0.338 vs 0.25 chance) and
kept it deliberately: "Splitting the outcome into two binary outputs would raise the headline
number but would depart from the task specification of 'trial outcome' as one categorical
variable", noting that the paper itself reports false-alarm decoding as "very low" (Fig. S22) while
hit-vs-miss re-run on the converted data gives 0.726.

---

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. **Decisions** — The script distinguishes *expected* imperfections (handled explicitly) from
*unexpected* ones (fail loudly / skip the session):

| Issue | Handling |
|---|---|
| Session has no eye-tracking table | Session skipped with reason `'no eye tracking'` (1 session, 805989030) |
| Blink NaNs in pupil, gap ≤ 1 s | Linearly interpolated |
| Pupil gap > 1 s, or gap at the recording edge | Left NaN → the trial is dropped |
| Any running/pupil bin with no valid sample | Trial dropped (599 / 22,778 = 2.6 %) |
| Session left with < 2 usable trials | Session skipped |
| Any other exception in a session | Caught, session skipped, traceback printed; the run continues |
| pandas nullable `<NA>` in `omitted` / `is_change` | `.fillna(False).astype(bool)` |
| Omitted flashes (no image shown) | Image identity forward-filled from the previous flash |
| `np.str_` vs `str` in `brain_regions` | Explicitly cast to `str` |
| Mixed frame rates (31 vs 11 Hz) | Bins hold the *mean*, so values are frame-rate independent |
| Missing `change_time`, non-partitioning outcome flags, window outside the trial, empty time bin, bin centre before the first flash | **Assertions** — these never occur in this dataset, so the script refuses to continue if they ever do |

ii. **Code snippets**

```python
except Exception as ex:
    import traceback
    return dict(session_id=int(session_id), skip='ERROR: ' + repr(ex),
                traceback=traceback.format_exc(), timings=timings)
```

```python
omitted = stim['omitted'].fillna(False).astype(bool).values
is_change = stim['is_change'].fillna(False).astype(bool).values
```

```python
skipped = [(r['session_id'], r['skip']) for r in results if r.get('skip')]
for sid, why in skipped:
    print(f'[skip] session {sid}: {why}')
    for r in results:
        if r['session_id'] == sid and 'traceback' in r:
            print(r['traceback'])
good = [r for r in results if not r.get('skip')]
print(f'[main] {len(good)} sessions converted, {len(skipped)} skipped')
```

iii. **Justification** — Step 6: "Assertions built into the conversion (fail loudly rather than
silently producing bad data)". Step 10, Check 5 enumerates each edge case and the empirical
evidence behind it. The conversion funnel table (Step 9) accounts for every dropped experiment,
session and trial so that "no data is silently lost". The AI's stated principle for behavioural
gaps is to drop rather than impute, because an imputed pupil/running value becomes a *decoder
target label*, so a fabricated value would be a fabricated ground truth.

---

## 9-a. What are the most time-consuming steps of the code?

i. **Decisions / findings** — The script instruments itself with a `timings` dict per session
(`load`, `neural`, `stim`, `behavior`, `total`). The dominant cost is **reading the NWB files**
(`bc.get_behavior_ophys_experiment` + materialising `events`, `trials`,
`stimulus_presentations`, `running_speed`, `eye_tracking`), which is I/O-bound and scales with the
number of imaging planes: ~3 s per plane, so a 7-plane Mesoscope session costs ~24 s versus ~7 s
for a single-plane session. Everything downstream (binning, labelling, discretisation) is
vectorised and negligible. Constructing the `VisualBehaviorOphysProjectCache` inside each worker is
a second, smaller fixed cost paid once per session.

The AI mitigated the bottleneck with an 8-process pool, bringing the full conversion of 92 sessions
/ 110 planes to **68.9 s** wall clock.

ii. **Code snippets**

```python
t0 = time.time()
planes = []
for eid in exp_ids:
    ds = bc.get_behavior_ophys_experiment(int(eid))
    ...
timings['load'] = time.time() - t0
...
timings['neural'] = time.time() - t0
timings['stim']   = time.time() - t0
timings['behavior'] = time.time() - t0
timings['total'] = time.time() - t_start
```

```python
with Pool(min(args.workers, len(jobs))) as pool:
    for i, r in enumerate(pool.imap_unordered(convert_session, jobs)):
        results.append(r)
        el = time.time() - t_start
        print(f'[{i+1}/{len(jobs)}] session {r["session_id"]} ... '
              f'({el:.0f}s elapsed, {el/(i+1):.1f}s/session)', flush=True)
```

iii. **Justification** — Step 7 Run Time Estimates: "per-plane cost ~3 s → 110 planes ≈ 330 s
serial; full run with 8 workers ≈ 1–3 min (well under the 15 min budget)"; Step 6 lists "each NWB
file is opened exactly once per session" and "8-way multiprocessing over sessions" as the main
speed-ups. The actual full run (68.9 s) came in at the fast end of the estimate.

---

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. **Decisions / findings** — The expensive inner loops were already eliminated: binning of all
neurons × trials × bins is done with a single `cumsum` + `searchsorted` (`bin_sum_count`), and the
image-identity / image-change lookups are a single vectorised `searchsorted` over the whole
`(ntrials, nbins)` centre matrix. The remaining Python-level loops are all O(n_trials) or
O(n_planes) with trivial bodies and are not bottlenecks:

- `for i in range(n_good)` twice, building `neural_trials` and `output_trials` — this could be a
  single `np.moveaxis` + `list(...)` and a vectorised `np.stack` of the five rows;
- the per-trial list comprehensions inside `sanity_checks` (`ok_change_img`, `ok_init_img`,
  `change_pattern_go`, `change_zero_catch`) — each iterates over every trial in Python and rebuilds
  `np.array_equal` comparisons that could be one broadcast comparison;
- `for i, o in enumerate(r['output']): o[2] = run_cls[i]` in `main()` — a per-trial write-back that
  exists only because the quintile rows are placeholders filled in later;
- the NaN-run loop in `interpolate_short_gaps` (bounded by the number of blink runs);
- `for p in planes` (3–7 iterations) and the per-session loops in `main()`.

ii. **Code snippets**

```python
# already vectorised: whole (ncells, ntrials, nbins) tensor in three array ops
idx = np.searchsorted(timestamps, edges_flat)
cs = np.concatenate([np.zeros((v.shape[0], 1)), np.cumsum(v, axis=1)], axis=1)
sums = cs[:, idx[:, 1:]] - cs[:, idx[:, :-1]]
```

```python
# remaining per-trial python loops
neural_trials = [np.ascontiguousarray(neural[:, i, :]) for i in range(n_good)]
output_trials = []
for i in range(n_good):
    output_trials.append(np.stack([...]).astype(np.int64))
...
ok_change_img = [img_names[ident[i, bin0]] == change_img[i] for i in range(len(tr)) if go[i]]
...
for i, o in enumerate(r['output']):
    o[2] = run_cls[i]
```

iii. **Justification** — Step 6: "binning implemented with cumulative sums + `searchsorted`
instead of per-trial/per-bin masks (all neurons, trials and bins in 3 array ops)", estimated at
"~10–50× on the binning step". The residual loops were left in place because the target format
itself requires a *list* of per-trial arrays, so the final materialisation must produce one array
object per trial regardless, and because loading dominates the runtime by more than an order of
magnitude.

---

## 9-c. What processing does the code repeat multiple times?

i. **Decisions / findings**

- **`get_cache()` is called once per session inside every worker** (plus twice in the parent via
  `local_experiment_table()`), so the `VisualBehaviorOphysProjectCache` and its metadata tables are
  re-constructed ~93 times instead of once. This is the main repeated work; it is partly forced by
  `multiprocessing.Pool` (the cache object is not shared between processes), but it could be done
  once per worker with a `Pool(initializer=...)` rather than once per task.
- `quantile_bin(run_all)` / `quantile_bin(pupil_all)` compute a full `np.digitize` over every time
  bin of the entire dataset and then **discard the labels**, keeping only the edges; the labels are
  immediately recomputed per session with `np.digitize(r['run_binned'], run_edges)`.
- `np.stack(result['neural'])` is rebuilt inside `sanity_checks` after the per-trial list has just
  been built, and again in `plot_processing`.
- `np.interp` in `interpolate_short_gaps` interpolates the *entire* pupil time series even though
  only the samples inside short gaps are used.
- The final summary re-stacks and re-reshapes every session's outputs
  (`np.stack(r['output']).transpose(1, 0, 2).reshape(5, -1)`) to recompute class fractions that the
  per-session `checks` already hold in part.

Things that are deliberately **not** repeated: each NWB file is opened exactly once; behaviour
streams are read from one plane only, not from all 7; sessions are never re-loaded for the
discretisation pass (the binned running/pupil values are carried forward in memory and deleted
afterwards with `del r['run_binned'], r['pupil_binned']`).

ii. **Code snippets**

```python
def convert_session(args):
    ...
    bc = get_cache()          # re-created for every session/task
```

```python
_, run_edges_g = quantile_bin(run_all)      # labels computed then thrown away
...
run_cls = np.digitize(r['run_binned'], run_edges)   # recomputed per session
```

```python
del r['run_binned'], r['pupil_binned']      # deliberate: large arrays freed after use
```

iii. **Justification** — Step 6 lists the deliberate de-duplication ("each NWB file is opened
exactly once per session", "behavior streams read once per session — avoids 7× redundant reads on
Mesoscope sessions"). The repeated cache construction and the discarded `quantile_bin` labels are
not discussed in CONVERSION_NOTES; they cost a small constant per session and are invisible against
the I/O bottleneck.

---

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. **Decisions / findings**

- **`cell_ids` and `depths` are collected for every neuron of every plane and stored in the
  per-session result dict, but never written to the output pickle** (`brain_region_idx` is built
  from `regions` only). This is the clearest instance of computed-then-discarded work.
- `quantile_bin` returns labels for the pooled dataset array that are immediately discarded (see
  9-c).
- Running speed and pupil diameter are binned for **all** selected trials, including the 599 trials
  later dropped for missing behaviour, and the per-neuron z-score statistics are likewise computed
  over trials that are later dropped (the latter is deliberate and documented).
- `centers` is computed for every session even when the stimulus labels could reuse `edges`.
- `plot_processing` keeps the full per-plane `events` arrays alive in `result['_plotargs']` for the
  two plotted sessions — a memory cost with no effect on the output.
- Several `metadata` fields (`session_checks`, `trial_ids`, per-session quintile edges, frame
  rates) are stored and never consumed by the decoder — though these are documentation/audit
  artefacts, not waste.
- The `--neural-signal filtered_events`, `--normalize {none,std}` and `--quantile-scope session`
  code paths exist only for the ablations reported in CONVERSION_NOTES and are dead in the default
  run.

ii. **Code snippets**

```python
neural_blocks, region_list, depth_list, cellid_list = [], [], [], []
for p in planes:
    ...
    depth_list += [p['depth']] * p['events'].shape[0]
    cellid_list += list(p['cell_ids'])
...
result = dict(
    ...
    depths=np.array(depth_list),       # never used downstream
    cell_ids=np.array(cellid_list),    # never used downstream
    ...)
```

```python
# only `regions` is consumed when assembling the dataset
brain_region_idx=[np.array([reg_to_idx[str(x)] for x in r['regions']], dtype=np.int64)
                  for r in good],
```

iii. **Justification** — CONVERSION_NOTES does not flag any of these as waste; `cell_ids`/`depths`
appear to be provenance fields that were collected with the intent of exporting them (the human
reference likewise encodes depth into its region label as `{area}_{depth}um`) but were left out of
the final dictionary. All of these costs are negligible relative to NWB I/O, and the AI's stated
efficiency target (Step 7: keep the full run well under 15 minutes; actual 68.9 s) was met.

---
