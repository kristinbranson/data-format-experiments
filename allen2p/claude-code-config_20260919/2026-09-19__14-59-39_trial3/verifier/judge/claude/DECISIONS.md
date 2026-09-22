# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. <Decisions>

The AI loads the Allen Visual Behavior 2P release **directly from the local NWB files** rather than
through the AllenSDK S3/cloud cache (it found that the local cache directory lacks the
`visual-behavior-ophys/manifests` entry that `VisualBehaviorOphysProjectCache` needs, and noted that
`get_behavior_ophys_experiment()` itself ultimately calls `from_nwb`, so the resulting object is
identical). Session discovery uses the released metadata table
`project_metadata/ophys_experiment_table.csv` — the same table served by
`bc.get_ophys_experiment_table()`. Each experiment is opened with
`BehaviorOphysExperiment.from_nwb_path()` and all streams are taken from the standard SDK
attributes: `.ophys_timestamps`, `.dff_traces`, `.cell_specimen_table`, `.trials`,
`.stimulus_presentations`, `.running_speed`, `.eye_tracking`, `.metadata`.
Sessions are processed independently and in parallel (`ProcessPoolExecutor`, 16 workers); results
are collected and assembled into the target dict in `main()`. 168 experiments were selected,
165 kept (37 mice, 28,821 neurons, 42,410 trials, 11.18 M timepoints), in 61 s wall time.

ii. <Code snippets>

```python
def select_experiments():
    et = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    present = set()
    for fn in os.listdir(NWB_DIR):
        if fn.endswith('.nwb'):
            present.add(int(fn.split('_')[-1].split('.')[0]))
    sel = et[et.ophys_experiment_id.isin(present)
             & (et.project_code == PROJECT_CODE)
             & (~et.passive.astype(bool))].copy()
    sel = sel.sort_values('ophys_experiment_id').reset_index(drop=True)
    return sel
```

```python
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
    BehaviorOphysExperiment)

path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{oeid}.nwb')
ds = BehaviorOphysExperiment.from_nwb_path(path)
ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
```

```python
with ProcessPoolExecutor(nworkers) as ex:
    for n, res in enumerate(ex.map(process_experiment, jobs), start=1):
        results.append(res)
```

iii. <Justification>

From CONVERSION_NOTES.md Step 10, Check 3(a): *"`BehaviorOphysExperiment.from_nwb_path(...)` and the
same attributes. The cache path is unusable locally (missing `visual-behavior-ophys/manifests`), and
`get_behavior_ophys_experiment` itself ends in `from_nwb`, so the loaded object is identical.
Session selection uses the released `ophys_experiment_table.csv`, the same table
`bc.get_ophys_experiment_table()` serves."* The AI cross-checked the loaded counts against
`ophys_cells_table.csv` (29,097 cells for the 168 experiments) and confirmed no locally-present file
was missed.

---

## 1-b. How are the data split into subjects?

i. <Decisions>

Subjects are mice, identified by `metadata['mouse_id']` read from each NWB file (equivalently the
`mouse_id` column of the experiment table). The `subjects` list is the sorted set of mouse ids over
the sessions that survive curation, and `subject_idx` indexes it per session. 37 mice, 2–9 sessions
each (mean 4.5).

ii. <Code snippets>

```python
info = {
    ...
    'mouse_id': str(md['mouse_id']),
    ...
}
return {..., 'subject': str(md['mouse_id']), ...}
```

```python
subjects = sorted({r['subject'] for r in kept})
subject_index = {s: i for i, s in enumerate(subjects)}
data = {
    ...
    'subjects': subjects,
    'subject_idx': np.array([subject_index[r['subject']] for r in kept], dtype=np.int64),
```

iii. <Justification>

`mouse_id` is the SDK's canonical animal identifier. The AI validated the resulting count against
the metadata table (Step 9 consistency table: 37 subjects from the table, 37 in the converted data)
and listed sessions-per-subject in `verification_full_out.txt`.

---

## 1-c. How are the data split into sessions?

i. <Decisions>

One converted "session" = one `ophys_experiment_id` = one NWB file. The AI explicitly verified that
in the `VisualBehavior` (single-plane Scientifica) project **every ophys session contains exactly one
experiment/imaging plane**, so experiment ≡ session and no cross-plane merging is needed.

Three session-level selection rules are applied:
1. `project_code == 'VisualBehavior'` — excludes the 45 `VisualBehaviorMultiscope` files (one mouse)
   because they run at 10.73 Hz vs 30.94 Hz and would break the "same time bin for all trials and
   sessions" requirement.
2. `passive == False` — the 71 passive experiments are dropped.
3. Sessions with an empty `eye_tracking` table are dropped when opened (3 sessions:
   795953296, 806456687, 833631914).

Result: 239 local VisualBehavior experiments → 168 active → 165 converted.

ii. <Code snippets>

```python
sel = et[et.ophys_experiment_id.isin(present)
         & (et.project_code == PROJECT_CODE)
         & (~et.passive.astype(bool))].copy()
```

```python
try:
    eye = ds.eye_tracking
except Exception:
    eye = None
if eye is None or len(eye) == 0 or not np.isfinite(eye['pupil_area'].values).any():
    return {'oeid': int(oeid), 'skip': 'no eye tracking data'}
```

iii. <Justification>

CONVERSION_NOTES.md Step 4 discrepancy table:
- Project: *"it is the only choice compatible with 'time bins the same size for all sessions':
  single-plane = 30.9406 Hz for every session, Multiscope = 10.726 Hz. Including the Multiscope mouse
  would either break the uniform bin size or force resampling of every session, and would add only 1
  subject."*
- Passive: *"hit = false_alarm = 0 in every passive session ⇒ trial outcome is degenerate (only
  miss / correct_reject) … Paper: passive viewing 'was not analyzed here' ⇒ Exclude the 71 passive
  VisualBehavior experiments."*
- Eye tracking: *"3/168 active sessions have an empty table … pupil diameter is a required decoder
  output and cannot be produced for them."*
- Session ≡ experiment: *"In the VisualBehavior project every ophys session has exactly 1
  experiment."*

---

## 1-d. How are the data split into trials?

i. <Decisions>

Trials are the rows of the SDK `trials` table selected with `go | catch` (which by construction
excludes `aborted` and `auto_rewarded`; the AI verified mutual exclusivity by checking
`hit+miss+false_alarm+correct_reject == n(go ∪ catch)` in every session). The trial window is the
experiment's own **half-open `[trials.start_time, trials.stop_time)`**, sampled at the ophys frame
timestamps that fall inside it — i.e. variable-length trials (7.26–12.56 s, mean T = 262 frames ≈
8.47 s), spanning the pre-change flashes and the post-change response window. Because the window is
half-open, consecutive trials can never share a frame.

ii. <Code snippets>

```python
trials = ds.trials
go = trials['go'].values.astype(bool)
catch = trials['catch'].values.astype(bool)
keep = go | catch
tr = trials[keep]
...
starts = tr['start_time'].values.astype(np.float64)
stops = tr['stop_time'].values.astype(np.float64)
i0 = np.searchsorted(ts, starts, side='left')
i1 = np.searchsorted(ts, stops, side='left')
```

```python
for k in sel:
    a, b = i0[k], i1[k]
    T = b - a
    neural.append(np.ascontiguousarray(traces[:, a:b]))
```

iii. <Justification>

Step 5: *"Window `[trials.start_time, trials.stop_time)` — the experiment's own trial definition,
and exactly the window the reference tutorial plots."* and *"Trials: `trials[(go) | (catch)]` →
excludes `aborted` and `auto_rewarded` by construction."* This follows the task instruction to
"segment each recording session into individual trials based on how they are defined in the
experiment. Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded'
trials." `off_start = 0.0`, `off_end = None` because the length is the experiment's natural
variable length.

---

## 1-e. How are trials filtered based on quality controls?

i. <Decisions>

On top of the `go | catch` selection:
- a trial is dropped if it contains **no valid (non-blink) pupil sample**, because its pupil trace
  would be pure interpolation/extrapolation (60 trials of 42,470 dropped, 0.14 %);
- a trial is dropped if it contains fewer than 2 ophys frames (0 occurred);
- a session is dropped if fewer than 2 usable trials remain (the decoder needs ≥2), or if it has 0
  valid ROIs (0 occurred).
An assertion enforces that every kept go/catch trial has exactly one of hit/miss/FA/CR.
No filtering on engagement, reward rate, lick behaviour, or trial length is applied.

ii. <Code snippets>

```python
j0 = np.searchsorted(pupil_valid_t, starts, side='left')
j1 = np.searchsorted(pupil_valid_t, stops, side='left')

n_drop_short, n_drop_pupil = 0, 0
sel = []
for k in range(len(tr)):
    if i1[k] - i0[k] < MIN_FRAMES_PER_TRIAL:
        n_drop_short += 1
        continue
    if j1[k] - j0[k] < 1:
        n_drop_pupil += 1
        continue
    sel.append(k)
sel = np.asarray(sel, dtype=int)
if len(sel) < MIN_TRIALS_PER_SESSION:
    return {'oeid': int(oeid), 'skip': f'only {len(sel)} usable trials'}
```

```python
assert outcome_cols.sum(axis=1).min() == 1 and outcome_cols.sum(axis=1).max() == 1, \
    f'{oeid}: go/catch trial without a unique outcome'
```

iii. <Justification>

Step 5: *"Drop a trial if it contains no valid (non-blink) pupil sample — pupil would be pure
extrapolation. Drop a trial with < 2 ophys frames. Drop a session left with < 2 trials or 0 neurons
(none expected)."* Step 10 Check 3(b): *"Extra: 3 sessions with no eye tracking and 60 trials with
no valid pupil sample — required because pupil is a mandatory decoder output."* The AI also audited
(Step 10 Check 5) that no trial window falls outside the ophys / running / eye-tracking timestamp
ranges, that no `change_time` is NaN on a go/catch trial, and that no trials overlap.

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. <Decisions>

`neural` = `dff_traces['dff']` — the released, neuropil-subtracted, demixed, baseline-normalised and
detrended ΔF/F traces — for the cells with `cell_specimen_table.valid_roi == True`, sampled on
`ophys_timestamps`. The AI explicitly considered and rejected the paper's signal
(`events` / `events.filtered_events`) after an empirical benchmark.

ii. <Code snippets>

```python
cst = ds.cell_specimen_table
valid_roi = cst['valid_roi'].values.astype(bool)
cell_ids = cst.index.values[valid_roi]

signal = job.get('neural_signal', 'dff')
if signal == 'dff':
    col = ds.dff_traces['dff']
else:
    col = ds.events[signal]
traces = np.empty((len(cell_ids), len(ts)), dtype=np.float32)
for i, cid in enumerate(cell_ids):
    traces[i] = col.loc[cid]
```

iii. <Justification>

Step 7 "Neural signal benchmark (why dF/F)": the AI converted the same data three ways and trained
the reference decoder on each (20-session benchmark, validation balanced accuracy):
`filtered_events` 0.292 / `filtered_events` z-scored 0.372 / **`dff` 0.470** on image identity, with
dF/F winning on 4 of 5 outputs. Rationale quoted: *"The detected calcium events used by the reference
paper are ~0.25 % non-zero per 32 ms frame; the SDK tutorial itself states their effective resolution
is ~200 ms. A per-timepoint decoder therefore sees almost no signal … dF/F is the primary neural
product of the release (a whole whitepaper section, 'DF/F CALCULATION', is devoted to it) … and it is
the signal the reference tutorial that defines our trial segmentation plots per trial."* This is
logged as a deliberate, measured deviation from the *paper* (Step 10, "Documented differences #1").

---

## 2-b. How is the `neural` data processed?

i. <Decisions>

**No processing at all beyond materialising the trace matrix and slicing it by trial.** The values
stored in `neural` are exactly the released dF/F values, cast to `float32`, ordered by
`cell_specimen_table` index. No smoothing, no z-scoring, no baseline subtraction, no normalisation,
no re-binning. (A `--zscore` option exists but is off by default; it was benchmarked and made no
difference.) Because the VisualBehavior project has one plane per session, no cross-plane stacking is
needed. Each neuron is labelled with the session's `targeted_structure` (all VISp).

ii. <Code snippets>

```python
traces = np.empty((len(cell_ids), len(ts)), dtype=np.float32)
for i, cid in enumerate(cell_ids):
    traces[i] = col.loc[cid]
if job.get('zscore', False):
    sd = traces.std(axis=1, keepdims=True)
    traces = (traces - traces.mean(axis=1, keepdims=True)) / np.maximum(sd, 1e-6)
```

```python
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

```python
region = str(md['targeted_structure'])
brain_region_name = [region] * len(cell_ids)
```

iii. <Justification>

Step 7: *"z-scoring adds nothing on top of dF/F (differences ≤0.02, both directions), so the raw
released traces are kept — the converted `neural` values *are* the released dF/F."* Step 10 Check 3
(d): *"native ophys frames (32.32 ms), no re-binning, no smoothing."* The AI then verified this by
re-reading `processing/ophys/dff/traces/data` with raw h5py and confirming
`np.allclose(atol=1e-6)` on 15 random (session, trial, neuron, timepoint) samples.

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. <Decisions>

Only `cell_specimen_table.valid_roi == True` cells are kept — which turns out to be a no-op, since
the public release contains 29,097/29,097 valid ROIs. No additional QC (no SNR, event-rate, or
activity threshold) is applied, because the release has already been through the Allen pipeline's
ROI classifier, demixing, neuropil correction and container QC. Sessions with 0 valid ROIs would be
dropped (none occurred); the smallest kept session has 6 neurons, the largest 666.

ii. <Code snippets>

```python
cst = ds.cell_specimen_table
valid_roi = cst['valid_roi'].values.astype(bool)
cell_ids = cst.index.values[valid_roi]
if len(cell_ids) == 0:
    return {'oeid': int(oeid), 'skip': 'no valid ROIs'}
```

iii. <Justification>

Step 5 Key Decision 4: *"No extra neuron filtering — the release already contains only `valid_roi`
cells; verified 29,097/29,097."* Step 10 Check 3(b): *"pipeline QC already applied to the release
(ROI classifier, demixing, container QC) … no further filtering, matching the papers, which describe
none."* Edge-case audit confirmed 0 duplicate/NaN `cell_specimen_id` and 0 NaN/Inf in dF/F.

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. <Decisions>

Everything is aligned to the **ophys frame clock** (`ophys_timestamps`), as the task instruction
requires ("Temporally align based on ophys timestamp"). The alignment event for a trial is
`trials.start_time`; the kept frames are those with `start_time <= t < stop_time`, found with two
vectorised `np.searchsorted(..., side='left')` calls. The *same* frame index range `a:b` slices the
neural matrix and all five output traces, so alignment between streams is exact by construction.
Metadata records `temporal_alignment_event = 'Start of each change-detection trial
(trials.start_time)'`, `off_start = 0.0`, `off_end = None` (variable-length trials).

ii. <Code snippets>

```python
i0 = np.searchsorted(ts, starts, side='left')
i1 = np.searchsorted(ts, stops, side='left')
...
for k in sel:
    a, b = i0[k], i1[k]
    T = b - a
    neural.append(np.ascontiguousarray(traces[:, a:b]))
    out = np.empty((5, T), dtype=np.int16)
    out[0] = image_all[a:b]
    out[1] = change_all[a:b]
    out[2] = run_bin_all[a:b]
    out[3] = pup_bin_all[a:b]
    out[4] = outcome[k]
```

iii. <Justification>

Step 10 Check 3(c): *"whitepaper: all clocks recorded on one 100 kHz IO board, so every SDK timestamp
is on a single session clock … identical window `[start_time, stop_time)`; every stream resampled
onto `ophys_timestamps` by linear interpolation (running, pupil) or interval lookup (stimulus). No
time shift is introduced anywhere."* Alignment was checked visually in
`processing_<oeid>.png` (raw 60 Hz running / 30 Hz pupil traces overlaid on the resampled traces, the
stimulus flash spans shaded, and the trials-table `change_time` drawn as a red line falling exactly
where `image_change` rises) and numerically by re-deriving the trial index ranges from raw h5py.

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. <Decisions>

**No rebinning, no resampling, no smoothing of the neural data.** The converted time base *is* the
native ophys frame grid: 32.32 ms bins (30.9406 Hz). Metadata stores
`time_bin_size = 32.32` ms (median frame period across sessions) plus the observed range
[32.3100, 32.3300] ms, and the script asserts the period is uniform across all sessions to within
1e-4 s — which is exactly why the 10.73 Hz Multiscope project was excluded. All other streams
(60 Hz running, 30 Hz eye tracking, stimulus table) are resampled *onto* this grid, not the reverse.

ii. <Code snippets>

```python
'frame_period_s': float(np.median(np.diff(ts))),
...
frame_periods = np.array([r['info']['frame_period_s'] for r in kept])
'time_bin_size': float(np.median(frame_periods) * 1000.0),   # ms
'time_bin_size_range_ms': [float(frame_periods.min() * 1000),
                           float(frame_periods.max() * 1000)],
...
assert np.isclose(frame_periods.min(), frame_periods.max(), atol=1e-4), \
    'frame period is not uniform across sessions'
```

iii. <Justification>

Step 5: *"Time base: the ophys frame timestamps inside that window (`ophys_timestamps`). All other
streams are resampled onto those timestamps. Bin size = 0.0323193 s (30.9406 Hz), identical for
every trial and session."* Step 10 Check 3(d): *"no binning in the reference — the SDK's native ophys
frame grid"*, matched. Step 9 consistency table cross-checks the rate against the whitepaper's
"31 Hz for single plane" and `metadata.ophys_frame_rate`.

---

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. <Decisions>

From the **`stimulus_presentations` table** (restricted to the `change_detection` stimulus block):
`start_time`, `end_time`, `image_name` and `omitted`. It is *not* derived from the trials table's
`initial_image_name` / `change_image_name`. For every ophys frame the AI looks up the flash that is
on screen at that instant; frames not inside any flash (the 500 ms grey inter-stimulus interval, and
omitted flashes, whose `end_time` is NaN) get their own class, `grey` (code 0).

ii. <Code snippets>

```python
sp = stimulus_presentations
sp = sp[sp['stimulus_block_name'].astype(str).str.contains('change_detection')]
start = sp['start_time'].values.astype(np.float64)
end = sp['end_time'].values.astype(np.float64)
names = sp['image_name'].astype(str).values
is_change = sp['is_change'].values.astype(bool)
order = np.argsort(start)
start, end, names, is_change = start[order], end[order], names[order], is_change[order]

codes = np.array([IMAGE_CODE.get(n, 0) for n in names], dtype=np.int16)

idx = np.searchsorted(start, ts, side='right') - 1
valid = idx >= 0
idx_c = np.clip(idx, 0, len(start) - 1)
on_screen = valid & (ts < np.nan_to_num(end[idx_c], nan=-np.inf))

image = np.where(on_screen, codes[idx_c], 0).astype(np.int16)
```

iii. <Justification>

Step 5 Key Decision 6: *"Image identity includes a `grey` class — the spec defines it as 'the image
presented during the non-grey screen', so the grey inter-stimulus interval (and omitted flashes,
which are grey continuations) need their own category."* Step 10 Check 3(f): the flash spans come
from *"exactly those flash spans"* that the reference tutorial's `plot_stimuli` shades. The AI
cross-checked the resulting on-screen fraction (0.3351) against the independently integrated flash
durations (0.3349) and against the nominal 250/750 ms duty cycle (0.3333).

---

## 3-b. What processing is involved in computing `output` *Image identity*?

i. <Decisions>

Image names are mapped to a **fixed global 17-value code book**: 0 = `grey`, 1–16 = the 16 natural
images of image sets A and B, in sorted name order. The list is hard-coded (and was verified against
the data) rather than being built from whatever appeared in the loaded sessions, so codes are
identical regardless of which sessions are converted. Per-session `image_index` (0–7) is deliberately
*not* used. The result is a per-frame `int16` row of the `(5, T)` output array.

ii. <Code snippets>

```python
IMAGE_NAMES = ['im000', 'im031', 'im035', 'im045', 'im054', 'im061', 'im062',
               'im063', 'im065', 'im066', 'im069', 'im073', 'im075', 'im077',
               'im085', 'im106']
IMAGE_VALUES = ['grey'] + IMAGE_NAMES
IMAGE_CODE = {name: i + 1 for i, name in enumerate(IMAGE_NAMES)}
...
codes = np.array([IMAGE_CODE.get(n, 0) for n in names], dtype=np.int16)
image = np.where(on_screen, codes[idx_c], 0).astype(np.int16)
```

iii. <Justification>

Step 5 Key Decision 6: *"16 image names are kept globally distinct rather than collapsed to the
per-session 0–7 `image_index`, because image sets A and B are different physical stimuli."*
Consequence documented in Step 10 Check 1: the globally shared 17-class readout means the decoder can
predict an image-set-B class on an image-set-A session, which produces a benign sklearn
`y_pred contains classes not in y_true` warning — the AI judged this unavoidable and not a data
defect. Resulting distribution: grey 0.669, each image ≈0.020–0.021.

---

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. <Decisions>

The full-session per-frame image code is computed once on the same `ophys_timestamps` vector used for
the neural traces, then sliced with the identical `a:b` trial index range. Alignment is therefore
exact by construction — there is no separate interpolation or offset. A frame is non-grey iff its
timestamp lies in `[flash.start_time, flash.end_time)`.

ii. <Code snippets>

```python
image_all, change_all, sp = stimulus_traces(
    ds.stimulus_presentations, ts, job.get('change_window', 'flash'))
...
out[0] = image_all[a:b]
```

iii. <Justification>

Step 7 plot review, panel 3: *"the code is non-grey exactly inside a non-omitted flash and grey
everywhere else; `image_change` rises exactly at the red `change_time` line and the identity code
changes at the same frame."* Step 10 Check 2: the whole-session image-identity trace was rebuilt
independently from raw h5py (`intervals/*_presentations/{start_time,stop_time,image_name}`) and was
**100 % identical**. The AI also recovered a 258.6 ms mean flash duration (8 frames × 32.3 ms) from
the converted trace, matching the whitepaper's 250 ms.

---

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. <Decisions>

From `stimulus_presentations.is_change` (plus the same flash `start_time` / `end_time` spans). The
SDK sets `is_change = True` only on the flash at which the image identity actually changed; catch
trials carry `is_sham_change` instead and therefore get 0 everywhere. The trials-table `change_time`
/ `go` columns are used only as a cross-check, not to build the variable.

ii. <Code snippets>

```python
is_change = sp['is_change'].values.astype(bool)
...
if change_window == 'interval':
    change = (valid & is_change[idx_c]).astype(np.int16)
else:
    change = (on_screen & is_change[idx_c]).astype(np.int16)
```

iii. <Justification>

Step 5 variable mapping: *"`stimulus_presentations.is_change` → `output[1]` `image_change` … binary;
catch trials have `is_sham_change` and correctly get 0."* Sanity check (Step 10 Check 2): *"one
change per GO trial / none per CATCH trial — PASS in every session"*, and the whole-session
`image_change` trace rebuilt from raw h5py `is_change` was 100 % identical.

---

## 4-b. What processing is involved in computing `output` *Image change*?

i. <Decisions>

None beyond the per-frame indicator: the binary trace is 1 on exactly the frames of the changed
flash. It is stored as an `int16` row of the `(5, T)` output array, with value names
`['no_change', 'change']`. Overall 2.57 % of timepoints are 1, which the AI showed matches the
analytic expectation P(go) × 0.2511 s / mean trial length = 0.875 × 0.2511 / 8.47 = 0.0259.

ii. <Code snippets>

```python
change = (on_screen & is_change[idx_c]).astype(np.int16)
...
out[1] = change_all[a:b]
```

iii. <Justification>

Step 9 footnote 3: *"expected = P(go) × 0.2511 s / mean trial length = 0.875 × 0.2511 / 8.47 =
0.0259"* vs converted 0.0257.

---

## 4-c. How is `output` *Image change* thresholded into categories?

i. <Decisions>

Binary, with the positive class defined as **the 250 ms during which the changed image is actually on
screen** (typically 8 ophys frames). The AI implemented and benchmarked the alternative definition —
the full 750 ms image-presentation interval (flash + following grey), which is the unit the reference
paper assigns events to — as `--change-window interval`, and kept the 250 ms flash.

ii. <Code snippets>

```python
if change_window == 'interval':
    # the whole 750 ms image-presentation interval that starts at the change
    change = (valid & is_change[idx_c]).astype(np.int16)
else:
    # only the 250 ms during which the changed image is actually on screen
    change = (on_screen & is_change[idx_c]).astype(np.int16)
```

iii. <Justification>

Step 7 "Change-window benchmark (why the 250 ms flash)": 20-session validation balanced accuracy
0.640 (250 ms) vs 0.624 (750 ms) for `image_change`, with the other outputs unchanged. *"The 250 ms
flash both decodes better and is the reading consistent with how `image_identity` is defined ('the
image presented during the non-grey screen')."* The AI also noted (Step 12 Check 1) that this makes
the label harder than the paper's, because the GCaMP6f response peaks 200–400 ms after the flash.

---

## 4-d. How is `output` *Image change* aligned with the neural data?

i. <Decisions>

Identically to image identity: computed once per session on the `ophys_timestamps` grid, then sliced
with the same `a:b` trial index range as the neural matrix. No interpolation or shift.

ii. <Code snippets>

```python
image_all, change_all, sp = stimulus_traces(ds.stimulus_presentations, ts, ...)
...
out[1] = change_all[a:b]
```

iii. <Justification>

Same as 3-c. Additionally verified in the diagnostic figure (panel 3 and panel 7), where the
`image_change` pulse is plotted against the independent `trials.change_time` red line and against the
shaded stimulus spans, and in Step 10 Check 2 ("one change per GO trial / none per CATCH trial").

---

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. <Decisions>

`ds.running_speed` — the SDK's 60 Hz running-wheel stream, columns `timestamps` and `speed` (cm/s).
The AI relies on the SDK's own preprocessing (unwrapping, transient correction, 10 Hz low-pass) and
adds none of its own.

ii. <Code snippets>

```python
def resample_running(running_speed, ts):
    rt = running_speed['timestamps'].values.astype(np.float64)
    rv = running_speed['speed'].values.astype(np.float64)
    good = np.isfinite(rt) & np.isfinite(rv)
    rt, rv = rt[good], rv[good]
    order = np.argsort(rt)
    return np.interp(ts, rt[order], rv[order])
```

iii. <Justification>

Docstring: *"The SDK already returns the unwrapped / transient-corrected / 10 Hz low-passed speed, so
nothing but resampling is required."* Step 10 Check 3(f): the reference tutorial's `plot_running`
uses the same `running_speed.speed`.

---

## 5-b. What processing is involved in computing `output` *Running speed*?

i. <Decisions>

Two steps: (1) drop non-finite samples, sort by time, and **linearly interpolate onto the ophys frame
timestamps** (`np.interp`, once per session for the whole session); (2) discretise into 5 equal
percentile (quintile) bins. Nothing else — no smoothing, no absolute value, no clipping.

ii. <Code snippets>

```python
run_all = resample_running(ds.running_speed, ts)
...
frame_idx = np.concatenate([np.arange(i0[k], i1[k]) for k in sel])
run_bin_all = np.zeros(len(ts), dtype=np.int16)
rb, run_edges = quantile_bins(run_all[frame_idx])
run_bin_all[frame_idx] = rb
```

iii. <Justification>

Step 5 mapping table: *"linear interpolation onto ophys timestamps → per-session quintile bins."*
Verified in Step 7 plot panel 1 (*"raw 60 Hz running speed vs. the ophys-resampled trace — curves
superimpose exactly"*) and in Step 10 Check 2, where the whole running-speed bin trace was rebuilt
from raw h5py `processing/running/speed/{timestamps,data}` and was **100.0000 % identical**, with
`np.allclose` bin edges.

---

## 5-c. How is `output` *Running speed* thresholded into categories?

i. <Decisions>

Five equal-percentile bins, with the quintile edges computed **per session**, and computed over
**exactly the timepoints that enter the dataset** (the concatenation of the kept trial windows), not
over the whole session and not globally over all sessions. Bin index = `searchsorted(edges, value,
'right')` ∈ {0..4}. Result: exactly 0.200 of timepoints in each bin, both per session and overall.

ii. <Code snippets>

```python
def quantile_bins(values, nbins=N_QUANTILE_BINS):
    """Assign `values` to `nbins` equal-percentile bins computed from `values`."""
    edges = np.quantile(values, np.arange(1, nbins) / nbins)
    return np.searchsorted(edges, values, side='right').astype(np.int16), edges
```

```python
frame_idx = np.concatenate([np.arange(i0[k], i1[k]) for k in sel])
rb, run_edges = quantile_bins(run_all[frame_idx])
run_bin_all[frame_idx] = rb
```

iii. <Justification>

Step 5 Key Decision 8: *"Quintile bins for running speed and pupil are computed per session over
exactly the timepoints that enter the dataset. Rationale: pupil size is in camera pixels and depends
on zoom/eye position/rig, so it is not comparable across sessions — global bins would mostly encode
session identity rather than arousal. Per-session bins also guarantee the 'five equal percentile
bins' property holds within every session and give a balanced 20/20/20/20/20 target."* The edges are
stored per session in `metadata['session_info'][i]['running_quintile_edges']`.

---

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. <Decisions>

The interpolation target is the full `ophys_timestamps` vector, so the binned running trace lives on
the same grid as the neural matrix and is sliced with the same `a:b` index range. No per-trial
interpolation and therefore no per-trial alignment error. The AI audited that no trial window falls
outside the running-speed timestamp range (worst margin 300 s before / 601 s after), so no
extrapolation occurs.

ii. <Code snippets>

```python
run_all = resample_running(ds.running_speed, ts)
...
out[2] = run_bin_all[a:b]
```

iii. <Justification>

Step 10 Check 3(c): all streams share one 100 kHz hardware clock, so a linear interpolation onto the
ophys timestamps is the correct (and shift-free) resampling. Step 10 Check 5 confirms no trial window
extends beyond the running-speed range.

---

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. <Decisions>

`ds.eye_tracking['pupil_area']` (the fitted-ellipse area from the 30 Hz eye camera) together with its
`timestamps`. Blink samples are removed by dropping non-finite / non-positive areas; the AI verified
that `pupil_area` is NaN **exactly** where the SDK's `likely_blink` flag is True, so this is
equivalent to masking on `likely_blink`. The area is converted to an effective diameter
`2·sqrt(area/π)`.

ii. <Code snippets>

```python
def resample_pupil(eye_tracking, ts):
    t = eye_tracking['timestamps'].values.astype(np.float64)
    a = eye_tracking['pupil_area'].values.astype(np.float64)
    good = np.isfinite(t) & np.isfinite(a) & (a > 0)
    t, a = t[good], a[good]
    order = np.argsort(t)
    t, a = t[order], a[order]
    diam = 2.0 * np.sqrt(a / np.pi)
    return np.interp(ts, t, diam), t
```

iii. <Justification>

Step 5 Key Decisions 9–10: *"Pupil diameter = `2*sqrt(pupil_area/π)` (effective diameter of the
fitted ellipse). Monotonic in area, so quintiles are identical to area quintiles; reported as a
diameter as requested."* and *"`pupil_area` is NaN exactly where `likely_blink`; these are removed
and linearly interpolated from the surrounding valid samples (standard blink handling). Trials with
no valid sample at all are dropped."* Step 10 Check 3(f): the reference tutorial's `plot_pupil` also
uses `eye_tracking.pupil_area`.

---

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. <Decisions>

(1) Blink removal (drop NaN/non-positive area); (2) area → effective diameter `2·sqrt(A/π)`;
(3) linear interpolation of the remaining valid samples onto the ophys frame timestamps — which also
performs the standard blink-gap interpolation; (4) quintile discretisation. The function additionally
returns the valid-sample timestamps so that trials containing no real measurement can be rejected
(see 1-e). Blink fraction per session is recorded in metadata.

ii. <Code snippets>

```python
pupil_all, pupil_valid_t = resample_pupil(eye, ts)
...
pb, pup_edges = quantile_bins(pupil_all[frame_idx])
pup_bin_all[frame_idx] = pb
...
'pupil_blink_fraction': float(np.mean(~np.isfinite(eye['pupil_area'].values))),
```

iii. <Justification>

Step 7 plot review panel 2: *"raw 30 Hz pupil samples vs. the blink-interpolated ophys-resampled
trace — the interpolated line passes through every raw sample."* Step 10 Check 2: the whole-session
pupil bin trace was rebuilt from raw h5py `acquisition/EyeTracking/pupil_tracking/area` +
`likely_blink`, with the same diameter conversion, interpolation and quantiles, and was
**100.0000 % identical** with `np.allclose` edges.

---

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. <Decisions>

Same scheme as running speed: five equal-percentile bins, edges computed **per session** over exactly
the timepoints that enter the dataset. Observed distribution 0.200 per bin.

ii. <Code snippets>

```python
pb, pup_edges = quantile_bins(pupil_all[frame_idx])
pup_bin_all[frame_idx] = pb
...
'pupil_quintile_edges': pup_edges.tolist(),
```

iii. <Justification>

Step 5 Key Decision 8 (quoted in 5-c). The pupil-specific part of the argument is the stronger one:
*"pupil size is in camera pixels and depends on zoom/eye position/rig, so it is not comparable across
sessions — global bins would mostly encode session identity rather than arousal."* Metadata records
the units as *"effective pupil diameter 2*sqrt(pupil_area/pi) in camera pixels (quintile-binned per
session)"*.

---

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. <Decisions>

Identical to running speed: interpolated once onto the full `ophys_timestamps` vector, then sliced
with the same `a:b` trial index range. The AI verified that no trial window falls outside the valid
eye-tracking range (worst margin 278 s / 588 s), and drops any trial with no non-blink sample inside
it, so a trial's pupil trace is never pure extrapolation.

ii. <Code snippets>

```python
pupil_all, pupil_valid_t = resample_pupil(eye, ts)
...
j0 = np.searchsorted(pupil_valid_t, starts, side='left')
j1 = np.searchsorted(pupil_valid_t, stops, side='left')
...
out[3] = pup_bin_all[a:b]
```

iii. <Justification>

Step 10 Check 3(c) (single hardware clock ⇒ direct interpolation is correct) and Check 5 (edge-case
audit of stream coverage). Step 5 rule 4 justifies the per-trial pupil-validity requirement.

---

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. <Decisions>

The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`,
`correct_reject`, in that fixed order. An assertion enforces that every kept go/catch trial has
exactly one True among them.

ii. <Code snippets>

```python
OUTCOME_VALUES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome_cols = np.stack([tr['hit'].values.astype(bool),
                         tr['miss'].values.astype(bool),
                         tr['false_alarm'].values.astype(bool),
                         tr['correct_reject'].values.astype(bool)], axis=1)
assert outcome_cols.sum(axis=1).min() == 1 and outcome_cols.sum(axis=1).max() == 1, \
    f'{oeid}: go/catch trial without a unique outcome'
outcome = np.argmax(outcome_cols, axis=1).astype(np.int16)
```

iii. <Justification>

Step 5: *"trials = `go | catch`, which is exactly the union of those four outcomes (verified:
`hit+miss+FA+CR == n(go∪catch)` in every session)."* Step 10 Check 3(f): the reference tutorial
selects trial types with `.query('hit')`, `.query('miss')`, etc. — the same four flags. Converted
per-trial distribution hit 0.3196 / miss 0.5550 / FA 0.0191 / CR 0.1062, matching a direct scan of
the raw trials tables.

---

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. <Decisions>

`np.argmax` over the four boolean columns gives a code 0–3; the code is then **broadcast across every
timepoint of the trial** so it occupies row 4 of the same `(5, T)` output array as the four
time-varying outputs. Value names `['hit', 'miss', 'false_alarm', 'correct_reject']`.

ii. <Code snippets>

```python
outcome = np.argmax(outcome_cols, axis=1).astype(np.int16)
...
out[4] = outcome[k]
```

iii. <Justification>

Step 5 Key Decision 11: *"Trial outcome is stored time-varying (constant across the trial's
timepoints) so it lives in the same `(5, T)` array as the other four outputs, per the spec's 'If at
all possible, make it time-varying'."* The consequence is discussed honestly in Step 12 Check 3: the
train/val gap for this output (1.49) is the largest, because the effective sample size is the number
of trials while the model sees timepoints.

---

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. <Decisions>

- **Sessions with no eye tracking** (empty table or all-NaN `pupil_area`): skipped with a logged
  reason (3 sessions).
- **Blink samples**: removed and interpolated over.
- **Trials with no valid pupil sample**: dropped (60 trials) rather than filled — so no fabricated
  pupil label enters the dataset.
- **Omitted flashes** (`end_time` NaN): mapped to `grey`, since the screen really is grey.
- **Frames before the first flash** (`searchsorted` index −1): mapped to `grey` via the `valid` mask.
- **Non-finite running/eye samples**: dropped before interpolation.
- **Degenerate sessions/trials**: `< 2` ophys frames per trial or `< 2` trials or 0 ROIs per session →
  dropped (none actually occurred, but guarded).
- **Plot failures**: caught so they cannot kill a conversion run.
- **Assertions** on unique trial outcome and on uniform frame period across sessions.
- A dedicated edge-case audit over all 165 sessions checked for NaN `change_time`, duplicate/NaN
  `cell_specimen_id`, NaN/Inf dF/F, non-monotonic timestamps, overlapping trials, unknown image
  names, and trial windows outside each stream's coverage — all clean.

ii. <Code snippets>

```python
if eye is None or len(eye) == 0 or not np.isfinite(eye['pupil_area'].values).any():
    return {'oeid': int(oeid), 'skip': 'no eye tracking data'}
```

```python
good = np.isfinite(t) & np.isfinite(a) & (a > 0)
t, a = t[good], a[good]
```

```python
on_screen = valid & (ts < np.nan_to_num(end[idx_c], nan=-np.inf))
image = np.where(on_screen, codes[idx_c], 0).astype(np.int16)
```

```python
if i1[k] - i0[k] < MIN_FRAMES_PER_TRIAL:
    n_drop_short += 1
    continue
if j1[k] - j0[k] < 1:
    n_drop_pupil += 1
    continue
```

```python
except Exception as exc:                      # plotting must not kill a run
    print(f'  [warn] plotting failed for {oeid}: {exc!r}', flush=True)
```

iii. <Justification>

Step 10 Check 5 table enumerates each edge case and its handling, e.g. *"frames before the first
flash (`searchsorted` index −1) → handled → grey"*, *"omitted flashes → mapped to grey (screen really
is grey)"*, *"session with <2 usable trials, or 0 neurons — none occurred; guarded anyway"*. Every
drop is counted and reported in `metadata['session_info']`
(`n_trials_dropped_no_pupil`, `n_trials_dropped_short`) and `metadata['skipped_sessions']`, so no
data loss is silent.

---

## 9-a. What are the most time-consuming steps of the code?

i. <Decisions>

The AI instrumented the code with per-stage timers (`open`, `neural`, `behavior`, `assemble`, `total`)
printed for every session. Opening the ~0.9 GB NWB file and materialising the
`(n_cells × n_frames)` trace matrix dominates completely: 2.7–6.4 s per session, versus ~0.05 s for
resampling all behaviour/stimulus streams and ~0.05 s for assembling trials. The second cost is I/O:
pickling the 8.38 GB output takes 9.2 s. Total: 51 s of processing (16 workers) + 9 s write = 61 s.

ii. <Code snippets>

```python
t0 = time.time()
ds = BehaviorOphysExperiment.from_nwb_path(path)
ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
timing['open'] = time.time() - t0
...
print(f'[{n}/{len(jobs)}] {res["oeid"]}: ... '
      f'(open {tm["open"]:.1f}s neural {tm["neural"]:.1f}s '
      f'behav {tm["behavior"]:.1f}s asm {tm["assemble"]:.1f}s '
      f'tot {tm["total"]:.1f}s)', flush=True)
```

iii. <Justification>

Step 6: *"Opening the NWB and materialising the (n_cells x n_frames) trace matrix dominates runtime
(~3-6 s/session, ~0.9 GB files)."* Step 7 run-time table gives the measured breakdown and the
resulting 65 s estimate, well inside the 15-minute budget, so no further optimisation was pursued.

---

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. <Decisions>

Most loops were already removed by design. What remains:
- **Per-neuron trace copy** `for i, cid in enumerate(cell_ids): traces[i] = col.loc[cid]` — one
  pandas `.loc` lookup per cell (up to 666 per session); `np.vstack(col.values)` (what the reference
  solution uses) or a single reindex would avoid the repeated index lookups. This sits inside the
  dominant cost.
- **Per-trial selection loop** `for k in range(len(tr))` applying the two drop rules — fully
  vectorisable as a boolean mask over the already-vectorised `i0/i1/j0/j1` arrays.
- **Per-trial assembly loop** `for k in sel` — unavoidable, since trials have different lengths and
  the target format is a list of arrays.
- `np.concatenate([np.arange(i0[k], i1[k]) for k in sel])` builds `frame_idx` with a Python
  comprehension over trials.

ii. <Code snippets>

```python
traces = np.empty((len(cell_ids), len(ts)), dtype=np.float32)
for i, cid in enumerate(cell_ids):
    traces[i] = col.loc[cid]
```

```python
for k in range(len(tr)):
    if i1[k] - i0[k] < MIN_FRAMES_PER_TRIAL:
        n_drop_short += 1
        continue
    if j1[k] - j0[k] < 1:
        n_drop_pupil += 1
        continue
    sel.append(k)
```

```python
i0 = np.searchsorted(ts, starts, side='left')   # already vectorised over all trials
i1 = np.searchsorted(ts, stops, side='left')
```

iii. <Justification>

Step 6 "Code speedups added": *"Every stream is resampled once per session onto the full ophys
timestamp vector; trials are then pure array slices. Trial boundaries found with two vectorised
`np.searchsorted` calls instead of a loop. `ProcessPoolExecutor(16)` over sessions."* The AI
identified the naive alternative it avoided: *"Naive per-trial `np.interp`/`searchsorted` would
repeat work 250+ times per session."* The residual loops are over ≤666 cells or ≤400 trials and are
negligible next to the 2.7–6.4 s file open, so the AI did not pursue them.

---

## 9-c. What processing does the code repeat multiple times?

i. <Decisions>

Very little is repeated within a run — each session is opened once, each stream resampled once, and
trials are pure slices. Minor repetitions:
- Two independent `searchsorted` passes over trial boundaries (one against `ophys_timestamps`, one
  against the valid pupil sample times).
- The full-session traces are materialised and then **copied again** per trial
  (`np.ascontiguousarray(traces[:, a:b])`), so the neural data exists twice in memory during a
  session (unlike the reference, which also copies but discards the session array immediately).
- In `--show-processing` mode, `make_processing_plot` re-reads `ds.running_speed` and
  `ds.eye_tracking` and recomputes the raw pupil diameter that `resample_pupil` already computed.
- Across the project (not within one run) the full conversion was re-run several times for the
  neural-signal and change-window benchmarks and after adding metadata fields.

ii. <Code snippets>

```python
i0 = np.searchsorted(ts, starts, side='left')
i1 = np.searchsorted(ts, stops, side='left')
j0 = np.searchsorted(pupil_valid_t, starts, side='left')
j1 = np.searchsorted(pupil_valid_t, stops, side='left')
```

```python
neural.append(np.ascontiguousarray(traces[:, a:b]))
```

```python
raw_eye_d = 2.0 * np.sqrt(eye['pupil_area'].values / np.pi)   # recomputed for the plot
```

iii. <Justification>

Step 6: *"Every stream is resampled once per session onto the full ophys timestamp vector; trials are
then pure array slices."* The AI did not flag the remaining repetitions explicitly; they are
immaterial (microseconds, or plotting-only) next to the file-open cost it measured.

---

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. <Decisions>

- **Whole-session resampling and binning.** `run_all`, `pupil_all`, `image_all`, `change_all`,
  `run_bin_all` and `pup_bin_all` are computed for *every* ophys frame of the session (~110 k), but
  only the frames inside kept trials (11.18 M of ~18 M total, i.e. ~40 % discarded) are ever stored.
  This is a deliberate trade: one vectorised pass beats 250+ per-trial interpolations.
- **Full trace matrix.** All `n_cells × n_frames` dF/F values are materialised even though only the
  in-trial columns are kept.
- **Diagnostic metadata** computed per session and never used downstream: `pupil_blink_fraction`,
  `mean_change_latency_s`, `n_aborted`, `n_auto_rewarded`, `n_trials_dropped_*`,
  `running/pupil_quintile_edges`, `cell_specimen_ids`, `trial_ids`. (These exist to support the
  sanity checks and the whitepaper comparisons, not the decoder.)
- **Static outcome replicated across time**: `out[4]` stores one value 262× per trial (11.18 M
  int16s) although it carries only 42,410 bits of information — done to satisfy the spec's preference
  for time-varying outputs.
- **Unused CLI paths** kept in the shipped script: `--neural-signal events/filtered_events`,
  `--zscore`, `--change-window interval`, `--limit` — all benchmark-only.
- `make_processing_plot` renders 7 large panels (only run under `--show-processing`).

ii. <Code snippets>

```python
run_all = resample_running(ds.running_speed, ts)          # whole session
pupil_all, pupil_valid_t = resample_pupil(eye, ts)        # whole session
image_all, change_all, sp = stimulus_traces(ds.stimulus_presentations, ts, ...)
```

```python
run_bin_all = np.zeros(len(ts), dtype=np.int16)           # whole session, mostly unused
pup_bin_all = np.zeros(len(ts), dtype=np.int16)
run_bin_all[frame_idx] = rb
pup_bin_all[frame_idx] = pb
```

```python
'pupil_blink_fraction': float(np.mean(~np.isfinite(eye['pupil_area'].values))),
'mean_change_latency_s': float(np.nanmean(tr['change_time'].values[sel] - starts[sel])),
```

iii. <Justification>

Step 6: the whole-session-then-slice design is listed under "Code speedups added" (*"~250x fewer
interp calls"*), so the discarded inter-trial frames are an accepted cost of vectorisation. The extra
metadata is justified in Step 10 "Issues Found and Resolved" #2: *"`cell_specimen_ids` / `trial_ids`
were not stored, which made the h5py sanity checks impossible to anchor. Added to
`metadata['session_info']`."* Storage was explicitly optimised instead (*"float32 neural / int16
outputs — 8.4 GB instead of ~17 GB"*).
