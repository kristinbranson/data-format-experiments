# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. <Decisions>

The AI does **not** use the AllenSDK project-cache API. It reads the released project metadata CSV
(`/app/data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv`) to enumerate
experiments, intersects that table with the NWB files actually present locally in
`behavior_ophys_experiments/`, and then loads each experiment with
`BehaviorOphysExperiment.from_nwb_path(path)`. Its stated reason (CONVERSION_NOTES Step 10, Check 3a) is
that "the cache API cannot read this non-cache-layout copy", and that `from_nwb_path` returns the
identical `BehaviorOphysExperiment` object that `cache.get_behavior_ophys_experiment(eid)` would return.

Each NWB file is loaded exactly once, inside a worker process of a `multiprocessing.Pool`
(24 workers used for the full run). From the loaded object it pulls `dff_traces`, `ophys_timestamps`,
`cell_specimen_table`, `stimulus_presentations`, `running_speed`, `eye_tracking`, `trials` and `metadata`.
284 local NWBs → 202 active-behaviour → 168 `VisualBehavior` (single-plane) → 165 kept after the
eye-tracking requirement. 37 mice, 42,470 trials, 28,821 neurons, 11,192,974 timepoints.

ii. <Code snippets>

```python
DATA_ROOT = '/app/data/visual-behavior-ophys-1.1.0'
NWB_DIR = os.path.join(DATA_ROOT, 'behavior_ophys_experiments')
META_DIR = os.path.join(DATA_ROOT, 'project_metadata')

def select_experiments():
    et = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    local = {int(re.findall(r'(\d+)', f)[0]): os.path.join(NWB_DIR, f)
             for f in os.listdir(NWB_DIR) if f.endswith('.nwb')}
    et = et[et.ophys_experiment_id.isin(local.keys())].copy()
    et['path'] = et.ophys_experiment_id.map(local)
    n_local = len(et)
    et = et[~et.session_type.str.contains('passive')]
    n_active = len(et)
    et = et[et.project_code == 'VisualBehavior']
    ...
    et = et.sort_values(['mouse_id', 'date_of_acquisition']).reset_index(drop=True)
    return et
```

```python
def process_session(args):
    path, want_raw, signal, normalize = args
    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
        BehaviorOphysExperiment)
    ...
    ds = BehaviorOphysExperiment.from_nwb_path(path)
    md = ds.metadata
```

```python
    if args.workers > 1 and len(jobs) > 1:
        with Pool(min(args.workers, len(jobs))) as pool:
            for i, r in enumerate(pool.imap(process_session, jobs)):
                _report(i, len(jobs), r, t_proc)
                results.append(r)
```

iii. <Justification>

From CONVERSION_NOTES Step 10 Check 3: the SDK cache object cannot be constructed over this
non-cache-layout local copy, so the AI uses the lower-level `from_nwb_path` constructor, which yields
"an identical object". It sorts by `(mouse_id, date_of_acquisition)` for a deterministic session order,
and parallelises across NWB files because "NWB load (`from_nwb_path`) 2.4 s" is the dominant per-session
cost (Step 7 timing table). The AI cross-checked that the per-session go/catch/hit/miss/FA/CR and neuron
counts it obtains are *identical in all 165 sessions* to `behavior_session_table.csv` and
`ophys_cells_table.csv` (Step 9 consistency table), confirming nothing was lost in loading.

---

## 1-b. How are the data split into subjects?

i. <Decisions>

Subjects are the unique `mouse_id` values, taken from each loaded experiment's `metadata['mouse_id']`
(cast to `str`). The global `subjects` list is the sorted set of mouse ids over the kept sessions, and
`subject_idx` is the index of each session's mouse into that list. 37 mice result, ranging from 2 to 9
sessions each.

ii. <Code snippets>

```python
    res = {'eid': eid, 'mouse': str(md['mouse_id']), ...}
```

```python
    subjects = sorted(set(r['mouse'] for r in good))
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    ...
        subject_idx.append(subj_to_idx[r['mouse']])
    ...
        'subjects': subjects,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
```

iii. <Justification>

`mouse_id` is the Allen identifier for the animal. The AI's Step 9 consistency table records
"Subjects (mice): 37 in the local active single-plane subset → **37** ✓" and "Sessions per subject 2–9
(mean 4.46) ✓", i.e. it verified the subject split against the project metadata table.

---

## 1-c. How are the data split into sessions?

i. <Decisions>

One converted "session" = one `ophys_experiment` = one NWB file. For the `VisualBehavior` project code
this is exact: each ophys session contains a single imaging plane, so experiment and session are
one-to-one (239 experiments ↔ 239 `ophys_session_id`s in the local metadata). The AI nonetheless records
`ophys_session_id` and `ophys_container_id` in `metadata['session_info']`.

Three *session-selection* filters are applied before any data are read:
1. `project_code == 'VisualBehavior'` — drops the 45 local Multiscope (`VisualBehaviorMultiscope`) planes.
2. `~session_type.str.contains('passive')` — drops the `OPHYS_2_*_passive` and `OPHYS_5_*_passive`
   sessions (168 of 239 VisualBehavior experiments survive).
3. Eye tracking must be present and non-empty (3 further sessions dropped).

Result: 165 sessions.

ii. <Code snippets>

```python
    et = et[~et.session_type.str.contains('passive')]
    n_active = len(et)
    et = et[et.project_code == 'VisualBehavior']
    print(f"  local NWB files: {n_local}; active behaviour: {n_active}; "
          f"active single-plane (VisualBehavior): {len(et)}")
```

```python
        session_info.append({
            'ophys_experiment_id': r['eid'], 'ophys_session_id': r['ophys_session_id'],
            'ophys_container_id': r['container'], 'mouse_id': r['mouse'], ...})
```

iii. <Justification>

Step 5 decision 1: the Multiscope rig is excluded because (a) it samples at 11 Hz vs 31 Hz on the
Scientifica rigs and "there is no way to keep both without resampling one of them off its native ophys
timestamps, which the task explicitly asks us to align to"; (b) the local Multiscope data is "34 imaging
planes from only 6 behaviour sessions of a single mouse", so including it would replicate the same six
sets of behavioural labels 34 times across the train/validation split and over-weight one animal.

Passive sessions are excluded with the stated reason "they have no lick spout and no trials at all"
(Step 5 decision 1, repeated in the final summary). The first half of that claim is true; **the second
half is factually wrong** — the passive NWBs do contain a full `trials` table (e.g. experiment 796108483:
402 trials, 351 go, 51 catch, all labelled miss/correct-reject because licking is impossible). The AI
never verified it.

---

## 1-d. How are the data split into trials?

i. <Decisions>

Trials come from the SDK `trials` table. The AI keeps rows where `go == True` or `catch == True`, which is
exactly the complement of aborted + auto-rewarded trials. Each trial spans the half-open interval
`[start_time, stop_time)`, converted to ophys frame indices with `np.searchsorted(..., side='left')`, so
trial length is variable (217–389 frames, mean 262 frames = 8.47 s) while the bin size is constant.
`start_time`, `stop_time` and `change_time` for every kept trial are stored in
`metadata['session_info'][i]`.

ii. <Code snippets>

```python
    trials = ds.trials
    res['n_trials_all'] = int(len(trials))
    res['n_aborted'] = int(trials.aborted.sum())
    res['n_auto_rewarded'] = int(trials.auto_rewarded.sum())
    sel = trials[(trials.go.values | trials.catch.values)].copy()
```

```python
    i0 = np.searchsorted(ots, sel.start_time.values, side='left')
    i1 = np.searchsorted(ots, sel.stop_time.values, side='left')
    ...
    frame_idx = np.concatenate([np.arange(a, b) for a, b in zip(i0, i1)])
    assert np.all(np.diff(frame_idx) > 0), 'trials overlap in time'
```

```python
    for k in range(n_trials):
        a, b = i0[k], i1[k]
        neural.append(np.ascontiguousarray(neural_full[:, a:b]))
        out = np.empty((5, b - a), dtype=np.int8)
```

iii. <Justification>

Step 5 decision 5: "Trial = `[trials.start_time, trials.stop_time)`, i.e. the experiment's own trial
definition, sampled on the native ophys frame times", matching the instruction "segment each recording
session into individual trials based on how they are defined in the experiment" and "include both the Go
and Catch trials, but exclude the Aborted and Auto-rewarded trials". The AI checked in the data that
consecutive trials never overlap and that `stop_time − change_time` is a constant ≈ 4.24 s, "so every
trial contains its (sham-)change and a fixed post-change period", and that mean `change_time − start_time`
= 4.28 s (range 2.79–8.29 s), consistent with the whitepaper's 2.25–8.25 s change-time distribution.
The per-session go (37,143) and catch (5,327) counts match `behavior_session_table.csv` exactly.

---

## 1-e. How are trials filtered based on quality controls?

i. <Decisions>

Trial-level filters:
- Only `go | catch` trials (aborted and auto-rewarded excluded) — see 1-d.
- Trials spanning fewer than 2 ophys frames are dropped (`keep = (i1 - i0) > 1`), with a printed message.
  In the full run no trial was dropped this way (minimum trial length 217 frames).
- No engagement / performance / reaction-time filtering is applied.

Session-level filters (see also 1-c): sessions with missing or empty `eye_tracking`, or with fewer than
100 finite pupil samples, or with fewer than 2 usable trials, are skipped and reported.

Assertions that would abort the run: every go/catch trial has exactly one of hit/miss/FA/CR; trials do not
overlap; trials do not start more than one flash cycle before the first flash of the behaviour block;
hit/miss occur only on go trials and FA/CR only on catch trials.

ii. <Code snippets>

```python
    oc = np.full(len(sel), -1, dtype=np.int8)
    for k, name in enumerate(OUTCOME_NAMES):
        oc[sel[name].values.astype(bool)] = k
    assert np.all(oc >= 0), 'go/catch trial without an outcome'
    assert (sel[OUTCOME_NAMES].values.astype(int).sum(axis=1) == 1).all(), \
        'trial with more than one outcome'
```

```python
    keep = (i1 - i0) > 1
    if not keep.all():
        print(f'    [{eid}] dropping {int((~keep).sum())} trials with <2 ophys frames')
    assert np.all(sel.start_time.values[keep] > f_start[0] - 0.1), \
        'trial starts before the stimulus block'
    sel = sel[keep]; oc = oc[keep]; i0 = i0[keep]; i1 = i1[keep]
    n_trials = len(sel)
    ...
    if n_trials < 2:
        res['skip'] = f'only {n_trials} usable trials'
        return res
```

```python
    try:
        eye = ds.eye_tracking
        if eye is None or len(eye) == 0:
            raise ValueError('empty')
    except Exception as exc:
        res['skip'] = f'no eye tracking ({exc})'
        return res
    ...
    if good_eye.sum() < 100:
        res['skip'] = 'pupil data all NaN'
        return res
```

iii. <Justification>

Step 5 decision 10: "**No trial filtering beyond the specified go/catch rule**, and **no
engagement/performance filtering** — neither reference paper excludes disengaged trials." Step 5
decision 8: "Sessions without eye tracking are dropped (3 of 168 …). Pupil diameter is a required output
and cannot be imputed for a whole session." The ≥ 2-frame rule and ≥ 2-trial rule come from the target
format's requirement that "there needs to be at least two trials within each session in order to evaluate
the decoder performance". Step 10 Check 5 verifies `n_trials == n_go + n_catch` per session and
42,470 == 42,470 overall, "nothing silently dropped".

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. <Decisions>

`neural` is `ds.dff_traces.dff` — the Allen pipeline's released ΔF/F traces, one row per
`cell_specimen_id`, stacked into an `(n_neurons, T)` float32 array on the native `ophys_timestamps` grid.
A `--neural {dff,events,filtered_events}` switch exists; the default and the value used for the delivered
dataset is `dff`.

ii. <Code snippets>

```python
    ots = np.asarray(ds.ophys_timestamps, dtype=np.float64)
    if signal == 'dff':
        neural_full = np.vstack(ds.dff_traces.dff.values).astype(np.float32)
    else:
        neural_full = np.vstack(ds.events[signal].values).astype(np.float32)
    assert neural_full.shape[1] == ots.size, 'trace length != ophys timestamps'
```

iii. <Justification>

Step 5 decision 3: dF/F is "the released, baseline-normalised and detrended dF/F; whitepaper section
'DF/F CALCULATION'". The Neuron paper instead used detected calcium `events`, so the AI ran an explicit
A/B: with `events`/`filtered_events` the decoder is far worse on every output (image identity 0.089 /
0.166 vs 0.372 for dF/F) and 103 trials trip the "all neural data is zero" warning, because events are
exactly zero in ~95% of 32 ms frames while "the decoder used here reads one timepoint at a time". The AI
classes this as the task's explicit exception ("except where … training a neural decoder require
otherwise") and documents that `--neural events` reproduces the paper's choice.

---

## 2-b. How is the `neural` data processed?

i. <Decisions>

Essentially no processing beyond what the Allen pipeline already did: stack the per-cell dF/F rows,
cast to `float32`, slice out each trial's frames with `np.ascontiguousarray`. Per-neuron normalisation is
available (`--normalize zscore|noise_std`) but the delivered dataset uses `--normalize none`. No
smoothing, no detrending, no re-binning, no spike deconvolution.

ii. <Code snippets>

```python
NEURAL_SIGNAL = 'dff'              # see CONVERSION_NOTES.md, Step 5 decision 3
NEURAL_NORMALIZE = 'none'          # see CONVERSION_NOTES.md, Step 5 decision 4
```

```python
    if normalize == 'zscore':
        sd = neural_full.std(axis=1, keepdims=True)
        neural_full = neural_full / np.maximum(sd, 1e-6)
    elif normalize == 'noise_std':
        sd = ds.events['noise_std'].values.astype(np.float32)[:, None]
        neural_full = neural_full / np.maximum(sd, 1e-6)
```

```python
        neural.append(np.ascontiguousarray(neural_full[:, a:b]))
```

iii. <Justification>

Step 5 decision 4: "dF/F is already a normalised quantity; per-neuron unit-variance scaling changes
validation accuracy by ≤ 0.02 (A/B table), so the raw released values are kept." float32 is used because
it "halves memory vs the float64 in the NWB" (8.3 GB instead of ~17 GB). Step 10 Check 2 re-derived three
random `(trial, neuron, timepoint)` samples and a whole `(n_neurons × T)` trial block straight from the
NWB HDF5 with `h5py` and matched them with `np.allclose` (max abs diff 9.2e-9, float32 rounding).

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. <Decisions>

No neuron-level filtering is applied. Instead the AI *asserts* that the released data contains only valid
ROIs (`cell_specimen_table.valid_roi.all()`), that the number of dF/F rows equals the number of rows in
the cell specimen table, and that the trace length equals the number of ophys timestamps. Neuron counts
range 6–666 per session (28,821 total).

ii. <Code snippets>

```python
    cst = ds.cell_specimen_table
    assert bool(cst.valid_roi.all()), 'invalid ROIs present in released data'
    assert len(cst) == neural_full.shape[0]
    res['n_neurons'] = neural_full.shape[0]
    res['dt_median'] = float(np.median(np.diff(ots)))
    res['cell_specimen_ids'] = cst.index.values.astype(np.int64)
```

iii. <Justification>

Step 5 decision 10: "**No neuron filtering**: only `valid_roi == True` cells exist in the released NWBs
(verified in all 165 sessions), the upstream pipeline already removed border/union/duplicate/non-somatic
ROIs, and neither paper applies a further cut." Step 9 confirms the per-session neuron counts are
identical to `ophys_cells_table.csv` in 165/165 sessions, and Step 10 Check 2 confirms the dF/F row
ordering is the `cell_specimen_table` ordering.

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. <Decisions>

Everything is aligned on `ophys_timestamps`, as the instructions require. Each trial's frames are those
with `start_time ≤ t < stop_time`, found by `np.searchsorted(ophys_timestamps, …, side='left')`; the
neural matrix is the corresponding column slice, and *all* output rows are sliced with the same `[a:b]`
indices, so neural and outputs are aligned by construction. The metadata records
`temporal_alignment_event = 'trial start (trials.start_time …)'`, `off_start = 0.0`, `off_end = None`
(variable length). Behavioural streams are brought onto the same grid with `np.interp` before slicing.

ii. <Code snippets>

```python
    i0 = np.searchsorted(ots, sel.start_time.values, side='left')
    i1 = np.searchsorted(ots, sel.stop_time.values, side='left')
    ...
    for k in range(n_trials):
        a, b = i0[k], i1[k]
        neural.append(np.ascontiguousarray(neural_full[:, a:b]))
        out = np.empty((5, b - a), dtype=np.int8)
        out[0] = image_per_frame[a:b]
        out[1] = change_per_frame[a:b]
        out[2] = speed_bin[a:b]
        out[3] = pupil_bin[a:b]
        out[4] = oc[k]
```

```python
            'temporal_alignment_event':
                'trial start (trials.start_time from the AllenSDK trials table); each '
                'trial spans [start_time, stop_time) and is sampled on the native ophys '
                'frame times (ophys_timestamps), so trial length varies',
            'off_start': 0.0,
            'off_end': None,
```

iii. <Justification>

The instruction says "Temporally align based on ophys timestamp", and Step 10 Check 3c notes that all
streams are already on the same 100 kHz sync clock in the SDK, so `ophys_timestamps`,
`running_speed.timestamps` and `eye_tracking.timestamps` are directly comparable seconds. The AI's
alignment sanity check is the change-triggered average of the population mean: it "peaks 0.05–0.3 s
*after* the change, never before" (Step 7 plot review, panel 7 of `processing_*.png`), plus a
flash-triggered average panel. Edge case documented in Step 10 Check 5: trials start ~21 ms before a
flash onset, so the first frame of a trial inherits the previous flash's identity — which is the same
image, and this is asserted.

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. <Decisions>

No rebinning of any kind. The data stay on the native ophys frame grid of the Scientifica single-plane
rigs: 31 Hz, median inter-frame interval 32.31–32.33 ms per session. The reported `time_bin_size` is the
mean over sessions of the per-session median Δt = **32.3193 ms**. Keeping one common bin size is the
stated reason for dropping the 11 Hz Multiscope sessions.

ii. <Code snippets>

```python
    res['dt_median'] = float(np.median(np.diff(ots)))
```

```python
    dt = float(np.mean([r['dt_median'] for r in good]))
    data = { ...
        'metadata': { ...
            'time_bin_size': dt * 1000.0,
            ...
            'mean_frame_interval_s': dt,
```

iii. <Justification>

Step 5 decision 5: "Trial length therefore varies (217–389 frames, mean 262 = 8.47 s); the time-bin size
(32.32 ms) is the same for every trial and session." Step 10 Check 3d: "ophys frames (31 Hz) are the
native bins; the paper aggregates them into 750 ms image presentation intervals / 400 ms windows for *its*
analyses; native ophys frames kept (task: 'temporally align based on ophys timestamp'); the 750 ms
interval convention is used for the *labels*, as in the paper." Step 9 verifies 31 Hz / 32.31–32.33 ms
against the whitepaper.

---

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. <Decisions>

From `ds.stimulus_presentations`, restricted to `stimulus_block_name == 'change_detection_behavior'`,
using the columns `start_time`, `image_name` and `omitted`. It is **not** derived from the trials table's
`initial_image_name` / `change_image_name` (which is what the human reference used) — although the AI
asserts at conversion time that the two definitions agree on every one of the 42,470 trials.

ii. <Code snippets>

```python
    stim = ds.stimulus_presentations
    flashes = stim[stim.stimulus_block_name == 'change_detection_behavior'].copy()
    flashes = flashes.sort_values('start_time')
    f_start = flashes.start_time.values.astype(np.float64)
    f_omitted = flashes.omitted.values.astype(bool)
    f_change = flashes.is_change.values.astype(bool)
    f_name = flashes.image_name.values.astype(str)
```

```python
        # before the change the trial shows `initial_image_name`; after a real
        # change it shows `change_image_name`; catch trials never change
        if pre.any() and not np.all(ident[pre] == img_to_local[init_img[k]]):
            n_bad_id += 1
        if post.any():
            want = img_to_local[chg_img[k] if go[k] else init_img[k]]
            if not np.all(ident[post] == want):
                n_bad_id += 1
```

iii. <Justification>

Step 5 decision 6 cites the paper's analysis unit: "the 750 ms interval beginning with each image
presentation … for image omissions we used the 750 ms following the time of the omission", so each ophys
frame is labelled with the image of the flash interval it falls in. Deriving it from the flash table
rather than from the trial summary makes the label follow the actual stimulus timeline and makes the 5%
omitted flashes explicit. Step 10 Check 2 independently re-derived image identity by brute-force scanning
the HDF5 stimulus table (last non-omitted flash at or before t) at 40 timepoints per session: PASS.

---

## 3-b. What processing is involved in computing `output` *Image identity*?

i. <Decisions>

Three stages:
1. Per session, the 8 non-omitted image names are sorted alphabetically into a local vocabulary and each
   flash is mapped to its local index.
2. Omitted flashes are forward-filled with the previous flash's image ("the ongoing image is carried
   forward"), so identity changes only at real image changes.
3. Each ophys frame takes the image of the flash interval it falls in, giving a piecewise-constant
   per-frame label held across the 250 ms image + 500 ms grey ISI.
4. At assembly, each session's local indices are remapped onto the **global 16-image vocabulary**
   (the union of image set A and image set B, which are disjoint), stored as `output_values[0]`.

ii. <Code snippets>

```python
    images = sorted(str(x) for x in set(f_name[~f_omitted]))
    res['images'] = images
    img_to_local = {n: i for i, n in enumerate(images)}

    local_idx = np.array([img_to_local.get(n, -1) for n in f_name], dtype=np.int16)
    carry = local_idx.copy()
    for i in range(1, carry.size):          # forward-fill through omissions
        if f_omitted[i]:
            carry[i] = carry[i - 1]
    assert np.all(carry >= 0), 'image identity undefined at start of block'
```

```python
        remap = np.array([image_vocab.index(n) for n in r['images']], dtype=np.int8)
        for o in r['output']:
            o[0] = remap[o[0]]
```

```python
    image_vocab = sorted(set(sum([r['images'] for r in good], [])))
```

iii. <Justification>

Step 5 decision 2: both image sets and all experience levels are kept, because "the papers restrict
*their* analyses to familiar images, but that is an analysis choice for a novelty-specific question; the
decoder task says only 'the Visual Behavior task' and dropping half the sessions would halve the data.
The two 8-image sets are disjoint, so `output_values[0]` is the 16-image union and only 8 classes occur in
any one session (this is fine: balanced accuracy is computed per class over the sessions in which the
class occurs)." Step 5 decision 6 justifies the omission carry-forward: it "keeps outputs 0 and 1 mutually
consistent ('image change = 1 right after a change in image identity')". Step 9 verifies 8 images per
session and 16 classes overall.

---

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. <Decisions>

A whole-session per-frame array `image_per_frame` is built once by mapping every ophys timestamp onto its
flash interval with `np.searchsorted(f_start, ots, side='right') - 1`, then clipping the index to the
valid flash range. Trials then slice this array with exactly the same `[a:b]` indices used for the neural
matrix, so alignment is identical by construction.

ii. <Code snippets>

```python
    # map every ophys frame onto its flash interval [start_i, start_{i+1})
    fi = np.searchsorted(f_start, ots, side='right') - 1
    fi_clipped = np.clip(fi, 0, f_start.size - 1)
    image_per_frame = carry[fi_clipped]
    change_per_frame = f_change[fi_clipped].astype(np.int8)
    # frames before the first flash of the behaviour block carry no stimulus
    change_per_frame[fi < 0] = 0
```

```python
        out[0] = image_per_frame[a:b]
```

iii. <Justification>

Step 5 decision 6 and the `processing_*.png` panel 5 review: "identity steps exactly at the red change
line". The two documented edge cases (Step 10 Check 5) are that a trial's first frame can fall ~21 ms
before the first flash onset (same image, asserted) and that at most one trial per session extends past
the last flash of the behaviour block, by ≤ 0.23 s, where the clip makes the last flash's label persist.

---

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. <Decisions>

From `stimulus_presentations.is_change` (together with `start_time`) of the change-detection block. The
SDK sets `is_change` only for real image changes; sham changes on catch trials are flagged separately as
`is_sham_change`, so catch trials are 0 throughout. (The human reference instead used `trials.change_time`
gated on `trials.go`; the two are equivalent.)

ii. <Code snippets>

```python
    f_change = flashes.is_change.values.astype(bool)
    ...
    change_per_frame = f_change[fi_clipped].astype(np.int8)
    change_per_frame[fi < 0] = 0
```

```python
        out[1] = change_per_frame[a:b]
```

iii. <Justification>

Step 5 variable-mapping table: "`stimulus_presentations.is_change` → `output[...][1]` image_change:
1 for every ophys frame inside the flash interval that starts at a change (= the 750 ms image
presentation interval); 0 on catch (sham-change) trials". Step 10 Check 2 re-derived it independently from
the HDF5 stimulus table: PASS.

---

## 4-b. What processing is involved in computing `output` *Image change*?

i. <Decisions>

None beyond the interval lookup: the per-frame value is the `is_change` flag of the flash interval the
frame falls in, i.e. a single contiguous pulse covering the 250 ms change image + the following 500 ms
grey ISI = 750 ms, starting at the change flash onset. No smoothing, no widening, no per-trial recentering.
Overall `image_change` fraction = 0.0776 of all converted timepoints.

ii. <Code snippets>

```python
    change_per_frame = f_change[fi_clipped].astype(np.int8)
```

```python
        # image_change must be 1 only in the 750 ms interval after a real change
        if go[k]:
            if chg.sum() == 0 or not np.all(t[chg == 1] >= change_time[k] - 1e-9):
                n_bad_change += 1
            dur = t[chg == 1].max() - t[chg == 1].min() if chg.sum() else 0
            if dur > 0.8:
                n_bad_change += 1
        else:
            if chg.sum() != 0:
                n_bad_change += 1
    assert n_bad_change == 0, f'[{eid}] image change mismatch in {n_bad_change} trials'
```

iii. <Justification>

Step 5 decision 6: "Changes and the flash before them are never omitted, so the change interval is always
a full 750 ms." Step 9 sanity check: "image_change fraction — analytic: n_go × 0.75 s / total time =
0.0770 vs converted **0.0776** ✓". Step 10 Check 5: "`image_change` is exactly one contiguous pulse per go
trial and identically 0 on catch trials; image identity switches exactly once per go trial, at the frame
where `image_change` turns on; the number of trials containing a change equals the number of go trials
(37,143)."

---

## 4-c. How is `output` *Image change* thresholded into categories?

i. <Decisions>

It is natively binary — no thresholding of a continuous quantity is needed. Values are `int8` 0/1 with
`output_values[1] = ['no_change', 'change']`.

ii. <Code snippets>

```python
        'output_values': [
            image_vocab,
            ['no_change', 'change'],
            [f'speed_pct_{i*20}_{(i+1)*20}' for i in range(N_BEHAVIOR_BINS)],
            [f'pupil_pct_{i*20}_{(i+1)*20}' for i in range(N_BEHAVIOR_BINS)],
            OUTCOME_NAMES,
        ],
```

iii. <Justification>

The task specifies "Image change, binary variable. Have value of 1 right after a change in image identity,
otherwise 0." The only design choice is the *duration* of the 1, and the AI set it to the paper's 750 ms
image-presentation interval (`metadata['image_change_definition']`: "1 during the 750 ms image
presentation interval that starts at a real image change, 0 otherwise (always 0 on catch/sham-change
trials)").

---

## 4-d. How is `output` *Image change* aligned with the neural data?

i. <Decisions>

Identically to image identity: one whole-session per-frame array built from the same
`fi = searchsorted(f_start, ots) - 1` flash-interval mapping, then sliced per trial with the same `[a:b]`
indices as the neural matrix.

ii. <Code snippets>

```python
    fi = np.searchsorted(f_start, ots, side='right') - 1
    fi_clipped = np.clip(fi, 0, f_start.size - 1)
    change_per_frame = f_change[fi_clipped].astype(np.int8)
    ...
        out[1] = change_per_frame[a:b]
```

iii. <Justification>

Same as 3-c. The alignment was checked visually in `processing_*.png` panel 5 ("the change pulse is
exactly one 750 ms flash interval") and against the trials table's `change_time` in `_check_session` for
all 42,470 trials.

---

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. <Decisions>

From `ds.running_speed`, using `.speed` (cm/s, already filtered by the SDK) and `.timestamps`. Non-finite
speed samples are dropped before interpolation.

ii. <Code snippets>

```python
    run = ds.running_speed
    run_t = run.timestamps.values.astype(np.float64)
    run_v = run.speed.values.astype(np.float64)
    good = np.isfinite(run_v)
    speed_per_frame = np.interp(ots, run_t[good], run_v[good])
```

iii. <Justification>

Step 5 variable-mapping table: "`running_speed.speed` / `.timestamps` → `output[...][2]` …
speed is already filtered by the SDK" (the SDK applies the low-pass/zero-phase filter to the encoder
signal before exposing `speed`).

---

## 5-b. What processing is involved in computing `output` *Running speed*?

i. <Decisions>

Linear resampling onto the ophys frame grid with `np.interp` (which clamps rather than extrapolating at
the edges, so no NaNs are produced), followed by percentile discretisation into 5 bins (see 5-c). The bin
labels are computed once per session over exactly the frames that end up inside retained trials, then
scattered back onto a session-length array for per-trial slicing.

ii. <Code snippets>

```python
    def percentile_bins(x):
        edges = np.percentile(x, np.linspace(0, 100, N_BEHAVIOR_BINS + 1)[1:-1])
        return np.digitize(x, edges).astype(np.int8), edges

    speed_bin_all, speed_edges = percentile_bins(speed_per_frame[frame_idx])
    ...
    speed_bin = np.zeros(ots.size, dtype=np.int8)
    speed_bin[frame_idx] = speed_bin_all
```

iii. <Justification>

Step 5 variable-mapping table: "`np.interp` onto ophys frame times, then `np.digitize` into 5 per-session
equal-percentile bins". Step 7 plot review: the resampled values "lie exactly on the native trace" when
overlaid on the raw 60 Hz signal. Step 10 Check 2: "output2/3 percentile edges == recomputed from HDF5
pupil area (+ blink mask) and the SDK running speed: PASS", and "all ~80–96k bin labels per session ==
independently digitised values: PASS".

---

## 5-c. How is `output` *Running speed* thresholded into categories?

i. <Decisions>

Five equal-percentile bins (quintiles) computed **separately for each session**, over the timepoints kept
in that session. The four interior edges (20/40/60/80th percentiles) are passed to `np.digitize`, giving
labels 0–4 with exactly 20% occupancy in every session (measured deviation ≤ 6e-5). The per-session edges
are stored in `metadata['session_info'][i]['running_speed_bin_edges']`. This differs from the human
reference, which pools all sessions and computes one global set of edges.

ii. <Code snippets>

```python
N_BEHAVIOR_BINS = 5                 # "five equal percentile bins"
...
    def percentile_bins(x):
        edges = np.percentile(x, np.linspace(0, 100, N_BEHAVIOR_BINS + 1)[1:-1])
        return np.digitize(x, edges).astype(np.int8), edges
```

```python
            'behavior_discretization':
                'running speed and pupil diameter are binned into five equal-percentile '
                'bins computed separately for each session over the timepoints retained '
                'in that session',
```

iii. <Justification>

Step 5 decision 9 and Step 12 Check 1. The AI ran a 20-session A/B with everything else held fixed:

| binning | speed bin | pupil bin |
|---|---|---|
| per session (used) | 0.276 | 0.259 |
| dataset-wide | **0.381** | **0.445** |

It deliberately kept the *lower* number, arguing the dataset-wide gain is an artefact: "running propensity
varies enormously between mice", pupil is in camera pixels with a rig dependence, whole sessions fall in a
single bin, "so the decoder scores by identifying the session — which it can always do, because each
session has its own projection matrix — rather than by reading out arousal". Per-session quintiles "make
the class label mean the same thing ('this animal's slowest/fastest fifth') everywhere". The AI flagged
this explicitly in its final report as one of "two decisions worth flagging".

---

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. <Decisions>

`np.interp` puts running speed on the ophys timestamps before any trial cutting, and the per-frame bin
labels are then sliced with the same `[a:b]` indices as the neural matrix.

ii. <Code snippets>

```python
    speed_per_frame = np.interp(ots, run_t[good], run_v[good])
    ...
    speed_bin = np.zeros(ots.size, dtype=np.int8)
    speed_bin[frame_idx] = speed_bin_all
    ...
        out[2] = speed_bin[a:b]
```

iii. <Justification>

Step 10 Check 3c: all streams already share the SDK's sync clock, so the timestamps are directly
comparable and linear interpolation is valid. Panel 3 of `processing_*.png` overlays the native ~60 Hz
running trace with the ophys-resampled trace as a visual alignment proof.

---

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. <Decisions>

From `ds.eye_tracking`, using `pupil_area` and `timestamps`. Blink frames — which the SDK has already set
to NaN wherever `likely_blink` is True — are dropped via a finite mask; the blink fraction is recorded per
session. Area is converted to an equivalent-circle diameter, `2√(area/π)`, in pixels. (The human reference
used the `pupil_width` column with an explicit `~likely_blink` mask.)

ii. <Code snippets>

```python
    eye_t = eye.timestamps.values.astype(np.float64)
    pupil_area = eye.pupil_area.values.astype(np.float64)
    good_eye = np.isfinite(pupil_area)
    res['blink_fraction'] = float(1.0 - good_eye.mean())
    if good_eye.sum() < 100:
        res['skip'] = 'pupil data all NaN'
        return res
    # pupil diameter of the fitted ellipse, in pixels (area -> diameter is a
    # monotone transform, so the percentile bins are unaffected by the choice)
    pupil_diam = 2.0 * np.sqrt(pupil_area / np.pi)
    pupil_per_frame = np.interp(ots, eye_t[good_eye], pupil_diam[good_eye])
```

iii. <Justification>

Step 5 decision 7: "Pupil diameter from `pupil_area` as `2√(area/π)`; blink frames (which the SDK has
already set to NaN via `likely_blink`) are dropped and bridged by linear interpolation. Percentile binning
is invariant to the monotone area→diameter transform, so this choice only affects the recorded bin edges,
not the labels." Step 10 Check 2 re-derived the edges from the raw HDF5 pupil area plus the blink mask:
PASS.

---

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. <Decisions>

Drop blink NaNs → convert area to diameter → `np.interp` onto the ophys frames (which bridges the blink
gaps and clamps at the recording edges) → per-session quintile discretisation over the retained frames →
scatter back into a session-length `int8` array for slicing.

ii. <Code snippets>

```python
    pupil_bin_all, pupil_edges = percentile_bins(pupil_per_frame[frame_idx])
    res['pupil_edges'] = pupil_edges
    res['pupil_range'] = (float(pupil_per_frame[frame_idx].min()),
                          float(pupil_per_frame[frame_idx].max()))
    ...
    pupil_bin = np.zeros(ots.size, dtype=np.int8)
    pupil_bin[frame_idx] = pupil_bin_all
```

iii. <Justification>

Same as 5-b/6-a. Panel 4 of `processing_*.png` overlays the native pupil trace (with NaN blink gaps)
against the blink-interpolated, ophys-resampled trace. The edge-case check confirms "no NaN or Inf
anywhere" in the converted arrays.

---

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. <Decisions>

Five equal-percentile bins computed **per session** over the retained timepoints, exactly as for running
speed, with the edges stored per session in `metadata['session_info'][i]['pupil_diameter_bin_edges']`.
Occupancy is exactly 20% per class in every session.

ii. <Code snippets>

```python
    speed_bin_all, speed_edges = percentile_bins(speed_per_frame[frame_idx])
    pupil_bin_all, pupil_edges = percentile_bins(pupil_per_frame[frame_idx])
```

iii. <Justification>

Step 12 Check 1: "pupil is measured in **camera pixels**, and the per-session median pupil diameter ranges
45–120 px with a clear rig dependence (CAM2P.3 mean 98.6 px vs CAM2P.4 83.3 px vs CAM2P.5 81.7 px) and a
between-mouse s.d. (16.3 px) twice the within-mouse s.d. (8.4 px). With dataset-wide edges whole sessions
fall in a single bin (e.g. 99% of one session in bin 0, 100% of another in bin 4), so the decoder scores
by identifying the session … rather than by reading out arousal." The AI accepted a validation accuracy of
0.259 instead of 0.445 for this reason.

---

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. <Decisions>

Same mechanism as running speed: interpolated onto `ophys_timestamps` before trial cutting, then sliced
with the same `[a:b]` indices as the neural matrix.

ii. <Code snippets>

```python
    pupil_per_frame = np.interp(ots, eye_t[good_eye], pupil_diam[good_eye])
    ...
        out[3] = pupil_bin[a:b]
```

iii. <Justification>

Step 10 Check 3c (common sync clock); panel 4 of the processing figure is the visual alignment check.

---

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. <Decisions>

From the four boolean columns of the SDK `trials` table — `hit`, `miss`, `false_alarm`, `correct_reject` —
in that fixed order, giving codes 0–3 with `output_values[4] = ['hit','miss','false_alarm','correct_reject']`.
The code asserts these are mutually exclusive and exhaustive on go|catch trials, and that hit/miss occur
only on go trials while FA/CR occur only on catch trials.

ii. <Code snippets>

```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
    oc = np.full(len(sel), -1, dtype=np.int8)
    for k, name in enumerate(OUTCOME_NAMES):
        oc[sel[name].values.astype(bool)] = k
    assert np.all(oc >= 0), 'go/catch trial without an outcome'
    assert (sel[OUTCOME_NAMES].values.astype(int).sum(axis=1) == 1).all(), \
        'trial with more than one outcome'
```

```python
    assert np.all((oc < 2) == sel.go.values.astype(bool)), \
        f'[{eid}] hit/miss must be go trials, FA/CR must be catch trials'
```

iii. <Justification>

These are the SDK's canonical outcome labels for the change-detection task. Step 9's consistency table
shows the converted totals 13,569 / 23,574 / 814 / 4,513 are "identical in all 165 sessions" to the
published `behavior_session_table.csv` values, and Step 10 Check 2 re-derived the labels from the HDF5
`intervals/trials` group.

---

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. <Decisions>

The single `int8` code per trial is broadcast across all timepoints of that trial, so `output` row 4 is a
constant time series rather than a scalar. Resulting distribution over timepoints: hit 0.316, miss 0.559,
false alarm 0.018, correct reject 0.107.

ii. <Code snippets>

```python
        out = np.empty((5, b - a), dtype=np.int8)
        ...
        out[4] = oc[k]
```

iii. <Justification>

The target-format spec says outputs should be time-varying "if at all possible", and all five output rows
must share one array shape, so the static label is repeated. The AI notes in Step 8/Step 12 Check 1 that
this label's "effective sample size is 42,470 trials rather than 11.2 M timepoints", explaining its lower
accuracy (0.295 vs 0.25 chance) and its larger train/validation gap (ratio 1.48), and verifies in Check 3
that `train_validate_decoder` splits by trial so there is no leakage.

---

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. <Decisions>

- **Missing eye tracking**: `ds.eye_tracking` absent/empty, or fewer than 100 finite pupil samples →
  whole session skipped with a printed reason (3 sessions: 795953296, 806456687, 833631914).
- **Blinks**: NaN pupil samples (the SDK's `likely_blink` frames) are dropped and bridged by linear
  interpolation rather than being labelled as a bin.
- **Non-finite running speed**: dropped before interpolation.
- **Out-of-range timestamps**: `np.interp` clamps to the first/last valid sample instead of producing
  NaNs, so no missing-value sentinel ever reaches the output.
- **Degenerate trials/sessions**: trials with < 2 ophys frames dropped; sessions with < 2 usable trials
  skipped.
- **Stimulus-timeline edges**: frames before the first flash get `change = 0`; frames past the last flash
  are clipped to the last flash interval (documented as ≤ 0.23 s in at most one trial per session).
- Everything else is handled by *assertions that fail loudly* rather than silent repair (trace length vs
  timestamps, monotone timestamps, valid ROIs, exactly one outcome per trial, no overlapping trials,
  image identity/change consistent with the trials table on every trial).

ii. <Code snippets>

```python
    try:
        eye = ds.eye_tracking
        if eye is None or len(eye) == 0:
            raise ValueError('empty')
    except Exception as exc:
        res['skip'] = f'no eye tracking ({exc})'
        return res
```

```python
    good = np.isfinite(run_v)
    speed_per_frame = np.interp(ots, run_t[good], run_v[good])
    ...
    good_eye = np.isfinite(pupil_area)
    res['blink_fraction'] = float(1.0 - good_eye.mean())
    if good_eye.sum() < 100:
        res['skip'] = 'pupil data all NaN'
        return res
```

```python
    assert np.all(np.diff(ots) > 0), 'ophys timestamps not increasing'
    assert neural_full.shape[1] == ots.size, 'trace length != ophys timestamps'
    assert bool(cst.valid_roi.all()), 'invalid ROIs present in released data'
    assert np.all(carry >= 0), 'image identity undefined at start of block'
    assert np.all(np.diff(frame_idx) > 0), 'trials overlap in time'
```

iii. <Justification>

Step 5 decision 8 ("Pupil diameter is a required output and cannot be imputed for a whole session") and
Step 10 Check 5, which enumerates the edge cases tested (`cache/edge_checks.py`, all PASS): every trial
≥ 2 timepoints, every session ≥ 2 trials, dtypes float32/float32/int8, "no NaN or Inf anywhere", every
output within `[0, len(output_values)-1]`, `n_trials == n_go + n_catch` for every session. The AI's
preference for assertions over silent fallbacks is visible throughout: `_check_session` re-validates
image identity, image change and outcome for *every* trial of *every* session at conversion time.

---

## 9-a. What are the most time-consuming steps of the code?

i. <Decisions>

The AI measured and published a per-step timing table. NWB loading dominates (2.4 s of the 3.2 s CPU per
session, ~75%); trace + behaviour processing + trial cutting is 0.8 s; pickle writing the 8.34 GB file is
13 s. It mitigated the dominant cost with a 24-process `multiprocessing.Pool`, so the full run is 56 s
wall (43 s of session processing + 13 s write) instead of ~9 min serial. Per-session timing and a running
ETA are printed.

ii. <Code snippets>

```python
    t_start = time.time()
    ds = BehaviorOphysExperiment.from_nwb_path(path)
    md = ds.metadata
    t_load = time.time() - t_start
    ...
    res['t_load'] = t_load
    res['t_total'] = time.time() - t_start
```

```python
def _report(i, n, r, t_proc):
    el = time.time() - t_proc
    msg = (f"  [{i+1}/{n}] {r['eid']} ... {r.get('t_total', float('nan')):.1f}s "
           f"(elapsed {el:.0f}s, eta {el/(i+1)*(n-i-1):.0f}s)")
```

iii. <Justification>

Step 7 "Run Time Estimates": "NWB load (`from_nwb_path`) 2.4 s | trace + behaviour processing + trial
cutting 0.8 s | total per session (CPU) 3.2 s → 9 min serial → **49 s with 24 workers** | assembly +
pickle write 13 s | measured full run 56 s." This is I/O + HDF5 deserialisation bound, which is why the
AI parallelised across files rather than optimising the numerics.

---

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. <Decisions>

The AI states that it deliberately vectorised all per-frame computation: "All per-frame quantities (image
identity, change, speed, pupil, percentile bins) are computed **once per session** as vectorised
whole-session arrays (`np.searchsorted` onto the flash grid, `np.interp`, `np.digitize`) and then only
sliced per trial; the only per-trial Python loop is the slicing itself."

Loops that remain in the delivered script and were *not* flagged by the AI:
- the omission forward-fill `for i in range(1, carry.size)` over every flash of the session (~4,800
  iterations/session) — vectorisable with `np.maximum.accumulate` over the indices of non-omitted flashes;
- `_check_session`'s per-trial validation loop, which re-derives identity/change for all 42,470 trials;
- `frame_idx = np.concatenate([np.arange(a, b) for a, b in zip(i0, i1)])` and the per-trial slicing loop;
- `remap = np.array([image_vocab.index(n) for n in r['images']])`, an O(n²) list lookup (trivial at n=8);
- the per-outcome assignment loop `for k, name in enumerate(OUTCOME_NAMES)` (4 iterations).

None of these matter: together they are a small fraction of the 0.8 s of non-I/O work per session.

ii. <Code snippets>

```python
    carry = local_idx.copy()
    for i in range(1, carry.size):          # forward-fill through omissions
        if f_omitted[i]:
            carry[i] = carry[i - 1]
```

```python
    fi = np.searchsorted(f_start, ots, side='right') - 1     # vectorised frame→flash map
    speed_per_frame = np.interp(ots, run_t[good], run_v[good])
    speed_bin_all, speed_edges = percentile_bins(speed_per_frame[frame_idx])
```

```python
    for k in range(n_trials):               # only remaining per-trial loop: slicing
        a, b = i0[k], i1[k]
        neural.append(np.ascontiguousarray(neural_full[:, a:b]))
```

iii. <Justification>

Step 6: the vectorisation and the worker pool are listed as the two "code inefficiencies identified /
speed-ups added", together with float32/int8 casting ("8.3 GB instead of ~17 GB"). Since loading is
I/O-bound and the whole conversion takes 56 s, the AI did not pursue the remaining Python loops.

---

## 9-c. What processing does the code repeat multiple times?

i. <Decisions>

Genuine repeats in the delivered script:
- **`_check_session`** re-derives, for every one of the 42,470 trials, the expected image identity
  (from `initial_image_name`/`change_image_name`) and the expected change window, and compares them with
  the already-computed outputs. This duplicates the output construction on purpose, as a self-check.
- **`frame_idx`** is rebuilt from `i0`/`i1` a second time inside `show_processing`, and the speed/pupil
  bin arrays are scattered back into session-length arrays a second time for plotting.
- `speed_per_frame` / `pupil_per_frame` / `image_per_frame` / `change_per_frame` are computed for **all**
  session frames, including the ~30–45% that fall outside retained trials and are then discarded.
- Each session's trial table is filtered once and its outcome columns are read twice (once for `oc`, once
  in the `sum(axis=1) == 1` assertion).

Outside the script, the *conversion itself* was re-run end-to-end three times (events → filtered_events →
dF/F) for the neural-signal A/B, and the decoder was retrained for the binning A/B.

ii. <Code snippets>

```python
    _check_session(res, sel, oc, i0, i1, ots, output, images)   # re-derives every trial's labels
```

```python
    fidx = np.concatenate([np.arange(a, b) for a, b in zip(r['i0'], r['i1'])])   # in show_processing
    sb = np.full(ots.size, np.nan); sb[r['frame_idx']] = r['speed_bin_all']
    pb = np.full(ots.size, np.nan); pb[r['frame_idx']] = r['pupil_bin_all']
```

iii. <Justification>

The duplicated label derivation is the point: Step 5's sanity-check list includes "Image identity must
equal `initial_image_name` before the change and `change_image_name` after it on every go trial, and be
constant on every catch trial (asserted for all 42,470 trials during conversion)". The AI treats the cost
(a fraction of 0.8 s/session) as worth paying for a hard guarantee on every trial. The duplicated work in
`show_processing` only runs under `--show-processing` on at most 2 sessions.

---

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. <Decisions>

- **`res['raw']`** — under `--show-processing` the worker returns the *entire* session's dF/F matrix plus
  every intermediate per-frame array through the multiprocessing pipe, purely to draw the figure; it is
  popped and discarded immediately afterwards.
- **Diagnostic statistics** computed for every session but never used by the decoder: `blink_fraction`,
  `speed_range`, `pupil_range`, `n_trials_all`, `n_aborted`, `n_auto_rewarded`, `n_go`, `n_catch`,
  `cell_specimen_ids`, `trial_start_times`, `trial_stop_times`, `change_times`, the per-session bin edges,
  rig name, cre line, imaging depth, container id — all serialised into `metadata['session_info']`.
- **`_check_session`** (see 9-c) contributes nothing to the output arrays.
- The whole-session per-frame arrays are computed over frames that never enter a trial.
- `brain_region_idx` is a constant array (every neuron is VISp, so `brain_regions` has length 1).

ii. <Code snippets>

```python
    if want_raw:   # extra material for the --show-processing plots
        res['raw'] = dict(
            ots=ots, neural_full=neural_full, f_start=f_start, f_omitted=f_omitted, ...)
    ...
    for r in results:
        r.pop('raw', None)
```

```python
        session_info.append({
            'ophys_experiment_id': r['eid'], ..., 'blink_fraction': r['blink_fraction'],
            'running_speed_bin_edges': r['speed_edges'].tolist(),
            'pupil_diameter_bin_edges': r['pupil_edges'].tolist(),
            'cell_specimen_ids': r['cell_specimen_ids'],
            'trial_start_times': r['trial_start'], 'trial_stop_times': r['trial_stop'],
            'change_times': r['change_time'],
        })
```

iii. <Justification>

The task's target format explicitly invites extra metadata ("Add other relevant fields, e.g.
`session_info`"), and the AI used exactly these fields for its Step 9/Step 10 consistency tables and for
the independent `h5py` sanity checks, so the "unnecessary" material is provenance rather than waste. It is
also tiny next to the 8.34 GB of neural data. The `raw` bundle is gated behind `--show-processing` and
explicitly freed (`r.pop('raw', None)`) before assembly so it never reaches the pickle.
