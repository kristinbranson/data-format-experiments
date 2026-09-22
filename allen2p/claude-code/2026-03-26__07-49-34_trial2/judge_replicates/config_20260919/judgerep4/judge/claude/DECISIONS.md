# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. <Decisions>

The AI did **not** use the AllenSDK loading API. It read the on-disk NWB (HDF5) files directly with `h5py`, and used the project metadata CSV `project_metadata/ophys_experiment_table.csv` to enumerate what is available. Discovery works as follows:

- Glob all `*.nwb` files in `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/` (284 files), parse the `ophys_experiment_id` out of each filename, and build an id → path map.
- Intersect that map with the rows of `ophys_experiment_table.csv`.
- Filter to **active behavior** experiments only (`passive == False`) → 202 experiments, 38 mice, brain regions VISp (185) and VISl (17).
- **No `project_code` filter is applied**, so both `VisualBehavior` (168 active experiments) and `VisualBehaviorMultiscope` (34 active experiments) are included.
- Each surviving experiment is then opened once per pass and every needed stream is read wholesale: dF/F traces + ophys timestamps, the cell specimen table, the natural-images stimulus presentations table, the trials table, running speed, and eye tracking.

The script runs **two full passes** over all 202 files: Pass 1 to accumulate running-speed and pupil values for the global percentile edges, Pass 2 to build the actual output.

ii. <Code snippets>

```python
NWB_DIR = 'data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/'
METADATA_DIR = 'data/visual-behavior-ophys-1.1.0/project_metadata/'

def get_experiment_metadata():
    """Load experiment metadata and identify available NWB files."""
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    nwb_map = {}
    for f in nwb_files:
        eid = int(os.path.basename(f).replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
        nwb_map[eid] = f

    our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
    # Only active behavior sessions
    active_exps = our_exps[our_exps['passive'] == False].copy()
    active_exps = active_exps.sort_values('ophys_experiment_id').reset_index(drop=True)
    return active_exps, nwb_map
```

```python
with h5py.File(nwb_path, 'r') as f:
    dff_data = f['processing']['ophys']['dff']['traces']['data'][:]      # (timepoints, neurons)
    ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
    cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
    valid_roi = cell_table['valid_roi'][:].astype(bool)
    ...
    stim  = f['intervals'][stim_key]
    trials = f['intervals']['trials']
    running_speed = f['processing']['running']['speed']['data'][:]
    running_ts    = f['processing']['running']['speed']['timestamps'][:]
    has_eye_tracking = 'EyeTracking' in f['acquisition']
```

```python
# Pass 1 (statistics only) and Pass 2 (full conversion) each iterate all experiments
for idx, (_, row) in enumerate(active_exps.iterrows()):
    stats = process_experiment(nwb_map[row['ophys_experiment_id']], row, collect_stats_only=True)
...
for idx, (_, row) in enumerate(active_exps.iterrows()):
    result = process_experiment(nwb_map[row['ophys_experiment_id']], row, collect_stats_only=False)
```

iii. <Justification>

From CONVERSION_NOTES.md Step 10 Check 3: "Data loading | h5py direct NWB read | `BehaviorOphysExperiment.from_nwb_path()` | Equivalent". The AI's Step 1 exploration mapped each SDK accessor to its NWB location (`DFFTraces.from_nwb()` → `processing['ophys']['dff']['traces']`, `RunningSpeed.from_nwb()`, `EyeTrackingTable.from_nwb()`, `Trials.from_nwb()`, `Presentations.from_nwb()`) and then replicated those reads directly, presumably to avoid SDK/S3 overhead when the files are already local. Step 5 decision 3: "Active sessions only: Passive sessions have no meaningful trial outcomes (no licking)." The trajectory (step 31) shows the AI first considered including passive sessions because the task says "Visual Behavior", then rejected that: "passive sessions lack the trial outcomes (Hit/Miss/FA/CR) that the task requires, so I should focus on active sessions only."

---

## 1-b. How are the data split into subjects?

i. <Decisions>

Subjects are the unique `mouse_id` values of the 202 active experiments, sorted as strings. `subject_idx` for each emitted session is the index of that experiment's `mouse_id` in this list. Result: 38 subjects.

ii. <Code snippets>

```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
subject_idx = subjects_list.index(result['mouse_id'])
all_subject_idx.append(subject_idx)
...
'subjects': subjects_list,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. <Justification>

`mouse_id` is the canonical animal identifier in the Allen experiment table. CONVERSION_NOTES.md Step 9 records "Subjects | 82 (full dataset) | 38 in our NWB subset | 38 | Yes (subset)", i.e. the AI cross-checked its 38 against the whitepaper's 82 mice for the full release and attributed the difference to the on-disk subset.

---

## 1-c. How are the data split into sessions?

i. <Decisions>

**Each NWB experiment file (i.e. each imaging plane) is treated as one "session"** in the output. There is no grouping by `ophys_session_id`. This gives 202 sessions from 202 active experiments.

For the 168 single-plane `VisualBehavior` experiments this is a 1:1 mapping (1 experiment = 1 session). For the 34 `VisualBehaviorMultiscope` experiments, however, the 34 planes belong to only **6 real ophys sessions** (3–7 planes each), so the same behavioural session is emitted 3–7 times as separate "sessions", each carrying that plane's neurons and an identical copy of the behaviour/trial data. This is visible in `verification_full_out.txt` as one mouse (457841) with 34 sessions, and as runs of identical trial counts (`... 209, 209, 209, 209, 209, 209, 209, ... 287 ×7, 309 ×7, 239 ×5, 196 ×5 ...`).

ii. <Code snippets>

```python
active_exps = our_exps[our_exps['passive'] == False].copy()
active_exps = active_exps.sort_values('ophys_experiment_id').reset_index(drop=True)
...
for idx, (_, row) in enumerate(active_exps.iterrows()):
    result = process_experiment(nwb_map[row['ophys_experiment_id']], row, collect_stats_only=False)
    ...
    all_sessions_neural.append(result['neural_trials'])    # one output session per experiment
    all_sessions_input.append(result['input_trials'])
    all_sessions_output.append(session_output)
    region_idx = np.full(result['n_neurons'],
                         brain_regions_list.index(result['brain_region']), dtype=np.int64)
    all_brain_region_idx.append(region_idx)
```

(`ophys_session_id` is present in the metadata table but is never read by the script.)

iii. <Justification>

Trajectory step 31: "Each NWB file is one 'experiment' (one imaging plane) — treat each as a session for the decoder. ... Since each NWB file contains its own set of neurons, I should treat each experiment as a separate session for the decoder rather than grouping by ophys_session." CONVERSION_NOTES.md Step 2 records "Each NWB file = one imaging plane from one session."

Notably, the AI later *noticed* the consequence but did not act on it. Trajectory step 79: "Interesting that one subject (457841) has 34 sessions. ... This might be because MESO sessions with multiple imaging planes get split into separate experiments — so 5 actual MESO sessions could generate 34 'sessions' in our output if each has 7 planes." No fix was made and no rationale for leaving it was written into CONVERSION_NOTES.md.

---

## 1-d. How are the data split into trials?

i. <Decisions>

Trials come from the NWB `intervals/trials` table. A trial is kept if `(go OR catch) AND NOT aborted AND NOT auto_rewarded`. The **temporal extent** of a trial is not taken from `start_time`/`stop_time`; instead it is the set of stimulus presentations whose `trials_id` equals that trial's `id`, and each such presentation becomes one 750 ms time bin. Trials therefore have variable length (10–17 bins, mean 11.6; verified `T: mean 11.61, median 11.67, min 10, max 17`). In the example session this window spans roughly −6.8 s to +3.75 s relative to the (sham) change, i.e. it covers the pre-change flashes and the post-change response window — essentially the same span as `start_time`→`stop_time`.

Total: 51,992 trials over 202 sessions (mean 257.4, min 39, max 409).

ii. <Code snippets>

```python
trial_go      = trials['go'][:].astype(bool)
trial_catch   = trials['catch'][:].astype(bool)
trial_aborted = trials['aborted'][:].astype(bool)
trial_auto    = trials['auto_rewarded'][:].astype(bool)

# Filter: Go + Catch, exclude Aborted and Auto-rewarded
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
valid_trial_ids = trial_ids[trial_mask]
```

```python
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]

    # Find stimulus presentations for this trial
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue

    n_bins = len(trial_stim_indices)
```

iii. <Justification>

CONVERSION_NOTES.md Step 3 "Trial curation rules": "Per task instructions: Include Go and Catch trials, exclude Aborted and Auto-rewarded"; Step 4: "Trial types | go/catch/aborted/auto_rewarded | Confirmed in NWB trials table | Same definitions | Include Go+Catch only per task instructions". Step 5 decision 1 explains the stimulus-presentation-indexed window: "Time bin = 750ms (1 per stimulus flash): Natural task unit, consistent across all equipment types (MESO/CAM2P), aligned to image presentations." Trajectory step 41: "For each trial, I'll bin the data into 750ms windows aligned with stimulus presentations."

---

## 1-e. How are trials filtered based on quality controls?

i. <Decisions>

Filtering is minimal and all of it is structural rather than quality-based:

1. Experiment level: `passive == False` (active behaviour only).
2. Experiment level: skipped if 0 valid ROIs, or if no natural-images stimulus table is found.
3. Trial level: `(go|catch) & ~aborted & ~auto_rewarded`.
4. Trial level: trials with zero associated stimulus presentations are skipped (`continue`).
5. Session level: sessions with `< 2` trials are dropped (to satisfy the format requirement of ≥2 trials/session).

No engagement/lick-bout filter, no d-prime session filter, no per-trial behavioural quality filter is applied. In the full run, 0 experiments and 0 trials were actually dropped by (2), (4), or (5) — minimum trials/session was 39. I independently confirmed that the AI's trial mask selects exactly 51,992 trials across the 202 active experiments, matching the output, and that all 51,992 have a non-NaN `change_time`.

ii. <Code snippets>

```python
if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None

stim_key = find_stim_key(f)
if stim_key is None:
    print(f"  WARNING: No stimulus presentations found for {experiment_id}, skipping")
    return None
```

```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
if len(trial_stim_indices) == 0:
    continue
```

```python
if result is None or result['n_trials'] < 2:
    skipped += 1
    reason = 'no result' if result is None else f'only {result["n_trials"]} trials'
    print(f"  Skipped experiment {eid}: {reason}")
    continue
```

iii. <Justification>

CONVERSION_NOTES.md Step 3 lists the whitepaper's session-level QC (z-drift, baseline drop, d-prime ≥ 1, etc.) but treats it as already applied upstream in the released dataset; Step 3 also notes "Paper excludes images where licking bout already ongoing" without adopting that rule. Step 10 Check 5: "Minimum trials per session: 39 (above the 2-trial minimum)". The ≥2-trial rule is directly from the target format spec ("There needs to be at least two trials within each session in order to evaluate the decoder performance").

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. <Decisions>

`neural` is derived from the pre-computed dF/F traces stored in the NWB at `processing/ophys/dff/traces/data`, with `processing/ophys/dff/traces/timestamps` (the ophys frame timestamps) as the time base. The array is stored (timepoints × ROIs) and columns are subset by the `valid_roi` column of `processing/ophys/image_segmentation/cell_specimen_table`. Calcium `events` / `filtered_events` (which the reference paper uses) were deliberately **not** used.

ii. <Code snippets>

```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]        # (timepoints, neurons)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]

cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi  = cell_table['valid_roi'][:].astype(bool)

dff_valid = dff_data[:, valid_roi]   # (timepoints, n_valid_neurons)
n_neurons = dff_valid.shape[1]
```

iii. <Justification>

CONVERSION_NOTES.md Step 1: "dF/F is pre-computed in NWB files — no need to compute from raw fluorescence"; Step 4: "Neural data type | dF/F and events both available | Both in NWB | Paper uses events for analysis | Use dF/F — standard for decoding, pre-computed". Step 5 decision 2: "Use dF/F (not events): dF/F is the standard neural signal for decoding; pre-computed in NWB; events are sparser and may not decode as well for time-varying outputs." Step 3 also documents the SDK's dF/F algorithm (noise estimation → 600 s median-filter baseline → (F−F0)/F0 → detrending), confirming the AI checked that no dF/F computation was needed.

---

## 2-b. How is the `neural` data processed?

i. <Decisions>

Three operations, in order:

1. **Neuron selection**: keep columns with `valid_roi == True` (matching the SDK's `exclude_invalid_rois=True` default). In these 284 NWB files this is a no-op — I verified all 29,444 ROIs are flagged valid — but it is the correct SDK-equivalent step.
2. **Temporal rebinning**: for each stimulus presentation, average dF/F over the ophys frames falling in `[stim_start, stim_start + 0.75 s)`. Implemented with a cumulative-sum trick plus `np.searchsorted` bin boundaries.
3. **Assembly**: each trial becomes a `(n_neurons, n_bins)` float32 matrix.

No z-scoring, baseline subtraction, smoothing, or normalisation is applied. Planes are **not** merged across simultaneously-recorded multiscope planes (see 1-c) — each plane's neurons form their own session, so `neural` per session is single-plane.

ii. <Code snippets>

```python
# Precompute cumulative sum for fast bin averaging of neural data
dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])

bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
all_stim_starts = stim_start
all_stim_ends   = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends   = np.searchsorted(ophys_ts, all_stim_ends,   side='left')
```

```python
# --- Neural data: vectorized bin averaging ---
neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. <Justification>

CONVERSION_NOTES.md Step 5 decision 4: "valid_roi filter: Use only cells marked valid_roi=True in NWB, matching SDK default behavior." Step 10 Check 3: "Neuron filtering | valid_roi boolean filter | exclude_invalid_rois=True in CellSpecimens | Same". Binning justification is in Step 5 decision 1 and is expanded in 2-e below. Step 6 lists the efficiency rationale: "Cumulative sum-based bin averaging (avoids per-bin boolean masking)".

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. <Decisions>

The only neuron-level quality control is the `valid_roi` flag from the NWB cell specimen table, which mirrors the AllenSDK default (`exclude_invalid_rois=True`). No SNR / event-rate / trace-variance threshold is added, and no minimum-neurons-per-session rule is imposed other than "≥ 1 valid neuron" (the smallest emitted session has 4 neurons, the largest 666, mean 145.8). All 29,444 ROIs in the 202 active experiments pass, so nothing is actually removed.

ii. <Code snippets>

```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
n_total_rois = dff_data.shape[1]

dff_valid = dff_data[:, valid_roi]
n_neurons = dff_valid.shape[1]

if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
```

iii. <Justification>

CONVERSION_NOTES.md Step 3 "Neuron curation rules (from SDK/whitepaper)": "`valid_roi` filter: excludes unions of cells, duplicates (>70% overlap), edge ROIs, apical dendrites, too small/narrow/dim, ghost cells (crosstalk), negative/zero traces. Classification via linear SVC on features: depth, shape, area, intensity, SNR, etc." The AI's position is that the released dataset's own QC (plus `valid_roi`) is the curation the reference pipeline applies, so no further filtering is warranted. Step 10 Check 5 additionally verified "No NaN/Inf in neural data".

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. <Decisions>

Alignment is to the **stimulus-flash grid**, anchored at the start of each trial. Every time bin `b` of a trial is the interval `[stim_start[b], stim_start[b] + 0.75 s)` where `stim_start` comes from the stimulus presentations table, and the bin's dF/F is the mean over the ophys frames in that interval (`np.searchsorted` on `ophys_timestamps`). The trial's bins are all presentations with that trial's `trials_id`, so bin 0 is the first flash of the trial (≈ trial `start_time`), not the change. Trials are therefore trial-onset-aligned with variable length, and the change falls at a variable bin index within the trial.

Metadata records `temporal_alignment_event = 'Stimulus presentation onset (each 750ms image flash)'` and `off_start = off_end = None` (since trial length is variable).

ii. <Code snippets>

```python
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends   = np.searchsorted(ophys_ts, all_stim_ends,   side='left')
...
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

```python
'temporal_alignment_event': 'Stimulus presentation onset (each 750ms image flash)',
'off_start': None,
'off_end': None,
```

iii. <Justification>

Trajectory step 41: "For a common time bin, I'll use the stimulus presentation interval approach (750ms bins). This is: 1. Consistent across all equipment types 2. Aligned to the natural task structure 3. Each time bin corresponds to one image presentation 4. The task variables (image identity, change) naturally align to this 5. Behavioral variables (running, pupil) can be averaged within each bin 6. Neural activity can be averaged within each bin." CONVERSION_NOTES.md Step 10 Check 3: "Temporal alignment | Average dF/F in 750ms stimulus bins | ophys_timestamps as temporal reference | Consistent."

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. <Decisions>

**Yes — aggressive rebinning is applied.** `TIME_BIN_MS = 750.0`: one bin per stimulus presentation (250 ms image + 500 ms grey), with all streams averaged inside the bin. The native ophys rates in this dataset are ~30.94 Hz (CAM2P, 168 experiments) and ~10.73 Hz (MESO.1, 34 experiments), i.e. native bins of ~32 ms and ~93 ms. Rebinning to 750 ms therefore discards ~23× (CAM2P) or ~8× (MESO) of the native temporal resolution. The resulting trials contain 10–17 bins (mean 11.6, ≈8.7 s).

The AI explicitly considered and rejected: using native timestamps per session (rejected: "the format requires uniform bin sizes across all trials and sessions"), resampling everything to 30 Hz like the paper (rejected: upsamples the mesoscope data), resampling everything to ~11 Hz / ~91 ms (rejected late in favour of 750 ms), and rounding to 100 ms.

ii. <Code snippets>

```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
...
bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
...
'time_bin_size': TIME_BIN_MS,
```

iii. <Justification>

Trajectory step 39/41 is a long deliberation: "CAM2P sessions: ~31 Hz; MESO.1 sessions: ~11 Hz per plane. If I resample to 31 Hz, mesoscope data gets upsampled (bad). If I resample to 11 Hz, single-plane data gets downsampled... The format says 'Time bins should be the same size for all trials and sessions'... The cleanest approach is to use 750ms bins aligned to stimulus presentations — one bin per flash. This works because the image identity stays constant within each bin, image changes happen at the flash level, and I can meaningfully average running speed and pupil data. It gives me 8–15 bins per trial, which is enough for the decoder to learn temporal patterns, and it's consistent across all sessions regardless of imaging equipment."

CONVERSION_NOTES.md Step 4: "Frame rate | 11 Hz (MESO) or 31 Hz (CAM2P) | Confirmed: MESO=10.73 Hz, CAM2P=30.94 Hz | Use 750ms bins (1 per stimulus flash) for consistency"; Step 5 decision 1: "Time bin = 750ms (1 per stimulus flash): Natural task unit, consistent across all equipment types (MESO/CAM2P), aligned to image presentations."

The AI also recorded the counter-argument it overrode (trajectory step 41): "750ms bins would align perfectly with stimulus presentations, but that's too coarse for capturing the temporal dynamics the decoder needs — onset transients, sustained responses, offset effects all get averaged together" — before reversing back to 750 ms.

---

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. <Decisions>

From the `image_name` column of the natural-images stimulus presentations table (`intervals/<Natural_Images_...>_presentations/image_name`), read per stimulus flash and restricted to the flashes belonging to each trial via `trials_id`. The stimulus table key is located by a heuristic scan that skips `trials`, `spontaneous*`, and `natural_movie*`. `is_change`/`omitted`/`trials_id` come from the same table. The trials table's `initial_image_name` / `change_image_name` columns are **not** used.

ii. <Code snippets>

```python
def find_stim_key(f):
    """Find the stimulus presentations key for change detection task."""
    for key in f['intervals']:
        if key == 'trials' or key.startswith('spontaneous') or key.startswith('natural_movie'):
            continue
        return key
    return None
```

```python
stim = f['intervals'][stim_key]
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in stim['image_name'][:]])
stim_trials_id  = stim['trials_id'][:]
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. <Justification>

CONVERSION_NOTES.md Step 5 variable mapping: "image_name | output[0] | Categorical encoding (8 images) | stimulus_presentations | Forward-fill for omitted". Step 4: "Image identity | 8 images per session | Confirmed: 8 unique image names + 'omitted' | 8 natural scene images | Forward-fill for omitted presentations." Using the presentations table is the direct record of what was on screen at each flash, which is why the AI preferred it over reconstructing from the trials table.

---

## 3-b. What processing is involved in computing `output` *Image identity*?

i. <Decisions>

1. **Omission handling**: flashes with `image_name == 'omitted'` (~5% of presentations, ~196 per session) are **forward-filled** from the previous flash; if the omission is the first flash of a trial, the next non-omitted name is back-filled instead.
2. **Global categorical encoding**: a single global image vocabulary is built and each name mapped to an integer. The vocabulary is collected by opening only the **first 10 experiments** (after sorting by `ophys_experiment_id`) and unioning their non-omitted image names; unknown names at encode time silently fall back to code 0.
3. Result: 16 unique images (8 from set A, 8 from set B), each ~5.8–6.7% of all bins. Sessions using set A never emit set-B codes and vice versa.

I verified that the 10 sampled experiments do in fact cover all 16 images present across all 202 active experiments, so the `.get(img, 0)` fallback never fired in this run.

ii. <Code snippets>

```python
def collect_all_image_names(active_exps, nwb_map):
    """Collect all unique image names across experiments (sample a few to be fast)."""
    all_images = set()
    # Sample up to 10 experiments to collect image names (they use the same 8 images)
    sample_exps = active_exps.head(min(10, len(active_exps)))
    for _, row in sample_exps.iterrows():
        ...
            for n in names:
                name = n.decode() if isinstance(n, bytes) else str(n)
                if name != 'omitted':
                    all_images.add(name)
    return sorted(all_images)
```

```python
images = stim_image_name[trial_stim_indices].copy()
# Forward-fill omitted presentations
for bi in range(len(images)):
    if images[bi] == 'omitted':
        if bi > 0:
            images[bi] = images[bi - 1]
        else:
            for bj in range(bi + 1, len(images)):
                if images[bj] != 'omitted':
                    images[bi] = images[bj]
                    break

image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. <Justification>

CONVERSION_NOTES.md Step 5 decision 5: "Omitted stimuli: Forward-fill image identity from previous presentation. Neural and behavioral data still recorded during omissions." Trajectory step 41 shows the AI first wanted to drop omitted flashes and then reversed: "that still violates the uniformity requirement because I'd have gaps. Let me settle on including all presentations and forward-filling the image identity for omissions, setting image change to 0."

For the global 16-code vocabulary, trajectory step 73: "16 unique images (not 8!) — this is because different sessions use different image sets (A and B)... The key question is whether to encode image identity per-session (0-7) or globally (0-15), and global encoding is correct since the decoder needs consistent output dimensions. ... Looking at the decoder implementation, it handles this through class fractions — sessions that never see certain images get zero weight for those classes." Step 6 lists "Sample 10 experiments for image name collection (not all 202)" as a deliberate speed optimisation.

---

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. <Decisions>

Perfectly co-indexed by construction: image identity is emitted once per stimulus presentation, and the neural bin for that same presentation is the dF/F averaged over `[stim_start, stim_start+0.75 s)`. Both use the identical `trial_stim_indices` ordering, so output column `b` and neural column `b` describe the same 750 ms flash cycle. The bin is labelled with the image shown during the 250 ms non-grey portion that opens the bin.

ii. <Code snippets>

```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
n_bins = len(trial_stim_indices)

neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    ...
images = stim_image_name[trial_stim_indices].copy()
...
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ...
], axis=0)  # (5, n_bins)
```

iii. <Justification>

Trajectory step 41: "For each trial, I'll bin the data into 750ms windows aligned with stimulus presentations. Within each window, I'll average the neural activity, record the image identity, and compute the mean running speed and pupil diameter." CONVERSION_NOTES.md Step 10 Check 2 reports spot-checks against raw NWB on sessions 0, 50, 150 with "Image ID Match: True" and `np.allclose(atol=1e-5)` passing.

---

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. <Decisions>

From the `is_change` column of the same stimulus presentations table, subset to the trial's flashes. This is the SDK-computed change flag (produced by `is_change_event()` in `stimulus_processing.py`), which is True only on the first flash of a genuinely new image — i.e. only on Go trials; catch ("sham change") trials have no `is_change` flash.

ii. <Code snippets>

```python
stim_is_change = stim['is_change'][:]
...
# --- Image change ---
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. <Justification>

CONVERSION_NOTES.md Step 1 lists "`is_change_event()` | stimulus_processing.py | PROCESSING | Identify change events in stimulus presentations" as a key reference function; Step 5 mapping: "is_change | output[1] | Binary (0/1) | stimulus_presentations | 1 at change flash only". Step 10 Check 4 validates the semantics against the task design: "Go trials all have change | Yes | 45,477 with change, 0 without | Exact" and "Catch trials no change | Yes | 0 with change, 6,515 without | Exact", which reproduces the 7/8–1/8 Go/Catch split of the transition matrix.

---

## 4-b. What processing is involved in computing `output` *Image change*?

i. <Decisions>

Essentially none: the per-flash boolean is taken as-is, with NaN (which occurs on omitted flashes) coerced to 0, and cast to int64. Exactly one bin per Go trial is 1; catch trials are all 0. Across the whole dataset 45,477 / 606,328 bins (7.5%) are 1, i.e. 87.5% of the 51,992 trials contain a change bin.

ii. <Code snippets>

```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. <Justification>

CONVERSION_NOTES.md Step 9: "Image change rate | ~7.5% of presentations | ~7.5% | 7.5% | Yes". Trajectory step 57: "Image change is ~7.6% — this is the fraction of stimulus presentations that are changes. Since Go trials make up ~87.5% of the data but each contains multiple non-change flashes before the actual change, the low overall change flash fraction makes sense." The NaN coercion follows the Step 5 decision to keep omitted flashes as bins with `image_change = 0`.

---

## 4-c. How is `output` *Image change* thresholded into categories?

i. <Decisions>

No thresholding is needed — the variable is natively binary. Categories are declared as `['no_change', 'change']` = `[0, 1]`. The temporal extent of the "change" label is exactly one 750 ms bin (the change flash plus its following grey period), which is the instruction's "right after a change in image identity".

ii. <Code snippets>

```python
change_value_names = ['no_change', 'change']
...
'output_values': [
    image_value_names,
    change_value_names,
    running_value_names,
    pupil_value_names,
    outcome_value_names,
],
```

iii. <Justification>

CONVERSION_NOTES.md Step 5 decision 1 ties the label width to the bin width: "Each bin = 250ms image + 500ms grey." Trajectory step 41: "The image change flag triggers on the first flash of a new image, which aligns perfectly with this binning scheme."

---

## 4-d. How is `output` *Image change* aligned with the neural data?

i. <Decisions>

Same mechanism as image identity: indexed by `trial_stim_indices`, so column `b` of the output corresponds to the neural bin averaged over `[stim_start[b], stim_start[b]+0.75 s)`. The change bin is the bin that opens at `change_time`.

ii. <Code snippets>

```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
...
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ot['image_change'].astype(np.int64),
    running_disc.astype(np.int64),
    pupil_disc.astype(np.int64),
    np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
], axis=0)  # (5, n_bins)
```

iii. <Justification>

Same as 3-c. CONVERSION_NOTES.md Step 10 Check 2 records "Change Match: True" for the three spot-checked sessions against raw NWB.

---

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. <Decisions>

From `processing/running/speed` in the NWB (`data` + `timestamps`). This is the **filtered** running speed (the NWB also stores `speed_unfiltered` and `dx`, neither of which is used); it corresponds to the SDK's `RunningSpeed.from_nwb()` / `dataset.running_speed`, i.e. the 10 Hz low-pass Butterworth-filtered trace with z ≥ 10 transients removed. Native rate ~60 Hz.

ii. <Code snippets>

```python
# --- Load running speed ---
running_speed = f['processing']['running']['speed']['data'][:]
running_ts    = f['processing']['running']['speed']['timestamps'][:]
```

iii. <Justification>

CONVERSION_NOTES.md Step 1: "Running speed has both raw and 10 Hz lowpass Butterworth filtered versions"; Step 3: "Running speed: 10 Hz lowpass Butterworth filter, z-score ≥ 10 transients set to NaN"; Step 10 Check 3: "Running speed | Filtered speed from NWB | RunningSpeed.from_nwb() | Same source". The AI chose the filtered version to match what the SDK exposes by default.

---

## 5-b. What processing is involved in computing `output` *Running speed*?

i. <Decisions>

1. Bin-average onto the 750 ms stimulus grid: for each flash, mean speed over the ~45 encoder samples in `[stim_start, stim_start+0.75 s)`, computed via a cumulative-sum + `searchsorted` scheme. Bins with no samples are left at 0.0.
2. Discretise into 5 bins using **global** percentile edges computed in Pass 1 over the bin-averaged speeds of **all** trials of **all** 202 experiments.

Resulting edges: `[-inf, 0.00408, 0.788, 15.82, 33.13, +inf]` cm/s.

ii. <Code snippets>

```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends   = np.searchsorted(running_ts, all_stim_ends,   side='left')
run_cumsum = np.cumsum(running_speed)
run_cumsum = np.concatenate([[0], run_cumsum])
...
# --- Running speed: vectorized bin averaging ---
running_binned = np.zeros(n_bins, dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
    n_pts = e - s
    if n_pts > 0:
        running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
running_values.extend(running_binned.tolist())
```

```python
# Pass 1 over every experiment, then:
all_running = np.array(all_running, dtype=np.float64)
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
```

iii. <Justification>

CONVERSION_NOTES.md Step 5 mapping: "running speed | output[2] | Average in 750ms bins, discretize to 5 percentile bins | running/speed in NWB | 10 Hz Butterworth filtered", and decision 7: "Discretization: Compute percentile bin edges across ALL valid time points in dataset (2-pass), then assign bins." Averaging within the bin follows directly from the 750 ms binning decision (trajectory step 41: "Behavioral variables (running, pupil) can be averaged within each bin").

---

## 5-c. How is `output` *Running speed* thresholded into categories?

i. <Decisions>

Five equal-count percentile bins (0/20/40/60/80/100th percentiles) computed once globally, applied with `np.digitize` against the interior edges, then clipped to `[0, 4]`. Outer edges are replaced with ±inf so that values outside the training range still land in bin 0 or 4. Labels are `speed_q1 … speed_q5`. Edges are stored in `metadata['running_speed_bin_edges']`.

ii. <Code snippets>

```python
def discretize_values(values, n_bins, bin_edges=None):
    """Discretize continuous values into equal percentile bins."""
    if bin_edges is None:
        valid = values[~np.isnan(values)]
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(valid, percentiles)
        # Ensure unique edges
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

    binned = np.digitize(values, bin_edges[1:-1]).astype(np.int64)  # 0 to n_bins-1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned, bin_edges
```

```python
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
...
running_value_names = [f'speed_q{i+1}' for i in range(N_PERCENTILE_BINS)]
'running_speed_bin_edges': running_bin_edges.tolist(),
```

iii. <Justification>

Directly from the task instruction "Running speed, discretized into five equal percentile bins." CONVERSION_NOTES.md Step 5 decision 7 explains the global (rather than per-session) edges: computing across the whole dataset keeps the category definitions comparable across sessions. Step 7 records the sanity check "Running speed bins | 20% each (uniform)" for the sample, and trajectory step 57 notes that per-session distributions are correctly non-uniform ("Running speed distribution is similarly session-dependent, which is normal").

---

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. <Decisions>

The running trace is averaged over exactly the same 750 ms windows used for the neural bins (`[stim_start, stim_start+0.75 s)`), indexed by the same `trial_stim_indices`, so output row 2 column `b` and neural column `b` cover the same wall-clock interval. Crucially the running bin boundaries are computed by `searchsorted` on the *running* timestamps rather than on ophys timestamps, so the two streams are aligned through absolute session time rather than by resampling one onto the other.

ii. <Code snippets>

```python
all_stim_starts = stim_start
all_stim_ends   = all_stim_starts + bin_duration

ophys_bin_starts = np.searchsorted(ophys_ts,    all_stim_starts, side='left')
ophys_bin_ends   = np.searchsorted(ophys_ts,    all_stim_ends,   side='left')
run_bin_starts   = np.searchsorted(running_ts,  all_stim_starts, side='left')
run_bin_ends     = np.searchsorted(running_ts,  all_stim_ends,   side='left')
```

iii. <Justification>

CONVERSION_NOTES.md Step 3: "Temporal alignment: All data streams synchronized via NI PCI-6612 at 100 kHz" — i.e. all timestamps are already on a common hardware clock, so binning each stream by absolute time is sufficient. Step 10 Check 2 reports "Running Match: True" for the spot-checked sessions.

---

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. <Decisions>

From `acquisition/EyeTracking/pupil_tracking/area` (+ its `timestamps`) together with `acquisition/EyeTracking/likely_blink/data`. The AI converts area to an **equivalent-circle diameter**, `d = 2·sqrt(area/π)`. The `pupil_tracking/width` and `height` fields (the fitted ellipse axes) are present but not used. Three of the 202 active experiments have no `EyeTracking` group at all; those are detected and handled downstream (see 8).

ii. <Code snippets>

```python
has_eye_tracking = 'EyeTracking' in f['acquisition']
if has_eye_tracking:
    pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
    pupil_ts       = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
    likely_blink   = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
    ...
    # Compute diameter from area: d = 2*sqrt(area/pi)
    pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
else:
    pupil_ts = None
    pupil_diameter = None
```

iii. <Justification>

CONVERSION_NOTES.md Step 4: "Pupil tracking | area, width, height available | pupil_area with NaN for blinks | DeepLabCut, blink detection z>3 | Use area → compute diameter, interpolate NaNs"; Step 5 mapping: "pupil area → diameter | 2*sqrt(area/pi), interpolate NaN, avg in bins, 5 percentile bins".

The trajectory (step 41) shows the AI explicitly weighed using `width` instead and chose area anyway: "the whitepaper specifies it as the major axis of the ellipse fit, so I should use the width field directly from the NWB data rather than computing it from area. ... For simplicity, I'll just compute diameter as 2*sqrt(area/pi) which gives the diameter of a circle with the same area. This is the most common definition and matches the whitepaper description."

---

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. <Decisions>

1. Blink rejection: frames flagged `likely_blink` are set to NaN; frames with `area <= 0` are also set to NaN.
2. Convert area → equivalent-circle diameter.
3. Session-level NaN gap filling: linear interpolation over the NaN runs, performed in **sample index space** (`np.interp` over `np.arange(len)`), with no maximum-gap limit; leading/trailing NaNs are effectively held constant by `np.interp`'s edge behaviour.
4. Bin-average onto the 750 ms flash grid using a NaN-aware cumsum (sum of valid values ÷ count of valid values); bins with no valid samples stay NaN.
5. Second, per-trial `interpolate_nans` pass over the binned values if any NaNs remain.
6. Discretise into 5 global percentile bins (edges from Pass 1 over all non-NaN binned values across all experiments): `[-inf, 73.88, 83.84, 92.87, 105.38, +inf]` px.

ii. <Code snippets>

```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
# Also NaN out negative or zero values
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
```

```python
def interpolate_nans(values):
    """Linearly interpolate NaN values in a 1D array."""
    valid = ~np.isnan(values)
    if valid.sum() == 0:
        return values
    if valid.sum() == len(values):
        return values
    result = values.copy()
    x = np.arange(len(values))
    result[~valid] = np.interp(x[~valid], x[valid], values[valid])
    return result
```

```python
pup_valid  = ~np.isnan(pupil_diameter)
pup_filled = np.where(pup_valid, pupil_diameter, 0.0)
pup_cumsum = np.concatenate([[0], np.cumsum(pup_filled)])
pup_count_cumsum = np.concatenate([[0], np.cumsum(pup_valid.astype(np.float64))])
...
for bi, si in enumerate(trial_stim_indices):
    s, e = pup_bin_starts[si], pup_bin_ends[si]
    n_valid = pup_count_cumsum[e] - pup_count_cumsum[s]
    if n_valid > 0:
        pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. <Justification>

CONVERSION_NOTES.md Step 3: "Pupil: DeepLabCut tracking, ellipse fit, blinks detected by z-score > 3, dilated 2 frames, set to NaN"; Step 1 lists `determine_likely_blinks()` as the relevant SDK function. Step 5 decision 6: "Pupil NaN handling: Linear interpolation for blink frames before computing diameter and averaging." Step 10 Check 3: "Pupil | Area → diameter, NaN interpolation | EyeTrackingTable with blink detection | Same source."

---

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. <Decisions>

Identical machinery to running speed: 5 equal-count percentile bins over the global pool of non-NaN bin-averaged diameters, ±inf outer edges, `np.digitize` + clip, labels `pupil_q1 … pupil_q5`, edges saved in `metadata['pupil_diameter_bin_edges']`.

ii. <Code snippets>

```python
# For pupil, exclude NaN
valid_pupil = all_pupil[~np.isnan(all_pupil)]
if len(valid_pupil) > 0:
    _, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
else:
    pupil_bin_edges = np.array([-np.inf, 0, 1, 2, 3, np.inf])
```

```python
pupil_raw = ot['pupil_diameter_raw']
nan_mask = np.isnan(pupil_raw)
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. <Justification>

From the task instruction "Pupil diameter, discretized into five equal percentile bins." CONVERSION_NOTES.md Step 5 decision 7 (global 2-pass percentile edges) and Step 7's sample check "Pupil diameter bins | 20% each (uniform global)". Trajectory step 57 confirms the AI expected and accepted per-session non-uniformity: "The pupil_diameter distribution for session 0 is very skewed (0.611 in q5, 0.005 in q1) — this is because we compute global percentile bins from the full sample, but session 0 might have different pupil characteristics. That's expected behavior."

---

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. <Decisions>

Same absolute-time binning as running speed: `searchsorted` on the eye-tracking timestamps with the same `[stim_start, stim_start+0.75 s)` windows and the same `trial_stim_indices` ordering, so pupil column `b` matches neural column `b`. Eye tracking is ~30 Hz, giving ~22 samples per bin.

ii. <Code snippets>

```python
if has_eye_tracking and pupil_diameter is not None:
    pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
    pup_bin_ends   = np.searchsorted(pupil_ts, all_stim_ends,   side='left')
...
pupil_binned = np.full(n_bins, np.nan, dtype=np.float32)
if has_eye_tracking and pupil_diameter is not None:
    for bi, si in enumerate(trial_stim_indices):
        s, e = pup_bin_starts[si], pup_bin_ends[si]
        n_valid = pup_count_cumsum[e] - pup_count_cumsum[s]
        if n_valid > 0:
            pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. <Justification>

Same as running speed — CONVERSION_NOTES.md Step 3 notes all streams share a 100 kHz hardware-synchronised clock, so binning by absolute time is the alignment. Step 2 records the eye-tracking rate ("Eye tracking: 135,981 timestamps at ~30 Hz") that the AI used to check the per-bin sample counts were sensible.

---

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. <Decisions>

From the four mutually exclusive boolean columns of the NWB trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that order. Any trial matching none of the four falls back to `miss` (code 1). I verified this fallback never fires: across all 202 active experiments, all 51,992 Go/Catch non-aborted non-auto-rewarded trials match exactly one of the four.

ii. <Code snippets>

```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
...
if trial_hit[trial_idx]:
    outcome = 0  # Hit
elif trial_miss[trial_idx]:
    outcome = 1  # Miss
elif trial_fa[trial_idx]:
    outcome = 2  # False Alarm
elif trial_cr[trial_idx]:
    outcome = 3  # Correct Rejection
else:
    outcome = 1  # Default to Miss
```

iii. <Justification>

CONVERSION_NOTES.md Step 5 mapping: "hit/miss/fa/cr | output[4] | Categorical (4 classes) | trials table | Static per trial"; Step 1 records these as canonical columns of `Trials.from_nwb()`. Step 10 Check 4: "Trial outcomes | Hit+Miss for Go, FA+CR for Catch | Confirmed | Correct" — the AI validated the outcome/trial-type consistency rather than trusting the columns blindly.

---

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. <Decisions>

The integer code is a static per-trial scalar, but it is **broadcast across all time bins** of the trial so that the output block stays a uniform `(5, n_bins)` array. Value names: `['hit', 'miss', 'false_alarm', 'correct_rejection']`. Full-dataset distribution: hit 30.7%, miss 56.8%, false_alarm 1.8%, correct_rejection 10.7%.

ii. <Code snippets>

```python
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ot['image_change'].astype(np.int64),
    running_disc.astype(np.int64),
    pupil_disc.astype(np.int64),
    np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
], axis=0)  # (5, n_bins)
```

```python
outcome_value_names = ['hit', 'miss', 'false_alarm', 'correct_rejection']
'output_names': ['image_identity', 'image_change', 'running_speed',
                 'pupil_diameter', 'trial_outcome'],
```

iii. <Justification>

The target format requires all rows of an `output` block to share `n_timepoints`; the instruction also says "If at all possible, make it time-varying", and replication is the standard way to express a static label on a time-varying grid. Trajectory step 41: "The trial outcome stays constant across the entire trial." CONVERSION_NOTES.md Step 12 Check 3 discusses the consequence: "Trial outcome is static per trial (repeated across time bins), so there are effectively fewer independent samples", which the AI cites as the reason for its 1.53× train/val gap.

---

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. <Decisions>

| Problem | Handling |
|---|---|
| Experiment with 0 valid ROIs | Return `None`, print warning, skip experiment |
| No natural-images stimulus table found | Return `None`, print warning, skip experiment |
| Trial with no associated stimulus presentations | `continue` (drop the trial) |
| Session with < 2 trials after filtering | Skip session |
| Stimulus bin containing no ophys / running samples | Neural and running bin left at **0.0** (not NaN, not skipped) |
| Blink frames / non-positive pupil area | Set to NaN, then linearly interpolated over |
| Whole experiment missing `EyeTracking` (3 of 202) | Pupil is all-NaN → **every bin assigned the middle bin (code 2)** |
| Pupil bins still NaN after binning | Per-trial linear interpolation |
| `is_change` NaN on omitted flashes | `np.nan_to_num(..., nan=0)` |
| `image_name == 'omitted'` | Forward-fill (back-fill at trial start) |
| Image name absent from the 10-experiment vocabulary | Silent fallback to code 0 |
| Trial matching none of hit/miss/FA/CR | Silent fallback to `miss` |

There is no `try/except` around per-experiment processing, so an unexpected read error would abort the whole run rather than skip one file.

ii. <Code snippets>

```python
if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
if stim_key is None:
    print(f"  WARNING: No stimulus presentations found for {experiment_id}, skipping")
    return None
```

```python
nan_mask = np.isnan(pupil_raw)
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

```python
neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:           # else: bin stays 0.0
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. <Justification>

CONVERSION_NOTES.md Step 5 decisions 5 and 6 cover omissions and pupil NaNs. Step 10 Check 5 "Check for edge cases" reports: "Minimum trials per session: 39 (above the 2-trial minimum); No NaN/Inf in neural data; All output values are valid integers; No non-integer outputs found." The instruction's requirement that every output be a valid integer class label is why the AI chose to impute rather than emit NaN/-1: a sentinel class would have to be declared in `output_values` and would become a spurious decodable category. The "middle bin" choice for eye-tracking-less sessions is not separately justified in CONVERSION_NOTES.md.

---

## 9-a. What are the most time-consuming steps of the code?

i. <Decisions>

Measured from `conversion_full_out.txt`, the full run took 510.4 s:

| Step | Time | Share |
|---|---|---|
| Pass 1 (statistics) | 261.6 s | 51% |
| Pass 2 (conversion) | 246.6 s | 48% |
| Image-name collection (10 files) | 1.0 s | <1% |
| Metadata CSV load | 0.0 s | ~0 |
| Pickle write (385.6 MB) | 1.1 s | <1% |

Both passes are dominated by the same thing: reading each NWB's full dF/F array (`(140k × N)` float64 per experiment) plus computing its cumulative sum, and then the ~600k iterations of the three per-bin Python loops. Because the two passes are almost identical work, the wall-clock cost is roughly 2× the necessary minimum. The script prints per-pass timing and a running ETA, which is how the AI tracked this.

ii. <Code snippets>

```python
print(f"  Pass 1 completed in {time.time()-t0:.1f}s")
...
elapsed = time.time() - t0
rate = (idx + 1) / elapsed
remaining = (len(active_exps) - idx - 1) / rate
print(f"  Pass 2: {idx+1}/{len(active_exps)} experiments "
      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
```

```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]   # full (T x N) read, both passes
...
dff_cumsum = np.cumsum(dff_valid, axis=0)                          # both passes
```

iii. <Justification>

CONVERSION_NOTES.md Step 7 "Run Time Estimates" projects "Pass 1 | 5.7s → ~575s; Pass 2 | 8.7s → ~878s; Total ~1453s (~24min)" from the 2-session sample. Trajectory step 61: "the sample with 2 experiments took about 14.6 seconds total, which projects to roughly 25 minutes for all 202 active experiments. Since that exceeds my 15-minute target, I need to optimize. The main bottleneck appears to be the per-bin loop that searches for ophys frames within each 750ms window." After the cumsum/searchsorted rework the real run came in at 8.5 min, under the 15-minute budget, so the AI stopped optimising there.

---

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. <Decisions>

The AI vectorised the *bin-boundary search* (one `np.searchsorted` call for the whole session) and replaced per-bin masking with cumulative sums, but left the bin-averaging itself as plain Python loops. Specifically, inside `process_experiment`'s per-trial loop there are **three separate per-bin Python loops** — neural, running, pupil — each executing ~606k times per pass (≈1.2M per pass across all three, ≈3.6M total over both passes). Each is a one-liner away from full vectorisation, since the cumsum arrays and the start/end index arrays already exist:

```python
s = ophys_bin_starts[trial_stim_indices]; e = ophys_bin_ends[trial_stim_indices]
n = np.maximum(e - s, 1)
neural_matrix = ((dff_cumsum[e] - dff_cumsum[s]) / n[:, None]).T
```

Other loops that could be vectorised or eliminated:
- The omitted-image forward-fill loop (`for bi in range(len(images))` plus an inner `for bj`) — a standard `np.maximum.accumulate` forward-fill.
- `image_indices = np.array([img_to_idx.get(img, 0) for img in images])` — a `np.searchsorted` into the sorted vocabulary.
- `np.where(stim_trials_id == tid)[0]` inside the trial loop — an O(n_stim) scan repeated once per trial (O(n_trials × n_stim) ≈ 365 × 4806 per experiment); a single `np.argsort`/`np.searchsorted` on `stim_trials_id` would give all trials' slices at once.
- `running_values.extend(...)` / `pupil_values.extend(...)` accumulate into Python lists of ~606k floats before being converted to arrays.

Note that the source comments label these loops "vectorized bin averaging", which they are not.

ii. <Code snippets>

```python
# --- Neural data: vectorized bin averaging ---
neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames

# --- Running speed: vectorized bin averaging ---
running_binned = np.zeros(n_bins, dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
    n_pts = e - s
    if n_pts > 0:
        running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts

# --- Pupil diameter: vectorized bin averaging ---
pupil_binned = np.full(n_bins, np.nan, dtype=np.float32)
if has_eye_tracking and pupil_diameter is not None:
    for bi, si in enumerate(trial_stim_indices):
        s, e = pup_bin_starts[si], pup_bin_ends[si]
        n_valid = pup_count_cumsum[e] - pup_count_cumsum[s]
        if n_valid > 0:
            pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

```python
for bi in range(len(images)):
    if images[bi] == 'omitted':
        if bi > 0:
            images[bi] = images[bi - 1]
        else:
            for bj in range(bi + 1, len(images)):
                if images[bj] != 'omitted':
                    images[bi] = images[bj]
                    break
```

iii. <Justification>

CONVERSION_NOTES.md Step 6 "Optimizations" claims: "Cumulative sum-based bin averaging (avoids per-bin boolean masking); np.searchsorted for efficient bin boundary finding; Sample 10 experiments for image name collection (not all 202)." Trajectory step 61 shows the intended plan went further than what was implemented: "The key optimization is to use np.digitize to assign each ophys frame to its corresponding stimulus bin, then leverage np.bincount or groupby operations to compute the averages in one pass instead of iterating through each presentation individually." That `bincount`/one-pass step was never written; the loops remained. The AI stopped once the measured runtime (8.5 min) was under the instructions' 15-minute threshold.

---

## 9-c. What processing does the code repeat multiple times?

i. <Decisions>

The dominant repetition is that **`process_experiment` is called twice on every experiment**, once with `collect_stats_only=True` and once with `False`, and the `collect_stats_only` flag is only consulted at the *very end* of the per-trial body (line 285). Everything before that point is executed identically in both passes:

- full read of `dff_data` and `ophys_ts`, plus the `valid_roi` subset;
- `np.cumsum(dff_valid, axis=0)` over the whole `(140k × N)` array;
- reading the stimulus table, the full trials table, running speed, eye tracking;
- pupil blink masking, area→diameter conversion, and whole-session NaN interpolation;
- per-trial `np.where(stim_trials_id == tid)`, the neural bin-averaging loop, the omitted forward-fill, `image_indices`, `change_flags`, and the trial-outcome branch.

Only the running and pupil binned values are actually consumed from Pass 1.

Smaller repetitions:
- `collect_all_image_names` re-opens 10 NWB files that Pass 1 and Pass 2 will open again, purely to read `image_name`.
- Pupil NaN interpolation runs twice: once on the full-resolution session trace, and again per trial on the already-binned values.
- `brain_regions_list.index(...)` and `subjects_list.index(...)` are linear list scans done per session instead of dict lookups (negligible).

ii. <Code snippets>

```python
# Pass 1
for idx, (_, row) in enumerate(active_exps.iterrows()):
    stats = process_experiment(nwb_path, row, collect_stats_only=True)
    ...
# Pass 2
for idx, (_, row) in enumerate(active_exps.iterrows()):
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

```python
def process_experiment(nwb_path, exp_info, collect_stats_only=False):
    ...
    dff_data = f['processing']['ophys']['dff']['traces']['data'][:]     # always
    dff_cumsum = np.cumsum(dff_valid, axis=0)                           # always
    ...
    for trial_idx in np.where(trial_mask)[0]:
        ...
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames   # always
        ...
        if collect_stats_only:
            continue                     # <-- flag checked only here, at the end
        neural_trials.append(neural_matrix)
```

iii. <Justification>

CONVERSION_NOTES.md Step 6 presents the two passes as the design: "convert_data.py with two-pass approach: 1. Pass 1: Collect running speed and pupil statistics for percentile bin computation; 2. Pass 2: Full conversion with discretization using global percentile bins." Two passes are genuinely necessary to compute global percentile edges before assigning bins, but nothing in the notes or trajectory addresses restricting Pass 1 to only the two streams it needs, or caching the binned values from Pass 1 for reuse in Pass 2.

---

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. <Decisions>

1. **The whole neural pipeline in Pass 1.** Because `if collect_stats_only: continue` sits after the neural bin-averaging, Pass 1 reads every dF/F array, cumsums it, and computes all ~606k neural bin averages — then returns only `running_values` and `pupil_values`. This is roughly half of the 510 s runtime spent producing arrays that are immediately garbage-collected. The same applies to `image_indices`, the omitted forward-fill, `change_flags`, and the trial-outcome branch in Pass 1.
2. **Unused loaded variables** in *both* passes: `stim_stop`, `stim_omitted`, `trial_start`, `trial_stop`, `trial_change_time`, `trial_catch` (only used inside the mask), `valid_trial_ids`, and `n_total_rois` are read and never used afterwards. `trial_start`/`trial_stop`/`trial_change_time` in particular are loaded on every experiment and never referenced, a leftover from the earlier start_time→stop_time windowing design.
3. **Empty input arrays**: `np.zeros((0, n_bins), dtype=np.float32)` is allocated per trial (51,992 allocations) for an input specification that is empty, and `input_names` is `[]`. The format only needs a placeholder.
4. **Raw running/pupil values are carried per trial** in `output_trials` dicts (`running_speed_raw`, `pupil_diameter_raw`) for the whole session and then discarded after discretisation.
5. **`collect_all_image_names` does an extra I/O pass** over 10 NWB files for information that Pass 1 already touches.
6. **Double pupil interpolation** (full-rate trace, then binned per trial).
7. `plot_processing` builds a 6×2 figure for the first 2 sessions under `--show-processing`; harmless but pure debug output.
8. `neural_matrix` is allocated with `np.zeros` and then fully overwritten column-by-column.

ii. <Code snippets>

```python
        # --- Trial outcome ---
        if trial_hit[trial_idx]:
            outcome = 0  # Hit
        ...
        if collect_stats_only:
            continue          # everything above was computed and is now thrown away

        neural_trials.append(neural_matrix)
```

```python
        trial_start = trials['start_time'][:]        # never used
        trial_stop = trials['stop_time'][:]          # never used
        trial_change_time = trials['change_time'][:] # never used
        stim_stop = stim['stop_time'][:]             # never used
        stim_omitted = stim['omitted'][:]            # never used
        n_total_rois = dff_data.shape[1]             # never used
```

```python
input_trials.append(np.zeros((0, n_bins), dtype=np.float32))
...
'input_names': [],
```

iii. <Justification>

Nothing in CONVERSION_NOTES.md or the trajectory acknowledges the Pass 1 waste; the notes' only efficiency claims are the cumsum binning, `searchsorted`, and the 10-file image-name sampling. The empty `input` arrays are required by the format ("input: ... No inputs for this task"), so item 3 is a format obligation rather than a mistake, just allocated more heavily than needed. The unused trials-table columns appear to be residue from the AI's earlier plan (trajectory step 35: "The trial window spans from start_time to stop_time") that was superseded by the stimulus-presentation windowing in step 41 without removing the loads.
