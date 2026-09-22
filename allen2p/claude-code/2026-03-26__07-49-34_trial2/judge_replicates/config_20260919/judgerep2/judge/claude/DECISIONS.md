# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK object API. It reads the released NWB files directly with `h5py`, and uses the project metadata CSV (`project_metadata/ophys_experiment_table.csv`) only as an experiment-level index. Discovery works as follows:

- Glob every `*.nwb` in `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/` and parse the `ophys_experiment_id` out of each filename (284 files present).
- Inner-join that id set against `ophys_experiment_table.csv`.
- Keep only **active** behavior experiments (`passive == False`) → 202 experiments, 38 mice.
- **No `project_code` filter is applied**, so the 34 active `VisualBehaviorMultiscope` (mesoscope) experiments are included alongside the 168 active `VisualBehavior` (single-plane) experiments.
- Experiments are sorted by `ophys_experiment_id` and each one is opened twice (Pass 1 for discretization statistics, Pass 2 for the real conversion).

Inside each NWB file the following groups are read directly:
`processing/ophys/dff/traces/{data,timestamps}`, `processing/ophys/image_segmentation/cell_specimen_table/valid_roi`, `intervals/<images>_presentations`, `intervals/trials`, `processing/running/speed/{data,timestamps}`, `acquisition/EyeTracking/{pupil_tracking/area, likely_blink}`.

ii.
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
    dff_data = f['processing']['ophys']['dff']['traces']['data'][:]   # (timepoints, neurons)
    ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
    ...
    stim = f['intervals'][stim_key]
    trials = f['intervals']['trials']
    running_speed = f['processing']['running']['speed']['data'][:]
    ...
```

iii. From CONVERSION_NOTES.md Step 1/Step 4: "**dF/F is pre-computed** in NWB files — no need to compute from raw fluorescence", and the Step 10 Check-3 table states the direct h5py read is "Equivalent" to `BehaviorOphysExperiment.from_nwb_path()`. The exclusion of passive experiments is justified in Step 5, Key Decision 3: "**Active sessions only**: Passive sessions have no meaningful trial outcomes (no licking)." The trajectory (step 39) shows the same reasoning: "passive sessions wouldn't have meaningful trial outcomes like hits or misses". No justification is given anywhere for including the Multiscope project alongside the VisualBehavior project — the agent noticed the distinction (trajectory step 31: "For multiscope sessions, one session can have multiple experiments (planes)") but never filtered on `project_code`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the unique `mouse_id` values of the 202 selected active experiments, cast to `str` and sorted. `subject_idx` is built per session by a `list.index()` lookup of the experiment's `mouse_id`. Result: 38 subjects.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
subject_idx = subjects_list.index(result['mouse_id'])
all_subject_idx.append(subject_idx)
...
'subjects': subjects_list,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. Not separately argued; `mouse_id` is taken as the canonical animal identifier from the experiment table (CONVERSION_NOTES Step 2 lists "Unique subjects in metadata | 107" and Step 9 reports "Subjects | 38 in our NWB subset").

## 1-c. How are the data split into sessions?

i. **Each NWB file (i.e. each `ophys_experiment_id`, one imaging plane) is treated as one "session".** The `ophys_session_id` column is never used, so simultaneously-recorded mesoscope planes are emitted as separate sessions. This yields 202 sessions where there are only 174 distinct `ophys_session_id`s (6 mesoscope sessions contribute 3–7 planes each). The consequence is visible in the output: identical trial counts repeat in blocks (`..., 209, 209, 209, 209, 209, 209, 209, ...`, `287 ×7`, `309 ×7`, `239 ×5`, `196 ×5`), and mouse 457841 is reported as having **34 sessions**. The behavioural output streams (image identity, image change, running, pupil, outcome) for those trials are therefore duplicated up to 7 times in the dataset with different neuron sets.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
    ...
    all_sessions_neural.append(result['neural_trials'])
    all_sessions_input.append(result['input_trials'])
    all_sessions_output.append(session_output)
    ...
    region_idx = np.full(result['n_neurons'],
                        brain_regions_list.index(result['brain_region']),
                        dtype=np.int64)
```
(there is no grouping key anywhere; one loop iteration = one output session)

iii. Trajectory step 31: "Each experiment = one imaging plane … For multiscope sessions, one session can have multiple experiments (planes) … Each NWB file is one 'experiment' (one imaging plane) - treat each as a session for the decoder … Since each NWB file contains its own set of neurons, I should treat each experiment as a separate session for the decoder rather than grouping [them]." The CONVERSION_NOTES do not revisit this; Step 9 simply reports "Sessions | 202" and Step 11 notes without comment that "one subject (457841) has 34 sessions".

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept if `(go | catch) & ~aborted & ~auto_rewarded`. The **temporal extent** of a trial is not taken from `start_time`/`stop_time`; instead it is the set of stimulus presentations in the images-presentation table whose `trials_id` equals the trial id. Each such flash becomes one 750 ms time bin, so trials are variable length (10–17 bins, mean 11.66, i.e. ~7.5–12.8 s). A trial with zero matching stimulus presentations is skipped.

I verified on experiment 792815735 that `(go|catch) & ~aborted & ~auto` selects exactly the same 188 trials as the reference's `~aborted & ~auto_rewarded & change_time.notna()` rule, and that the flash set for a trial spans the trial window (first flash ≈ 21 ms after `start_time`, last flash ≈ `stop_time`).

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
valid_trial_ids = trial_ids[trial_mask]
...
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]

    # Find stimulus presentations for this trial
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue

    n_bins = len(trial_stim_indices)
```

iii. CONVERSION_NOTES Step 3 ("Trial curation rules"): "Per task instructions: Include Go and Catch trials, exclude Aborted and Auto-rewarded". Step 4 discrepancy table: "Trial types | go/catch/aborted/auto_rewarded | Confirmed in NWB trials table | Same definitions | Include Go+Catch only per task instructions". Using the stimulus-presentation list as the trial's time base follows from Key Decision 1 (750 ms bins, "Each time bin corresponds to one image presentation").

## 1-e. How are trials filtered based on quality controls?

i. Filters applied, in order:
- **Session level**: `passive == True` experiments dropped (82 of 284); experiments with 0 `valid_roi` neurons dropped; experiments yielding `< 2` trials dropped (0 actually dropped — the minimum was 39).
- **Trial level**: aborted and auto-rewarded trials dropped; trials with no stimulus presentation carrying that `trials_id` dropped.
- No explicit `change_time` non-NaN check (implied by `go | catch`), no engagement/d-prime filter, no clipping needed because bins are defined by flash onsets rather than by the recording end.

Result: 202 sessions, 51,992 trials, 29,444 neurons.

ii.
```python
if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
...
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
if len(trial_stim_indices) == 0:
    continue
...
if result is None or result['n_trials'] < 2:
    skipped += 1
    reason = 'no result' if result is None else f'only {result["n_trials"]} trials'
    print(f"  Skipped experiment {eid}: {reason}")
    continue
```

iii. CONVERSION_NOTES Step 5 Key Decision 3 ("Active sessions only: Passive sessions have no meaningful trial outcomes (no licking)") and Step 3 trial-curation rules. Step 10 Check 5 reports "Minimum trials per session: 39 (above the 2-trial minimum)". The paper's engagement filter (72.2 % engaged) was noted in Step 3 but deliberately not applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. dF/F calcium traces, read directly from `processing/ophys/dff/traces/data` (stored as (timepoints, ROIs)), with the corresponding frame times from `processing/ophys/dff/traces/timestamps`. Detected calcium `events`/`filtered_events` were considered and rejected.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
```

iii. CONVERSION_NOTES Step 4: "Neural data type | dF/F and events both available | Both in NWB | Paper uses events for analysis | **Use dF/F - standard for decoding, pre-computed**". Step 5 Key Decision 2: "**Use dF/F (not events)**: dF/F is the standard neural signal for decoding; pre-computed in NWB; events are sparser and may not decode as well for time-varying outputs."

## 2-b. How is the `neural` data processed?

i. Two operations, nothing else (no z-scoring, no smoothing, no baseline subtraction, no cross-plane merging):
1. Column selection by `valid_roi` (see 2-c).
2. **Temporal rebinning**: for every stimulus flash `i` of the trial, the mean dF/F over the ophys frames falling in `[stim_start[i], stim_start[i] + 0.75 s)` is computed for every neuron, giving a `(n_neurons, n_bins)` `float32` matrix. The average is computed with a prefix-sum (`cumsum`) over the time axis so each bin is an O(1) difference, inside a Python loop over bins.

If a bin happens to contain no ophys frame the entry is left at 0.

ii.
```python
dff_valid = dff_data[:, valid_roi]  # (timepoints, n_valid_neurons)
n_neurons = dff_valid.shape[1]
...
bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends   = np.searchsorted(ophys_ts, all_stim_ends,   side='left')

dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])
...
neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. CONVERSION_NOTES Step 5 Key Decision 1: "**Time bin = 750ms (1 per stimulus flash)**: Natural task unit, consistent across all equipment types (MESO/CAM2P), aligned to image presentations. Each bin = 250ms image + 500ms grey." Step 6 lists the cumsum trick as the main speed optimisation. dF/F itself is treated as already fully processed by the Allen pipeline (Step 3: "dF/F: Pre-computed in NWB. Algorithm: noise estimation → 600s median filter baseline → (F-F0)/F0 → detrending").

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the SDK's ROI validity flag: columns of the dF/F matrix are kept where `cell_specimen_table/valid_roi == True`. Experiments left with 0 neurons are skipped. No SNR, event-rate, or activity-based filtering; no session-level z-drift / d-prime QC (those were noted from the whitepaper but not implemented, since the released files are already QC-passed).

Empirically this filter is a **no-op on this release**: I checked the first 40 NWB files (730 ROIs) and every ROI has `valid_roi == True`.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
n_total_rois = dff_data.shape[1]

dff_valid = dff_data[:, valid_roi]  # (timepoints, n_valid_neurons)
n_neurons = dff_valid.shape[1]

if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
```

iii. CONVERSION_NOTES Step 5 Key Decision 4: "**valid_roi filter**: Use only cells marked valid_roi=True in NWB, matching SDK default behavior." Step 1 identifies the corresponding SDK behaviour: "`CellSpecimens.__init__` with `exclude_invalid_rois=True` | cell_specimens.py | CURATION | Filter to valid ROIs only", and Step 3 lists what that flag excludes ("unions of cells, duplicates (>70% overlap), edge ROIs, apical dendrites, too small/narrow/dim, ghost cells (crosstalk), negative/zero traces").

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is **flash-locked**: bin `k` of a trial starts exactly at the onset time of the `k`-th stimulus presentation belonging to that trial, and ophys frames are assigned to it by `np.searchsorted` on the dF/F timestamps. Because the first flash of a trial begins ~20 ms after the trials-table `start_time`, the trial is effectively trial-start-aligned, with every subsequent bin re-locked to the true stimulus onset (so there is no accumulating drift from the nominal 750 ms period). The change flash therefore always falls on a bin boundary. `metadata['temporal_alignment_event']` is recorded as `'Stimulus presentation onset (each 750ms image flash)'`, and `off_start`/`off_end` are `None` because trial length is variable.

ii.
```python
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends   = np.searchsorted(ophys_ts, all_stim_starts + bin_duration, side='left')
...
s, e = ophys_bin_starts[si], ophys_bin_ends[si]
neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / (e - s)
```
```python
'temporal_alignment_event': 'Stimulus presentation onset (each 750ms image flash)',
'off_start': None,
'off_end': None,
```

iii. CONVERSION_NOTES Step 10 Check 3: "Temporal alignment | Average dF/F in 750ms stimulus bins | ophys_timestamps as temporal reference | Consistent". Step 5 Key Decision 1 argues the flash is the natural task unit so that "The task variables (image identity, change) naturally align to this". Step 3 records that all data streams are hardware-synchronised ("Temporal alignment: All data streams synchronized via NI PCI-6612 at 100 kHz"), which is what licenses putting all four streams on the same stimulus-defined bin grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **750 ms per bin** (`metadata['time_bin_size'] = 750.0`), i.e. one bin per stimulus presentation (250 ms image + 500 ms grey). This is a heavy rebinning of the native data: single-plane rigs run at ~30.94 Hz (32.3 ms/frame) and the mesoscope at ~10.73 Hz (93.2 ms/frame), so each bin averages ~23 or ~8 frames respectively. Trials end up 10–17 bins long (mean 11.66). Running speed (~60 Hz) and pupil (~30 Hz) are averaged into the same bins.

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
...
'time_bin_size': TIME_BIN_MS,
```

iii. CONVERSION_NOTES Step 4: "Frame rate | 11 Hz (MESO) or 31 Hz (CAM2P) | Confirmed: MESO=10.73 Hz, CAM2P=30.94 Hz | … | **Use 750ms bins (1 per stimulus flash) for consistency**". Trajectory step 35 frames the problem as needing "a SINGLE time bin size for the whole dataset", and step 41 lists the accepted rationale: "For a common time bin, I'll use the stimulus presentation interval approach (750ms bins). This is: 1. Consistent across all equipment types 2. Aligned to the natural task structure 3. Each time bin corresponds to one image presentation 4. The task variables (image identity, change) naturally align to this 5. Behavioral variables (running, pupil) can be averaged within each bin". Notably, the same trajectory step had *previously* argued against it ("Using 750ms bins would align perfectly with stimulus presentations, but that's too coarse for capturing the temporal dynamics the decoder needs — onset transients, sustained responses, offset effects all get averaged together. So I'm going with the 11 Hz resampling approach instead"); that objection was reversed without being answered.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `intervals/<Natural_Images_…>_presentations/image_name`, selected per trial via the `trials_id` column of the same table. The trials-table fields `initial_image_name` / `change_image_name` are not used.

ii.
```python
stim = f['intervals'][stim_key]
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in stim['image_name'][:]])
stim_trials_id = stim['trials_id'][:]
...
trial_stim_indices = np.where(stim_trials_id == tid)[0]
images = stim_image_name[trial_stim_indices].copy()
```

iii. CONVERSION_NOTES Step 5 variable-mapping table: "image_name | output[0] | Categorical encoding (8 images) | stimulus_presentations | Forward-fill for omitted". Step 4: "Image identity | 8 images per session | Confirmed: 8 unique image names + 'omitted' | 8 natural scene images | Forward-fill for omitted presentations."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Two steps:
1. **Omission forward-fill** — flashes whose `image_name` is the literal `'omitted'` (5 % of non-change flashes) inherit the previous flash's image; if the *first* flash of a trial is omitted, the first non-omitted flash later in the trial is used instead (backward-fill).
2. **Global integer coding** — a single name→code dict is built from a sorted list of all image names, giving 16 codes (image sets A and B pooled). Unknown names fall back to code 0.

The global name list is collected by opening only the **first 10 experiments** (sorted by id), not all 202. I confirmed by scanning all 202 active NWB files that the complete set really is those 16 names and that the first 10 experiments already cover all 16, so the shortcut happened to be safe for this run — but the `.get(img, 0)` fallback would silently mislabel an unseen image (and would do so in `--sample` mode, where fewer experiments are scanned).

ii.
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
img_to_idx = {name: i for i, name in enumerate(IMAGE_NAMES_GLOBAL)}
...
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

iii. CONVERSION_NOTES Step 5 Key Decision 5: "**Omitted stimuli**: Forward-fill image identity from previous presentation. Neural and behavioral data still recorded during omissions." Step 6 lists the sampling shortcut as an optimisation: "Sample 10 experiments for image name collection (not all 202)". Step 10 Check 4 verifies "8 images per session | Yes | 8 per session (16 total across sets A/B) | Correct".

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Exactly one image code per time bin, and the bins are the same `trial_stim_indices` used to build the neural matrix — so image identity and neural activity are on the same grid by construction. The code changes at the change flash, i.e. at a bin boundary, with no interpolation or searchsorted needed.

ii.
```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
n_bins = len(trial_stim_indices)
# neural
for bi, si in enumerate(trial_stim_indices): ...
# image identity, same indices, same order
images = stim_image_name[trial_stim_indices].copy()
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
...
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ...
], axis=0)  # (5, n_bins)
```

iii. Implicit in Key Decision 1 — the entire point of 750 ms bins is that "Each time bin corresponds to one image presentation", so identity is exact per bin rather than interpolated. Step 10 Check 2 reports the per-bin image identity matched a raw-NWB recomputation for sessions 0, 50 and 150 ("Image ID Match: True").

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` column of the stimulus-presentations table, subset to the trial's flashes. `change_time` and the `go`/`catch` trials-table columns are not used for this output.

ii.
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. CONVERSION_NOTES Step 1 identifies `is_change_event()` in `stimulus_processing.py` as the SDK function that "Identify[s] change events in stimulus presentations". Step 5 mapping table: "is_change | output[1] | Binary (0/1) | stimulus_presentations | **1 at change flash only**".

## 4-b. What processing is involved in computing `output` *Image change*?

i. Essentially none: the stored `is_change` is a float 0/1 column, so NaNs are coerced to 0 and the result is cast to int64. Exactly one bin per go trial is set to 1 (the change flash); catch (sham-change) trials get all zeros because the SDK's `is_change` is False for a sham change. Because a bin is 750 ms, the "1" covers the change image flash plus its following grey period.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. CONVERSION_NOTES Step 10 Check 4 verifies the semantics end-to-end: "Go trials all have change | Yes | 45,477 with change, 0 without | Exact" and "Catch trials no change | Yes | 0 with change, 6,515 without | Exact". The overall rate matches expectation: "Image change rate | ~7.5% of presentations | 7.5% | Match".

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required — the variable is natively binary. `output_values[1] = ['no_change', 'change']`, and the realised distribution is 92.5 % / 7.5 %.

ii.
```python
change_value_names = ['no_change', 'change']
...
'output_values': [
    image_value_names,
    change_value_names,
    ...
]
```

iii. The instructions specify image change as a binary variable; CONVERSION_NOTES Step 5 records it as "Binary (0/1)".

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity: indexed by `trial_stim_indices`, one value per 750 ms bin, sharing the bin grid with the neural matrix. The change bin starts exactly at the change-image onset, so the "1" bin is the bin containing the evoked response to the change.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
...
output_arr = np.stack([..., ot['image_change'].astype(np.int64), ...], axis=0)
```

iii. Same justification as 3-c (flash-locked bins make the alignment exact). Step 10 Check 2: "Change Match: True" for the three spot-checked sessions.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/{data,timestamps}` — the SDK's filtered running speed (10 Hz low-pass Butterworth, transients z ≥ 10 removed), in cm/s at ~60 Hz. The unfiltered variant `speed_unfiltered` present in the same group is not used.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. CONVERSION_NOTES Step 1: "`RunningSpeed.from_nwb()` | running_speed.py | LOADING | Read running speed (filtered)". Step 3: "Running speed: 10 Hz lowpass Butterworth filter, z-score ≥ 10 transients set to NaN". Step 10 Check 3: "Running speed | Filtered speed from NWB | RunningSpeed.from_nwb() | Same source".

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Mean speed within each 750 ms bin, computed by prefix sum over the ~45 encoder samples per bin (`np.searchsorted` on the running timestamps to get bin boundaries). No interpolation onto the ophys clock; the running stream is binned directly onto the stimulus-onset grid. A bin containing no samples would be left at 0.0 (does not occur at 60 Hz). Then global percentile discretisation (5-c).

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends   = np.searchsorted(running_ts, all_stim_ends,   side='left')
run_cumsum = np.cumsum(running_speed)
run_cumsum = np.concatenate([[0], run_cumsum])
...
running_binned = np.zeros(n_bins, dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
    n_pts = e - s
    if n_pts > 0:
        running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
```

iii. CONVERSION_NOTES Step 5 mapping table: "running speed | output[2] | **Average in 750ms bins**, discretize to 5 percentile bins | running/speed in NWB | 10 Hz Butterworth filtered". Trajectory step 41 point 5: "Behavioral variables (running, pupil) can be averaged within each bin".

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five **equal-percentile** bins. A first full pass over all 202 experiments accumulates every bin-averaged running value in the dataset; `np.percentile` at 0/20/40/60/80/100 gives the edges, with the outer two edges replaced by `±inf` so nothing falls outside. `np.digitize` against the 4 interior edges assigns labels 0–4, then `np.clip` to [0, 4]. Edges used: `[-inf, 0.00408, 0.788, 15.82, 33.13, inf]` cm/s. The realised global distribution is 20.000 %/20.000 %/20.000 %/20.000 %/20.000 % (verified by me from `converted_data.pkl`).

ii.
```python
def discretize_values(values, n_bins, bin_edges=None):
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
# Pass 1 over every experiment
stats = process_experiment(nwb_path, row, collect_stats_only=True)
all_running.extend(stats['running_values'])
...
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
# Pass 2, per trial
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. The instructions require "five equal percentile bins"; CONVERSION_NOTES Step 5 Key Decision 7: "**Discretization**: Compute percentile bin edges across ALL valid time points in dataset (2-pass), then assign bins." Step 7 verifies "Running speed bins | 20% each (uniform)".

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running is binned onto the same stimulus-onset grid as the neural data (`all_stim_starts` / `all_stim_starts + 0.75`), using `searchsorted` on the running clock rather than the ophys clock. Since both rigs' clocks are hardware-synced to a common time base, bin *k* of the running row covers exactly the same wall-clock window as bin *k* of the neural matrix.

ii.
```python
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts,   all_stim_starts, side='left')
run_bin_starts   = np.searchsorted(running_ts, all_stim_starts, side='left')
```

iii. Step 3: "Temporal alignment: All data streams synchronized via NI PCI-6612 at 100 kHz". Step 10 Check 2 verified "Running Match: True" against raw NWB for three sessions. (I independently reproduced session 0 / trial 0: raw bin means `[31.83, 44.64, 40.82, 45.22, 43.69, 31.72, 5.01, 34.56, 50.65, 49.29]` → labels `[3 4 4 4 4 3 2 4 4 4]`, identical to the stored row.)

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (the ellipse-fit pupil **area**, already NaN on blink frames) together with `acquisition/EyeTracking/likely_blink/data`. Diameter is derived from area as an equivalent-circle diameter. `pupil_tracking/width` and `height` are available but not used. Three of the 202 experiments have no `EyeTracking` group at all.

ii.
```python
has_eye_tracking = 'EyeTracking' in f['acquisition']
if has_eye_tracking:
    pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
    pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
    likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. CONVERSION_NOTES Step 4: "Pupil tracking | area, width, height available | pupil_area with NaN for blinks | DeepLabCut, blink detection z>3 | **Use area → compute diameter, interpolate NaNs**". Step 1 flags the SDK's blink logic: "`determine_likely_blinks()` | eye_tracking_processing.py | PROCESSING | Z-score blink detection (threshold=3.0, dilation=2 frames)".

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pipeline:
1. Blink frames → NaN (redundant: the stored `area` is already NaN there — I confirmed `area[likely_blink]` is 100 % NaN), and non-positive areas → NaN.
2. `diameter = 2·sqrt(area/π)`.
3. Linear interpolation over NaN gaps in **index space** across the whole session (`np.interp`, flat extrapolation at the edges), so blinks are imputed rather than dropped.
4. Mean within each 750 ms bin via prefix sums of valid samples (`pup_cumsum / pup_count_cumsum`); after step 3 there are no NaNs left, so this is a plain mean.
5. Per trial, if all bins are NaN (i.e. no eye tracking at all) the whole trial is assigned the **middle bin (2)**; otherwise any residual NaN bins are interpolated within the trial before discretisation.

4 of 202 sessions end up with a constant pupil row; this pushes bin 2 to 21.4 % of all timepoints instead of 20 % (verified from `converted_data.pkl`: `[0.1964, 0.1964, 0.2143, 0.1964, 0.1964]`).

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
# Also NaN out negative or zero values
pupil_area[pupil_area <= 0] = np.nan
# Compute diameter from area: d = 2*sqrt(area/pi)
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
# Interpolate NaNs
pupil_diameter = interpolate_nans(pupil_diameter)
...
pup_valid = ~np.isnan(pupil_diameter)
pup_filled = np.where(pup_valid, pupil_diameter, 0.0)
pup_cumsum = np.concatenate([[0], np.cumsum(pup_filled)])
pup_count_cumsum = np.concatenate([[0], np.cumsum(pup_valid.astype(np.float64))])
...
    n_valid = pup_count_cumsum[e] - pup_count_cumsum[s]
    if n_valid > 0:
        pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
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

iii. CONVERSION_NOTES Step 5 Key Decision 6: "**Pupil NaN handling**: Linear interpolation for blink frames before computing diameter and averaging", and the mapping table: "pupil area → diameter | output[3] | 2*sqrt(area/pi), interpolate NaN, avg in bins, 5 percentile bins | pupil_tracking/area | Interpolate blinks". The middle-bin fallback is not separately justified in the notes.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: 5 equal-percentile bins with edges computed globally in Pass 1 over all bin-averaged pupil values, NaNs excluded before `np.percentile`, outer edges set to `±inf`. Edges used: `[-inf, 73.88, 83.84, 92.87, 105.38, inf]` px.

ii.
```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
if len(valid_pupil) > 0:
    _, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
else:
    pupil_bin_edges = np.array([-np.inf, 0, 1, 2, 3, np.inf])
...
pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. Same as 5-c — Step 5 Key Decision 7 (global 2-pass percentile edges). Step 7 verifies "Pupil diameter bins | 20% each (uniform global)". Note the edges are pooled across mice, so between-animal differences in absolute pupil size contribute to the labels.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same stimulus-onset bin grid, with `searchsorted` performed on the eye-tracking clock (~30 Hz, ~22 samples per bin). Bin *k* of the pupil row therefore covers the same window as bin *k* of the neural matrix.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends   = np.searchsorted(pupil_ts, all_stim_ends,   side='left')
```

iii. Same as 5-d — all streams share the hardware-synced clock, and all are binned on the common `all_stim_starts` grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of `intervals/trials`: `hit`, `miss`, `false_alarm`, `correct_reject`.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. CONVERSION_NOTES Step 5 mapping table: "hit/miss/fa/cr | output[4] | Categorical (4 classes) | trials table | Static per trial". Step 10 Check 4: "Trial outcomes | Hit+Miss for Go, FA+CR for Catch | Confirmed | Correct".

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. A priority `if/elif` chain maps the booleans to codes 0=hit, 1=miss, 2=false_alarm, 3=correct_rejection, with an unreachable fallback to 1 (miss). The trial-level scalar is then **broadcast across all `n_bins`** of the trial so that row 4 of the output is a constant time series, satisfying the "static per-trial" spec while keeping the output array rectangular. Realised distribution: hit 30.7 %, miss 56.8 %, FA 1.8 %, CR 10.7 %.

ii.
```python
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
```

iii. CONVERSION_NOTES Step 5 says trial outcome is "Static per trial"; the target format allows either per-trial or time-varying outputs, and replicating keeps a single `(5, n_bins)` array per trial. Step 12 notes the consequence for decoding: "Trial outcome is static per trial (repeated across time bins), so there are effectively fewer independent samples".

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Experiments with 0 valid ROIs** → the experiment is skipped with a printed warning.
- **Experiments with no images-presentation interval** (`find_stim_key` returns None) → skipped with a warning.
- **Experiments producing < 2 trials** → skipped with a printed reason.
- **Trials with no stimulus presentations carrying their `trials_id`** → silently skipped.
- **Missing eye tracking** (3 experiments) or all-NaN pupil in a trial → the whole trial's pupil row is set to the middle bin (2).
- **Blinks / non-positive pupil area** → NaN, then linearly interpolated over the session; residual NaN bins interpolated within the trial.
- **`is_change` NaN** → coerced to 0.
- **`omitted` stimulus flashes** → image identity forward-filled (backward-filled if the first flash is omitted).
- **Empty bins** (no ophys frame or no running sample in a 750 ms window) → the value stays 0 rather than being flagged.
- **Unrecognised image name** → silently coded 0 (`img_to_idx.get(img, 0)`).
- **Trial matching none of hit/miss/FA/CR** → silently coded 1 (miss).

There is **no** `try/except` around per-experiment processing, so an unexpected read error would abort the whole run rather than skip one experiment.

ii.
```python
if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
...
if stim_key is None:
    print(f"  WARNING: No stimulus presentations found for {experiment_id}, skipping")
    return None
...
if result is None or result['n_trials'] < 2:
    skipped += 1
    reason = 'no result' if result is None else f'only {result["n_trials"]} trials'
    print(f"  Skipped experiment {eid}: {reason}")
    continue
```
```python
def interpolate_nans(values):
    """Linearly interpolate NaN values in a 1D array."""
    valid = ~np.isnan(values)
    if valid.sum() == 0:
        return values  # All NaN, can't interpolate
    if valid.sum() == len(values):
        return values  # No NaN
    result = values.copy()
    x = np.arange(len(values))
    result[~valid] = np.interp(x[~valid], x[valid], values[valid])
    return result
```
```python
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
```

iii. CONVERSION_NOTES Step 5 Key Decisions 5 and 6 cover omissions and pupil NaNs. Step 10 Check 5 ("Check for edge cases") reports: "Minimum trials per session: 39 (above the 2-trial minimum); No NaN/Inf in neural data; All output values are valid integers; No non-integer outputs found", and Step 9 reports 0 experiments skipped in the full run. The middle-bin pupil fallback, the `get(img, 0)` fallback and the miss fallback are not individually justified in the notes.

## 9-a. What are the most time-consuming steps of the code?

i. Measured from `conversion_full_out.txt`: Pass 1 = 261.6 s, Pass 2 = 246.6 s, pickling = 1.1 s, total 510.4 s for 202 experiments (~2.5 s per experiment per pass). The dominant cost in both passes is the same: opening each NWB file and reading the full-session dF/F array (`f[...]['data'][:]`, e.g. 140,208 × 27 … up to 666 neurons) plus the `np.cumsum` over it, i.e. the work is I/O- and memory-bandwidth-bound on the neural traces. Because Pass 1 performs that identical work and then throws it away, roughly **half the total runtime is avoidable**. Secondary costs are the per-bin Python loops (~600 k iterations per pass for neural, running and pupil each) and the `list.extend(...tolist())` accumulation of ~600 k Python floats for running and pupil.

The script prints per-pass timing and an ETA, and `process_experiment` returns an `elapsed` field, but the notes never name the bottleneck explicitly.

ii.
```python
    for idx, (_, row) in enumerate(active_exps.iterrows()):
        ...
        stats = process_experiment(nwb_path, row, collect_stats_only=True)   # Pass 1
        ...
        if (idx + 1) % 20 == 0 or idx == len(active_exps) - 1:
            print(f"  Pass 1: {idx+1}/{len(active_exps)} experiments processed")
    ...
            rate = (idx + 1) / elapsed
            remaining = (len(active_exps) - idx - 1) / rate
            print(f"  Pass 2: {idx+1}/{len(active_exps)} experiments "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
```

iii. CONVERSION_NOTES Step 6 lists the optimisations that were made — "Cumulative sum-based bin averaging (avoids per-bin boolean masking); np.searchsorted for efficient bin boundary finding; Sample 10 experiments for image name collection (not all 202)" — and Step 7 estimates ~24 min for the full run against the 15 min guidance; the run finished in 8.5 min, so no further optimisation was pursued.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python loops are trivially vectorisable, all of them inside the per-trial loop of `process_experiment`:
- the neural bin loop — `(dff_cumsum[ends] - dff_cumsum[starts]).T / counts` computes all bins at once;
- the running bin loop and the pupil bin loop — same prefix-sum trick applied with array indices;
- the `omitted` forward-fill loop — a standard `np.maximum.accumulate` over the indices of non-omitted flashes.

Beyond that, the outer loop over trials could be dropped entirely: every bin in the session can be averaged in one vectorised statement and then `np.split` on the `trials_id` boundaries. The inline comments already claim vectorisation (`# --- Neural data: vectorized bin averaging ---`, `# --- Running speed: vectorized bin averaging ---`) but the code underneath is a per-bin `for` loop — only the O(1) per-bin cost is vectorised, not the iteration.

Also `running_values.extend(running_binned.tolist())` / `pupil_values.extend(...)` build multi-hundred-thousand-element Python lists that are then converted with `np.array`; collecting per-experiment arrays and `np.concatenate`-ing once would be cheaper.

ii.
```python
            # --- Neural data: vectorized bin averaging ---
            neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
            for bi, si in enumerate(trial_stim_indices):
                s, e = ophys_bin_starts[si], ophys_bin_ends[si]
                n_frames = e - s
                if n_frames > 0:
                    neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```
```python
            # Forward-fill omitted presentations
            for bi in range(len(images)):
                if images[bi] == 'omitted':
                    if bi > 0:
                        images[bi] = images[bi - 1]
```
```python
            running_values.extend(running_binned.tolist())
            ...
            pupil_values.extend(pupil_binned.tolist())
```

iii. CONVERSION_NOTES Step 6 frames the cumsum as the vectorisation win ("Cumulative sum-based bin averaging (avoids per-bin boolean masking)"). Since the measured run time (8.5 min) came in under the 15-minute threshold in the instructions, no further vectorisation was attempted.

## 9-c. What processing does the code repeat multiple times?

i. **Every experiment is fully processed twice.** `process_experiment(..., collect_stats_only=True)` is called for all 202 experiments in Pass 1 and `process_experiment(..., collect_stats_only=False)` for the same 202 in Pass 2. In the `collect_stats_only` branch the `continue` that discards the work sits *after* everything has been computed, so Pass 1 re-opens each NWB file, re-reads the full dF/F matrix, recomputes the ROI filter, recomputes `dff_cumsum`, recomputes every trial's `neural_matrix`, the image forward-fill, the image codes, the change flags and the trial outcome — then keeps only the running and pupil lists. This is the single largest inefficiency in the script (~260 s of the 510 s total).

Additionally, `collect_all_image_names` opens and reads 10 of the NWB files a third time, and `subjects_list.index(...)` / `brain_regions_list.index(...)` do a linear scan per session (negligible).

ii.
```python
            # --- Neural data: vectorized bin averaging ---
            neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
            for bi, si in enumerate(trial_stim_indices):
                ...
            # --- Image identity ---
            images = stim_image_name[trial_stim_indices].copy()
            ...
            # --- Trial outcome ---
            if trial_hit[trial_idx]:
                outcome = 0  # Hit
            ...
            if collect_stats_only:
                continue          # <-- everything above is thrown away
```
```python
    for idx, (_, row) in enumerate(active_exps.iterrows()):
        stats = process_experiment(nwb_path, row, collect_stats_only=True)   # Pass 1
    ...
    for idx, (_, row) in enumerate(active_exps.iterrows()):
        result = process_experiment(nwb_path, row, collect_stats_only=False) # Pass 2
```

iii. The two-pass design is documented in CONVERSION_NOTES Step 6: "`convert_data.py` with two-pass approach: 1. Pass 1: Collect running speed and pupil statistics for percentile bin computation 2. Pass 2: Full conversion with discretization using global percentile bins." A second pass is genuinely required to compute *global* percentile edges before discretising, but the notes do not acknowledge that Pass 1 re-does the neural work, nor that the Pass-2 trial dictionaries could have been cached from Pass 1 (they hold `running_speed_raw`/`pupil_diameter_raw` already).

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Discarded work:
- **All of Pass 1's neural/image/change/outcome computation** (see 9-c) — computed and dropped at the `continue`.
- **Unused NWB datasets read into memory every call**: `stim_stop`, `stim_omitted` (omissions are detected from the string `'omitted'` instead), `trial_start`, `trial_stop`, `trial_change_time`, and `valid_trial_ids` / `n_total_rois` / `pupil_area_raw`'s intermediate copy are all loaded or computed and never used.
- **Redundant blink masking**: `pupil_area[likely_blink] = np.nan` has no effect because the stored `area` is already NaN on exactly those frames (verified: `area[likely_blink]` is 100 % NaN).
- **Dead NaN bookkeeping for pupil**: after `interpolate_nans`, `pup_valid` is all-True, so `pup_count_cumsum` is just an integer ramp and the "valid count" division is equivalent to a plain mean.
- **Empty input arrays**: `np.zeros((0, n_bins), dtype=np.float32)` is allocated for every one of the 51,992 trials even though the task specifies no decoder inputs.
- **Raw running/pupil per-trial arrays** are stored in the intermediate `output_trials` dicts and then replaced by their discretised versions; only the labels reach the pickle.
- `elapsed` is measured and returned per experiment but never printed or used.

ii.
```python
        stim_stop = stim['stop_time'][:]
        stim_omitted = stim['omitted'][:]
        ...
        trial_start = trials['start_time'][:]
        trial_stop = trials['stop_time'][:]
        trial_change_time = trials['change_time'][:]
        ...
        valid_trial_ids = trial_ids[trial_mask]
```
```python
        pupil_area = pupil_area_raw.copy().astype(float)
        pupil_area[likely_blink] = np.nan     # already NaN in the file
```
```python
            input_trials.append(np.zeros((0, n_bins), dtype=np.float32))
```
```python
    elapsed = time.time() - t0
    if collect_stats_only:
        return {'running_values': running_values, 'pupil_values': pupil_values}
```

iii. Not discussed in CONVERSION_NOTES. Step 6 only records the three optimisations that were added; Step 7's run-time table concluded the estimated ~24 min was acceptable, and the actual 8.5 min run removed any pressure to trim the redundant work.
