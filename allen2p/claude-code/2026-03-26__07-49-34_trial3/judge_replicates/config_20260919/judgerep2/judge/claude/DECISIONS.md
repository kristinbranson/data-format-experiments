# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI bypassed the AllenSDK entirely and read the released NWB/HDF5 files directly with `h5py`, using the shipped metadata CSV as the index. The experiment inventory comes from `data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv`. That table lists all 1,936 released experiments, so the AI intersects it with the set of NWB files actually present on disk (284 files) by parsing the experiment id out of each filename. It then further restricts to four "active behavior" session types (`OPHYS_1_images_A`, `OPHYS_3_images_A`, `OPHYS_4_images_B`, `OPHYS_6_images_B`), dropping the passive `OPHYS_2`/`OPHYS_5` sessions. No `project_code` filter is applied, so both `VisualBehavior` (168 experiments) and `VisualBehaviorMultiscope` (34 experiments) are included, for a final selection of **202 experiments from 38 mice**. Each selected file is then opened three separate times: Pass 1 to harvest the global image-name vocabulary, Pass 2 to accumulate running-speed and pupil samples for percentile bin edges, Pass 3 to load everything and build trials.

ii.
```python
DATA_DIR = Path('data/visual-behavior-ophys-1.1.0')
NWB_DIR = DATA_DIR / 'behavior_ophys_experiments'
META_DIR = DATA_DIR / 'project_metadata'

ACTIVE_SESSION_TYPES = [
    'OPHYS_1_images_A', 'OPHYS_3_images_A',
    'OPHYS_4_images_B', 'OPHYS_6_images_B',
]

def get_experiment_list(sample=False):
    """Get list of active experiment IDs with metadata."""
    exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')

    # Get NWB files on disk
    nwb_files = list(NWB_DIR.glob('*.nwb'))
    nwb_ids = set()
    for f in nwb_files:
        try:
            eid = int(f.stem.split('_')[-1])
            nwb_ids.add(eid)
        except ValueError:
            pass

    # Filter to on-disk, active sessions
    mask = (
        exp_table['ophys_experiment_id'].isin(nwb_ids) &
        exp_table['session_type'].isin(ACTIVE_SESSION_TYPES)
    )
    active_exps = exp_table[mask].copy()
```

```python
def load_experiment_data(nwb_path, experiment_id):
    with h5py.File(nwb_path, 'r') as f:
        cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
        valid_roi = cell_table['valid_roi'][()].astype(bool)
        ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][()]
        events_data = f['processing']['ophys']['event_detection']['data'][()]
        events_valid = events_data[:, valid_roi]
        trials_grp = f['intervals']['trials']
        ...
        running_speed = f['processing']['running']['speed']['data'][()]
        running_ts = f['processing']['running']['speed']['timestamps'][()]
        pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
```

iii. From CONVERSION_NOTES.md Step 1/Step 4: the AI verified in the SDK source that dF/F, events, trials, running speed, and eye tracking are all **pre-computed and stored in the NWB file**, so `h5py` reads return the same arrays the SDK would return, without the SDK's overhead. Step 4 records that only 284 of the 1,936 catalogued experiments are on disk and resolves the discrepancy as "We have a subset. 38 mice is consistent with partial download." Passive sessions were excluded because "OPHYS_2, OPHYS_5 are passive viewing (no lick spout, satiated mice). No meaningful trial outcomes." (Step 5, Key Decision 2). The trajectory shows the AI reasoning that the instruction "Collect and convert data under the 'Visual Behavior' task" meant the active change-detection behavior sessions.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the selected experiments, sorted and cast to `str`. A dict maps mouse id string to its index, and each session appends its mouse's index to `subject_idx`. This yields 38 subjects.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
...
'subjects': [str(s) for s in subjects],
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 12: "**Subject IDs**: Use mouse_id from experiment table." `mouse_id` is the canonical per-animal identifier in the Allen metadata; sorting makes the index assignment deterministic. Step 4 cross-checks the count (38 on disk vs. 82 in the full release) and attributes the difference to the partial download.

## 1-c. How are the data split into sessions?

i. **Each NWB experiment file is treated as one "session" in the output.** The `ophys_session_id` column is never used. For the 168 single-plane `VisualBehavior` experiments this is equivalent to a session, because those rigs record one imaging plane per session. For the 34 `VisualBehaviorMultiscope` experiments it is not: those 34 experiments come from only **6 distinct `ophys_session_id`s** recorded on the MESO.1 rig, so each real multiscope session is emitted as 5–8 separate output "sessions" that share identical trial timing, identical behavior, and identical outputs, but carry disjoint 4–22-neuron subsets of the simultaneously recorded population. The result is 202 output sessions instead of the 174 actual sessions, with one mouse (457841) contributing 34 "sessions". This is visible in the conversion log as runs of sessions with identical trial counts (`209, 209, 209, 209, 209, 209, 209, 309, 309, ...`) and tiny neuron counts.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    ...
    raw_data = load_experiment_data(nwb_path, eid)
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
    all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])

    # Brain region index for each neuron
    region_idx = np.full(result['n_cells'], region_to_idx[row['targeted_structure']], dtype=np.int64)
    all_brain_region_idx.append(region_idx)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 13: "**Each NWB experiment = one 'session'** in output format (one imaging plane with its own neurons)." The AI was aware of the mismatch: Step 4 explicitly records "Active sessions (OPHYS_1,3,4,6): 202 experiments, 174 sessions, 38 mice" and "Equipment: CAM2P.3/4/5 (168 single-plane), MESO.1 (34 multi-plane)", and the Step 9 consistency table lists "202 experiments" against "174 active (on-disk)" with the verdict "Yes (experiments from 174 sessions)". No rationale is given for preferring the plane-level grouping over the session-level grouping, and the resulting duplication of behavioral outputs across pseudo-sessions is not discussed anywhere.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. The trial mask keeps trials that are `go` **or** `catch` and are neither `aborted` nor `auto_rewarded`. For each kept trial the AI takes the half-open window `[start_time, stop_time)` on the resampled 30 Hz time grid, giving variable-length trials (mean 254 bins ≈ 8.5 s, range 210–377). Trials with fewer than 3 time bins are dropped, as are trials for which none of `hit/miss/false_alarm/correct_reject` is set.

ii.
```python
# Filter trials: keep Go and Catch, exclude Aborted and Auto-rewarded
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]

for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop = trials['stop_time'][trial_idx]

    # Find regular grid indices within this trial
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]

    if len(trial_time_indices) < 3:
        continue

    trial_ts = regular_ts[trial_time_indices]
    n_tp = len(trial_ts)
```

iii. CONVERSION_NOTES.md Step 5, Key Decisions 3 and 5: "Exclude aborted and auto-rewarded trials: Per task instructions" and "**Trial window**: Use trial start_time to stop_time from trials table. Variable length across trials." Step 3 records the trial taxonomy from the whitepaper (Go 87.5%, Catch 12.5%, aborted = premature lick, auto-rewarded = 5 free rewards at session start plus after 10 consecutive misses). Step 10 Check 4 reports the realized Go/Catch split as 87.5%/12.5%, matching the whitepaper. The full window (rather than a fixed window around the change) is used so that image identity and image change can be time-varying within the trial.

## 1-e. How are trials filtered based on quality controls?

i. Five filters, in order: (1) trial-type filter — `go|catch`, `~aborted`, `~auto_rewarded`; (2) experiments with fewer than 2 such trials are skipped before any processing; (3) trials shorter than 3 time bins on the 30 Hz grid are skipped; (4) trials with no outcome flag set are skipped; (5) experiments that end up with fewer than 2 usable trials after per-trial processing are skipped entirely. No session-level d-prime / engagement / behavioral-performance filter is applied. 51,992 trials survive across 202 sessions (min 39 trials/session).

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]

if len(valid_indices) < 2:
    print(f"  WARNING: Only {len(valid_indices)} valid trials in experiment {raw_data['experiment_id']}, skipping")
    return None
...
    if len(trial_time_indices) < 3:
        continue
...
    if trials['hit'][trial_idx]:
        outcome = 0  # hit
    elif trials['miss'][trial_idx]:
        outcome = 1  # miss
    elif trials['false_alarm'][trial_idx]:
        outcome = 2  # false_alarm
    elif trials['correct_reject'][trial_idx]:
        outcome = 3  # correct_reject
    else:
        continue  # Unknown outcome, skip
...
if len(neural_trials) < 2:
    print(f"  WARNING: Only {len(neural_trials)} processed trials in experiment {raw_data['experiment_id']}, skipping")
    return None
```

iii. CONVERSION_NOTES.md Step 3 "Trial curation rules": "Aborted trials: premature lick before change → excluded from performance calculations; Auto-rewarded: 5 free rewards at session start + after 10 consecutive misses; Per task instructions: include Go and Catch trials, exclude Aborted and Auto-rewarded." The ≥2-trial rule is driven by the format requirement that "There needs to be at least two trials within each session in order to evaluate the decoder performance." Step 10 Check 5 confirms "All sessions have >= 2 trials (min: 39)". The AI listed the whitepaper's 10-point session QC criteria in Step 3 but noted these are already applied upstream by the Allen release pipeline, so no session-level QC was re-implemented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from **deconvolved calcium events**, i.e. the NWB dataset `processing/ophys/event_detection/data` (shape `(n_timepoints, n_all_cells)`), not from dF/F. Columns are subset to ROIs with `valid_roi == True` from `processing/ophys/image_segmentation/cell_specimen_table`. The timebase is `processing/ophys/dff/traces/timestamps`, which is the same 140,204-frame timebase the event array uses.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][()]

# --- Neural events (filtered by valid_roi) ---
events_data = f['processing']['ophys']['event_detection']['data'][()]
# events_data shape: (n_timepoints, n_all_cells)
events_valid = events_data[:, valid_roi]  # (n_timepoints, n_valid_cells)
```

```python
'neural_signal': 'calcium events (FastLZeroSpikeInference)',
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 1: "**Neural signal: events (not dF/F)**: Paper says 'We performed our analyses on discrete calcium events.' Use raw events from event_detection." The trajectory shows the AI first leaning toward dF/F ("dF/F is the standard pre-computed fluorescence signal ... and seems like the most appropriate") and then switching: "the paper explicitly states they performed analyses on discrete calcium events ... The instructions emphasize matching the reference processing, so I should use events instead of dF/F." Step 1 also records that the SDK offers an optional causal half-gaussian filter on events (`scale=2.0/31.0 s, n_steps=20`); the AI deliberately used the **unfiltered** events because "The reference paper uses raw calcium events detected via Fast LZeroSpikeInference, not the filtered versions from the SDK."

## 2-b. How is the `neural` data processed?

i. Three operations, and no others: (1) ROI subsetting by `valid_roi`; (2) linear interpolation of every neuron's event trace from the native ophys timebase onto a uniform 30 Hz grid spanning the whole session (`np.arange(ophys_ts[0], ophys_ts[-1], 1/30)`), done column-by-column in a Python loop; (3) clipping to non-negative. No normalization, no smoothing, no z-scoring, no baseline subtraction. Crucially, **neurons from different imaging planes of the same multiscope session are never merged** — each plane stays in its own output session (see 1-c). Per-trial slices are stored as `float32` arrays of shape `(n_cells, n_timepoints)`.

ii.
```python
def interpolate_to_regular_grid(timestamps, data, target_timestamps):
    if data.ndim == 1:
        return np.interp(target_timestamps, timestamps, data)
    else:
        result = np.zeros((len(target_timestamps), data.shape[1]), dtype=np.float32)
        for i in range(data.shape[1]):
            result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
        return result
```

```python
t_start = ophys_ts[0]
t_end = ophys_ts[-1]
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)

# Interpolate events to regular grid
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
# Ensure non-negative (events should be >= 0)
events_resampled = np.maximum(events_resampled, 0)
...
# Neural data: (n_cells, n_timepoints)
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 6: "Resamples all data streams to 30 Hz via linear interpolation ... Events are clipped to >=0 after interpolation." Step 5, Key Decision 4: "**Resample to 30 Hz**: Paper interpolates to 30 Hz. Needed for consistent time bins across Scientifica (31 Hz) and Multiscope (11 Hz). time_bin_size = 33.33 ms." Step 10 Check 3(c/d) claims a MATCH to the paper: "Paper: 'linearly interpolating onto a consistent set of 30hz timestamps'. We interpolate all data to 30 Hz using np.interp, matching the paper's approach." No justification is given for keeping planes separate rather than stacking them per session.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level quality control is the `valid_roi` boolean from the cell specimen table: ROIs with `valid_roi == False` are dropped. Experiments where zero ROIs are valid are skipped. No activity-based, SNR-based, or event-rate-based filtering is applied. In practice this filter removes nothing on this data subset — 29,444 ROIs are present and 29,444 survive — because the public NWB release already ships only valid ROIs.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
cell_specimen_ids = cell_table['cell_specimen_id'][()]

n_valid = valid_roi.sum()
if n_valid == 0:
    print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping")
    return None
...
events_valid = events_data[:, valid_roi]
```

iii. CONVERSION_NOTES.md Step 1 records the SDK finding: "**Cell filtering**: Only automatic filter is `valid_roi` boolean (SVM binary classifier output). `exclude_invalid_rois=True` by default." Step 3 "Neuron curation rules": "`valid_roi == True` (SVM classifier output); Exclusion reasons: union of cells, duplicate, edge/motion affected, apical dendrite, too small/narrow/dim." Step 5, Key Decision 10: "**valid_roi filtering**: Only include neurons with valid_roi=True." Step 10 Check 3(b) argues this reproduces the SDK's own default. Step 4 flags the need for this filter as a discrepancy to resolve ("Need to filter by valid_roi when loading NWB").

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start** (`trials.start_time`), in absolute session time. The whole session is first placed on one uniform 30 Hz grid built from the ophys timestamps, and a trial is then the set of grid points satisfying `start_time <= t < stop_time`. Because every stream (events, running, pupil, image identity, image change) is evaluated on that same `regular_ts` grid and indexed by the same `trial_time_indices`, all streams are aligned to each other by construction. Metadata records `temporal_alignment_event = 'Trial start time (first stimulus onset of trial)'`, `off_start = 0.0`, `off_end = None` (variable-length trials). The residual alignment error is the grid phase, at most one 33.3 ms bin.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
...
trial_ts = regular_ts[trial_time_indices]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
...
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
running_trial = running_resampled[trial_time_indices]
pupil_trial = pupil_resampled[trial_time_indices]
```

```python
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
'off_start': 0.0,
'off_end': None,
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 6: "**Alignment event**: Trial start time (stimulus onset). off_start=0, off_end=None (variable)." Step 3 notes the Allen hardware sync ("NI PCI-6612 board at 100 kHz, all clocks synchronized"), which is what licenses interpolating the 60 Hz running and 30 Hz eye-tracking streams onto the ophys clock. Step 10 Check 5 reports "Neural/output length alignment at trial boundaries" as passing, and Check 2 verified a spot-check of image identity at (session 0, trial 5, timepoint 10) against an independent NWB read.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. `time_bin_size = 33.333 ms` (30 Hz), uniform across every trial, session, and rig. Rebinning **is** applied: it is a resampling, not an integrating rebin — every stream is linearly interpolated onto `np.arange(ophys_ts[0], ophys_ts[-1], 1/30)`. For the 168 single-plane experiments this is a slight downsample from the native ~30.95 Hz (32.31 ms); for the 34 multiscope experiments it is a ~3× **upsample** from the native ~11 Hz. Because the neural signal is a sparse impulse train (~0.25 % of samples nonzero), linear interpolation spreads each event over two grid bins and changes total event mass slightly (measured on one file: total event mass 1084.97 → 1051.73, a 3.1 % loss; nonzero fraction 0.246 % → 0.492 %).

ii.
```python
TARGET_RATE_HZ = 30.0  # Paper: "linearly interpolating onto a consistent set of 30hz timestamps"
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms
...
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
'time_bin_size': TIME_BIN_MS,
'target_rate_hz': TARGET_RATE_HZ,
```

iii. CONVERSION_NOTES.md Step 3 records the paper quote as the source: "Paper analysis time bin | 30 Hz (interpolated) | Paper: 'linearly interpolating onto a consistent set of 30hz timestamps'." Step 4's discrepancy table resolves "Frame rate | 31 Hz (CAM2P), 11 Hz (MESO) | Confirmed ~30.95 Hz for CAM2P | 30 Hz interpolated (paper) | Will resample all to 30 Hz." Step 5, Key Decision 4 adds the format-driven reason: a single common bin size is required because the AI's session set spans two rigs with different native rates. Step 10 Check 4 lists "Time bin: 33.33 ms (30 Hz) - matches paper."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the **stimulus presentations table** in `intervals/<Natural_Images_..._presentations>`, specifically the `image_name`, `start_time`, and `omitted` columns — not from the trials table. The presentations group is located by scanning `f['intervals']` and taking the first key that is not `trials`, not a `spontaneous` table, and not a `movie` table. Omitted flashes are dropped from the lookup list so that an omission does not blank out the image identity.

ii.
```python
stim_key = None
for k in f['intervals'].keys():
    if k != 'trials' and 'spontaneous' not in k.lower() and 'movie' not in k.lower():
        stim_key = k
        break

stim = f['intervals'][stim_key]
stim_data = {
    'start_time': stim['start_time'][()],
    'stop_time': stim['stop_time'][()],
    'image_name': stim['image_name'][()],
    'is_change': stim['is_change'][()].astype(bool),
    'omitted': stim['omitted'][()],
}
```

```python
non_omitted = ~stim_data['omitted']
stim_starts = stim_data['start_time'][non_omitted]
stim_names = stim_data['image_name'][non_omitted]
```

iii. CONVERSION_NOTES.md Step 5 Variable Mapping: "`stimulus_presentations.image_name` | output[0]: image_identity | Map to categorical int, time-varying per ophys frame | 8 natural images." Key Decisions 7 and 8 explain the two edge-case rules: "**Image identity during gray screen**: Use the identity of the image that was just shown (last presented image)" and "**Image identity for omitted flashes**: Continue with previous image identity." The AI's Step 3 notes record the 250 ms image + 500 ms gray = 750 ms flash cycle and the 5 % omission probability, which is what motivated both rules.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Two-stage integer coding. Locally, each experiment's own sorted set of non-omitted image names is built and `np.searchsorted(..., side='right') - 1` finds, for every trial time bin, the index of the most recent flash onset; a Python loop then maps that flash's name to the local index. Globally, Pass 1 over all 202 files collects the union of image names (16 images, `im000`…`im106`, i.e. 8 from image set A and 8 from image set B), sorts them, and a per-experiment `local_to_global` dict remaps local indices to global codes so that codes are consistent across sessions. Names not found map to 0. Output dtype `int64`.

ii.
```python
def get_image_at_timepoints(timepoints, stim_data, all_image_names):
    n_tp = len(timepoints)
    image_idx = np.zeros(n_tp, dtype=np.int64)
    non_omitted = ~stim_data['omitted']
    stim_starts = stim_data['start_time'][non_omitted]
    stim_names = stim_data['image_name'][non_omitted]
    name_to_idx = {name: i for i, name in enumerate(all_image_names)}

    # For each timepoint, find the most recent stimulus onset
    insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1

    for i in range(n_tp):
        if insert_idx[i] >= 0:
            img_name = stim_names[insert_idx[i]]
            image_idx[i] = name_to_idx.get(img_name, 0)
        else:
            image_idx[i] = 0  # Before first stimulus
    return image_idx
```

```python
all_stim_images = sorted(set(
    raw_data['stim']['image_name'][raw_data['stim']['image_name'] != 'omitted']
))
...
global_image_names = sorted(all_image_names_set)
...
local_to_global = {}
for local_idx, name in enumerate(result['all_image_names']):
    if name in global_image_names:
        local_to_global[local_idx] = global_image_names.index(name)
    else:
        local_to_global[local_idx] = 0
...
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. CONVERSION_NOTES.md Step 6: "Image identity at each timepoint determined by most recent non-omitted stimulus onset." Step 9 explains the 16-class vocabulary: "Image identity classes | 16 (8 per image set, 2 image sets)", consistent with the whitepaper's "Each session included 8 images" plus the A→B image-set switch between OPHYS_3 and OPHYS_4. Step 7 sanity-checks the within-session distribution as "~12.5% each (8 images) - uniform", and Step 10 Check 2 verifies a specific value against a fresh NWB read ("Image identity (session 0, trial 5, timepoint 10): expected 'im063', got 'im063'. PASS").

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated at exactly the trial's neural time bins. `get_image_at_timepoints` is called with `trial_ts = regular_ts[trial_time_indices]`, the same array of 30 Hz grid timestamps used to slice `events_resampled`. The resulting vector therefore has the same length as the trial's neural matrix and is index-for-index aligned; it is stacked as row 0 of the `(5, n_timepoints)` output block.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
n_tp = len(trial_ts)

# Neural data: (n_cells, n_timepoints)
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)

# Image identity at each timepoint
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

```python
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)  # (4, n_tp)
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)  # (5, n_tp)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 6 (single alignment event, single grid) and Step 6 ("Resamples all data streams to 30 Hz"). The AI's argument is that alignment is guaranteed rather than asserted, because all streams are looked up on one absolute-time grid before segmentation. Step 10 Check 5 lists "Neural/output length alignment at trial boundaries" as passing; Step 12 also confirms image-change consistency against trial type as a proxy for alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the **stimulus presentations** table: `is_change` combined with `~omitted` selects the flashes that were actual image changes, and `start_time` gives their onsets. The trials table's `change_time`/`go`/`catch` columns are *not* used for this output (the trials-table `is_change` column is read into memory but never referenced).

ii.
```python
def get_image_change_at_timepoints(timepoints, stim_data):
    n_tp = len(timepoints)
    change_signal = np.zeros(n_tp, dtype=np.int64)

    # Get change stimulus presentations (non-omitted, is_change=True)
    change_mask = stim_data['is_change'] & ~stim_data['omitted']
    change_starts = stim_data['start_time'][change_mask]
    change_stops = stim_data['stop_time'][change_mask]
```

iii. CONVERSION_NOTES.md Step 5 Variable Mapping lists "`trials.is_change` + stimulus timing | output[1]: image_change | Binary 1 at change timepoint, 0 otherwise, time-varying | 1 for one 750ms window at change" — the notes name the trials table but the code actually uses the stimulus-presentations `is_change`, which is the flash-level ground truth. Step 12 "Additional Verification" defends the choice empirically: "Catch trials: 0/6,515 have image_change signal (correct: catch = sham change); Go trials: 0/45,477 missing image_change signal (all have change)."

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary indicator built by marking, for each change flash in the session, all trial time bins in the half-open 750 ms window `[change_onset, change_onset + 0.75)`. The loop runs over all change flashes of the whole session for every trial, and any flash whose window does not intersect the trial simply marks nothing. Result is `int64` in `{0, 1}`. Realized fraction of 1s across the whole dataset: 7.7 %.

ii.
```python
    # For each change, mark timepoints within the 750ms image presentation interval
    for cs, ce in zip(change_starts, change_stops):
        # Mark the full image interval (stimulus + gray) as change
        # Use 750ms window from change start
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1

    return change_signal
```

```python
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
...
img_change = trial_data_out['image_change'].astype(np.int64)
```

iii. CONVERSION_NOTES.md Step 6: "Image change signal: 1 during 750ms window starting at change onset." The 750 ms width comes from Step 3's record of the stimulus cycle: "Image presentation | 250 ms stimulus + 500 ms gray = 750 ms | Paper" — i.e. the changed image flash plus its following gray interval, which is the natural unit of the change event. Step 9's consistency table cross-checks the resulting density: "Image change fraction | ~1/13 flashes | 7.7% | Yes (~1/13=7.7%)", using the whitepaper's mean change time of ~4.25 s within a ~8.5 s trial.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — the variable is natively binary. `output_values[1] = ['no_change', 'change']`, two classes, encoded 0/1. The only implicit "threshold" is the choice of the 750 ms post-onset window that defines how many bins carry the label 1 (roughly 22–23 bins at 30 Hz).

ii.
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
...
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1
```

```python
output_values = [
    global_image_names,  # image identity values
    ['no_change', 'change'],  # image change values
    ...
]
```

iii. The task specification states "Image change, binary variable. Have value of 1 right after a change in image identity, otherwise 0." CONVERSION_NOTES.md Step 5 restates this as "1 for one 750ms window at change", and Step 9 validates the resulting class balance (92.3 % / 7.7 %) against the expected ~1-in-13 flash rate.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity: `get_image_change_at_timepoints` is evaluated on `trial_ts`, the exact 30 Hz grid timestamps of the trial's neural matrix, so the indicator is index-for-index aligned and becomes row 1 of the output block. Because the window is defined in absolute time and applied to the trial's own timestamps, a change that begins near the end of a trial correctly carries its remaining window into the following trial's opening bins.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
...
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
...
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)
```

iii. Same single-grid argument as 3-c (CONVERSION_NOTES.md Step 5 Key Decision 6, Step 6). Step 12's "Additional Verification" is the AI's alignment evidence: the change indicator appears in exactly the Go trials and never in the Catch trials, which would not hold if the change windows were misregistered relative to trial boundaries.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. From the NWB running module: `processing/running/speed/data` (speed in cm/s, ~60 Hz) with its companion `processing/running/speed/timestamps`.

ii.
```python
# --- Running speed ---
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

```python
'running_speed': running_speed,
'running_ts': running_ts,
```

iii. CONVERSION_NOTES.md Step 1 identifies `RunningSpeed.from_nwb()` as the SDK loader for "running speed (cm/s)", and Step 2 records the stream properties ("Running speed: 270,240 samples (~60 Hz)"). Step 5 Variable Mapping: "`running/speed/data` | output[2]: running_speed | Interpolate to 30 Hz, discretize into 5 percentile bins, time-varying." This is the same underlying array the SDK's `dataset.running_speed` exposes.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation from the native ~60 Hz timestamps onto the session's 30 Hz grid via `np.interp` (which clamps rather than extrapolating, so no NaNs are produced), then per-trial slicing, then discretization into 5 bins using globally pre-computed percentile edges. Note that the percentile edges are computed in Pass 2 from the **raw, native-rate, whole-session** running traces concatenated over all 202 experiments — not from the trial-segmented, resampled values that actually end up in the output.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
...
'running_speed': running_trial.astype(np.float32),
```

```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...
        running = f['processing']['running']['speed']['data'][()]
        all_running_values.append(running.astype(np.float32))
...
all_running_cat = np.concatenate(all_running_values) if all_running_values else np.array([0.0])
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
```

iii. CONVERSION_NOTES.md Step 6 describes the 3-pass design: "(2) compute global percentile bins for running/pupil, (3) process experiments", and "Resamples all data streams to 30 Hz via linear interpolation". Global (rather than per-session) edges are justified in Step 7: "Running speed bins slightly unequal within sessions (expected, bins computed globally)" — i.e. the AI accepted within-session imbalance in exchange for a single, consistent class definition across the dataset.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins. `compute_percentile_bins` takes `np.percentile(valid, np.linspace(0, 100, 6))`, nudges any non-increasing edge by 1e-10 to guarantee strict monotonicity, and `digitize_to_bins` applies `np.digitize(values, bin_edges[1:-1])` and clips to `[0, 4]`; NaN maps to bin 0. Edges: `[-24.13, -0.0041, 0.336, 14.18, 33.13, 99.92]` cm/s. The realized class distribution in the converted data is `{0: 0.199, 1: 0.203, 2: 0.184, 3: 0.209, 4: 0.205}` — close to but not exactly uniform, because the edges were derived from whole-session 60 Hz samples rather than from the in-trial 30 Hz samples.

ii.
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.linspace(0, 1, n_bins + 1)
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    # Ensure unique edges (handle constant values)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-10
    return edges


def digitize_to_bins(values, bin_edges):
    n_bins = len(bin_edges) - 1
    binned = np.digitize(values, bin_edges[1:-1])  # 0 to n_bins-1
    binned = np.clip(binned, 0, n_bins - 1)
    binned[np.isnan(values)] = 0
    return binned.astype(np.int64)
```

```python
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. Directly from the task spec ("Running speed, discretized into five equal percentile bins"). CONVERSION_NOTES.md Step 5 planned sanity check: "Running speed and pupil bins are roughly equal-sized (by definition of percentile bins)"; Step 9's consistency table reports "Running speed bins | 5 equal percentile | 18-21% each | Yes (equal percentile)". The epsilon nudge is documented in-code as handling degenerate/constant-valued edges.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the session-wide 30 Hz grid *before* trial segmentation, then sliced with the same `trial_time_indices` used for the neural matrix. Alignment is therefore exact by construction, and the discretized vector becomes row 2 of the `(5, n_timepoints)` output block.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
...
running_trial = running_resampled[trial_time_indices]
```

iii. Step 3 of CONVERSION_NOTES.md records the hardware basis for cross-stream interpolation: "**Temporal sync**: NI PCI-6612 board at 100 kHz, all clocks synchronized." Step 5 Key Decision 6 fixes a single alignment event and a single grid for all streams, and Step 6 states that all streams are resampled to 30 Hz, which is what makes the shared-index slicing valid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. From `acquisition/EyeTracking/pupil_tracking/area` (the fitted **pupil area**, ~30 Hz) with `acquisition/EyeTracking/pupil_tracking/timestamps`, plus the `acquisition/EyeTracking/likely_blink/data` boolean used to censor blinks. The AI deliberately used area rather than a width/diameter column, treating it as a proxy for diameter. If the EyeTracking group is absent the whole stream is set to `None` and the session's pupil output becomes all-NaN; this happened for 3 of the 202 experiments.

ii.
```python
pupil_area = None
pupil_ts = None
likely_blink = None
try:
    pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
    pupil_area = pupil_tracking['area'][()]
    pupil_ts = pupil_tracking['timestamps'][()]
    likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
except (KeyError, Exception):
    print(f"  WARNING: No pupil tracking data in {experiment_id}")
```

iii. CONVERSION_NOTES.md Step 5 Variable Mapping is explicit about the substitution: "`EyeTracking/pupil_tracking/area` | output[3]: pupil_diameter | Interpolate to 30 Hz, handle blinks (NaN→interpolate), discretize into 5 percentile bins, time-varying | **Use pupil area as proxy for diameter**." Step 4 flags the data issue that motivates blink handling: "Pupil | area with NaNs during blinks | ~9% blinks | 30 Hz camera | Need to handle NaN/blinks."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Four steps: (1) frames flagged `likely_blink` are set to NaN; (2) `interpolate_nans` linearly fills every NaN gap (blink-induced and otherwise) along the native pupil timeline, clamping at the edges, and returns all-zeros if the entire trace is NaN; (3) `np.interp` resamples the cleaned trace onto the session's 30 Hz grid; (4) values are discretized with globally pre-computed percentile edges. As with running speed, the edges come from Pass 2's concatenation of raw, whole-session, non-blink pupil samples across all experiments, not from the trial-segmented values.

ii.
```python
def interpolate_nans(arr):
    """Linearly interpolate NaN values in a 1D array."""
    nans = np.isnan(arr)
    if not nans.any():
        return arr.copy()
    if nans.all():
        return np.zeros_like(arr)
    result = arr.copy()
    x = np.arange(len(arr))
    result[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return result
```

```python
pupil_resampled = None
if raw_data['pupil_area'] is not None:
    pupil_area = raw_data['pupil_area'].copy().astype(float)
    pupil_ts = raw_data['pupil_ts']
    likely_blink = raw_data['likely_blink']

    # Set blink frames to NaN
    if likely_blink is not None:
        pupil_area[likely_blink] = np.nan

    # Interpolate NaNs in pupil area before resampling
    pupil_area = interpolate_nans(pupil_area)

    # Resample to regular grid
    pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
```

```python
pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()].astype(float)
blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
pupil_area[blink] = np.nan
valid_pupil = pupil_area[~np.isnan(pupil_area)]
if len(valid_pupil) > 0:
    all_pupil_values.append(valid_pupil.astype(np.float32))
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 9: "**Pupil NaN handling**: During blinks (likely_blink=True or NaN), linearly interpolate. Compute percentile bins from non-blink data." Step 6 restates: "Pupil blinks (likely_blink=True) set to NaN and interpolated before resampling." The stated rationale is to keep the pupil trace continuous on the ophys grid while preventing blink artifacts (and the NaNs they leave behind) from contaminating either the resampled signal or the percentile edges.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The same `compute_percentile_bins` / `digitize_to_bins` machinery as running speed: 5 equal-percentile bins with globally computed edges `[125.6, 4374.1, 5527.6, 6747.1, 8598.7, 323783.3]` (in pupil-area units), NaN → bin 0. Realized distribution: `{0: 0.228, 1: 0.185, 2: 0.191, 3: 0.190, 4: 0.206}`. Bin 0 is over-represented mainly because the 3 experiments with no eye-tracking data contribute entire sessions of NaN that collapse to bin 0.

ii.
```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

```python
if pupil_resampled is not None:
    pupil_trial = pupil_resampled[trial_time_indices]
else:
    pupil_trial = np.full(n_tp, np.nan)
```

iii. Task spec: "Pupil diameter, discretized into five equal percentile bins." CONVERSION_NOTES.md Step 9 reports "Pupil diameter bins | 5 equal percentile | 19-23% each | Yes (roughly equal)", and Step 5's planned sanity check anticipated exactly this. The AI's argument for percentile binning of *area* is that a rank-preserving transform of diameter yields the same percentile bins, so the substitution is harmless for a discretized output.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Identical to running speed: resampled onto the session-wide 30 Hz grid before segmentation, then sliced with the same `trial_time_indices` as the neural matrix, giving exact index-for-index alignment. It becomes row 3 of the output block. Sessions without eye tracking get an all-NaN vector of the correct length, so shapes stay consistent.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
...
if pupil_resampled is not None:
    pupil_trial = pupil_resampled[trial_time_indices]
else:
    pupil_trial = np.full(n_tp, np.nan)
```

iii. Same rationale as 5-d: CONVERSION_NOTES.md Step 3's note on the 100 kHz NI hardware sync licenses interpolating the 30 Hz eye camera onto the ophys clock, and Step 5 Key Decision 6 plus Step 6 put every stream on one grid so that a single index array aligns them all.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. From the four mutually exclusive boolean columns of the NWB trials table: `hit`, `miss`, `false_alarm`, `correct_reject`. They are checked in that order and mapped to codes 0–3; a trial with none of them set is dropped.

ii.
```python
trials = {
    ...
    'hit': trials_grp['hit'][()].astype(bool),
    'miss': trials_grp['miss'][()].astype(bool),
    'false_alarm': trials_grp['false_alarm'][()].astype(bool),
    'correct_reject': trials_grp['correct_reject'][()].astype(bool),
    ...
}
```

```python
# Trial outcome (static)
if trials['hit'][trial_idx]:
    outcome = 0  # hit
elif trials['miss'][trial_idx]:
    outcome = 1  # miss
elif trials['false_alarm'][trial_idx]:
    outcome = 2  # false_alarm
elif trials['correct_reject'][trial_idx]:
    outcome = 3  # correct_reject
else:
    continue  # Unknown outcome, skip
```

iii. CONVERSION_NOTES.md Step 5 Variable Mapping: "`trials.hit/miss/false_alarm/correct_reject` | output[4]: trial_outcome | Static per-trial categorical | 4 classes." Step 3 records the task structure from the whitepaper ("Hit, Miss, False Alarm, or Correct Rejection, with responses detected in a 150-750 ms window after stimulus change"), and Step 5's planned sanity check "Trial outcome distribution: Go trials → hit or miss; Catch trials → FA or CR" was confirmed in Step 12.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The integer code is broadcast to a constant row spanning all of the trial's time bins, so the nominally static variable is stored in the same `(5, n_timepoints)` time-varying layout as the other four outputs. `output_values[4] = ['hit', 'miss', 'false_alarm', 'correct_reject']`. Realized distribution: hit 0.302, miss 0.572, false_alarm 0.017, correct_reject 0.108.

ii.
```python
# 5. Trial outcome (static)
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
...
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)  # (4, n_tp)
# Append trial outcome as constant across time
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)  # (5, n_tp)
```

```python
output_names = ['image_identity', 'image_change', 'running_speed', 'pupil_diameter', 'trial_outcome']
output_values = [
    global_image_names,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
    [f'bin_{i}' for i in range(5)],
    ['hit', 'miss', 'false_alarm', 'correct_reject'],
]
```

iii. Broadcasting is driven by the format requirement that all five outputs share one `(n_output, n_timepoints)` array per trial; the task spec itself marks trial outcome as "Static per-trial", and the AI's in-code comment flags the mixed static/time-varying handling. Step 10 Check 5 verifies "Trial outcome constant within trials" as one of its edge-case checks. Note the dead local `trial_outcome = np.array([...])`, which is computed and never used.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The handling is layered:
- **Unreadable / malformed file**: the entire `load_experiment_data` body is wrapped in `try/except Exception`, which prints an error and returns `None`; the experiment is skipped.
- **No valid ROIs**: experiment skipped with a warning.
- **No stimulus-presentations table**: experiment skipped with a warning.
- **Missing eye tracking** (3 experiments): caught by `except (KeyError, Exception)`, pupil set to `None`, per-trial pupil filled with NaN, and NaN is mapped to bin 0 by `digitize_to_bins` — so those sessions carry a constant pupil output.
- **Blinks and NaN gaps in pupil**: censored to NaN then linearly interpolated; an all-NaN trace becomes all zeros.
- **Byte-string columns**: `image_name`, `initial_image_name`, `change_image_name` are decoded if stored as bytes; `omitted` is cast from float to bool when needed.
- **Interpolation artifacts**: resampled events are clipped to `>= 0`; `np.interp` clamps at the stream edges instead of extrapolating, so no out-of-range NaNs are created for running/pupil.
- **Degenerate percentile edges**: non-increasing bin edges are nudged apart by 1e-10.
- **Short / outcomeless trials**: skipped; **sessions with < 2 usable trials**: skipped.
- **All-zero neural trials** (2,602 of 51,992): *not* treated as an error — left in the dataset.

ii.
```python
    except Exception as e:
        print(f"  ERROR loading {experiment_id}: {e}")
        return None
```

```python
if n_valid == 0:
    print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping")
    return None
...
if stim_key is None:
    print(f"  WARNING: No stimulus presentations found in {experiment_id}, skipping")
    return None
...
except (KeyError, Exception):
    print(f"  WARNING: No pupil tracking data in {experiment_id}")
```

```python
if isinstance(trials['initial_image_name'][0], bytes):
    trials['initial_image_name'] = np.array([x.decode() for x in trials['initial_image_name']])
    trials['change_image_name'] = np.array([x.decode() for x in trials['change_image_name']])
...
if stim_data['omitted'].dtype == float:
    stim_data['omitted'] = stim_data['omitted'].astype(bool)
```

```python
def interpolate_nans(arr):
    nans = np.isnan(arr)
    if not nans.any():
        return arr.copy()
    if nans.all():
        return np.zeros_like(arr)
    ...
binned[np.isnan(values)] = 0
events_resampled = np.maximum(events_resampled, 0)
for i in range(1, len(edges)):
    if edges[i] <= edges[i-1]:
        edges[i] = edges[i-1] + 1e-10
```

iii. CONVERSION_NOTES.md Step 4 anticipated the pupil issue ("Pupil | area with NaNs during blinks | ~9% blinks | Need to handle NaN/blinks") and Step 5 Key Decision 9 set the policy. Step 10 Check 1 addresses the all-zero-trial warnings and argues they are not a defect: "This is NOT a bug: calcium events are sparse (~0.25% of timepoints nonzero). Sessions with few neurons (e.g., 4-6) will have many trials with no detected events." Step 10 Check 5 enumerates eight edge-case checks that all passed (no NaN in neural data, all events non-negative, all output values in range, outcome constant within trials, valid region/subject indices, ≥2 trials per session).

## 9-a. What are the most time-consuming steps of the code?

i. The dominant cost is NWB file I/O, and it is paid three times per file because of the 3-pass design: Pass 1 opens all 202 files to read the `image_name` column, Pass 2 re-opens all 202 to read the full running and pupil arrays, Pass 3 re-opens all 202 and reads everything including the full `(n_timepoints, n_cells)` event matrix. Within Pass 3 the second cost is the per-neuron resampling loop in `interpolate_to_regular_grid`, which runs one `np.interp` per neuron over ~140k source samples; the log shows this splitting roughly evenly with loading (~0.2–0.3 s each per small experiment). Finally, pickling the 8.3 GB result takes ~9 s. Total wall clock was 358 s (6 min). Timing is instrumented and printed, but only for Pass 3.

ii.
```python
t0 = time.time()
print(f"\n[{idx+1}/{len(exp_table)}] Processing experiment {eid}...", flush=True)
raw_data = load_experiment_data(nwb_path, eid)
...
t_load = time.time() - t0
...
result = process_single_experiment(raw_data, exp_meta)
...
t_process = time.time() - t0 - t_load
...
t_total = time.time() - t0
print(f"  Loaded in {t_load:.1f}s, processed in {t_process:.1f}s, total {t_total:.1f}s")
```

iii. CONVERSION_NOTES.md Step 6 "Code inefficiencies identified": "Sequential processing of experiments (could parallelize); Multiple passes over NWB files." Step 7's runtime table estimated ~0.25 s load + ~0.25 s process per session ⇒ "~2-3 minutes (well under 15 min limit)", and the actual 6 min run was within the instructions' 15-minute budget, so the AI chose not to optimize further.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python-level loops are vectorizable:
- `interpolate_to_regular_grid` loops over neurons, calling `np.interp` once per column. `scipy.interpolate.interp1d(..., axis=0)` or a shared-index gather/lerp would do all neurons at once — this is the single largest avoidable cost.
- `get_image_at_timepoints` already computes `insert_idx` with a vectorized `searchsorted`, then throws that away by looping over every timepoint in Python to do a dict lookup. Precomputing a name→code integer array and doing `codes[insert_idx]` makes the whole function branch-free.
- `get_image_change_at_timepoints` loops over *every change flash in the session* for *every trial*, building a full boolean mask each time — O(n_trials × n_changes). A single `searchsorted` of the trial timestamps into the change-onset array would be O(n_tp log n_changes).
- The per-trial `img_id_global = np.array([local_to_global.get(v, 0) for v in ...])` remap is a Python list comprehension over all timepoints; it should be a lookup-table index. `global_image_names.index(name)` inside the mapping loop is also a linear scan.

Additionally, the outer experiment loop is fully independent across files and the unused `ProcessPoolExecutor` import shows the AI considered parallelizing it but never did.

ii.
```python
from concurrent.futures import ProcessPoolExecutor, as_completed   # imported, never used
```

```python
        result = np.zeros((len(target_timestamps), data.shape[1]), dtype=np.float32)
        for i in range(data.shape[1]):
            result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
```

```python
    insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
    for i in range(n_tp):
        if insert_idx[i] >= 0:
            img_name = stim_names[insert_idx[i]]
            image_idx[i] = name_to_idx.get(img_name, 0)
        else:
            image_idx[i] = 0
```

```python
    for cs, ce in zip(change_starts, change_stops):
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1
```

```python
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. CONVERSION_NOTES.md Step 6 "Code speedups added" claims "Vectorized interpolation using np.interp" and "Efficient searchsorted for image identity assignment". Both are true only partially — `np.interp` is vectorized across target timepoints but still called once per neuron, and the `searchsorted` result is consumed by a per-timepoint Python loop. The AI's stated position is that the total runtime (6 min) was comfortably under the 15-minute threshold in the instructions, so no further optimization was warranted.

## 9-c. What processing does the code repeat multiple times?

i. Repeated work, in order of cost:
- **Every NWB file is opened and read three times.** Pass 1 reads the stimulus `image_name` column; Pass 2 reads the full running-speed array and the full pupil area + blink arrays; Pass 3 reads all of those again (plus events, trials, timestamps). The Pass 2 quantities are exactly the ones Pass 3 re-reads.
- **The `f['intervals']` stimulus-key search is re-run** in Pass 1 and again in Pass 3 for each file, with the same filtering logic duplicated inline in `convert_all` and in `load_experiment_data`.
- **Per-trial recomputation of session-constant quantities**: `get_image_at_timepoints` rebuilds `non_omitted`, `stim_starts`, `stim_names`, and `name_to_idx` on every trial; `get_image_change_at_timepoints` rebuilds `change_mask`, `change_starts`, `change_stops` on every trial. With ~250 trials per experiment these session-level arrays are constructed ~250 times each.
- `global_image_names.index(name)` is a linear search repeated per image per experiment.

ii.
```python
    # First pass: determine global image names
    for _, row in exp_table.iterrows():
        with h5py.File(nwb_path, 'r') as f:
            for k in f['intervals'].keys():
                if k != 'trials' and 'spontaneous' not in k.lower() and 'movie' not in k.lower():
                    img_names = f['intervals'][k]['image_name'][()]
    ...
    # Pass 2
    for idx, (_, row) in enumerate(exp_table.iterrows()):
        with h5py.File(nwb_path, 'r') as f:
            running = f['processing']['running']['speed']['data'][()]
            pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()].astype(float)
    ...
    # Pass 3
    for idx, (_, row) in enumerate(exp_table.iterrows()):
        raw_data = load_experiment_data(nwb_path, eid)   # reads all of the above again
```

```python
def get_image_at_timepoints(timepoints, stim_data, all_image_names):
    non_omitted = ~stim_data['omitted']            # recomputed every trial
    stim_starts = stim_data['start_time'][non_omitted]
    stim_names = stim_data['image_name'][non_omitted]
    name_to_idx = {name: i for i, name in enumerate(all_image_names)}
```

iii. CONVERSION_NOTES.md Step 6 lists "Multiple passes over NWB files" under "Code inefficiencies identified" but no fix was applied, and the per-trial recomputation of session-level arrays is not mentioned anywhere. Step 10's "Issues Found and Resolved" reports "No issues found requiring fixes." The implicit justification is again the 6-minute runtime.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Work performed whose product is never consumed:
- **Whole-session neural resampling.** `interpolate_to_regular_grid` resamples every neuron across the *entire* session, but only the `[start_time, stop_time)` windows of go/catch trials are kept. All aborted-trial and auto-rewarded-trial periods, plus pre/post-task time, are resampled and discarded.
- **3× upsampling of multiscope data.** The 34 MESO.1 experiments are interpolated from ~11 Hz to 30 Hz, so roughly two thirds of their stored neural samples are interpolated filler carrying no new information — a major contributor to the 8.3 GB output file.
- **Dead values.** `change_stops` is extracted and zipped but `ce` is never referenced; `cell_specimen_ids` is subset and returned but never used; `valid_roi` is returned in the dict and never used downstream; the trials-table `is_change` column is loaded but never read (the stimulus-table one is used instead); `n_trials` is computed and discarded; `trial_outcome = np.array([...])` is built and then ignored in favour of `outcome_broadcast`.
- **Unused imports**: `ProcessPoolExecutor`, `as_completed`, `sys`.
- **Redundant Pass 2 reads**: full-resolution running and pupil arrays are loaded for percentile edges and then thrown away, only to be re-loaded in Pass 3.

ii.
```python
    change_starts = stim_data['start_time'][change_mask]
    change_stops = stim_data['stop_time'][change_mask]
    for cs, ce in zip(change_starts, change_stops):   # `ce` unused
```

```python
            n_trials = len(trials_grp['start_time'][()])     # never used
            'is_change': trials_grp['is_change'][()].astype(bool),   # never used
...
            return {
                'valid_roi': valid_roi,                      # never used by caller
                'cell_specimen_ids': cell_specimen_ids[valid_roi],   # never used by caller
```

```python
            # 5. Trial outcome (static)
            trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)   # never used
            ...
            outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
```

```python
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)  # whole session
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)   # only trial windows kept
```

iii. None of this is documented. CONVERSION_NOTES.md Step 6 lists only "Sequential processing of experiments" and "Multiple passes over NWB files" as inefficiencies, and Step 10 concludes "No issues found requiring fixes." The AI's general position throughout Steps 6–12 is that because the conversion finished in 6 minutes — well under the instructions' 15-minute threshold — no further efficiency work was needed.
