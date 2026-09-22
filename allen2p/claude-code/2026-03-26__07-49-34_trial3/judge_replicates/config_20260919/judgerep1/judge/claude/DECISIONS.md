# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did **not** use the AllenSDK `VisualBehaviorOphysProjectCache`. Instead it reads the NWB (HDF5) files on disk directly with `h5py`, and uses the project metadata CSV `project_metadata/ophys_experiment_table.csv` as the index of what exists. The experiment list is built by intersecting (a) the experiment ids for which a `behavior_ophys_experiment_<id>.nwb` file is actually present in `/app/data/.../behavior_ophys_experiments/`, with (b) a hard-coded list of *active* session types (`OPHYS_1_images_A`, `OPHYS_3_images_A`, `OPHYS_4_images_B`, `OPHYS_6_images_B`). Passive sessions (`OPHYS_2_*_passive`, `OPHYS_5_*_passive`) are dropped. No `project_code` filter is applied, so the selection contains both `VisualBehavior` (168 experiments) and `VisualBehaviorMultiscope` (34 experiments) data. Result: 202 experiments / 38 mice / 51,992 trials / 29,444 neurons.

Everything needed from a file (cell table, ophys timestamps, event traces, trials, stimulus presentations, running speed, eye tracking) is pulled inside a single `h5py.File` context in `load_experiment_data()`. The whole file set is traversed three times: Pass 1 for the global image-name vocabulary, Pass 2 for the running/pupil percentile statistics, Pass 3 for the actual conversion.

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
    exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
    nwb_files = list(NWB_DIR.glob('*.nwb'))
    nwb_ids = set()
    for f in nwb_files:
        try:
            eid = int(f.stem.split('_')[-1])
            nwb_ids.add(eid)
        except ValueError:
            pass
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

iii. From CONVERSION_NOTES Step 1/2/4: dF/F and events are **pre-computed** in the NWB, so the SDK adds nothing that the raw file does not already contain; reading with `h5py` is faster and avoids SDK overhead. Restricting to files actually on disk is the AI's handling of the data-limited subset ("284 NWB files on disk … subset of full 1,936 experiments in metadata"). Passive sessions were excluded because "OPHYS_2, OPHYS_5 are passive viewing (no lick spout, satiated mice). No meaningful trial outcomes." The AI decided the task phrase "Visual Behavior task" meant the active behaving sessions rather than the `VisualBehavior` project code, and never explicitly considered restricting by `project_code`.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the selected experiment table, cast to `str` and sorted. `subject_idx` for each output session is the index of that experiment's `mouse_id` in this list. 38 subjects result (37 from `VisualBehavior` + 1 Multiscope mouse, 457841).

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

iii. CONVERSION_NOTES Step 5, decision 12: "Subject IDs: Use mouse_id from experiment table." The AI cross-checked the count against the papers (82 mice in the full release vs 38 on disk) and recorded the difference as expected for a partial download.

## 1-c. How are the data split into sessions?

i. **One NWB experiment (= one imaging plane) is one output "session."** The AI does not group experiments by `ophys_session_id`. For the 168 single-plane `VisualBehavior` experiments this is equivalent to one session per recording. For the 34 `VisualBehaviorMultiscope` experiments that were also included, the 3–7 simultaneously-recorded planes of a single physical session each become their own output session: 6 real Multiscope sessions are emitted as 34 sessions, all sharing the identical trial structure, running speed, pupil trace and behavioural outcomes. That single Multiscope mouse therefore contributes 34 of the 202 sessions (visible in the verification log: "Subject 457841: 34 sessions", versus 3–9 for every other mouse).

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
    all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
    region_idx = np.full(result['n_cells'], region_to_idx[row['targeted_structure']], dtype=np.int64)
    all_brain_region_idx.append(region_idx)
```

iii. CONVERSION_NOTES Step 5, decision 13: "Each NWB experiment = one 'session' in output format (one imaging plane with its own neurons)." Trajectory step 36: "Each NWB file = one experiment = one imaging plane. This should be one 'session' in our output, since each has its own set of neurons." The rationale is that the target format needs a fixed neuron set per session; the AI did not discuss the consequence of duplicating one animal's behaviour across planes.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. Each kept trial spans `start_time` → `stop_time` (half-open), i.e. the full trial window including the pre-change flashes and the post-change response window, so trials are variable length (mean 254 bins ≈ 8.5 s at 30 Hz, range 210–377). Trial membership is determined on the resampled 30 Hz grid by a boolean mask.

ii.
```python
trials = {
    'start_time': trials_grp['start_time'][()],
    'stop_time': trials_grp['stop_time'][()],
    'change_time': trials_grp['change_time'][()],
    'go': trials_grp['go'][()].astype(bool),
    'catch': trials_grp['catch'][()].astype(bool),
    'aborted': trials_grp['aborted'][()].astype(bool),
    'auto_rewarded': trials_grp['auto_rewarded'][()].astype(bool),
    ...
}
...
for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop = trials['stop_time'][trial_idx]
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
```

iii. CONVERSION_NOTES Step 5, decision 5: "Trial window: Use trial start_time to stop_time from trials table. Variable length across trials." The AI initially considered a fixed window centred on `change_time` (trajectory step 36) but settled on the full native trial so the time-varying outputs (image identity, image change) would have both pre- and post-change structure.

## 1-e. How are trials filtered based on quality controls?

i. Four filters:
1. Trial-type filter: `(go | catch) & ~aborted & ~auto_rewarded`. (I verified on experiment 1007107386 that this selects exactly the same 365 trials as the reference's `~aborted & ~auto_rewarded & change_time.notna()`.)
2. Trials whose 30 Hz window contains fewer than 3 bins are dropped.
3. Trials with none of `hit/miss/false_alarm/correct_reject` set are dropped.
4. Experiments with fewer than 2 valid trials (before or after processing) are dropped entirely; likewise experiments with zero `valid_roi` cells.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
if len(valid_indices) < 2:
    print(f"  WARNING: Only {len(valid_indices)} valid trials ... skipping")
    return None
...
    if len(trial_time_indices) < 3:
        continue
...
    if trials['hit'][trial_idx]: outcome = 0
    elif trials['miss'][trial_idx]: outcome = 1
    elif trials['false_alarm'][trial_idx]: outcome = 2
    elif trials['correct_reject'][trial_idx]: outcome = 3
    else:
        continue  # Unknown outcome, skip
...
if len(neural_trials) < 2:
    print(f"  WARNING: Only {len(neural_trials)} processed trials ... skipping")
    return None
```

iii. CONVERSION_NOTES Step 5, decision 3: "Exclude aborted and auto-rewarded trials: Per task instructions." Step 3 notes the reasons from the whitepaper (aborted = premature lick before the change; auto-rewarded = 5 free rewards at session start and after 10 consecutive misses). The ≥2-trial rule follows the target-format requirement "There needs to be at least two trials within each session." No further quality gate (e.g. d-prime, engagement) was applied; the AI recorded the whitepaper's 10 session-QC criteria but noted they are already applied upstream by Allen.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/event_detection/data` — the pre-computed deconvolved calcium events (FastLZeroSpikeInference), **not** dF/F. Columns are subset to `valid_roi == True` from `processing/ophys/image_segmentation/cell_specimen_table`. The time base is `processing/ophys/dff/traces/timestamps`.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][()].astype(bool)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][()]
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]  # (n_timepoints, n_valid_cells)
```

iii. Trajectory step 36 records the AI explicitly weighing dF/F against events and choosing events: "the reference paper explicitly uses deconvolved calcium events detected via FastLZeroSpikeInference, which are much closer to spike-like activity than raw dF/F. The instructions emphasize matching the reference processing, so I should use events instead of dF/F." It also decided to use the *raw* events rather than the SDK's causally half-Gaussian–filtered events: "The reference paper uses raw calcium events detected via Fast LZeroSpikeInference, not the filtered versions from the SDK." CONVERSION_NOTES Step 5 mapping table: "`event_detection/data` (filtered by valid_roi) … Paper uses events, not dF/F."

## 2-b. How is the `neural` data processed?

i. Three operations, no smoothing and no normalisation:
1. Column subset to valid ROIs.
2. Linear interpolation of every neuron's event trace from the native ophys timestamps onto a uniform 30 Hz grid `np.arange(ophys_ts[0], ophys_ts[-1], 1/30)` spanning the whole session. Each neuron is interpolated in a Python `for` loop over columns.
3. Clip to non-negative, then slice out the per-trial windows and transpose to `(n_neurons, n_timepoints)`, stored as `float32`.

No causal half-Gaussian filtering is applied, so the stored traces are the extremely sparse raw L0 event amplitudes (the AI measured ~0.25 % non-zero samples). This produced 2,602 trials (5 % of 51,992) whose neural matrix is entirely zero.

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
t_start = ophys_ts[0]; t_end = ophys_ts[-1]; dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. CONVERSION_NOTES Step 5, decision 4: "Resample to 30 Hz: Paper interpolates to 30 Hz. Needed for consistent time bins across Scientifica (31 Hz) and Multiscope (11 Hz). time_bin_size = 33.33 ms", quoting the paper's "linearly interpolating onto a consistent set of 30hz timestamps." Clipping to ≥0 is justified as "events should be >= 0". The all-zero trials were explicitly examined in Step 10/12 and dismissed: "This is NOT a bug: calcium events are sparse (~0.25% of timepoints nonzero). Sessions with few neurons (e.g., 4-6) will have many trials with no detected events."

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single neuron-level filter: keep only ROIs with `valid_roi == True` in the cell specimen table. Experiments with zero valid ROIs are skipped. No activity-based, SNR-based or event-rate-based neuron filtering, and no manual exclusion list. 29,444 neurons survive across the 202 sessions (4–666 per session).

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
cell_specimen_ids = cell_table['cell_specimen_id'][()]
n_valid = valid_roi.sum()
if n_valid == 0:
    print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping")
    return None
events_valid = events_data[:, valid_roi]
```

iii. CONVERSION_NOTES Step 1: "Cell filtering: Only automatic filter is `valid_roi` boolean (SVM binary classifier output). `exclude_invalid_rois=True` by default." Step 5, decision 10: "valid_roi filtering: Only include neurons with valid_roi=True." Because the AI bypassed the SDK it had to reproduce the SDK's default ROI exclusion by hand, and it identified this explicitly in Step 4 ("Need to filter by valid_roi when loading NWB").

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**. All streams (events, running, pupil) are first resampled onto one shared session-wide 30 Hz grid derived from the ophys timestamps, and a trial is the set of grid bins with `start_time <= t < stop_time`. Because every stream lives on the same grid, index `k` of the neural matrix and index `k` of each output row refer to the same wall-clock bin. Metadata records `temporal_alignment_event = 'Trial start time (first stimulus onset of trial)'`, `off_start = 0.0`, `off_end = None` (variable length).

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
trial_ts = regular_ts[trial_time_indices]
n_tp = len(trial_ts)
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
running_trial = running_resampled[trial_time_indices]
pupil_trial   = pupil_resampled[trial_time_indices]
image_idx     = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. CONVERSION_NOTES Step 5, decision 6: "Alignment event: Trial start time (stimulus onset). off_start=0, off_end=None (variable)." Step 3 records that all clocks are hardware-synchronised by the NI PCI-6612 board at 100 kHz, so simple interpolation onto a common time base is legitimate. Step 10 Check 5 verified neural/output length agreement at trial boundaries, and Check 2 spot-checked image identity at (session 0, trial 5, timepoint 10) against the raw NWB.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 33.33 ms (30 Hz) uniformly for every trial and session; `metadata['time_bin_size'] = 1000/30`. Yes — rebinning is applied: every stream is linearly interpolated onto a regular 30 Hz grid. For the Scientifica single-plane rigs this is a very slight downsampling (30.95 Hz → 30 Hz); for the 34 Multiscope plane-experiments it is a ~2.7× **up**sampling (11 Hz → 30 Hz), which adds no information and inflates storage (final pickle 8.3 GB).

ii.
```python
TARGET_RATE_HZ = 30.0  # Paper: "linearly interpolating onto a consistent set of 30hz timestamps"
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms
...
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
...
'time_bin_size': TIME_BIN_MS,
'target_rate_hz': TARGET_RATE_HZ,
```

iii. Two reasons given in CONVERSION_NOTES Step 4/5: the paper's own analysis pipeline interpolates to 30 Hz, and the target format requires "Time bins … the same size for all trials and sessions", which cannot hold across the 31 Hz Scientifica and 11 Hz Multiscope rigs that the AI chose to pool without resampling.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The **stimulus presentations** table (`intervals/<stimulus table>`), specifically `image_name`, `start_time` and `omitted` — not the trials table's `initial_image_name` / `change_image_name`. The stimulus interval group is located by name, skipping `trials`, `spontaneous*` and `*movie*`.

ii.
```python
stim_data = {
    'start_time': stim['start_time'][()],
    'stop_time': stim['stop_time'][()],
    'image_name': stim['image_name'][()],
    'is_change': stim['is_change'][()].astype(bool),
    'omitted': stim['omitted'][()],
}
```

iii. CONVERSION_NOTES Step 5 mapping table: "`stimulus_presentations.image_name` → output[0]: image_identity … Map to categorical int, time-varying per ophys frame | 8 natural images." Using the flash table rather than the trial table means the label is read straight off what was actually on the monitor at each moment, and it degrades gracefully across omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Per-experiment, the sorted set of non-`omitted` image names is built. For each 30 Hz bin inside a trial, `np.searchsorted(..., side='right') - 1` finds the most recent non-omitted flash onset, and that flash's image name is the label; during the 500 ms grey period and during omitted flashes the previous image is carried forward. Bins before the first flash fall back to local index 0. The per-experiment index is then remapped to a **global** vocabulary of 16 image names (8 from set A + 8 from set B) collected in Pass 1 over all files, so codes are consistent across sessions. The verification log shows a near-uniform distribution (each image 5.8–6.7 %).

ii.
```python
def get_image_at_timepoints(timepoints, stim_data, all_image_names):
    n_tp = len(timepoints)
    image_idx = np.zeros(n_tp, dtype=np.int64)
    non_omitted = ~stim_data['omitted']
    stim_starts = stim_data['start_time'][non_omitted]
    stim_names = stim_data['image_name'][non_omitted]
    name_to_idx = {name: i for i, name in enumerate(all_image_names)}
    insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
    for i in range(n_tp):
        if insert_idx[i] >= 0:
            image_idx[i] = name_to_idx.get(stim_names[insert_idx[i]], 0)
        else:
            image_idx[i] = 0  # Before first stimulus
    return image_idx
```

```python
local_to_global = {}
for local_idx, name in enumerate(result['all_image_names']):
    local_to_global[local_idx] = global_image_names.index(name) if name in global_image_names else 0
...
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5, decisions 7 and 8: "Image identity during gray screen: Use the identity of the image that was just shown (last presented image)" and "Image identity for omitted flashes: Continue with previous image identity." A global mapping is used so that the categorical code means the same thing in every session, and the AI checked in Step 9 that 16 classes appear (two image sets).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated at exactly the same 30 Hz bin centres (`trial_ts = regular_ts[trial_time_indices]`) used to slice the neural matrix, so both have identical length `n_tp` and identical time base; the image row is then stacked into the `(5, n_tp)` output array.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
...
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)  # (4, n_tp)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)  # (5, n_tp)
```

iii. Sharing one resampled grid for all streams is the AI's stated mechanism for guaranteeing alignment (Step 5, decision 6; Step 10 Check 5 "Neural/output length alignment at trial boundaries"). Step 10 Check 2 also re-derived the image label for (session 0, trial 5, timepoint 10) from the raw NWB and matched `im063`.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The stimulus presentations table's `is_change` flag combined with `omitted` and `start_time`. The trials table's `change_time` / `is_change` columns are read into memory but never used for this output.

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. CONVERSION_NOTES Step 5 mapping table: "`trials.is_change` + stimulus timing → output[1]: image_change | Binary 1 at change timepoint, 0 otherwise, time-varying | 1 for one 750ms window at change". Because `is_change` in the flash table is only true for genuine image changes, catch (sham-change) trials automatically get an all-zero change signal — the AI verified this in Step 12: "Catch trials: 0/6,515 have image_change signal (correct: catch = sham change)."

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each change flash in the session, all trial bins in the half-open window `[change_start, change_start + 0.75 s)` are set to 1; everything else is 0. 0.75 s is one full image interval (250 ms image + 500 ms grey). The loop runs over every change flash of the whole session for every trial. The resulting global rate is 7.7 % of bins.

ii.
```python
def get_image_change_at_timepoints(timepoints, stim_data):
    n_tp = len(timepoints)
    change_signal = np.zeros(n_tp, dtype=np.int64)
    change_mask = stim_data['is_change'] & ~stim_data['omitted']
    change_starts = stim_data['start_time'][change_mask]
    change_stops = stim_data['stop_time'][change_mask]
    for cs, ce in zip(change_starts, change_stops):
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1
    return change_signal
```

iii. CONVERSION_NOTES Step 3 records "Image presentation | 250 ms stimulus + 500 ms gray = 750 ms", and Step 6: "Image change signal: 1 during 750ms window starting at change onset." Step 9/10 sanity-checked the resulting rate: "Image change fraction | ~1/13 flashes | 7.7% | Yes (~1/13=7.7%)."

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is natively binary — no thresholding of a continuous quantity is involved. Values are `{0, 1}` with `output_values` `['no_change', 'change']`; verification reports `no_change 0.923 / change 0.077`.

ii.
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
...
img_change = trial_data_out['image_change'].astype(np.int64)
...
['no_change', 'change'],  # image change values
```

iii. The instructions define the variable as binary ("Have value of 1 right after a change in image identity, otherwise 0"), so the only design freedom the AI exercised was the width of the "right after" window, which it set to one 750 ms image interval to match the stimulus cycle described in the whitepaper.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Computed on the same `trial_ts` bins as the neural matrix and stacked as row 1 of the `(5, n_tp)` output array, so it is bin-for-bin aligned. Note that because the change list is session-wide rather than trial-specific, a change flash belonging to a neighbouring trial would also be marked if it fell inside this trial's `[start_time, stop_time)` window.

ii.
```python
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
...
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)
```

iii. Same rationale as 3-c: one shared 30 Hz grid for every stream. Step 12 additionally verified consistency with the trial labels ("Go trials: 0/45,477 missing image_change signal").

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` and its `timestamps` (~60 Hz), i.e. the filtered running speed in cm/s that the SDK exposes as `dataset.running_speed`.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. CONVERSION_NOTES Step 5 mapping table: "`running/speed/data` → output[2]: running_speed | Interpolate to 30 Hz, discretize into 5 percentile bins, time-varying." Step 1 identified `RunningSpeed.from_nwb()` as the corresponding SDK accessor.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation from the ~60 Hz encoder time base onto the session's 30 Hz grid with `np.interp` (so effectively a decimation with interpolation, no anti-alias filtering), then per-bin discretisation with globally-computed percentile edges.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. Step 5, decision 4 (30 Hz common grid) plus the task requirement that the output be discrete. The AI noted in Step 3 that all acquisition clocks are hardware-synced, so direct interpolation between streams is valid.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins. The edges are the 0/20/40/60/80/100th percentiles of **all raw 60 Hz running samples pooled over every selected experiment** (Pass 2), computed *before* and independently of trial segmentation and resampling. Ties are broken by nudging non-increasing edges by 1e-10; values are digitised with `np.digitize` on the interior edges and clipped into `[0, 4]`; NaN maps to bin 0. Edges came out as `[-24.13, -0.0041, 0.336, 14.18, 33.13, 99.92]` cm/s, and the realised per-bin occupancy in the converted data is 18.4–20.9 % rather than exactly 20 %.

ii.
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-10
    return edges

def digitize_to_bins(values, bin_edges):
    n_bins = len(bin_edges) - 1
    binned = np.digitize(values, bin_edges[1:-1])
    binned = np.clip(binned, 0, n_bins - 1)
    binned[np.isnan(values)] = 0
    return binned.astype(np.int64)
```

```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        running = f['processing']['running']['speed']['data'][()]
        all_running_values.append(running.astype(np.float32))
running_bin_edges = compute_percentile_bins(np.concatenate(all_running_values), n_bins=5)
```

iii. The instruction is "discretized into five equal percentile bins". The AI computed the edges globally so that "bin 2" means the same speed range in every session (Step 6: "compute global percentile bins for running/pupil"), and it accepted the resulting slight imbalance, documenting it in Step 7: "Running speed bins slightly unequal within sessions (expected, bins computed globally)" and in Step 9: "Running speed bins | 5 equal percentile | 18-21% each | Yes (equal percentile)".

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated onto the same session-wide 30 Hz grid before trial segmentation and then indexed with the identical `trial_time_indices`, so it is bin-for-bin aligned with the neural matrix; it is row 2 of the `(5, n_tp)` output.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
neural_trial  = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. Same single-grid argument as 2-d/3-c; hardware clock synchronisation (Step 3, item 4) is cited as the justification for interpolating across streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (pupil **area**, px²) with its `timestamps` (~30 Hz), plus `acquisition/EyeTracking/likely_blink/data` for blink rejection. The AI used area as a stand-in for diameter; the `width`/`height` datasets available in the same group were not used. Three of 202 experiments have no eye-tracking group at all.

ii.
```python
try:
    pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
    pupil_area = pupil_tracking['area'][()]
    pupil_ts = pupil_tracking['timestamps'][()]
    likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
except (KeyError, Exception):
    print(f"  WARNING: No pupil tracking data in {experiment_id}")
```

iii. CONVERSION_NOTES Step 5 mapping table: "`EyeTracking/pupil_tracking/area` → output[3]: pupil_diameter … Use pupil area as proxy for diameter." Step 4 recorded that the pupil signal contains "area with NaNs during blinks | ~9% blinks", motivating the `likely_blink` handling.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) Samples flagged `likely_blink` are set to NaN; (2) remaining NaN runs are filled by linear interpolation in sample index space (`interpolate_nans`) — i.e. the pupil trace is bridged across blinks rather than left missing; (3) the bridged trace is linearly interpolated onto the 30 Hz grid; (4) percentile-binned with globally-computed edges. If the whole experiment lacks eye tracking, the trial pupil vector is filled with NaN.

ii.
```python
pupil_area = raw_data['pupil_area'].copy().astype(float)
if likely_blink is not None:
    pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
if pupil_resampled is not None:
    pupil_trial = pupil_resampled[trial_time_indices]
else:
    pupil_trial = np.full(n_tp, np.nan)
```

```python
def interpolate_nans(arr):
    nans = np.isnan(arr)
    if not nans.any(): return arr.copy()
    if nans.all():     return np.zeros_like(arr)
    result = arr.copy()
    x = np.arange(len(arr))
    result[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return result
```

iii. CONVERSION_NOTES Step 5, decision 9: "Pupil NaN handling: During blinks (likely_blink=True or NaN), linearly interpolate. Compute percentile bins from non-blink data." The AI's stated aim is to keep a continuous pupil trace without letting blink artefacts contaminate either the signal or the bin edges.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five equal-percentile bins with the same `compute_percentile_bins` / `digitize_to_bins` machinery as running speed. Edges are computed in Pass 2 from the pooled **non-blink raw** pupil-area samples of all experiments: `[125.6, 4374.1, 5527.6, 6747.1, 8598.7, 323783.3]` px². NaN → bin 0. Realised occupancy 18.5–22.8 %.

ii.
```python
pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()].astype(float)
blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
pupil_area[blink] = np.nan
valid_pupil = pupil_area[~np.isnan(pupil_area)]
if len(valid_pupil) > 0:
    all_pupil_values.append(valid_pupil.astype(np.float32))
...
pupil_bin_edges = compute_percentile_bins(np.concatenate(all_pupil_values), n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. Same reasoning as 5-c — global edges for cross-session comparability; blinks excluded from the percentile computation so that artefactual values do not distort the edges (Step 5, decision 9). Step 9 records "Pupil diameter bins | 5 equal percentile | 19-23% each | Yes (roughly equal)".

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Resampled onto the shared 30 Hz grid before trial segmentation and indexed with the same `trial_time_indices`; it is row 3 of the `(5, n_tp)` output array.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. Identical single-grid / hardware-sync justification as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually-exclusive boolean columns of the NWB trials table: `hit`, `miss`, `false_alarm`, `correct_reject`.

ii.
```python
'hit': trials_grp['hit'][()].astype(bool),
'miss': trials_grp['miss'][()].astype(bool),
'false_alarm': trials_grp['false_alarm'][()].astype(bool),
'correct_reject': trials_grp['correct_reject'][()].astype(bool),
```

iii. CONVERSION_NOTES Step 5 mapping table: "`trials.hit/miss/false_alarm/correct_reject` → output[4]: trial_outcome | Static per-trial categorical | 4 classes." Step 4 confirmed on the data that "Each valid trial has exactly 1 outcome".

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. A fixed priority chain maps the booleans to codes 0=hit, 1=miss, 2=false_alarm, 3=correct_reject; a trial matching none of the four is dropped. Although the variable is conceptually static per trial, it is broadcast to a constant row of length `n_tp` so that the output array is a clean `(5, n_timepoints)`. Resulting distribution: hit 30.2 %, miss 57.2 %, false alarm 1.7 %, correct reject 10.8 %.

ii.
```python
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
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)  # (5, n_tp)
```

iii. The four flags are the SDK's canonical change-detection outcomes; broadcasting keeps every output on a single `(n_output, n_timepoints)` array, which the target format allows and which keeps the decoder's output tensor rectangular. Step 10 Check 5 verified "Trial outcome constant within trials". Step 9 compared the implied Go/Catch split (hit+miss ≈ 87.4 %, FA+CR ≈ 12.5 %) with the whitepaper's 87.5 %/12.5 %.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is defensive at several levels:
- Whole-file load wrapped in `try/except`; any exception prints an error and the experiment is skipped.
- Experiment skipped if no valid ROIs, if no stimulus presentation table can be identified, if fewer than 2 valid trials before processing, or fewer than 2 after.
- Missing eye tracking (3 of 202 experiments): a warning is printed, `pupil_resampled` stays `None`, and every trial's pupil vector becomes all-NaN, which `digitize_to_bins` then maps to **bin 0** — so those sessions carry a constant, fabricated pupil class rather than being excluded.
- Blink samples → NaN → linearly bridged (`interpolate_nans`); an all-NaN trace would become all zeros.
- Byte-string image names decoded; `omitted` coerced from float to bool if needed.
- Trials shorter than 3 bins, and trials with no outcome flag, are dropped.
- Interpolated events clipped to ≥0; percentile edges nudged to be strictly increasing when the data are degenerate.

ii.
```python
    except Exception as e:
        print(f"  ERROR loading {experiment_id}: {e}")
        return None
...
            except (KeyError, Exception):
                print(f"  WARNING: No pupil tracking data in {experiment_id}")
...
        else:
            pupil_trial = np.full(n_tp, np.nan)
...
    binned[np.isnan(values)] = 0
...
    if isinstance(trials['initial_image_name'][0], bytes):
        trials['initial_image_name'] = np.array([x.decode() for x in trials['initial_image_name']])
    if stim_data['omitted'].dtype == float:
        stim_data['omitted'] = stim_data['omitted'].astype(bool)
...
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-10
```

iii. CONVERSION_NOTES Step 5, decision 9 (pupil NaN) and Step 10 Check 5 ("There may be minor issues in the data that your code must handle") list the edge-case checks the AI ran: no NaN in neural data, all events non-negative, all output values in range, trial outcome constant within trials, valid brain-region and subject indices, all sessions ≥2 trials. The `try/except` wrappers are justified so that one corrupt file cannot abort a 6-minute full run.

## 9-a. What are the most time-consuming steps of the code?

i. Measured from `/app/conversion_full_out.txt`: the full run took 358 s. Pass 3 accounts for 312 s, split into 183 s of loading (59 % of the run) and 125 s of processing (40 %). Passes 1 and 2 (re-opening all 202 NWB files to collect image names and running/pupil samples) account for roughly the remaining 37 s, and pickling the 8.3 GB output takes 9 s. Within loading, the dominant cost is `events_data = f[...]['event_detection']['data'][()]` which materialises the entire `(n_frames, n_cells)` array — up to 666 neurons × ~140 k frames — before subsetting to valid ROIs. Within processing, the dominant cost is the per-neuron `np.interp` loop that resamples every neuron's full-session trace to 30 Hz. The largest single experiment took 5.1 s (3.3 s load + 1.7 s process).

ii.
```python
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
...
for i in range(data.shape[1]):
    result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
```

iii. CONVERSION_NOTES Step 7 estimated ~0.5 s/session and a ~2–3 min total ("well under 15 min limit"); the realised 6 min is within 1.5× of that so the AI did not re-optimise. Step 6 lists the known inefficiencies it chose to accept: "Sequential processing of experiments (could parallelize)" and "Multiple passes over NWB files."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python-level loops remain that NumPy could absorb:
1. `interpolate_to_regular_grid` interpolates one neuron column at a time; `scipy.interpolate.interp1d(..., axis=0)` would do all neurons in one call.
2. `get_image_at_timepoints` already computes `insert_idx` vectorially but then walks every timepoint in Python to look names up in a dict; a NumPy index array (`code_lut[insert_idx]`) would remove the loop entirely. This runs ~13 M times over the full dataset.
3. `get_image_change_at_timepoints` loops over every change flash of the whole session and builds a full-length boolean mask per flash, for every trial — O(n_trials × n_changes × n_tp). Two `searchsorted` calls on the trial's own change time would suffice.
4. Trial windowing builds `(regular_ts >= start) & (regular_ts < stop)` — a scan over the entire session grid (~135 k bins) for each of ~300 trials — where `np.searchsorted` is O(log T), which is exactly what the reference implementation uses.
5. `img_id_global = np.array([local_to_global.get(v, 0) for v in ...])` is a per-timepoint Python comprehension that a lookup-table gather would replace.

Additionally, `ProcessPoolExecutor` is imported but never used, so the per-experiment loop stays single-threaded despite being embarrassingly parallel.

ii.
```python
from concurrent.futures import ProcessPoolExecutor, as_completed   # imported, never used
...
    for i in range(data.shape[1]):
        result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
...
    for i in range(n_tp):
        if insert_idx[i] >= 0:
            image_idx[i] = name_to_idx.get(stim_names[insert_idx[i]], 0)
...
    for cs, ce in zip(change_starts, change_stops):
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1
...
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
...
    img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 6 claims "Code speedups added: Vectorized interpolation using np.interp; Efficient searchsorted for image identity assignment" — which is only partly accurate, since the interpolation is still a per-column loop and the searchsorted result is consumed by a Python loop. The AI's stated reason for not going further is that the run already fit comfortably inside the 15-minute budget (Step 7).

## 9-c. What processing does the code repeat multiple times?

i. Every NWB file is opened and read three times:
- Pass 1 reads the stimulus `image_name` array from all 202 files purely to build the 16-name vocabulary.
- Pass 2 re-opens all 202 files and reads the full running-speed array and the full pupil-area + blink arrays to compute percentile edges.
- Pass 3 re-opens all 202 files and re-reads running speed, pupil area, blink flags and image names (again) along with everything else.

So the running-speed and pupil arrays are read from disk twice and the image-name array twice; the whole-session 30 Hz interpolation of running and pupil is also recomputed in Pass 3. The reference implementation loads each session exactly once and derives the bin edges and image vocabulary from the already-extracted trial data held in memory.

ii.
```python
# Pass 1
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        img_names = f['intervals'][k]['image_name'][()]
        all_image_names_set.update([n for n in img_names if n != 'omitted'])

# Pass 2
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        running = f['processing']['running']['speed']['data'][()]
        pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()].astype(float)

# Pass 3
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)   # reads all of the above again
```

iii. The AI acknowledged this in CONVERSION_NOTES Step 6 under "Code inefficiencies identified: … Multiple passes over NWB files", but kept the design because global percentile edges and a global image vocabulary must be known before any trial can be encoded, and it considered the cost acceptable given the 6-minute total runtime. (Caching the per-session extracts in memory during a single pass, as the reference does, would have achieved the same result without the extra I/O.)

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several things are computed or stored and then never used:
- **Whole-session neural resampling.** Every neuron's trace is interpolated to 30 Hz across the entire ~75-minute recording, but only the bins falling inside the ~300 kept trials are retained. Aborted trials (133 of 503 in the file I checked, ~26 %) and all inter-trial time are interpolated and thrown away.
- **Upsampling Multiscope data.** The 34 Multiscope plane-experiments are upsampled 11 Hz → 30 Hz, tripling their storage without adding information, and outputs are stored as `int64` rather than `int8`; the resulting pickle is 8.3 GB.
- **Trial columns loaded but never used**: `change_time`, `is_change`, `initial_image_name`, `change_image_name` are all read (and the image names byte-decoded) but the outputs are built from the stimulus table instead.
- `cell_specimen_ids` is subset by `valid_roi` and returned, but never referenced afterwards.
- `stim_data['stop_time']` is read and `change_stops` is zipped into the change loop as `ce`, which is unused (the window is always `cs + 0.75`).
- `ProcessPoolExecutor`, `as_completed` and `sys` are imported and unused.
- Pass 2 concatenates every running-speed and pupil sample of the entire dataset into memory just to take six percentiles.

ii.
```python
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)   # whole session
...
'change_time': trials_grp['change_time'][()],
'is_change': trials_grp['is_change'][()].astype(bool),
'initial_image_name': trials_grp['initial_image_name'][()],
'change_image_name': trials_grp['change_image_name'][()],
...
'cell_specimen_ids': cell_specimen_ids[valid_roi],
...
for cs, ce in zip(change_starts, change_stops):   # ce unused
...
all_running_cat = np.concatenate(all_running_values) if all_running_values else np.array([0.0])
```

iii. The AI did not flag any of these. Its only related remarks are Step 6's acknowledgement of "Multiple passes over NWB files" and Step 7's conclusion that runtime was acceptable, so the unused loads and the whole-session interpolation were never revisited. Resampling the full session before segmenting is a reasonable simplification (it guarantees one common grid), but it means roughly a third of the interpolation work is discarded.
