# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI bypassed the AllenSDK entirely and read the NWB/HDF5 files directly with `h5py`. Discovery of what exists is done from the metadata CSV `project_metadata/ophys_experiment_table.csv`, which is then intersected with the set of `behavior_ophys_experiment_<id>.nwb` files actually present on disk (284 files), and further filtered to four "active" session types. Everything needed (events, trials, stimulus presentations, running, eye tracking) is pulled out of the one NWB file per experiment in `load_experiment_data()`. The full dataset is walked three separate times: Pass 1 to build the global image-name vocabulary, Pass 2 to pool running/pupil samples for percentile bin edges, Pass 3 to actually convert.

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
        running_ts    = f['processing']['running']['speed']['timestamps'][()]
        pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
```

iii. From CONVERSION_NOTES Step 1/2: the AI first explored the AllenSDK (`BehaviorOphysExperiment.from_nwb`, `CellSpecimens.from_nwb`, `Events.from_nwb`, …) and concluded that everything the SDK exposes is pre-computed and stored in the NWB file, so reading HDF5 directly is equivalent and avoids SDK/S3 overhead. It restricted to files physically on disk because "284 NWB files on disk (subset of full 1,936 experiments in metadata)", so the metadata table over-lists what is available. Step 10 Check 3 claims "h5py reads same data as AllenSDK NWB reader", and a Step-10 sanity check re-read an NWB file independently and confirmed neuron counts, neural values (`np.allclose`) and image identity matched.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values from the (already filtered) experiment table, sorted and cast to string. Each output session gets a `subject_idx` pointing into that list. Result: 38 mice.

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

iii. CONVERSION_NOTES Step 5 key decision 12: "**Subject IDs**: Use mouse_id from experiment table." Step 4 cross-checked the count: 38 mice on disk versus 82 in the full release, and the AI recorded this as an expected consequence of having only a subset of the release ("We have a subset. 38 mice is consistent with partial download").

## 1-c. How are the data split into sessions?

i. **One NWB experiment (i.e. one imaging plane) is treated as one output "session".** There is no grouping by `ophys_session_id`. 202 active experiments therefore become 202 sessions. Only the four *active* behavior session types are kept; the two passive session types (`OPHYS_2_images_A_passive`, `OPHYS_5_images_B_passive`, 71 experiments on disk) are dropped.

ii.
```python
ACTIVE_SESSION_TYPES = [
    'OPHYS_1_images_A', 'OPHYS_3_images_A',
    'OPHYS_4_images_B', 'OPHYS_6_images_B',
]
...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    ...
    all_neural.append(session_neural)      # one list entry per experiment
    all_input.append(session_input)
    all_output.append(session_output)
    all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
    region_idx = np.full(result['n_cells'], region_to_idx[row['targeted_structure']], dtype=np.int64)
    all_brain_region_idx.append(region_idx)
```

iii. From the trajectory (reasoning at step 36) and CONVERSION_NOTES Step 5 key decision 13: "**Each NWB experiment = one 'session'** in output format (one imaging plane with its own neurons)" — the argument is that each plane carries its own distinct neuron population, so it should get its own entry. The AI was explicitly aware that "Each **session** can contain multiple **experiments** (imaging planes), especially for Multiscope recordings (up to 7 planes)" but chose the per-plane split anyway. For passive sessions, Step 5 decision 2: "**Exclude passive sessions**: OPHYS_2, OPHYS_5 are passive viewing (no lick spout, satiated mice). No meaningful trial outcomes."

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept if it is a Go or Catch trial and is neither aborted nor auto-rewarded. The trial window is the full `start_time` → `stop_time` interval of the trials table (variable length, mean ≈ 254 bins ≈ 8.5 s at 30 Hz), selected on the resampled 30 Hz grid.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
...
for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop  = trials['stop_time'][trial_idx]
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
    if len(trial_time_indices) < 3:
        continue
    trial_ts = regular_ts[trial_time_indices]
    neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 key decisions 3 and 5: "**Exclude aborted and auto-rewarded trials**: Per task instructions" and "**Trial window**: Use trial start_time to stop_time from trials table. Variable length across trials." Step 4 recorded that the trials table has exactly the `go/catch/aborted/auto_rewarded` flags needed and that the resulting Go/Catch split (87.5%/12.5%) matches the whitepaper.

## 1-e. How are trials filtered based on quality controls?

i. Four filters are applied: (1) Go or Catch only, excluding aborted and auto-rewarded; (2) trials with fewer than 3 timepoints on the 30 Hz grid are dropped; (3) trials that do not carry exactly one of `hit/miss/false_alarm/correct_reject` are dropped; (4) whole experiments with fewer than 2 valid trials (checked both before and after trial construction) are dropped, as are experiments with zero valid ROIs or no natural-image stimulus table. No neural-quality- or behavior-quality-based trial rejection is applied.

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
    if trials['hit'][trial_idx]:      outcome = 0
    elif trials['miss'][trial_idx]:   outcome = 1
    elif trials['false_alarm'][trial_idx]: outcome = 2
    elif trials['correct_reject'][trial_idx]: outcome = 3
    else:
        continue  # Unknown outcome, skip
...
if len(neural_trials) < 2:
    print(f"  WARNING: Only {len(neural_trials)} processed trials ... skipping")
    return None
```

iii. CONVERSION_NOTES Step 3 "Trial curation rules": aborted trials are premature licks before the change so are "excluded from performance calculations"; auto-rewarded are the free rewards at session start / after 10 misses; "Per task instructions: include Go and Catch trials, exclude Aborted and Auto-rewarded". Step 10 Check 5 verified "All sessions have >= 2 trials (min: 39)". The AI deliberately did *not* drop trials for having all-zero neural data: "This is NOT a bug: calcium events are sparse (~0.25% of timepoints nonzero)."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/event_detection/data` — the pre-computed discrete calcium events (FastLZeroSpikeInference), column-subset to ROIs with `valid_roi == True`. dF/F traces are *not* used for the neural signal; only the dF/F **timestamps** group is read, to get the ophys frame times (verified identical to the event_detection timestamps).

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][()].astype(bool)
cell_specimen_ids = cell_table['cell_specimen_id'][()]

ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][()]

events_data = f['processing']['ophys']['event_detection']['data'][()]
# events_data shape: (n_timepoints, n_all_cells)
events_valid = events_data[:, valid_roi]  # (n_timepoints, n_valid_cells)
```

iii. CONVERSION_NOTES Step 5 key decision 1: "**Neural signal: events (not dF/F)**: Paper says 'We performed our analyses on discrete calcium events.' Use raw events from event_detection." The quote is genuine — the paper (Piet et al.) states "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f". Step 1 also notes "**Events are PRE-COMPUTED** via FastLZeroSpikeInference. Optional filtering with causal half-gaussian (scale=2.0/31.0 sec, n_steps=20)" — the AI identified the SDK's event-smoothing option but chose not to apply it.

## 2-b. How is the `neural` data processed?

i. Three operations: (1) ROI selection by `valid_roi`; (2) linear interpolation of each neuron's event trace from the native ophys timestamps onto a uniform 30 Hz grid spanning the session; (3) clipping to be non-negative after interpolation. No smoothing/filtering, no normalisation, no z-scoring, no baseline subtraction. Per-trial slices are taken out of the resampled matrix and transposed to `(n_neurons, n_timepoints)`, stored as float32.

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

events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
# Ensure non-negative (events should be >= 0)
events_resampled = np.maximum(events_resampled, 0)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. CONVERSION_NOTES Step 6: "Resamples all data streams to 30 Hz via linear interpolation … Events are clipped to >=0 after interpolation." The 30 Hz choice is justified by the paper quote "linearly interpolating onto a consistent set of 30hz timestamps relative to the triggering behavioral event" (Step 3 table) and by the need for one common bin size across the 31 Hz Scientifica and 11 Hz Multiscope rigs (Step 4 discrepancy table). The AI acknowledged the consequence of unsmoothed events — "2,602 trials with all-zero neural data" in the verification log — but declared it expected ("calcium events are sparse (~0.25% of timepoints nonzero)") rather than revisiting the choice, and in Step 12 attributed the near-chance decoding of 4 of 5 outputs to the decoder architecture, sparsity and trial length rather than to the neural representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Exactly one neuron-level filter: keep only ROIs with `valid_roi == True` from the NWB `cell_specimen_table`. An experiment with zero valid ROIs is skipped entirely. No activity-based, SNR-based or event-rate-based neuron rejection is applied.

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

iii. CONVERSION_NOTES Step 1: "**Cell filtering**: Only automatic filter is `valid_roi` boolean (SVM binary classifier output). `exclude_invalid_rois=True` by default." Step 3: "`valid_roi == True` (SVM classifier output). Exclusion reasons: union of cells, duplicate, edge/motion affected, apical dendrite, too small/narrow/dim." Step 5 key decision 10: "**valid_roi filtering**: Only include neurons with valid_roi=True." This is exactly what the AllenSDK does by default when loading via `BehaviorOphysExperiment`, so it reproduces SDK behaviour while reading raw HDF5. Step 10 sanity check: "Neuron count: NWB valid_roi=12, converted session 0 = 12. PASS."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the **trial start time** from the trials table. All data streams (events, running, pupil) are first put on one common 30 Hz grid derived from the ophys timestamps, then a single boolean mask `start_time <= t < stop_time` selects the indices used for the neural matrix and for every output row — so all streams share the identical time index by construction. Metadata records `temporal_alignment_event = 'Trial start time (first stimulus onset of trial)'`, `off_start = 0.0`, `off_end = None` (variable-length trials).

ii.
```python
t_trial_start = trials['start_time'][trial_idx]
t_trial_stop = trials['stop_time'][trial_idx]
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
trial_ts = regular_ts[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
image_idx     = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
running_trial = running_resampled[trial_time_indices]
pupil_trial   = pupil_resampled[trial_time_indices]
```
```python
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
'off_start': 0.0,
'off_end': None,
```

iii. CONVERSION_NOTES Step 5 key decisions 5 and 6: "**Trial window**: Use trial start_time to stop_time from trials table"; "**Alignment event**: Trial start time (stimulus onset). off_start=0, off_end=None (variable)." The instruction to "temporally align based on ophys timestamp" is honoured by deriving the common grid from the ophys timestamps. Step 3 records that the Allen rig synchronises all clocks on an NI PCI-6612 board at 100 kHz, which is the AI's justification for interpolating streams onto a shared time base. Step 10 Check 5 verified "Neural/output length alignment at trial boundaries".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — everything is rebinned. All sessions are resampled to a uniform 30 Hz grid (`time_bin_size = 33.333 ms`) by linear interpolation. Native rates are ~31 Hz (Scientifica single-plane) and ~11 Hz (Multiscope), so single-plane data is slightly downsampled and Multiscope data is upsampled ~3x. The grid is `np.arange(ophys_ts[0], ophys_ts[-1], 1/30)`.

ii.
```python
TARGET_RATE_HZ = 30.0  # Paper: "linearly interpolating onto a consistent set of 30hz timestamps"
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms
...
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
...
'time_bin_size': TIME_BIN_MS,
'target_rate_hz': TARGET_RATE_HZ,
```

iii. CONVERSION_NOTES Step 5 key decision 4: "**Resample to 30 Hz**: Paper interpolates to 30 Hz. Needed for consistent time bins across Scientifica (31 Hz) and Multiscope (11 Hz). time_bin_size = 33.33 ms." Step 4 discrepancy table: "Frame rate | 31 Hz (CAM2P), 11 Hz (MESO) | Confirmed ~30.95 Hz for CAM2P | 30 Hz interpolated (paper) | Will resample all to 30 Hz." This is also required by the target-format rule that "Time bins should be the same size for all trials and sessions."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The natural-image **stimulus presentations** table inside `intervals/` (`Natural_Images_..._presentations`), specifically `image_name`, `start_time` and `omitted`. It is *not* derived from the trials table's `initial_image_name` / `change_image_name`. The stimulus table is located by scanning `f['intervals']` and taking the first key that is not `trials` and does not contain "spontaneous" or "movie".

ii.
```python
stim_key = None
for k in f['intervals'].keys():
    if k != 'trials' and 'spontaneous' not in k.lower() and 'movie' not in k.lower():
        stim_key = k
        break
...
stim_data = {
    'start_time': stim['start_time'][()],
    'stop_time': stim['stop_time'][()],
    'image_name': stim['image_name'][()],
    'is_change': stim['is_change'][()].astype(bool),
    'omitted': stim['omitted'][()],
}
if isinstance(stim_data['image_name'][0], bytes):
    stim_data['image_name'] = np.array([x.decode() for x in stim_data['image_name']])
if stim_data['omitted'].dtype == float:
    stim_data['omitted'] = stim_data['omitted'].astype(bool)
```

iii. CONVERSION_NOTES Step 5 variable mapping: "`stimulus_presentations.image_name` → output[0]: image_identity — Map to categorical int, time-varying per ophys frame — 8 natural images". Using the flash table rather than the trials table gives the image that is physically on screen at every moment, including across omitted flashes, which the AI wanted because image identity must be time-varying within the trial.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each 30 Hz timepoint in the trial, `searchsorted` finds the most recent non-omitted flash onset and its image name is assigned to that timepoint (so identity is held constant through the 500 ms grey period and through omitted flashes). Names are first mapped to a per-experiment sorted index, then remapped to a **global** sorted vocabulary built in Pass 1 over every NWB file (16 names total: 8 in image set A + 8 in image set B). Result: a `(n_timepoints,)` int64 row with values 0–15.

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
            img_name = stim_names[insert_idx[i]]
            image_idx[i] = name_to_idx.get(img_name, 0)
        else:
            image_idx[i] = 0  # Before first stimulus
    return image_idx
```
```python
# Pass 1 global vocabulary
all_image_names_set.update([n for n in img_names if n != 'omitted'])
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

iii. CONVERSION_NOTES Step 5 key decisions 7 and 8: "**Image identity during gray screen**: Use the identity of the image that was just shown (last presented image)"; "**Image identity for omitted flashes**: Continue with previous image identity." Step 6: "Image identity at each timepoint determined by most recent non-omitted stimulus onset." The global vocabulary is needed because sessions 1/3 use image set A and sessions 4/6 use image set B; the converted data shows 16 near-uniform classes (~0.058–0.067 each), which the AI cites as a sanity check.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated at exactly the same 30 Hz timestamps (`trial_ts = regular_ts[trial_time_indices]`) that index the neural matrix, so alignment is guaranteed by construction; the row is stacked into the `(5, n_timepoints)` output array for the trial.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
n_tp = len(trial_ts)
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
...
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)  # (4, n_tp)
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)  # (5, n_tp)
```

iii. CONVERSION_NOTES Step 10 Check 2 sanity check: "Image identity (session 0, trial 5, timepoint 10): expected 'im063', got 'im063'. PASS", verified by re-reading the NWB stimulus table independently of the conversion code. Step 10 Check 5 also verified neural/output length agreement at trial boundaries.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The stimulus-presentations table's `is_change` flag combined with `omitted` and the flash `start_time`. It is *not* derived from the trials table `change_time`/`go` columns. Because the SDK sets `is_change = False` for sham (catch) changes, catch trials automatically get an all-zero change row.

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. CONVERSION_NOTES Step 5 variable mapping: "`trials.is_change` + stimulus timing → output[1]: image_change — Binary 1 at change timepoint, 0 otherwise, time-varying — 1 for one 750ms window at change". Step 12 "Additional Verification": "Catch trials: 0/6,515 have image_change signal (correct: catch = sham change). Go trials: 0/45,477 missing image_change signal (all have change)."

## 4-b. What processing is involved in computing `output` *Image change*?

i. For every change flash in the session, all trial timepoints falling in the window `[change_onset, change_onset + 0.75 s)` are set to 1; everything else is 0. The 750 ms window is one image flash (250 ms) plus the following grey interval (500 ms). The loop runs over every change in the whole session for every trial.

ii.
```python
def get_image_change_at_timepoints(timepoints, stim_data):
    n_tp = len(timepoints)
    change_signal = np.zeros(n_tp, dtype=np.int64)
    change_mask = stim_data['is_change'] & ~stim_data['omitted']
    change_starts = stim_data['start_time'][change_mask]
    change_stops = stim_data['stop_time'][change_mask]
    # For each change, mark timepoints within the 750ms image presentation interval
    for cs, ce in zip(change_starts, change_stops):
        # Mark the full image interval (stimulus + gray) as change
        # Use 750ms window from change start
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1
    return change_signal
```

iii. CONVERSION_NOTES Step 3: "Image presentation | 250 ms stimulus + 500 ms gray = 750 ms". Step 6: "Image change signal: 1 during 750ms window starting at change onset." Step 9/10 consistency check: "Image change fraction | ~1/13 flashes | 7.7% | Yes (~1/13=7.7%)" — the observed 0.077 fraction of change timepoints is offered as the sanity check.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — the variable is constructed directly as a binary 0/1 indicator with value names `['no_change', 'change']`. The final distribution is `no_change 0.923 / change 0.077`.

ii.
```python
img_change = trial_data_out['image_change'].astype(np.int64)
...
output_values = [
    global_image_names,          # image identity values
    ['no_change', 'change'],     # image change values
    ...
]
```

iii. The Decoder Output spec calls for "Image change, binary variable. Have value of 1 right after a change in image identity, otherwise 0. Time-varying." The AI implements exactly that, choosing the 750 ms post-change flash+grey interval as the extent of "right after a change" (CONVERSION_NOTES Step 6).

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity: computed on `trial_ts`, the identical 30 Hz timestamps used to slice the neural matrix, then stacked as row 1 of the `(5, n_timepoints)` output.

ii.
```python
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
...
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)
```

iii. Alignment is guaranteed because all streams are placed on the one `regular_ts` grid before trial segmentation (CONVERSION_NOTES Step 6). Step 12 verified that the change row is present for 100% of go trials and absent for 100% of catch trials.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` (cm/s) and its `timestamps` (~60 Hz) from the NWB file — the same object the SDK exposes as `dataset.running_speed`.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. CONVERSION_NOTES Step 1 lists `RunningSpeed.from_nwb()` as the SDK loader for "running speed (cm/s)"; Step 2 records the stream as "(270240,) ~60 Hz". Step 5 maps "`running/speed/data` → output[2]: running_speed".

## 5-b. What processing is involved in computing `output` *Running speed*?

i. A single linear interpolation from the ~60 Hz running timestamps onto the 30 Hz grid (`np.interp`, which clamps rather than extrapolating outside the recorded range), then per-trial slicing, then discretisation using globally pre-computed bin edges. No smoothing, no absolute value, no outlier removal (the resulting bin edges run from −24.1 to 99.9 cm/s, i.e. negative/backward running is retained).

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. CONVERSION_NOTES Step 5 mapping: "Interpolate to 30 Hz, discretize into 5 percentile bins, time-varying." Step 6: "Resamples all data streams to 30 Hz via linear interpolation" and "Vectorized interpolation using np.interp" listed as a speedup.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins. Crucially, the bin edges are computed in a **separate pass over the raw, native-rate (~60 Hz), whole-session running traces of all 202 experiments**, not over the trial-window resampled values that actually get binned. `np.percentile` at 0/20/40/60/80/100 gives the edges; a monotonicity guard nudges duplicate edges by 1e-10; `np.digitize` on the interior edges maps to 0–4; NaN maps to bin 0. Observed class fractions in the converted data are 0.199 / 0.203 / 0.184 / 0.209 / 0.205 (not exactly 20% each, because the edge-defining distribution differs from the binned distribution).

ii.
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.linspace(0, 1, n_bins + 1)
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
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
# Pass 2
running = f['processing']['running']['speed']['data'][()]
all_running_values.append(running.astype(np.float32))
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
```

iii. The Decoder Output spec requires "Running speed, discretized into five equal percentile bins." CONVERSION_NOTES Step 6 documents the "3-pass approach: … (2) compute global percentile bins for running/pupil". Step 9 consistency check records "Running speed bins | 5 equal percentile | 18-21% each | Yes (equal percentile)" — i.e. the AI noticed and accepted the deviation from exactly 20%. Step 7 also notes "Running speed bins slightly unequal within sessions (expected, bins computed globally)."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is put on the shared 30 Hz `regular_ts` grid before trial segmentation, then indexed with the identical `trial_time_indices` used for the neural matrix, so it is aligned by construction.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
neural_trial  = events_resampled[trial_time_indices, :].T.astype(np.float32)
running_trial = running_resampled[trial_time_indices]
```

iii. CONVERSION_NOTES Step 3 notes "Temporal sync: NI PCI-6612 board at 100 kHz, all clocks synchronized", which is the AI's stated grounds for cross-interpolating behavioral and ophys streams onto one timebase. Step 10 Check 5 confirmed "Neural/output length alignment at trial boundaries".

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (pupil **area**, not width or height) plus its `timestamps` (~30 Hz) and the `acquisition/EyeTracking/likely_blink/data` boolean. If the EyeTracking group is missing, the experiment is still processed but pupil is left as NaN (this happened for 3 experiments: 795953296, 806456687, 833631914).

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

iii. CONVERSION_NOTES Step 5 variable mapping: "`EyeTracking/pupil_tracking/area` → output[3]: pupil_diameter — Interpolate to 30 Hz, handle blinks (NaN→interpolate), discretize into 5 percentile bins — **Use pupil area as proxy for diameter**." Step 4 flagged "Pupil | area with NaNs during blinks | ~9% blinks | 30 Hz camera | Need to handle NaN/blinks."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) Frames flagged `likely_blink` are set to NaN; (2) all NaN runs are filled by linear interpolation **in sample-index space** over the native ~30 Hz trace; (3) the gap-filled trace is linearly interpolated onto the 30 Hz `regular_ts` grid; (4) per-trial slicing; (5) discretisation with globally pre-computed percentile edges. Because `np.interp` clamps at the ends, no NaN survives for experiments that have eye data at all.

ii.
```python
def interpolate_nans(arr):
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
pupil_area = raw_data['pupil_area'].copy().astype(float)
pupil_ts = raw_data['pupil_ts']
likely_blink = raw_data['likely_blink']
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

iii. CONVERSION_NOTES Step 5 key decision 9: "**Pupil NaN handling**: During blinks (likely_blink=True or NaN), linearly interpolate. Compute percentile bins from non-blink data." Step 6: "Pupil blinks (likely_blink=True) set to NaN and interpolated before resampling."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same machinery as running speed: five equal-percentile bins, edges computed in Pass 2 over the pooled **blink-excluded, native-rate, whole-session** pupil-area samples of all experiments; `np.digitize` + clip to 0–4; NaN → bin 0. Observed class fractions are 0.228 / 0.185 / 0.191 / 0.190 / 0.206 — bin 0 is over-represented partly because the three experiments with no eye-tracking data have every timepoint mapped to bin 0.

ii.
```python
# Pass 2
pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()].astype(float)
blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
pupil_area[blink] = np.nan
valid_pupil = pupil_area[~np.isnan(pupil_area)]
if len(valid_pupil) > 0:
    all_pupil_values.append(valid_pupil.astype(np.float32))
...
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. The Decoder Output spec requires "Pupil diameter, discretized into five equal percentile bins." CONVERSION_NOTES Step 9 consistency check: "Pupil diameter bins | 5 equal percentile | 19-23% each | Yes (roughly equal)." The AI's stated rationale for excluding blinks from the edge computation is Step 5 decision 9 ("Compute percentile bins from non-blink data"). The pupil-area bin edges (125 → 323,783) show the distribution is heavy-tailed, which percentile binning handles by construction.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Identical to running speed — interpolated onto the shared 30 Hz grid before segmentation, then indexed with the same `trial_time_indices` as the neural matrix.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. Same justification as running speed (hardware-synchronised clocks, Step 3), and the same construction guarantee of identical indices (Step 6).

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the NWB trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that order.

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
```

iii. CONVERSION_NOTES Step 5 mapping: "`trials.hit/miss/false_alarm/correct_reject` → output[4]: trial_outcome — Static per-trial categorical — 4 classes." Step 4 confirmed "Trial outcomes | Each valid trial has exactly 1 outcome | Confirmed". Step 5 planned sanity check: "Trial outcome distribution: Go trials → hit or miss; Catch trials → FA or CR."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Mapped to a fixed integer code 0–3 with value names `['hit','miss','false_alarm','correct_reject']`. Although it is a static per-trial variable, it is broadcast to a constant row of length `n_timepoints` so that the output array is a uniform `(5, n_timepoints)` matrix. Trials with no outcome flag set are dropped rather than assigned a sentinel. Final distribution: hit 0.302, miss 0.572, false_alarm 0.017, correct_reject 0.108.

ii.
```python
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
...
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)  # (5, n_tp)
...
output_values = [
    ...,
    ['hit', 'miss', 'false_alarm', 'correct_reject'],  # trial outcomes
]
```

iii. The format spec says "Can be time-varying or discrete values per trial. If at all possible, make it time-varying" and requires a single `(n_output, n_timepoints)` array, so the AI broadcast the static label across time (comment in the code: "For mixed time-varying and static, we need to handle carefully … Combined: (n_output, n_timepoints) where static is broadcast"). Step 10 Check 5 verified "Trial outcome constant within trials". Step 12 checked that outcome proportions vary sensibly across sessions.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The handling is:
- **Unreadable / malformed NWB file**: the whole `load_experiment_data` body is wrapped in try/except; on any exception the experiment is skipped with an `ERROR` message.
- **No valid ROIs**: experiment skipped.
- **No natural-image stimulus table**: experiment skipped.
- **Missing eye tracking**: caught, warned, `pupil_area = None`; every pupil value for that experiment becomes NaN and therefore bin 0 (3 experiments).
- **Blink frames / NaNs in pupil**: set to NaN then linearly interpolated over.
- **Behavior samples outside the ophys time range**: `np.interp` clamps to the endpoint value rather than producing NaN.
- **NaNs reaching discretisation**: mapped to bin 0.
- **Degenerate percentile edges** (constant signal): nudged by 1e-10 so edges are strictly increasing.
- **Interpolation-induced negative event values**: clipped to 0.
- **Too-short trials** (<3 bins) and **trials with no outcome flag**: dropped.
- **Experiments with <2 usable trials**: dropped.
- **Byte-string vs str and float-vs-bool column dtypes** in HDF5: explicitly normalised.

ii.
```python
try:
    with h5py.File(nwb_path, 'r') as f:
        ...
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
```
```python
if n_valid == 0:
    print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping"); return None
if stim_key is None:
    print(f"  WARNING: No stimulus presentations found in {experiment_id}, skipping"); return None
except (KeyError, Exception):
    print(f"  WARNING: No pupil tracking data in {experiment_id}")
```
```python
if isinstance(trials['initial_image_name'][0], bytes):
    trials['initial_image_name'] = np.array([x.decode() for x in trials['initial_image_name']])
if stim_data['omitted'].dtype == float:
    stim_data['omitted'] = stim_data['omitted'].astype(bool)
```
```python
for i in range(1, len(edges)):
    if edges[i] <= edges[i-1]:
        edges[i] = edges[i-1] + 1e-10
...
binned[np.isnan(values)] = 0
...
events_resampled = np.maximum(events_resampled, 0)
...
if nans.all():
    return np.zeros_like(arr)
```

iii. CONVERSION_NOTES Step 6: "Handle missing data appropriately… Pupil blinks (likely_blink=True) set to NaN and interpolated before resampling; Events are clipped to >=0 after interpolation." Step 10 Check 5 lists 8 edge-case checks that all passed: "Neural/output length alignment at trial boundaries; No NaN in neural data; All events non-negative; All output values in valid range; Trial outcome constant within trials; Brain region indices valid; Subject indices valid; All sessions have >= 2 trials (min: 39)." The 2,602 all-zero-neural-data warnings are explicitly argued not to be data errors: "calcium events are sparse (~0.25% of timepoints nonzero). Sessions with few neurons (e.g., 4-6) will have many trials with no detected events."

## 9-a. What are the most time-consuming steps of the code?

i. Total full conversion was 358 s for 202 experiments. The dominant costs are (1) reading the full `event_detection/data` matrix (e.g. 140,204 × N float array) out of HDF5 — 0.2–1.0 s per experiment; (2) the whole-session 30 Hz resampling of that matrix, done neuron-by-neuron in a Python loop — 0.2–0.6 s per experiment; (3) the two extra full passes over all 284→202 NWB files (Pass 1 for image names, Pass 2 for running/pupil), which together account for roughly a third of wall-clock time before any real conversion starts; (4) pickling and writing the 8.3 GB output (9.1 s).

ii.
```python
events_data = f['processing']['ophys']['event_detection']['data'][()]   # full read
...
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
...
t_load = time.time() - t0
...
t_process = time.time() - t0 - t_load
print(f"  Loaded in {t_load:.1f}s, processed in {t_process:.1f}s, total {t_total:.1f}s")
```

iii. CONVERSION_NOTES Step 6 "Code inefficiencies identified: Sequential processing of experiments (could parallelize); Multiple passes over NWB files." Step 7 estimated ~0.5 s/session and ~2 min total; the actual run took 6 min, still comfortably under the 15-minute budget, so the AI did not optimise further. The script imports `ProcessPoolExecutor` but never uses it.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several, in rough order of cost:
- `get_image_at_timepoints`: after a vectorised `searchsorted` it falls back to a **per-timepoint Python loop** to look up names in a dict. With ~52,000 trials × ~254 bins that is ~13 M Python iterations. It could be one fancy-index into an integer-coded array.
- `interpolate_to_regular_grid`: a Python loop calling `np.interp` once **per neuron** (up to 666 per experiment). `scipy.interpolate.interp1d(..., axis=0)` handles the whole matrix at once.
- `get_image_change_at_timepoints`: loops over **every change in the whole session** for **every trial**, each iteration doing a full `(timepoints >= cs) & (timepoints < cs+0.75)` scan — O(n_trials × n_changes × n_bins). A single `searchsorted` per trial would do.
- Trial window selection: `trial_mask = (regular_ts >= start) & (regular_ts < stop)` scans the entire session grid (~140 k samples) once per trial — ~300 × 140 k ≈ 4×10⁷ comparisons per experiment. `np.searchsorted` is O(log T).
- The per-trial `[local_to_global.get(v, 0) for v in ...]` list comprehension, replaceable with a lookup-table index.
- Pass 1 / Pass 2 `exp_table.iterrows()` loops re-open every file sequentially and could be merged and/or parallelised.

ii.
```python
    insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
    for i in range(n_tp):                      # <-- per-timepoint Python loop
        if insert_idx[i] >= 0:
            img_name = stim_names[insert_idx[i]]
            image_idx[i] = name_to_idx.get(img_name, 0)
```
```python
        for i in range(data.shape[1]):         # <-- per-neuron loop
            result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
```
```python
    for cs, ce in zip(change_starts, change_stops):   # <-- all changes x all trials
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1
```
```python
        trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
        trial_time_indices = np.where(trial_mask)[0]
```
```python
            img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 6 claims the opposite for two of these: "Code speedups added: Vectorized interpolation using np.interp; Efficient searchsorted for image identity assignment." Both claims are only half true — `np.interp` is called inside a per-column loop, and the `searchsorted` result is consumed by a per-element Python loop. The AI did not pursue further optimisation because the full run (6 min) already met the 15-minute target (Step 7/Step 9).

## 9-c. What processing does the code repeat multiple times?

i.
- **Every NWB file is opened and read three times**: Pass 1 (stimulus `image_name`), Pass 2 (running + pupil + blink), Pass 3 (everything, including running/pupil again). Passes 1 and 2 could have been folded into Pass 3 with a deferred discretisation step (which is what the reference does).
- **Running speed and pupil area are decoded twice** — once at native rate in Pass 2 for the bin edges, once again in Pass 3 for the actual conversion.
- **Image names are collected twice**: globally in Pass 1, then again per-experiment as `all_stim_images` inside `process_single_experiment`, and then reconciled through the `local_to_global` remap — the local vocabulary is redundant with the global one.
- **Dictionaries are rebuilt inside hot functions**: `name_to_idx` is constructed on every call of `get_image_at_timepoints` (i.e. once per trial), and `local_to_global` uses `global_image_names.index(name)`, a linear search, rather than a prebuilt dict.
- `get_image_change_at_timepoints` re-derives the session-wide `change_mask`/`change_starts` from scratch for every trial.

ii.
```python
    # Pass 1
    for _, row in exp_table.iterrows():
        with h5py.File(nwb_path, 'r') as f:
            ...  img_names = f['intervals'][k]['image_name'][()]
    # Pass 2
    for idx, (_, row) in enumerate(exp_table.iterrows()):
        with h5py.File(nwb_path, 'r') as f:
            running = f['processing']['running']['speed']['data'][()]
            pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()]
    # Pass 3
    for idx, (_, row) in enumerate(exp_table.iterrows()):
        raw_data = load_experiment_data(nwb_path, eid)   # reads running + pupil again
```
```python
    name_to_idx = {name: i for i, name in enumerate(all_image_names)}   # rebuilt per trial
...
        local_to_global[local_idx] = global_image_names.index(name)      # linear search
```
```python
    all_stim_images = sorted(set(
        raw_data['stim']['image_name'][raw_data['stim']['image_name'] != 'omitted']
    ))                                                                   # already known from Pass 1
```

iii. CONVERSION_NOTES Step 6 acknowledges this explicitly as a known inefficiency: "Code inefficiencies identified: … Multiple passes over NWB files." The AI's reason for the multi-pass design is that both the image vocabulary and the percentile bin edges must be global across the whole dataset before any trial can be encoded, and it judged the extra I/O acceptable given the 6-minute total runtime.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Whole-session resampling**: the entire session's event matrix, running trace and pupil trace are interpolated onto the 30 Hz grid, but only the kept Go/Catch trial windows survive. Aborted trials (a large fraction — 8,057 of 13,709 trials in the first 25 files are kept, so ~40% of the timeline is aborted/inter-trial) are resampled and thrown away. This is the single largest piece of wasted compute.
- **`interpolate_nans` over the full pupil trace**, including the discarded portions.
- **Dead variables**: `trial_outcome = np.array([...])` is built for every trial and never used (`outcome_broadcast` is what goes into the output); `change_stops`/`ce` is loaded and unpacked but never read; `n_trials` is computed and unused.
- **Unused loaded fields**: `cell_specimen_ids` is extracted, sliced by `valid_roi`, returned, and never consumed; `valid_roi` itself is returned but only used inside the loader; `trials['is_change']` and `stim['stop_time']` are read from disk and never used.
- **Redundant per-experiment image vocabulary** (`all_stim_images`) that is immediately remapped onto the global vocabulary.
- **Output rows are stored as `int64`**, i.e. 8 bytes per categorical value in the range 0–15, and neural data as dense float32 despite being ~99.75% zeros; the pickle is 8.3 GB where a compact dtype (or sparse storage) would be far smaller. The `--show-processing` plotting path also re-derives several quantities purely for figures.

ii.
```python
    events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)  # whole session
    running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
    pupil_area = interpolate_nans(pupil_area)
    pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
```
```python
            # 5. Trial outcome (static)
            trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)   # never used
            ...
            outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
```
```python
    for cs, ce in zip(change_starts, change_stops):   # `ce` never used
```
```python
            n_trials = len(trials_grp['start_time'][()])   # never used
            'cell_specimen_ids': cell_specimen_ids[valid_roi],   # never used downstream
            'is_change': trials_grp['is_change'][()].astype(bool),  # never used
```

iii. The AI did not identify any of these in CONVERSION_NOTES; its self-assessment of inefficiency stopped at "Sequential processing of experiments" and "Multiple passes over NWB files" (Step 6). Because the full conversion finished in 6 minutes — well inside the 15-minute threshold that would have forced optimisation under Step 7/Step 9 — there was no pressure to remove this waste. The one cost the AI did feel was memory/disk: the 8,325.7 MB pickle is recorded in Step 9 without comment on dtype choice.
