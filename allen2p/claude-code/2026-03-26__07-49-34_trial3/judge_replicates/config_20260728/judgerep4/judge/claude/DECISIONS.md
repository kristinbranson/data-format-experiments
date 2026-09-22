# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using `h5py`, rather than using the AllenSDK's `VisualBehaviorOphysProjectCache`. It reads the experiment metadata CSV (`ophys_experiment_table.csv`) to discover experiments, then filters to NWB files present on disk. It also filters by `session_type` to include only active behavior sessions (`OPHYS_1_images_A`, `OPHYS_3_images_A`, `OPHYS_4_images_B`, `OPHYS_6_images_B`). Three passes are made over the NWB files: (1) collect global image names, (2) collect running speed and pupil data for percentile bin computation, (3) full processing.

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
    ...

def load_experiment_data(nwb_path, experiment_id):
    with h5py.File(nwb_path, 'r') as f:
        ...
```

iii. The AI chose to use `h5py` directly rather than the AllenSDK to avoid SDK overhead and directly access NWB data structures. It filtered to active session types because passive sessions (OPHYS_2, OPHYS_5) lack meaningful behavioral responses. From CONVERSION_NOTES.md: "Passive sessions (OPHYS_2,5): 82 experiments - will EXCLUDE (no active behavior)."

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata table. They are sorted and converted to strings.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The `mouse_id` field is the standard identifier for each animal in the Allen SDK metadata.

## 1-c. How are the data split into sessions?

i. Each individual NWB experiment (one imaging plane) is treated as a separate "session" in the output format. The AI does NOT group multiple imaging planes from the same `ophys_session_id` into a single session.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    ...
    raw_data = load_experiment_data(nwb_path, eid)
    ...
    result = process_single_experiment(raw_data, exp_meta)
    ...
    all_neural.append(session_neural)
```

iii. From CONVERSION_NOTES.md Step 5, Decision 13: "Each NWB experiment = one 'session' in output format (one imaging plane with its own neurons)." This means each experiment/imaging plane is a separate entry in the sessions list, rather than combining multiple planes from the same recording session.

## 1-d. How are the data split into trials?

i. Trials are defined using the trials table from the NWB file. The AI filters to Go and Catch trials, excluding Aborted and Auto-rewarded. For each valid trial, the time window from `start_time` to `stop_time` is used on the 30 Hz resampled grid. Trials with fewer than 3 timepoints are skipped.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
...
for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop = trials['stop_time'][trial_idx]
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
    if len(trial_time_indices) < 3:
        continue
```

iii. The instruction states to "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be Go or Catch (not Aborted or Auto-rewarded), (2) must have at least 3 timepoints on the resampled grid, (3) trials with unknown outcome (not hit/miss/FA/CR) are skipped, (4) experiments with fewer than 2 valid trials are skipped entirely.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
...
if len(trial_time_indices) < 3:
    continue
...
if trials['hit'][trial_idx]:
    outcome = 0
elif trials['miss'][trial_idx]:
    outcome = 1
elif trials['false_alarm'][trial_idx]:
    outcome = 2
elif trials['correct_reject'][trial_idx]:
    outcome = 3
else:
    continue  # Unknown outcome, skip
...
if len(neural_trials) < 2:
    return None
```

iii. The AI requires trials to have a known outcome (one of hit/miss/FA/CR) in addition to being Go or Catch. The 2-trial minimum prevents degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `event_detection` data (calcium events from FastLZeroSpikeInference), NOT from `dff_traces` (dF/F). Only neurons with `valid_roi == True` are included.

ii.
```python
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]  # (n_timepoints, n_valid_cells)
```

iii. From CONVERSION_NOTES.md Step 5, Decision 1: "Neural signal: events (not dF/F): Paper says 'We performed our analyses on discrete calcium events.' Use raw events from event_detection." The AI explicitly chose events over dF/F based on the paper's methodology.

## 2-b. How is the `neural` data processed?

i. The event data is filtered by `valid_roi`, then interpolated from the native ophys frame rate to a regular 30 Hz grid using linear interpolation. After interpolation, values are clipped to be non-negative.

ii.
```python
regular_ts = np.arange(t_start, t_end, dt)  # dt = 1/30

events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
```

iii. From CONVERSION_NOTES.md Step 3: "Paper analysis: Neural data interpolated to 30 Hz." The 30 Hz resampling follows the paper's methodology. Non-negativity clipping ensures events remain physically meaningful after interpolation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the `valid_roi` boolean flag from the NWB cell specimen table. Only neurons where `valid_roi == True` are included.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
cell_specimen_ids = cell_table['cell_specimen_id'][()]
n_valid = valid_roi.sum()
if n_valid == 0:
    return None
events_valid = events_data[:, valid_roi]
```

iii. From CONVERSION_NOTES.md: "valid_roi == True (SVM classifier output)." The AI applies this filter as it matches the SDK's default behavior (`exclude_invalid_rois=True`).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start time. After resampling to 30 Hz, the trial window is extracted by finding timepoints between `start_time` and `stop_time` on the regular grid.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The alignment event is documented as "Trial start time (first stimulus onset of trial)". The `start_time` from the trials table marks the onset of the first stimulus in the trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is resampled to 30 Hz (time bin size = 33.33 ms). This is a resampling from the native ophys frame rate (~31 Hz for Scientifica, ~11 Hz for Multiscope) to a consistent 30 Hz grid via linear interpolation.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms

regular_ts = np.arange(t_start, t_end, dt)  # dt = 1/30
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
```

iii. From CONVERSION_NOTES.md: "Paper analysis time bin: 30 Hz (interpolated). Paper: 'linearly interpolating onto a consistent set of 30hz timestamps'." The AI followed the paper's approach to unify different frame rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table in the NWB file, specifically the `image_name`, `start_time`, and `omitted` fields. For each timepoint, the most recent non-omitted stimulus onset determines the current image.

ii.
```python
def get_image_at_timepoints(timepoints, stim_data, all_image_names):
    non_omitted = ~stim_data['omitted']
    stim_starts = stim_data['start_time'][non_omitted]
    stim_names = stim_data['image_name'][non_omitted]
    name_to_idx = {name: i for i, name in enumerate(all_image_names)}
    insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
    for i in range(n_tp):
        if insert_idx[i] >= 0:
            img_name = stim_names[insert_idx[i]]
            image_idx[i] = name_to_idx.get(img_name, 0)
```

iii. The AI chose to use the stimulus_presentations table because it provides exact timing of each stimulus flash, handling omitted flashes and gray screen periods. From CONVERSION_NOTES.md: "Image identity at each timepoint determined by most recent non-omitted stimulus onset."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping built from all unique non-omitted image names across all experiments. At each timepoint within a trial, the image index is determined by the most recent non-omitted stimulus onset using `np.searchsorted`.

ii.
```python
global_image_names = sorted(all_image_names_set)
...
local_to_global = {}
for local_idx, name in enumerate(result['all_image_names']):
    if name in global_image_names:
        local_to_global[local_idx] = global_image_names.index(name)
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. A two-level mapping (local per-experiment then global) ensures consistent integer codes across all experiments.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at each timepoint on the 30 Hz resampled grid, using the same trial time indices as the neural data.

ii.
```python
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. Both neural data and image identity are evaluated at the same regular 30 Hz timepoints within each trial window.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table, using the `is_change` and `omitted` fields to identify change stimulus presentations. It is NOT derived from the trials table's `go` column.

ii.
```python
def get_image_change_at_timepoints(timepoints, stim_data):
    change_mask = stim_data['is_change'] & ~stim_data['omitted']
    change_starts = stim_data['start_time'][change_mask]
    change_stops = stim_data['stop_time'][change_mask]
```

iii. The AI used the stimulus presentations' `is_change` field, which directly identifies stimulus changes at the presentation level.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each stimulus presentation where `is_change == True` and `omitted == False`, a 750 ms window starting at the change onset is marked as 1 in a binary signal. All other timepoints are 0.

ii.
```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The 750 ms window corresponds to one stimulus flash (250 ms) plus the following gray inter-stimulus interval (500 ms).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is a binary variable: 0 (no change) and 1 (change). No thresholding is needed beyond the 750 ms window construction.

ii.
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
...
change_signal[mask] = 1
```

iii. Binary by definition per the instructions.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same alignment as image identity and neural data - computed at 30 Hz regular grid timepoints within the trial window.

ii.
```python
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. All outputs share the same timepoint grid as the neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the `processing/running/speed` data and timestamps in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. This is the standard running speed data path in the Allen NWB format.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is resampled to 30 Hz via `np.interp`, then discretized into 5 equal percentile bins. Bin edges are computed globally across all experiments from the full running speed timeseries (not just trial-segmented data).

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
# In pass 2, bin edges computed from entire sessions:
all_running_values.append(running.astype(np.float32))
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. Global percentile bins ensure consistent categories across all experiments.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins (0-4) using globally computed bin edges. NaN values are assigned to bin 0.

ii.
```python
def digitize_to_bins(values, bin_edges):
    n_bins = len(bin_edges) - 1
    binned = np.digitize(values, bin_edges[1:-1])
    binned = np.clip(binned, 0, n_bins - 1)
    binned[np.isnan(values)] = 0
    return binned.astype(np.int64)
```

iii. Percentile-based binning ensures roughly equal class counts.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is resampled to the same 30 Hz regular grid as the neural data, then trial-segmented using the same time indices.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
```

iii. All data streams share the same 30 Hz timebase.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_tracking/area` in the NWB file's eye tracking data, with blink frames identified by `likely_blink`.

ii.
```python
pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The AI used pupil **area** rather than pupil width or diameter. From CONVERSION_NOTES.md: "Use pupil area as proxy for diameter."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN, then NaN values are linearly interpolated within the pupil area timeseries before resampling to 30 Hz. Discretization into 5 percentile bins follows. Bin edges are computed globally from non-blink data across all experiments.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. Blink removal and NaN interpolation prevent blink artifacts from corrupting the signal.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal percentile bins (0-4) with NaN mapped to bin 0.

ii. See `digitize_to_bins` function in 5-c.

iii. Same rationale as running speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil data is resampled to 30 Hz and trial-segmented using the same time indices as neural data.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. Same 30 Hz timebase alignment as all other data streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

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
```

iii. These four columns are the canonical trial outcome labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) and broadcast as a constant across all timepoints within the trial.

ii.
```python
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)  # (5, n_tp)
```

iii. The outcome is per-trial but represented as time-varying (constant) to match the output format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiment loading**: Returns None, experiment is skipped.
- **No valid ROIs**: Experiment skipped with warning.
- **No stimulus presentations**: Experiment skipped.
- **Missing pupil data**: If eye tracking data is absent, pupil values are filled with NaN.
- **Pupil blinks**: Set to NaN and linearly interpolated before resampling.
- **Short trials**: Trials with fewer than 3 timepoints are skipped.
- **Unknown outcome**: Trials without a recognized outcome are skipped.
- **Few valid trials**: Experiments with fewer than 2 valid trials are skipped.
- **NaN in running/pupil**: Mapped to bin 0 during discretization.

ii.
```python
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
...
if n_valid == 0:
    return None
...
if len(trial_time_indices) < 3:
    continue
...
else:
    continue  # Unknown outcome, skip
...
if len(neural_trials) < 2:
    return None
...
binned[np.isnan(values)] = 0
```

iii. The approach ensures robustness by gracefully handling missing or incomplete data at multiple levels.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading NWB files via h5py, which is done three times (pass 1: image names, pass 2: running/pupil stats, pass 3: full processing). Each pass reopens and reads from all NWB files.

ii.
```python
# Pass 1:
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...

# Pass 2:
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        ...

# Pass 3:
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
    ...
```

iii. The three-pass architecture was chosen for simplicity (global statistics needed before final processing) but results in redundant I/O.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `get_image_at_timepoints` function uses a Python loop over all timepoints when the `insert_idx` to image name mapping could be vectorized with numpy indexing. The `interpolate_to_regular_grid` function loops over features when multi-dimensional. The local-to-global image name mapping also uses a Python list comprehension.

ii.
```python
# Loop in get_image_at_timepoints:
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)

# Loop in interpolate_to_regular_grid:
for i in range(data.shape[1]):
    result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])

# Loop for global mapping:
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. These loops are not major bottlenecks compared to I/O, but could be vectorized for cleaner code.

## 9-c. What processing does the code repeat multiple times?

i. NWB files are opened and read three separate times across the three passes. Pass 1 reads stimulus image names, pass 2 reads running speed and pupil data, and pass 3 reads everything again for full processing. The running speed and pupil data read in pass 2 are discarded and re-read in pass 3.

ii.
```python
# Pass 1: reads image_name from intervals
# Pass 2: reads running/speed/data and EyeTracking/pupil_tracking/area
# Pass 3: reads ALL data again via load_experiment_data()
```

iii. This redundancy is a design tradeoff for simpler code structure but adds significant I/O overhead.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes percentile bin edges from ALL running speed and pupil data across entire sessions (including non-trial periods like inter-trial intervals and pre/post-session time), whereas only trial-segmented data is used in the final output. The stimulus presentations data and processing (searching for stim_key, handling omitted flashes) adds complexity that could be avoided by using the simpler trials table approach.

ii.
```python
# Pass 2 collects ALL running data, not just trial-segmented:
running = f['processing']['running']['speed']['data'][()]
all_running_values.append(running.astype(np.float32))
```

iii. Including non-trial data in percentile computation may shift bin edges compared to computing from trial data only.
