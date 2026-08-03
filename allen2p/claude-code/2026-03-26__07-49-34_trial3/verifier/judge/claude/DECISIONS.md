# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files on disk using `h5py`, rather than through the AllenSDK's `VisualBehaviorOphysProjectCache`. It reads the experiment metadata CSV (`ophys_experiment_table.csv`), filters to experiments whose NWB files exist on disk and whose `session_type` is in a list of active session types (`OPHYS_1_images_A`, `OPHYS_3_images_A`, `OPHYS_4_images_B`, `OPHYS_6_images_B`), then processes each matching NWB file individually. A 3-pass approach is used: (1) collect global image names, (2) compute global percentile bins for running/pupil, (3) process each experiment.

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
    return active_exps
```

iii. The AI chose to load NWB files directly via h5py rather than the AllenSDK cache, extracting events, trials, running speed, and pupil tracking from the HDF5 structure. It explicitly filtered to active session types (excluding passive OPHYS_2/OPHYS_5), reasoning that passive sessions lack meaningful trial outcomes.

## 1-b. How are the data split into subjects?

i. Subjects are derived from the `mouse_id` column in the experiment metadata table. Unique mouse IDs are sorted and assigned integer indices.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The `mouse_id` field from the experiment table uniquely identifies each animal. This is the same identifier used in the reference.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as a separate "session" in the output format. The AI does NOT group multiple experiments (imaging planes) from the same `ophys_session_id` into a single session. Each experiment (one imaging plane) becomes one entry in the session lists.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    # ... processes each experiment independently as a "session" ...
    all_neural.append(session_neural)
```

iii. The AI's CONVERSION_NOTES.md (Step 5, Decision 13) states: "Each NWB experiment = one 'session' in output format (one imaging plane with its own neurons)." This means multi-plane sessions are split into separate entries, each with its own set of neurons.

## 1-d. How are the data split into trials?

i. Trials are defined using the trials table from the NWB file. Go and Catch trials are included; Aborted and Auto-rewarded trials are excluded. The trial window spans from `start_time` to `stop_time`. Valid trials are identified by the mask `(go | catch) & ~aborted & ~auto_rewarded`.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]

for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop = trials['stop_time'][trial_idx]
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
    if len(trial_time_indices) < 3:
        continue
```

iii. The AI follows the instructions to include Go and Catch trials and exclude Aborted and Auto-rewarded. Unlike the reference which also filters on `change_time.notna()`, the AI uses the `go | catch` boolean mask directly.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be Go or Catch (not Aborted, not Auto-rewarded), (2) must have at least 3 timepoints after resampling, (3) must have a known trial outcome (hit/miss/false_alarm/correct_reject — trials with unknown outcome are skipped). Experiments with fewer than 2 valid processed trials are skipped.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
# ...
if len(trial_time_indices) < 3:
    continue
# ...
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
# ...
if len(neural_trials) < 2:
    return None
```

iii. The filtering follows task instructions. The minimum trial count ensures decoder training has sufficient data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `event_detection/data` (calcium events from FastLZeroSpikeInference), NOT from `dff_traces` (dF/F). This is filtered by the `valid_roi` boolean from the cell specimen table.

ii.
```python
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]  # (n_timepoints, n_valid_cells)
```

iii. The AI's CONVERSION_NOTES.md (Step 5, Decision 1) states: "Neural signal: events (not dF/F): Paper says 'We performed our analyses on discrete calcium events.' Use raw events from event_detection." The AI chose to follow the paper's analysis approach rather than using raw dF/F.

## 2-b. How is the `neural` data processed?

i. Neural events are linearly interpolated from native ophys timestamps to a regular 30 Hz grid, then clipped to be non-negative. Only valid ROIs are included.

ii.
```python
regular_ts = np.arange(t_start, t_end, dt)  # dt = 1/30
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
```

iii. The paper states neural data was "linearly interpolating onto a consistent set of 30hz timestamps." The AI followed this to unify the different native frame rates (31 Hz for Scientifica, 11 Hz for Multiscope).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by the `valid_roi` boolean from the cell specimen table. Only cells with `valid_roi == True` are included.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
events_valid = events_data[:, valid_roi]
```

iii. The Allen SDK applies this filter by default (`exclude_invalid_rois=True`). The AI explicitly applies it when reading from h5py.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. The resampled 30 Hz timestamps within each trial's `[start_time, stop_time)` window are extracted, giving a variable-length neural matrix per trial.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
trial_ts = regular_ts[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The metadata records `temporal_alignment_event: 'Trial start time (first stimulus onset of trial)'` and `off_start: 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is resampled to 30 Hz (time bin size = 33.33 ms). All data streams (neural, running, pupil) are linearly interpolated to a regular 30 Hz grid.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
```

iii. The paper states "linearly interpolating onto a consistent set of 30hz timestamps." The AI followed this to ensure consistent temporal resolution across all experiments.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table (`intervals/<stim_key>/image_name`, `start_time`, `stop_time`, `is_change`, `omitted`), not from the trials table's `initial_image_name`/`change_image_name`.

ii.
```python
stim = f['intervals'][stim_key]
stim_data = {
    'start_time': stim['start_time'][()],
    'stop_time': stim['stop_time'][()],
    'image_name': stim['image_name'][()],
    'is_change': stim['is_change'][()].astype(bool),
    'omitted': stim['omitted'][()],
}
```

iii. The AI uses the stimulus presentation times to determine which image is on screen at each timepoint, using `searchsorted` to find the most recent non-omitted stimulus onset.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each timepoint, the most recent non-omitted stimulus onset is found. The image name from that stimulus is mapped to a global integer index. During gray screen periods, the last shown image identity persists. A global mapping of sorted image names to integer codes is built across all experiments.

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
        else:
            image_idx[i] = 0
    return image_idx
```

iii. The AI used the stimulus presentations table as the source of truth for which image is displayed at each timepoint, rather than deriving it from trial-level variables.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 30 Hz resampled timepoints as the neural data, using the trial-level timestamp array `trial_ts`.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. Both neural data and image identity are evaluated at the same regular 30 Hz timestamps, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentations table, specifically from entries where `is_change == True` and `omitted == False`. The `start_time` and `stop_time` of change stimuli are used.

ii.
```python
def get_image_change_at_timepoints(timepoints, stim_data):
    change_mask = stim_data['is_change'] & ~stim_data['omitted']
    change_starts = stim_data['start_time'][change_mask]
    change_stops = stim_data['stop_time'][change_mask]
```

iii. The AI uses `stim_data['is_change']` from the stimulus presentations to identify change events, rather than using the trials table's `go` column.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary signal is created: 1 during the 750 ms window starting at each change stimulus onset, 0 otherwise. This marks the image change period including the stimulus (250 ms) plus the following gray inter-stimulus interval (500 ms).

ii.
```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
return change_signal
```

iii. The 750 ms window matches the stimulus presentation cycle (250 ms image + 500 ms gray).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is a binary variable (0 or 1), not thresholded from a continuous value. No thresholding is needed.

ii. N/A

iii. The signal is inherently binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed at the 30 Hz resampled timepoints.

ii.
```python
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. Aligned via shared timestamp array.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. This is the standard running speed data from the Allen SDK's NWB format.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 30 Hz regular grid. Then it is discretized into 5 equal percentile bins computed globally across all experiments. NaN values are mapped to bin 0.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
# ... later ...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. Percentile-based binning ensures roughly equal class counts, important for balanced decoding.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using percentile-based edges computed globally from all experiments' running speed data (including non-trial periods in Pass 2). `np.digitize` maps values to bin indices 0-4.

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

iii. The percentile bin edges are computed from the full running speed traces of all experiments (entire sessions, not just trial periods), ensuring equal data counts per bin globally.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 30 Hz regular grid as neural data, then sliced using the same trial time indices.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
running_trial = running_resampled[trial_time_indices]
```

iii. Shared timestamp grid ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` (pupil AREA, not width/diameter) and `acquisition/EyeTracking/likely_blink/data`.

ii.
```python
pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The AI uses pupil area as a proxy for pupil diameter, noting in CONVERSION_NOTES.md (Step 5): "Use pupil area as proxy for diameter."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames (`likely_blink == True`) are set to NaN, then NaN values are linearly interpolated before resampling. The cleaned pupil area is resampled to 30 Hz, then discretized into 5 percentile bins.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)  # fills NaN by interpolation
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
# ... later ...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. Blinks are removed and interpolated over before resampling to prevent artifacts.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal percentile bins computed globally, using `compute_percentile_bins` and `digitize_to_bins`.

ii. See 5-c code snippets (same functions applied to pupil values).

iii. Global percentile bins ensure consistent categories across experiments.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed — resampled to 30 Hz grid, then sliced with trial time indices.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. Shared timestamp grid ensures alignment.

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

iii. These four outcomes are mutually exclusive for non-aborted, non-auto-rewarded trials. Trials without a matching outcome are skipped.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) and broadcast as a constant value across all timepoints in the trial.

ii.
```python
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)  # (5, n_tp)
```

iii. Trial outcome is static per trial, so the same code is repeated across all timepoints.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Failed experiment loading**: If `load_experiment_data` throws an exception, the experiment is skipped with a warning.
- **No valid ROIs**: Experiments with zero valid ROIs are skipped.
- **No stimulus presentations**: Experiments without a matching stimulus key are skipped.
- **Missing pupil data**: If eye tracking is unavailable, pupil values are set to NaN for all timepoints.
- **Pupil blinks**: Blink frames set to NaN, then linearly interpolated before resampling.
- **Short trials**: Trials with fewer than 3 timepoints after resampling are skipped.
- **Unknown trial outcomes**: Trials that don't match any of the 4 outcomes are skipped.
- **Few trials**: Experiments with fewer than 2 valid trials are skipped.
- **NaN in outputs**: NaN values in running speed or pupil are mapped to bin 0 during discretization.

ii.
```python
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
# ...
if n_valid == 0:
    return None
# ...
if pupil_resampled is not None:
    pupil_trial = pupil_resampled[trial_time_indices]
else:
    pupil_trial = np.full(n_tp, np.nan)
# ...
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
# ...
binned[np.isnan(values)] = 0
```

iii. The approach is robust — individual failures don't crash the pipeline.

## 9-a. What are the most time-consuming steps of the code?

i. The code uses 3 passes over NWB files: (1) image name collection, (2) running/pupil percentile computation, (3) full processing. The loading of NWB files is the main bottleneck. Each NWB file is opened up to 3 times.

ii. N/A

iii. CONVERSION_NOTES.md reports total conversion time of ~6 minutes for 202 experiments, with ~0.25s load + 0.25s processing per experiment.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `get_image_at_timepoints` function uses a Python for-loop over all timepoints to map image indices, which could be fully vectorized with NumPy indexing. The `interpolate_to_regular_grid` loops over features when data is 2D.

ii.
```python
# In get_image_at_timepoints - inner loop over timepoints:
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)
    else:
        image_idx[i] = 0
```

iii. The `searchsorted` call is vectorized, but the subsequent mapping loop is not.

## 9-c. What processing does the code repeat multiple times?

i. NWB files are opened and partially read 3 times: once for image names (Pass 1), once for running/pupil statistics (Pass 2), and once for full processing (Pass 3). Running speed and pupil area are loaded in both Pass 2 and Pass 3.

ii.
```python
# Pass 1: Collecting global image names
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        # reads image names

# Pass 2: Collecting running speed and pupil data
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        running = f['processing']['running']['speed']['data'][()]
        # ... pupil data ...

# Pass 3: Processing experiments
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
```

iii. The multi-pass design was chosen for clarity and to compute global statistics before final processing, but it means each file is read multiple times.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In Pass 2, running speed and pupil data are collected from the ENTIRE session (not just trial periods) for computing percentile bins. This means inter-trial and pre/post-session data contribute to bin edges, which may differ from computing bins only on trial data. Additionally, the `interpolate_nans` function fills in blink periods with interpolated values, but then `digitize_to_bins` still maps NaN to bin 0 — since blinks were already interpolated, the NaN handling in `digitize_to_bins` is redundant for pupil data.

ii. See Pass 2 code in 9-c.

iii. Computing percentile bins from full-session data rather than trial-only data means the bins reflect different statistics than what appears in the final output.
