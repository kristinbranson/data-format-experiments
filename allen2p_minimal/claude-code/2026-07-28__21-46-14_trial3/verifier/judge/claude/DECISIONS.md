# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the experiment table from a CSV file (`ophys_experiment_table.csv`) in the metadata directory. It filters to experiments that have corresponding NWB files on disk and are "active behavior" sessions (not passive). Each experiment's NWB file is loaded individually using `BehaviorOphysExperiment.from_nwb_path()`.

ii.
```python
exp = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_ids = set()
for f in os.listdir(NWB_DIR):
    if f.endswith('.nwb'):
        eid = int(f.replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
        nwb_ids.add(eid)
exp = exp[exp['ophys_experiment_id'].isin(nwb_ids)]
exp = exp[exp['passive'] == False]
# ...
dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
```

iii. The agent explored the data directory and found the NWB files and metadata CSVs. It loaded the experiment table to discover all available experiments, cross-referenced with actually available NWB files, and filtered to active behavior sessions. It chose to load all active behavior experiments (both VisualBehavior and VisualBehaviorMultiscope project codes) rather than filtering to a single project code.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `mouse_id` field from the experiment metadata. All unique mouse IDs across processed experiments are collected and sorted.

ii.
```python
all_subjects = sorted(set(r['mouse_id'] for r in results))
# ...
subject_idx.append(all_subjects.index(r['mouse_id']))
```

iii. The agent used the experiment metadata's `mouse_id` as the unique identifier for each subject, which is the standard approach.

## 1-c. How are the data split into sessions?

i. Each experiment (NWB file / imaging plane) is treated as a separate session. There is no grouping of experiments by `ophys_session_id` to form multi-plane sessions.

ii.
```python
for i, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(row, sample_mode=args.sample)
    if result is not None:
        results.append(result)
# Each result becomes one session in the final output
```

iii. The agent treated each NWB file (one per experiment/imaging plane) as a separate session. The agent noted that VisualBehavior experiments are single-plane and VisualBehaviorMultiscope are multi-plane. It did not group multi-plane experiments from the same recording session together.

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `dataset.trials` table. Valid trials are those where `go == True` or `catch == True`, `aborted == False`, and `auto_rewarded == False`. Each trial spans from `start_time` to `stop_time` (variable length). Neural and behavioral data are resampled to common time bins within this window.

ii.
```python
valid_trials = trials[
    ((trials['go'] == True) | (trials['catch'] == True)) &
    (trials['aborted'] == False) &
    (trials['auto_rewarded'] == False)
]
# ...
t_start = trial['start_time']
t_end = trial['stop_time']
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
```

iii. The agent used the SDK's built-in trials table to define trial boundaries (start_time to stop_time), filtering for Go and Catch trials only as specified in the instructions.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if: (1) `aborted == True`, (2) `auto_rewarded == True`, (3) `go` and `catch` are both False, (4) `start_time` or `stop_time` is NaN or `stop_time <= start_time`, (5) fewer than `MIN_TRIAL_FRAMES=3` ophys frames in the trial window, (6) fewer than `MIN_TRIAL_FRAMES` resampled bins, (7) trial outcome is unknown (-1). Experiments with fewer than 2 valid trials are skipped. Experiments with fewer than `MIN_NEURONS=5` neurons are also skipped.

ii.
```python
if np.isnan(t_start) or np.isnan(t_end) or t_end <= t_start:
    continue
if frame_mask.sum() < MIN_TRIAL_FRAMES:
    continue
if neural_resampled is None or len(bin_centers) < MIN_TRIAL_FRAMES:
    continue
if outcome == -1:
    continue
# ...
if n_neurons < MIN_NEURONS:
    return None
if trial_count < 2:
    return None
```

iii. The agent applied multiple layers of quality control: excluding aborted/auto-rewarded trials per instructions, filtering out degenerate trials with insufficient data, and requiring minimum neuron and trial counts per experiment.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces` (dF/F calcium fluorescence traces) from each experiment, accessed via `dataset.dff_traces`.

ii.
```python
dff_df = dataset.dff_traces
neural_data = np.array([row['dff'] for _, row in dff_df.iterrows()])  # (n_neurons, n_frames)
```

iii. The agent initially used `events` (detected calcium events) to match the paper's methodology, but switched to `dff_traces` after finding events were 99.7% sparse, making them impractical for decoding. dF/F traces are the standard continuous measure of neural activity for two-photon imaging.

## 2-b. How is the `neural` data processed?

i. The dF/F data is extracted for each neuron, then resampled to a common time bin size (~91ms, matching ~11Hz multiplane frame rate) within each trial window. The resampling averages neural data within each bin, or uses nearest-neighbor for bins with no data.

ii.
```python
COMMON_BIN_SIZE = 1.0 / 11.0  # ~91ms
# ...
def resample_to_common_bins(timestamps, data, trial_start, trial_end):
    bin_centers = np.arange(trial_start + COMMON_BIN_SIZE / 2, trial_end, COMMON_BIN_SIZE)
    for b, tc in enumerate(bin_centers):
        t_lo = tc - COMMON_BIN_SIZE / 2
        t_hi = tc + COMMON_BIN_SIZE / 2
        mask = (timestamps >= t_lo) & (timestamps < t_hi)
        if mask.sum() > 0:
            resampled[:, b] = data[:, mask].mean(axis=1)
        else:
            idx = np.argmin(np.abs(timestamps - tc))
            resampled[:, b] = data[:, idx]
    return resampled, bin_centers
```

iii. The agent resampled all data to a common ~11Hz bin size to match the multiplane mesoscope rate referenced in the paper. This ensures consistent temporal resolution across all experiments regardless of their native frame rate (31Hz for single-plane, 11Hz for multi-plane).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Experiments with fewer than `MIN_NEURONS=5` neurons are excluded entirely. No per-neuron quality filtering is applied beyond what the Allen SDK already provides.

ii.
```python
MIN_NEURONS = 5
# ...
if n_neurons < MIN_NEURONS:
    print(f"  SKIP {eid}: only {n_neurons} neurons (min {MIN_NEURONS})")
    return None
```

iii. The agent added a minimum neuron threshold as a basic quality control. No further neuron-level filtering (e.g., by signal quality or responsiveness) was applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to trial start time (`start_time` from the trials table). The ophys frames within the trial window (`start_time` to `stop_time`) are extracted and resampled to common time bins starting from `trial_start + COMMON_BIN_SIZE/2`.

ii.
```python
t_start = trial['start_time']
t_end = trial['stop_time']
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
trial_ophys_ts = ophys_ts[frame_mask]
trial_neural_raw = neural_data[:, frame_mask]
neural_resampled, bin_centers = resample_to_common_bins(
    trial_ophys_ts, trial_neural_raw, t_start, t_end
)
```

iii. The agent uses the trial start time as the alignment event, extracting neural data from `start_time` to `stop_time` and resampling to common bins within that window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is resampled to a common bin size of ~91ms (1/11 Hz), matching the multiplane mesoscope frame rate. This means single-plane experiments (originally ~31Hz) are downsampled by averaging ~3 frames per bin, while multiplane experiments are kept at approximately their native rate.

ii.
```python
COMMON_BIN_SIZE = 1.0 / 11.0  # ~91ms, matching multiplane frame rate
# ...
'time_bin_size': COMMON_BIN_SIZE * 1000,  # in ms
```

iii. The agent chose to resample to the multiplane rate (~11Hz) to ensure consistent temporal resolution across all experiments. This is referenced in the paper's methodology.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name` column and `start_time` of each stimulus flash. The most recent non-omitted stimulus at each time bin determines the image identity.

ii.
```python
def get_image_identity_at_times(stim_presentations, bin_centers, image_to_idx):
    stim = stim_presentations[stim_presentations['image_name'] != 'omitted'].sort_values('start_time')
    stim_times = stim['start_time'].values
    stim_idx = np.array([image_to_idx.get(n, 0) for n in stim['image_name'].values])
    indices = np.searchsorted(stim_times, bin_centers, side='right') - 1
    result = np.where(indices >= 0, stim_idx[np.clip(indices, 0, len(stim_idx) - 1)], 0)
    return result.astype(int)
```

iii. The agent used the stimulus presentations table to determine which image was on screen at each time point. It uses `searchsorted` to find the most recent stimulus onset before each bin center, effectively implementing a "last shown image" approach including during grey inter-stimulus intervals.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping built from all unique image names across all experiments. The mapping is built per-experiment first (local), then remapped to global indices during final assembly.

ii.
```python
# Per-experiment local mapping
image_names = sorted(stim_cd[...]['image_name'].unique().tolist())
img_identity = get_image_identity_at_times(stim_cd, bin_centers,
                                            {name: i for i, name in enumerate(image_names)})
# Global remapping
all_image_names = sorted(all_image_names)
global_img_to_idx = {name: i for i, name in enumerate(all_image_names)}
local_to_global_arr = np.array([global_img_to_idx[name] for name in r['image_names']])
img_id_global = local_to_global_arr[out['image_identity']]
```

iii. The agent built a two-stage mapping: first a local per-experiment mapping, then a global remapping. This ensures consistent integer codes across all sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same resampled bin centers as the neural data, using `searchsorted` on stimulus presentation times. This ensures frame-by-frame alignment.

ii.
```python
img_identity = get_image_identity_at_times(stim_cd, bin_centers,
                                            {name: i for i, name in enumerate(image_names)})
```

iii. Both neural and image identity data use the same `bin_centers` time points, guaranteeing alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table's `is_change` column, which indicates stimulus presentations where the image identity changed from the previous presentation.

ii.
```python
def get_image_change_at_times(stim_presentations, bin_centers):
    result = np.zeros(len(bin_centers), dtype=int)
    changes = stim_presentations[stim_presentations['is_change'] == True]
    change_times = changes['start_time'].values
    for ct in change_times:
        mask = (bin_centers >= ct) & (bin_centers < ct + COMMON_BIN_SIZE * 2)
        result[mask] = 1
    return result
```

iii. The agent used the `is_change` flag from stimulus presentations to identify when image changes occurred, marking a window of approximately 2 time bins (~182ms) as the change period.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary indicator is created where bins within `COMMON_BIN_SIZE * 2` (~182ms) after each change event are marked as 1, all others as 0. This is computed from `is_change == True` rows in the stimulus presentations table.

ii. See 4-a above.

iii. The 2-bin window was chosen to ensure at least one time bin captures the change event given the ~91ms bin size.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change), so no thresholding is applied. The output values are `['no_change', 'change']`.

ii.
```python
output_values = [
    # ...
    ['no_change', 'change'],  # image change values
    # ...
]
```

iii. N/A - binary by definition.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed at the same resampled bin centers as the neural data, ensuring alignment.

ii. See 4-a code - uses the same `bin_centers` as neural data.

iii. Same alignment approach as image identity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides speed and timestamps from the running wheel encoder.

ii.
```python
running = dataset.running_speed
running_ts = running['timestamps'].values
running_speed = running['speed'].values
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the resampled bin centers, then discretized into 5 percentile-based bins computed globally across all experiments. The discretization uses `np.digitize` with edges set to `-inf` and `+inf` at the boundaries.

ii.
```python
run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
# ...
def discretize_continuous(all_values, n_bins=5):
    valid = all_values[~np.isnan(all_values)]
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges

def apply_discretization(values, edges):
    result = np.digitize(values, edges[1:-1])
    result = np.clip(result, 0, len(edges) - 2)
    return result
```

iii. Linear interpolation resamples to the common time bins. Percentile-based binning ensures balanced class counts. Global computation ensures consistent bin edges across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins (0-20th, 20-40th, 40-60th, 60-80th, 80-100th percentile). Bin edges are set to `-inf` and `+inf` at the extremes to capture all values.

ii. See 5-b code above.

iii. The 5-bin discretization follows the instructions. The `-inf`/`+inf` boundary edges ensure no values fall outside the bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same resampled bin centers as the neural data.

ii.
```python
run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
```

iii. Using the same `bin_centers` for both neural and running speed data ensures temporal alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, computed as the mean of `pupil_width` and `pupil_height`. Likely blink frames are set to NaN before interpolation.

ii.
```python
eye = dataset.eye_tracking
pupil_diam = ((eye['pupil_width'].values + eye['pupil_height'].values) / 2.0)
likely_blink = eye['likely_blink'].values
pupil_diam[likely_blink] = np.nan
```

iii. The agent averaged width and height to get a more robust diameter estimate. Blink frames were excluded to prevent contamination of the signal.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter (after blink removal) is linearly interpolated to the resampled bin centers, with NaN values filled with the global median pupil value before discretization. Discretization uses 5 percentile-based bins computed globally.

ii.
```python
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
# ...
pupil_fill_value = np.nanmedian(all_pupil_flat)
# ...
pupil_raw = out['pupil_diam_raw'].copy()
nan_mask = np.isnan(pupil_raw)
pupil_raw[nan_mask] = pupil_fill_value
pupil_disc = apply_discretization(pupil_raw, pupil_edges)
```

iii. Same approach as running speed. NaN filling with median ensures missing data gets a neutral bin assignment rather than an edge bin.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal percentile bins with `-inf`/`+inf` at the boundaries. NaN values are filled with the global median before discretization.

ii. See 6-b code above.

iii. Same percentile-based approach as running speed ensures balanced classes.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed - interpolated to the same resampled bin centers as neural data.

ii.
```python
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
```

iii. Same alignment approach as other time-varying outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
def get_trial_outcome(trial):
    if trial['hit']:
        return 0  # hit
    elif trial['miss']:
        return 1  # miss
    elif trial['false_alarm']:
        return 2  # false_alarm
    elif trial['correct_reject']:
        return 3  # correct_reject
    else:
        return -1  # unknown
```

iii. These are the SDK's canonical trial outcome labels, which are mutually exclusive for non-aborted, non-auto-rewarded trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) using a conditional chain. The code is constant across all time bins within a trial. Trials with unknown outcomes (code -1) are excluded.

ii.
```python
outcome = get_trial_outcome(trial)
if outcome == -1:
    continue
# ...
output_data[4, :] = out['trial_outcome']  # static, same value each bin
```

iii. The fixed mapping (hit=0, miss=1, false_alarm=2, correct_reject=3) matches the reference approach.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiment loading**: If loading an NWB file throws an exception, the experiment is skipped with an error message.
- **Missing running speed**: If running speed is unavailable, zeros are used.
- **Missing eye tracking**: If eye tracking is unavailable, NaN values are used (later filled with median).
- **Invalid trials**: Trials with NaN start/stop times, or stop <= start, are skipped.
- **Short trials**: Trials with fewer than `MIN_TRIAL_FRAMES=3` frames are skipped.
- **Few neurons**: Experiments with fewer than `MIN_NEURONS=5` neurons are skipped.
- **Few trials**: Experiments with fewer than 2 valid trials are skipped.
- **NaN pupil values**: Filled with global median before discretization.

ii.
```python
try:
    dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
except Exception as e:
    print(f"  ERROR loading {eid}: {e}")
    return None
# ...
if running_ts is None:
    run_at_bins = np.zeros(n_bins)
if eye_ts is None:
    pupil_at_bins = np.full(n_bins, np.nan)
# ...
pupil_raw[nan_mask] = pupil_fill_value
```

iii. The agent used defensive programming with try/except blocks and fallback values to handle various data quality issues gracefully.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading each NWB file via `BehaviorOphysExperiment.from_nwb_path()`, and (2) the `resample_to_common_bins` function which loops over every time bin for every trial.

ii. N/A

iii. NWB file loading is I/O bound. The resampling function iterates over each bin center in a Python loop, which is inherently slow.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `resample_to_common_bins` function has a Python loop over bin centers that could be vectorized using numpy operations. The `get_image_identity_at_times` function is already vectorized using `searchsorted`.

ii.
```python
for b, tc in enumerate(bin_centers):
    t_lo = tc - COMMON_BIN_SIZE / 2
    t_hi = tc + COMMON_BIN_SIZE / 2
    mask = (timestamps >= t_lo) & (timestamps < t_hi)
    if mask.sum() > 0:
        resampled[:, b] = data[:, mask].mean(axis=1)
```

iii. The per-bin loop in resampling creates O(n_bins * n_timestamps) comparisons, which could be replaced with vectorized binning operations.

## 9-c. What processing does the code repeat multiple times?

i. The code does not repeat any major processing. Each experiment is loaded and processed once, and the results are stored for later assembly.

ii. N/A

iii. The two-pass structure (process all experiments, then assemble) avoids redundant computation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores detailed session metadata (`session_info` list) and bin label strings with numeric ranges (e.g., `speed_bin0_-42.3_0.1`). The temporal resampling to ~91ms bins is unnecessary for single-plane experiments that are already at ~31Hz, and the resampling introduces averaging that loses temporal resolution. Additionally, a `pupil_fill_value` is precomputed from the global median and used to fill NaN pupil values before discretization, which adds processing that could be avoided.

ii.
```python
session_info.append({
    'ophys_experiment_id': r['ophys_experiment_id'],
    'mouse_id': r['mouse_id'],
    # ... many fields
})
```

iii. The detailed metadata is not used by the decoder but provides useful provenance information.
