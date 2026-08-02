# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files on disk using `BehaviorOphysExperiment.from_nwb_path()`, rather than via the AllenSDK S3 cache. It reads an experiment table CSV from the metadata directory, filters to experiments with available NWB files and `passive == False`, then processes each experiment individually.

ii.
```python
NWB_DIR = '/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/'
METADATA_DIR = '/app/data/visual-behavior-ophys-1.1.0/project_metadata/'

def load_experiment_table():
    exp = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_ids = set()
    for f in os.listdir(NWB_DIR):
        if f.endswith('.nwb'):
            eid = int(f.replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
            nwb_ids.add(eid)
    exp = exp[exp['ophys_experiment_id'].isin(nwb_ids)]
    exp = exp[exp['passive'] == False]
    return exp
...
dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
```

iii. The AI chose to load NWB files directly rather than using the SDK cache because the data was already available as NWB files on disk. It filters to active behavior experiments only.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `mouse_id` field from experiment metadata. All unique mouse IDs across processed experiments are collected and sorted.

ii.
```python
all_subjects = sorted(set(r['mouse_id'] for r in results))
```

iii. The mouse_id field uniquely identifies each animal.

## 1-c. How are the data split into sessions?

i. Each experiment (NWB file) is treated as an independent session. The AI does NOT group multiple imaging planes from the same ophys_session_id into a single session. Each experiment = one session in the output.

ii.
```python
for i, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(row, sample_mode=args.sample)
    if result is not None:
        results.append(result)
```

iii. The AI processes each experiment independently. For single-plane (VisualBehavior) data this is equivalent to one session = one experiment. However, for multi-plane (VisualBehaviorMultiscope) data, this results in multiple "sessions" per actual recording session, each with only the neurons from one imaging plane.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `trials` table. Each trial spans from `start_time` to `stop_time` (variable length). The AI filters to Go and Catch trials, excluding aborted and auto-rewarded trials.

ii.
```python
valid_trials = trials[
    ((trials['go'] == True) | (trials['catch'] == True)) &
    (trials['aborted'] == False) &
    (trials['auto_rewarded'] == False)
]
...
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    t_start = trial['start_time']
    t_end = trial['stop_time']
    if np.isnan(t_start) or np.isnan(t_end) or t_end <= t_start:
        continue
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
```

iii. The AI follows the instructions to include Go and Catch trials while excluding Aborted and Auto-rewarded trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be Go or Catch (not aborted, not auto-rewarded), (2) valid start/stop times, (3) at least MIN_TRIAL_FRAMES=3 time bins, (4) valid trial outcome (not -1), (5) experiments with fewer than MIN_NEURONS=5 neurons are excluded entirely, (6) experiments with fewer than 2 valid trials are skipped.

ii.
```python
n_neurons = neural_data.shape[0]
if n_neurons < MIN_NEURONS:
    print(f"  SKIP {eid}: only {n_neurons} neurons (min {MIN_NEURONS})")
    return None
...
if frame_mask.sum() < MIN_TRIAL_FRAMES:
    continue
...
outcome = get_trial_outcome(trial)
if outcome == -1:
    continue
...
if trial_count < 2:
    return None
```

iii. The MIN_NEURONS filter removes experiments with very few neurons that would be poor for decoding. The MIN_TRIAL_FRAMES filter ensures trials have enough temporal data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces` (dF/F calcium fluorescence traces) from each experiment.

ii.
```python
dff_df = dataset.dff_traces
neural_data = np.array([row['dff'] for _, row in dff_df.iterrows()])  # (n_neurons, n_frames)
```

iii. The AI notes in CONVERSION_NOTES.md: "The paper used detected calcium events (events), but these are 99.7% sparse (zero-valued) which makes them unsuitable for time-bin-based decoding. dF/F provides continuous neural activity suitable for the decoder framework."

## 2-b. How is the `neural` data processed?

i. Neural data is resampled from the native ophys frame rate to a common bin size of 1/11 seconds (~90.9ms). For single-plane experiments at 31Hz, neural data is averaged within each time bin. No other processing is applied.

ii.
```python
COMMON_BIN_SIZE = 1.0 / 11.0  # ~91ms, matching multiplane frame rate

def resample_to_common_bins(timestamps, data, trial_start, trial_end):
    bin_centers = np.arange(trial_start + COMMON_BIN_SIZE / 2, trial_end, COMMON_BIN_SIZE)
    ...
    for b, tc in enumerate(bin_centers):
        t_lo = tc - COMMON_BIN_SIZE / 2
        t_hi = tc + COMMON_BIN_SIZE / 2
        mask = (timestamps >= t_lo) & (timestamps < t_hi)
        if mask.sum() > 0:
            resampled[:, b] = data[:, mask].mean(axis=1)
        else:
            idx = np.argmin(np.abs(timestamps - tc))
            resampled[:, b] = data[:, idx]
    ...
```

iii. The AI chose to resample to a common bin size to handle the different frame rates across VisualBehavior (31 Hz) and VisualBehaviorMultiscope (11 Hz) experiments.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Experiments with fewer than MIN_NEURONS=5 neurons are excluded entirely. No per-neuron quality filtering is applied beyond what the AllenSDK pipeline already provides.

ii.
```python
MIN_NEURONS = 5
...
if n_neurons < MIN_NEURONS:
    print(f"  SKIP {eid}: only {n_neurons} neurons (min {MIN_NEURONS})")
    return None
```

iii. The AI applied a minimum neuron threshold to exclude experiments that would be poor for decoding.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time (`start_time`). Ophys frames within the trial window (start_time to stop_time) are extracted, then resampled to common time bins.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
trial_ophys_ts = ophys_ts[frame_mask]
trial_neural_raw = neural_data[:, frame_mask]
neural_resampled, bin_centers = resample_to_common_bins(
    trial_ophys_ts, trial_neural_raw, t_start, t_end
)
```

iii. The trial window from start_time to stop_time includes both pre-change stimulus flashes and post-change response period.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI resamples all data to a common bin size of 1/11 seconds (~90.9 ms). This matches the multiplane mesoscope frame rate. For single-plane experiments (31 Hz), data is downsampled by averaging within bins.

ii.
```python
COMMON_BIN_SIZE = 1.0 / 11.0  # ~91ms, matching multiplane frame rate
...
'time_bin_size': COMMON_BIN_SIZE * 1000,  # in ms
```

iii. The AI chose this bin size to maintain consistency across different experiment types (single-plane at 31 Hz and multi-plane at 11 Hz).

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name` and `start_time` columns. It uses the most recently presented non-omitted image at each time bin.

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

iii. The AI uses the stimulus_presentations table rather than the trials table's initial_image_name/change_image_name columns. It assigns the most recently presented image to each time bin.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping built from all unique image names across all experiments. At each time bin, the most recent non-omitted stimulus presentation determines the image identity. A per-experiment local-to-global index remapping is applied.

ii.
```python
all_image_names = sorted(all_image_names)
global_img_to_idx = {name: i for i, name in enumerate(all_image_names)}
...
local_to_global_arr = np.array([global_img_to_idx[name] for name in r['image_names']])
img_id_global = local_to_global_arr[out['image_identity']]
```

iii. A global mapping ensures consistent image codes across all experiments. Local per-experiment indices are remapped to global indices during assembly.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same resampled time bin centers as the neural data, so they are inherently aligned.

ii.
```python
img_identity = get_image_identity_at_times(stim_cd, bin_centers,
                                            {name: i for i, name in enumerate(image_names)})
```

iii. Both neural data and image identity are evaluated at the same bin_centers array.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` flag in the `stimulus_presentations` table.

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

iii. The AI uses the `is_change` column from stimulus_presentations to identify change events.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary indicator is set to 1 for time bins within 2 bin widths (~182 ms) of each change onset. This covers the change event itself but is shorter than the reference's 750ms window.

ii.
```python
mask = (bin_centers >= ct) & (bin_centers < ct + COMMON_BIN_SIZE * 2)
result[mask] = 1
```

iii. The 2-bin window was chosen to mark the immediate change event.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary (0 or 1). No thresholding is needed.

ii. See 4-a/4-b code.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - computed at the same resampled bin centers as the neural data.

ii. See 4-a code.

iii. Both are evaluated at the same bin_centers.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, providing speed and timestamps from the running wheel encoder.

ii.
```python
running = dataset.running_speed
running_ts = running['timestamps'].values
running_speed = running['speed'].values
```

iii. The running_speed attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the resampled bin centers, then discretized into 5 equal percentile bins computed across all experiments. The bin edges have -inf and +inf as the first and last edges.

ii.
```python
def interpolate_to_bins(timestamps, values, bin_centers):
    valid = ~np.isnan(values)
    f = interpolate.interp1d(timestamps[valid], values[valid],
        kind='linear', bounds_error=False, fill_value='extrapolate')
    return f(bin_centers)
...
def discretize_continuous(all_values, n_bins=5):
    valid = all_values[~np.isnan(all_values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges
```

iii. Linear interpolation resamples running speed to the common time bins. Percentile-based binning ensures balanced class counts. Using -inf/+inf for outer edges ensures all values are captured.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Discretized into 5 bins using percentile-based edges. `np.digitize` maps values to bin indices 0-4.

ii.
```python
def apply_discretization(values, edges):
    result = np.digitize(values, edges[1:-1])
    result = np.clip(result, 0, len(edges) - 2)
    return result
```

iii. Percentile-based bins ensure roughly equal class counts.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same resampled bin centers as neural data, ensuring alignment.

ii.
```python
run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
```

iii. Both neural and running data are evaluated at the same time points.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, computed as the mean of `pupil_width` and `pupil_height`. Frames with `likely_blink == True` are set to NaN.

ii.
```python
eye = dataset.eye_tracking
pupil_diam = ((eye['pupil_width'].values + eye['pupil_height'].values) / 2.0)
likely_blink = eye['likely_blink'].values
pupil_diam[likely_blink] = np.nan
```

iii. The AI uses the average of width and height as a more robust diameter estimate. Blink frames are excluded.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. After setting blinks to NaN, pupil diameter is linearly interpolated to the resampled bin centers. Remaining NaNs are filled with the global median pupil value. Then discretized into 5 percentile-based bins across all experiments.

ii.
```python
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
...
pupil_fill_value = np.nanmedian(all_pupil_flat)
pupil_raw = out['pupil_diam_raw'].copy()
nan_mask = np.isnan(pupil_raw)
pupil_raw[nan_mask] = pupil_fill_value
pupil_disc = apply_discretization(pupil_raw, pupil_edges)
```

iii. The AI fills NaN values with the global median rather than mapping to bin 0. This places missing data in a "neutral" bin rather than the lowest bin.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: discretized into 5 bins using percentile-based edges with -inf/+inf bounds.

ii. See 5-c code (same `apply_discretization` function).

iii. Same approach as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed: interpolated to the same resampled bin centers.

ii.
```python
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
```

iii. Same alignment approach.

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

iii. The four outcome categories are the standard trial outcomes for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). Trials with no matching outcome (returns -1) are skipped. The outcome code is broadcast to all time bins within the trial.

ii.
```python
outcome = get_trial_outcome(trial)
if outcome == -1:
    continue
...
output_data[4, :] = out['trial_outcome']  # trial outcome (static, same value each bin)
```

iii. The mapping order matches the reference solution.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Failed experiment loads**: Wrapped in try/except, skipped with error message.
- **Invalid trial times**: Trials with NaN start/end times or end <= start are skipped.
- **Too few frames**: Trials with fewer than MIN_TRIAL_FRAMES=3 bins are skipped.
- **Missing running speed**: If running_speed is unavailable, zeros are used.
- **Missing eye tracking**: If eye tracking is unavailable, NaN is used (later filled with global median).
- **NaN pupil values**: Filled with global median before discretization.
- **Too few neurons**: Experiments with fewer than 5 neurons are skipped.
- **Too few trials**: Experiments with fewer than 2 valid trials are skipped.

ii.
```python
try:
    dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
except Exception as e:
    print(f"  ERROR loading {eid}: {e}")
    return None
...
if running_ts is not None and running_speed is not None:
    run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
else:
    run_at_bins = np.zeros(n_bins)
...
pupil_raw[nan_mask] = pupil_fill_value
```

iii. The AI takes a defensive approach, gracefully handling various failure modes.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via `BehaviorOphysExperiment.from_nwb_path()`, which reads large neural and behavioral data arrays from disk. The resampling step (`resample_to_common_bins`) is also computationally expensive due to its per-bin loop.

ii. N/A

iii. Each NWB file contains full-session data for all neurons plus behavioral streams. The per-bin resampling loop iterates over every bin in every trial.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `resample_to_common_bins` function iterates over each bin center individually with a Python for loop, computing a mask and mean for each bin. This could be vectorized using `np.digitize` or similar approaches.

ii.
```python
for b, tc in enumerate(bin_centers):
    t_lo = tc - COMMON_BIN_SIZE / 2
    t_hi = tc + COMMON_BIN_SIZE / 2
    mask = (timestamps >= t_lo) & (timestamps < t_hi)
    if mask.sum() > 0:
        resampled[:, b] = data[:, mask].mean(axis=1)
```

iii. This loop is called for every trial in every experiment, making it a significant performance bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. No major processing is repeated. Each experiment is loaded once and processed sequentially.

ii. N/A

iii. The AI's architecture processes each experiment once and stores results for later assembly.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes running speed bin labels with detailed edge values (e.g., `speed_bin0_-0.0_0.4`) and pupil bin labels, which are descriptive but not strictly necessary. The local-to-global image index remapping step adds complexity compared to building a global mapping upfront.

ii.
```python
running_bin_labels = []
for i in range(5):
    lo = running_edges[i] if i > 0 else valid_running.min()
    hi = running_edges[i + 1] if i < 4 else valid_running.max()
    running_bin_labels.append(f"bin{i}_{lo:.1f}_{hi:.1f}")
```

iii. These labels provide useful metadata but are not required by the decoder.
