# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files stored locally. It first reads an experiment metadata CSV (`ophys_experiment_table.csv`) to identify available experiments, filters to those with matching NWB files and `passive == False`, then iterates over each experiment row, loading each NWB file via `BehaviorOphysExperiment.from_nwb_path()`. Each NWB file corresponds to one ophys experiment (one imaging plane).

ii.
```python
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

# In main():
for i, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(row, sample_mode=args.sample)
```

iii. The AI justified including all active behavior experiments (202 found, 200 successfully processed) to "build a meaningful decoder with sufficient data diversity," noting that restricting to only the paper's multiscope sessions would yield only 22 experiments from 1 mouse.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by `mouse_id` from each experiment's metadata. A sorted list of unique mouse IDs forms the `subjects` list, and each session is mapped to its subject via `subject_idx`.

ii.
```python
all_subjects = sorted(set(r['mouse_id'] for r in results))
# ...
subject_idx.append(all_subjects.index(r['mouse_id']))
```

iii. The AI extracted mouse_id from `dataset.metadata['mouse_id']` for each NWB file. 38 unique mice were identified.

## 1-c. How are the data split into sessions?

i. Each ophys experiment (each NWB file / imaging plane) is treated as a separate session. For multiplane mesoscope recordings, this means a single behavioral session (one mouse doing one task at one time) is split into multiple "sessions" in the output - one per imaging plane.

ii.
```python
def process_experiment(exp_row, sample_mode=False):
    eid = exp_row['ophys_experiment_id']
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
    dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
    # ... processes as one session
```

iii. The AI's CONVERSION_NOTES.md acknowledges this produces 200 sessions from 38 mice. Mouse 457841 alone has 32 "sessions" because its multiplane recordings produce one NWB file (and thus one "session") per imaging plane. The AI did not merge planes from the same behavioral session.

## 1-d. How are the data split into trials?

i. Trials are defined using the `trials` table from the SDK (`dataset.trials`). Each trial has `start_time` and `stop_time`. The time window [start_time, stop_time) defines the trial's temporal extent. Trials with NaN start/stop times or stop <= start are excluded.

ii.
```python
trials = dataset.trials
valid_trials = trials[
    ((trials['go'] == True) | (trials['catch'] == True)) &
    (trials['aborted'] == False) &
    (trials['auto_rewarded'] == False)
]
# ...
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    t_start = trial['start_time']
    t_end = trial['stop_time']
    if np.isnan(t_start) or np.isnan(t_end) or t_end <= t_start:
        continue
```

iii. The AI followed the instructions to use Go and Catch trials while excluding Aborted and Auto-rewarded trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in multiple ways: (1) Must be Go or Catch (not Aborted or Auto-rewarded), (2) Must have valid start/stop times, (3) Must have at least MIN_TRIAL_FRAMES=3 ophys frames in the trial window, (4) Must have a known trial outcome (not -1). Additionally, entire experiments are skipped if they have fewer than MIN_NEURONS=5 neurons or fewer than 2 valid trials.

ii.
```python
MIN_TRIAL_FRAMES = 3
MIN_NEURONS = 5

# Experiment-level filter
if n_neurons < MIN_NEURONS:
    return None

# Trial-level filters
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
if frame_mask.sum() < MIN_TRIAL_FRAMES:
    continue

outcome = get_trial_outcome(trial)
if outcome == -1:
    continue

if trial_count < 2:
    return None
```

iii. The AI applied minimal quality control thresholds (5 neurons minimum, 3 frames minimum per trial) to retain as much data as possible while ensuring decoder viability.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `dff_traces` (delta-F/F normalized fluorescence traces) accessed via `dataset.dff_traces`. The AI initially planned to use `events` (detected calcium events) matching the paper, but switched to dFF after finding events were 99.7% sparse.

ii.
```python
dff_df = dataset.dff_traces
neural_data = np.array([row['dff'] for _, row in dff_df.iterrows()])  # (n_neurons, n_frames)
```

iii. From the trajectory: "Events are 99.7% sparse! That's why we get all-zero trials. DFF is continuous and more suitable for decoding." The AI acknowledged the paper used events but chose dFF for practical decoder performance.

## 2-b. How is the `neural` data processed?

i. The dFF traces are extracted from the NWB file (already processed by the Allen pipeline: motion corrected, cell segmented, ROI filtered, demixed, neuropil subtracted, baseline normalized, detrended). The AI then resamples to a common time bin size of ~91ms (1/11 Hz) by averaging frames within each bin. Data is stored as float32.

ii.
```python
COMMON_BIN_SIZE = 1.0 / 11.0  # ~91ms

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

iii. The AI chose to resample all data to 11 Hz to match the multiplane mesoscope frame rate, noting that single-plane experiments (31 Hz) would be averaged down.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied beyond what the Allen pipeline already does. The only filter is at the experiment level: experiments with fewer than 5 neurons are excluded. The AI relies on the Allen pipeline's existing cell segmentation and ROI filtering.

ii.
```python
n_neurons = neural_data.shape[0]
if n_neurons < MIN_NEURONS:
    print(f"  SKIP {eid}: only {n_neurons} neurons (min {MIN_NEURONS})")
    return None
```

iii. The CONVERSION_NOTES.md states: "dF/F traces are already processed by the Allen pipeline: motion corrected, cell segmented, ROI filtered, demixed, neuropil subtracted, baseline normalized, and detrended."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. For each trial, frames within [start_time, stop_time) are selected using `ophys_timestamps`, then resampled to common time bins starting from trial start_time.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
trial_ophys_ts = ophys_ts[frame_mask]
trial_neural_raw = neural_data[:, frame_mask]

neural_resampled, bin_centers = resample_to_common_bins(
    trial_ophys_ts, trial_neural_raw, t_start, t_end
)
```

iii. The instructions say "Temporally align based on ophys timestamp." The AI uses ophys_timestamps as the reference timebase and aligns all other data streams to it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is ~90.9 ms (1/11 seconds). Temporal rebinning IS applied: all data is resampled to this common bin size regardless of original frame rate. Single-plane experiments at 31 Hz are downsampled; multiplane experiments at 11 Hz are approximately preserved.

ii.
```python
COMMON_BIN_SIZE = 1.0 / 11.0  # ~91ms, matching multiplane frame rate
```

iii. The AI chose 11 Hz to match the multiplane mesoscope frame rate since the paper focused on multiplane data. The metadata reports `time_bin_size: 90.9 ms`.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations` table, specifically the `image_name` and `start_time` columns. The `stimulus_block_name` column is used to filter to change detection blocks.

ii.
```python
stim = dataset.stimulus_presentations
if 'stimulus_block_name' in stim.columns:
    stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)]
else:
    stim_cd = stim[stim['image_name'] != 'spontaneous']
```

iii. The AI used the stimulus_presentations table which records each image flash with its name and timing.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each time bin, the most recently presented non-omitted image is assigned. A global mapping of image names to integer indices (0–15 for 16 unique images) is constructed across all experiments. Local per-experiment indices are remapped to global indices.

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

iii. The AI noted: "At each time bin, the most recently presented non-omitted image is recorded. During grey screen periods (inter-stimulus intervals), the last shown image persists."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same bin_centers as the neural data, so it is inherently aligned. The bin_centers are derived from the trial start/end times at the common bin size.

ii.
```python
img_identity = get_image_identity_at_times(stim_cd, bin_centers,
                                            {name: i for i, name in enumerate(image_names)})
```

iii. Using the same bin_centers for both neural and output data ensures temporal alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the `stimulus_presentations` table and the `start_time` of those change events.

ii.
```python
def get_image_change_at_times(stim_presentations, bin_centers):
    changes = stim_presentations[stim_presentations['is_change'] == True]
    change_times = changes['start_time'].values
```

iii. The `is_change` flag in stimulus_presentations marks flashes where the image identity changed from the previous flash.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary vector is created (length = number of time bins). For each change event, time bins within 2 bin widths (~182 ms) of the change onset are set to 1; all others are 0.

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

iii. The instructions say "Have value of 1 right after a change in image identity, otherwise 0." The AI implemented this as marking bins within ~2 bin widths of the change onset.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is needed. The output values are labeled `['no_change', 'change']`.

ii.
```python
output_values = [
    all_image_names,
    ['no_change', 'change'],
    # ...
]
```

iii. Binary variable as specified in the instructions.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed at the same bin_centers as the neural data, ensuring alignment.

ii.
```python
img_change = get_image_change_at_times(stim_cd, bin_centers)
```

iii. Same alignment approach as image identity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides timestamps and speed values from the running wheel encoder.

ii.
```python
running = dataset.running_speed
running_ts = running['timestamps'].values
running_speed = running['speed'].values
```

iii. The SDK's `running_speed` property returns already-processed running speed data (low-pass filtered at 10 Hz by the Allen pipeline).

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed values are linearly interpolated to the bin_centers of each trial. Then, percentile-based discretization bin edges are computed globally across all sessions using all running speed values. Values are discretized into 5 bins using `np.digitize`.

ii.
```python
run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)

# Global discretization
all_running_flat = np.concatenate(all_running)
running_edges = discretize_continuous(all_running_flat, n_bins=5)
run_disc = apply_discretization(out['running_speed_raw'], running_edges)
```

iii. The AI interpolated running speed to match neural data timing, then discretized into 5 equal percentile bins across all data.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using percentile-based edges computed across all valid running speed values globally. The `discretize_continuous` function computes edges at 0th, 20th, 40th, 60th, 80th, and 100th percentiles, then replaces the first/last with -inf/+inf.

ii.
```python
def discretize_continuous(all_values, n_bins=5):
    valid = all_values[~np.isnan(all_values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges

def apply_discretization(values, edges):
    result = np.digitize(values, edges[1:-1])
    result = np.clip(result, 0, len(edges) - 2)
    return result
```

iii. The resulting bin edges were [-inf, -0.002, 0.353, 16.35, 33.68, inf] cm/s.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same bin_centers as neural data, ensuring temporal alignment.

ii.
```python
run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
```

iii. Linear interpolation from the running speed's native timestamps (~60 Hz) to the common bin_centers.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, specifically the `pupil_width`, `pupil_height`, and `likely_blink` columns.

ii.
```python
eye = dataset.eye_tracking
eye_ts = eye['timestamps'].values
pupil_diam = ((eye['pupil_width'].values + eye['pupil_height'].values) / 2.0)
likely_blink = eye['likely_blink'].values
pupil_diam[likely_blink] = np.nan
```

iii. The AI computed pupil diameter as the mean of `pupil_width` and `pupil_height`, which are the semi-axes of the fitted pupil ellipse.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed as `(pupil_width + pupil_height) / 2`. Frames marked as `likely_blink` are set to NaN. The values are then linearly interpolated to bin_centers (with NaN-aware interpolation). For sessions without eye tracking data, a zero-filled array with NaN values is used. NaN values remaining after interpolation are filled with the global median pupil value before discretization. Finally, values are discretized into 5 equal percentile bins.

ii.
```python
pupil_diam = ((eye['pupil_width'].values + eye['pupil_height'].values) / 2.0)
pupil_diam[likely_blink] = np.nan
# ...
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
# ...
pupil_fill_value = np.nanmedian(all_pupil_flat)
pupil_raw[nan_mask] = pupil_fill_value
pupil_disc = apply_discretization(pupil_raw, pupil_edges)
```

iii. The CONVERSION_NOTES.md states: "Computed as mean of pupil_width and pupil_height from eye tracking ellipse fits. Frames marked as likely_blink set to NaN and interpolated."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into 5 equal percentile bins globally, using the same approach as running speed. NaN values are replaced with the global median before applying discretization.

ii.
```python
pupil_edges = discretize_continuous(all_pupil_flat, n_bins=5)
# NaN handling:
pupil_raw[nan_mask] = pupil_fill_value  # global median
pupil_disc = apply_discretization(pupil_raw, pupil_edges)
```

iii. Bin edges were [-inf, 35.47, 40.32, 44.75, 50.74, inf] pixels.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated from its native timestamps (~30 Hz) to the same bin_centers as neural data.

ii.
```python
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
```

iii. Same interpolation approach as running speed.

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

iii. The four outcome categories directly correspond to the Go/no-go task structure: hits and misses for Go trials, false alarms and correct rejects for Catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The trial outcome is determined by checking boolean flags in priority order (hit > miss > false_alarm > correct_reject). The resulting integer (0-3) is static per trial but broadcast to all time bins in the output array.

ii.
```python
output_data[4, :] = out['trial_outcome']  # trial outcome (static, same value each bin)
```

iii. The instructions specify trial outcome as "Static per-trial," so the AI broadcasts the single outcome value across all time bins.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing data scenarios are handled:
- **Missing eye tracking**: Sessions without eye tracking use NaN-filled arrays, which are later filled with the global median pupil value.
- **Missing running speed**: Sessions without running speed use zero-filled arrays.
- **Blink artifacts**: Frames with `likely_blink == True` are set to NaN and handled by the interpolation function.
- **Missing trial outcomes**: Trials with no outcome flag set (returning -1) are skipped.
- **Invalid trial times**: Trials with NaN start/stop times or stop <= start are skipped.
- **Too few neurons/trials**: Experiments with <5 neurons or <2 valid trials are skipped.

ii.
```python
# Missing eye tracking
if eye_ts is None:
    pupil_at_bins = np.full(n_bins, np.nan)
# Missing running speed
if running_ts is None:
    run_at_bins = np.zeros(n_bins)
# NaN filling for pupil
pupil_fill_value = np.nanmedian(all_pupil_flat)
pupil_raw[nan_mask] = pupil_fill_value
# Blinks
pupil_diam[likely_blink] = np.nan
```

iii. The AI documented that 3 sessions had eye tracking errors and were filled with the global median.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading NWB files via `BehaviorOphysExperiment.from_nwb_path()` for each of 200+ experiments, (2) The per-trial resampling loop in `resample_to_common_bins` which iterates over every bin for every trial, (3) Extracting dff_traces using `iterrows()` on the DataFrame.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
# ...
neural_data = np.array([row['dff'] for _, row in dff_df.iterrows()])
# ...
for b, tc in enumerate(bin_centers):  # iterates over every bin
    mask = (timestamps >= t_lo) & (timestamps < t_hi)
```

iii. Each NWB file must be opened and parsed, which involves disk I/O and HDF5 reading. The resampling loop runs per-bin per-trial per-experiment.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could be vectorized is the bin-by-bin resampling in `resample_to_common_bins`. Instead of iterating over each bin center, the entire binning operation could use vectorized `np.digitize` or `np.searchsorted` to assign timestamps to bins and then compute means. The `get_image_change_at_times` function also loops over change times when a vectorized approach with `np.searchsorted` would suffice.

ii.
```python
# Could be vectorized:
for b, tc in enumerate(bin_centers):
    t_lo = tc - COMMON_BIN_SIZE / 2
    t_hi = tc + COMMON_BIN_SIZE / 2
    mask = (timestamps >= t_lo) & (timestamps < t_hi)
    if mask.sum() > 0:
        resampled[:, b] = data[:, mask].mean(axis=1)

# Also could be vectorized:
for ct in change_times:
    mask = (bin_centers >= ct) & (bin_centers < ct + COMMON_BIN_SIZE * 2)
    result[mask] = 1
```

iii. The AI partially vectorized some functions (e.g., `get_image_identity_at_times` uses `searchsorted`) but left the core resampling loop as a Python for-loop.

## 9-c. What processing does the code repeat multiple times?

i. The code loads and processes each experiment independently, meaning shared data across multiplane experiments (same behavioral session) is read and processed multiple times. For mouse 457841, the same behavioral data (running speed, pupil, stimulus presentations, trials) is loaded 32 times (once per imaging plane). The global discretization also requires two passes: first to collect all raw values, then to apply discretization.

ii.
```python
# Each experiment independently loads behavioral data:
for i, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(row, sample_mode=args.sample)
    # This reloads running_speed, eye_tracking, stimulus_presentations, trials
    # for every plane in a multiplane session
```

iii. The AI did not implement any caching or deduplication for shared behavioral data across imaging planes of the same session.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code resamples neural data to a common 11 Hz bin size, which involves averaging for 31 Hz single-plane experiments. This downsampling loses temporal resolution that the native ophys timestamps already provide. Additionally, the code computes and stores bin labels with exact percentile edge values in the output_values, which is cosmetic and not used by the decoder. The code also computes detailed per-session metadata (session_info) that is stored but not used by the decoder. The interpolation of behavioral data (running speed, pupil diameter) to bin_centers is an extra step beyond simply finding the nearest ophys-aligned value.

ii.
```python
# Resampling to common bins (lossy for 31Hz data)
COMMON_BIN_SIZE = 1.0 / 11.0

# Detailed bin labels (cosmetic)
running_bin_labels.append(f"bin{i}_{lo:.1f}_{hi:.1f}")

# Session info metadata (unused by decoder)
session_info.append({...})
```

iii. The instructions state "Temporally align based on ophys timestamp," which could be interpreted as using the native ophys timestamps directly without rebinning. The resampling adds processing overhead and loses precision.
