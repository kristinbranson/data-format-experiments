# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading a CSV experiment table from the project metadata directory, filtering to experiments that have NWB files available and are active behavior (not passive). Each experiment is loaded individually via `BehaviorOphysExperiment.from_nwb_path()` from the NWB files on disk, rather than using the AllenSDK's S3 cache.

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
...
dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
```

iii. The AI chose to load directly from NWB files rather than using the SDK cache. It filtered to active behavior experiments with available NWB files (202 total). The CONVERSION_NOTES.md states this approach was taken because the data was available as NWB files on disk.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `mouse_id` field from experiment metadata. All unique mouse IDs across processed experiments are collected and sorted.

ii.
```python
all_subjects = sorted(set(r['mouse_id'] for r in results))
...
mouse_id = str(meta['mouse_id'])
```

iii. The AI used the same `mouse_id` field as the reference to identify subjects. 38 mice were found across all experiments.

## 1-c. How are the data split into sessions?

i. The AI treats each individual **experiment** (single imaging plane / NWB file) as a separate "session" in the output. It does NOT group experiments by `ophys_session_id` to reconstruct multi-plane sessions. Each experiment is iterated independently.

ii.
```python
for i, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(row, sample_mode=args.sample)
    if result is not None:
        results.append(result)
```

iii. The AI's CONVERSION_NOTES.md acknowledges that 200 sessions were produced from 38 mice. It does not discuss grouping experiments by session. This approach results in each imaging plane being a separate "session" rather than combining all planes from the same recording session.

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `dataset.trials` table. The AI filters to Go and Catch trials, excluding Aborted and Auto-rewarded trials. Each trial spans from `start_time` to `stop_time`. Trials with NaN start/end times or fewer than MIN_TRIAL_FRAMES (3) frames are skipped.

ii.
```python
valid_trials = trials[
    ((trials['go'] == True) | (trials['catch'] == True)) &
    (trials['aborted'] == False) &
    (trials['auto_rewarded'] == False)
]
...
t_start = trial['start_time']
t_end = trial['stop_time']
if np.isnan(t_start) or np.isnan(t_end) or t_end <= t_start:
    continue
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
if frame_mask.sum() < MIN_TRIAL_FRAMES:
    continue
```

iii. The AI used the standard trials table filtering approach. The explicit `go | catch` filter is equivalent to the reference's `~aborted & ~auto_rewarded` since non-aborted, non-auto-rewarded trials should be either go or catch.

## 1-e. How are trials filtered based on quality controls?

i. Trial quality filters include: (1) must be Go or Catch (not Aborted/Auto-rewarded), (2) start/stop times must be valid (not NaN, end > start), (3) at least MIN_TRIAL_FRAMES=3 ophys frames in the trial window, (4) trial outcome must be one of hit/miss/false_alarm/correct_reject (outcome != -1), (5) experiments with fewer than MIN_NEURONS=5 neurons are skipped, (6) experiments with fewer than 2 valid trials are skipped.

ii.
```python
if n_neurons < MIN_NEURONS:
    print(f"  SKIP {eid}: only {n_neurons} neurons (min {MIN_NEURONS})")
    return None
...
if outcome == -1:
    continue
...
if trial_count < 2:
    return None
```

iii. The MIN_NEURONS=5 filter is an additional quality control not present in the reference. The AI's CONVERSION_NOTES states 2 experiments were skipped due to having fewer than 5 neurons.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces` (dF/F calcium fluorescence traces), accessed via `dataset.dff_traces`.

ii.
```python
dff_df = dataset.dff_traces
neural_data = np.array([row['dff'] for _, row in dff_df.iterrows()])  # (n_neurons, n_frames)
```

iii. The AI's CONVERSION_NOTES.md states: "The paper used detected calcium events (events), but these are 99.7% sparse which makes them unsuitable for time-bin-based decoding. dF/F provides continuous neural activity suitable for the decoder framework." Both the AI and reference use dF/F.

## 2-b. How is the `neural` data processed?

i. The neural data is resampled from its native frame rate to a common bin size of 1/11 seconds (~90.9ms). For single-plane experiments (31 Hz), neural data is averaged within each time bin. The resampling is done per-trial using the `resample_to_common_bins` function which creates common time bins from trial start to trial end and averages neural activity within each bin.

ii.
```python
COMMON_BIN_SIZE = 1.0 / 11.0  # ~91ms, matching multiplane frame rate
...
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

iii. The AI chose a common bin size matching the multiplane mesoscope rate (11 Hz). The reference code does not resample and keeps the native ophys frame rate. This is a significant processing difference.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Experiments with fewer than MIN_NEURONS=5 neurons are excluded entirely. No per-neuron quality filtering is applied beyond what the Allen SDK already provides.

ii.
```python
if n_neurons < MIN_NEURONS:
    print(f"  SKIP {eid}: only {n_neurons} neurons (min {MIN_NEURONS})")
    return None
```

iii. The reference applies no additional neural data filtering. The AI adds a minimum neuron count threshold of 5.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start time. Ophys frames within the trial window (start_time to stop_time) are selected, then resampled to common time bins starting from trial start.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
trial_ophys_ts = ophys_ts[frame_mask]
trial_neural_raw = neural_data[:, frame_mask]
neural_resampled, bin_centers = resample_to_common_bins(
    trial_ophys_ts, trial_neural_raw, t_start, t_end
)
```

iii. Both AI and reference align to trial start time. The reference uses `np.searchsorted` to find frame indices, while the AI uses a boolean mask and then resamples. The AI's frame selection uses `>=` for start and `<` for end, slightly different from the reference's `searchsorted` approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning to a common bin size of 1/11 seconds (~90.9 ms), regardless of the native frame rate. For 31 Hz single-plane data, this means averaging ~3 frames per bin. For 11 Hz multiplane data, bins roughly match the native rate.

ii.
```python
COMMON_BIN_SIZE = 1.0 / 11.0  # ~91ms, matching multiplane frame rate
...
'time_bin_size': COMMON_BIN_SIZE * 1000,  # in ms
```

iii. The reference does NOT resample and keeps the native ophys frame rate (~11 Hz for multiplane). Since the reference only processes VisualBehavior (single-plane, ~31 Hz), its native rate differs. The AI chose to standardize across experiment types.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name` and `start_time` columns. The AI uses the most recently presented non-omitted image at each time bin.

ii.
```python
stim = dataset.stimulus_presentations
...
stim = stim_presentations[stim_presentations['image_name'] != 'omitted'].sort_values('start_time')
stim_times = stim['start_time'].values
stim_idx = np.array([image_to_idx.get(n, 0) for n in stim['image_name'].values])
indices = np.searchsorted(stim_times, bin_centers, side='right') - 1
result = np.where(indices >= 0, stim_idx[np.clip(indices, 0, len(stim_idx) - 1)], 0)
```

iii. The reference uses `initial_image_name` and `change_image_name` from the trials table, switching at `change_time`. The AI instead uses the full `stimulus_presentations` table to look up which image was most recently shown at each time bin.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to global integer indices across all sessions. At each time bin, the most recently presented non-omitted stimulus is found via `searchsorted`. During grey screen intervals, the last shown image persists. Omitted stimuli are excluded.

ii.
```python
all_image_names = sorted(all_image_names)
global_img_to_idx = {name: i for i, name in enumerate(all_image_names)}
...
def get_image_identity_at_times(stim_presentations, bin_centers, image_to_idx):
    stim = stim_presentations[stim_presentations['image_name'] != 'omitted'].sort_values('start_time')
    stim_times = stim['start_time'].values
    stim_idx = np.array([image_to_idx.get(n, 0) for n in stim['image_name'].values])
    indices = np.searchsorted(stim_times, bin_centers, side='right') - 1
    result = np.where(indices >= 0, stim_idx[np.clip(indices, 0, len(stim_idx) - 1)], 0)
    return result.astype(int)
```

iii. The AI builds a per-experiment local image mapping, then remaps to global indices. The reference directly uses trial-level image names without consulting the stimulus_presentations table.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same bin centers as the resampled neural data, ensuring alignment. The `bin_centers` from `resample_to_common_bins` are passed to `get_image_identity_at_times`.

ii.
```python
neural_resampled, bin_centers = resample_to_common_bins(...)
img_identity = get_image_identity_at_times(stim_cd, bin_centers, ...)
```

iii. Both AI and reference align image identity to the same time points as neural data. The AI uses resampled bin centers; the reference uses ophys frame indices directly.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table, specifically the `is_change` column and `start_time`. The AI looks for stimulus presentations where `is_change == True`.

ii.
```python
changes = stim_presentations[stim_presentations['is_change'] == True]
change_times = changes['start_time'].values
```

iii. The reference derives image change from the `change_time` and `go` columns in the trials table. The AI uses the `is_change` flag from stimulus_presentations instead. The `is_change` flag may capture changes differently than the go/catch trial logic.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary indicator is set to 1 at time bins within 2 bin widths (~182ms) of a change onset time from the stimulus presentations table. This is narrower than the reference's 750ms window.

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

iii. The reference uses a 750ms window (one flash + grey period) and restricts to go trials only. The AI uses a 2-bin (~182ms) window and uses the `is_change` flag which may include catch trial "changes" (though catch trials have `is_change=False` in the Allen data, so this may not matter in practice).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), no thresholding is needed.

ii. See 4-b.

iii. Both AI and reference produce a binary output. No additional thresholding.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed at the same resampled bin centers as the neural data.

ii.
```python
img_change = get_image_change_at_times(stim_cd, bin_centers)
```

iii. Same alignment approach as image identity - uses the resampled bin centers.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides speed and timestamps from the running wheel encoder.

ii.
```python
running = dataset.running_speed
running_ts = running['timestamps'].values
running_speed = running['speed'].values
```

iii. Both AI and reference use the same `running_speed` data source from the Allen SDK.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to the resampled bin centers, then discretized into 5 equal percentile bins computed globally across all sessions. NaN values in running speed default to bin 0 (no special handling mentioned, but `interpolate_to_bins` uses `fill_value='extrapolate'` which may avoid NaN).

ii.
```python
def interpolate_to_bins(timestamps, values, bin_centers):
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return np.full(len(bin_centers), np.nan)
    f = interpolate.interp1d(timestamps[valid], values[valid],
        kind='linear', bounds_error=False, fill_value='extrapolate')
    return f(bin_centers)
...
running_edges = discretize_continuous(all_running_flat, n_bins=5)
run_disc = apply_discretization(out['running_speed_raw'], running_edges)
```

iii. The reference also interpolates linearly and discretizes into 5 percentile bins. Key difference: the AI uses `fill_value='extrapolate'` (linear extrapolation beyond data bounds) instead of `fill_value=np.nan` (the reference's approach).

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 percentile-based bins using `np.digitize` with precomputed edges. The edges are modified: first edge set to -inf, last to +inf. Values are clipped to valid bin range [0, 4].

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

iii. The reference computes percentile edges without -inf/+inf modification. Both produce 5 bins (0-4). The AI's -inf/+inf edges ensure all values are captured, which is functionally similar.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same resampled bin centers as the neural data.

ii.
```python
run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
```

iii. Same alignment approach as other outputs - uses the resampled bin centers from `resample_to_common_bins`.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using the **average** of `pupil_width` and `pupil_height`. Blink frames (where `likely_blink` is True) are set to NaN before interpolation.

ii.
```python
eye = dataset.eye_tracking
eye_ts = eye['timestamps'].values
pupil_diam = ((eye['pupil_width'].values + eye['pupil_height'].values) / 2.0)
likely_blink = eye['likely_blink'].values
pupil_diam[likely_blink] = np.nan
```

iii. The reference uses only `pupil_width` for pupil diameter. The AI averages width and height, which computes a mean diameter of the pupil ellipse. This is a different measurement.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter (mean of width and height) has blink frames set to NaN, then is linearly interpolated (with extrapolation) to the resampled bin centers. For sessions without eye tracking, values are filled with NaN (then later filled with global median). Finally, discretized into 5 percentile bins globally.

ii.
```python
pupil_diam[likely_blink] = np.nan
...
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
...
pupil_fill_value = np.nanmedian(all_pupil_flat)
pupil_raw[nan_mask] = pupil_fill_value
pupil_disc = apply_discretization(pupil_raw, pupil_edges)
```

iii. The reference removes blink rows before interpolation (filters the dataframe), while the AI sets blink values to NaN and relies on `interpolate_to_bins` to skip NaN values. For NaN handling during discretization, the reference maps NaN to bin 0; the AI fills NaN with the global median pupil value before discretizing.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 percentile-based bins with edges[0]=-inf, edges[-1]=inf, using `np.digitize` and `np.clip`.

ii.
```python
pupil_edges = discretize_continuous(all_pupil_flat, n_bins=5)
pupil_disc = apply_discretization(pupil_raw, pupil_edges)
```

iii. Same approach as running speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same resampled bin centers as the neural data.

ii.
```python
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
```

iii. Same alignment approach as running speed and other outputs.

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

iii. Both AI and reference use the same four trial outcome categories with the same integer codes. The AI returns -1 for unknown outcomes and skips those trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) and broadcast to all time bins in the trial as a static per-trial variable.

ii.
```python
outcome = get_trial_outcome(trial)
if outcome == -1:
    continue
...
output_data[4, :] = out['trial_outcome']  # trial outcome (static, same value each bin)
```

iii. Same approach as reference. Both encode outcome as an integer repeated across all time bins.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Failed experiment loading**: Wrapped in try/except, skipped with error message.
- **Missing eye tracking**: If eye tracking unavailable, pupil values set to NaN, later filled with global median.
- **Missing running speed**: If unavailable, running speed set to zeros.
- **Invalid trials**: Trials with NaN start/stop times, insufficient frames, or unknown outcome are skipped.
- **Experiments with few neurons**: MIN_NEURONS=5 threshold.
- **Experiments with few trials**: Minimum 2 valid trials required.
- **NaN pupil values**: Filled with global median before discretization.

ii.
```python
except Exception as e:
    print(f"  ERROR loading {eid}: {e}")
    return None
...
if running_ts is None:
    run_at_bins = np.zeros(n_bins)
...
if eye_ts is None:
    pupil_at_bins = np.full(n_bins, np.nan)
...
pupil_raw[nan_mask] = pupil_fill_value  # global median
```

iii. The reference maps NaN to bin 0 for both running and pupil. The AI fills pupil NaN with global median and running NaN implicitly through extrapolation. The AI also fills entirely missing running speed with zeros, which is different from the reference approach.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via `BehaviorOphysExperiment.from_nwb_path()`, which reads large neural and behavioral data arrays from disk. The per-trial resampling loop (`resample_to_common_bins`) also adds overhead since it iterates over every bin for every trial.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
...
for b, tc in enumerate(bin_centers):  # per-bin loop in resample_to_common_bins
```

iii. Data loading is I/O bound and dominates runtime, similar to the reference.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `resample_to_common_bins` function contains a per-bin Python loop that creates a boolean mask and computes mean for each time bin. This could be vectorized using `np.histogram`-style binning or `scipy.interpolate.interp1d`. The per-trial loop over `valid_trials.iterrows()` also uses slow pandas iteration.

ii.
```python
for b, tc in enumerate(bin_centers):
    t_lo = tc - COMMON_BIN_SIZE / 2
    t_hi = tc + COMMON_BIN_SIZE / 2
    mask = (timestamps >= t_lo) & (timestamps < t_hi)
    if mask.sum() > 0:
        resampled[:, b] = data[:, mask].mean(axis=1)
```

iii. The per-bin loop is the main vectorization opportunity. Each iteration creates a full-size boolean mask over all timestamps, making it O(n_bins * n_timestamps).

## 9-c. What processing does the code repeat multiple times?

i. The AI processes each experiment independently, so experiments from the same session load overlapping behavioral data (running speed, pupil, trials table) multiple times from the same recording session. Also, `get_image_identity_at_times` and `get_image_change_at_times` each sort the stimulus presentations table redundantly per trial.

ii.
```python
# Each experiment loads its own running/pupil/trials even if from the same session
running = dataset.running_speed
eye = dataset.eye_tracking
trials = dataset.trials
```

iii. Since each experiment is processed independently, behavioral data shared across imaging planes in the same session is loaded and processed multiple times.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI resamples neural data to a common 11 Hz bin size for all experiments, including 31 Hz single-plane experiments. This loses temporal resolution unnecessarily. The stimulus_presentations table is loaded and filtered for every experiment to compute image identity, when simpler trial-level information (initial/change image) could suffice. The code also computes descriptive bin labels (`running_bin_labels`, `pupil_bin_labels`) with actual edge values that add complexity without clear benefit.

ii.
```python
running_bin_labels = []
for i in range(5):
    lo = running_edges[i] if i > 0 else valid_running.min()
    hi = running_edges[i + 1] if i < 4 else valid_running.max()
    running_bin_labels.append(f"bin{i}_{lo:.1f}_{hi:.1f}")
```

iii. The descriptive bin labels are cosmetic; the stimulus_presentations approach is more complex than needed.
