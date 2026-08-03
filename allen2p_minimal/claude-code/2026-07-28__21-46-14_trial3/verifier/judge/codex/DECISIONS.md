# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a local CSV metadata table, filters it to experiments that both have NWB files on disk and are marked `passive == False`, then opens each NWB file one experiment at a time with `BehaviorOphysExperiment.from_nwb_path()`. It does not use `VisualBehaviorOphysProjectCache`, and it includes both `VisualBehavior` and `VisualBehaviorMultiscope` projects rather than restricting to the reference `VisualBehavior` project.

ii.
```python
def load_experiment_table():
    exp = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    ...
    exp = exp[exp['ophys_experiment_id'].isin(nwb_ids)]
    exp = exp[exp['passive'] == False]
    return exp

...

for i, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(row, sample_mode=args.sample)

...

dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
```

iii. `CONVERSION_NOTES.md` says the AI intentionally used "all active behavior ophys experiments with available NWB files" and included both single-plane and multiscope data to increase dataset size, because it believed the paper subset available on disk was too small.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values extracted from the processed experiment results.

ii.
```python
mouse_id = str(meta['mouse_id'])
...
all_subjects = sorted(set(r['mouse_id'] for r in results))
...
subject_idx.append(all_subjects.index(r['mouse_id']))
```

iii. The AI treats mouse identity as the canonical subject identifier; this is also how it reports subject counts in `CONVERSION_NOTES.md`.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` / NWB file as one session. It does not group multiple experiments that share an `ophys_session_id`.

ii.
```python
for i, (_, row) in enumerate(exp_table.iterrows()):
    print(f"\nProcessing experiment {i+1}/{len(exp_table)}: {row['ophys_experiment_id']}")
    result = process_experiment(row, sample_mode=args.sample)
    if result is not None:
        results.append(result)

...

return {
    ...
    'ophys_experiment_id': eid,
    ...
}
```

iii. `CONVERSION_NOTES.md` explicitly describes the output as 200 "sessions" built from 200 experiments, and notes that multiplane data are handled as separate experiments rather than reconstructed into multi-plane sessions.

## 1-d. How are the data split into trials?

i. Within each experiment, trials come from `dataset.trials`. For each valid row, the AI uses `start_time` and `stop_time` to select ophys frames in that interval, then resamples the selected data into common ~11 Hz bins.

ii.
```python
trials = dataset.trials
...
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    t_start = trial['start_time']
    t_end = trial['stop_time']
    ...
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
    ...
    neural_resampled, bin_centers = resample_to_common_bins(
        trial_ophys_ts, trial_neural_raw, t_start, t_end
    )
```

iii. The notes say trials are defined from `start_time` to `stop_time`, with a minimum of 3 common bins per trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `go` or `catch`, while excluding `aborted` and `auto_rewarded`. Trials are also skipped if start/stop times are invalid, if fewer than 3 ophys frames fall in the window, if resampling yields fewer than 3 bins, or if the outcome flags do not map to one of the 4 named outcomes. Entire experiments are skipped if fewer than 2 valid trials remain.

ii.
```python
valid_trials = trials[
    ((trials['go'] == True) | (trials['catch'] == True)) &
    (trials['aborted'] == False) &
    (trials['auto_rewarded'] == False)
]
...
if np.isnan(t_start) or np.isnan(t_end) or t_end <= t_start:
    continue
...
if frame_mask.sum() < MIN_TRIAL_FRAMES:
    continue
...
if neural_resampled is None or len(bin_centers) < MIN_TRIAL_FRAMES:
    continue
...
if outcome == -1:
    continue
...
if trial_count < 2:
    return None
```

iii. `CONVERSION_NOTES.md` only justifies the go/catch and aborted/auto-rewarded filter directly. The minimum-frame and minimum-trial filters are code decisions added to keep the decoder input non-degenerate.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The code derives neural data from `dataset.dff_traces`, specifically each row's `dff` array. This conflicts with the module docstring, which says "detected calcium events," but matches the actual implementation and the later notes.

ii.
```python
# Get neural data (dF/F traces - continuous fluorescence measure)
try:
    dff_df = dataset.dff_traces
    neural_data = np.array([row['dff'] for _, row in dff_df.iterrows()])
```

iii. `CONVERSION_NOTES.md` says the AI deliberately chose dF/F instead of `events`, arguing that events were too sparse for its decoder setup.

## 2-b. How is the `neural` data processed?

i. The AI does not use session-level plane merging. Instead, within each experiment it keeps the experiment's dF/F matrix, cuts out each trial window, and resamples that trial's neural activity into fixed ~91 ms bins by averaging all samples in each bin and falling back to nearest-neighbor when a bin is empty.

ii.
```python
def resample_to_common_bins(timestamps, data, trial_start, trial_end):
    bin_centers = np.arange(trial_start + COMMON_BIN_SIZE / 2, trial_end, COMMON_BIN_SIZE)
    ...
    for b, tc in enumerate(bin_centers):
        ...
        if mask.sum() > 0:
            resampled[:, b] = data[:, mask].mean(axis=1)
        else:
            idx = np.argmin(np.abs(timestamps - tc))
            resampled[:, b] = data[:, idx]

...

trial_neural_raw = neural_data[:, frame_mask]
neural_resampled, bin_centers = resample_to_common_bins(
    trial_ophys_ts, trial_neural_raw, t_start, t_end
)
```

iii. The notes justify this as matching a common 11 Hz mesoscope time base across single-plane and multiscope recordings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code applies an extra neural QC threshold: experiments with fewer than 5 neurons are skipped entirely. There is no further cell-level QC.

ii.
```python
MIN_NEURONS = 5
...
n_neurons = neural_data.shape[0]
if n_neurons < MIN_NEURONS:
    print(f"  SKIP {eid}: only {n_neurons} neurons (min {MIN_NEURONS})")
    return None
```

iii. `CONVERSION_NOTES.md` calls out this `MIN_NEURONS=5` rule as a practical filter; it is not tied to a paper- or SDK-described QC step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data are aligned to trial-local common bins spanning `start_time` to `stop_time`. The bin centers begin at `trial_start + COMMON_BIN_SIZE / 2`, so alignment is to trial time, not to native frame indices.

ii.
```python
bin_centers = np.arange(trial_start + COMMON_BIN_SIZE / 2, trial_end, COMMON_BIN_SIZE)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
trial_ophys_ts = ophys_ts[frame_mask]
trial_neural_raw = neural_data[:, frame_mask]
neural_resampled, bin_centers = resample_to_common_bins(
    trial_ophys_ts, trial_neural_raw, t_start, t_end
)
```

iii. The notes say the AI wanted all outputs on one shared ~11 Hz time grid, with trial start as the implicit within-trial origin.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output time bin size is fixed at `1/11` seconds, about 90.9 ms. Yes, temporal rebinning is applied to neural and behavioral traces.

ii.
```python
COMMON_BIN_SIZE = 1.0 / 11.0  # ~91ms, matching multiplane frame rate
...
'time_bin_size': COMMON_BIN_SIZE * 1000,
```

iii. `CONVERSION_NOTES.md` says the AI chose a common 11 Hz bin size to match the mesoscope frame rate and downsample single-plane 31 Hz data onto that grid.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `dataset.stimulus_presentations`, specifically the `image_name` and `start_time` columns after filtering to change-detection stimulus blocks and excluding `omitted` rows.

ii.
```python
stim = dataset.stimulus_presentations
if 'stimulus_block_name' in stim.columns:
    stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)]
...
stim = stim_presentations[stim_presentations['image_name'] != 'omitted'].sort_values('start_time')
stim_times = stim['start_time'].values
stim_idx = np.array([image_to_idx.get(n, 0) for n in stim['image_name'].values])
```

iii. The notes justify this as tracking "the most recently presented non-omitted image" throughout the continuous flashed-image task.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each trial bin center, the AI looks up the most recent stimulus presentation at or before that bin and uses its image label. During grey intervals, the last shown image persists. Image names are first indexed locally within an experiment, then remapped to a global image vocabulary across experiments.

ii.
```python
indices = np.searchsorted(stim_times, bin_centers, side='right') - 1
result = np.where(indices >= 0, stim_idx[np.clip(indices, 0, len(stim_idx) - 1)], 0)
...
image_names = sorted(stim_cd[
    (stim_cd['image_name'] != 'omitted') &
    (stim_cd['image_name'].notna())
]['image_name'].unique().tolist())
...
local_to_global_arr = np.array([global_img_to_idx[name] for name in r['image_names']])
img_id_global = local_to_global_arr[out['image_identity']]
```

iii. `CONVERSION_NOTES.md` explicitly says the last image should persist through the grey screen because the task is a flashed image stream.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated at the same per-trial `bin_centers` used for resampled neural data, so image identity and neural activity share the same rebinned time axis.

ii.
```python
img_identity = get_image_identity_at_times(stim_cd, bin_centers,
                                            {name: i for i, name in enumerate(image_names)})
...
trial_neural.append(neural_resampled)
trial_outputs.append({
    'image_identity': img_identity,
    ...
})
```

iii. The alignment rationale is the same shared 11 Hz binning scheme described in the notes.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `dataset.stimulus_presentations`, using the `is_change` flag and each change row's `start_time`.

ii.
```python
changes = stim_presentations[stim_presentations['is_change'] == True]
...
change_times = changes['start_time'].values
```

iii. `CONVERSION_NOTES.md` says changes are identified from the `is_change` flag in the stimulus table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI initializes a zero vector for each trial, then marks bins as 1 when their center lies within two common-bin widths after any `is_change` onset found in the stimulus table.

ii.
```python
result = np.zeros(len(bin_centers), dtype=int)
...
for ct in change_times:
    mask = (bin_centers >= ct) & (bin_centers < ct + COMMON_BIN_SIZE * 2)
    result[mask] = 1
```

iii. The notes describe this as "time bins immediately after a change in image identity (within 2 bin widths of change onset)."

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary categorical output: 0 for `no_change`, 1 for `change`.

ii.
```python
output_values = [
    all_image_names,
    ['no_change', 'change'],
    ...
]
```

iii. The task asked for a binary change variable, and the notes use the same two-category interpretation.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is computed on the same per-trial rebinned `bin_centers` used for neural data.

ii.
```python
img_change = get_image_change_at_times(stim_cd, bin_centers)
...
output_data[1, :] = out['image_change']
```

iii. The AI's general alignment strategy was to express every time-varying output on the shared common-bin grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `dataset.running_speed`, using its `timestamps` and `speed` columns.

ii.
```python
running = dataset.running_speed
running_ts = running['timestamps'].values
running_speed = running['speed'].values
```

iii. The notes describe this as wheel-encoder speed already processed by the Allen pipeline.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The raw running-speed time series is linearly interpolated onto per-trial common bin centers, then all values across all trials are pooled to compute 5 percentile bins, and each trial trace is discretized with those global edges.

ii.
```python
f = interpolate.interp1d(
    timestamps[valid], values[valid],
    kind='linear', bounds_error=False, fill_value='extrapolate'
)
return f(bin_centers)
...
all_running_flat = np.concatenate(all_running)
running_edges = discretize_continuous(all_running_flat, n_bins=5)
...
run_disc = apply_discretization(out['running_speed_raw'], running_edges)
```

iii. `CONVERSION_NOTES.md` says the AI used linear interpolation and 5 equal percentile bins across all sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into 5 percentile bins using global bin edges. The stored category names are bin-label strings derived from the numeric edges.

ii.
```python
running_edges = discretize_continuous(all_running_flat, n_bins=5)
...
result = np.digitize(values, edges[1:-1])
...
output_values = [
    ...,
    [f"speed_{running_bin_labels[i]}" for i in range(5)],
    ...
]
```

iii. The notes explicitly say running speed is discretized into "5 equal percentile bins across all sessions."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same per-trial `bin_centers` used for rebinned neural data, so alignment is on the common-bin grid rather than native ophys frames.

ii.
```python
run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
...
trial_outputs.append({
    ...
    'running_speed_raw': run_at_bins,
    ...
})
```

iii. The notes justify the same shared temporal grid for all time-varying outputs.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using `timestamps`, `pupil_width`, `pupil_height`, and `likely_blink`.

ii.
```python
eye = dataset.eye_tracking
eye_ts = eye['timestamps'].values
pupil_diam = ((eye['pupil_width'].values + eye['pupil_height'].values) / 2.0)
likely_blink = eye['likely_blink'].values
pupil_diam[likely_blink] = np.nan
```

iii. The notes say the AI used the mean of width and height from ellipse fits and masked likely blinks.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI computes mean(width, height), sets blink-marked samples to `NaN`, linearly interpolates valid samples to the common bin centers, pools all trial values to build 5 global percentile bins, and later replaces remaining `NaN`s with the global median before discretization.

ii.
```python
pupil_diam = ((eye['pupil_width'].values + eye['pupil_height'].values) / 2.0)
likely_blink = eye['likely_blink'].values
pupil_diam[likely_blink] = np.nan
...
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
...
pupil_edges = discretize_continuous(all_pupil_flat, n_bins=5)
...
pupil_fill_value = np.nanmedian(all_pupil_flat)
...
pupil_raw[nan_mask] = pupil_fill_value
pupil_disc = apply_discretization(pupil_raw, pupil_edges)
```

iii. `CONVERSION_NOTES.md` says sessions without eye tracking were filled with a global median and that blink frames were interpolated through.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into 5 global percentile bins. The stored category labels encode the numeric bin ranges.

ii.
```python
pupil_edges = discretize_continuous(all_pupil_flat, n_bins=5)
...
output_values = [
    ...,
    [f"pupil_{pupil_bin_labels[i]}" for i in range(5)],
    ...
]
```

iii. The notes explicitly describe 5 equal percentile bins across all sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation to the same per-trial common `bin_centers` used for neural data.

ii.
```python
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
...
trial_outputs.append({
    ...
    'pupil_diam_raw': pupil_at_bins,
    ...
})
```

iii. This follows the code's uniform "everything to 11 Hz bins" alignment strategy.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
def get_trial_outcome(trial):
    if trial['hit']:
        return 0
    elif trial['miss']:
        return 1
    elif trial['false_alarm']:
        return 2
    elif trial['correct_reject']:
        return 3
```

iii. The notes describe these four values as the intended outcome categories for go and catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI converts the outcome booleans into a fixed 0-3 code and broadcasts that code across all time bins in the trial.

ii.
```python
outcome = get_trial_outcome(trial)
...
output_data = np.zeros((5, n_bins), dtype=np.int64)
...
output_data[4, :] = out['trial_outcome']
```

iii. `CONVERSION_NOTES.md` says trial outcome is a static per-trial variable broadcast to each bin.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles bad or missing data by skipping experiments that fail to load, skipping experiments with too few neurons or trials, skipping invalid/too-short trials, filling missing running speed with zeros if the whole stream is unavailable, filling missing pupil data with `NaN` first and then with the global pupil median before discretization, and using interpolation with extrapolation for continuous signals.

ii.
```python
except Exception as e:
    print(f"  ERROR loading {eid}: {e}")
    return None
...
if n_neurons < MIN_NEURONS:
    return None
...
if np.isnan(t_start) or np.isnan(t_end) or t_end <= t_start:
    continue
if frame_mask.sum() < MIN_TRIAL_FRAMES:
    continue
...
except Exception as e:
    running_ts = None
    running_speed = None
...
if running_ts is not None and running_speed is not None:
    run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
else:
    run_at_bins = np.zeros(n_bins)
...
if eye_ts is not None and pupil_diam is not None:
    pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
else:
    pupil_at_bins = np.full(n_bins, np.nan)
...
pupil_fill_value = np.nanmedian(all_pupil_flat)
...
pupil_raw[nan_mask] = pupil_fill_value
```

iii. The notes justify the pupil median fill as a way to keep sessions with eye-tracking failures in the dataset. The rest of the handling is pragmatic error tolerance visible in the code rather than strongly justified from the reference materials.

## 9-a. What are the most time-consuming steps of the code?

i. The code structure implies that the slowest steps are loading every NWB file into `BehaviorOphysExperiment`, iterating over every trial in every experiment, and within each trial scanning through time bins to resample neural data. The trajectory notes also mention that full conversion over ~202 experiments takes about 20 minutes.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
...
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    ...
    neural_resampled, bin_centers = resample_to_common_bins(
        trial_ophys_ts, trial_neural_raw, t_start, t_end
    )
...
for b, tc in enumerate(bin_centers):
    ...
```

iii. The trajectory memory notes say the full conversion takes about 20 minutes and explicitly call out the need to vectorize image lookup and to precompute large-array medians for performance.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is the per-bin loop in `resample_to_common_bins`, which repeatedly constructs boolean masks and averages slices one bin at a time. Smaller targets include the per-trial `iterrows()` loop and the per-change loop in `get_image_change_at_times`.

ii.
```python
for b, tc in enumerate(bin_centers):
    t_lo = tc - COMMON_BIN_SIZE / 2
    t_hi = tc + COMMON_BIN_SIZE / 2
    mask = (timestamps >= t_lo) & (timestamps < t_hi)
    if mask.sum() > 0:
        resampled[:, b] = data[:, mask].mean(axis=1)

...

for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    ...

...

for ct in change_times:
    mask = (bin_centers >= ct) & (bin_centers < ct + COMMON_BIN_SIZE * 2)
    result[mask] = 1
```

iii. The trajectory notes show the AI was already thinking about vectorization for image lookup and median computation, but it left these other loops in scalar form.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly interpolates running speed and pupil data trial-by-trial instead of once onto a session-level time base, repeatedly scans the full stimulus table for each trial when computing image identity and image change, and does a second full pass over all trials later for discretization and assembly.

ii.
```python
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    ...
    img_identity = get_image_identity_at_times(stim_cd, bin_centers, ...)
    img_change = get_image_change_at_times(stim_cd, bin_centers)
    run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
    pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)

...

for r in results:
    for out in r['outputs']:
        all_running.append(out['running_speed_raw'])
        all_pupil.append(out['pupil_diam_raw'])

...

for r in results:
    ...
    for trial_idx, (neural, out) in enumerate(zip(r['neural'], r['outputs'])):
        ...
```

iii. The AI partially acknowledges this in the trajectory notes by storing raw trial outputs first and only discretizing after global edges are known, but it does not discuss the repeated stimulus and interpolation work.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does extra work that the downstream decoder does not need: it computes local per-experiment image indices only to remap them later to global indices, builds verbose running/pupil bin-label strings for `output_values`, constructs large `session_info` metadata records, prints extensive summary statistics, and computes `n_images` without ever using it. More importantly, it rebins 31 Hz data down to 11 Hz even though the reference decoder format can consume native ophys-frame trials.

ii.
```python
image_names = sorted(stim_cd[...]['image_name'].unique().tolist())
...
local_to_global_arr = np.array([global_img_to_idx[name] for name in r['image_names']])
img_id_global = local_to_global_arr[out['image_identity']]
...
n_images = len(all_image_names)
...
running_bin_labels.append(f"bin{i}_{lo:.1f}_{hi:.1f}")
pupil_bin_labels.append(f"bin{i}_{lo:.1f}_{hi:.1f}")
...
session_info.append({
    'ophys_experiment_id': r['ophys_experiment_id'],
    'mouse_id': r['mouse_id'],
    ...
})
```

iii. The trajectory notes emphasize performance concerns, but the code still keeps several convenience or reporting steps that do not affect the decoder inputs/outputs. The rebinning decision is also unnecessary relative to the reference solution and loses information.
