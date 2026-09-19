# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a local metadata CSV (`ophys_experiment_table.csv`), scans the NWB directory for available files, filters to experiments that both have an NWB file and are marked `passive == False`, and then loads each experiment directly from disk with `BehaviorOphysExperiment.from_nwb_path()`. It does not use the SDK project cache or filter to `project_code == 'VisualBehavior'`.

ii. 
```python
def load_experiment_table():
    exp = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    ...
    exp = exp[exp['ophys_experiment_id'].isin(nwb_ids)]
    exp = exp[exp['passive'] == False]
    return exp
...
dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
```

iii. In trajectory step 38, the agent said its key decision was to use "ALL active behavior experiments (both VisualBehavior + VisualBehaviorMultiscope)". The code implements that local-NWB, active-only strategy rather than the SDK cache strategy.

## 1-b. How are the data split into subjects?

i. Subjects are defined as unique `mouse_id` values collected from the processed experiment results, then sorted.

ii. 
```python
all_subjects = sorted(set(r['mouse_id'] for r in results))
...
subject_idx.append(all_subjects.index(r['mouse_id']))
```

iii. The trajectory does not dwell on this choice, but the agent consistently treated `mouse_id` as the subject identifier in logs and output summaries.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` as one session. It processes experiment-table rows independently and appends one entry to `results` per experiment; it does not group multiple experiments by `ophys_session_id`.

ii. 
```python
for i, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(row, sample_mode=args.sample)
    if result is not None:
        results.append(result)
...
return {
    'ophys_experiment_id': eid,
    ...
}
```

iii. In step 38, the agent explicitly chose to use all active behavior experiments. Nothing in the trajectory indicates an attempt to reconstruct multi-experiment behavioral sessions, so the code follows an experiment-level notion of "session".

## 1-d. How are the data split into trials?

i. Trials are taken from `dataset.trials`. For each valid row, the AI uses `start_time` and `stop_time` to form a time window, collects ophys frames in that window, and keeps the trial if enough frames remain after resampling.

ii. 
```python
trials = dataset.trials
...
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    t_start = trial['start_time']
    t_end = trial['stop_time']
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
    if frame_mask.sum() < MIN_TRIAL_FRAMES:
        continue
```

iii. Step 38 says the agent would use "Go + Catch only" trials. The rest of the trajectory is consistent with using the Allen trial table as the source of trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `go` or `catch`, excluding `aborted` and `auto_rewarded`. The AI also drops trials with invalid start/end times, trials with fewer than 3 ophys frames, trials whose resampled output has fewer than 3 bins, and trials whose outcome does not map to one of four canonical labels. Entire experiments are skipped if fewer than 2 trials survive.

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

iii. Step 38 lists the intended trial filter as "Go + Catch only (exclude Aborted + Auto-rewarded)". The later checks are implementation-level safeguards added in code rather than explicitly justified in the trajectory.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data comes from `dataset.dff_traces`, specifically each row's `dff` array.

ii. 
```python
dff_df = dataset.dff_traces
neural_data = np.array([row['dff'] for _, row in dff_df.iterrows()])
```

iii. The trajectory shows an explicit change of mind: in step 38 the agent first intended to use events, but in steps 45 and 48 it switched to `dff_traces` after observing that events were 99.7% zero and too sparse for decoding.

## 2-b. How is the `neural` data processed?

i. The AI keeps one experiment's neurons together, slices each trial's ophys window, and resamples that trial-level neural matrix into a common 11 Hz grid by averaging within bins and falling back to nearest-neighbor when a bin is empty.

ii. 
```python
COMMON_BIN_SIZE = 1.0 / 11.0
...
trial_neural_raw = neural_data[:, frame_mask]
neural_resampled, bin_centers = resample_to_common_bins(
    trial_ophys_ts, trial_neural_raw, t_start, t_end
)
...
if mask.sum() > 0:
    resampled[:, b] = data[:, mask].mean(axis=1)
else:
    idx = np.argmin(np.abs(timestamps - tc))
    resampled[:, b] = data[:, idx]
```

iii. Step 38 states the agent's key choice to "Resample to common bin size (~91ms / 11Hz)". After switching from events to dF/F, the same resampling strategy remained in place.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no per-neuron quality filter, but the AI excludes entire experiments with fewer than 5 neurons and excludes trials with too few frames/bins.

ii. 
```python
MIN_NEURONS = 5
...
n_neurons = neural_data.shape[0]
if n_neurons < MIN_NEURONS:
    return None
...
if frame_mask.sum() < MIN_TRIAL_FRAMES:
    continue
if neural_resampled is None or len(bin_centers) < MIN_TRIAL_FRAMES:
    continue
```

iii. The trajectory does not provide a scientific justification for `MIN_NEURONS` or `MIN_TRIAL_FRAMES`; these appear to be pragmatic dataset-validity guards added during implementation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start. For each trial, the AI uses all ophys samples with timestamps in `[start_time, stop_time)`, then creates evenly spaced bin centers beginning half a bin after `start_time`.

ii. 
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
trial_ophys_ts = ophys_ts[frame_mask]
...
bin_centers = np.arange(trial_start + COMMON_BIN_SIZE / 2, trial_end, COMMON_BIN_SIZE)
...
'temporal_alignment_event': 'Trial start time (onset of first stimulus in trial)',
```

iii. The trajectory emphasizes ophys-time alignment and common resampling rather than alignment to `change_time`. The metadata string makes the intended alignment event explicit.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed bin size of `1/11` seconds, about 90.9 ms. Yes: the AI explicitly rebins/resamples neural and behavioral signals to that common grid.

ii. 
```python
COMMON_BIN_SIZE = 1.0 / 11.0  # ~91ms, matching multiplane frame rate
...
'time_bin_size': COMMON_BIN_SIZE * 1000,
```

iii. Step 38 says the agent chose to resample to "~91ms / 11Hz" to match the paper's multiplane rate. That rationale drove the fixed-bin design.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `dataset.stimulus_presentations`, mainly `image_name` and `start_time`, after filtering to the change-detection block and excluding `omitted` stimuli.

ii. 
```python
stim = dataset.stimulus_presentations
if 'stimulus_block_name' in stim.columns:
    stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)]
else:
    stim_cd = stim[stim['image_name'] != 'spontaneous']
...
stim = stim_presentations[stim_presentations['image_name'] != 'omitted'].sort_values('start_time')
stim_times = stim['start_time'].values
```

iii. Step 38 lists image identity as one of the required outputs, and the code shows the agent chose to derive it from the full stimulus presentation stream rather than the trial table's initial/change image fields.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI builds a per-experiment image vocabulary, maps each non-omitted presented image to a local integer, assigns each time bin the most recent presented image, then later remaps those local codes into a global image-code list assembled across all experiments.

ii. 
```python
image_names = sorted(stim_cd[
    (stim_cd['image_name'] != 'omitted') &
    (stim_cd['image_name'].notna())
]['image_name'].unique().tolist())
...
indices = np.searchsorted(stim_times, bin_centers, side='right') - 1
result = np.where(indices >= 0, stim_idx[np.clip(indices, 0, len(stim_idx) - 1)], 0)
...
all_image_names = sorted(all_image_names)
global_img_to_idx = {name: i for i, name in enumerate(all_image_names)}
...
local_to_global_arr = np.array([global_img_to_idx[name] for name in r['image_names']])
img_id_global = local_to_global_arr[out['image_identity']]
```

iii. The trajectory around steps 92, 94, and 135 shows the agent treated this as a performance-sensitive part of the pipeline and optimized it by vectorizing the image lookup and later the local-to-global remapping.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated on the same `bin_centers` used for resampled neural data, so each neural time bin gets one categorical image label.

ii. 
```python
neural_resampled, bin_centers = resample_to_common_bins(...)
...
img_identity = get_image_identity_at_times(stim_cd, bin_centers,
                                            {name: i for i, name in enumerate(image_names)})
```

iii. The trajectory repeatedly frames the conversion around a shared common-bin timebase, so the image labels were intentionally built on the same bins as neural activity.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `dataset.stimulus_presentations`, specifically rows where `is_change == True`, using their `start_time`.

ii. 
```python
changes = stim_presentations[stim_presentations['is_change'] == True]
...
change_times = changes['start_time'].values
```

iii. Step 38 identifies image change as a required output. The implementation shows the agent decided to use stimulus-presentation change annotations instead of the trial table's `change_time` plus `go` logic.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI creates a binary vector over the common bin centers. For every stimulus-presentation change time, it marks bins from `ct` to `ct + 2 * COMMON_BIN_SIZE` as 1 and leaves the rest at 0.

ii. 
```python
result = np.zeros(len(bin_centers), dtype=int)
...
for ct in change_times:
    mask = (bin_centers >= ct) & (bin_centers < ct + COMMON_BIN_SIZE * 2)
    result[mask] = 1
```

iii. The trajectory around steps 92-96 shows the agent regarded this as part of the common-bin representation and optimized the function for speed, but it did not provide any paper-based justification for the 2-bin window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is directly represented as a binary categorical output: 0 for `no_change`, 1 for `change`.

ii. 
```python
output_values = [
    all_image_names,
    ['no_change', 'change'],
    ...
]
...
output_data[1, :] = out['image_change']
```

iii. The trajectory consistently refers to image change as a binary output variable; the code implements that without any additional thresholding step.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed on the same `bin_centers` as the neural data, so its binary labels are time-locked to the rebinned neural matrices.

ii. 
```python
neural_resampled, bin_centers = resample_to_common_bins(...)
...
img_change = get_image_change_at_times(stim_cd, bin_centers)
...
output_data[1, :] = out['image_change']
```

iii. The common-bin design announced in step 38 applies here too: the agent wanted all outputs aligned to the same 11 Hz bins as neural activity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `dataset.running_speed`, using its `timestamps` and `speed` columns.

ii. 
```python
running = dataset.running_speed
running_ts = running['timestamps'].values
running_speed = running['speed'].values
```

iii. The trajectory does not question this source; it follows the standard AllenSDK running-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to the trial's common bin centers, stored as a raw continuous per-bin series, and later discretized into 5 global percentile bins across all experiments.

ii. 
```python
run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
...
all_running_flat = np.concatenate(all_running)
running_edges = discretize_continuous(all_running_flat, n_bins=5)
...
run_disc = apply_discretization(out['running_speed_raw'], running_edges)
```

iii. Step 38 lists "running speed (5 bins)" as a key output, and the general resampling rationale in that step explains why the agent interpolated it onto the shared 11 Hz timebase first.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The AI computes global percentile edges from all non-NaN running-speed samples and applies `np.digitize` to obtain 5 discrete bins.

ii. 
```python
def discretize_continuous(all_values, n_bins=5):
    valid = all_values[~np.isnan(all_values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
...
result = np.digitize(values, edges[1:-1])
result = np.clip(result, 0, len(edges) - 2)
```

iii. Step 38 explicitly mentions "5 bins" for running speed. The code implements equal-percentile binning globally rather than per session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is evaluated at the same `bin_centers` as resampled neural activity, so both share the same number of timepoints per trial.

ii. 
```python
run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
...
output_data[2, :] = run_disc
...
assert out.shape[1] == neural_all[si][ti].shape[1]
```

iii. The common-bin architecture from step 38 is the justification: the agent wanted all outputs directly aligned to the neural matrix columns.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using `pupil_width`, `pupil_height`, `timestamps`, and `likely_blink`.

ii. 
```python
eye = dataset.eye_tracking
eye_ts = eye['timestamps'].values
pupil_diam = ((eye['pupil_width'].values + eye['pupil_height'].values) / 2.0)
likely_blink = eye['likely_blink'].values
pupil_diam[likely_blink] = np.nan
```

iii. Step 65 notes that "running speed and pupil diameter vary over time" in the sample plots, but the trajectory does not otherwise justify the specific choice to average width and height; that choice is only visible in code.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI averages pupil width and height into one scalar, blanks blink frames with `NaN`, linearly interpolates valid samples to the trial's common bin centers, computes global percentile edges, and later fills remaining NaNs with the global median pupil value before discretization.

ii. 
```python
pupil_diam = ((eye['pupil_width'].values + eye['pupil_height'].values) / 2.0)
pupil_diam[likely_blink] = np.nan
...
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
...
pupil_edges = discretize_continuous(all_pupil_flat, n_bins=5)
...
pupil_fill_value = np.nanmedian(all_pupil_flat)
...
pupil_raw = out['pupil_diam_raw'].copy()
nan_mask = np.isnan(pupil_raw)
pupil_raw[nan_mask] = pupil_fill_value
pupil_disc = apply_discretization(pupil_raw, pupil_edges)
```

iii. The trajectory mainly justifies this at the level of the shared-bin representation and performance tuning. Step 132 specifically mentions precomputing `np.nanmedian(all_pupil_flat)` after it became a bottleneck.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The AI discretizes pupil values into 5 global percentile bins using edges from all pooled pupil samples, after replacing per-trial NaNs with the global median.

ii. 
```python
pupil_edges = discretize_continuous(all_pupil_flat, n_bins=5)
...
pupil_raw[nan_mask] = pupil_fill_value
pupil_disc = apply_discretization(pupil_raw, pupil_edges)
```

iii. Step 38 lists "pupil diameter (5 bins)" as a key output. The median-fill behavior is an implementation detail rather than something explicitly justified in the trajectory.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same `bin_centers` as the resampled neural data, then stored as one categorical value per neural time bin.

ii. 
```python
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
...
output_data[3, :] = pupil_disc
```

iii. As with running speed and image labels, the trajectory presents shared common-bin alignment as the organizing principle for all outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table boolean fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

iii. Trial outcome appears in step 38 as one of the required outputs. The code follows the standard Allen trial-outcome booleans.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps those four outcome booleans to integer codes 0-3 and writes the chosen code into every time bin of that trial.

ii. 
```python
outcome = get_trial_outcome(trial)
...
output_data[4, :] = out['trial_outcome']
...
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
```

iii. The trajectory treats trial outcome as a static per-trial variable, so repeating the same label across bins is consistent with the agent's chosen output format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI skips experiments that fail to load, experiments with too few neurons, and experiments with fewer than 2 surviving trials. It skips trials with invalid times, too few frames, too few resampled bins, or unmapped outcomes. Missing running data defaults to zeros. Missing eye data defaults to all-NaN, which is later filled with the global median pupil value before discretization. Interpolation uses extrapolation rather than leaving edge values missing.

ii. 
```python
try:
    dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
except Exception as e:
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
if running_ts is not None and running_speed is not None:
    run_at_bins = interpolate_to_bins(...)
else:
    run_at_bins = np.zeros(n_bins)
...
if eye_ts is not None and pupil_diam is not None:
    pupil_at_bins = interpolate_to_bins(...)
else:
    pupil_at_bins = np.full(n_bins, np.nan)
...
f = interpolate.interp1d(..., fill_value='extrapolate')
...
pupil_raw[nan_mask] = pupil_fill_value
```

iii. The trajectory mostly frames these as robustness or speed measures rather than paper-driven methodological choices. Step 132 is the clearest explicit justification, focusing on efficient handling of pupil NaNs.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading many NWB experiments from disk and the large Python/NumPy loops used for per-trial resampling, image-identity assignment, and final assembly over tens of thousands of trials.

ii. 
```python
for i, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(row, sample_mode=args.sample)
...
for b, tc in enumerate(bin_centers):
    ...
for r in results:
    for out in r['outputs']:
        ...
for trial_idx, (neural, out) in enumerate(zip(r['neural'], r['outputs'])):
    ...
```

iii. The trajectory explicitly identifies multiple bottlenecks: step 74 says processing 202 NWBs takes a while, step 94 says `get_image_identity_at_times` was very slow, and steps 123, 125, and 132 say assembly and repeated `np.nanmedian` calls were major runtime costs.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code's per-bin loop in `resample_to_common_bins`, the per-change loop in `get_image_change_at_times`, the experiment/trial loops that build `all_running` and `all_pupil`, and the final nested trial-assembly loops are vectorization targets. The agent already vectorized image-identity lookup and local-to-global image remapping.

ii. 
```python
for b, tc in enumerate(bin_centers):
    ...
for ct in change_times:
    ...
for r in results:
    for out in r['outputs']:
        all_running.append(out['running_speed_raw'])
        all_pupil.append(out['pupil_diam_raw'])
...
for trial_idx, (neural, out) in enumerate(zip(r['neural'], r['outputs'])):
    ...
```

iii. Steps 94 and 135 explicitly mention vectorizing image processing for speed. The remaining loops are visible in the final code and are the same kind of hotspots the trajectory complains about.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats several transformations: it first assigns image identities with a local experiment-specific code and later remaps them to a global code; it makes one full pass to collect all running/pupil values for discretization and another pass to apply those bins trial by trial; and it traverses the assembled data again for summary statistics and sanity checks.

ii. 
```python
image_names = sorted(... unique().tolist())
...
img_identity = get_image_identity_at_times(...)
...
all_image_names = sorted(all_image_names)
global_img_to_idx = {name: i for i, name in enumerate(all_image_names)}
...
img_id_global = local_to_global_arr[out['image_identity']]
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
for si, session in enumerate(output_all):
    for ti, out in enumerate(session):
        ...
```

iii. Steps 123, 125, 127, 132, and 135 all describe the agent investigating repeated assembly work and image remapping as performance bottlenecks.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does extra work that is not needed by downstream decoding: it computes and prints extensive summary statistics and sanity checks; it builds verbose bin-label strings and rich `session_info` metadata; it stores local image codes only to remap them later; and it computes counts like `n_go_trials`/`n_catch_trials` that are only used for metadata.

ii. 
```python
running_bin_labels.append(f"bin{i}_{lo:.1f}_{hi:.1f}")
pupil_bin_labels.append(f"bin{i}_{lo:.1f}_{hi:.1f}")
...
session_info.append({
    'ophys_experiment_id': r['ophys_experiment_id'],
    ...
    'n_go_trials': r['n_go_trials'],
    'n_catch_trials': r['n_catch_trials'],
})
...
print("DATASET SUMMARY")
...
for si, session in enumerate(neural_all):
    ...
for si, session in enumerate(output_all):
    ...
```

iii. The trajectory repeatedly talks about runtime spent in assembly and sanity-check-like loops rather than anything decoder-specific. Those costs do not improve the final neural/output tensors used downstream.
