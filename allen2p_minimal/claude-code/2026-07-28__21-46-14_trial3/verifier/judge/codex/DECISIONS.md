# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a local CSV experiment table from the Allen metadata directory, scans the NWB directory for available experiment files, filters to experiments that both have an NWB file and are not passive, and then loads each experiment independently with `BehaviorOphysExperiment.from_nwb_path`. It does not use the Allen SDK project cache or restrict to `project_code == "VisualBehavior"`.

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

iii. `CONVERSION_NOTES.md` says the dataset source is the Visual Behavior 2P dataset accessed through NWB files and the AllenSDK, and explicitly justifies including “all active behavior ophys experiments” rather than matching the narrower reference subset.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values taken from the processed experiments, then stored as strings in sorted order.

ii.
```python
all_subjects = sorted(set(r['mouse_id'] for r in results))
...
'subjects': all_subjects,
'subject_idx': np.array(subject_idx, dtype=np.int64),
```

iii. The justification is implicit in the metadata design: `mouse_id` is treated as the animal identifier. The notes summarize the resulting dataset as 38 mice.

## 1-c. How are the data split into sessions?

i. The AI treats each individual `ophys_experiment_id` / NWB file as one session. It does not group experiments by `ophys_session_id`.

ii.
```python
for i, (_, row) in enumerate(exp_table.iterrows()):
    print(f"\nProcessing experiment {i+1}/{len(exp_table)}: {row['ophys_experiment_id']}")
    result = process_experiment(row, sample_mode=args.sample)
    if result is not None:
        results.append(result)
...
session_info.append({
    'ophys_experiment_id': r['ophys_experiment_id'],
    ...
})
```

iii. `CONVERSION_NOTES.md` justifies this by choosing “all active behavior ophys experiments” and describes the final dataset in terms of experiments/sessions interchangeably.

## 1-d. How are the data split into trials?

i. Trials come from `dataset.trials`. For each valid row, the AI uses `start_time` and `stop_time` to define the trial window, extracts ophys frames in that interval, and then resamples those frames to common 11 Hz bins.

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

iii. The notes state “Each trial defined by `start_time` to `stop_time` from the trials table” and justify that choice as matching the task structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only go or catch trials that are not aborted and not auto-rewarded, then drops trials with invalid or too-short time windows. It also skips whole experiments with fewer than 2 valid trials, and skips experiments with fewer than 5 neurons.

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
if len(valid_trials) < 2:
    return None
...
if n_neurons < MIN_NEURONS:
    return None
```

iii. `CONVERSION_NOTES.md` explicitly justifies including go and catch trials and excluding aborted and auto-rewarded trials. The minimum-neuron and minimum-frame filters are not justified there beyond producing a workable decoder dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final code derives neural data from `dataset.dff_traces`, specifically the per-cell `dff` arrays.

ii.
```python
dff_df = dataset.dff_traces
neural_data = np.array([row['dff'] for _, row in dff_df.iterrows()])
```

iii. The trajectory shows the AI first planned to use events, then switched after checking sparsity: “Events are 99.7% zero ... I’ll switch to `dff_traces`.” The notes repeat that rationale.

## 2-b. How is the `neural` data processed?

i. The AI keeps dF/F as the neural signal, slices it per trial, and resamples each trial to a common ~11 Hz time grid by averaging values in each bin, with nearest-neighbor fallback for empty bins. No additional normalization is applied.

ii.
```python
COMMON_BIN_SIZE = 1.0 / 11.0
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
```

iii. `CONVERSION_NOTES.md` says dF/F is already processed by the Allen pipeline and explains the extra resampling as a way to match a common 11 Hz bin size across experiments.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not filter neurons individually, but it excludes any experiment with fewer than 5 neurons.

ii.
```python
MIN_NEURONS = 5
...
n_neurons = neural_data.shape[0]
if n_neurons < MIN_NEURONS:
    print(f"  SKIP {eid}: only {n_neurons} neurons (min {MIN_NEURONS})")
    return None
```

iii. `CONVERSION_NOTES.md` reports that 2 experiments were skipped for having fewer than 5 neurons, but does not tie that threshold to the paper or reference code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start: the trial window is `[start_time, stop_time)`, and the resampled bins start at `trial_start + COMMON_BIN_SIZE / 2`.

ii.
```python
bin_centers = np.arange(trial_start + COMMON_BIN_SIZE / 2, trial_end, COMMON_BIN_SIZE)
...
t_start = trial['start_time']
t_end = trial['stop_time']
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
```

iii. The metadata states `temporal_alignment_event` is “Trial start time (onset of first stimulus in trial)”, and the notes describe trial segmentation from start to stop time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed common bin size of `1/11` seconds, about 90.9 ms. Yes, the AI rebins all neural and behavioral time series to this common grid.

ii.
```python
COMMON_BIN_SIZE = 1.0 / 11.0  # ~91ms, matching multiplane frame rate
...
'time_bin_size': COMMON_BIN_SIZE * 1000,
```

iii. `CONVERSION_NOTES.md` explicitly justifies the 11 Hz choice as matching the multiplane mesoscope frame rate and says single-plane data were averaged into that bin size.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `dataset.stimulus_presentations`, using the `image_name` and `start_time` columns within the change-detection block.

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
stim_idx = np.array([image_to_idx.get(n, 0) for n in stim['image_name'].values])
```

iii. The notes justify this by saying image identity should reflect “the most recently presented non-omitted image,” including grey-screen periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI filters out omitted stimuli, sorts presentations by time, assigns each bin the most recent image shown, uses local image indices within each experiment, and later remaps them to a global image vocabulary across experiments.

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

iii. `CONVERSION_NOTES.md` explains that during grey periods the “last shown image persists,” and that the task happens within change-detection blocks where images keep flashing.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on exactly the same `bin_centers` used for the resampled neural trial, so it is aligned per common 11 Hz trial bin rather than per original ophys frame.

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

iii. The code comments and notes both frame this as a shared time-base alignment choice after rebinning.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations['is_change']` and the corresponding `start_time` values.

ii.
```python
changes = stim_presentations[stim_presentations['is_change'] == True]
...
change_times = changes['start_time'].values
```

iii. `CONVERSION_NOTES.md` says changes are identified from the `is_change` flag in the stimulus table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For every detected change time, the AI marks bins as 1 from the change onset through the next two common bins; otherwise the value is 0.

ii.
```python
result = np.zeros(len(bin_centers), dtype=int)
...
for ct in change_times:
    mask = (bin_centers >= ct) & (bin_centers < ct + COMMON_BIN_SIZE * 2)
    result[mask] = 1
```

iii. The notes justify this as “time bins immediately after a change in image identity (within 2 bin widths of change onset).”

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary category: bins inside the short post-change window are labeled 1 and all other bins are 0.

ii.
```python
result = np.zeros(len(bin_centers), dtype=int)
...
result[mask] = 1
...
output_values = [
    all_image_names,
    ['no_change', 'change'],
    ...
]
```

iii. The justification is the same as in the notes: image change is treated as a brief change-event indicator rather than a sustained post-change state.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is defined on the trial’s common `bin_centers`, so it is aligned to the resampled neural bins.

ii.
```python
img_change = get_image_change_at_times(stim_cd, bin_centers)
...
output_data[1, :] = out['image_change']
```

iii. The code treats neural, image identity, and image change as sharing one rebinned trial time base.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, specifically its `timestamps` and `speed` columns.

ii.
```python
running = dataset.running_speed
running_ts = running['timestamps'].values
running_speed = running['speed'].values
```

iii. The notes describe this as wheel-encoder running speed already processed by the Allen pipeline.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed onto each trial’s common bin centers and later discretizes all values into five global percentile bins.

ii.
```python
f = interpolate.interp1d(
    timestamps[valid], values[valid],
    kind='linear', bounds_error=False, fill_value='extrapolate'
)
...
run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
...
running_edges = discretize_continuous(all_running_flat, n_bins=5)
```

iii. `CONVERSION_NOTES.md` explicitly says behavior data were linearly interpolated to ophys time bins and then discretized into 5 equal percentile bins across all sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded by global percentile edges into 5 discrete bins using `np.digitize`.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
edges = np.percentile(valid, percentiles)
edges[0] = -np.inf
edges[-1] = np.inf
...
result = np.digitize(values, edges[1:-1])
result = np.clip(result, 0, len(edges) - 2)
```

iii. The notes justify this as equal-percentile discretization for decoder outputs.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation to the same per-trial `bin_centers` used for resampled neural activity.

ii.
```python
run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
...
output_data[2, :] = run_disc
```

iii. The notes explicitly state that behavior data were interpolated to the ophys time bins chosen for the converted dataset.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking` using `pupil_width`, `pupil_height`, `timestamps`, and `likely_blink`.

ii.
```python
eye = dataset.eye_tracking
eye_ts = eye['timestamps'].values
pupil_diam = ((eye['pupil_width'].values + eye['pupil_height'].values) / 2.0)
likely_blink = eye['likely_blink'].values
pupil_diam[likely_blink] = np.nan
```

iii. `CONVERSION_NOTES.md` says the AI computed pupil diameter as the mean of width and height from the eye-tracking ellipse fits and set likely blinks to NaN.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI averages width and height, masks likely blinks as NaN, interpolates to the common trial bins, and later replaces NaNs with the global median before discretization.

ii.
```python
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
...
pupil_fill_value = np.nanmedian(all_pupil_flat)
...
pupil_raw = out['pupil_diam_raw'].copy()
nan_mask = np.isnan(pupil_raw)
pupil_raw[nan_mask] = pupil_fill_value
pupil_disc = apply_discretization(pupil_raw, pupil_edges)
```

iii. The notes justify the width/height averaging and mention that sessions without eye tracking were filled with the global median.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into 5 global percentile bins with `np.digitize`; missing values are first imputed with the global median so they fall into a middle bin rather than a dedicated missing bin.

ii.
```python
pupil_edges = discretize_continuous(all_pupil_flat, n_bins=5)
...
pupil_raw[nan_mask] = pupil_fill_value
pupil_disc = apply_discretization(pupil_raw, pupil_edges)
```

iii. `CONVERSION_NOTES.md` explicitly says pupil diameter used 5 equal percentile bins and that sessions without eye tracking data were filled with the global median.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same per-trial common bin centers used for resampled neural data.

ii.
```python
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
...
output_data[3, :] = pupil_disc
```

iii. The notes describe a shared 11 Hz binned time base for neural and behavioral outputs.

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

iii. The notes describe the same four categories as the trial outcome classes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four mutually exclusive flags to integer codes 0-3 and stores that code as a constant across all time bins in a trial.

ii.
```python
outcome = get_trial_outcome(trial)
...
output_data[4, :] = out['trial_outcome']
...
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
```

iii. `CONVERSION_NOTES.md` justifies this as a 4-category static per-trial output, broadcast across bins to fit the decoder’s array format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI skips experiments that fail to load or lack enough neurons/trials, skips trials with invalid timing or too few frames, fills missing running data with zeros when the whole stream is unavailable, fills missing pupil data with NaNs and later the global median, and uses extrapolating interpolation for partially missing behavioral streams.

ii.
```python
try:
    dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
except Exception as e:
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
run_at_bins = np.zeros(n_bins)
...
pupil_at_bins = np.full(n_bins, np.nan)
...
pupil_raw[nan_mask] = pupil_fill_value
```

iii. The notes explicitly call out missing eye-tracking sessions being filled with the global median and frame-level blink handling with NaNs.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading and parsing each NWB file experiment-by-experiment, plus the per-trial/per-bin resampling work. The trajectory also shows the AI identified an inner-loop `np.nanmedian` bottleneck and moved it out of the loop.

ii.
```python
for i, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(row, sample_mode=args.sample)
...
for b, tc in enumerate(bin_centers):
    ...
...
pupil_fill_value = np.nanmedian(all_pupil_flat)
```

iii. The trajectory repeatedly mentions that processing 202 NWB files takes 10-20+ minutes and later says “Found the bottleneck! `np.nanmedian(all_pupil_flat)` is called inside the inner loop.”

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still leaves several expensive loops unvectorized: the per-bin loop in `resample_to_common_bins`, the per-change loop in `get_image_change_at_times`, the per-trial loop over `valid_trials`, and the experiment loop over the entire table. The AI did vectorize image lookup and global remapping, but not these other steps.

ii.
```python
for b, tc in enumerate(bin_centers):
    ...
for ct in change_times:
    ...
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    ...
for i, (_, row) in enumerate(exp_table.iterrows()):
    ...
indices = np.searchsorted(stim_times, bin_centers, side='right') - 1
```

iii. The only explicit justification is in code comments such as “Vectorized for performance” on image functions; the trajectory does not claim the remaining loops were optimal.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly filters and sorts `stim_presentations` inside image helper calls for every trial, rebuilds the local image-name mapping inside each trial, interpolates behavioral streams separately for every trial, and processes experiments one at a time even though many operations are structurally identical.

ii.
```python
stim = stim_presentations[stim_presentations['image_name'] != 'omitted'].sort_values('start_time')
...
img_identity = get_image_identity_at_times(stim_cd, bin_centers,
                                            {name: i for i, name in enumerate(image_names)})
...
run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
```

iii. The trajectory confirms at least one case of avoidable repeated processing: computing a global median inside the inner loop before the AI fixed it.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extra metadata and reporting fields that are not used by downstream decoding, such as detailed `session_info`, per-experiment counts, verbose bin labels, dataset summaries, and sanity-check printouts. It also constructs local image lists and remapping structures that are only intermediate scaffolding.

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
print("SANITY CHECKS")
```

iii. The notes emphasize validation and documentation, so these extras were likely intentional for reporting, but they are not needed for the downstream decoder arrays themselves.
