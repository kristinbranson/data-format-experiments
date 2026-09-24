# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local `ophys_experiment_table.csv`, enumerates locally available NWB files, retains experiments with a local file and `passive == False`, and loads each retained NWB independently with `BehaviorOphysExperiment.from_nwb_path`. This includes active experiments from both `VisualBehavior` and `VisualBehaviorMultiscope`, rather than restricting to the reference's `VisualBehavior` project.

ii.
```python
exp = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
...
exp = exp[exp['ophys_experiment_id'].isin(nwb_ids)]
exp = exp[exp['passive'] == False]
...
dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
```

iii. The trajectory says the agent found 202 active experiments across both projects and chose all active data to maximize mouse and session coverage. It recognized that the paper emphasized familiar multiscope data, but considered the available strict subset too small and prioritized decoder diversity.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id`s from successfully processed experiments. Each output session receives the index of its experiment's mouse in a sorted global subject list.

ii.
```python
mouse_id = str(meta['mouse_id'])
...
all_subjects = sorted(set(r['mouse_id'] for r in results))
...
subject_idx.append(all_subjects.index(r['mouse_id']))
```

iii. The trajectory treats the metadata mouse ID as the animal identifier and reports 38 mice in the final dataset. No alternative subject grouping was considered.

## 1-c. How are the data split into sessions?

i. Every `ophys_experiment_id`/NWB file is emitted as a separate session. Simultaneously recorded imaging planes sharing an `ophys_session_id` are not combined.

ii.
```python
for i, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(row, sample_mode=args.sample)
...
for r in results:
    ...
    neural_all.append(session_neural)
```

iii. The trajectory initially noted that each NWB is an imaging plane, but ultimately described each file as a separate session with different neurons. The choice simplified mixed-rate processing but did not preserve the SDK's behavioral-session grouping.

## 1-d. How are the data split into trials?

i. The SDK `trials` table defines trials. For each valid row, the agent uses the full half-open interval `[start_time, stop_time)`, creates common 90.9 ms bin centers, and stores a variable-length trial.

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
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
neural_resampled, bin_centers = resample_to_common_bins(
    trial_ophys_ts, trial_neural_raw, t_start, t_end)
```

iii. The agent explicitly chose Go and Catch trials and the built-in trial boundaries because the instructions define those trial types and require ophys-time alignment. It retained the complete trial so pre- and post-change information remained available.

## 1-e. How are trials filtered based on quality controls?

i. Trials must be Go or Catch, non-aborted, and non-auto-rewarded. Invalid or reversed start/stop times, trials with fewer than three raw frames or fewer than three resampled bins, and trials without one of the four recognized outcomes are dropped. Experiments need at least two valid trials; experiments with fewer than five neurons are also dropped.

ii.
```python
if n_neurons < MIN_NEURONS:
    return None
...
if np.isnan(t_start) or np.isnan(t_end) or t_end <= t_start:
    continue
if frame_mask.sum() < MIN_TRIAL_FRAMES:
    continue
...
if outcome == -1:
    continue
...
if trial_count < 2:
    return None
```

iii. Excluding aborted and auto-rewarded trials directly follows the task. The trajectory emphasizes format validity and decoder usability; the three-frame, five-neuron, recognized-outcome, and two-trial requirements are defensive additions rather than paper-derived controls.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Final neural activity is derived from every ROI's SDK `dff_traces`/`dff` array. The agent initially implemented detected events, then changed to dF/F.

ii.
```python
dff_df = dataset.dff_traces
neural_data = np.array([row['dff'] for _, row in dff_df.iterrows()])
```

iii. The trajectory says detected events were about 99.7% zero and produced all-zero trial warnings. Although it acknowledged the paper used events, it selected continuous dF/F as more usable for decoder training and verified above-chance performance.

## 2-b. How is the `neural` data processed?

i. Trial-window dF/F is averaged within fixed 1/11-second bins. If a bin has no ophys sample, the nearest raw frame is used. The result is cast to `float32` during assembly.

ii.
```python
for b, tc in enumerate(bin_centers):
    mask = (timestamps >= t_lo) & (timestamps < t_hi)
    if mask.sum() > 0:
        resampled[:, b] = data[:, mask].mean(axis=1)
    else:
        idx = np.argmin(np.abs(timestamps - tc))
        resampled[:, b] = data[:, idx]
...
session_neural.append(neural.astype(np.float32))
```

iii. The agent wanted a common resolution across approximately 31 Hz single-plane and 11 Hz multiscope recordings and chose the slower multiscope rate to avoid upsampling artifacts and to resemble the paper's rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The SDK-provided dF/F ROIs are otherwise unfiltered, but whole experiments with fewer than five neurons and trial slices with fewer than three frames/bins are excluded.

ii.
```python
MIN_TRIAL_FRAMES = 3
MIN_NEURONS = 5
...
if n_neurons < MIN_NEURONS:
    return None
```

iii. The trajectory does not cite a paper-based neuronal QC criterion. These thresholds appear intended to avoid degenerate decoder sessions and very short samples.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials begin at `trial['start_time']`, end at `trial['stop_time']`, and use absolute ophys timestamps to select frames and form bin centers. Metadata names the temporal alignment event as trial start.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
bin_centers = np.arange(trial_start + COMMON_BIN_SIZE / 2,
                        trial_end, COMMON_BIN_SIZE)
...
'temporal_alignment_event': 'Trial start time (onset of first stimulus in trial)',
```

iii. The agent chose trial start because full trial segmentation preserves both pre-change and post-change periods and all streams can be placed on the ophys clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The declared resolution is `1/11` second, or about 90.9 ms. All neural data are explicitly rebinned to these common bins, including native ~11 Hz data.

ii.
```python
COMMON_BIN_SIZE = 1.0 / 11.0
...
'time_bin_size': COMMON_BIN_SIZE * 1000,
```

iii. The trajectory says common 11 Hz bins reconcile the two project frame rates, match the multiplane acquisition rate, and avoid upsampling the slower data.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from `dataset.stimulus_presentations`, principally `image_name` and `start_time`, after restricting to change-detection presentations and removing omitted images.

ii.
```python
stim = dataset.stimulus_presentations
stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)]
...
stim = stim_presentations[stim_presentations['image_name'] != 'omitted'].sort_values('start_time')
```

iii. The trajectory examined stimulus presentations as the flash-level source and used it to represent time-varying identity throughout a trial.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Non-omitted image names are sorted into per-experiment integer categories. At each bin center, `searchsorted` selects the most recently started image, thereby carrying its identity through grey periods. Local codes are then remapped to a sorted global vocabulary.

ii.
```python
indices = np.searchsorted(stim_times, bin_centers, side='right') - 1
result = np.where(indices >= 0,
                  stim_idx[np.clip(indices, 0, len(stim_idx) - 1)], 0)
...
local_to_global_arr = np.array([global_img_to_idx[name] for name in r['image_names']])
img_id_global = local_to_global_arr[out['image_identity']]
```

iii. The agent states that identity during grey should be the last shown image. It later vectorized the lookup because the initial per-bin implementation was a runtime bottleneck.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated at the exact same common `bin_centers` used for rebinned neural activity, producing one category per neural time bin.

ii.
```python
img_identity = get_image_identity_at_times(stim_cd, bin_centers,
                                            {name: i for i, name in enumerate(image_names)})
```

iii. The shared absolute-time bin centers were intended to guarantee alignment across neural, stimulus, and behavior streams.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations['is_change']` and each matching presentation's `start_time`.

ii.
```python
changes = stim_presentations[stim_presentations['is_change'] == True]
change_times = changes['start_time'].values
```

iii. The agent used the flash table's explicit change marker rather than reconstructing changes from trial image fields.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created and bins from each change time through two common-bin widths later are marked one.

ii.
```python
for ct in change_times:
    mask = (bin_centers >= ct) & (bin_centers < ct + COMMON_BIN_SIZE * 2)
    result[mask] = 1
```

iii. The docstring says this marks the time bin immediately after an identity change. The implementation uses a two-bin window, apparently to ensure the event is captured at the chosen resolution; no deeper trajectory justification is given.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is directly binary: bins in the post-change window are category 1 and all others category 0. No numeric threshold is estimated.

ii.
```python
result = np.zeros(len(bin_centers), dtype=int)
...
result[mask] = 1
```

iii. This directly implements the requested binary output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change windows are evaluated against the same common bin centers as the neural data.

ii.
```python
img_change = get_image_change_at_times(stim_cd, bin_centers)
```

iii. Shared bin centers provide frame-for-frame output alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `dataset.running_speed['timestamps']` and `dataset.running_speed['speed']`.

ii.
```python
running = dataset.running_speed
running_ts = running['timestamps'].values
running_speed = running['speed'].values
```

iii. The agent identified the SDK running-speed table as the synchronized wheel-derived locomotion stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Valid raw speed samples are linearly interpolated/extrapolated at bin centers. Values from all included trials and experiments are pooled, global percentile edges are computed, and values are digitized into five bins.

ii.
```python
f = interpolate.interp1d(timestamps[valid], values[valid],
                         kind='linear', bounds_error=False, fill_value='extrapolate')
...
edges = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
...
run_disc = apply_discretization(out['running_speed_raw'], running_edges)
```

iii. The trajectory chose interpolation onto the common ophys-aligned grid and equal-percentile categories to satisfy the five-bin decoder requirement and balance class frequencies.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four internal global percentile cut points split valid running values into five zero-indexed categories using `np.digitize`; outer edges are replaced with infinities.

ii.
```python
edges[0] = -np.inf
edges[-1] = np.inf
result = np.digitize(values, edges[1:-1])
result = np.clip(result, 0, len(edges) - 2)
```

iii. Equal percentile bins were selected because the task explicitly requests them and they provide approximately balanced decoder classes.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated at each trial's neural `bin_centers`, so its length and timestamps correspond exactly to neural bins.

ii.
```python
run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
```

iii. The agent reasoned that timestamp interpolation onto the common ophys-derived grid provides synchronized alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It is derived from eye-tracking `pupil_width`, `pupil_height`, `timestamps`, and `likely_blink`. Diameter is defined as the arithmetic mean of width and height.

ii.
```python
eye_ts = eye['timestamps'].values
pupil_diam = ((eye['pupil_width'].values + eye['pupil_height'].values) / 2.0)
likely_blink = eye['likely_blink'].values
pupil_diam[likely_blink] = np.nan
```

iii. The trajectory considered width, height, and area and settled on averaging width and height as a diameter proxy, while using the SDK blink flag to reject artifacts.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples become NaN; remaining values are linearly interpolated/extrapolated to bin centers. Global non-NaN percentiles define five bins. Before discretization, remaining NaNs are replaced by the global median pupil value.

ii.
```python
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
...
pupil_fill_value = np.nanmedian(all_pupil_flat)
...
pupil_raw[nan_mask] = pupil_fill_value
pupil_disc = apply_discretization(pupil_raw, pupil_edges)
```

iii. The agent removed likely blinks to avoid artifacts, used the same interpolation and percentile strategy as running speed, and precomputed the median after discovering that recomputing it in every trial was a major bottleneck.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global percentile cut points divide valid pupil values into five zero-indexed categories. Missing values are median-filled first and thus enter the middle percentile category.

ii.
```python
pupil_edges = discretize_continuous(all_pupil_flat, n_bins=5)
...
pupil_raw[nan_mask] = pupil_fill_value
pupil_disc = apply_discretization(pupil_raw, pupil_edges)
```

iii. Percentile thresholds satisfy the required five equal-percentile bins; median filling was chosen to avoid invalid category values.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Eye-tracking values are interpolated at the same trial bin centers used for neural activity.

ii.
```python
pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
```

iii. Shared absolute bin-center timestamps were intended to align eye and ophys clocks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome uses the trial table's boolean `hit`, `miss`, `false_alarm`, and `correct_reject` columns.

ii.
```python
if trial['hit']:
    return 0
elif trial['miss']:
    return 1
elif trial['false_alarm']:
    return 2
elif trial['correct_reject']:
    return 3
```

iii. The agent regarded these as the canonical mutually exclusive outcomes for valid Go/Catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four outcomes are mapped in fixed order to integers 0–3. Unknown outcomes are assigned -1 and their trials are discarded. During assembly, the chosen integer is broadcast across every time bin of its trial.

ii.
```python
outcome = get_trial_outcome(trial)
if outcome == -1:
    continue
...
output_data[4, :] = out['trial_outcome']
```

iii. Broadcasting allows a static per-trial output to coexist in the common `(n_outputs, n_timepoints)` representation and matches the decoder's expected layout.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. NWB load failures and missing mandatory neural/stimulus/trial tables cause an experiment to be skipped. Missing running data become zeros; missing pupil data become NaNs and later the global median. Too few valid interpolation samples return NaNs. Blinks are NaN. Invalid/short/unknown-outcome trials and undersized experiments are skipped. Interpolation extrapolates beyond behavior timestamp ranges.

ii.
```python
except Exception as e:
    print(f"  ERROR loading {eid}: {e}")
    return None
...
run_at_bins = np.zeros(n_bins)
...
pupil_at_bins = np.full(n_bins, np.nan)
...
if valid.sum() < 2:
    return np.full(len(bin_centers), np.nan)
```

iii. The trajectory emphasizes keeping the full conversion running despite individual bad streams/files and ensuring all final outputs are valid discrete values. It validated the assembled data with sanity checks and the decoder.

## 9-a. What are the most time-consuming steps of the code?

i. Loading and processing roughly 202 large NWB experiments is I/O/CPU intensive. The original image-identity per-bin loop was slow until vectorized. The worst discovered assembly bottleneck was recomputing `np.nanmedian(all_pupil_flat)` inside every one of roughly 51,000 trial iterations. Pickle assembly/saving and large-memory handling also took substantial time.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
...
pupil_fill_value = np.nanmedian(all_pupil_flat)
```

iii. The trajectory records an initial 20-minute run, profiling by inspection, vectorization of image lookup, and discovery of the repeated median calculation. After moving the median outside the loop, the full run completed with 200 experiments and 51,557 trials.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The neural rebinning loop over every bin and each change-time loop could be vectorized. The outer experiment/trial loops are structurally useful, but parts of trial assembly were vectorized: image lookup now uses `searchsorted`, and local-to-global image remapping uses an array lookup.

ii.
```python
for b, tc in enumerate(bin_centers):
    ...
for ct in change_times:
    ...
indices = np.searchsorted(stim_times, bin_centers, side='right') - 1
img_id_global = local_to_global_arr[out['image_identity']]
```

iii. The trajectory explicitly identifies and fixes the image-identity loop. It did not vectorize bin averaging, likely because trial lengths and sample membership vary.

## 9-c. What processing does the code repeat multiple times?

i. Each experiment independently loads identical session-level behavioral/stimulus/trial information even when several NWBs are planes from the same `ophys_session_id`. It also recomputes filtered/sorted stimulus arrays and interpolation objects for every trial, and scans all trials again for global binning and final assembly.

ii.
```python
for i, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(row, sample_mode=args.sample)
...
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    img_identity = get_image_identity_at_times(stim_cd, bin_centers, ...)
    run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
```

iii. The trajectory focused on the especially costly repeated median and eliminated it. It did not address duplication caused by processing simultaneous planes as separate sessions.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It builds descriptive bin-label strings and extensive `session_info`, computes/prints summary distributions and trial-duration statistics, and runs full post-save sanity-check traversals; these support reporting but are not decoder inputs. It also retains several metadata fields unused by downstream decoding.

ii.
```python
running_bin_labels.append(f"bin{i}_{lo:.1f}_{hi:.1f}")
...
session_info.append({...})
...
for si, session in enumerate(output_all):
    for ti, out in enumerate(session):
        assert out.shape[0] == 5
```

iii. The trajectory valued explicit validation, plots, summary statistics, and decoder verification. Thus this processing was intentionally used for QA, even though it is discarded by downstream model training.
