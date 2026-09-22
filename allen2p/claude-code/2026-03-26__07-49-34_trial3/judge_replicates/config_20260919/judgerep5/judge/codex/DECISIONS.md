# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment metadata CSV, inventories local NWB files, keeps on-disk experiments from four active session types, and loads each NWB directly with `h5py`. It does not use the SDK cache or include every VisualBehavior session type.

ii.
```python
exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
nwb_files = list(NWB_DIR.glob('*.nwb'))
mask = (exp_table['ophys_experiment_id'].isin(nwb_ids) &
        exp_table['session_type'].isin(ACTIVE_SESSION_TYPES))
with h5py.File(nwb_path, 'r') as f:
```

iii. The notes say direct HDF5 access reads the same NWB content faster, and active sessions were selected because the task concerns active Visual Behavior. The agent considered the 284 local files a subset of the full release.

## 1-b. How are the data split into subjects?

i. Subjects are sorted unique `mouse_id` values among selected experiments; every experiment receives that mouse's index.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. The notes identify `mouse_id` as the subject identifier.

## 1-c. How are the data split into sessions?

i. Each NWB `ophys_experiment_id` (one imaging plane) is made a separate output session. Simultaneously acquired planes sharing an `ophys_session_id` are not combined.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    all_neural.append(session_neural)
```

iii. The notes explicitly decide: “Each NWB experiment = one ‘session’ ... (one imaging plane with its own neurons).”

## 1-d. How are the data split into trials?

i. Trial rows come from the NWB `intervals/trials` table. For each valid row, samples on the regular 30-Hz grid satisfying `start_time <= t < stop_time` form the trial; trials with fewer than three samples are skipped.

ii.
```python
valid_indices = np.where(valid_mask)[0]
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
if len(trial_time_indices) < 3:
    continue
```

iii. The agent chose full trial start-to-stop windows so pre-change and response-period outputs remain time varying.

## 1-e. How are trials filtered based on quality controls?

i. It retains Go or Catch trials, removes aborted and auto-rewarded trials, skips unknown outcomes and very short trials, and discards experiments with fewer than two valid/processed trials.

ii.
```python
valid_mask = ((trials['go'] | trials['catch']) &
              ~trials['aborted'] & ~trials['auto_rewarded'])
if len(valid_indices) < 2: return None
...
else: continue  # Unknown outcome
```

iii. The notes tie Go/Catch inclusion and aborted/auto-reward exclusion directly to the task, and require two trials for decoder validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from NWB event-detection `data`, filtered by the cell table's `valid_roi`; it is not derived from dF/F traces.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
```

iii. The notes chose sparse FastLZeroSpikeInference calcium events and regarded `valid_roi` as the SDK's quality filter.

## 2-b. How is the `neural` data processed?

i. Valid event traces are linearly interpolated from ophys timestamps to a 30-Hz regular grid, clipped nonnegative, trial-sliced, transposed to neuron-by-time, and cast to float32.

ii.
```python
regular_ts = np.arange(t_start, t_end, 1.0 / target_rate)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes cite paper text about interpolation to consistent 30-Hz timestamps and say clipping preserves the nonnegative nature of events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `valid_roi=True` cells are retained; experiments with no valid ROIs are skipped. No other neuron filtering is performed.

ii.
```python
n_valid = valid_roi.sum()
if n_valid == 0: return None
events_valid = events_data[:, valid_roi]
```

iii. The agent says this matches the SDK default invalid-ROI exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are placed on the common absolute 30-Hz grid and sliced from each trial's start through stop; metadata calls trial start the alignment event.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
neural_trial = events_resampled[trial_time_indices, :].T
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)'
```

iii. The justification is that one grid aligns neural and all outputs while preserving the full variable-length trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 33.33 ms (30 Hz). Linear temporal resampling is applied to neural events and behavioral signals.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ
```

iii. The agent cites the paper's 30-Hz interpolation, despite the reference conversion retaining native ~11-Hz ophys frames.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from stimulus-presentation `start_time`, `image_name`, and `omitted`, rather than trial initial/change image columns.

ii.
```python
stim_data = {'start_time': stim['start_time'][()],
             'image_name': stim['image_name'][()],
             'omitted': stim['omitted'][()]}
```

iii. The agent wanted the actually presented image at every sample and to carry the prior image through gray/omitted periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Omitted rows are removed, `searchsorted` finds the most recent stimulus onset, local sorted image codes are then remapped to a global sorted 16-image vocabulary.

ii.
```python
insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
image_idx[i] = name_to_idx.get(stim_names[insert_idx[i]], 0)
img_id_global = np.array([local_to_global.get(v, 0) for v in
                          trial_data_out['image_identity']])
```

iii. The notes say global coding makes classes consistent, gray retains the last image, and omitted flashes do not change identity.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at the exact `trial_ts` used to slice resampled neural data, so its length and grid match neural timepoints.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. The common 30-Hz timestamps are the stated alignment mechanism.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses stimulus-presentation `is_change`, `omitted`, and `start_time` (the loaded `stop_time` is unused).

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
```

iii. The agent treats non-omitted `is_change` presentations as real changes, naturally leaving Catch sham changes at zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For every change onset, it marks a 750-ms window as one and leaves all other samples zero.

ii.
```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The notes interpret 750 ms as the changed flash plus gray interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric thresholding occurs: membership in a change-onset window directly yields categorical 1; otherwise 0.

ii.
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
change_signal[mask] = 1
```

iii. This directly implements a binary event indicator.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change windows are tested at the same per-trial regular timestamps as neural samples.

ii.
```python
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. The shared grid guarantees equal time axes.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses NWB running `speed/data` and `speed/timestamps`.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. The notes identify the Allen running-wheel speed stream as the canonical source.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated to 30 Hz and discretized using five global percentile bins. Edges are computed from all full-session raw speed samples, not only retained trial samples.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'],
                              raw_data['running_speed'])
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The agent says global percentiles provide consistent, approximately balanced decoder classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Percentiles 0,20,...,100 define edges; internal four edges are passed to `np.digitize`, clipped to 0–4, with NaNs assigned to class 0. Duplicate edges are nudged upward.

ii.
```python
edges = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
binned = np.digitize(values, bin_edges[1:-1])
binned[np.isnan(values)] = 0
```

iii. The justification is five equal-percentile categories as requested.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated to `regular_ts` and sliced with the same trial indices as neural events.

ii.
```python
running_trial = running_resampled[trial_time_indices]
```

iii. The agent relies on synchronized NWB clocks and the common grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Despite the output name, it uses EyeTracking pupil `area`, timestamps, and `likely_blink`, not pupil width/diameter.

ii.
```python
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()]
```

iii. The notes chose pupil area as the available pupil-size measure and used the blink flag for artifact removal.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples become NaN, internal NaNs are linearly filled (all-NaN streams become zero), then area is interpolated to 30 Hz and globally percentile-binned. Missing pupil streams become NaNs and hence class 0.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
```

iii. The agent says interpolation prevents blink artifacts and maintains a complete aligned signal.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five 20-percentile bins are computed from all non-blink full-session area samples; digitization is 0–4 and NaNs map to 0.

ii.
```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. The justification mirrors running speed: globally consistent, approximately balanced classes.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Filled pupil area is interpolated to the common 30-Hz grid and sliced using neural trial indices.

ii.
```python
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. Shared timestamps and indices are the stated alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It uses trial booleans `hit`, `miss`, `false_alarm`, and `correct_reject` in that priority order.

ii.
```python
if trials['hit'][trial_idx]: outcome = 0
elif trials['miss'][trial_idx]: outcome = 1
elif trials['false_alarm'][trial_idx]: outcome = 2
elif trials['correct_reject'][trial_idx]: outcome = 3
```

iii. The agent identifies these as the four mutually exclusive canonical outcomes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes are encoded 0–3; unknown outcomes are skipped. Although called static, the code broadcasts the code across every trial timepoint in the combined output matrix.

ii.
```python
outcome_broadcast = np.full((1, n_tp),
                            trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. Broadcasting was chosen to fit mixed static/time-varying outputs into one rectangular per-trial array.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. File-open errors return `None`; missing pupil data becomes all NaN then bin 0; blinks/NaNs are interpolated (all NaN becomes zero); missing stimuli or zero valid ROIs skip an experiment; malformed/short/unknown-outcome trials are skipped. `np.interp` endpoint behavior fills out-of-range running/pupil with boundary values.

ii.
```python
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
...
if nans.all(): return np.zeros_like(arr)
...
pupil_trial = np.full(n_tp, np.nan)
```

iii. The agent prioritized completing conversion despite isolated bad experiments and avoiding NaNs in decoder outputs.

## 9-a. What are the most time-consuming steps of the code?

i. Repeated NWB I/O and per-neuron interpolation dominate: files are opened in three passes, and 2-D interpolation loops over every cell.

ii.
```python
for i in range(data.shape[1]):
    result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
```

iii. The notes measured the full conversion at 358 seconds and identified sequential processing and multiple passes as inefficiencies.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cell interpolation loop, per-timepoint image-code loop, local-to-global list comprehension, trial loop, and per-change mask loop are candidates. `searchsorted` itself is already vectorized.

ii.
```python
for i in range(data.shape[1]): ...
for i in range(n_tp): ...
for trial_idx in valid_indices: ...
```

iii. The notes claim vectorized `np.interp` and efficient `searchsorted`, but `np.interp` is still invoked once per neuron and image assignment still loops.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB is opened once to collect image names, again to collect running/pupil statistics, and again for full loading. Stimulus-table discovery and several datasets are therefore read repeatedly.

ii.
```python
# Pass 1 ... with h5py.File(nwb_path, 'r')
# Pass 2 ... with h5py.File(nwb_path, 'r')
# Pass 3 ... load_experiment_data(nwb_path, eid)
```

iii. The agent explicitly documented the three-pass design and acknowledged “multiple passes over NWB files” as an inefficiency.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads unused cell IDs, several unused trial fields, and stimulus stop times; computes an unused `ce`; creates an unused scalar `trial_outcome`; and, when plotting is disabled, still retains some raw structures until each iteration ends. Full-session samples outside retained trials are processed for bin edges and all neural events are resampled before only trial slices are kept.

ii.
```python
cell_specimen_ids = cell_table['cell_specimen_id'][()]
'is_change': trials_grp['is_change'][()]
for cs, ce in zip(change_starts, change_stops):
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
```

iii. The notes do not justify these discarded values; they arise from broad extraction, diagnostics, and the multi-pass implementation.
