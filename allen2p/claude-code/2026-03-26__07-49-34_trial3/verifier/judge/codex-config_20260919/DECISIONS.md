# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local `ophys_experiment_table.csv`, enumerates NWB files physically present in `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, and keeps only four active session types. Each selected NWB is opened directly with `h5py`. Thus “all” means all on-disk active experiments, not every experiment advertised by the Allen cache metadata; passive OPHYS_2/5 data are deliberately excluded.

ii.
```python
nwb_files = list(NWB_DIR.glob('*.nwb'))
mask = (
    exp_table['ophys_experiment_id'].isin(nwb_ids) &
    exp_table['session_type'].isin(ACTIVE_SESSION_TYPES)
)
active_exps = exp_table[mask].copy()
...
with h5py.File(nwb_path, 'r') as f:
```

iii. The notes say the available files are a partial download and justify restricting processing to them. They also exclude passive sessions because those sessions lack active behavior and meaningful trial outcomes. The agent chose direct HDF5 access as equivalent to reading the same NWB content through AllenSDK.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique `mouse_id` values in the already-filtered experiment table. Every experiment receives the corresponding index into that list.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. The notes identify `mouse_id` as the subject identifier and report that the 38 mice in the output match the on-disk subset.

## 1-c. How are the data split into sessions?

i. Each selected `ophys_experiment_id`/NWB file is emitted as one output session. Experiments sharing an `ophys_session_id` are not combined, so separate imaging planes from the same behavioral session become separate decoder sessions.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    all_neural.append(session_neural)
    all_output.append(session_output)
```

iii. The planning notes explicitly state: “Each NWB experiment = one ‘session’ in output format (one imaging plane with its own neurons).” The full-run notes consequently describe 202 experiments from 174 physical sessions as 202 output sessions.

## 1-d. How are the data split into trials?

i. Trial boundaries come from the NWB `intervals/trials` table. For every retained trial, the agent selects points on its regular 30 Hz grid satisfying `start_time <= t < stop_time`; trials with fewer than three points or no recognized outcome are dropped.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
if len(trial_time_indices) < 3:
    continue
...
else:
    continue  # Unknown outcome, skip
```

iii. The notes justify the variable-length start-to-stop window as preserving pre-change flashes and the post-change response period, and identify trial start as the alignment event.

## 1-e. How are trials filtered based on quality controls?

i. Only trials flagged Go or Catch are retained; aborted and auto-rewarded trials are removed. Experiments must have at least two eligible trials, and after segmentation must still have at least two trials. Very short and unknown-outcome trials are also removed.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
if len(valid_indices) < 2:
    return None
...
if len(neural_trials) < 2:
    return None
```

iii. Excluding aborted and auto-rewarded trials is directly tied to the task instructions. The minimum of two trials is required by the target format. The notes also treat active-only session filtering as behavioral curation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the NWB `processing/ophys/event_detection/data` calcium-event matrix, restricted using the cell table’s `valid_roi` flags. They do not come from dF/F traces.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
```

iii. The agent cites the paper’s statement that its analyses used discrete calcium events and identifies the events as FastLZeroSpikeInference output.

## 2-b. How is the `neural` data processed?

i. Valid-ROI event traces are linearly interpolated neuron by neuron onto a session-wide regular 30 Hz grid, clipped to nonnegative values, sliced by trial, transposed to neuron-by-time, and cast to `float32`.

ii.
```python
regular_ts = np.arange(t_start, t_end, 1.0 / target_rate)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes justify 30 Hz interpolation from the paper and clipping because event magnitudes should not be negative. They describe vectorized `np.interp` as a speed optimization, although the helper still loops over neurons.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are filtered to `valid_roi == True`; experiments with zero valid ROIs are skipped. No further cell-level threshold is applied.

ii.
```python
n_valid = valid_roi.sum()
if n_valid == 0:
    return None
events_valid = events_data[:, valid_roi]
```

iii. The notes connect `valid_roi` to the Allen SVM ROI-quality classifier and list invalid segmentation, duplicates, edge/motion effects, dendrites, and small/dim ROIs as reasons for exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials begin at the first point of the 30 Hz grid at or after `trials.start_time` and end before `trials.stop_time`. Metadata calls the alignment event trial start, with `off_start=0` and variable end.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
...
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
'off_start': 0.0,
'off_end': None,
```

iii. The agent says the entire trial is needed for time-varying stimulus and behavior outputs and consistently uses the same regular-grid indices for neural and outputs.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is fixed at 30 Hz, or 33.333 ms. Native event traces are linearly resampled, including upsampling lower-rate recordings; this is interpolation rather than count-preserving temporal binning.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ
regular_ts = np.arange(t_start, t_end, dt)
```

iii. The notes quote the paper’s use of a consistent 30 Hz timestamp grid and argue that this unifies approximately 31 Hz Scientifica and 11 Hz Multiscope recordings.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from the stimulus-presentations table’s `start_time`, `image_name`, and `omitted` fields, not from the trial table’s initial/change image columns.

ii.
```python
stim_data = {
    'start_time': stim['start_time'][()],
    'image_name': stim['image_name'][()],
    'omitted': stim['omitted'][()],
}
```

iii. The notes say image identity should represent the last non-omitted image, including during the gray interval, and should persist through omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each time point, `searchsorted` locates the most recent non-omitted stimulus onset. Its image is first encoded in a session-local sorted vocabulary and then remapped to a sorted global vocabulary collected in a separate NWB scan. Before the first presentation, code 0 is used.

ii.
```python
non_omitted = ~stim_data['omitted']
insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
...
image_idx[i] = name_to_idx.get(img_name, 0)
...
img_id_global = np.array([local_to_global.get(v, 0)
                          for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. The stated reason is to preserve image identity across gray and omission intervals while maintaining one deterministic category mapping across experiments.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image codes are evaluated directly at `trial_ts`, the exact regular-grid timestamps used to slice neural events, so every code corresponds one-to-one with a neural column.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. The agent’s sanity check independently compared a converted image label against the NWB and reported a match.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change uses stimulus-presentation `is_change`, `omitted`, and `start_time`. The code reads `stop_time` too, but does not use it. Only non-omitted change presentations create positives.

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. The notes define a real image change as a non-omitted presentation flagged `is_change`, distinguishing it from Catch/sham changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created and set to one for every regular-grid point in the 750 ms interval beginning at each actual change onset.

ii.
```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The 750 ms duration is justified as one 250 ms image flash plus its following 500 ms gray interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: category 1 denotes the `[change onset, change onset + 0.75 s)` window and category 0 denotes every other point. No numeric thresholding is applied.

ii.
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
change_signal[mask] = 1
```

iii. The output categories are documented as `['no_change', 'change']`; the notes verify that Catch trials contain no positive points and Go trials do.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The binary signal is evaluated on `trial_ts`, exactly the timestamps of the trial’s neural columns.

ii.
```python
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. The agent uses one shared regular timebase for all time-varying signals and reports boundary/length alignment checks as passing.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from NWB `processing/running/speed/data` with `timestamps`.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. The notes identify this as the wheel-encoder speed signal and treat the NWB path as equivalent to the AllenSDK `running_speed` interface.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated to the regular 30 Hz grid. Five global percentile edges are computed in a preliminary pass from every native running sample in every selected experiment—not just retained trial samples—and trial values are digitized with those edges.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
all_running_values.append(running.astype(np.float32))
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The agent says interpolation synchronizes behavior with neural data and percentile bins balance decoder classes globally. The notes report roughly 18–21% occupancy per bin.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 0th, 20th, 40th, 60th, 80th, and 100th percentiles define five categories; interior edges are passed to `np.digitize`. Duplicate edges are nudged upward by `1e-10`, results are clipped to 0–4, and NaNs map to category 0.

ii.
```python
edges = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
...
binned = np.digitize(values, bin_edges[1:-1])
binned[np.isnan(values)] = 0
```

iii. Equal-percentile bins were chosen to approximate balanced class sizes for decoding and one mapping is shared across sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The full speed trace is interpolated to `regular_ts`, then indexed by the same `trial_time_indices` used for neural activity.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
running_trial = running_resampled[trial_time_indices]
```

iii. The notes cite hardware synchronization of data streams and use the shared resampled grid to guarantee equal lengths.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Despite the output name, the source is pupil `area`, plus eye-tracking timestamps and `likely_blink`; pupil width/diameter is not read.

ii.
```python
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The planning notes explicitly call pupil area a proxy for diameter. They chose it because it was available in the NWB eye-tracking group.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples are replaced with NaN, all NaNs are linearly filled over sample index (all-missing arrays become zero), and the result is linearly resampled to 30 Hz. Global percentile edges are separately computed from all nonblink native pupil-area samples across selected experiments, not from the imputed trial-aligned values.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
```

iii. The notes say blink interpolation prevents artifacts and missing values from contaminating the signal, followed by the same global percentile strategy as running speed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five categories are formed from global pupil-area quintiles using the same digitization, duplicate-edge, clipping, and NaN-to-zero rules as running speed. A wholly missing experiment therefore produces all category 0.

ii.
```python
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. The agent sought approximately equal class sizes; its notes report 19–23% occupancy per pupil bin in the full conversion.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Imputed pupil area is interpolated to the same `regular_ts` used for neural events and sliced using identical trial indices.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. The shared grid is the agent’s alignment guarantee; its final review reports that neural/output lengths matched at trial boundaries.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome derives from the trial table’s `hit`, `miss`, `false_alarm`, and `correct_reject` Boolean columns.

ii.
```python
'hit': trials_grp['hit'][()].astype(bool),
'miss': trials_grp['miss'][()].astype(bool),
'false_alarm': trials_grp['false_alarm'][()].astype(bool),
'correct_reject': trials_grp['correct_reject'][()].astype(bool),
```

iii. The agent describes these as mutually exclusive canonical outcomes for retained Go and Catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. A fixed precedence maps hit, miss, false alarm, and correct reject to 0, 1, 2, and 3. Trials matching none are skipped. Although described as static, the code broadcasts the code across every trial time point.

ii.
```python
if trials['hit'][trial_idx]: outcome = 0
elif trials['miss'][trial_idx]: outcome = 1
elif trials['false_alarm'][trial_idx]: outcome = 2
elif trials['correct_reject'][trial_idx]: outcome = 3
else: continue
...
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
```

iii. The notes use the fixed four-class ordering required for decoder labels and verify that outcome is constant within every trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. File-level load exceptions return `None` and skip the experiment. Experiments without stimuli or valid ROIs are skipped. Missing pupil data generates an all-NaN trial trace which becomes category 0; blink/NaN gaps are interpolated, and wholly NaN arrays become zeros. Unknown outcomes and too-short trials are dropped. Image lookup failures/default pre-stimulus points use code 0. Duplicate percentile edges are perturbed. `np.interp` also extends endpoint values outside source ranges.

ii.
```python
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
...
if nans.all():
    return np.zeros_like(arr)
...
pupil_trial = np.full(n_tp, np.nan)
```

iii. The notes emphasize robustness to absent pupil tracking, blink artifacts, partial local data, and isolated bad experiments. They report final checks for NaNs, valid category ranges, and aligned shapes.

## 9-a. What are the most time-consuming steps of the code?

i. The costly work is opening every NWB three times, reading large event arrays, interpolating every neuron to 30 Hz, retaining an approximately 8.3 GB assembled dataset, and serializing it. The full conversion reportedly took 358 seconds; repeated file I/O and per-neuron interpolation dominate conversion work.

ii.
```python
# Pass 1: open every file for image names
# Pass 2: open every file for running/pupil
# Pass 3: load_experiment_data(...), interpolate events, assemble output
for i in range(data.shape[1]):
    result[:, i] = np.interp(...)
```

iii. The notes identify sequential experiment processing and multiple NWB passes as inefficiencies, while their timing estimate divides work into load and process phases. The final file size makes pickle writing another material cost even though it was not highlighted in the planning notes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The neuron-by-neuron interpolation loop, per-time-point image assignment loop, per-trial local-to-global image remapping comprehension, and repeated full-grid masking for each image change could be vectorized or replaced by indexed operations. Trial extraction itself remains a natural loop because trials are variable length.

ii.
```python
for i in range(data.shape[1]):
    result[:, i] = np.interp(...)
for i in range(n_tp):
    ...
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
```

iii. The notes claim interpolation and `searchsorted` speedups, but also acknowledge sequential processing. They do not specifically call out that `searchsorted` is followed by a Python loop over all time points.

## 9-c. What processing does the code repeat multiple times?

i. Every selected NWB is opened in three passes: image vocabulary, behavior values for percentile edges, and full conversion. Running and pupil arrays are read in both passes 2 and 3; stimulus image names are read in passes 1 and 3. Local image dictionaries and local-to-global mappings are rebuilt per experiment, and outputs are traversed again for plotting when requested.

ii.
```python
print("--- Pass 1: Collecting global image names ---")
...
print("--- Pass 2: Collecting running speed and pupil data ... ---")
...
print("--- Pass 3: Processing experiments ---")
```

iii. The notes explicitly identify “multiple passes over NWB files” as an inefficiency, accepted to obtain global image categories and percentile thresholds before final assembly.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads and returns `cell_specimen_ids` and several trial fields (`change_time`, `is_change`, initial/change image names) that downstream conversion never uses. It computes `change_stops` and a `trial_outcomes` list variable that are unused, constructs a one-element `trial_outcome` array that is discarded in favor of broadcasting from the scalar, imports unused concurrency helpers, and optionally generates plots not used by the decoder. Full-session resampling also processes long periods outside retained trials.

ii.
```python
cell_specimen_ids = cell_table['cell_specimen_id'][()]
...
change_stops = stim_data['stop_time'][change_mask]
trial_outcomes = []
...
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
```

iii. The agent did not document these discarded intermediates. Its notes instead state that the three-pass design and sequential processing were the principal inefficiencies.
