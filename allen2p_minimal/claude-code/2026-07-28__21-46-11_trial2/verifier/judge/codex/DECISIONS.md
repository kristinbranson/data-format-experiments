# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from the local Allen Visual Behavior NWB release directly with `h5py`, starting from `project_metadata/ophys_experiment_table.csv`. It filters to experiments that have a local NWB file and `passive == False`, then opens each NWB file itself. It does not use the AllenSDK cache or filter to only `project_code == "VisualBehavior"`.

ii.
```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
...
nwb_files = [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
nwb_ids = [int(f.split('_')[-1].split('.')[0]) for f in nwb_files]
...
exp_table = exp_table[exp_table['ophys_experiment_id'].isin(nwb_ids)]
exp_table = exp_table[exp_table['passive'] == False]
...
with h5py.File(nwb_path, 'r') as f:
```

iii. The justification in the notes is that the local NWB files contain the full Visual Behavior 2P release, that only active behavior should be kept, and that direct NWB loading “mirrors what the SDK does under the hood.” The notes also explicitly say it intentionally included all active project codes, not only `VisualBehavior`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are unique `mouse_id` values from the experiment metadata table, converted to strings and sorted.

ii.
```python
all_subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
mouse_id = str(row['mouse_id'])
...
subject_idx_list.append(subject_to_idx[mouse_id])
```

iii. The implicit justification is that `mouse_id` is the dataset’s animal identifier. The notes report subject counts from the filtered experiment table and treat mice as the top-level grouping.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` as one output session. It does not group multiple experiments that share an `ophys_session_id`; the notes explicitly say “Each experiment = 1 session in the output.”

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
    nwb_data = load_nwb_data(nwb_path)
    result = process_experiment(nwb_data, all_image_names)
    ...
    neural_all.append(final_neural)
```

iii. The notes justify this by describing the output as experiment-wise rather than session-wise and by distinguishing single-plane and multiscope experiments by their individual NWB files.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table. For each included trial, the AI uses the full `[start_time, stop_time)` trial window and extracts all ophys frames whose timestamps fall inside that interval.

ii.
```python
trials = f['intervals/trials']
data['trial_start_times'] = trials['start_time'][:]
data['trial_stop_times'] = trials['stop_time'][:]
...
t_start = nwb_data['trial_start_times'][trial_idx]
t_stop = nwb_data['trial_stop_times'][trial_idx]
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]
```

iii. The notes justify this as matching the experiment-defined trial segmentation and preserving the whole behavior trial, including pre-change and post-change periods.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if they are aborted, auto-rewarded, neither go nor catch, have an unknown outcome, or contain fewer than two ophys frames. Experiments with fewer than two valid trials are dropped.

ii.
```python
if nwb_data['trial_aborted'][trial_idx]:
    return False
if nwb_data['trial_auto_rewarded'][trial_idx]:
    return False
if not (nwb_data['trial_go'][trial_idx] or nwb_data['trial_catch'][trial_idx]):
    return False
...
outcome = get_trial_outcome(trial_idx, nwb_data)
if outcome == -1:
    continue
...
if len(frame_indices) < 2:
    continue
...
if neural_trials is None or len(neural_trials) < 2:
    ...
```

iii. The notes justify the aborted and auto-reward exclusions as matching the task spec. The extra `go`/`catch`, valid-outcome, and minimum-frame filters are code-level safeguards rather than decisions described in the notes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/event_detection/data`, filtered by the `valid_roi` field in the cell specimen table. The script also loads dF/F traces, but only as an unused fallback/comparison source.

ii.
```python
events = f['processing/ophys/event_detection/data'][:]
events_ts = f['processing/ophys/event_detection/timestamps'][:]
data['events'] = events
...
dff = f['processing/ophys/dff/traces/data'][:]
data['dff'] = dff
...
valid = nwb_data['valid_roi']
events = events[:, valid]
```

iii. The notes justify this by citing the paper’s statement about using “discrete calcium events” rather than raw dF/F, and by saying the direct `valid_roi` filter reproduces the SDK’s invalid-ROI exclusion.

## 2-b. How is the `neural` data processed?

i. The AI transposes the events matrix from time-by-cell to cell-by-time, filters to valid ROIs, and then slices it into per-trial arrays. It does not merge multiple planes into a single session because it processes each experiment independently.

ii.
```python
events = nwb_data['events']  # (n_timepoints, n_cells)
valid = nwb_data['valid_roi']
events = events[:, valid]
...
trial_neural = events[frame_indices, :].T  # (n_neurons, n_timepoints)
...
neural_trials.append(trial_neural.astype(np.float32))
```

iii. The notes justify the events choice as matching the paper and describe the output as per-experiment neural trials, with all neurons in an experiment sharing one targeted structure.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered by the `valid_roi` boolean from the NWB cell specimen table. If no valid neurons remain, the experiment is skipped.

ii.
```python
valid = nwb_data['valid_roi']
events = events[:, valid]
...
if n_neurons == 0:
    return None, None, None, None, None, None
...
if n_neurons < 1:
    print(f"    SKIPPED: no valid neurons")
```

iii. The notes justify this as keeping only valid ROIs and matching the AllenSDK’s default invalid-ROI exclusion when loading experiments.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns all streams to ophys frame timestamps first, then defines each trial by the ophys frames whose timestamps fall between the trial `start_time` and `stop_time`. There is no additional re-alignment to `change_time`.

ii.
```python
ophys_ts = f['processing/ophys/dff/traces/timestamps'][:]
...
t_start = nwb_data['trial_start_times'][trial_idx]
t_stop = nwb_data['trial_stop_times'][trial_idx]
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]
...
trial_neural = events[frame_indices, :].T
```

iii. The notes justify this by saying all data are put on the ophys timestamp timebase and then segmented by trial boundaries from the task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Each trial stays at the experiment’s native ophys frame rate. Because the AI mixes single-plane and multiscope experiments, this means some sessions are about 31 Hz and some about 11 Hz. Metadata reports a single `time_bin_size` computed from the median imaging rate across sessions.

ii.
```python
data['imaging_rate'] = f['general/optophysiology/imaging_plane_1/imaging_rate'][()]
...
imaging_rates = [s['imaging_rate_hz'] for s in session_info]
median_rate = np.median(imaging_rates)
time_bin_ms = 1000.0 / median_rate
...
'time_bin_size': time_bin_ms,
```

iii. The notes justify this as keeping the native sampling and acknowledging both ~31 Hz single-plane and ~11 Hz multiscope data, with metadata summarizing the median frame interval.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation interval table, specifically `image_name`, `start_time`, `stop_time`, and `omitted`. It is not derived from the trial table’s `initial_image_name` and `change_image_name`.

ii.
```python
stim = f[f'intervals/{stim_key}']
data['stim_start_times'] = stim['start_time'][:]
data['stim_stop_times'] = stim['stop_time'][:]
data['stim_image_names'] = np.array([s.decode() if isinstance(s, bytes) else s for s in stim['image_name'][:]])
data['stim_omitted'] = stim['omitted'][:]
```

iii. The notes justify this by saying the output should reflect “the image presented during the non-grey screen,” so the exact stimulus interval table is used rather than only trial-level before/after change labels.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI builds a full-session image index at ophys resolution by marking stimulus intervals, skipping omitted flashes, and then carrying the last shown image forward through grey-screen periods. Within each trial, any leading grey frames are backfilled from the first non-grey image. Image names are then mapped to sorted global integer codes.

ii.
```python
image_at_ophys = np.full(len(ophys_ts), -1, dtype=int)
...
for i in range(len(stim_starts)):
    if stim_omitted[i] == 1.0:
        continue
    ...
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    image_at_ophys[mask] = img_idx
...
trial_image = image_at_ophys[frame_indices]
...
for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
    elif last_img >= 0:
        trial_image[k] = last_img
```

iii. The notes justify the carry-forward rule by saying omitted and grey periods should preserve the identity of the last shown non-grey image for continuity.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is first mapped onto the same ophys timestamps used for neural data, and trial slices use the exact same `frame_indices` as the neural matrix.

ii.
```python
mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
image_at_ophys[mask] = img_idx
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]
...
trial_image = image_at_ophys[frame_indices]
trial_neural = events[frame_indices, :].T
```

iii. The notes justify this as putting every stream onto ophys timestamps before trial slicing so that image identity and neural data share a common frame index.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation `is_change` flag together with the stimulus interval `start_time` and `stop_time`. It is not derived from the trial table’s `change_time`.

ii.
```python
data['stim_is_change'] = stim['is_change'][:]
...
is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
stim_is_change = nwb_data['stim_is_change']
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        is_change_at_ophys[mask] = 1
```

iii. The notes justify this as directly marking the stimulus presentation during which an image change actually occurs.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI creates a full-session binary time series at ophys resolution and sets it to 1 only during stimulus intervals whose `is_change` flag is true. Within each trial it simply slices that binary series.

ii.
```python
is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
...
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        is_change_at_ophys[mask] = 1
...
trial_change = is_change_at_ophys[frame_indices]
```

iii. The notes justify this by defining image change as the stimulus period in which the changed image is shown, rather than a trial-level before/after marker.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is applied beyond turning it into a binary variable with values 0 and 1.

ii.
```python
output_combined = np.stack([
    img_id.astype(np.int64),
    img_change.astype(np.int64),
    running_binned.astype(np.int64),
    pupil_binned.astype(np.int64),
    np.full(n_t, outcome, dtype=np.int64),
], axis=0)
...
['no_change', 'change']
```

iii. The notes explicitly describe image change as a binary 0/1 signal.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned exactly like image identity: the change signal is defined on ophys timestamps and then indexed by the same trial frame indices as the neural data.

ii.
```python
mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
is_change_at_ophys[mask] = 1
...
frame_indices = np.where(frame_mask)[0]
trial_change = is_change_at_ophys[frame_indices]
trial_neural = events[frame_indices, :].T
```

iii. The notes justify this as part of the general “all streams on ophys timestamps” alignment strategy.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the NWB running processing group: `processing/running/speed/data` and `processing/running/speed/timestamps`.

ii.
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. The notes justify this as using the standard running-wheel signal supplied in the dataset.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated with `np.interp` onto ophys timestamps. Global percentile edges are then computed from all interpolated running samples across the filtered experiments, and each trial’s running values are digitized into 5 bins.

ii.
```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts):
    return np.interp(ophys_ts, signal_ts, signal)
...
running = interpolate_to_ophys(
    nwb_data['running_speed'],
    nwb_data['running_timestamps'],
    nwb_data['ophys_timestamps']
)
all_running.append(running)
...
running_edges = compute_percentile_bins(all_running_flat, n_bins=5)
...
running_binned = digitize_to_bins(running, running_edges)
```

iii. The notes justify this as synchronizing behavior to ophys timestamps and using equal-percentile bins to make the decoder target balanced.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five global percentile bins using `np.percentile` and `np.digitize`.

ii.
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    return edges
...
running_binned = digitize_to_bins(running, running_edges)
```

iii. The notes justify this as implementing the requested five equal-percentile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the ophys timebase before trial segmentation, then sliced with the same trial frame indices as the neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'],
    nwb_data['running_timestamps'],
    ophys_ts
)
...
trial_running = running_at_ophys[frame_indices]
trial_neural = events[frame_indices, :].T
```

iii. The notes justify this with the general choice to use ophys timestamps as the common alignment clock.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the eye-tracking pupil width signal, together with eye-tracking timestamps and the `likely_blink` mask.

ii.
```python
data['pupil_width'] = pupil['width'][:]
...
data['eye_timestamps'] = eye_ts
data['likely_blink'] = f['acquisition/EyeTracking/likely_blink/data'][:]
```

iii. The notes justify using pupil width as the diameter measure and the blink mask to remove bad eye-tracking samples.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI copies pupil width, sets blink samples to `NaN`, interpolates the remaining valid samples to ophys timestamps with `np.interp`, computes global percentile edges across all valid pupil values, and digitizes each trial into 5 bins. If a session has no usable pupil data, the code fills that trial with the median bin `2`.

ii.
```python
pupil_raw = nwb_data['pupil_width'].copy()
blink_mask = nwb_data['likely_blink']
if blink_mask is not None:
    pupil_raw[blink_mask] = np.nan
...
valid_mask = ~np.isnan(pupil_raw)
if np.sum(valid_mask) > 10:
    pupil_at_ophys = np.interp(
        ophys_ts,
        nwb_data['eye_timestamps'][valid_mask],
        pupil_raw[valid_mask]
    )
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
if pupil_edges is not None and not np.all(np.isnan(pupil)):
    pupil_binned = digitize_to_bins(pupil, pupil_edges)
else:
    pupil_binned = np.full(len(pupil), 2, dtype=int)
```

iii. The notes justify blink removal before interpolation and describe the median-bin fallback for sessions without usable pupil data.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five global percentile bins using percentile edges computed from all valid interpolated pupil samples.

ii.
```python
if all_pupil:
    all_pupil_flat = np.concatenate(all_pupil)
    pupil_edges = compute_percentile_bins(all_pupil_flat, n_bins=5)
...
pupil_binned = digitize_to_bins(pupil, pupil_edges)
```

iii. The notes justify this as implementing the required five equal-percentile pupil categories.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to ophys timestamps before trial segmentation, and trial slices use the same frame indices as the neural data.

ii.
```python
pupil_at_ophys = np.interp(
    ophys_ts,
    nwb_data['eye_timestamps'][valid_mask],
    pupil_raw[valid_mask]
)
...
trial_pupil = pupil_at_ophys[frame_indices]
trial_neural = events[frame_indices, :].T
```

iii. The notes justify this with the common-clock alignment strategy based on ophys timestamps.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome comes from the NWB trial booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
data['trial_hit'] = trials['hit'][:]
data['trial_miss'] = trials['miss'][:]
data['trial_false_alarm'] = trials['false_alarm'][:]
data['trial_correct_reject'] = trials['correct_reject'][:]
...
if nwb_data['trial_hit'][trial_idx]:
    return 0
elif nwb_data['trial_miss'][trial_idx]:
    return 1
elif nwb_data['trial_false_alarm'][trial_idx]:
    return 2
elif nwb_data['trial_correct_reject'][trial_idx]:
    return 3
```

iii. The notes justify this as the standard trial outcome taxonomy for the change-detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four outcome booleans to fixed integer codes `0..3` and then repeats the resulting code across every timepoint of the trial when building the output matrix.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
output_combined = np.stack([
    img_id.astype(np.int64),
    img_change.astype(np.int64),
    running_binned.astype(np.int64),
    pupil_binned.astype(np.int64),
    np.full(n_t, outcome, dtype=np.int64),
], axis=0)
```

iii. The notes justify this as a static per-trial label that is replicated over time to fit the decoder’s time-varying output format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing or awkward data in several ad hoc ways: it skips trials with too few frames or no valid stimulus, skips trials with unknown outcomes, carries the last image forward through grey periods, sets blink-contaminated pupil samples to `NaN` before interpolation, fills all-missing pupil trials with bin `2`, and drops experiments with too few valid trials or no valid neurons.

ii.
```python
if len(frame_indices) < 2:
    continue
...
if trial_image[0] < 0:
    first_valid = np.where(trial_image >= 0)[0]
    if len(first_valid) > 0:
        trial_image[:first_valid[0]] = trial_image[first_valid[0]]
    else:
        continue
...
pupil_raw[blink_mask] = np.nan
...
pupil_binned = np.full(len(pupil), 2, dtype=int)
...
if neural_trials is None or len(neural_trials) < 2:
    ...
if n_neurons < 1:
    ...
```

iii. The notes explicitly justify blink masking, carry-forward image identity, and the median-bin fallback for missing pupil data. The remaining guards are practical code-level filters rather than separately documented methodological decisions.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are repeated NWB file I/O and repeated full-session passes over the dataset: one pass to collect image names, one to collect running and pupil samples for global binning, and one to do the final trial extraction. Within each loaded experiment, building full-session image and change series with per-stimulus boolean masks is also relatively expensive.

ii.
```python
for _, row in exp_table.iterrows():
    ...
    with h5py.File(nwb_path, 'r') as f:
        ...
...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...
    nwb_data = load_nwb_data(nwb_path)
    ...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...
    nwb_data = load_nwb_data(nwb_path)
```

iii. There is no explicit performance discussion in the notes, so this is inferred from the structure of the code: the same NWB files are opened multiple times and each pass touches large time-series arrays.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are left unvectorized: iterating over all stimulus intervals to fill `image_at_ophys` and `is_change_at_ophys`, iterating frame-by-frame to carry forward image identity through grey periods, and repeated row-wise `iterrows()` passes over the experiment table.

ii.
```python
for i in range(len(stim_starts)):
    ...
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    image_at_ophys[mask] = img_idx
...
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        is_change_at_ophys[mask] = 1
...
for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
```

iii. No explicit justification is documented. The code comments instead emphasize clarity and behavioral meaning, especially for grey-period carry-forward.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats NWB loading and some preprocessing in multiple passes. It scans all experiments once for image names, once for global running/pupil statistics, and once again for final processing. Running and pupil interpolation are also performed in the statistics pass and then recomputed inside `process_experiment`.

ii.
```python
# Collect all image names across all experiments
for _, row in exp_table.iterrows():
    ...
    with h5py.File(nwb_path, 'r') as f:
        ...
...
# First pass: collecting behavioral statistics for binning...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...
    running = interpolate_to_ophys(...)
...
# Second pass: processing experiments...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...
    result = process_experiment(nwb_data, all_image_names)
```

iii. No explicit justification is given beyond the need for global percentile bin edges. The notes do not acknowledge that this requires repeated full-dataset reads and duplicated interpolation work.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and stores dF/F traces only for fallback/comparison but never uses them, loads `events_timestamps` without checking them against ophys timestamps, returns several full-session arrays from `process_experiment` that are not used downstream, and accumulates `trial_outcomes` in a list that is never used.

ii.
```python
dff = f['processing/ophys/dff/traces/data'][:]
data['dff'] = dff  # shape: (n_timepoints, n_cells)
...
events_ts = f['processing/ophys/event_detection/timestamps'][:]
data['events_timestamps'] = events_ts
...
trial_outcomes = []
...
trial_outcomes.append(outcome)
...
return neural_trials, output_trials, trial_outcomes, running_at_ophys, pupil_at_ophys, events
...
neural_trials, raw_output_trials, trial_outcomes, running_at_ophys, pupil_at_ophys, events_full = result
```

iii. The only explicit justification is the comment “for fallback / comparison” on dF/F loading. The rest appears to be leftover intermediate-state handling rather than an intentional downstream requirement.
