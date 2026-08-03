# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads experiment-level metadata from `ophys_experiment_table.csv`, filters it to NWB files that are physically present, then filters to `passive == False`. It then loops over every retained `ophys_experiment_id`, opens the corresponding NWB file with `h5py`, and reads the neural, stimulus, trial, running, and eye-tracking groups directly from the NWB. It also makes separate dataset-wide passes to collect image names and percentile-bin statistics before the final conversion pass.

ii. 
```python
def load_experiment_table():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
    nwb_ids = [int(f.split('_')[-1].split('.')[0]) for f in nwb_files]
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(nwb_ids)]
    exp_table = exp_table[exp_table['passive'] == False]
    return exp_table
```

```python
for _, row in exp_table.iterrows():
    eid = row['ophys_experiment_id']
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
    nwb_data = load_nwb_data(nwb_path)
```

iii. In `CONVERSION_NOTES.md`, the agent says it used the Allen Visual Behavior 2P NWB files directly and filtered to active behavior only. In the trajectory, it also justified direct `h5py` reads as mirroring the SDK “under the hood.”

## 1-b. How are the data split into subjects?

i. Subjects are split by `mouse_id` from the experiment table. The code converts mouse IDs to strings, stores all unique mice in sorted order, and assigns each retained session an integer `subject_idx`.

ii. 
```python
all_subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}

for idx, (_, row) in enumerate(exp_table.iterrows()):
    mouse_id = str(row['mouse_id'])
    ...
    subject_idx_list.append(subject_to_idx[mouse_id])
```

iii. `CONVERSION_NOTES.md` reports summary counts in “mice,” and the trajectory consistently discusses filtering and sampling at the mouse level.

## 1-c. How are the data split into sessions?

i. The agent treats each NWB file, i.e. each `ophys_experiment_id`, as one output session. It does not merge multiple experiments that belong to the same `ophys_session_id`, so multiscope recording sessions remain split into separate plane-level sessions.

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
    input_all.append(final_input)
    output_all.append(final_output_trials)
```

iii. In trajectory step 31, the agent explicitly states “Each experiment = 1 session in the output.” In step 34, it notes that multiscope sessions can have multiple experiments but still keeps native experiment files separate.

## 1-d. How are the data split into trials?

i. Trials are split using the NWB `intervals/trials` table. For each included trial, the code uses `start_time` and `stop_time` to find all ophys frames with timestamps in `[start_time, stop_time)`, then slices neural and output streams to those frame indices.

ii. 
```python
t_start = nwb_data['trial_start_times'][trial_idx]
t_stop = nwb_data['trial_stop_times'][trial_idx]

frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]

trial_neural = events[frame_indices, :].T
trial_image = image_at_ophys[frame_indices]
trial_change = is_change_at_ophys[frame_indices]
trial_running = running_at_ophys[frame_indices]
trial_pupil = pupil_at_ophys[frame_indices]
```

iii. `CONVERSION_NOTES.md` says “Trials segmented using `start_time` and `stop_time` from NWB trials table,” with ophys frames extracted inside that interval.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to Go or Catch trials only, and Aborted and Auto-rewarded trials are excluded. The code also drops trials with unknown outcome, fewer than 2 ophys frames, or no valid image assignment after processing.

ii. 
```python
def get_trial_mask(trial_idx, nwb_data):
    if nwb_data['trial_aborted'][trial_idx]:
        return False
    if nwb_data['trial_auto_rewarded'][trial_idx]:
        return False
    if not (nwb_data['trial_go'][trial_idx] or nwb_data['trial_catch'][trial_idx]):
        return False
    return True
```

```python
outcome = get_trial_outcome(trial_idx, nwb_data)
if outcome == -1:
    continue
...
if len(frame_indices) < 2:
    continue
...
if len(first_valid) == 0:
    continue
```

iii. The notes say trial inclusion was “Go and Catch only” and exclusion was “Aborted” plus “Auto-rewarded,” matching the task text. The extra frame-count and image-availability checks are implementation safeguards rather than paper-derived curation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `processing/ophys/event_detection/data`, using the event-detection timestamps and ROI validity table from `processing/ophys/image_segmentation/cell_specimen_table`.

ii. 
```python
ophys_ts = f['processing/ophys/dff/traces/timestamps'][:]
events = f['processing/ophys/event_detection/data'][:]
events_ts = f['processing/ophys/event_detection/timestamps'][:]
...
cst = f['processing/ophys/image_segmentation/cell_specimen_table']
data['valid_roi'] = cst['valid_roi'][:]
```

iii. `CONVERSION_NOTES.md` states that the agent used detected calcium events rather than raw dF/F, and the trajectory says this matched the paper’s use of discrete calcium events.

## 2-b. How is the `neural` data processed?

i. The neural data are not temporally rebinned or transformed beyond selecting event-detection values, filtering invalid ROIs, slicing by trial, transposing from time-by-cell to neuron-by-time, and casting to `float32`.

ii. 
```python
events = nwb_data['events']
valid = nwb_data['valid_roi']
events = events[:, valid]
...
trial_neural = events[frame_indices, :].T
...
neural_trials.append(trial_neural.astype(np.float32))
```

iii. The notes frame this as using “detected calcium events” directly. In the trajectory, the agent contrasts this with dF/F and decides not to use raw fluorescence traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC is limited to excluding ROIs where `valid_roi` is false, skipping experiments with zero remaining neurons, and later skipping experiments with fewer than two valid trials. No additional cell-level filtering is applied in `convert_data.py`.

ii. 
```python
valid = nwb_data['valid_roi']
events = events[:, valid]

if n_neurons == 0:
    return None, None, None, None, None, None
...
if neural_trials is None or len(neural_trials) < 2:
    ...
if n_neurons < 1:
    ...
```

iii. In trajectory step 92, the agent says the SDK default is `exclude_invalid_rois=True`, and that its manual `valid_roi` filtering is intended to match that default behavior.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Neural data are aligned to the experiment’s ophys imaging timestamps, which serve as the common clock for all modalities. Within each trial, the code keeps the subset of event frames whose ophys timestamps fall inside that trial’s time window.

ii. 
```python
ophys_ts = nwb_data['ophys_timestamps']
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]
trial_neural = events[frame_indices, :].T
```

iii. Both the script header and `CONVERSION_NOTES.md` say the pipeline is “temporally aligned to ophys timestamps,” and the trajectory repeats that ophys timestamps are the reference frame.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep each experiment’s native ophys frame times. No temporal rebinning is applied. Despite that, the metadata stores a single `time_bin_size` equal to the median imaging rate across sessions, even though multiscope sessions have coarser sampling.

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

iii. `CONVERSION_NOTES.md` explicitly says the time bin is the inter-frame interval and that metadata reports the median across sessions. In trajectory step 34, the agent notes the mismatch between ~31 Hz single-plane and ~11 Hz multiscope data and decides not to resample.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation interval table selected from the NWB `intervals` group, specifically `image_name`, `start_time`, `stop_time`, and `omitted`.

ii. 
```python
stim = f[f'intervals/{stim_key}']
data['stim_start_times'] = stim['start_time'][:]
data['stim_stop_times'] = stim['stop_time'][:]
data['stim_image_names'] = np.array([s.decode() if isinstance(s, bytes) else s
                                     for s in stim['image_name'][:]])
data['stim_omitted'] = stim['omitted'][:]
```

iii. The notes say image identity is assigned from stimulus-presentation intervals and that omitted stimuli are treated as gray-screen periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code first builds a global mapping from all non-omitted image names to sorted integer IDs. It then labels ophys frames during stimulus presentations with the corresponding image ID, skips omitted presentations, and forward-fills the last seen image across gray intervals within each trial. If a trial begins on gray, it backfills from the first valid image in that trial.

ii. 
```python
all_image_names = sorted(all_image_names_set)
image_name_to_idx = {name: idx for idx, name in enumerate(all_image_names)}

image_at_ophys = np.full(len(ophys_ts), -1, dtype=int)
for i in range(len(stim_starts)):
    if stim_omitted[i] == 1.0:
        continue
    img_name = stim_names[i]
    if img_name == 'omitted':
        continue
    img_idx = image_name_to_idx[img_name]
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    image_at_ophys[mask] = img_idx
```

```python
last_img = -1
for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
    elif last_img >= 0:
        trial_image[k] = last_img
if trial_image[0] < 0:
    first_valid = np.where(trial_image >= 0)[0]
    if len(first_valid) > 0:
        trial_image[:first_valid[0]] = trial_image[first_valid[0]]
```

iii. `CONVERSION_NOTES.md` explicitly justifies the carry-forward rule by saying that during gray screens “the last shown image identity is carried forward,” including omitted stimuli.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is first written onto the ophys timestamp grid and then sliced with the same `frame_indices` used for neural events in each trial, so image labels and neural frames are aligned one-to-one within a trial.

ii. 
```python
mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
image_at_ophys[mask] = img_idx
...
frame_indices = np.where(frame_mask)[0]
trial_image = image_at_ophys[frame_indices]
trial_neural = events[frame_indices, :].T
```

iii. The notes describe all streams as aligned to ophys timestamps, and image identity is described as assigned “for each ophys timepoint.”

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation `is_change` flag together with stimulus `start_time` and `stop_time`.

ii. 
```python
data['stim_start_times'] = stim['start_time'][:]
data['stim_stop_times'] = stim['stop_time'][:]
data['stim_is_change'] = stim['is_change'][:]
```

iii. `CONVERSION_NOTES.md` says image change is “derived from `is_change` field in stimulus presentations table.”

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code initializes a zero vector on the ophys frame grid, then sets it to 1 for every frame whose timestamp falls inside any stimulus interval marked as a change.

ii. 
```python
is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
stim_is_change = nwb_data['stim_is_change']
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        is_change_at_ophys[mask] = 1
```

iii. The notes justify this as a binary signal that is “1 during the change stimulus presentation, 0 otherwise.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. No additional thresholding is applied. The raw `is_change` flag is already binary, so the code keeps it as category 0 or 1 and later casts it to `int64`.

ii. 
```python
if stim_is_change[i] == 1.0:
    ...
...
output_combined = np.stack([
    img_id.astype(np.int64),
    img_change.astype(np.int64),
    ...
], axis=0)
```

iii. The agent’s notes describe image change as intrinsically binary rather than derived from a continuous quantity.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is first placed on the ophys frame grid and then trial-sliced using the same `frame_indices` that define each neural trial.

ii. 
```python
mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
is_change_at_ophys[mask] = 1
...
frame_indices = np.where(frame_mask)[0]
trial_change = is_change_at_ophys[frame_indices]
trial_neural = events[frame_indices, :].T
```

iii. The notes say all outputs are aligned to ophys timestamps before trial extraction.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the NWB running processing module, specifically `processing/running/speed/data` and its timestamps.

ii. 
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. The trajectory and notes both describe running speed as coming from the NWB running-speed stream and being interpolated to ophys time.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to the ophys timestamps. In a first full-dataset pass, the agent collects all interpolated running values and computes global percentile edges. In the second pass, each trial’s interpolated running vector is digitized into those bins.

ii. 
```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts):
    return np.interp(ophys_ts, signal_ts, signal)
```

```python
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

iii. `CONVERSION_NOTES.md` explicitly says running speed is linearly interpolated to ophys timestamps and binned using global percentile cutoffs across all sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into 5 equal-percentile bins computed from all interpolated running-speed values pooled across the full dataset.

ii. 
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    return edges

running_edges = compute_percentile_bins(all_running_flat, n_bins=5)
running_binned = digitize_to_bins(running, running_edges)
```

iii. The notes list the exact five-percentile scheme and the resulting bin edges, indicating this was a deliberate global discretization choice.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation onto the ophys frame times and then sliced by the same per-trial frame indices as the neural matrix.

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

iii. The notes say the common temporal reference is the ophys frame clock, with running speed interpolated onto that clock.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking fields under `acquisition/EyeTracking`, mainly `pupil_tracking/width`, plus `eye_tracking/timestamps` and `likely_blink/data`. The code also reads `pupil_tracking/area`, but does not use it downstream.

ii. 
```python
pupil = f['acquisition/EyeTracking/pupil_tracking']
data['pupil_width'] = pupil['width'][:]
data['pupil_area'] = pupil['area'][:]
eye_ts = f['acquisition/EyeTracking/eye_tracking/timestamps'][:]
data['eye_timestamps'] = eye_ts
data['likely_blink'] = f['acquisition/EyeTracking/likely_blink/data'][:]
```

iii. The notes justify this by stating that pupil width from the fitted ellipse was used as the diameter measure.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code copies pupil width, sets likely-blink samples to `NaN`, discards `NaN` samples for interpolation, and linearly interpolates the remaining valid width values onto the ophys frame grid. If there are too few valid samples, it fills the whole session with `NaN`.

ii. 
```python
pupil_raw = nwb_data['pupil_width'].copy()
blink_mask = nwb_data['likely_blink']
if blink_mask is not None:
    pupil_raw[blink_mask] = np.nan

valid_mask = ~np.isnan(pupil_raw)
if np.sum(valid_mask) > 10:
    pupil_at_ophys = np.interp(
        ophys_ts,
        nwb_data['eye_timestamps'][valid_mask],
        pupil_raw[valid_mask]
    )
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
```

iii. `CONVERSION_NOTES.md` says blink periods are excluded before interpolation and that valid pupil values are interpolated to ophys timestamps.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into 5 equal-percentile bins computed globally from all valid interpolated pupil values. If an experiment has no usable pupil values, the code assigns every frame to the median bin (`2`) instead of leaving the value missing.

ii. 
```python
all_pupil_flat = np.concatenate(all_pupil)
pupil_edges = compute_percentile_bins(all_pupil_flat, n_bins=5)
...
if pupil_edges is not None and not np.all(np.isnan(pupil)):
    pupil_binned = digitize_to_bins(pupil, pupil_edges)
else:
    pupil_binned = np.full(len(pupil), 2, dtype=int)
```

iii. The notes say pupil diameter used global percentile bins and explicitly document the fallback “Sessions without pupil data: assigned median bin (bin 2).”

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil width is aligned by interpolation onto ophys timestamps, then per-trial pupil vectors are extracted using the same frame indices as the neural data.

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

iii. The notes group pupil with running speed as behavioral signals recorded at different rates and interpolated to the ophys frame clock.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB `intervals/trials` table.

ii. 
```python
data['trial_hit'] = trials['hit'][:]
data['trial_miss'] = trials['miss'][:]
data['trial_false_alarm'] = trials['false_alarm'][:]
data['trial_correct_reject'] = trials['correct_reject'][:]
```

```python
if nwb_data['trial_hit'][trial_idx]:
    return 0
elif nwb_data['trial_miss'][trial_idx]:
    return 1
elif nwb_data['trial_false_alarm'][trial_idx]:
    return 2
elif nwb_data['trial_correct_reject'][trial_idx]:
    return 3
```

iii. The notes describe trial outcome as being determined directly from those four trial flags.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code converts the mutually exclusive outcome flags into integer labels `0..3`, skips trials where none of the four outcome flags is set, and then broadcasts the chosen label across all timepoints in the trial when constructing the final output array.

ii. 
```python
outcome = get_trial_outcome(trial_idx, nwb_data)
if outcome == -1:
    continue
...
output_combined = np.stack([
    img_id.astype(np.int64),
    img_change.astype(np.int64),
    running_binned.astype(np.int64),
    pupil_binned.astype(np.int64),
    np.full(n_t, outcome, dtype=np.int64),
], axis=0)
```

iii. `CONVERSION_NOTES.md` calls trial outcome “static per trial,” and the implementation makes it time-constant for compatibility with the decoder format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or problematic values are handled with several ad hoc rules: blink-flagged pupil samples are set to `NaN`; sessions with too little valid pupil data remain all-`NaN`; experiments with no usable pupil values are later assigned the median pupil bin; omitted stimuli are skipped; gray periods are filled with the previous image; trials with no valid image, unknown outcome, or fewer than 2 frames are dropped.

ii. 
```python
if blink_mask is not None:
    pupil_raw[blink_mask] = np.nan
...
if np.sum(valid_mask) > 10:
    pupil_at_ophys = np.interp(...)
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
```

```python
if pupil_edges is not None and not np.all(np.isnan(pupil)):
    pupil_binned = digitize_to_bins(pupil, pupil_edges)
else:
    pupil_binned = np.full(len(pupil), 2, dtype=int)
```

```python
if stim_omitted[i] == 1.0:
    continue
...
elif last_img >= 0:
    trial_image[k] = last_img
...
if outcome == -1:
    continue
if len(frame_indices) < 2:
    continue
```

iii. These behaviors are documented mainly in `CONVERSION_NOTES.md` rather than derived from the prompt. The notes specifically call out blink exclusion, omitted-stimulus handling, and the “median bin” fallback for missing pupil data.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive parts are repeated full-dataset passes over all NWB files and repeated per-file HDF5 reads. The code scans all experiments once to collect image names, a second time to collect running and pupil values for global bin edges, and a third time to build trials. Within each experiment it also loops over every stimulus interval and every trial.

ii. 
```python
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...

for idx, (_, row) in enumerate(exp_table.iterrows()):
    nwb_data = load_nwb_data(nwb_path)
    ...

for idx, (_, row) in enumerate(exp_table.iterrows()):
    nwb_data = load_nwb_data(nwb_path)
    result = process_experiment(nwb_data, all_image_names)
```

```python
for i in range(len(stim_starts)):
    ...
for trial_idx in range(n_trials_total):
    ...
```

iii. The agent did not explicitly benchmark the hotspots, but the control flow in `convert_data.py` and the notes’ “first pass / second pass” description make these costs clear.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious vectorization opportunities are the per-stimulus masking loops for image identity and image change, the per-timepoint forward-fill loop over `trial_image`, and the repeated Python loops over trials when constructing outputs and running validation checks.

ii. 
```python
for i in range(len(stim_starts)):
    ...
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    image_at_ophys[mask] = img_idx
```

```python
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        is_change_at_ophys[mask] = 1
```

```python
for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
    elif last_img >= 0:
        trial_image[k] = last_img
```

iii. This is an inference from the implementation rather than something the agent documented directly.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats NWB loading and interpolation work. Running speed interpolation and pupil interpolation are performed once during the first pass for global percentile estimation and again inside `process_experiment` during the second pass. NWB files are also reopened multiple times for image-name collection and then again for conversion.

ii. 
```python
running = interpolate_to_ophys(
    nwb_data['running_speed'],
    nwb_data['running_timestamps'],
    nwb_data['ophys_timestamps']
)
all_running.append(running)
```

```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'],
    nwb_data['running_timestamps'],
    ophys_ts
)
```

```python
with h5py.File(nwb_path, 'r') as f:
    ...
...
nwb_data = load_nwb_data(nwb_path)
...
nwb_data = load_nwb_data(nwb_path)
```

iii. The notes’ explicit “first pass” and “second pass” structure matches this repeated processing pattern.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads and stores several raw fields that are never used in the final dataset: dF/F traces, event timestamps, cell specimen IDs, trial change times, trial image-name fields, pupil area, and the full `events` matrix returned from `process_experiment`. It also computes `trial_outcomes` and `subjects_list` without using them downstream.

ii. 
```python
dff = f['processing/ophys/dff/traces/data'][:]
data['dff'] = dff
...
data['cell_specimen_ids'] = cst['cell_specimen_id'][:]
...
data['trial_change_times'] = trials['change_time'][:]
data['trial_change_image'] = ...
data['trial_initial_image'] = ...
...
data['pupil_area'] = pupil['area'][:]
```

```python
return neural_trials, output_trials, trial_outcomes, running_at_ophys, pupil_at_ophys, events
...
neural_trials, raw_output_trials, trial_outcomes, running_at_ophys, pupil_at_ophys, events_full = result
```

iii. The agent’s notes do not call these out, but they are visible from the implementation and from which fields are ultimately written into the final output dictionary.
