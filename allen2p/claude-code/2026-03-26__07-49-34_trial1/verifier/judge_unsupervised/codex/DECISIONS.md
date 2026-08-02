# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local metadata CSV `ophys_experiment_table.csv`, enumerates downloaded NWB files under `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, filters the table down to those experiment IDs and to non-passive sessions, then opens each NWB directly with `h5py`. Within each NWB it loads ophys timestamps, dF/F traces, running speed, pupil data, trials, and stimulus-presentations.

ii. ```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)].copy()
exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)].copy()
...
with h5py.File(nwb_path, 'r') as f:
    data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
    data['dff_traces'] = f['processing']['ophys']['dff']['traces']['data'][:].T
    data['running_speed'] = f['processing']['running']['speed']['data'][:]
    ...
    data['trials'] = trial_data
    data['stimulus'] = stim_data
``` 

iii. `CONVERSION_NOTES.md` says the agent chose direct NWB reads because they are "fast, no AllenSDK overhead" and wanted to use all active downloaded experiments while excluding passive sessions.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the filtered experiment table. Each processed experiment stores its `mouse_id`, and a `subject_map` converts those IDs into `subjects` and `subject_idx`.

ii. ```python
mouse_id = str(exp_row['mouse_id'])
...
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
...
all_subject_idx.append(subject_map[mouse_id])
```

iii. The notes explicitly map `mouse_id` to `subjects`/`subject_idx` and describe unique mice as the subject definition.

## 1-c. How are the data split into sessions?

i. The agent treats each `ophys_experiment_id` as one output session. It iterates row-by-row through the filtered experiment table and appends one entry to `neural`, `input`, `output`, `subject_idx`, and `brain_region_idx` per experiment.

ii. ```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    ...
    all_neural.append(result['neural'])
    all_output.append(result['output'])
    all_subject_idx.append(subject_map[mouse_id])
```

iii. In the notes the agent says "Each experiment (plane) is a separate 'session' in the output" because different planes have different neurons even when they share behavior in Multiscope recordings.

## 1-d. How are the data split into trials?

i. Trials are read from the NWB `intervals/trials` table. For each retained trial, the agent uses the trial’s `start_time` and `stop_time` and extracts all ophys frames satisfying `start_time <= t < stop_time`, producing variable-length trials.

ii. ```python
for trial_idx in valid_trial_idx:
    t_start = trial_data['start_time'][trial_idx]
    t_stop = trial_data['stop_time'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    n_trial_frames = frame_mask.sum()
    if n_trial_frames < 2:
        continue
    neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes say the agent uses `start_time`/`stop_time` from the trials table for Go and Catch trials and aligns everything on ophys timestamps across the full trial window.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `(go | catch) & ~aborted & ~auto_rewarded`. After that, trials with fewer than two ophys frames are skipped, and experiments with fewer than two valid processed trials are discarded.

ii. ```python
go = trial_data['go'].astype(bool)
catch = trial_data['catch'].astype(bool)
aborted = trial_data['aborted'].astype(bool)
auto_rewarded = trial_data['auto_rewarded'].astype(bool)
valid = (go | catch) & ~aborted & ~auto_rewarded
...
if n_trial_frames < 2:
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. The notes justify this directly from the task instructions: include Go and Catch, exclude Aborted and Auto-rewarded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `processing/ophys/dff/traces/data` plus its matching timestamps in `processing/ophys/dff/traces/timestamps`.

ii. ```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes explicitly say the agent chose dF/F traces rather than events because dF/F is the standard precomputed calcium-imaging signal in this dataset.

## 2-b. How is the `neural` data processed?

i. The agent applies minimal processing: transpose the NWB dF/F array to `(n_cells, n_frames)`, then slice trial windows from it and cast to `float32`. There is no deconvolution, normalization, or cross-trial averaging.

ii. ```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes state that dF/F is already precomputed in the NWB files and "no need to compute from scratch."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neuron-level QC filter is applied in `convert_data.py`. All cells present in the dF/F array are kept as long as the experiment has at least one cell.

ii. ```python
n_cells, n_frames = dff.shape
if n_cells == 0:
    print(f"  WARNING: No cells in experiment {exp_id}")
    return None
```

iii. The notes acknowledge AllenSDK’s `exclude_invalid_rois=True` behavior, but the implemented script does not reproduce that filter and instead assumes the NWB dF/F traces are usable as-is.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to ophys timestamps and segmented per trial from trial `start_time` to `stop_time`. The temporal reference is the native ophys timebase, not a fixed window around change onset.

ii. ```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes repeatedly say "Align to ophys timestamps" and describe each trial as the ophys frames between `trial start_time` and `stop_time`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The converted data use the native ophys frame grid for each experiment. The script records `dt` from the median frame interval per experiment and stores the median across experiments in metadata.

ii. ```python
dt = np.median(np.diff(ophys_ts))  # time bin size
...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
...
'time_bin_size': median_dt,
```

iii. The notes explicitly say to use native ophys timestamps, about 31 Hz for Scientifica and 11 Hz for Multiscope, without resampling neural data.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentations interval table: `start_time`, `stop_time`, and `image_name`, plus the trial bounds and ophys timestamps.

ii. ```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    if key in stim:
        stim_data[key] = stim[key][:]
...
stim_starts = stim_data['start_time']
stim_stops = stim_data['stop_time']
stim_names = stim_data['image_name']
```

iii. The notes say image identity should come from stimulus presentations and include a separate gray-screen category during ISIs.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent first builds a global list of image categories across experiments, prepends a `"gray"` label, initializes each trial to gray, then rasterizes overlapping stimulus-presentation intervals onto ophys frames. `"omitted"` presentations remain gray.

ii. ```python
all_image_names = get_all_image_names(exp_table)
image_names_list = [GRAY_LABEL] + all_image_names
...
gray_idx = image_names_list.index(GRAY_LABEL)
trace = np.full(n_frames, gray_idx, dtype=np.int64)
...
if name == 'omitted':
    continue
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. `CONVERSION_NOTES.md` explicitly says the agent chose to map stimulus presentations onto ophys timepoints and use `"gray"` during the 500 ms inter-stimulus interval.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned on the same per-trial ophys frame mask used for neural data, and each stimulus presentation writes into the matching ophys frames of that trial.

ii. ```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. The notes say to "align to ophys timestamps" for all streams and verify that image changes line up with image-identity transitions.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentations table, specifically `is_change` and `start_time`, together with the trial window and ophys timestamps.

ii. ```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']
...
if not is_change[si]:
    continue
```

iii. The notes state that image change should come from stimulus presentations and be aligned to the first frame after a stimulus change.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The agent creates a binary vector of zeros for the trial and sets a single `1` at the first ophys frame at or after each change-onset stimulus presentation.

ii. ```python
trace = np.zeros(n_frames, dtype=np.int64)
...
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. The notes describe image change as a binary event aligned to stimulus onset, not as a prolonged post-change state.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is directly represented as two categories: `0 = no_change`, `1 = change`.

ii. ```python
output_values = [
    image_names_list,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
    [f'bin_{i}' for i in range(5)],
    outcome_names,
]
```

iii. The notes call image change a binary variable and use the same two labels.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned on the same per-trial ophys frame grid as `neural`, with onset determined by `np.searchsorted(trial_ts, change_start)`.

ii. ```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_idx = np.searchsorted(trial_ts, s_start)
trace[frame_idx] = 1
```

iii. The notes say image-change events were checked against image-identity transitions on the ophys-aligned trial plots.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and its `timestamps`.

ii. ```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. The notes identify running speed as the 60 Hz running-wheel signal that should be interpolated to the ophys clock.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The agent linearly interpolates running speed to all ophys timestamps for the experiment, computes percentile edges from the session-wide interpolated values, then bins each trial’s values with those per-session edges.

ii. ```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
...
running_trial = running_at_ophys[frame_mask]
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. The notes explicitly justify linear interpolation to ophys timestamps and using five equal percentile bins computed across the session.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into five equal percentile bins within each experiment/session. NaNs are assigned to bin 0.

ii. ```python
edges = np.percentile(valid, percentiles)
edges[0] = -np.inf
edges[-1] = np.inf
...
result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
result[~valid] = 0
```

iii. The notes say to use 5 equal bins `(0-20, 20-40, ..., 80-100)` computed across the entire session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running is interpolated to the experiment’s ophys timestamp vector before trial segmentation, then trial slices use the same `frame_mask` as neural data.

ii. ```python
running_at_ophys = interpolate_to_ophys(..., ophys_ts)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
running_trial = running_at_ophys[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes explicitly call for interpolation from 60 Hz to ophys timestamps so all streams share one clock.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking `pupil_tracking/area`, `timestamps`, and `likely_blink`.

ii. ```python
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. The notes say the agent chose pupil area and converted it to diameter after removing blink frames.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The agent copies pupil area, sets blink frames to `NaN`, converts valid areas to diameter with `2 * sqrt(area / pi)`, interpolates the resulting signal to ophys timestamps, computes session-wide percentile edges, and bins each trial.

ii. ```python
pupil_area = nwb_data['pupil_area'].copy()
pupil_area[likely_blink] = np.nan
pupil_diameter = np.full_like(pupil_area, np.nan)
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
```

iii. The notes explicitly justify using the area-to-diameter formula, setting blinks to NaN, and then discretizing into five equal percentile bins.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into five equal percentile bins within each experiment/session. NaNs are mapped to bin 0.

ii. ```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
...
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
...
result[~valid] = 0
```

iii. The notes say NaN-to-bin-0 was a deliberate design choice because the task specified exactly five bins and not a separate blink category.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the experiment’s ophys timestamps first and then sliced per trial using the same `frame_mask` as the neural data.

ii. ```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
pupil_trial = pupil_at_ophys[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes say all streams are synchronized to ophys timestamps and that pupil bins were spot-checked against raw NWB data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
if trial_data['hit'][idx]:
    return 'hit'
elif trial_data['miss'][idx]:
    return 'miss'
elif trial_data['false_alarm'][idx]:
    return 'false_alarm'
elif trial_data['correct_reject'][idx]:
    return 'correct_reject'
```

iii. The notes identify those four trial outcomes as the task-relevant categories.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The agent maps the string outcome into a fixed category index and then broadcasts that scalar across all time bins in the trial output array.

ii. ```python
outcome = get_trial_outcome(trial_data, trial_idx)
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
...
output_full[4] = outcome_idx
```

iii. The notes describe trial outcome as "static per trial," but the implementation repeats that static label across time because the saved output array is time-varying.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The script skips missing or unusable experiments (`NWB` missing, no cells, no stimulus, fewer than two valid trials). It skips too-short trials, fills missing pupil with all-NaN when absent, maps NaNs to bin 0 during discretization, and leaves omitted stimuli as gray by not writing any image label for them.

ii. ```python
if not os.path.exists(nwb_path):
    return None
if n_cells == 0:
    return None
if stim_data is None:
    return None
...
if n_trial_frames < 2:
    continue
...
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
result[~valid] = 0
...
if name == 'omitted':
    continue
```

iii. The notes explicitly defend NaN-to-bin-0 for pupil and describe these skip conditions as practical safeguards during conversion.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive work is reading NWB files and scanning them for image names. The agent’s notes estimate NWB load time dominates per experiment, with a separate extra pass for image-name collection.

ii. ```python
all_image_names = get_all_image_names(exp_table)
...
nwb_data = load_nwb_data(nwb_path)
...
't_load': t_load,
't_process': t_process,
```

iii. `CONVERSION_NOTES.md` reports about `~1.7s` load time versus `~0.4s` processing time per session and explicitly says "Image name collection adds ~14s overhead."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization targets: iterating through every NWB file to collect image names, iterating through every stimulus presentation inside each trial to build image traces, iterating through every trial in each experiment, and repeated `list.index(...)` lookups for image names.

ii. ```python
for _, row in exp_table.iterrows():
    ...
for si in range(len(stim_starts)):
    ...
for trial_idx in valid_trial_idx:
    ...
img_idx = image_names_list.index(name)
```

iii. The trajectory and notes focus on practicality rather than optimization, but they explicitly time loading and processing and note the image-name scan as extra overhead.

## 9-c. What processing does the code repeat multiple times?

i. The code rereads NWB files once in `get_all_image_names()` and again in `process_experiment()`. Within each trial it also recomputes the same `trial_mask` separately in `process_experiment()`, `build_image_identity_trace()`, and `build_image_change_trace()`.

ii. ```python
all_image_names = get_all_image_names(exp_table)
...
nwb_data = load_nwb_data(nwb_path)
...
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
...
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
```

iii. The notes themselves mention the extra image-name pass, and the repeated trial-mask construction is visible in the code.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A few values are computed but not used downstream: `cell_roi_ids` are loaded and never consumed, `trial_ts` is created in `process_experiment()` but unused there, and `output_tv` / `output_static` are built and then ignored in favor of `output_full`. Plotting and `show_processing` metadata are also optional diagnostics rather than part of the saved dataset.

ii. ```python
data['cell_roi_ids'] = seg[key]['id'][:]
...
trial_ts = ophys_ts[frame_mask]
...
output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
output_static = np.array([outcome_idx], dtype=np.int64)
...
if show_processing:
    result['ophys_ts'] = ophys_ts
    ...
```

iii. The trajectory shows the agent intentionally added plotting and diagnostics for validation; those computations help debugging but are not needed for downstream decoding.
