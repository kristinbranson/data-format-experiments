# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the Allen SDK cache. It loaded `ophys_experiment_table.csv` directly, scanned the local NWB directory for downloaded experiment IDs, filtered out passive sessions, then opened each NWB file with `h5py` and extracted arrays from HDF5 groups. Trials were then processed inside `process_experiment()` from the per-file contents.

ii.
```python
def load_experiment_table():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    ...
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)].copy()
    exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)].copy()

def load_nwb_data(nwb_path):
    with h5py.File(nwb_path, 'r') as f:
        data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
        data['dff_traces'] = f['processing']['ophys']['dff']['traces']['data'][:].T
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as a speed/overhead decision: "Loads NWB files directly via h5py (fast, no AllenSDK overhead)." It also treated the downloaded local subset as the available corpus rather than reconstructing the full Allen SDK project view.

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values from the experiment table. A `subject_map` assigns each mouse string to a subject index the first time it appears.

ii.
```python
subject_map = {}
all_subjects = []
...
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
all_subject_idx.append(subject_map[mouse_id])
```

iii. The notes map `mouse_id` to `subjects`/`subject_idx` explicitly and do not mention any more complex subject definition.

## 1-c. How are the data split into sessions?

i. The AI treated each `ophys_experiment_id` as one output "session". It did not group multiple experiments from the same `ophys_session_id`; instead, each imaging plane became a separate session in the saved dataset.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    ...
    all_neural.append(result['neural'])
    session_metadata.append({
        'exp_id': result['exp_id'],
        'ophys_session_id': result['ophys_session_id'],
```

iii. The explicit rationale in `CONVERSION_NOTES.md` was: "Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data."

## 1-d. How are the data split into trials?

i. Trials are defined from the NWB `intervals/trials` table. For each valid trial, the code uses the full `start_time` to `stop_time` window and includes all ophys frames whose timestamps fall within that interval.

ii.
```python
for trial_idx in valid_trial_idx:
    t_start = trial_data['start_time'][trial_idx]
    t_stop = trial_data['stop_time'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    n_trial_frames = frame_mask.sum()
    if n_trial_frames < 2:
        continue
```

iii. The notes say: "Use `start_time` and `stop_time` from trials table for Go and Catch trials only" and "For each trial, extract the ophys frames between trial start_time and stop_time."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `(go | catch) & ~aborted & ~auto_rewarded`. Trials with fewer than 2 aligned ophys frames are dropped, and entire experiments are dropped if they have fewer than 2 remaining trials. There is no explicit `change_time.notna()` filter.

ii.
```python
def get_valid_trials(trial_data):
    go = trial_data['go'].astype(bool)
    catch = trial_data['catch'].astype(bool)
    aborted = trial_data['aborted'].astype(bool)
    auto_rewarded = trial_data['auto_rewarded'].astype(bool)
    valid = (go | catch) & ~aborted & ~auto_rewarded
    return np.where(valid)[0]

if n_trial_frames < 2:
    continue
...
if len(valid_trial_idx) < 2:
    return None
```

iii. The notes justify this with the task instructions: include Go and Catch trials, exclude Aborted and Auto-rewarded trials. No extra justification was found for omitting the reference solution's `change_time` validity filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB dF/F array at `processing/ophys/dff/traces/data`, plus the matching ophys timestamps from `processing/ophys/dff/traces/timestamps`.

ii.
```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. The notes say dF/F is already precomputed in the NWB files and should be used directly rather than recomputed.

## 2-b. How is the `neural` data processed?

i. Processing is minimal: the dF/F matrix is transposed to `(cells, frames)`, then trial windows are sliced out with a boolean frame mask and cast to `float32`. The AI did not merge multiple planes belonging to the same behavioral session.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T
...
neural = dff[:, frame_mask].astype(np.float32)
neural_trials.append(neural)
```

iii. `CONVERSION_NOTES.md` says: "Neural data: Use dF/F traces (not events/deconvolved). dF/F is the standard calcium imaging signal." It also says each experiment is kept separate rather than merged across multiscope planes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit ROI or neuron-quality filtering in the code. The AI simply loads all cells present in the NWB dF/F dataset, only skipping experiments with zero cells.

ii.
```python
dff = nwb_data['dff_traces']
n_cells, n_frames = dff.shape

if n_cells == 0:
    print(f"  WARNING: No cells in experiment {exp_id}")
    return None
```

iii. The notes acknowledge AllenSDK's `exclude_invalid_rois=True` default, but the AI concluded: "No explicit valid_roi filter ... OK - all ROIs in downloaded NWB files are valid."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to the ophys timebase, then segmented by trial window. Each trial contains all ophys frames from `start_time` inclusive to `stop_time` exclusive.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes say: "Align to ophys timestamps. For each trial, extract the ophys frames between trial start_time and stop_time."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Each experiment stays at its native ophys sampling interval, with `dt` computed from the median frame-to-frame timestamp difference. The final metadata stores the median `dt` across all saved experiments.

ii.
```python
dt = np.median(np.diff(ophys_ts))  # time bin size
...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
...
'time_bin_size': median_dt,
```

iii. The notes justify this as "Use native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope)." The AI explicitly decided to keep both frame-rate regimes rather than resample them to a common grid.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation interval table, specifically `start_time`, `stop_time`, and `image_name` from the selected `Natural_Images...presentations` group. It is not derived from `initial_image_name` / `change_image_name` in the trials table.

ii.
```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    if key in stim:
        stim_data[key] = stim[key][:]
...
stim_starts = stim_data['start_time']
stim_stops = stim_data['stop_time']
stim_names = stim_data['image_name']
```

iii. The notes explicitly planned `image_name` "from stimulus presentations" and not from trial metadata.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code first collects all unique image names across experiments, prepends a special `gray` category, initializes every trial frame to `gray`, then overwrites frames that fall inside stimulus-presentation intervals with the corresponding image code. Omitted flashes remain gray.

ii.
```python
GRAY_LABEL = 'gray'
...
all_image_names = get_all_image_names(exp_table)
image_names_list = [GRAY_LABEL] + all_image_names
...
gray_idx = image_names_list.index(GRAY_LABEL)
trace = np.full(n_frames, gray_idx, dtype=np.int64)
...
if name == 'omitted':
    continue
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. The notes justify this with: "Image identity: Map stimulus presentations to ophys timepoints. During gray screen (ISI), use a 'gray' category."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The image-identity trace is built on the same per-trial ophys timestamps used for the neural slice. Each stimulus interval is projected onto the `trial_ts` frame grid with boolean masks.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. The notes say the variable is "aligned to ophys timestamps," and the code uses the same trial mask for both neural and image outputs.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `image_change` is derived from the stimulus-presentation table's `is_change` flag and each stimulus presentation's `start_time`. It is not derived from trial-table `change_time`.

ii.
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']
...
if not is_change[si]:
    continue
```

iii. The notes planned `is_change` from stimulus presentations as the source for this output.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each stimulus presentation marked `is_change`, the code finds the first ophys frame at or after the change onset and sets only that single frame to 1. All other frames remain 0.

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
...
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. The notes describe this as "Binary, 1 at change onset frame, 0 otherwise." That is the explicit rationale the AI recorded.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary. The code uses `0` for `no_change` and `1` for `change`.

ii.
```python
output_values = [
    image_names_list,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
```

iii. No additional thresholding justification appears beyond the notes' plan to represent change as a binary output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The trace is defined over the same trial-level ophys frames used for `neural`, using `trial_ts` and `np.searchsorted` to place the 1-valued frame.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_idx = np.searchsorted(trial_ts, s_start)
trace[frame_idx] = 1
```

iii. The code uses the same frame mask as the neural slice, so the alignment decision is "align on ophys frames inside the trial window."

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `processing/running/speed/data` and its timestamps from `processing/running/speed/timestamps` in each NWB file.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. The notes describe this as the standard running-speed stream that should be interpolated onto ophys timestamps.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates the continuous running-speed trace to all ophys timestamps for the experiment, computes experiment-wide percentile edges from those interpolated values, and then bins each trial's running values with those edges.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
...
running_trial = running_at_ophys[frame_mask]
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. The notes justify interpolation to ophys timestamps and say percentile bins should be computed "across the entire session (all valid timepoints), then apply per-trial."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The code uses five percentile bins, but the percentiles are computed separately per experiment/session rather than globally across all saved sessions.

ii.
```python
def compute_session_percentile_edges(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
...
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
```

iii. The justification in the notes was to use five equal percentile bins from "the entire session."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first interpolated onto the full ophys timestamp vector, then trial windows are cut out with the same `frame_mask` as the neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(..., ophys_ts)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
running_trial = running_at_ophys[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes explicitly planned "Interpolate from 60 Hz to ophys timestamps using linear interpolation."

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the eye-tracking `area` signal plus `timestamps` and the `likely_blink` mask from `acquisition/EyeTracking`.

ii.
```python
pt = et['pupil_tracking']
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. The notes say: "`pupil_tracking/area` -> diameter" and planned to compute diameter rather than use the precomputed width column described in the SDK path.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples are set to `NaN`, diameter is computed as `2 * sqrt(area / pi)` for positive non-NaN samples, the result is linearly interpolated to ophys timestamps, and then each trial is binned using experiment/session-specific percentile edges.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter = np.full_like(pupil_area, np.nan)
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
```

iii. The notes justify this formula explicitly and say blinks should become NaNs before interpolation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Like running speed, pupil diameter is put into five percentile bins computed separately for each experiment/session; NaNs are assigned to bin 0.

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
...
result = np.zeros(len(values), dtype=np.int64)
result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
result[~valid] = 0
```

iii. The notes state: "Discretize non-NaN values" and use five equal percentile bins over the session-wide values.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the experiment's ophys timestamps first, then the trial-level values are selected with the same `frame_mask` used for neural data.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
pupil_trial = pupil_at_ophys[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The recorded rationale is the same as for running: align all behavior outputs to the ophys timebase before trial extraction.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome comes from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
if trial_data['hit'][idx]:
    return 'hit'
elif trial_data['miss'][idx]:
    return 'miss'
elif trial_data['false_alarm'][idx]:
    return 'false_alarm'
elif trial_data['correct_reject'][idx]:
    return 'correct_reject'
```

iii. The notes map these four canonical outcomes directly to the decoder output.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code converts the boolean outcome columns to one label via `get_trial_outcome`, then converts that label to an index in `outcome_names`. The result is broadcast across all time bins in the trial as a constant fifth output row.

ii.
```python
outcome = get_trial_outcome(trial_data, trial_idx)
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
...
output_full[4] = outcome_idx
```

iii. The notes describe this variable as "Categorical, static per trial," which is why the scalar outcome is repeated across the whole trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases by skipping or filling: missing NWB files, missing stimulus tables, zero-cell experiments, and experiments with too few valid trials are skipped; trials with fewer than 2 frames are skipped; missing pupil data yields all-NaN pupil traces; NaNs after interpolation are assigned to bin 0.

ii.
```python
if not os.path.exists(nwb_path):
    return None
...
if stim_data is None:
    return None
...
if n_cells == 0:
    return None
...
if n_trial_frames < 2:
    continue
...
result[~valid] = 0
```

iii. The notes explicitly defend "NaN pupil values (blinks) mapped to bin 0" as a design choice. For the skip logic, no deeper justification was recorded beyond keeping processing moving.

## 9-a. What are the most time-consuming steps of the code?

i. The code is dominated by opening and reading every NWB file, and it also incurs an extra full pass over NWB files to collect all image names before the main conversion loop.

ii.
```python
def get_all_image_names(exp_table):
    for _, row in exp_table.iterrows():
        with h5py.File(nwb_path, 'r') as f:
            ...

for idx, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
```

iii. `CONVERSION_NOTES.md` explicitly says "Load NWB" is the slowest step and that image-name collection adds overhead.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code leaves several Python loops in place: iterating over every stimulus presentation when building each trial's image trace, iterating again over every stimulus presentation to place change flags, iterating over every experiment to collect image names, and iterating over every experiment/trial in Python. The most obvious vectorizable hot loop inside trial processing is the per-stimulus scan in `build_image_identity_trace`.

ii.
```python
for si in range(len(stim_starts)):
    ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx

for si in range(len(stim_starts)):
    if not is_change[si]:
        continue
```

iii. No explicit vectorization rationale was found in the notes. The code simply chooses readability and direct looping.

## 9-c. What processing does the code repeat multiple times?

i. It repeats a full NWB scan to gather image names before conversion, then reopens every NWB file again for actual conversion. It also repeatedly scans all stimulus presentations for every trial, even though those presentations are session-level data.

ii.
```python
all_image_names = get_all_image_names(exp_table)
...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
```

iii. No explicit justification for this repeated work was recorded. The notes only mention the extra image-name collection overhead.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads `cell_roi_ids` but never uses them, computes `output_tv` and `output_static` and then discards both in favor of `output_full`, and computes optional plotting-related intermediates that are irrelevant unless `--show-processing` is used.

ii.
```python
if 'id' in seg[key]:
    data['cell_roi_ids'] = seg[key]['id'][:]
...
output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
output_static = np.array([outcome_idx], dtype=np.int64)
...
output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
```

iii. No justification for these discarded intermediates was documented. They appear to be leftovers from development while the output representation was being decided.
