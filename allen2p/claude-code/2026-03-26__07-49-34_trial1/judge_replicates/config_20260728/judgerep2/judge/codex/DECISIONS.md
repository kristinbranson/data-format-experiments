# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the AllenSDK cache. It reads local metadata CSVs with `pandas`, discovers available experiments by scanning downloaded NWB filenames, filters to those downloaded files, removes sessions whose `session_type` contains `passive`, then loads each NWB directly with `h5py`.

ii.
```python
def load_experiment_table():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    downloaded_ids = set()
    for f in nwb_files:
        eid = int(os.path.basename(f).replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
        downloaded_ids.add(eid)
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)].copy()
    exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)].copy()
    exp_table = exp_table.sort_values('ophys_experiment_id').reset_index(drop=True)
    return exp_table

def load_nwb_data(nwb_path):
    with h5py.File(nwb_path, 'r') as f:
        data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
        dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
        data['dff_traces'] = dff_raw.T
```

iii. In `CONVERSION_NOTES.md`, Step 6 says the script "Loads NWB files directly via h5py (fast, no AllenSDK overhead)." Step 4 and Step 5 justify using all active downloaded sessions and handling Multiscope data directly from the local files.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are unique `mouse_id` values in the filtered experiment table. A subject index is assigned the first time each mouse appears during the experiment loop.

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

iii. Step 5 of `CONVERSION_NOTES.md` maps `mouse_id` to `subjects` and `subject_idx`, explicitly treating unique mice as subjects.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` as one output session. It does not group multiple experiments that share an `ophys_session_id`; Multiscope planes are therefore kept as separate sessions.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(
        exp_id, row, image_names_list, outcome_names,
        show_processing=args.show_processing
    )
    ...
    all_neural.append(result['neural'])
    all_output.append(result['output'])
```

iii. Step 5 of `CONVERSION_NOTES.md` states: "Multiscope handling: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data."

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table. For each valid trial, the AI uses `start_time` and `stop_time` and extracts all ophys frames with timestamps in `[start_time, stop_time)`, producing variable-length trials.

ii.
```python
valid_trial_idx = get_valid_trials(trial_data)
...
for trial_idx in valid_trial_idx:
    t_start = trial_data['start_time'][trial_idx]
    t_stop = trial_data['stop_time'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    trial_ts = ophys_ts[frame_mask]
    n_trial_frames = frame_mask.sum()
    if n_trial_frames < 2:
        continue
```

iii. Step 5 of `CONVERSION_NOTES.md` says: "Trial definition: Use `start_time` and `stop_time` from trials table for Go and Catch trials only." The notes repeatedly describe extracting the ophys frames between those trial bounds.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps `go` and `catch` trials and excludes `aborted` and `auto_rewarded` trials. It also skips experiments with fewer than 2 valid trials and skips individual trials with fewer than 2 ophys frames.

ii.
```python
def get_valid_trials(trial_data):
    go = trial_data['go'].astype(bool)
    catch = trial_data['catch'].astype(bool)
    aborted = trial_data['aborted'].astype(bool)
    auto_rewarded = trial_data['auto_rewarded'].astype(bool)
    valid = (go | catch) & ~aborted & ~auto_rewarded
    return np.where(valid)[0]
...
if len(valid_trial_idx) < 2:
    return None
...
if n_trial_frames < 2:
    continue
```

iii. Step 3 and Step 5 of `CONVERSION_NOTES.md` call out the decoder-task curation rule to include Go and Catch and exclude Aborted and Auto-rewarded trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB dF/F traces in `processing/ophys/dff/traces/data`, together with the corresponding ophys timestamps.

ii.
```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. Step 5 of `CONVERSION_NOTES.md` says "Neural data: Use dF/F traces (not events/deconvolved). dF/F is the standard calcium imaging signal."

## 2-b. How is the `neural` data processed?

i. The AI transposes the raw NWB trace matrix to `(n_cells, n_frames)`, then slices trial windows directly from that array. It does not deconvolve, normalize further, merge planes, or resample.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
...
neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
```

iii. Step 5 of `CONVERSION_NOTES.md` explicitly chooses dF/F rather than events. Step 6 says the conversion "Handles both Scientifica (~31 Hz) and Multiscope (~11 Hz) sessions," implying the traces are kept at their native sampling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neural QC filter is applied in the script. The code only checks for experiments with zero cells and otherwise keeps all traces present in the NWB dataset.

ii.
```python
n_cells, n_frames = dff.shape

if n_cells == 0:
    print(f"  WARNING: No cells in experiment {exp_id}")
    return None
```

iii. `CONVERSION_NOTES.md` Step 1 notes that AllenSDK normally exposes `exclude_invalid_rois=True` by default, but the final script bypasses that path and does not add a corresponding ROI-validity filter itself.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to ophys timestamps within each trial window. In practice, the extracted trial starts at `trial start_time` and ends at `trial stop_time`, using the subset of imaging frames whose timestamps fall inside that interval.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. Step 5 of `CONVERSION_NOTES.md` says: "Temporal alignment: Align to ophys timestamps. For each trial, extract the ophys frames between trial `start_time` and `stop_time`."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps native ophys frame timing per experiment and does not rebin the traces. It computes `dt` as the median inter-frame interval of each experiment, but stores a single median `time_bin_size` across all output sessions.

ii.
```python
dt = np.median(np.diff(ophys_ts))  # time bin size
...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
...
'time_bin_size': median_dt,
```

iii. Step 5 of `CONVERSION_NOTES.md` says: "Time bin: Use native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope)." The notes justify supporting both frame rates without rebinning.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation interval table, specifically each presentation's `start_time`, `stop_time`, and `image_name`. It is not derived from the trial table's `initial_image_name` and `change_image_name`.

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

iii. Step 5 of `CONVERSION_NOTES.md` maps "`image_name` from stimulus presentations" to the image-identity output and says it will be "time-varying at ophys rate."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI first builds a global list of image names by scanning all experiments, prepends a `gray` category, and then, for each trial, initializes every frame to `gray` and overwrites frames that overlap actual stimulus presentations with the corresponding image ID. `omitted` stimuli are treated as gray.

ii.
```python
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

iii. Step 5 of `CONVERSION_NOTES.md` says: "Image identity: Map stimulus presentations to ophys timepoints. During gray screen (ISI), use a 'gray' category." Step 7 says the gray-screen fraction looked correct and was used as a sanity check.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is constructed directly on the ophys frame timestamps within each trial, so the image-identity vector has one value per neural frame in that trial.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The design in Step 5 of `CONVERSION_NOTES.md` explicitly says to "Map stimulus presentations to ophys timepoints," so image identity is intended to share the ophys time base with neural activity.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table's `is_change` flags and presentation `start_time`s, not from the trial table's `change_time`.

ii.
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']

for si in range(len(stim_starts)):
    if not is_change[si]:
        continue
    s_start = stim_starts[si]
```

iii. Step 5 of `CONVERSION_NOTES.md` maps "`is_change` from stimulus presentations" to the image-change output and describes it as "1 at change onset frame, 0 otherwise."

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI builds a binary vector per trial and sets a single 1 at the first ophys frame at or after any stimulus presentation marked `is_change=True` that falls inside the trial window.

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
...
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. Step 5 of `CONVERSION_NOTES.md` justifies this as an onset-style event code: "`image_change` ... Binary, 1 at change onset frame, 0 otherwise."

## 4-c. How is `output` *Image change* thresholded into categories?

i. There is no additional thresholding beyond making the trace binary. The two categories are fixed as `0 = no_change` and `1 = change`.

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
...
output_values = [
    image_names_list,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
```

iii. Step 5 of `CONVERSION_NOTES.md` defines image change as a binary variable rather than a continuous quantity.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed on the trial's ophys timestamps and therefore has one value per neural frame.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_idx = np.searchsorted(trial_ts, s_start)
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes consistently describe temporal alignment on the ophys time base, so the event is intended to be frame-aligned with neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the NWB running-speed stream: `processing/running/speed/data` with its own timestamps.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. Step 5 of `CONVERSION_NOTES.md` maps `running_speed` directly to the decoder output and states it will be interpolated to ophys time.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed from wheel timestamps to the full ophys timestamp vector for the experiment, then computes percentile edges from that experiment's ophys-aligned running values and applies those per-trial.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
...
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
...
running_trial = running_at_ophys[frame_mask]
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. Step 5 of `CONVERSION_NOTES.md` says: "Running speed: Interpolate from 60 Hz to ophys timestamps using linear interpolation." It also says percentile bins are computed across the entire session and then applied per trial.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five equal-percentile bins, but the percentiles are computed separately for each experiment/session. Missing values are assigned to bin 0.

ii.
```python
def compute_session_percentile_edges(values, n_bins=5):
    valid = values[~np.isnan(values)]
    ...
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges

result[~valid] = 0
```

iii. Step 5 of `CONVERSION_NOTES.md` explicitly chooses "5 equal bins" computed over the whole session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the ophys time base before trial segmentation, then sliced with the same per-trial frame mask as the neural traces.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
...
running_trial = running_at_ophys[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The Step 5 plan in `CONVERSION_NOTES.md` says to interpolate running to ophys timestamps specifically so the output is aligned to neural frames.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking `area` values plus the `likely_blink` mask and eye-tracking timestamps. The AI does not use the SDK's precomputed `pupil_width`.

ii.
```python
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. Step 5 of `CONVERSION_NOTES.md` says: "`pupil_tracking/area` → `pupil_diameter`: Compute diameter, interpolate to ophys timestamps, discretize into 5 equal percentile bins."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink-marked samples are set to NaN, pupil diameter is computed from area as `2 * sqrt(area / pi)`, that signal is linearly interpolated to ophys timestamps, and percentile bins are computed per experiment/session and applied per trial.

ii.
```python
pupil_area = nwb_data['pupil_area'].copy()
likely_blink = nwb_data['likely_blink']
...
pupil_area[likely_blink] = np.nan
...
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
...
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
```

iii. Step 5 of `CONVERSION_NOTES.md` gives the same formula and says blinks are set to NaN before interpolation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five equal-percentile bins, computed separately within each experiment/session. Missing values are assigned to bin 0.

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
...
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
...
result[~valid] = 0
```

iii. Step 5 of `CONVERSION_NOTES.md` says the bins are session-wide percentile bins with 5 levels.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the full ophys timeline and then sliced with the same trial frame mask used for neural data.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
pupil_trial = pupil_at_ophys[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes repeatedly justify putting behavioral variables on the ophys time base so they can be decoded from frame-aligned neural activity.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table boolean fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
def get_trial_outcome(trial_data, idx):
    if trial_data['hit'][idx]:
        return 'hit'
    elif trial_data['miss'][idx]:
        return 'miss'
    elif trial_data['false_alarm'][idx]:
        return 'false_alarm'
    elif trial_data['correct_reject'][idx]:
        return 'correct_reject'
```

iii. Step 5 of `CONVERSION_NOTES.md` maps trial outcome to a 4-category output: "Trial outcome (hit/miss/FA/CR)."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The string label is converted to an integer by indexing into `outcome_names`, and that scalar outcome code is then broadcast across all frames of the trial in the final output matrix.

ii.
```python
outcome = get_trial_outcome(trial_data, trial_idx)
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
...
output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
...
output_full[4] = outcome_idx
```

iii. Step 5 of `CONVERSION_NOTES.md` describes trial outcome as a categorical static-per-trial variable; the final implementation represents that static value by repeating it across time bins.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI skips experiments when files, cells, stimulus data, or valid trials are missing; skips trials with fewer than 2 frames; fills absent pupil data with all-NaN arrays; assigns NaNs to the lowest percentile bin; and leaves omitted stimuli as gray by not overwriting the default gray trace.

ii.
```python
if not os.path.exists(nwb_path):
    return None
...
if n_cells == 0:
    return None
...
if stim_data is None:
    return None
...
if len(valid_trial_idx) < 2:
    return None
...
if n_trial_frames < 2:
    continue
...
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
result[~valid] = 0
```

iii. The notes describe these as practical safeguards for a local downloaded subset. No separate general-purpose exception handling rationale was documented beyond those checks.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive steps are repeated NWB reads and the prescan across all NWB files to collect image names, followed by per-experiment trial processing.

ii.
```python
all_image_names = get_all_image_names(exp_table)
...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(...)
```

iii. Step 7 of `CONVERSION_NOTES.md` reports approximate runtime and explicitly says "Image name collection adds ~14s overhead" and that NWB loading dominates per-session time.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the loops over stimulus presentations in `build_image_identity_trace` and `build_image_change_trace`, the repeated experiment-wide scan in `get_all_image_names`, and the per-trial loop inside `process_experiment`.

ii.
```python
for si in range(len(stim_starts)):
    ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
...
for si in range(len(stim_starts)):
    if not is_change[si]:
        continue
...
for trial_idx in valid_trial_idx:
    ...
```

iii. No explicit written justification for keeping these loops was recorded. The code and runtime notes imply readability and local-file I/O were prioritized over deeper vectorization.

## 9-c. What processing does the code repeat multiple times?

i. The code makes a full prescan of all experiments to build the global image-name list before processing experiments again; it also rebuilds per-trial frame masks separately for image identity, image change, running, pupil, and neural slicing.

ii.
```python
all_image_names = get_all_image_names(exp_table)
...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(...)
```

```python
img_trace, _ = build_image_identity_trace(
    ophys_ts, stim_data, t_start, t_stop, image_names_list
)
change_trace = build_image_change_trace(ophys_ts, stim_data, t_start, t_stop)
running_trial = running_at_ophys[frame_mask]
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. Step 7 of `CONVERSION_NOTES.md` explicitly notes the extra image-name collection overhead. No separate justification was given for the repeated per-trial masking.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads `cell_roi_ids` but never uses them, computes `output_tv` and `output_static` and then discards them, returns `trial_mask` from `build_image_identity_trace` but ignores it, and optionally stores large debug-only arrays for plotting that are not part of the final converted dataset.

ii.
```python
if 'image_segmentation' in f['processing']['ophys']:
    ...
        data['cell_roi_ids'] = seg[key]['id'][:]
```

```python
output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
output_static = np.array([outcome_idx], dtype=np.int64)
...
img_trace, _ = build_image_identity_trace(...)
```

```python
if show_processing:
    result['ophys_ts'] = ophys_ts
    result['running_at_ophys'] = running_at_ophys
    result['pupil_at_ophys'] = pupil_at_ophys
```

iii. No explicit justification was written for these discarded intermediates. They appear to be leftovers from development and visualization work documented in Step 6 and Step 7.
