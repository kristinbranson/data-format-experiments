# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads a metadata table from `ophys_experiment_table.csv`, filters it to downloaded NWB files and non-passive sessions, then iterates experiment-by-experiment. For each experiment it opens the NWB directly with `h5py` and reads ophys timestamps, dF/F traces, running speed, eye-tracking/pupil data, the trials table, and one image-presentation interval table.

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
        data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
        data['running_speed'] = f['processing']['running']['speed']['data'][:]
        ...
        trials = f['intervals']['trials']
        ...
        stim = f['intervals'][stim_key]
```

iii. In `CONVERSION_NOTES.md` Step 6 the agent says it chose direct NWB reads with `h5py` because it was "fast" and avoided AllenSDK overhead, while Step 10 claims this is equivalent to `BehaviorOphysExperiment.from_nwb()`.

## 1-b. How are the data split into subjects?

i. Subjects are defined by `mouse_id` from the experiment table. The agent converts each `mouse_id` to a string, assigns it a unique integer index the first time it appears, stores the unique subject list in encounter order, and appends one `subject_idx` per output session.

ii. 
```python
brain_region = exp_row['targeted_structure']
mouse_id = str(exp_row['mouse_id'])
...
subject_map = {}
all_subjects = []
...
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
...
all_subject_idx.append(subject_map[mouse_id])
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly maps `mouse_id` to `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. The agent treats each `ophys_experiment_id` as one output session. It does not merge multiple experiments that share the same `ophys_session_id`; instead, each experiment/plane is emitted as a separate session with its own neural array and repeated behavioral data.

ii. 
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    ...
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    ...
    all_neural.append(result['neural'])
    all_output.append(result['output'])
    session_metadata.append({
        'exp_id': result['exp_id'],
        'ophys_session_id': result['ophys_session_id'],
        ...
    })
```

iii. `CONVERSION_NOTES.md` Step 5 says: "Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data."

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table. For each valid trial, the code uses `start_time` and `stop_time` and slices every signal to the ophys frames satisfying `start_time <= t < stop_time`.

ii. 
```python
for key in ['start_time', 'stop_time', 'go', 'catch', 'aborted', 'auto_rewarded',
             'hit', 'miss', 'false_alarm', 'correct_reject', 'change_time',
             'initial_image_name', 'change_image_name', 'is_change']:
    if key in trials:
        trial_data[key] = trials[key][:]
...
for trial_idx in valid_trial_idx:
    t_start = trial_data['start_time'][trial_idx]
    t_stop = trial_data['stop_time'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
```

iii. `CONVERSION_NOTES.md` Step 5 says the trial definition is "Use `start_time` and `stop_time` from trials table."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to Go or Catch trials while excluding Aborted and Auto-rewarded trials. After that, any trial with fewer than 2 ophys frames is skipped, and any experiment with fewer than 2 retained trials is dropped entirely.

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
...
if len(neural_trials) < 2:
    return None
```

iii. This matches the task text verbatim, and `CONVERSION_NOTES.md` Step 3 and Step 5 both restate the same inclusion/exclusion rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `processing/ophys/dff/traces/data` plus the matching `processing/ophys/dff/traces/timestamps` in each NWB file.

ii. 
```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T
...
dff = nwb_data['dff_traces']
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 says: "`dff_traces` -> `neural`" and Step 6 says the script uses dF/F traces rather than events.

## 2-b. How is the `neural` data processed?

i. The agent uses precomputed dF/F without recomputing fluorescence baselines or deconvolution. Processing consists of transposing `(n_frames, n_cells)` to `(n_cells, n_frames)`, slicing each valid trial by frame mask, and casting each trial matrix to `float32`.

ii. 
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
...
neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
```

iii. In `CONVERSION_NOTES.md` Step 1 and Step 3 the agent notes that dF/F is already precomputed in the NWB files. Step 5 says "Use dF/F traces (not events/deconvolved). dF/F is the standard calcium imaging signal."

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit cell-quality filter in `convert_data.py`. The agent effectively uses every trace present in the NWB file, while skipping only experiments with zero cells. It also records `cell_roi_ids` but does not use `valid_roi`.

ii. 
```python
if 'image_segmentation' in f['processing']['ophys']:
    seg = f['processing']['ophys']['image_segmentation']
    for key in seg.keys():
        if 'id' in seg[key]:
            data['cell_roi_ids'] = seg[key]['id'][:]
            break
...
dff = nwb_data['dff_traces']
n_cells, n_frames = dff.shape
if n_cells == 0:
    return None
```

iii. `CONVERSION_NOTES.md` Step 1 acknowledges AllenSDK's `exclude_invalid_rois=True` default, but Step 10 says the script does not explicitly apply it and instead assumes the downloaded NWBs already contain only valid ROIs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to ophys timestamps and cut into trial windows using trial `start_time` and `stop_time`. The agent does not realign trials to `change_time`; the alignment anchor is the trial window on the ophys timebase.

ii. 
```python
ophys_ts = nwb_data['ophys_timestamps']
...
for trial_idx in valid_trial_idx:
    t_start = trial_data['start_time'][trial_idx]
    t_stop = trial_data['stop_time'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    neural = dff[:, frame_mask].astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 says: "Align to ophys timestamps. For each trial, extract the ophys frames between trial `start_time` and `stop_time`."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent uses each experiment's native ophys frame spacing, estimated as `median(diff(ophys_ts))`, with no rebinning. This means sessions remain mixed-rate: roughly 31 Hz for Scientifica and roughly 11 Hz for Multiscope. The saved metadata reports only the median dt across sessions.

ii. 
```python
dt = np.median(np.diff(ophys_ts))  # time bin size
...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
...
'time_bin_size': median_dt,
'ophys_frame_rate_hz': 1000.0 / median_dt,
```

iii. `CONVERSION_NOTES.md` Step 5 says "Use native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope)." Step 10 also explicitly notes different rates for Multiscope sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation table, specifically `start_time`, `stop_time`, and `image_name`, with an added synthetic `gray` label for the inter-stimulus interval.

ii. 
```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    if key in stim:
        stim_data[key] = stim[key][:]
...
gray_idx = image_names_list.index(GRAY_LABEL)
trace = np.full(n_frames, gray_idx, dtype=np.int64)
stim_starts = stim_data['start_time']
stim_stops = stim_data['stop_time']
stim_names = stim_data['image_name']
```

iii. `CONVERSION_NOTES.md` Step 5 maps "`image_name` from stimulus presentations" to the image-identity output and says gray should be an explicit category during the ISI.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent first builds a global image vocabulary by scanning all experiments, prepending `gray`, then initializes each trial trace to `gray` and overwrites frames falling inside non-omitted stimulus intervals with the corresponding image index.

ii. 
```python
all_image_names = get_all_image_names(exp_table)
image_names_list = [GRAY_LABEL] + all_image_names
...
trace = np.full(n_frames, gray_idx, dtype=np.int64)
...
if name == 'omitted':
    continue
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. `CONVERSION_NOTES.md` Step 5 says the output should be time-varying, should show gray during the ISI, and should be constructed from stimulus presentations rather than trial-level image labels.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The code aligns image identity by trial on the ophys frame grid. For each trial, it finds the ophys timestamps within the trial, then labels those same frames according to overlapping stimulus intervals.

ii. 
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
...
img_trace, _ = build_image_identity_trace(ophys_ts, stim_data, t_start, t_stop, image_names_list)
```

iii. `CONVERSION_NOTES.md` Step 5 says image identity is "mapped to ophys timepoints."

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table's `is_change` flag and `start_time`.

ii. 
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']
...
if not is_change[si]:
    continue
```

iii. `CONVERSION_NOTES.md` Step 5 maps "`is_change` from stimulus presentations" to the image-change output.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The agent constructs a zero vector for each trial and places a `1` at the first ophys frame at or after each change flash onset inside the trial.

ii. 
```python
trace = np.zeros(n_frames, dtype=np.int64)
...
s_start = stim_starts[si]
if s_start < trial_start or s_start >= trial_stop:
    continue
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. `CONVERSION_NOTES.md` Step 5 says the image-change output should be binary and "1 at change onset frame."

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is not thresholded from a continuous value; it is encoded directly as a binary categorical trace with categories `no_change` and `change`.

ii. 
```python
output_values = [
    image_names_list,
    ['no_change', 'change'],
    ...
]
```

iii. `CONVERSION_NOTES.md` Step 5 describes this output as binary rather than continuous.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Alignment is on the same per-trial ophys frame grid as the neural data. The `1` is assigned to the first neural frame after stimulus change onset.

ii. 
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. `CONVERSION_NOTES.md` Step 7 says the processing plots showed change events aligned to stimulus onset.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in each NWB file.

ii. 
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. `CONVERSION_NOTES.md` Step 5 maps running speed directly from the NWB running stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code linearly interpolates the full-session running-speed signal to the ophys timestamps, then slices the interpolated signal per trial and converts it to percentile bins.

ii. 
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
...
running_trial = running_at_ophys[frame_mask]
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. `CONVERSION_NOTES.md` Step 5 says running speed should be interpolated from 60 Hz to ophys timestamps.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The agent computes 5 equal-percentile bin edges from the full session's interpolated running-speed values, then applies those edges to each trial. Values are coded as `bin_0` through `bin_4`.

ii. 
```python
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
...
result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
...
[f'bin_{i}' for i in range(5)]
```

iii. `CONVERSION_NOTES.md` Step 5 says the bins are session-wide equal percentile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Alignment is performed by interpolating to the ophys timestamps first and then selecting the same per-trial ophys frame mask used for the neural data.

ii. 
```python
running_at_ophys = interpolate_to_ophys(..., ophys_ts)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
running_trial = running_at_ophys[frame_mask]
```

iii. `CONVERSION_NOTES.md` Step 10 says the running-speed sanity check was done after interpolation to the ophys grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the eye-tracking stream's pupil area and timestamps plus the `likely_blink` mask.

ii. 
```python
pt = et['pupil_tracking']
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `pupil_tracking/area` to pupil diameter and explicitly mentions blink handling.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink-marked frames are set to `NaN`, diameter is computed from area as `2 * sqrt(area / pi)`, then the resulting time series is linearly interpolated to the ophys timestamps.

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
```

iii. `CONVERSION_NOTES.md` Step 5 gives the same diameter formula, and Step 10 says this matches the whitepaper's area-to-diameter conversion choice.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Like running speed, pupil diameter is binned into 5 equal percentile bins using session-wide edges computed from the interpolated ophys-aligned pupil signal. Missing values are forced into bin 0.

ii. 
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
...
pupil_trial = pupil_at_ophys[frame_mask]
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
...
result[~valid] = 0
```

iii. `CONVERSION_NOTES.md` Step 5 says percentile bins are session-wide. Trajectory step 62 says the agent consciously left `NaN` pupil values mapped to bin 0 because the task asked for five bins rather than a sixth blink class.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The code interpolates pupil diameter to the ophys timestamps and then slices it by the same trial frame mask used for neural data.

ii. 
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. `CONVERSION_NOTES.md` Step 5 and Step 10 both describe the pupil output as aligned to ophys timestamps.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. 
```python
for key in ['start_time', 'stop_time', 'go', 'catch', 'aborted', 'auto_rewarded',
             'hit', 'miss', 'false_alarm', 'correct_reject', ...]:
    if key in trials:
        trial_data[key] = trials[key][:]
...
if trial_data['hit'][idx]:
    return 'hit'
elif trial_data['miss'][idx]:
    return 'miss'
elif trial_data['false_alarm'][idx]:
    return 'false_alarm'
elif trial_data['correct_reject'][idx]:
    return 'correct_reject'
```

iii. `CONVERSION_NOTES.md` Step 5 maps "Trial outcome (hit/miss/FA/CR)" to the final output dimension.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The agent chooses the first matching outcome label by fixed precedence (`hit`, `miss`, `false_alarm`, `correct_reject`), converts it to an integer category, and then broadcasts that scalar across all time bins of the trial instead of storing it separately as a static scalar.

ii. 
```python
outcome = get_trial_outcome(trial_data, trial_idx)
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
...
output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
...
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. The inline comments in `convert_data.py` show the agent debated mixed static/time-varying output shapes and deliberately chose a constant row per trial to keep one array per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable experiments are silently skipped when files, cells, stimulus tables, or at least two valid trials are absent. Omitted stimuli are treated as gray. Missing pupil data yields an all-`NaN` pupil trace, which later becomes bin 0. Blink-marked pupil samples are also set to `NaN` and then effectively become bin 0 after discretization. Unknown trial outcomes fall back to category index 0.

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
if name == 'omitted':
    continue
...
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
result[~valid] = 0
...
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
```

iii. `CONVERSION_NOTES.md` Step 10 explicitly defends the `NaN pupil -> bin 0` rule as a "design choice." The trajectory also shows the agent noticed this issue and intentionally left it unchanged.

## 9-a. What are the most time-consuming steps of the code?

i. The biggest costs are full NWB I/O for every experiment, the pre-pass that rescans every experiment to build the global image-name list, and repeated per-trial/per-stimulus loops for image identity and image change construction.

ii. 
```python
nwb_data = load_nwb_data(nwb_path)
...
all_image_names = get_all_image_names(exp_table)
...
for trial_idx in valid_trial_idx:
    ...
    img_trace, _ = build_image_identity_trace(...)
    change_trace = build_image_change_trace(...)
```

iii. `CONVERSION_NOTES.md` Step 7 reports timing estimates and specifically notes that "Image name collection adds ~14s overhead."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the nested loops over all stimulus presentations inside `build_image_identity_trace()` and `build_image_change_trace()`, the repeated scan through all experiments in `get_all_image_names()`, and the repeated boolean mask construction per trial.

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
for _, row in exp_table.iterrows():
    ...
```

iii. `CONVERSION_NOTES.md` Step 6 says efficiency mattered, but the final implementation keeps these Python loops.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly recomputes trial frame masks, repeatedly scans the same stimulus list once for image identity and again for image change for every trial, and repeatedly performs `image_names_list.index(name)` lookups instead of caching a mapping.

ii. 
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
...
for si in range(len(stim_starts)):
    ...
for si in range(len(stim_starts)):
    ...
if name in image_names_list:
    img_idx = image_names_list.index(name)
```

iii. This repetition is visible directly in the script; no note claims it was eliminated.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script reads and stores `cell_roi_ids` but never uses them, creates `output_tv` and `output_static` and then discards both, builds a global image-name vocabulary by reopening every experiment even though each session only uses 8 images, and computes some per-trial variables such as `trial_ts` or `n_frames` only to use them indirectly or not at all.

ii. 
```python
data['cell_roi_ids'] = seg[key]['id'][:]
...
output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
output_static = np.array([outcome_idx], dtype=np.int64)
...
all_image_names = get_all_image_names(exp_table)
```

iii. These are direct code observations rather than stated design goals. The notes mention the image-name scan overhead, but the other discarded intermediates are only visible in `convert_data.py`.
