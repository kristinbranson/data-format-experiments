# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a local metadata CSV (`ophys_experiment_table.csv`), filters it to experiments whose NWB files are present on disk, drops passive sessions, then opens each NWB file directly with `h5py`. It does not use the Allen SDK cache, does not filter by `project_code == 'VisualBehavior'`, and processes one experiment at a time.

ii. 
```python
def load_experiment_table():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    ...
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)].copy()
    exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)].copy()
    exp_table = exp_table.sort_values('ophys_experiment_id').reset_index(drop=True)
    return exp_table

...
with h5py.File(nwb_path, 'r') as f:
    data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
    dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
    data['dff_traces'] = dff_raw.T
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly chose to use the downloaded local subset and “load NWB files directly via h5py (fast, no AllenSDK overhead).” Its Step 4 notes say the available data are a downloaded subset and that it would “use all active downloaded sessions.”

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values in the filtered experiment table, and each mouse ID is mapped to a subject index the first time it appears.

ii. 
```python
print(f"  Unique mice: {exp_table['mouse_id'].nunique()}")

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

iii. The justification is implicit in the code and consistent with Step 5 of `CONVERSION_NOTES.md`, which maps `mouse_id` to `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` as a separate output session. It does not merge experiments sharing an `ophys_session_id`; multiscope planes are intentionally kept separate because they have different neurons.

ii. 
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names,
                                show_processing=args.show_processing)
    ...
    all_neural.append(result['neural'])
    all_output.append(result['output'])
```

iii. The justification is explicit in Step 5 of `CONVERSION_NOTES.md`: “Multiscope handling: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data.”

## 1-d. How are the data split into trials?

i. Trials are defined from the NWB `intervals/trials` table. The AI keeps trials where `(go | catch) & ~aborted & ~auto_rewarded`, then uses each trial’s `start_time` and `stop_time` to take all ophys frames satisfying `start_time <= t < stop_time`.

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
for trial_idx in valid_trial_idx:
    t_start = trial_data['start_time'][trial_idx]
    t_stop = trial_data['stop_time'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
```

iii. Step 5 of `CONVERSION_NOTES.md` states: “Trial definition: Use `start_time` and `stop_time` from trials table for Go and Catch trials only.” In the trajectory, the AI also justified low valid-trial counts by pointing to many aborted trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes aborted and auto-rewarded trials, requires trials to be either go or catch, skips experiments with fewer than 2 valid trials, and skips individual trial windows with fewer than 2 ophys frames. It does not explicitly require non-null `change_time`.

ii. 
```python
valid = (go | catch) & ~aborted & ~auto_rewarded
...
if len(valid_trial_idx) < 2:
    print(f"  WARNING: Only {len(valid_trial_idx)} valid trials in experiment {exp_id}")
    return None
...
if n_trial_frames < 2:
    continue
...
if len(neural_trials) < 2:
    print(f"  WARNING: Only {len(neural_trials)} valid trials after processing for experiment {exp_id}")
    return None
```

iii. Step 3 of `CONVERSION_NOTES.md` lists the curation rule “Include: Go trials and Catch trials; Exclude: Aborted trials and Auto-rewarded trials.” The trajectory also shows the AI checking that low trial counts were explained by many aborted trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB dF/F array at `processing/ophys/dff/traces/data`, transposed to neuron-by-time format.

ii. 
```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. Step 5 of `CONVERSION_NOTES.md` says: “Neural data: Use dF/F traces (not events/deconvolved). dF/F is the standard calcium imaging signal.”

## 2-b. How is the `neural` data processed?

i. The AI does almost no extra neural processing beyond transposing the NWB array and slicing it into trial windows. It does not merge planes within an ophys session; each experiment remains separate. Neural trials are cast to `float32`.

ii. 
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. Step 6 of `CONVERSION_NOTES.md` says the script “loads NWB files directly via h5py” and handles multiscope by keeping experiments separate. Step 5 also states the AI chose dF/F rather than events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neuron-level QC filter is applied in `convert_data.py`. The AI loads every trace present in the NWB dF/F dataset and only rejects experiments with zero cells.

ii. 
```python
dff = nwb_data['dff_traces']
n_cells, n_frames = dff.shape

if n_cells == 0:
    print(f"  WARNING: No cells in experiment {exp_id}")
    return None
```

iii. The AI’s notes acknowledge the Allen SDK’s `exclude_invalid_rois` logic, but Step 10 of `CONVERSION_NOTES.md` records the final decision as “No explicit valid_roi filter ... OK - all ROIs in downloaded NWB files are valid.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the ophys timestamp stream by selecting, for each trial, the ophys frames between trial `start_time` and `stop_time`. The alignment event is therefore the trial window on the ophys clock, not a fixed window around `change_time`.

ii. 
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. Step 5 of `CONVERSION_NOTES.md` says: “Temporal alignment: Align to ophys timestamps. For each trial, extract the ophys frames between trial start_time and stop_time.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Each experiment keeps its native ophys sampling, and the script records each experiment’s median frame interval as `dt`. The top-level metadata reports the median `dt` across processed experiments, even though the dataset includes both ~31 Hz and ~11 Hz recordings.

ii. 
```python
dt = np.median(np.diff(ophys_ts))  # time bin size
...
session_metadata.append({
    ...
    'dt_ms': result['dt'] * 1000,
})
...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
...
'time_bin_size': median_dt,
```

iii. Step 5 of `CONVERSION_NOTES.md` says: “Time bin: Use native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope). Each session's time bin is consistent within itself.”

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The AI derives image identity from the stimulus-presentation table, specifically each presentation’s `start_time`, `stop_time`, and `image_name`. It does not use the trial table’s `initial_image_name` and `change_image_name` to define the label.

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

iii. Step 5 of `CONVERSION_NOTES.md` maps “`image_name` from stimulus presentations” to image identity and says the variable should be time-varying at the ophys rate.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI first scans all experiments to collect unique image names, prepends a synthetic `gray` label, and then builds a per-trial framewise trace initialized to `gray`. For each overlapping stimulus presentation, it overwrites the corresponding ophys frames with the presented image ID; omitted stimuli are left as gray.

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
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. Step 5 of `CONVERSION_NOTES.md` explicitly justifies this choice: “During gray screen (ISI), use a ‘gray’ category.” The trajectory also notes that processing plots looked correct because they showed gray during ISI and images during flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is built directly on the same ophys frame mask used for each neural trial. Stimulus intervals are intersected with `trial_ts`, so the label changes on the ophys frames whose timestamps fall inside each image-presentation interval.

ii. 
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. The AI’s notes justify this by saying image identity should be “time-varying at ophys rate,” and the trajectory says the plots confirmed “image identity correctly shows gray during ISI, images during flashes.”

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `image_change` is derived from the stimulus-presentation table’s `is_change` field and each presentation’s `start_time`, rather than from the trials table’s `change_time` and `go` fields.

ii. 
```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    if key in stim:
        stim_data[key] = stim[key][:]

...
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']
```

iii. Step 5 of `CONVERSION_NOTES.md` maps “`is_change` from stimulus presentations” to `output[1]: image_change` and describes it as “1 at change onset frame.”

## 4-b. What processing is involved in computing `output` *Image change*?

i. For every stimulus presentation marked `is_change`, the AI finds the first ophys frame at or after that stimulus onset within the trial and sets only that frame to 1. All other frames stay 0.

ii. 
```python
trace = np.zeros(n_frames, dtype=np.int64)
...
for si in range(len(stim_starts)):
    if not is_change[si]:
        continue
    s_start = stim_starts[si]
    if s_start < trial_start or s_start >= trial_stop:
        continue
    frame_idx = np.searchsorted(trial_ts, s_start)
    if frame_idx < n_frames:
        trace[frame_idx] = 1
```

iii. Step 5 of `CONVERSION_NOTES.md` says the AI intended a binary output with “1 at change onset frame, 0 otherwise.” The trajectory later says processing plots showed change events aligned to stimulus onset.

## 4-c. How is `output` *Image change* thresholded into categories?

i. The AI uses a fixed binary categorization: `0 = no_change`, `1 = change`. There is no continuous value and no wider event window.

ii. 
```python
output_values = [
    image_names_list,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
    [f'bin_{i}' for i in range(5)],
    outcome_names,
]

...
trace = np.zeros(n_frames, dtype=np.int64)
...
trace[frame_idx] = 1
```

iii. The justification is implicit in Step 5 of `CONVERSION_NOTES.md`, which planned `image_change` as a binary change-onset indicator.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. `image_change` is aligned on the ophys clock, using the same per-trial ophys frame mask as the neural data. The positive label is attached to the first ophys frame at or after the changed stimulus onset.

ii. 
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. The trajectory states that “change events aligned to stimulus onset,” which is the AI’s stated rationale for this choice.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the NWB running-speed stream: `processing/running/speed/data` and its timestamps.

ii. 
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. Step 5 of `CONVERSION_NOTES.md` maps `running_speed` directly to the running-speed output and notes interpolation to ophys timestamps.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed from its native timestamps to all ophys timestamps for the experiment, computes percentile edges from the whole experiment’s interpolated running signal, and applies those edges to each trial.

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

iii. Step 5 of `CONVERSION_NOTES.md` explicitly says: “Interpolate from 60 Hz to ophys timestamps using linear interpolation” and “Compute percentiles across the entire session ... then apply per-trial.”

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into five percentile bins (`bin_0` through `bin_4`) using session-wide percentile edges from non-NaN interpolated values. NaNs are assigned to bin 0.

ii. 
```python
def compute_session_percentile_edges(values, n_bins=5):
    valid = values[~np.isnan(values)]
    ...
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges

def apply_percentile_bins(values, edges, n_bins=5):
    valid = ~np.isnan(values)
    result = np.zeros(len(values), dtype=np.int64)
    if valid.sum() > 0:
        result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
    result[~valid] = 0
    return result
```

iii. The notes justify this as “Use 5 equal bins (0-20th, 20-40th, ..., 80-100th percentile)” and the trajectory says NaN-to-bin-0 was a deliberate choice for discretized outputs.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The running signal is first interpolated onto the full ophys timestamp grid, then trial slices are taken with the same boolean frame mask used for neural data. That gives timepoint-by-timepoint alignment to neural frames.

ii. 
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
...
running_trial = running_at_ophys[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. Step 5 of `CONVERSION_NOTES.md` says running speed should be “interpolate[d] to ophys timestamps,” and this is the AI’s stated alignment strategy.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI derives pupil diameter from eye-tracking `pupil_area`, `pupil_timestamps`, and `likely_blink` values read directly from NWB. It does not use `pupil_width`.

ii. 
```python
if 'pupil_tracking' in et:
    pt = et['pupil_tracking']
    data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
    data['pupil_timestamps'] = pt['timestamps'][:]
    data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. Step 5 of `CONVERSION_NOTES.md` explicitly planned: “`pupil_tracking/area` → diameter” and described converting area to diameter after blink handling.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI copies pupil area, sets blink frames to NaN, converts non-NaN positive areas to an estimated diameter using `2 * sqrt(area / pi)`, interpolates that signal to ophys timestamps, then bins it with session-wide percentile edges.

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

iii. The notes justify this directly: “Compute diameter from pupil area as `2*sqrt(area/pi)`. Set blink frames to NaN, then interpolate.”

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into five percentile bins (`bin_0` through `bin_4`) using experiment-wide percentile edges from non-NaN values after interpolation. NaNs are assigned to bin 0.

ii. 
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
...
pupil_trial = pupil_at_ophys[frame_mask]
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
...
result[~valid] = 0
```

iii. Step 5 of `CONVERSION_NOTES.md` says pupil diameter should be discretized into “5 equal bins” and that blink-related NaNs should be handled conservatively.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the ophys timestamp grid first, then trial windows are extracted with the same frame mask as the neural data.

ii. 
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
pupil_trial = pupil_at_ophys[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. Step 5 of `CONVERSION_NOTES.md` says all streams should be aligned to ophys timestamps, and the trajectory says the processing plots showed pupil traces aligned with the trial windows.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. 
```python
for key in ['start_time', 'stop_time', 'go', 'catch', 'aborted', 'auto_rewarded',
             'hit', 'miss', 'false_alarm', 'correct_reject', 'change_time',
             'initial_image_name', 'change_image_name', 'is_change']:
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

iii. Step 5 of `CONVERSION_NOTES.md` maps “Trial outcome (hit/miss/FA/CR)” to the output and treats it as a categorical per-trial variable.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI resolves each trial to one of the four named outcomes, converts that name to an integer index via `outcome_names.index(...)`, and then broadcasts the scalar outcome across all time bins in the saved output array.

ii. 
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome = get_trial_outcome(trial_data, trial_idx)
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
...
output_full[4] = outcome_idx
```

iii. The notes say trial outcome should be “categorical, static per trial.” The code comments show the AI debated mixed static/time-varying formatting and chose to repeat the static label across time.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases by skipping bad experiments, skipping too-short trials, filling missing pupil with NaNs, mapping NaNs in discretized running/pupil to bin 0, and aborting if no valid sessions remain. It also skips experiments with no stimulus table or no cells. It does not have explicit special handling for missing `change_time` in trial selection.

ii. 
```python
if not os.path.exists(nwb_path):
    print(f"  WARNING: NWB file not found for experiment {exp_id}")
    return None
...
if stim_data is None:
    print(f"  WARNING: No stimulus data in experiment {exp_id}")
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
if len(all_neural) == 0:
    print("ERROR: No valid sessions processed!")
    sys.exit(1)
```

iii. The trajectory records an explicit justification for one of these choices: mapping NaN pupil bins to 0 was considered acceptable because adding a separate blink class would violate the requested 5-bin output. The rest are mostly implicit robustness decisions reflected in Step 6 and Step 10 notes.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive steps are reading each large NWB file from disk and the extra full-dataset scan in `get_all_image_names`, which opens every NWB once before the main processing pass. Trial-wise stimulus loops are additional per-experiment overhead but smaller.

ii. 
```python
all_image_names = get_all_image_names(exp_table)
...
for _, row in exp_table.iterrows():
    eid = row['ophys_experiment_id']
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
    with h5py.File(nwb_path, 'r') as f:
        ...

...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
```

iii. The trajectory says full conversion took about 9 minutes and that “Image name collection adds ~14s overhead.” Step 7 of `CONVERSION_NOTES.md` also breaks runtime down into load time vs. processing time and identifies NWB loading as the main cost.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunities are the Python loops over stimulus presentations in `build_image_identity_trace` and `build_image_change_trace`, the full-table `iterrows()` loop in `get_all_image_names`, and the per-experiment / per-trial Python loops that repeatedly build boolean masks and output arrays.

ii. 
```python
for si in range(len(stim_starts)):
    ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx

for si in range(len(stim_starts)):
    if not is_change[si]:
        continue
    ...

for _, row in exp_table.iterrows():
    ...

for trial_idx in valid_trial_idx:
    ...
```

iii. The AI did not write an explicit vectorization analysis, but Step 6 says it intended to “write efficient code,” and the runtime note that image-name collection adds measurable overhead supports this inference.

## 9-c. What processing does the code repeat multiple times?

i. The AI repeats a full pass over all NWB files to collect image names before doing the real processing pass, and then repeats per-trial timestamp comparisons inside both image-trace helper functions. It also recomputes boolean trial masks separately for neural, image identity, image change, running, and pupil extraction.

ii. 
```python
all_image_names = get_all_image_names(exp_table)
...
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...

...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
...
img_trace, _ = build_image_identity_trace(
    ophys_ts, stim_data, t_start, t_stop, image_names_list
)
change_trace = build_image_change_trace(ophys_ts, stim_data, t_start, t_stop)
```

iii. The justification is only indirect: the AI wanted a global image vocabulary up front, and the trajectory notes that this extra image-name pass cost about 14 seconds.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads `cell_roi_ids` but never uses them; it creates `output_tv` and `output_static` but discards both in favor of `output_full`; and it does an extra pre-pass solely to collect image names. These steps do work that is not used by downstream decoding.

ii. 
```python
if 'image_segmentation' in f['processing']['ophys']:
    ...
    data['cell_roi_ids'] = seg[key]['id'][:]

...
output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
output_static = np.array([outcome_idx], dtype=np.int64)
...
output_full = np.zeros((5, n_trial_frames), dtype=np.int64)

...
all_image_names = get_all_image_names(exp_table)
```

iii. There is no explicit justification in the notes for these discarded computations. The only related rationale is that the AI wanted image categories known in advance and was iterating on mixed static/time-varying output formatting.
