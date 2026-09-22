# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from local metadata CSVs and local NWB files with `h5py`, not through the Allen SDK cache. It reads `ophys_experiment_table.csv`, filters it to experiment IDs that have downloaded NWB files, excludes session types containing `"passive"`, then iterates experiment-by-experiment and loads each NWB file with `load_nwb_data()`.

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

for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names,
                                show_processing=args.show_processing)
```

iii. The clearest justification is in `CONVERSION_NOTES.md` Step 6: the AI says it “Loads NWB files directly via h5py (fast, no AllenSDK overhead)” and “Filters to active sessions (excludes passive).”

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the filtered experiment table. They are added lazily during processing, converted to strings, and indexed with `subject_map`.

ii.
```python
subject_map = {}  # mouse_id -> index
all_subjects = []
...
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
...
all_subject_idx.append(subject_map[mouse_id])
```

iii. No detailed separate justification was given beyond the standard use of `mouse_id` in the metadata and notes.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` as a separate output session. It does keep the original `ophys_session_id` in metadata, but it does not merge experiments that share the same `ophys_session_id`.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names,
                                show_processing=args.show_processing)
...
session_metadata.append({
    'exp_id': result['exp_id'],
    'ophys_session_id': result['ophys_session_id'],
    'session_type': result['session_type'],
    ...
})
```

iii. The explicit justification is in `CONVERSION_NOTES.md` Step 5, Key Decision 9: “Each experiment (plane) is a separate ‘session’ in the output, since they have different neurons but share the same behavioral data.”

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trial table and segmented with the trial `start_time` and `stop_time`. For each valid trial, the AI builds a boolean mask over ophys timestamps and extracts all frames in that variable-length window.

ii.
```python
trial_data = nwb_data['trials']
valid_trial_idx = get_valid_trials(trial_data)
...
for trial_idx in valid_trial_idx:
    t_start = trial_data['start_time'][trial_idx]
    t_stop = trial_data['stop_time'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    trial_ts = ophys_ts[frame_mask]
    n_trial_frames = frame_mask.sum()
```

iii. The notes justify using trial `start_time` and `stop_time` and aligning all outputs to ophys timestamps.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `(go | catch) & ~aborted & ~auto_rewarded`. The AI does not additionally require non-null `change_time`. Trials with fewer than 2 ophys frames are skipped, and experiments with fewer than 2 remaining trials are dropped.

ii.
```python
def get_valid_trials(trial_data):
    go = trial_data['go'].astype(bool)
    catch = trial_data['catch'].astype(bool)
    aborted = trial_data['aborted'].astype(bool)
    auto_rewarded = trial_data['auto_rewarded'].astype(bool)
    valid = (go | catch) & ~aborted & ~auto_rewarded
    return np.where(valid)[0]

if len(valid_trial_idx) < 2:
    return None
...
if n_trial_frames < 2:
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. `CONVERSION_NOTES.md` Step 6 says the script “Filters trials to Go+Catch, excluding Aborted and Auto-rewarded.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB `processing/ophys/dff/traces/data` array, with timestamps from `processing/ophys/dff/traces/timestamps`.

ii.
```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. The notes repeatedly state the AI chose dF/F as the standard calcium-imaging signal.

## 2-b. How is the `neural` data processed?

i. The AI transposes the NWB dF/F array to `(n_cells, n_frames)`, leaves it otherwise unnormalized, then slices it per trial and casts each trial matrix to `float32`. It does not merge multiple planes together because each experiment is treated as a separate session.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
...
neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
```

iii. `CONVERSION_NOTES.md` Step 5 says “Neural data: Use dF/F traces (not events/deconvolved). dF/F is the standard calcium imaging signal.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neuron-level QC is applied beyond dropping experiments with zero cells. The script does not consult `valid_roi` or AllenSDK ROI filtering.

ii.
```python
n_cells, n_frames = dff.shape
if n_cells == 0:
    print(f"  WARNING: No cells in experiment {exp_id}")
    return None
```

iii. The notes do mention AllenSDK `exclude_invalid_rois=True` during exploration, but the final script does not implement an equivalent filter and does not explain that omission.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the ophys timestamp stream and cut to the trial window from `start_time` to `stop_time`. It is not realigned to `change_time`.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes explicitly say to “Align to ophys timestamps” and to extract frames between trial `start_time` and `stop_time`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Each experiment stays on its native ophys sampling grid, and per-experiment `dt` is the median difference of `ophys_timestamps`. However, the final metadata stores a single median `time_bin_size` across all retained experiments.

ii.
```python
dt = np.median(np.diff(ophys_ts))  # time bin size
...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
...
'time_bin_size': median_dt,
```

iii. `CONVERSION_NOTES.md` Step 5 says “Use native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope)” and “Need to handle both ~31 Hz and ~11 Hz ophys frame rates.”

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The AI derives image identity from the stimulus-presentation interval table, specifically stimulus `start_time`, `stop_time`, and `image_name`, not from trial-level `initial_image_name` / `change_image_name`.

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

iii. `CONVERSION_NOTES.md` Step 5 maps “`image_name` from stimulus presentations” to `image_identity`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The script first scans all experiments to collect the global set of image names, prepends a synthetic `"gray"` class, then for each trial initializes every ophys frame to gray and overwrites the frames that fall within non-omitted stimulus-presentation intervals. Omitted flashes stay gray.

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

iii. The notes justify this as “Image identity correctly shows gray during ISI, images during flashes,” and Step 5 explicitly says “During gray screen (ISI), use a ‘gray’ category.”

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is constructed directly on the same per-trial ophys frames as the neural data. The trial uses one `frame_mask`, and within that trial the stimulus interval times are converted to frame-level masks on `trial_ts`.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. The script comments and notes both frame this as alignment to the ophys timestamp base.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The AI derives image change from the stimulus-presentation table’s `is_change` flag and the presentation `start_time`, not from the trial-table `change_time` and `go` columns.

ii.
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']
```

iii. The notes map “`is_change` from stimulus presentations” to `image_change`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each trial, the script creates a zero vector and sets exactly one ophys frame to 1 for each in-trial stimulus presentation whose `is_change` flag is true. It is therefore an impulse at change onset, not a 750 ms window.

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

iii. The README describes this as “1 at image change onset,” and the function docstring says “1 at the first frame after an image change.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary only: `0` for `no_change` and `1` for `change`. There is no additional thresholding beyond the boolean `is_change` flag.

ii.
```python
output_values = [
    image_names_list,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
    [f'bin_{i}' for i in range(5)],
    outcome_names,
]
```

iii. The justification is implicit from the task definition and the output metadata.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change indicator is aligned to the same ophys trial frames as the neural data. The onset time is mapped to the first frame at or after the stimulus change time with `np.searchsorted`.

ii.
```python
trial_ts = ophys_ts[trial_mask]
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. The code comments explicitly state it is “the first ophys frame at or after the change onset.”

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from NWB `processing/running/speed/data` and `processing/running/speed/timestamps`.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. The notes identify this as the standard locomotion stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates the full running signal to the ophys timestamp grid, computes percentile edges from the full interpolated experiment/session trace, then applies those edges to each trial segment.

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

iii. `CONVERSION_NOTES.md` Step 5 says “Interpolate from 60 Hz to ophys timestamps” and “Compute percentiles across the entire session.”

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins using per-experiment/per-session edges. NaNs are assigned to bin `0`.

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

iii. The notes justify 5 equal-percentile bins, but specify session-wide rather than global binning.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first interpolated to `ophys_ts`, then trial windows reuse the same `frame_mask` used for neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
...
running_trial = running_at_ophys[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes explicitly say to align behavioral outputs to ophys timestamps.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI uses `acquisition/EyeTracking/pupil_tracking/area`, `timestamps`, and `likely_blink`. It does not use the `pupil_width` column exposed by the SDK.

ii.
```python
pt = et['pupil_tracking']
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly planned: “`pupil_tracking/area` → diameter.”

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink-labeled samples are set to `NaN`, pupil area is converted to diameter with `2 * sqrt(area / pi)` for positive non-NaN entries, the result is interpolated to the ophys grid, and then binned by percentile.

ii.
```python
pupil_area = nwb_data['pupil_area'].copy()
likely_blink = nwb_data['likely_blink']
...
pupil_area[likely_blink] = np.nan
...
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
...
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
```

iii. The notes justify the formula directly: “Compute diameter from pupil area as `2*sqrt(area/pi)`.”

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Like running speed, pupil diameter is put into 5 percentile bins using per-experiment/per-session edges, with NaNs forced to bin `0`.

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
...
pupil_trial = pupil_at_ophys[frame_mask]
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
```

iii. The notes justify the use of 5 equal bins, but again specify session-wide rather than global binning.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the ophys clock before trial segmentation, then the same per-trial frame mask is used to extract aligned values.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
pupil_trial = pupil_at_ophys[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The script comments and notes both describe this as alignment to ophys timestamps.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the NWB trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

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
    else:
        return 'unknown'
```

iii. This follows the standard task outcome labels; no extra justification was needed.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The string label is mapped to its index in `outcome_names`. The chosen code is then broadcast across all time bins in the trial as the fifth output row. If no known label is found, the script falls back to `0`.

ii.
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome = get_trial_outcome(trial_data, trial_idx)
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
...
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. The code comments show the AI considered mixed static/time-varying formats and chose to repeat the scalar across time for simplicity.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data by skipping whole experiments when critical structures are missing (`NWB` file absent, no cells, no stimulus table, or fewer than 2 valid trials), skipping individual trials with fewer than 2 ophys frames, setting blinked/missing pupil samples to `NaN` and then bin `0`, letting interpolation outside the recorded range become `NaN`, and skipping unknown/omitted image presentations when constructing the image trace.

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
pupil_area[likely_blink] = np.nan
...
result[~valid] = 0
...
if name == 'omitted':
    continue
```

iii. The notes mention handling blinks, missing pupil data, and low-trial experiments, but there is no broader try/except-based session recovery in the final script.

## 9-a. What are the most time-consuming steps of the code?

i. The AI’s own notes identify two main costs: loading each NWB experiment and scanning all NWB files once up front to collect image names. The script also measures `t_load` and `t_process` per experiment.

ii.
```python
def get_all_image_names(exp_table):
    all_names = set()
    for _, row in exp_table.iterrows():
        eid = row['ophys_experiment_id']
        nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
        with h5py.File(nwb_path, 'r') as f:
            ...

t_load = time.time() - t0
...
print(f"  Cells: {result['n_cells']}, Trials: {result['n_trials']}, "
      f"dt: {result['dt']*1000:.1f}ms, "
      f"Time: {t_total:.1f}s (load={result['t_load']:.1f}s, process={result['t_process']:.1f}s)")
```

iii. `CONVERSION_NOTES.md` Step 7 says “Load NWB ~1.7s” per session and “Image name collection adds ~14s overhead.”

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves several heavy Python loops unvectorized: iterating over all experiments, all valid trials within an experiment, all stimulus presentations when building image identity, and all stimulus presentations again when building image change.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...

for trial_idx in valid_trial_idx:
    ...

for si in range(len(stim_starts)):
    ...

for si in range(len(stim_starts)):
    ...
```

iii. The trajectory shows the AI estimated the runtime was acceptable and “no optimization needed,” so it did not justify additional vectorization.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats at least two kinds of work. First, it opens every NWB once in `get_all_image_names()` and then opens the same NWB again in `process_experiment()`. Second, for every trial it rescans the full stimulus-presentation list in both `build_image_identity_trace()` and `build_image_change_trace()`.

ii.
```python
all_image_names = get_all_image_names(exp_table)
...
result = process_experiment(exp_id, row, image_names_list, outcome_names,
                            show_processing=args.show_processing)
```

```python
for si in range(len(stim_starts)):
    ...
```

iii. The only clear rationale is that the AI wanted a stable global image-name mapping before trial processing. The notes acknowledge the image-name scan overhead.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script performs some work that is never used downstream: it reads `cell_roi_ids` but never uses them, defines `discretize_percentile()` but never calls it, and allocates `output_tv` and `output_static` before discarding both in favor of `output_full`.

ii.
```python
if 'image_segmentation' in f['processing']['ophys']:
    ...
        data['cell_roi_ids'] = seg[key]['id'][:]
```

```python
def discretize_percentile(values, n_bins=5):
    ...
```

```python
output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
output_static = np.array([outcome_idx], dtype=np.int64)
...
output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
```

iii. No explicit justification for these discarded computations appears in the notes or trajectory.
