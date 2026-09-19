# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads a local metadata CSV (`ophys_experiment_table.csv`), enumerates downloaded NWB files under `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, filters the table to those experiment IDs and to non-passive sessions, and then opens each NWB directly with `h5py`. It does not use the Allen SDK cache or filter by `project_code == 'VisualBehavior'`.

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
    return exp_table

def load_nwb_data(nwb_path):
    with h5py.File(nwb_path, 'r') as f:
        ...
```

iii. In `CONVERSION_NOTES.md`, the agent justified this as a speed choice: "Loads NWB files directly via h5py (fast, no AllenSDK overhead)" and "Only use active behavior sessions."

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values in the filtered experiment table. A `subject_map` is built on the fly as experiments are processed.

ii. 
```python
print(f"  Unique mice: {exp_table['mouse_id'].nunique()}")

subject_map = {}
...
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
all_subject_idx.append(subject_map[mouse_id])
```

iii. The notes map `mouse_id` directly to `subjects` / `subject_idx` and treat it as the mouse identifier for the downloaded subset.

## 1-c. How are the data split into sessions?

i. The agent treats each `ophys_experiment_id` as one output session. It iterates row-by-row over the experiment table and appends one session per experiment, even though it also stores the underlying `ophys_session_id` in metadata.

ii. 
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    ...
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    ...
    all_neural.append(result['neural'])
    all_output.append(result['output'])
    ...
    session_metadata.append({
        'exp_id': result['exp_id'],
        'ophys_session_id': result['ophys_session_id'],
        ...
    })
```

iii. The explicit justification in the notes is: "Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data."

## 1-d. How are the data split into trials?

i. Trials are split from the NWB `intervals/trials` table. For each valid trial, the code takes all ophys frames with timestamps `>= start_time` and `< stop_time`, producing variable-length trials.

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
    n_trial_frames = frame_mask.sum()
```

iii. The notes say the trial definition uses `start_time`/`stop_time` from the trials table and extracts frames between those bounds on the ophys timebase.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `(go | catch) & ~aborted & ~auto_rewarded`. Trials with fewer than 2 ophys frames are skipped. Entire experiments are skipped if they have fewer than 2 valid trials, no cells, or no stimulus table.

ii. 
```python
valid = (go | catch) & ~aborted & ~auto_rewarded
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
if len(neural_trials) < 2:
    return None
```

iii. The justification in the notes and README is that the decoder task should include Go and Catch trials but exclude Aborted and Auto-rewarded trials; the minimum-trial checks are there to keep decoder-evaluable sessions only.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB dF/F dataset at `processing/ophys/dff/traces/data`, together with its `timestamps`.

ii. 
```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
...
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. The notes explicitly justify using dF/F rather than event traces: "Use dF/F traces (not events/deconvolved). dF/F is the standard calcium imaging signal."

## 2-b. How is the `neural` data processed?

i. Neural processing is minimal: the NWB dF/F array is transposed to `(n_cells, n_frames)`, then sliced into per-trial matrices using the ophys frame mask. The code does not merge multiple planes into one session; each experiment remains separate.

ii. 
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T
...
neural = dff[:, frame_mask].astype(np.float32)
...
result = {
    'neural': neural_trials,
    ...
    'ophys_session_id': int(exp_row['ophys_session_id']),
}
```

iii. The notes say the agent wanted to use precomputed dF/F and keep each experiment separate, especially to handle both single-plane and Multiscope recordings without extra merging logic.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neuron-quality filtering is applied beyond requiring `n_cells > 0`. The script loads whatever traces are present in the NWB dF/F dataset.

ii. 
```python
dff = nwb_data['dff_traces']  # (n_cells, n_frames)
n_cells, n_frames = dff.shape

if n_cells == 0:
    print(f"  WARNING: No cells in experiment {exp_id}")
    return None
```

iii. In the notes, the agent argued that an explicit `valid_roi` filter was unnecessary because the downloaded NWB files already contain valid ROIs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The per-trial neural data are aligned to the ophys timestamp stream and cropped to the trial window from `start_time` to `stop_time`. The practical alignment event is trial start on the ophys clock.

ii. 
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes say: "Align to ophys timestamps. For each trial, extract the ophys frames between trial start_time and stop_time."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Each experiment uses its native ophys frame times, with per-experiment `dt = median(diff(ophys_ts))`. The final metadata stores the median `dt` across all output sessions, even though the notes explicitly acknowledge both ~31 Hz and ~11 Hz sessions.

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

iii. The notes justify this as using "native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope)" with each session internally consistent.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation interval table, specifically `start_time`, `stop_time`, and `image_name`, rather than from the trial table’s `initial_image_name` / `change_image_name`.

ii. 
```python
if stim_key is not None:
    stim = f['intervals'][stim_key]
    stim_data = {}
    for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
        if key in stim:
            stim_data[key] = stim[key][:]

...
stim_starts = stim_data['start_time']
stim_stops = stim_data['stop_time']
stim_names = stim_data['image_name']
```

iii. The notes explicitly say: "Image identity: Map stimulus presentations to ophys timepoints."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code first collects a global image vocabulary across experiments, prepends a synthetic `gray` label, initializes each trial to the gray category, and then overwrites frames that fall within individual stimulus presentations. `omitted` flashes are left as gray.

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

iii. The justification in the notes is explicit: "During gray screen (ISI), use a 'gray' category." The README also describes the output as "gray + 16 images."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is aligned directly on the ophys timebase. The code computes `trial_ts = ophys_ts[trial_mask]` and assigns image categories to the same frame positions used for the neural trial slice.

ii. 
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. The notes and processing-plot review state that image identity was checked against stimulus onset timing and aligned to ophys timestamps.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change` and `start_time` fields, not from the trial table’s `change_time` and `go` fields.

ii. 
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']
```

iii. The notes map `is_change` from stimulus presentations directly to `output[1]` and describe it as "1 at change onset frame."

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each change stimulus that falls inside the trial window, the trace is set to 1 only at the first ophys frame at or after the stimulus onset. All other frames remain 0.

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

iii. The README and notes justify this as a binary indicator that is "1 at image change onset."

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary; there is no additional thresholding. Category 0 is `no_change` and category 1 is `change`.

ii. 
```python
trace = np.zeros(n_frames, dtype=np.int64)
...
output_values = [
    image_names_list,
    ['no_change', 'change'],
    ...
]
```

iii. The justification is simply that the decoder task requested a binary change variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like the other time-varying outputs, it is aligned on the ophys frame grid inside each trial window. Change onset is snapped to the first ophys frame at or after the stimulus-change timestamp.

ii. 
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_idx = np.searchsorted(trial_ts, s_start)
...
change_trace = build_image_change_trace(ophys_ts, stim_data, t_start, t_stop)
```

iii. The notes say the change events were checked visually and were "aligned to stimulus onset."

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and its `timestamps` in the NWB file.

ii. 
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. The notes treat this as the standard running-wheel stream and map it directly to the running-speed output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The signal is linearly interpolated from the running timestamps onto the full ophys timestamp vector, then each experiment’s interpolated values are discretized using percentile bin edges computed from that same experiment.

ii. 
```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts_trial):
    f = interpolate.interp1d(signal_ts, signal, kind='linear',
                             bounds_error=False, fill_value=np.nan)
    return f(ophys_ts_trial)

...
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
...
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. The notes justify this as interpolation from 60 Hz to ophys timestamps followed by session-wide percentile bins.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into 5 equal-percentile bins computed per experiment/session, with NaNs assigned to bin 0.

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
    ...
    result[~valid] = 0
```

iii. The notes explicitly say: "Compute percentiles across the entire session (all valid timepoints), then apply per-trial. Use 5 equal bins."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolating it to the session’s ophys timestamps before trial slicing, then indexing it with the same `frame_mask` used for neural data.

ii. 
```python
running_at_ophys = interpolate_to_ophys(..., ophys_ts)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
...
neural = dff[:, frame_mask].astype(np.float32)
running_trial = running_at_ophys[frame_mask]
```

iii. The notes explicitly state that running speed should be aligned to ophys timestamps.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the eye-tracking `pupil_tracking/area` values, plus `timestamps` and the `likely_blink` mask.

ii. 
```python
if 'pupil_tracking' in et:
    pt = et['pupil_tracking']
    data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
    data['pupil_timestamps'] = pt['timestamps'][:]
    data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. The notes justify this as: "Pupil diameter: Compute from pupil area as `2*sqrt(area/pi)`."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code copies pupil area, sets blink frames to NaN, converts positive areas to diameter using `2 * sqrt(area / pi)`, interpolates that diameter to ophys timestamps, and then bins it using per-experiment percentile edges.

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
...
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
```

iii. The notes justify the diameter conversion and blink handling explicitly, and say to discretize non-NaN values after interpolation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into 5 equal-percentile bins computed per experiment/session, with NaNs assigned to bin 0.

ii. 
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
...
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
```

iii. The notes use the same session-wide percentile strategy for pupil diameter as for running speed, and explicitly call out NaN/blink handling.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolation onto ophys timestamps first, then by taking the same trial `frame_mask` used for the neural slice.

ii. 
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
...
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. The notes repeatedly state that all time-varying outputs are aligned to ophys timestamps.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

iii. The notes describe this as the categorical per-trial outcome mapped from hit/miss/false-alarm/correct-reject.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code converts the first matching boolean outcome flag into an integer category via `outcome_names.index(...)` and then broadcasts that scalar across all frames in row 4 of the trial output matrix.

ii. 
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome = get_trial_outcome(trial_data, trial_idx)
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
...
output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
...
output_full[4] = outcome_idx
```

iii. The code comments show the agent debated how to encode a static per-trial output in a single array and chose to repeat the label across time so every trial output has shape `(5, T)`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing or problematic data mostly by skipping entire experiments or mapping missing values to NaNs and then bin 0. Missing NWB files, missing stimulus tables, zero-cell experiments, and experiments with too few valid trials are dropped. Missing pupil data becomes all-NaN; blink samples are set to NaN; NaNs in running/pupil are assigned to bin 0; short trials are skipped.

ii. 
```python
if not os.path.exists(nwb_path):
    return None
...
if stim_data is None:
    return None
...
if len(valid_trial_idx) < 2:
    return None
...
if nwb_data['pupil_area'] is not None:
    ...
    pupil_area[likely_blink] = np.nan
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
if n_trial_frames < 2:
    continue
...
result[~valid] = 0
```

iii. The notes justify NaN-to-bin-0 as a deliberate design choice for blinks/missing pupil data and say sessions with too few valid trials are retained only if they still have at least two usable trials.

## 9-a. What are the most time-consuming steps of the code?

i. The code’s main bottlenecks are repeatedly opening NWB files: once in the full-dataset image-name scan and once again during actual experiment processing. The notes also report NWB loading as the dominant per-session cost.

ii. 
```python
def get_all_image_names(exp_table):
    for _, row in exp_table.iterrows():
        ...
        with h5py.File(nwb_path, 'r') as f:
            ...

for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
```

iii. In the sample-run notes, the agent recorded "Load NWB ~1.7s/session" and "Image name collection adds ~14s overhead."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code uses several serial loops that could have been vectorized: the loop over every stimulus presentation inside `build_image_identity_trace`, the similar loop in `build_image_change_trace`, the loop over valid trials inside `process_experiment`, and the first pass over every NWB file in `get_all_image_names`.

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

for trial_idx in valid_trial_idx:
    ...

for _, row in exp_table.iterrows():
    ...
```

iii. The agent did not provide an explicit efficiency justification for these loops beyond preferring direct NWB access and adding processing plots; the structure appears chosen for simplicity.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats several pieces of work: it scans each NWB once to build the global image-name list and then opens the same NWBs again for full processing; within each trial it recomputes `trial_mask` in multiple helper functions; and during image-identity construction it repeatedly calls `image_names_list.index(name)` inside the stimulus loop.

ii. 
```python
all_image_names = get_all_image_names(exp_table)
...
result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)

gray_idx = image_names_list.index(GRAY_LABEL)
...
if name in image_names_list:
    img_idx = image_names_list.index(name)

trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
...
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
```

iii. The notes acknowledge the repeated image-name scan as overhead, but do not otherwise justify the repeated helper-level recomputation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of processing are unnecessary for the saved dataset: `cell_roi_ids` are loaded but never used; the temporary `output_tv` and `output_static` arrays are constructed and then discarded; the standalone `discretize_percentile` helper is defined but never called; and `--show-processing` stores extra arrays only for plotting.

ii. 
```python
if 'image_segmentation' in f['processing']['ophys']:
    ...
    data['cell_roi_ids'] = seg[key]['id'][:]

def discretize_percentile(values, n_bins=5):
    ...

output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
output_static = np.array([outcome_idx], dtype=np.int64)

if show_processing:
    result['ophys_ts'] = ophys_ts
    result['running_at_ophys'] = running_at_ophys
    ...
```

iii. There is no explicit justification beyond diagnostics and plotting support; these appear to be convenience leftovers rather than required downstream processing.
