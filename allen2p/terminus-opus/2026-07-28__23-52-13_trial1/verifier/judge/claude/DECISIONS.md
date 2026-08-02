# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading NWB files directly from the filesystem using `BehaviorOphysExperiment.from_nwb_path()`. It constructs an experiment table from CSV metadata files, filters to active behavior experiments, and excludes Multiscope (MESO.1) sessions. Each NWB file is loaded individually.

ii.
```python
def get_experiment_table():
    nwb_files = os.listdir(NWB_DIR)
    all_exp_ids = [int(re.search(r'(\d+)', f).group(1)) for f in nwb_files if f.endswith('.nwb')]
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(all_exp_ids)]
    exp_table = exp_table[exp_table['behavior_type'] == 'active_behavior']
    exp_table = exp_table[exp_table['equipment_name'] != 'MESO.1']
    return exp_table

def load_experiment(exp_id):
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{exp_id}.nwb')
    return BehaviorOphysExperiment.from_nwb_path(nwb_path)
```

iii. The AI chose to load NWB files directly rather than using the S3 cache because the data files were already available locally. It filtered by `behavior_type == 'active_behavior'` and excluded MESO.1 equipment to ensure consistent frame rates across sessions.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the experiment metadata. Each experiment's metadata contains the mouse_id, which is collected during Pass 2 and mapped to a subject index.

ii.
```python
subject_set = {}  # mouse_id -> index
# ...
meta = session_data['metadata']
mouse_id = str(meta['mouse_id'])
if mouse_id not in subject_set:
    subject_set[mouse_id] = len(subject_set)
all_subject_idx.append(subject_set[mouse_id])
```

iii. The mouse_id from the experiment metadata is the standard identifier for each animal.

## 1-c. How are the data split into sessions?

i. Each experiment (single imaging plane / NWB file) is treated as its own session. The AI does NOT group multiple experiments by `ophys_session_id`. Each of the 168 active-behavior Scientifica experiments becomes a separate session in the output.

ii.
```python
for i, exp_id in enumerate(exp_ids):
    # ...
    dataset = load_experiment(exp_id)
    session_data = extract_session_data(dataset, temp_image_to_idx)
    # Each experiment becomes one session
    all_neural.append(neural_trials)
```

iii. The AI treated each experiment as a session. For the Scientifica (single-plane) data this happens to be equivalent to one session = one experiment, so in practice the sessions match.

## 1-d. How are the data split into trials?

i. Trials are segmented using the built-in trials table from the dataset. Go and catch trials are included; aborted and auto-rewarded trials are excluded. Trial boundaries are defined by `start_time` to `stop_time` using ophys timestamps, yielding variable-length trials. A minimum of 5 frames per trial is required.

ii.
```python
valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]
# ...
for _, trial_row in valid_trials.iterrows():
    start_time = trial_row['start_time']
    stop_time = trial_row['stop_time']
    frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
    frame_indices = np.where(frame_mask)[0]
    if len(frame_indices) < 5:
        continue
```

iii. The AI used the SDK's trials table for segmentation, filtering to go and catch trials as specified in the instructions. The 5-frame minimum avoids degenerate trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be go or catch (not aborted, not auto-rewarded), (2) must have at least 5 ophys frames, (3) sessions with fewer than 2 valid trials are skipped, (4) sessions with fewer than 2 neurons are skipped.

ii.
```python
valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]
# ...
if len(frame_indices) < 5:
    continue
# ...
if n_neurons < 2:
    return None
# ...
if len(neural_trials) < 2:
    continue
```

iii. The filtering matches the instructions (go + catch, exclude aborted + auto-rewarded). Additional filters (min 5 frames, min 2 neurons) are conservative quality controls.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dataset.events` - the deconvolved calcium events computed using FastLZeroSpikeInference. This is NOT the dF/F traces.

ii.
```python
events = dataset.events
events_array = np.vstack(events['events'].values).astype(np.float32)
```

iii. The AI explicitly chose events over dF/F based on the methods.txt statement: "For all analysis of neural data we used the detected calcium events." The CONVERSION_NOTES.md documents: "Neural data: Use `events` (calcium events from FastLZeroSpikeInference) NOT `dff_traces`."

## 2-b. How is the `neural` data processed?

i. The events are extracted from the dataset, stacked into a (n_neurons, T) array, and sliced per trial. No additional processing (filtering, normalization) is applied.

ii.
```python
events = dataset.events
events_array = np.vstack(events['events'].values).astype(np.float32)
# ...
neural_trial = events_array[:, frame_indices]
```

iii. The AI noted that the Allen SDK pipeline already applies ROI filtering, neuropil subtraction, and event detection. No further processing was deemed necessary.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filtering is applied beyond the Allen pipeline's ROI classifier. Sessions with fewer than 2 neurons are excluded.

ii.
```python
if n_neurons < 2:
    return None
```

iii. The AI noted: "ROI filtering already applied by Allen pipeline. No additional filtering needed."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. Each trial's data is extracted using a boolean mask on `ophys_ts` for frames between `start_time` (inclusive) and `stop_time` (exclusive).

ii.
```python
frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
frame_indices = np.where(frame_mask)[0]
neural_trial = events_array[:, frame_indices]
```

iii. The instructions say to "temporally align based on ophys timestamp." The AI aligns all data streams to the ophys timebase.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The data is kept at the native ophys frame rate. The time bin size is hardcoded as `1000.0 / 31.0` (~32.3 ms), assuming 31 Hz Scientifica frame rate.

ii.
```python
'time_bin_size': 1000.0 / 31.0,  # ~32.3 ms
```

iii. The AI excluded Multiscope sessions to ensure a consistent frame rate across all sessions. The hardcoded value assumes 31 Hz.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations`, specifically the `image_name` column from the `change_detection_behavior` stimulus block. A forward-fill approach is used: during image presentation frames the current image name is used, and during gray-screen gaps the last-shown image is forward-filled.

ii.
```python
sp_active = sp[sp['stimulus_block_name'] == 'change_detection_behavior'].copy()
# ...
for j in range(len(starts)):
    img_name = names[j]
    if not isinstance(img_name, str) or img_name == 'omitted':
        continue
    mask = (ophys_ts >= starts[j]) & (ophys_ts < ends[j])
    image_idx[mask] = image_to_idx[img_name]
# Forward-fill
last_img = -1
for i in range(n_tp):
    if image_idx[i] >= 0:
        last_img = image_idx[i]
    elif last_img >= 0:
        image_idx[i] = last_img
```

iii. The AI used stimulus_presentations to get precise image timing, with forward-fill to handle gray-screen intervals. Omitted stimuli are skipped.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer indices via a global sorted mapping. The image identity at each ophys timepoint is determined from stimulus presentation start/end times, with forward-fill during gray screens. Any remaining -1 values (before the first stimulus) are filled with the first valid image index.

ii.
```python
all_image_names = sorted(list(all_image_names))
image_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
# ...
# Fix any -1 image indices (before first stimulus)
neg_mask = img_trial < 0
if neg_mask.any():
    first_valid = img_trial[~neg_mask][0] if (~neg_mask).any() else 0
    img_trial[neg_mask] = first_valid
```

iii. A global mapping ensures consistent codes across sessions. Forward-filling means image identity is never undefined during a trial.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at each ophys timestamp for the full session, then sliced using the same frame indices as the neural data.

ii.
```python
image_idx_full = get_image_at_ophys(sp_active, ophys_ts, image_to_idx)
# ...
img_trial = image_idx_full[frame_indices]
```

iii. By computing image identity at ophys timestamps, alignment with neural data is guaranteed.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations` using the `is_change` column. Presentations where `is_change == True` are marked as change events.

ii.
```python
def get_change_at_ophys(sp_active, ophys_ts):
    change = np.zeros(n_tp, dtype=np.int32)
    change_sp = sp_active[sp_active['is_change'] == True]
    for _, row in change_sp.iterrows():
        mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
        change[mask] = 1
    return change
```

iii. The AI used the stimulus presentations table's `is_change` flag, marking change=1 only during the presentation duration (start_time to end_time, ~250ms).

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary array is created at ophys timestamps. For each stimulus presentation where `is_change == True`, frames within [start_time, end_time) are set to 1. All other frames are 0.

ii. See 4-a code snippet.

iii. No additional processing beyond the binary encoding.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is applied.

ii. N/A - binary variable.

iii. The instructions specify "binary variable" so no thresholding is needed.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change signal is computed at ophys timestamps for the full session, then sliced using the same frame indices as neural data.

ii.
```python
change_full = get_change_at_ophys(sp_active, ophys_ts)
# ...
change_trial = change_full[frame_indices]
```

iii. Same alignment approach as image identity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides speed values and timestamps from the running wheel encoder.

ii.
```python
running = dataset.running_speed
running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
```

iii. Standard SDK interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is resampled from its native timestamps (~60 Hz) to ophys timestamps using `np.interp` (linear interpolation). Then discretized into 5 percentile-based bins using global percentile edges computed across all sessions. Bin edges are computed from full-session running data (not just trial data).

ii.
```python
def resample_to_ophys(signal_timestamps, signal_values, ophys_timestamps):
    return np.interp(ophys_timestamps, signal_timestamps, signal_values).astype(np.float32)

# Global percentile computation
all_running_concat = np.concatenate(all_running_values)
running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))

# Binning
running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. Linear interpolation preserves the signal shape. Global percentile binning ensures consistent categories across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins (quintiles). Bin edges are computed from all non-NaN running speed values across all sessions. `np.digitize` is used with the inner bin edges, then clipped to [0, 4].

ii.
```python
running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))
running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. Five percentile bins as specified in the instructions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is resampled to ophys timestamps before trial segmentation, so it uses the same time indices as neural data. The same `frame_indices` are used for both.

ii.
```python
running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
# ...
running_trial = running_binned[frame_indices]
```

iii. Resampling to ophys timebase before slicing ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using the `pupil_width` column.

ii.
```python
eye = dataset.eye_tracking
pupil_raw = eye['pupil_width'].values
pupil_ts = eye['timestamps'].values
```

iii. `pupil_width` is the SDK's measure of pupil diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaN values are filtered out from the raw pupil data before interpolation (rather than filtering by `likely_blink`). The valid values are linearly interpolated to ophys timestamps using `np.interp`. Then discretized into 5 percentile bins with global edges. If fewer than 10 valid pupil values exist, the entire session gets NaN pupil data.

ii.
```python
valid_mask = ~np.isnan(pupil_raw)
if valid_mask.sum() > 10:
    pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
else:
    pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
```

iii. The AI filtered NaN values rather than using `likely_blink`. This is a simpler approach but misses blink frames that have non-NaN but corrupted values.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 percentile bins with global edges. NaN pupil values get bin 0.

ii.
```python
pupil_binned = np.zeros(len(ophys_ts), dtype=np.int32)
if valid_pupil_mask.any() and pupil_edges is not None:
    pupil_binned[valid_pupil_mask] = np.clip(
        np.digitize(pupil_full[valid_pupil_mask], pupil_edges[1:-1]), 0, 4
    ).astype(np.int32)
```

iii. Five percentile bins as specified in the instructions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: interpolated to ophys timestamps before trial segmentation.

ii.
```python
pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
# ...
pupil_trial = pupil_binned[frame_indices]
```

iii. Same alignment as running speed and neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
def get_trial_outcome(trial_row):
    if trial_row['hit']: return 0
    if trial_row['miss']: return 1
    if trial_row['false_alarm']: return 2
    if trial_row['correct_reject']: return 3
    return -1
```

iii. These four columns are the SDK's canonical trial outcome labels.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) using the same ordering as the reference: hit=0, miss=1, false_alarm=2, correct_reject=3. The code is repeated for all timepoints in the trial (static per-trial variable broadcast to time-varying).

ii.
```python
TRIAL_OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
# ...
outcome = get_trial_outcome(trial_row)
outcome_trial = np.full(len(frame_indices), outcome, dtype=np.int32)
```

iii. The integer mapping matches the reference. Broadcasting to all timepoints makes it compatible with the time-varying output format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiment loading**: Caught with try/except, experiment skipped with error message.
- **Missing pupil data**: If fewer than 10 valid values, entire session gets NaN pupil (mapped to bin 0).
- **Negative image indices**: If image identity is -1 at trial start (before first stimulus), forward-filled with next valid image.
- **Insufficient neurons**: Sessions with < 2 neurons skipped.
- **Insufficient trials**: Sessions with < 2 trials after segmentation skipped.
- **Short trials**: Trials with < 5 frames skipped.

ii.
```python
try:
    dataset = load_experiment(exp_id)
except Exception as e:
    print(f'  ERROR loading experiment {exp_id}: {e}')
    all_session_data.append(None)
    continue

# Missing pupil
if valid_mask.sum() > 10:
    pupil_full = resample_to_ophys(...)
else:
    pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)

# Negative image indices
neg_mask = img_trial < 0
if neg_mask.any():
    first_valid = img_trial[~neg_mask][0] if (~neg_mask).any() else 0
    img_trial[neg_mask] = first_valid
```

iii. The try/except prevents crashes from bad NWB files. The NaN-to-0 mapping for pupil is a conservative default.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via `BehaviorOphysExperiment.from_nwb_path()`. The code also has a two-pass structure where data is loaded in Pass 1 and then image indices are recomputed in Pass 2 with the global image mapping.

ii. N/A

iii. NWB file loading is I/O-bound and dominates runtime. The two-pass approach adds overhead for recomputing image indices.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `get_image_at_ophys` function has a loop over stimulus presentations that could potentially be vectorized using `np.searchsorted`. The forward-fill loop over all timepoints is also sequential.

ii.
```python
for j in range(len(starts)):
    mask = (ophys_ts >= starts[j]) & (ophys_ts < ends[j])
    image_idx[mask] = image_to_idx[img_name]

# Forward-fill loop
last_img = -1
for i in range(n_tp):
    if image_idx[i] >= 0:
        last_img = image_idx[i]
    elif last_img >= 0:
        image_idx[i] = last_img
```

iii. The stimulus presentation loop creates boolean masks for each presentation. The forward-fill could use `pd.Series.ffill()`.

## 9-c. What processing does the code repeat multiple times?

i. The code uses a two-pass approach. In Pass 1, it loads all experiments and computes image indices with a temporary (incomplete) image mapping. In Pass 2, it recomputes image indices with the final global mapping. This means `get_image_at_ophys` is called twice per session.

ii.
```python
# Pass 1: temporary mapping
temp_image_to_idx = {name: idx for idx, name in enumerate(sorted(all_image_names))}
session_data = extract_session_data(dataset, temp_image_to_idx)

# Pass 2: recompute with global mapping
session_data['image_idx_full'] = get_image_at_ophys(session_data['sp_active'], session_data['ophys_ts'], image_to_idx)
```

iii. The two-pass approach is needed to compute global statistics first, but image index computation is done twice per session.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The temporary image index computation in Pass 1 is discarded and recomputed in Pass 2. The `sp_active` data is kept in memory between passes specifically for this recomputation.

ii.
```python
# Pass 1 computes image_idx_full with temp mapping - discarded
session_data = extract_session_data(dataset, temp_image_to_idx)
# Pass 2 recomputes
session_data['image_idx_full'] = get_image_at_ophys(...)
```

iii. The Pass 1 image indices serve no purpose since they use an incomplete mapping.
