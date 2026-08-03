# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by iterating over NWB files in the data directory. It reads an experiment metadata table CSV (`ophys_experiment_table.csv`), filters it to available NWB files, then loads each experiment individually using the AllenSDK `BehaviorOphysExperiment.from_nwb_path()` function. A two-pass approach is used: Pass 1 loads all experiments and collects global statistics (image names, running/pupil percentile edges), Pass 2 re-extracts image indices with the global mapping and segments trials.

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

iii. The agent identified `BehaviorOphysExperiment.from_nwb_path()` as the correct SDK function from exploring the reference code and tutorials. The two-pass approach was chosen to compute global percentile bins for running speed and pupil diameter across all sessions before segmenting trials.

## 1-b. How are the data split into subjects?

i. Subjects (mice) are identified by the `mouse_id` field in each experiment's metadata. A dictionary maps mouse IDs to indices. Each session is assigned to its subject via `subject_idx`.

ii.
```python
mouse_id = str(meta['mouse_id'])
if mouse_id not in subject_set:
    subject_set[mouse_id] = len(subject_set)
all_subject_idx.append(subject_set[mouse_id])
```

iii. The agent used the `mouse_id` from the experiment metadata, which is the standard Allen SDK field identifying the animal. CONVERSION_NOTES documents 37 unique mice in the converted data (subset of the 82 in the full dataset).

## 1-c. How are the data split into sessions?

i. Each ophys experiment (NWB file) is treated as one session. After filtering to active behavior and excluding Multiscope (MESO.1) equipment, 168 experiments remain. For Scientifica rigs, there is a 1:1 mapping between experiments and sessions (one imaging plane per session).

ii.
```python
exp_ids = exp_table['ophys_experiment_id'].values
# Each experiment is iterated as one session
for i, exp_id in enumerate(exp_ids):
    ...
    all_neural.append(neural_trials)
```

iii. The agent notes in CONVERSION_NOTES: "Each experiment = one session: Each imaging plane treated as independent session." This is correct for Scientifica rigs where each session has exactly one imaging plane.

## 1-d. How are the data split into trials?

i. Trials are segmented using the `start_time` and `stop_time` from the experiment's trials table. Ophys frames within each trial's time window are selected, and the neural/output data at those frames form the trial data. Trials with fewer than 5 frames are discarded.

ii.
```python
for _, trial_row in valid_trials.iterrows():
    start_time = trial_row['start_time']
    stop_time = trial_row['stop_time']
    frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
    frame_indices = np.where(frame_mask)[0]
    if len(frame_indices) < 5:
        continue
    neural_trial = events_array[:, frame_indices]
```

iii. The agent used the trial boundaries defined by the Allen SDK's trials table (start_time, stop_time), which define the experiment's trial structure. The minimum 5-frame threshold guards against degenerate trials.

## 1-e. How are trials filtered based on quality controls?

i. Only "Go" and "Catch" trials are included. "Aborted" and "Auto-rewarded" trials are excluded. Additionally, trials with fewer than 5 ophys frames are excluded, and sessions with fewer than 2 valid trials are skipped entirely.

ii.
```python
valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]
if len(valid_trials) < 2:
    return None
```

iii. The agent followed the instructions directly: "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials." The minimum 2-trial requirement ensures the decoder can have both training and validation data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `dataset.events`, which contains calcium events detected using the FastLZeroSpikeInference algorithm applied to dF/F traces.

ii.
```python
events = dataset.events
events_array = np.vstack(events['events'].values).astype(np.float32)
```

iii. The agent identified from the methods text that "For all analysis of neural data we used the detected calcium events" and from the SDK that `events` contains deconvolved calcium events. This is consistent with the reference paper's methodology.

## 2-b. How is the `neural` data processed?

i. The events data is extracted as a 2D array (n_neurons x n_timepoints) by vstacking the per-cell event traces. It is cast to float32. No additional filtering, normalization, or smoothing is applied. The data is then sliced per trial based on ophys frame indices.

ii.
```python
events_array = np.vstack(events['events'].values).astype(np.float32)
# Per trial:
neural_trial = events_array[:, frame_indices]
```

iii. The agent's CONVERSION_NOTES states: "Neural data: Use events (calcium events from FastLZeroSpikeInference) NOT dff_traces" and "ROI filtering already applied by Allen pipeline (classifier-based)." No additional processing is applied beyond slicing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level filtering is applied. The AI relies on the Allen pipeline's existing ROI filtering (classifier-based cell segmentation). Sessions with fewer than 2 neurons are skipped.

ii.
```python
n_neurons = events_array.shape[0]
if n_neurons < 2:
    return None
```

iii. From CONVERSION_NOTES: "Neuron curation rules: ROI filtering already applied by Allen pipeline. No additional filtering needed." The Allen SDK already applies quality filtering through a cell specimen classifier.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. Each trial's start and stop times define a window, and ophys frames within that window are selected. The alignment event is "Trial start time" - i.e., data is aligned to the beginning of each trial as defined by the trials table.

ii.
```python
frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
frame_indices = np.where(frame_mask)[0]
neural_trial = events_array[:, frame_indices]
```

iii. The instructions state "Temporally align based on ophys timestamp." The agent aligns all data streams (neural, running, pupil, stimulus) to the ophys timestamps, then segments by trial boundaries. Metadata records `temporal_alignment_event: 'Trial start time'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native ophys frame rate of ~31 Hz (Scientifica rigs), giving a time bin of approximately 32.3 ms. No temporal rebinning is applied. Multiscope sessions (11 Hz) are excluded entirely.

ii.
```python
# In metadata:
'time_bin_size': 1000.0 / 31.0,  # ~32.3 ms

# Multiscope exclusion:
exp_table = exp_table[exp_table['equipment_name'] != 'MESO.1']
```

iii. The agent excluded Multiscope sessions because they have a different frame rate (11 Hz vs 31 Hz), and the format requires consistent time bins across all sessions. CONVERSION_NOTES confirms: "Excluded Multiscope (11 Hz) sessions for consistent time bins."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name`, `start_time`, and `end_time` columns, filtered to the `change_detection_behavior` stimulus block.

ii.
```python
sp = dataset.stimulus_presentations
sp_active = sp[sp['stimulus_block_name'] == 'change_detection_behavior'].copy()

# In get_image_at_ophys:
starts = sp_active['start_time'].values
ends = sp_active['end_time'].values
names = sp_active['image_name'].values
```

iii. The agent identified 16 unique image names across the dataset (8 per image set, 2 sets: A and B), from the stimulus presentations table.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each stimulus presentation in the change detection block, ophys frames during that presentation are assigned the corresponding image index. Between presentations (gray screen), the last shown image index is forward-filled. Before the first image presentation, indices default to -1 and are then replaced with the first valid image index. Images named 'omitted' are skipped. A global mapping of all 16 image names to integer indices (0-15) is used.

ii.
```python
def get_image_at_ophys(sp_active, ophys_ts, image_to_idx):
    n_tp = len(ophys_ts)
    image_idx = np.full(n_tp, -1, dtype=np.int32)
    for j in range(len(starts)):
        img_name = names[j]
        if not isinstance(img_name, str) or img_name == 'omitted':
            continue
        if img_name not in image_to_idx:
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
    return image_idx
```

iii. The agent decided to forward-fill image identity during gray screens (between presentations) with the last shown image. This creates a continuous signal rather than having undefined periods. The global image mapping ensures consistent indices across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at each ophys timestamp directly (by checking which stimulus presentation window each timestamp falls in), so it is inherently aligned with the neural data at the same temporal resolution.

ii. Same as 3-b - the function operates on `ophys_ts` directly.

iii. Since both neural data and image identity are indexed by the same ophys timestamps and sliced by the same frame indices per trial, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the `stimulus_presentations` table (filtered to `change_detection_behavior` block), along with `start_time` and `end_time`.

ii.
```python
def get_change_at_ophys(sp_active, ophys_ts):
    n_tp = len(ophys_ts)
    change = np.zeros(n_tp, dtype=np.int32)
    change_sp = sp_active[sp_active['is_change'] == True]
    for _, row in change_sp.iterrows():
        mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
        change[mask] = 1
    return change
```

iii. The agent used the `is_change` flag from stimulus presentations to identify change events.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary signal is created: 1 during the duration of a change stimulus presentation (from start_time to end_time), and 0 otherwise. Only presentations marked with `is_change == True` are set to 1.

ii. Same as 4-a code snippet.

iii. The agent defined image change as a time-varying binary signal. The "change" value of 1 persists for the entire duration of the change presentation (~250ms), which is approximately 7-8 ophys frames.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is needed. Output values are `['no_change', 'change']`.

ii.
```python
output_values = [
    ...
    ['no_change', 'change'],
    ...
]
```

iii. The instructions specify "binary variable. Have value of 1 right after a change in image identity, otherwise 0." The agent implements this directly.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed at ophys timestamps and sliced by the same frame indices as neural data per trial, ensuring alignment.

ii.
```python
change_full = get_change_at_ophys(sp_active, ophys_ts)
# Per trial:
change_trial = change_full[frame_indices]
```

iii. Same alignment mechanism as image identity and neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides speed values with their own timestamps (at ~60 Hz).

ii.
```python
running = dataset.running_speed
running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
```

iii. The agent used the AllenSDK's `running_speed` property which returns a DataFrame with `timestamps` and `speed` columns.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed (~60 Hz) is resampled to ophys timestamps (~31 Hz) using linear interpolation (`np.interp`), then discretized into 5 equal percentile bins using globally-computed percentile edges.

ii.
```python
def resample_to_ophys(signal_timestamps, signal_values, ophys_timestamps):
    return np.interp(ophys_timestamps, signal_timestamps, signal_values).astype(np.float32)

# Global percentile computation:
all_running_concat = np.concatenate(all_running_values)
running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))

# Binning:
running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. The agent chose global percentiles (across all sessions) rather than per-session percentiles. The `np.linspace(0, 100, 6)` creates 6 edges defining 5 equal-frequency bins at [0%, 20%, 40%, 60%, 80%, 100%] percentiles.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (labeled bin_0 through bin_4) using `np.digitize` with global percentile edges. `np.clip` ensures values at the extremes map to bins 0 or 4.

ii.
```python
running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. The instructions specify "discretized into five equal percentile bins." The agent uses edges from the 0th, 20th, 40th, 60th, 80th, and 100th percentiles to create 5 bins with approximately equal numbers of samples globally.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is resampled to ophys timestamps via linear interpolation, then binned, and sliced per trial by the same frame indices as neural data.

ii.
```python
running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
# Per trial:
running_trial = running_binned[frame_indices]
```

iii. The interpolation to ophys timestamps ensures temporal alignment with the neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the `pupil_width` column in `dataset.eye_tracking`, along with the `timestamps` column.

ii.
```python
eye = dataset.eye_tracking
pupil_raw = eye['pupil_width'].values
pupil_ts = eye['timestamps'].values
```

iii. The agent chose `pupil_width` as the pupil diameter measure. The eye tracking data also contains `pupil_area`, `pupil_height`, and `pupil_phi`. The instructions refer to "pupil diameter" which maps most directly to `pupil_width` (diameter of the fitted ellipse).

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaN values are filtered out before interpolation. Valid pupil width values are resampled to ophys timestamps via linear interpolation, then discretized into 5 equal percentile bins using globally-computed edges. If fewer than 10 valid samples exist, the entire session's pupil data is set to NaN and binned as 0.

ii.
```python
valid_mask = ~np.isnan(pupil_raw)
if valid_mask.sum() > 10:
    pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
else:
    pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)

# Global percentile computation:
all_pupil_concat = np.concatenate(all_pupil_values)
pupil_edges = np.percentile(all_pupil_concat, np.linspace(0, 100, 6))

# Binning:
pupil_binned[valid_pupil_mask] = np.clip(
    np.digitize(pupil_full[valid_pupil_mask], pupil_edges[1:-1]), 0, 4
).astype(np.int32)
```

iii. The agent handles missing pupil data (NaN values) by excluding them from interpolation and percentile computation. Sessions without sufficient eye tracking data default to bin 0.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal percentile bins using global percentile edges, with `np.digitize` and `np.clip`.

ii. See 6-b code.

iii. Instructions specify "discretized into five equal percentile bins." The agent uses the same approach as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is resampled to ophys timestamps via linear interpolation (after NaN removal), then binned and sliced per trial by the same frame indices as neural data.

ii.
```python
pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
# Per trial:
pupil_trial = pupil_binned[frame_indices]
```

iii. Same alignment mechanism as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns in the trials table.

ii.
```python
def get_trial_outcome(trial_row):
    if trial_row['hit']: return 0
    if trial_row['miss']: return 1
    if trial_row['false_alarm']: return 2
    if trial_row['correct_reject']: return 3
    return -1
```

iii. The agent uses the four trial outcome categories as defined by the Allen SDK, mapping them to integers 0-3.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Each trial is assigned a single integer outcome (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) based on the boolean flags. This static per-trial value is repeated across all timepoints within the trial to create a time-varying output.

ii.
```python
outcome = get_trial_outcome(trial_row)
outcome_trial = np.full(len(frame_indices), outcome, dtype=np.int32)
```

iii. The instructions specify trial outcome as "Static per-trial." The agent represents it as a constant value repeated across all timepoints in the trial. The output values are ordered as `['hit', 'miss', 'false_alarm', 'correct_reject']`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of missing/problematic data are handled:
- **Missing pupil data**: NaN values are excluded from interpolation. Sessions with <10 valid pupil samples get NaN-filled arrays, which are binned as 0.
- **Missing image identity**: Before the first stimulus presentation, image index is -1; these are replaced with the first valid image index in the trial.
- **Failed experiment loading**: Wrapped in try/except; failed loads are skipped.
- **Short trials**: Trials with fewer than 5 frames are excluded.
- **All-zero neural data**: Acknowledged in verification but not filtered (genuine calcium events can be zero).
- **Omitted stimuli**: Images named 'omitted' are skipped in image identity computation.

ii.
```python
# Missing pupil
valid_mask = ~np.isnan(pupil_raw)
if valid_mask.sum() > 10:
    pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
else:
    pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)

# Missing image identity
neg_mask = img_trial < 0
if neg_mask.any():
    first_valid = img_trial[~neg_mask][0] if (~neg_mask).any() else 0
    img_trial[neg_mask] = first_valid

# Failed experiment loading
try:
    dataset = load_experiment(exp_id)
except Exception as e:
    print(f'  ERROR loading experiment {exp_id}: {e}')
    all_session_data.append(None)
    continue
```

iii. The agent documented edge cases in CONVERSION_NOTES Step 10: "4 trials had -1 image identity at first frame" and "some trials genuinely have no detected calcium events."

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is Pass 1 (loading all experiments), which took ~1277 seconds (21.3 minutes) for 168 experiments. Loading each NWB file via the AllenSDK takes 4-16 seconds depending on experiment size. Pass 2 (trial segmentation) is much faster at ~1 second per experiment. Total processing time was ~24.6 minutes.

ii.
```python
# Pass 1 timing output:
# Pass 1 completed in 1277.4s
# Pass 2 individual experiments: ~1.0-1.2s each
```

iii. The I/O-bound NWB file loading dominates the runtime. The agent prints timing information for each batch of experiments.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. **`get_image_at_ophys`**: The loop over stimulus presentations (line 81-88) could use `np.searchsorted` to vectorize assignment.
2. **Forward-fill loop in `get_image_at_ophys`** (lines 91-96): Could use `pd.Series.ffill()` or a vectorized approach.
3. **`get_change_at_ophys`**: The `iterrows()` loop over change presentations could be vectorized similarly to image identity.
4. **`segment_trials`**: The loop over `valid_trials.iterrows()` could potentially be vectorized, though trial-by-trial segmentation is inherently sequential.

ii.
```python
# Non-vectorized loop in get_image_at_ophys:
for j in range(len(starts)):
    ...
    mask = (ophys_ts >= starts[j]) & (ophys_ts < ends[j])
    image_idx[mask] = image_to_idx[img_name]

# Non-vectorized forward-fill:
for i in range(n_tp):
    if image_idx[i] >= 0:
        last_img = image_idx[i]
    elif last_img >= 0:
        image_idx[i] = last_img
```

iii. The agent's CONVERSION_NOTES do not specifically identify vectorization opportunities; they note the two-pass approach and overall timing but don't analyze specific loops.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats image index computation twice:
1. In Pass 1, `extract_session_data` calls `get_image_at_ophys` with a temporary (incomplete) `image_to_idx` mapping.
2. In Pass 2, `get_image_at_ophys` is called again with the global `image_to_idx` mapping.

Additionally, the NWB file is only loaded once (in Pass 1), but the session data is stored in memory. The stimulus presentation filtering (`sp_active`) is done in both `extract_session_data` (Pass 1) and stored in `session_data` for reuse in Pass 2.

ii.
```python
# Pass 1:
session_data = extract_session_data(dataset, temp_image_to_idx)

# Pass 2:
session_data['image_idx_full'] = get_image_at_ophys(session_data['sp_active'], session_data['ophys_ts'], image_to_idx)
```

iii. The agent acknowledges the two-pass approach in CONVERSION_NOTES but frames it as necessary for computing global statistics before final processing.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations may be considered unnecessary:
1. **Temporary image index computation in Pass 1**: The image indices computed with incomplete mappings are discarded and recomputed in Pass 2.
2. **Storing full session data in memory**: All session data (full ophys timeseries for events, running, pupil) is kept in memory across both passes, using significant RAM.
3. **Processing plots** (when `--show-processing` is enabled): Generated for visualization but not used in the final data.
4. **The `input` field**: Empty arrays `(0, n_timepoints)` are created for every trial since no decoder inputs are specified, but these must exist per the data format specification.

ii.
```python
# Temporary image mapping (discarded):
temp_image_to_idx = {name: idx for idx, name in enumerate(sorted(all_image_names))}
session_data = extract_session_data(dataset, temp_image_to_idx)

# Empty inputs:
input_trials = [np.zeros((0, t.shape[1]), dtype=np.float32) for t in neural_trials]
```

iii. The two-pass approach with temporary mapping is a design consequence of needing global image names before final assignment. The empty inputs are required by the data format specification despite having no decoder inputs.
