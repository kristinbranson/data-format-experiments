# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by listing NWB files in the data directory, reading the experiment table CSV, filtering to `active_behavior` experiments, and excluding Multiscope (`MESO.1`) sessions. Each experiment is loaded individually via `BehaviorOphysExperiment.from_nwb_path()`. This differs from the reference, which uses `VisualBehaviorOphysProjectCache.from_s3_cache()` and filters by `project_code == 'VisualBehavior'`.

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

iii. The AI chose to load NWB files directly rather than using the SDK cache, and explicitly excluded MESO.1 equipment for consistent frame rates. The CONVERSION_NOTES.md states: "Exclude Multiscope (11 Hz) sessions - different frame rate than Scientifica (31 Hz). The format requires consistent time bins across all sessions."

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the experiment metadata. Each unique mouse_id is assigned an index.

ii.
```python
mouse_id = str(meta['mouse_id'])
if mouse_id not in subject_set:
    subject_set[mouse_id] = len(subject_set)
all_subject_idx.append(subject_set[mouse_id])
```

iii. The AI used the metadata's `mouse_id` field to identify unique subjects, same as the reference.

## 1-c. How are the data split into sessions?

i. The AI treats each individual experiment (imaging plane) as a separate session. This differs from the reference, which groups experiments by `ophys_session_id` and merges neurons from multiple imaging planes within a session.

ii.
```python
for i, exp_id in enumerate(exp_ids):
    # ...
    session_data = extract_session_data(dataset, temp_image_to_idx)
    # ...
    neural_trials, output_trials = segment_trials(session_data, running_edges, pupil_edges)
    all_neural.append(neural_trials)
```

iii. The CONVERSION_NOTES.md states: "Each experiment = one session: Each imaging plane treated as independent session." This is because the AI loaded NWB files individually rather than grouping by session.

## 1-d. How are the data split into trials?

i. Trials are defined using the `dataset.trials` table. Go and catch trials are included; aborted and auto-rewarded trials are excluded. Trial boundaries are from `start_time` to `stop_time` (variable length), using `ophys_ts >= start_time & ophys_ts < stop_time`.

ii.
```python
valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]
# ...
frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
frame_indices = np.where(frame_mask)[0]
if len(frame_indices) < 5:
    continue
```

iii. The AI's trial filtering matches the reference (go + catch, excluding aborted and auto-rewarded). However, the AI requires `len(frame_indices) >= 5` instead of just `> 0`, and does not additionally filter on `change_time.notna()`. The AI uses `>=` for start_time (same as reference's `searchsorted`) but `<` for stop_time (reference uses `searchsorted` which gives `<=` behavior for the stop boundary).

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded trials are excluded. Trials with fewer than 5 frames are skipped. Sessions with fewer than 2 valid trials or fewer than 2 neurons are excluded.

ii.
```python
valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]
# ...
if len(frame_indices) < 5:
    continue
# ...
if n_neurons < 2:
    return None
if len(valid_trials) < 2:
    return None
```

iii. The AI does not explicitly filter on `change_time.notna()` as the reference does. Instead, it filters on `go | catch`, which is similar but not identical. The reference requires valid `change_time` to ensure a well-defined change point.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `dataset.events` (deconvolved calcium events from FastLZeroSpikeInference), NOT `dataset.dff_traces` (delta F/F). This is a significant difference from the reference, which uses `dff_traces`.

ii.
```python
events = dataset.events
events_array = np.vstack(events['events'].values).astype(np.float32)
```

iii. The AI's CONVERSION_NOTES.md explicitly states: "Neural data: Use events (calcium events from FastLZeroSpikeInference) NOT dff_traces. Per methods.txt: 'For all analysis of neural data we used the detected calcium events'." The AI followed the paper's methods section rather than using raw dF/F.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied beyond extracting the events array and casting to float32. Unlike the reference, which merges neurons across multiple imaging planes within a session, the AI treats each plane as a separate session so no merging is needed.

ii.
```python
events = dataset.events
events_array = np.vstack(events['events'].values).astype(np.float32)
```

iii. The AI noted that ROI filtering is already applied by the Allen pipeline, so no additional filtering was needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. Sessions with fewer than 2 neurons are skipped.

ii.
```python
if n_neurons < 2:
    return None
```

iii. The AI noted that ROI filtering is already applied by the Allen pipeline classifier.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to `trial_start` (start_time from the trials table). Ophys frames from `start_time` to `stop_time` are extracted, giving variable-length trials.

ii.
```python
frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
frame_indices = np.where(frame_mask)[0]
neural_trial = events_array[:, frame_indices]
```

iii. Same approach as the reference: align to trial start time and extract variable-length windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The AI hardcodes the time bin size as `1000.0 / 31.0` (~32.3 ms), unlike the reference which computes it from the median inter-frame interval.

ii.
```python
'time_bin_size': 1000.0 / 31.0,  # ~32.3 ms
```

iii. The AI excluded Multiscope sessions to maintain a consistent ~31 Hz frame rate. The hardcoded value assumes all remaining sessions are at 31 Hz.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations` (filtered to `change_detection_behavior` block), using the `image_name` column with forward-filling during gray periods. This differs from the reference, which uses `initial_image_name` and `change_image_name` from the trials table with the `change_time` as the switch point.

ii.
```python
sp = dataset.stimulus_presentations
sp_active = sp[sp['stimulus_block_name'] == 'change_detection_behavior'].copy()
# ...
def get_image_at_ophys(sp_active, ophys_ts, image_to_idx):
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

iii. The AI uses the stimulus_presentations table to determine which image is on screen at each timepoint, which provides more precise timing than the trials table approach. Omitted stimuli are skipped. Gray screen periods are forward-filled with the last shown image.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer indices using a global sorted mapping. The mapping is built after collecting all unique image names across all sessions. Gray screen periods and times before first stimulus are forward-filled.

ii.
```python
all_image_names = sorted(list(all_image_names))
image_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
# Forward-fill handles gray screens
```

iii. Same global mapping approach as reference, but the source of image identity differs (stimulus_presentations vs trials table).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at each ophys timestamp using the stimulus_presentations start/end times, then extracted per trial using the same frame indices as neural data.

ii.
```python
image_idx_full = get_image_at_ophys(sp_active, ophys_ts, image_to_idx)
# ...
img_trial = image_idx_full[frame_indices]
```

iii. The image identity is computed over the full session at ophys timestamps, ensuring alignment with neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in `stimulus_presentations`, filtered to the `change_detection_behavior` block. This differs from the reference, which computes it from `change_time` and the `go` flag in the trials table.

ii.
```python
def get_change_at_ophys(sp_active, ophys_ts):
    change_sp = sp_active[sp_active['is_change'] == True]
    for _, row in change_sp.iterrows():
        mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
        change[mask] = 1
    return change
```

iii. The AI marks change=1 only during the change stimulus presentation window (start_time to end_time of the change flash, ~250ms), while the reference marks a 750ms window (flash + gray period). Also, the AI does not distinguish go vs catch trials for the change signal -- `is_change` is True for both go and catch trials in the stimulus_presentations table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Binary 0/1 variable. 1 during the change stimulus presentation window (ophys frames within start_time to end_time of the change flash), 0 otherwise.

ii.
```python
change_sp = sp_active[sp_active['is_change'] == True]
for _, row in change_sp.iterrows():
    mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
    change[mask] = 1
```

iii. No additional processing beyond the binary indicator.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 or 1). No thresholding is needed.

ii. N/A - binary by construction.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Computed at ophys timestamps over the full session, then extracted per trial using the same frame indices.

ii.
```python
change_full = get_change_at_ophys(sp_active, ophys_ts)
# ...
change_trial = change_full[frame_indices]
```

iii. Same alignment approach as image identity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, same as the reference.

ii.
```python
running = dataset.running_speed
running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
```

iii. Standard SDK interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is resampled to ophys timestamps using `np.interp` (linear interpolation), then discretized into 5 percentile bins computed globally across all sessions.

ii.
```python
def resample_to_ophys(signal_timestamps, signal_values, ophys_timestamps):
    return np.interp(ophys_timestamps, signal_timestamps, signal_values).astype(np.float32)

# Global percentile edges
all_running_concat = np.concatenate(all_running_values)
running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))

# Binning
running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. The AI uses `np.interp` (which extrapolates edge values) rather than `scipy.interpolate.interp1d` with `fill_value=np.nan` as in the reference. This means edge cases are handled differently: `np.interp` extends the last value instead of producing NaN. Also, percentiles are computed from full-session data rather than trial-segmented data only.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Discretized into 5 bins using global percentile edges via `np.digitize`.

ii.
```python
running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. Same 5-bin approach as reference. The AI clips values to 0-4 range.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to ophys timestamps before trial segmentation, ensuring alignment.

ii.
```python
running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
# ...
running_trial = running_binned[frame_indices]
```

iii. Same alignment approach as reference.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using the `pupil_width` column. Same as reference.

ii.
```python
eye = dataset.eye_tracking
pupil_raw = eye['pupil_width'].values
pupil_ts = eye['timestamps'].values
```

iii. Standard SDK interface for eye tracking data.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil data is filtered for NaN values (not `likely_blink` as in the reference), then resampled to ophys timestamps using `np.interp`, then discretized into 5 percentile bins.

ii.
```python
pupil_raw = eye['pupil_width'].values
pupil_ts = eye['timestamps'].values
valid_mask = ~np.isnan(pupil_raw)
if valid_mask.sum() > 10:
    pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
```

iii. The AI filters NaN values rather than using the `likely_blink` flag. This may miss some blink artifacts that are not NaN but are flagged by the blink detector.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 percentile bins computed globally.

ii.
```python
pupil_binned = np.zeros(len(ophys_ts), dtype=np.int32)
if valid_pupil_mask.any() and pupil_edges is not None:
    pupil_binned[valid_pupil_mask] = np.clip(
        np.digitize(pupil_full[valid_pupil_mask], pupil_edges[1:-1]), 0, 4
    ).astype(np.int32)
```

iii. NaN pupil values are mapped to bin 0 (same default as reference).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: interpolated to ophys timestamps, then extracted per trial.

ii.
```python
pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
# ...
pupil_trial = pupil_binned[frame_indices]
```

iii. Same alignment approach.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns in the trials table. Same as reference.

ii.
```python
def get_trial_outcome(trial_row):
    if trial_row['hit']: return 0
    if trial_row['miss']: return 1
    if trial_row['false_alarm']: return 2
    if trial_row['correct_reject']: return 3
    return -1
```

iii. Same four categories and mapping order as reference.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Boolean columns mapped to integer codes (0-3). The outcome is constant across all timepoints in a trial.

ii.
```python
outcome = get_trial_outcome(trial_row)
outcome_trial = np.full(len(frame_indices), outcome, dtype=np.int32)
```

iii. Same approach as reference: static per-trial value replicated across all frames.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases handled:
- Failed experiment loads are caught by try/except and skipped.
- Trials with fewer than 5 frames are skipped.
- Sessions with < 2 neurons or < 2 trials are skipped.
- Image indices of -1 (before first stimulus) are forward-filled with the next valid image.
- Pupil NaN values are handled gracefully (fallback to all-NaN array).

ii.
```python
try:
    dataset = load_experiment(exp_id)
except Exception as e:
    print(f'  ERROR loading experiment {exp_id}: {e}')
    all_session_data.append(None)
    continue
# ...
neg_mask = img_trial < 0
if neg_mask.any():
    first_valid = img_trial[~neg_mask][0] if (~neg_mask).any() else 0
    img_trial[neg_mask] = first_valid
```

iii. The AI handles edge cases robustly but uses a minimum 5-frame threshold instead of any valid frames.

## 9-a. What are the most time-consuming steps of the code?

i. Loading each experiment via `BehaviorOphysExperiment.from_nwb_path()` is the most time-consuming step, as it reads large NWB files from disk.

ii. N/A (I/O bound, not a code inefficiency)

iii. The AI noted total processing time of ~24 minutes for full dataset.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `get_image_at_ophys` function has a loop over stimulus presentations that could be vectorized. The forward-fill loop is also sequential.

ii.
```python
for j in range(len(starts)):
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

iii. These loops iterate over stimulus presentations and timepoints. The forward-fill could use `pd.Series.ffill()` or similar.

## 9-c. What processing does the code repeat multiple times?

i. The AI uses a two-pass approach: Pass 1 loads all experiments and collects global statistics, then Pass 2 re-extracts image indices with the global mapping. The `get_image_at_ophys` function is called twice for each session (once in each pass). The reference avoids this by collecting image names as strings first and mapping to codes after all data is loaded.

ii.
```python
# Pass 1: temp_image_to_idx
session_data = extract_session_data(dataset, temp_image_to_idx)
# ...
# Pass 2: Re-compute image indices with global image_to_idx
session_data['image_idx_full'] = get_image_at_ophys(session_data['sp_active'], session_data['ophys_ts'], image_to_idx)
```

iii. The two-pass approach is needed because the global image mapping is not known until all sessions are processed.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI stores full-session `sp_active` (stimulus presentations) data in memory across both passes, which is only needed for computing image indices. The plot_processing function recomputes binned running and pupil data instead of reusing already-computed values.

ii.
```python
# sp_active stored in session_data for Pass 2
'sp_active': sp_active,
# ...
# plot_processing recomputes bins
running_binned = np.clip(np.digitize(session_data['running_full'], running_edges[1:-1]), 0, 4)
```

iii. These are minor inefficiencies that don't affect correctness.
