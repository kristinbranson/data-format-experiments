# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading NWB files directly from disk using `BehaviorOphysExperiment.from_nwb_path()`. It first reads the experiment metadata CSV to get a list of all experiment IDs, filters to active behavior experiments (and excludes Multiscope/MESO.1 sessions), then loads each NWB file individually. This contrasts with the reference, which uses the Allen SDK's `VisualBehaviorOphysProjectCache.from_s3_cache()` to load experiments via `get_behavior_ophys_experiment()`.

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

iii. The AI justified loading from NWB files directly rather than the SDK cache. It filtered to `behavior_type == 'active_behavior'` and excluded MESO.1 (Multiscope) sessions because they have a different frame rate (11 Hz vs 31 Hz), which violates the requirement that time bins be consistent across sessions.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the metadata of each loaded experiment using `meta['mouse_id']`. Each unique mouse_id is assigned an index.

ii.
```python
mouse_id = str(meta['mouse_id'])
if mouse_id not in subject_set:
    subject_set[mouse_id] = len(subject_set)
all_subject_idx.append(subject_set[mouse_id])
```

iii. The AI uses the mouse_id from each experiment's metadata to track subjects, which is equivalent to the reference's approach of using `mouse_id` from the experiment table.

## 1-c. How are the data split into sessions?

i. The AI treats each ophys experiment (one imaging plane) as a separate session. This differs from the reference, which groups experiments by `ophys_session_id` (combining multiple imaging planes from the same behavioral session into one session, merging neurons across planes).

ii.
```python
for i, exp_id in enumerate(exp_ids):
    # Each experiment is treated as its own session
    dataset = load_experiment(exp_id)
    session_data = extract_session_data(dataset, temp_image_to_idx)
    # ...
    all_neural.append(neural_trials)
```

iii. The AI's CONVERSION_NOTES state: "Each experiment = one session: Each imaging plane treated as independent session." Since the filtered data (Scientifica only) predominantly has 1 experiment per session, this results in a similar but not identical split. Multi-plane Scientifica sessions would be split into multiple sessions rather than merged.

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `dataset.trials` table. The AI filters to Go and Catch trials (excluding Aborted and Auto-rewarded), then extracts ophys frames from `start_time` to `stop_time` for each valid trial, giving variable-length trials.

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

iii. The AI used the SDK's trials table and filtered by go/catch status. The minimum trial length threshold of 5 frames is more aggressive than the reference's threshold of 0 frames (end_idx > start_idx). The AI uses `ophys_ts >= start_time` (inclusive) and `< stop_time` (exclusive), while the reference uses `np.searchsorted` which finds the first frame at or after the boundary.

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

iii. The reference additionally filters on `change_time.notna()`. The AI does not explicitly filter on `change_time` validity. The AI adds a neuron count filter (< 2) not in the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `dataset.events` (calcium events from FastLZeroSpikeInference deconvolution), NOT `dataset.dff_traces`. This is a significant difference from the reference, which uses `dff_traces`.

ii.
```python
events = dataset.events
events_array = np.vstack(events['events'].values).astype(np.float32)
```

iii. The AI justified this choice by citing methods.txt: "For all analysis of neural data we used the detected calcium events." The methods text from the paper does specify using calcium events, so the AI followed the paper's stated methodology. However, the reference solution uses dff_traces.

## 2-b. How is the `neural` data processed?

i. The neural data (events) is extracted as-is with no additional processing. Since the AI treats each experiment as a single session, there is no merging across imaging planes (unlike the reference which stacks dF/F from multiple planes).

ii.
```python
events = dataset.events
events_array = np.vstack(events['events'].values).astype(np.float32)
```

iii. The AI noted that the Allen SDK pipeline already applies quality control (ROI segmentation, filtering). No additional processing is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied beyond the Allen pipeline's built-in ROI filtering. Sessions with fewer than 2 neurons are excluded.

ii.
```python
if n_neurons < 2:
    return None
```

iii. The AI stated that ROI filtering is already applied by the Allen pipeline classifier. The reference also does not apply additional neural filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. Ophys frames from `start_time` to `stop_time` are extracted using boolean masking on the ophys timestamps array.

ii.
```python
frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
frame_indices = np.where(frame_mask)[0]
neural_trial = events_array[:, frame_indices]
```

iii. The AI uses `ophys_ts >= start_time` and `ophys_ts < stop_time`, which is slightly different from the reference's `np.searchsorted(ophys_ts, start_time)` and `np.searchsorted(ophys_ts, stop_time)`. Both approaches align to the ophys timebase; the difference is in boundary handling (inclusive vs exclusive at stop_time).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI hardcodes the time bin size as `1000.0 / 31.0` (~32.3 ms) rather than computing it from the data. No temporal rebinning is applied. The reference computes it from `np.median(np.diff(ophys_timestamps))`.

ii.
```python
'time_bin_size': 1000.0 / 31.0,  # ~32.3 ms
```

iii. The AI noted that Scientifica rigs run at 31 Hz. Hardcoding is functionally equivalent since only Scientifica sessions are included, but the reference dynamically computes this value.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The AI derives image identity from the `stimulus_presentations` table, specifically using `image_name`, `start_time`, and `end_time` columns. It filters to the `change_detection_behavior` stimulus block and excludes `omitted` stimuli. This contrasts with the reference which uses `initial_image_name` and `change_image_name` from the trials table.

ii.
```python
sp = dataset.stimulus_presentations
sp_active = sp[sp['stimulus_block_name'] == 'change_detection_behavior'].copy()

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

iii. The AI used the stimulus_presentations table to get precise timing of each image display, then forward-filled through gray screen periods. This provides frame-accurate image identity that accounts for the actual stimulus timing (250ms on, 500ms off) rather than using the simpler trial-level split at change_time.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global sorted mapping. During image presentations, the corresponding image index is assigned. Between presentations (gray screen), the last shown image index is forward-filled. Before the first stimulus in a trial, -1 values are replaced with the next valid image index.

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

iii. A global mapping ensures consistent codes across sessions. The forward-fill approach means gray screen periods carry the last-shown image identity, which differs from the reference approach of simply splitting the trial at `change_time`.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the full session level on ophys timestamps, then indexed per trial using the same frame indices as the neural data.

ii.
```python
image_idx_full = get_image_at_ophys(sp_active, ophys_ts, image_to_idx)
# ...
img_trial = image_idx_full[frame_indices]
```

iii. Using the same ophys timestamps and frame indices guarantees alignment with neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The AI derives image change from the `stimulus_presentations` table using the `is_change` column and the presentation's `start_time`/`end_time`. This differs from the reference which uses `change_time` from the trials table and a 750ms window.

ii.
```python
def get_change_at_ophys(sp_active, ophys_ts):
    change_sp = sp_active[sp_active['is_change'] == True]
    for _, row in change_sp.iterrows():
        mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
        change[mask] = 1
    return change
```

iii. The AI uses the stimulus presentation's own start/end times (which is the ~250ms image display period) for the change marker. The reference uses a 750ms window (one full image presentation + gray period) and only marks go trials.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary array is created at ophys timestamps, set to 1 only during the presentation time of stimuli marked as `is_change`. No additional processing.

ii.
```python
change_sp = sp_active[sp_active['is_change'] == True]
for _, row in change_sp.iterrows():
    mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
    change[mask] = 1
```

iii. The `is_change` flag in stimulus_presentations marks both go trial changes and catch trial "sham changes". The reference only marks go trials (where an actual image change occurs), not catch trials.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 or 1), no thresholding needed.

ii. N/A

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same approach as image identity - computed at full session level on ophys timestamps, then indexed per trial.

ii.
```python
change_full = get_change_at_ophys(sp_active, ophys_ts)
# ...
change_trial = change_full[frame_indices]
```

iii. Using the same frame indices guarantees alignment with neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, using the `timestamps` and `speed` columns.

ii.
```python
running = dataset.running_speed
running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
```

iii. Same source as the reference.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is resampled to ophys timestamps using `np.interp` (linear interpolation), then discretized into 5 percentile-based bins computed globally across all sessions.

ii.
```python
def resample_to_ophys(signal_timestamps, signal_values, ophys_timestamps):
    return np.interp(ophys_timestamps, signal_timestamps, signal_values).astype(np.float32)
# ...
running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))
running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. The AI uses `np.interp` which extrapolates at boundaries (using edge values) rather than `interp1d` with `fill_value=np.nan` as in the reference. The reference then handles NaN values by mapping them to bin 0. The AI's approach avoids NaN but may be slightly different at edges.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins using global percentiles across all sessions, with `np.digitize` and `np.clip` to ensure values stay in [0, 4].

ii.
```python
running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))
running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. Similar to the reference approach. The AI computes percentile bin edges from all sessions combined (global binning), matching the reference.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to ophys timestamps before trial segmentation, ensuring alignment with neural data via the same frame indices.

ii.
```python
running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
# ...
running_trial = running_binned[frame_indices]
```

iii. Same alignment approach as neural data - both use the same ophys frame indices.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using the `pupil_width` column.

ii.
```python
eye = dataset.eye_tracking
pupil_raw = eye['pupil_width'].values
pupil_ts = eye['timestamps'].values
```

iii. Same source variable as the reference (`pupil_width`).

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI filters out NaN values from pupil data (but does NOT filter blinks using `likely_blink` as the reference does), then resamples to ophys timestamps using `np.interp`, and discretizes into 5 percentile bins.

ii.
```python
pupil_raw = eye['pupil_width'].values
pupil_ts = eye['timestamps'].values
valid_mask = ~np.isnan(pupil_raw)
if valid_mask.sum() > 10:
    pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
```

iii. The AI removes NaN values before interpolation but does not explicitly filter blink frames using `likely_blink`. If blink frames produce NaN values in `pupil_width`, the NaN filtering may partially address blinks, but blink frames with non-NaN but corrupted values would not be filtered.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into 5 equal percentile bins using global percentiles, same approach as running speed.

ii.
```python
pupil_edges = np.percentile(all_pupil_concat, np.linspace(0, 100, 6))
pupil_binned[valid_pupil_mask] = np.clip(
    np.digitize(pupil_full[valid_pupil_mask], pupil_edges[1:-1]), 0, 4
).astype(np.int32)
```

iii. Same global percentile approach as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to ophys timestamps before trial segmentation, then indexed per trial using the same frame indices as neural data.

ii.
```python
pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
# ...
pupil_trial = pupil_binned[frame_indices]
```

iii. Same alignment approach as running speed and neural data.

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

iii. Same source as the reference.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) matching the order [hit, miss, false_alarm, correct_reject]. The outcome code is replicated across all time bins in the trial.

ii.
```python
outcome = get_trial_outcome(trial_row)
outcome_trial = np.full(len(frame_indices), outcome, dtype=np.int32)
```

iii. Same processing as the reference. The mapping order matches: hit=0, miss=1, false_alarm=2, correct_reject=3.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- Failed experiment loading: try/except with skip and error message
- Short trials (< 5 frames): skipped
- Missing eye tracking data: replaced with NaN array
- NaN pupil values: filtered before interpolation, remaining NaN set to bin 0
- Negative image indices (before first stimulus): forward/backward-filled
- Sessions with < 2 valid trials or < 2 neurons: skipped

ii.
```python
try:
    dataset = load_experiment(exp_id)
except Exception as e:
    print(f'  ERROR loading experiment {exp_id}: {e}')
    all_session_data.append(None)
    continue
# ...
if len(frame_indices) < 5:
    continue
# ...
try:
    eye = dataset.eye_tracking
    # ...
except:
    pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
```

iii. The AI handles edge cases robustly. The reference similarly uses try/except for sessions and handles truncated trials and NaN values.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB experiment file from disk via `BehaviorOphysExperiment.from_nwb_path()`. The AI uses a two-pass approach which loads experiments once in Pass 1 and reuses the data in Pass 2.

ii. N/A

iii. From CONVERSION_NOTES: "Processing time: ~24 min for full dataset (202 experiments)". Loading NWB files is I/O bound.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `get_image_at_ophys` function loops through all stimulus presentations individually, applying a boolean mask for each one. The forward-fill loop also iterates through all ophys timepoints. These could be vectorized using `np.searchsorted` and pandas forward-fill.

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

iii. The per-stimulus loop could be replaced with vectorized `np.searchsorted` calls. The forward-fill could use `pd.Series.ffill()`.

## 9-c. What processing does the code repeat multiple times?

i. The AI's two-pass approach means image identity indices are computed twice: once in Pass 1 (with a temporary mapping) and again in Pass 2 (with the global mapping). The Pass 1 computation is effectively wasted.

ii.
```python
# Pass 1: temporary mapping
temp_image_to_idx = {name: idx for idx, name in enumerate(sorted(all_image_names))}
session_data = extract_session_data(dataset, temp_image_to_idx)
# Pass 2: re-compute with global mapping
session_data['image_idx_full'] = get_image_at_ophys(session_data['sp_active'], session_data['ophys_ts'], image_to_idx)
```

iii. The two-pass approach was needed to compute global percentile edges and image mappings before final trial assembly, but the image identity computation in Pass 1 is redundant.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores full-session running speed and pupil values (`running_full`, `pupil_full`) in the session data during Pass 1, concatenates them all for global percentile computation, then also collects them per-trial. The full-session arrays are kept in memory across all passes even though only per-trial segments are needed for the output. Additionally, the `sp_active` stimulus presentations table is stored in session_data and persists through both passes.

ii.
```python
all_session_data.append(session_data)  # Keeps full session data in memory
```

iii. The reference avoids this by computing per-trial data immediately and discarding the full-session arrays.
