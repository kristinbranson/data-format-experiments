# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using `h5py`, rather than using the AllenSDK cache. It reads the experiment metadata table from a CSV file (`ophys_experiment_table.csv`), identifies available NWB files on disk, and loads each experiment's NWB file individually. It filters for active behavior sessions (`passive == False`), which includes both VisualBehavior (single-plane) and VisualBehaviorMultiscope (multi-plane) projects.

ii.
```python
def load_experiment_table():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
    nwb_ids = [int(f.split('_')[-1].split('.')[0]) for f in nwb_files]
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(nwb_ids)]
    exp_table = exp_table[exp_table['passive'] == False]
    return exp_table

def load_nwb_data(nwb_path):
    data = {}
    with h5py.File(nwb_path, 'r') as f:
        ophys_ts = f['processing/ophys/dff/traces/timestamps'][:]
        events = f['processing/ophys/event_detection/data'][:]
        # ... loads trials, running, pupil, stimulus presentations
    return data
```

iii. From the trajectory (step 31): "Each NWB file = one ophys experiment (one imaging plane)." The AI chose to read NWB files directly with h5py rather than using the AllenSDK cache, as the data was available on disk. The filtering for `passive == False` was meant to select active behavior sessions.

## 1-b. How are the data split into subjects?

i. Subjects are identified as unique `mouse_id` values from the experiment table, sorted and converted to strings.

ii.
```python
all_subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
```

iii. The AI used the standard `mouse_id` field from the experiment metadata. Sorting ensures deterministic ordering.

## 1-c. How are the data split into sessions?

i. Each individual experiment (one imaging plane) is treated as a separate session. The AI does NOT group multiple imaging planes from the same ophys session together. Each NWB file becomes one session in the output.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
    nwb_data = load_nwb_data(nwb_path)
    result = process_experiment(nwb_data, all_image_names)
    # ... each experiment becomes a session
    neural_all.append(final_neural)
```

iii. From the trajectory (step 31): "Each experiment is a 'session' for our purposes." The AI treated each experiment (one imaging plane) as a separate session rather than grouping multiple planes from the same ophys session.

## 1-d. How are the data split into trials?

i. Trials are segmented using the trials table from the NWB file, from `trial_start_times` to `trial_stop_times`. Only Go and Catch trials are included (Aborted and Auto-rewarded are excluded). Trials with unknown outcomes or fewer than 2 frames are skipped. The trial window is variable length.

ii.
```python
def get_trial_mask(trial_idx, nwb_data):
    if nwb_data['trial_aborted'][trial_idx]:
        return False
    if nwb_data['trial_auto_rewarded'][trial_idx]:
        return False
    if not (nwb_data['trial_go'][trial_idx] or nwb_data['trial_catch'][trial_idx]):
        return False
    return True

# In process_experiment:
for trial_idx in range(n_trials_total):
    if not get_trial_mask(trial_idx, nwb_data):
        continue
    t_start = nwb_data['trial_start_times'][trial_idx]
    t_stop = nwb_data['trial_stop_times'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    frame_indices = np.where(frame_mask)[0]
    if len(frame_indices) < 2:
        continue
```

iii. From the trajectory (step 31): "Segment into trials: Go and Catch only, exclude Aborted and Auto-rewarded." The AI explicitly checks the `go` and `catch` boolean flags rather than just excluding aborted/auto-rewarded.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) excluding Aborted trials, (2) excluding Auto-rewarded trials, (3) requiring Go or Catch flag, (4) requiring a known outcome (not -1), (5) requiring at least 2 ophys frames, (6) requiring at least 2 valid trials per session. Additionally, neurons are filtered to valid ROIs only.

ii.
```python
# Trial filtering
if not get_trial_mask(trial_idx, nwb_data):
    continue
outcome = get_trial_outcome(trial_idx, nwb_data)
if outcome == -1:
    continue
if len(frame_indices) < 2:
    continue

# Session-level filtering
if neural_trials is None or len(neural_trials) < 2:
    print(f"    SKIPPED: insufficient valid trials")
    continue
```

iii. The AI's trajectory shows it intended to follow the instructions exactly: "Go and Catch trials only (exclude Aborted and Auto-rewarded)."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses **detected calcium events** (`processing/ophys/event_detection/data`), NOT dF/F traces. This is a fundamentally different neural signal from the reference solution.

ii.
```python
# In load_nwb_data:
events = f['processing/ophys/event_detection/data'][:]
data['events'] = events  # shape: (n_timepoints, n_cells)

# In process_experiment:
events = nwb_data['events']  # (n_timepoints, n_cells)
valid = nwb_data['valid_roi']
events = events[:, valid]
```

iii. From the trajectory (step 31): "Neural data: use events (detected calcium events) - this is what the paper uses." The AI justified this by stating the paper uses detected events, and the code comment says "Neural data: detected calcium events (not raw DFF)."

## 2-b. How is the `neural` data processed?

i. The neural data (events) is filtered to valid ROIs only, then transposed from (n_timepoints, n_cells) to (n_neurons, n_timepoints) per trial. No additional processing (normalization, smoothing, etc.) is applied.

ii.
```python
events = nwb_data['events']  # (n_timepoints, n_cells)
valid = nwb_data['valid_roi']
events = events[:, valid]
# ...
trial_neural = events[frame_indices, :].T  # (n_neurons, n_timepoints)
neural_trials.append(trial_neural.astype(np.float32))
```

iii. The AI relies on the event detection pipeline already applied by the Allen SDK.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using the `valid_roi` flag from the cell specimen table. Only neurons marked as valid ROIs are included.

ii.
```python
valid = nwb_data['valid_roi']
events = events[:, valid]
```

iii. The AI explicitly checks `valid_roi` from the NWB file's cell specimen table. This is an additional quality filter not applied in the reference solution.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start (`trial_start_times`). The ophys frames from `start_time` to `stop_time` (exclusive) are extracted. The frame mask uses `>=` for start and `<` for stop.

ii.
```python
t_start = nwb_data['trial_start_times'][trial_idx]
t_stop = nwb_data['trial_stop_times'][trial_idx]
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]
trial_neural = events[frame_indices, :].T
```

iii. The AI aligns to trial start/stop times using boolean masking on ophys timestamps, giving variable-length trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Data is kept at native ophys frame rate. The time bin size is computed as 1000 / median imaging rate across sessions.

ii.
```python
imaging_rates = [s['imaging_rate_hz'] for s in session_info]
median_rate = np.median(imaging_rates)
time_bin_ms = 1000.0 / median_rate
```

iii. The AI noted (step 34) that different experiments might have different frame rates (~11 Hz for Multiscope vs ~31 Hz for single-plane) but chose to use native timestamps and report the median rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the **stimulus presentations table** (`stim_start_times`, `stim_stop_times`, `stim_image_names`), NOT from the trials table's `initial_image_name`/`change_image_name`.

ii.
```python
stim_starts = nwb_data['stim_start_times']
stim_stops = nwb_data['stim_stop_times']
stim_names = nwb_data['stim_image_names']
stim_omitted = nwb_data['stim_omitted']

for i in range(len(stim_starts)):
    if stim_omitted[i] == 1.0:
        continue
    img_name = stim_names[i]
    if img_name == 'omitted':
        continue
    img_idx = image_name_to_idx[img_name]
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    image_at_ophys[mask] = img_idx
```

iii. The AI built a full session-level image identity time series from stimulus presentations, then sliced per trial. Grey-screen periods (inter-stimulus intervals) are filled with the last shown image via carry-forward logic.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to sorted integer codes via a global mapping. Grey-screen periods (where `image_at_ophys = -1`) are filled using forward-fill logic: the last shown image is carried forward through grey-screen intervals. If the trial starts before the first stimulus, the first valid image is back-filled.

ii.
```python
all_image_names = sorted(all_image_names_set)
image_name_to_idx = {name: idx for idx, name in enumerate(all_image_names)}

# Forward fill grey screen periods
last_img = -1
for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
    elif last_img >= 0:
        trial_image[k] = last_img
if trial_image[0] < 0:
    first_valid = np.where(trial_image >= 0)[0]
    if len(first_valid) > 0:
        trial_image[:first_valid[0]] = trial_image[first_valid[0]]
```

iii. The AI wanted to handle the "image identity of the image presented during the non-grey screen" by identifying which image is being shown at each timepoint using stimulus presentations, with carry-forward for grey periods.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed for the full session at ophys resolution, then sliced using the same `frame_indices` as neural data. This ensures alignment.

ii.
```python
image_at_ophys = np.full(len(ophys_ts), -1, dtype=int)
# ... fill from stimulus presentations ...
trial_image = image_at_ophys[frame_indices]
```

iii. By computing the image identity at each ophys frame first, then using the same frame indices as neural data, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the **stimulus presentations table** `stim_is_change` flag, NOT from the trials table's `change_time` and `go` columns.

ii.
```python
stim_is_change = nwb_data['stim_is_change']
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        is_change_at_ophys[mask] = 1
```

iii. The AI used the `is_change` flag from stimulus presentations to mark change events. This marks only the stimulus presentation period as 1, not a 750ms window.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time series is created at ophys resolution. Timepoints during stimulus presentations where `is_change == True` are set to 1; all others are 0. No 750ms window is used. The change signal is limited to the actual stimulus presentation duration only.

ii.
```python
is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
stim_is_change = nwb_data['stim_is_change']
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        is_change_at_ophys[mask] = 1
```

iii. The AI derived image change from the stimulus presentations rather than the trials table. This approach marks only the stimulus presentation window (250ms) as change, rather than the 750ms window used in the reference.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 or 1). No thresholding is needed. The `stim_is_change` flag directly determines whether each timepoint is a change or not.

ii.
```python
is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
# ...
is_change_at_ophys[mask] = 1
```

iii. The binary nature comes directly from the `is_change` stimulus flag.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity: computed for the full session at ophys resolution, then sliced with the same `frame_indices`.

ii.
```python
trial_change = is_change_at_ophys[frame_indices]
```

iii. Uses the same ophys-aligned indices as all other data streams.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. This corresponds to the SDK's `dataset.running_speed`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the ophys timebase using `np.interp`. Then it is discretized into 5 percentile bins computed globally across all sessions. NaN values are not explicitly handled before interpolation (unlike the reference which uses scipy `interp1d` with `fill_value=np.nan`).

ii.
```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts):
    return np.interp(ophys_ts, signal_ts, signal)

running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'],
    nwb_data['running_timestamps'],
    nwb_data['ophys_timestamps']
)

# Global binning
all_running_flat = np.concatenate(all_running)
running_edges = compute_percentile_bins(all_running_flat, n_bins=5)
running_binned = digitize_to_bins(running, running_edges)
```

iii. The AI uses `np.interp` which extrapolates edge values (constant extrapolation) rather than returning NaN for out-of-range values.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 percentile-based bins using `np.digitize` with globally-computed bin edges. The bins are 0-indexed (0 to 4).

ii.
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    return edges

def digitize_to_bins(values, edges):
    bins = np.digitize(values, edges[1:-1])  # bins: 0 to n_bins-1
    return bins
```

iii. Same percentile-based approach as the reference. NaN handling differs: no explicit NaN-to-0 mapping.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to ophys timestamps before trial segmentation, then sliced with the same frame indices as neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'],
    nwb_data['running_timestamps'],
    nwb_data['ophys_timestamps']
)
# ...
trial_running = running_at_ophys[frame_indices]
```

iii. Same alignment strategy as reference: interpolate to ophys timebase first, then index with trial frames.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width` in the NWB file. Blink frames are identified via `acquisition/EyeTracking/likely_blink/data`.

ii.
```python
data['pupil_width'] = pupil['width'][:]
data['likely_blink'] = f['acquisition/EyeTracking/likely_blink/data'][:]
```

iii. The AI correctly uses `pupil_width` as the measure of pupil diameter, matching the reference.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN, then valid (non-NaN) values are linearly interpolated to the ophys timebase using `np.interp`. The result is discretized into 5 percentile bins globally. Sessions without eye tracking data get a constant fill of bin 2 (median bin).

ii.
```python
pupil_raw = nwb_data['pupil_width'].copy()
blink_mask = nwb_data['likely_blink']
if blink_mask is not None:
    pupil_raw[blink_mask] = np.nan
valid_mask = ~np.isnan(pupil_raw)
if np.sum(valid_mask) > 10:
    pupil_at_ophys = np.interp(
        ophys_ts,
        nwb_data['eye_timestamps'][valid_mask],
        pupil_raw[valid_mask]
    )
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
```

iii. The AI removes blinks before interpolation (like the reference), but uses `np.interp` instead of `scipy.interp1d`, which means out-of-range values are extrapolated rather than set to NaN.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 percentile-based bins computed globally. Sessions without pupil data get a constant fill of bin 2.

ii.
```python
if pupil_edges is not None and not np.all(np.isnan(pupil)):
    pupil_binned = digitize_to_bins(pupil, pupil_edges)
else:
    pupil_binned = np.full(len(pupil), 2, dtype=int)
```

iii. The fallback to bin 2 for missing pupil data is the AI's approach to handling missing data, contrasting with the reference's NaN-to-bin-0 mapping.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed: interpolated to ophys timestamps, then sliced with the same frame indices.

ii.
```python
pupil_at_ophys = np.interp(ophys_ts,
    nwb_data['eye_timestamps'][valid_mask],
    pupil_raw[valid_mask])
# ...
trial_pupil = pupil_at_ophys[frame_indices]
```

iii. Same alignment approach as all other data streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean `hit`, `miss`, `false_alarm`, and `correct_reject` fields in the trials interval of the NWB file.

ii.
```python
data['trial_hit'] = trials['hit'][:]
data['trial_miss'] = trials['miss'][:]
data['trial_false_alarm'] = trials['false_alarm'][:]
data['trial_correct_reject'] = trials['correct_reject'][:]

def get_trial_outcome(trial_idx, nwb_data):
    if nwb_data['trial_hit'][trial_idx]:
        return 0  # hit
    elif nwb_data['trial_miss'][trial_idx]:
        return 1  # miss
    elif nwb_data['trial_false_alarm'][trial_idx]:
        return 2  # false_alarm
    elif nwb_data['trial_correct_reject'][trial_idx]:
        return 3  # correct_reject
    else:
        return -1  # unknown
```

iii. The AI uses the same four outcome categories as the reference.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) in the same order as the reference: hit=0, miss=1, false_alarm=2, correct_reject=3. The outcome is constant across all timepoints within a trial. Trials with unknown outcomes (-1) are skipped.

ii.
```python
outcome = get_trial_outcome(trial_idx, nwb_data)
if outcome == -1:
    continue
# ...
np.full(n_t, outcome, dtype=np.int64)
```

iii. The integer mapping matches the reference's `TRIAL_OUTCOMES` order.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Missing eye tracking**: Sessions without pupil data get a constant fill of bin 2 (median bin).
- **Insufficient valid points for interpolation**: If fewer than 10 valid pupil samples, the session's pupil data is filled with NaN (then bin 2 at discretization).
- **Trials with no stimulus**: If no valid image identity is found in a trial, the trial is skipped.
- **Insufficient trials**: Sessions with fewer than 2 valid trials are skipped.
- **Unknown trial outcomes**: Trials with outcome -1 are skipped.

ii.
```python
# Missing pupil data
if nwb_data['pupil_width'] is None:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
# ...
pupil_binned = np.full(len(pupil), 2, dtype=int)  # fallback

# No stimulus in trial
if len(first_valid) == 0:
    continue  # No stimulus in this trial, skip

# Insufficient trials
if neural_trials is None or len(neural_trials) < 2:
    continue
```

iii. The AI handles missing data gracefully with fallback values and skipping, but the approach differs from the reference (which maps NaN to bin 0 rather than bin 2).

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files twice: once for collecting behavioral statistics (first pass) and once for processing experiments (second pass). Each NWB file is read from disk using h5py.

ii.
```python
# First pass
for idx, (_, row) in enumerate(exp_table.iterrows()):
    nwb_data = load_nwb_data(nwb_path)
    # collect running and pupil

# Second pass
for idx, (_, row) in enumerate(exp_table.iterrows()):
    nwb_data = load_nwb_data(nwb_path)
    result = process_experiment(nwb_data, all_image_names)
```

iii. The two-pass approach doubles the I/O cost compared to a single-pass solution.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-stimulus-presentation loop for building `image_at_ophys` iterates over every stimulus presentation and applies a boolean mask per stimulus. This could be vectorized using `np.searchsorted`. The per-trial forward-fill loop for grey-screen periods is also element-wise.

ii.
```python
# Per-stimulus loop
for i in range(len(stim_starts)):
    if stim_omitted[i] == 1.0:
        continue
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    image_at_ophys[mask] = img_idx

# Per-element forward fill
for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
    elif last_img >= 0:
        trial_image[k] = last_img
```

iii. These loops are O(n_stimuli * n_timepoints) and O(n_trial_frames) respectively. The stimulus loop is particularly expensive since it creates a boolean mask over all ophys timepoints for each stimulus.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is loaded twice: once in the first pass (collecting running/pupil statistics for global binning) and once in the second pass (processing experiments). This includes re-reading all the data from disk and re-interpolating running speed and pupil to ophys timestamps.

ii.
```python
# First pass: collect statistics
for idx, (_, row) in enumerate(exp_table.iterrows()):
    nwb_data = load_nwb_data(nwb_path)
    running = interpolate_to_ophys(...)
    pupil = np.interp(...)

# Second pass: process experiments
for idx, (_, row) in enumerate(exp_table.iterrows()):
    nwb_data = load_nwb_data(nwb_path)
    result = process_experiment(nwb_data, all_image_names)  # re-interpolates
```

iii. The two-pass design loads every NWB file twice and re-computes the interpolation. The reference avoids this by extracting trial data in a single pass and computing bin edges from already-extracted data.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI also loads dF/F traces from NWB files (`f['processing/ophys/dff/traces/data'][:]`) but never uses them for neural data (uses events instead). It also collects image names in a separate preliminary pass through all NWB files, reading them a third time just for image name collection.

ii.
```python
# In load_nwb_data:
dff = f['processing/ophys/dff/traces/data'][:]
data['dff'] = dff  # loaded but never used for neural data

# Image name collection pass (separate from the two main passes)
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        names = f[f'intervals/{stim_keys[0]}/image_name'][:]
```

iii. Loading dF/F traces adds unnecessary memory and I/O overhead. The image name collection could be folded into the first pass.
