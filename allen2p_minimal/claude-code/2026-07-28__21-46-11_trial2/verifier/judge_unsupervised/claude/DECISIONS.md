# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files from `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/` using h5py. It first loads an experiment metadata table (`ophys_experiment_table.csv`), filters for experiments that have corresponding NWB files and are active behavior sessions (`passive == False`), then iterates over all matching experiments. For each experiment, it opens the NWB file and extracts ophys timestamps, calcium events, DFF traces, cell specimen table, stimulus presentations, trial intervals, running speed, pupil tracking, and imaging rate. It makes two passes: a first pass to collect running and pupil statistics for global percentile binning, and a second pass to do the full processing.

ii. ```python
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
        # ... loads trials, stimuli, running, pupil, etc.
    return data
```

iii. The AI's CONVERSION_NOTES.md states: "284 NWB files in behavior_ophys_experiments/", "Active behavior only: Filtered passive == False from experiment table (202 of 284 NWB files)". The trajectory shows the agent explored the NWB file structure with h5py before writing the conversion script.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `mouse_id` column in the experiment metadata table. All unique mouse IDs are sorted and used as subject identifiers. A `subject_to_idx` mapping tracks which subject each session belongs to.

ii. ```python
all_subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
# ...
subject_idx_list.append(subject_to_idx[mouse_id])
```

iii. CONVERSION_NOTES: "202 sessions from 38 mice". This is a straightforward split derived from metadata.

## 1-c. How are the data split into sessions?

i. Each ophys experiment (each NWB file / imaging plane) is treated as a separate session. This means for Multiscope mice with multiple imaging planes recorded simultaneously, each plane becomes its own session even though they share the same behavioral data and trials. The output shows 202 sessions from 174 unique ophys_session_ids, meaning some physical recording sessions are split into multiple "sessions" in the output.

ii. ```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    # ...
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
    nwb_data = load_nwb_data(nwb_path)
    result = process_experiment(nwb_data, all_image_names)
    # Each experiment becomes one session in neural_all, output_all, etc.
    neural_all.append(final_neural)
```

iii. The trajectory (step 31 reasoning) shows the agent explicitly considered this: "Each NWB file = one ophys experiment (one imaging plane)" and "Each experiment is a 'session' for our purposes". The agent noted there are 174 unique ophys_session_ids but 202 experiments.

## 1-d. How are the data split into trials?

i. Trials are segmented using `start_time` and `stop_time` from the NWB `intervals/trials` table. For each trial, ophys frames within `[start_time, stop_time)` are extracted. Trials with fewer than 2 ophys frames are excluded.

ii. ```python
t_start = nwb_data['trial_start_times'][trial_idx]
t_stop = nwb_data['trial_stop_times'][trial_idx]
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]
if len(frame_indices) < 2:
    continue
trial_neural = events[frame_indices, :].T
```

iii. CONVERSION_NOTES: "Trials segmented using start_time and stop_time from NWB trials table. Ophys frames within [start_time, stop_time) extracted for each trial. Trials with < 2 ophys frames are excluded."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) excluding aborted trials, (2) excluding auto-rewarded trials, (3) only including trials marked as Go or Catch, (4) excluding trials where the outcome is unknown (not hit/miss/false_alarm/correct_reject), (5) excluding trials with fewer than 2 ophys frames, and (6) excluding trials where no stimulus was presented.

ii. ```python
def get_trial_mask(trial_idx, nwb_data):
    if nwb_data['trial_aborted'][trial_idx]:
        return False
    if nwb_data['trial_auto_rewarded'][trial_idx]:
        return False
    if not (nwb_data['trial_go'][trial_idx] or nwb_data['trial_catch'][trial_idx]):
        return False
    return True
```

iii. CONVERSION_NOTES: "Included: Go trials and Catch trials. Excluded: Aborted trials (mouse licked before change) and Auto-rewarded trials (free rewards). This matches the task specification and standard analysis practice from the reference papers."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the detected calcium events stored at `processing/ophys/event_detection/data` in the NWB files. This represents discrete calcium events that were regressed from the raw fluorescence traces, not the raw dF/F traces.

ii. ```python
events = f['processing/ophys/event_detection/data'][:]
data['events'] = events  # shape: (n_timepoints, n_cells)
```

iii. CONVERSION_NOTES: "Used detected calcium events (processing/ophys/event_detection/data) rather than raw dF/F traces. This matches the paper: 'We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces'."

## 2-b. How is the `neural` data processed?

i. The raw event detection data is filtered to include only valid ROIs (using the `valid_roi` flag from the cell specimen table). The data is then transposed from (n_timepoints, n_cells) to (n_cells, n_timepoints) format and cast to float32. No additional normalization, smoothing, or temporal rebinning is applied.

ii. ```python
valid = nwb_data['valid_roi']
events = events[:, valid]
# ...
trial_neural = events[frame_indices, :].T  # (n_neurons, n_timepoints)
neural_trials.append(trial_neural.astype(np.float32))
```

iii. The trajectory (step 31) notes the paper "regressed events from the traces to remove the slow decay dynamics of GCaMP6f", confirming events are already processed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered by the `valid_roi` flag from the cell specimen table. Only neurons with `valid_roi == True` are included. Sessions with zero valid neurons are skipped entirely. Sessions with fewer than 2 valid trials are also skipped.

ii. ```python
valid = nwb_data['valid_roi']
events = events[:, valid]
n_timepoints_total, n_neurons = events.shape
if n_neurons == 0:
    return None, None, None, None, None, None
```

iii. CONVERSION_NOTES: "Only valid ROIs included (filtered by valid_roi flag in cell specimen table)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the ophys imaging frame timestamps. For each trial, all ophys frames within the trial's `[start_time, stop_time)` interval are extracted. There is no alignment to a specific within-trial event (like the change time); the alignment is simply to the trial boundaries as defined in the NWB trials table.

ii. ```python
ophys_ts = nwb_data['ophys_timestamps']
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]
trial_neural = events[frame_indices, :].T
```

iii. CONVERSION_NOTES: "All data aligned to ophys imaging frame timestamps. Ophys timestamps serve as the common temporal reference." The metadata sets `temporal_alignment_event: 'Ophys imaging frame timestamps'` and `off_start: None, off_end: None`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The native ophys frame rate is used as-is, which differs across experiments: ~31 Hz for single-plane (VisualBehavior) and ~11 Hz for multi-plane (VisualBehaviorMultiscope). The reported `time_bin_size` in metadata is the median across sessions (32.26 ms, corresponding to 31 Hz). This means sessions have inconsistent temporal resolutions.

ii. ```python
imaging_rates = [s['imaging_rate_hz'] for s in session_info]
median_rate = np.median(imaging_rates)
time_bin_ms = 1000.0 / median_rate
```

iii. CONVERSION_NOTES: "Time bin size is the inter-frame interval (~32.26 ms for single-plane at ~31 Hz, ~93.2 ms for multi-plane at ~11 Hz). Metadata time_bin_size reports median across sessions (32.26 ms)." The trajectory (step 34) shows the agent deliberated on this issue but decided to keep native timestamps.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table in the NWB file, specifically the `image_name` field from the stimulus intervals (e.g., `intervals/Natural_Images_Lum_Matched_set_training_2017_presentations`), along with `start_time`, `stop_time`, and `omitted` fields.

ii. ```python
stim_starts = nwb_data['stim_start_times']
stim_stops = nwb_data['stim_stop_times']
stim_names = nwb_data['stim_image_names']
stim_omitted = nwb_data['stim_omitted']
```

iii. CONVERSION_NOTES: "For each ophys timepoint, the currently displayed image is determined from stimulus presentation intervals."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image identity is computed as a time-varying signal at ophys resolution. For each stimulus presentation (excluding omitted ones), the image name is mapped to an integer index (0-15, from sorted unique image names across all experiments). During gray screen periods (inter-stimulus intervals), the last shown image identity is carried forward. If a trial begins before the first stimulus, the first valid image is back-filled. Trials with no stimulus at all are skipped.

ii. ```python
image_at_ophys = np.full(len(ophys_ts), -1, dtype=int)
for i in range(len(stim_starts)):
    if stim_omitted[i] == 1.0:
        continue
    img_name = stim_names[i]
    if img_name == 'omitted':
        continue
    img_idx = image_name_to_idx[img_name]
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    image_at_ophys[mask] = img_idx

# Carry forward last image during gray screen
last_img = -1
for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
    elif last_img >= 0:
        trial_image[k] = last_img
```

iii. CONVERSION_NOTES: "During gray screen periods, the last shown image identity is carried forward. Omitted stimuli are treated as gray screen."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is directly mapped to ophys timestamps. For each ophys frame, the code checks which stimulus presentation interval contains that timestamp, assigning the corresponding image index. Since both neural data and image identity use the same ophys timestamp indices, they are inherently aligned.

ii. ```python
mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
image_at_ophys[mask] = img_idx
# Then for each trial:
trial_image = image_at_ophys[frame_indices]
```

iii. Both neural data and image identity use the same `frame_indices` into the ophys timestamps array.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table of the NWB file.

ii. ```python
data['stim_is_change'] = stim['is_change'][:]
```

iii. CONVERSION_NOTES: "Binary signal (0/1) marking timepoints during which a stimulus change occurred. Derived from is_change field in stimulus presentations table."

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time series is constructed at ophys resolution. For each stimulus presentation where `is_change == 1.0`, all ophys frames during that presentation interval are set to 1; all other frames are 0.

ii. ```python
is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
stim_is_change = nwb_data['stim_is_change']
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        is_change_at_ophys[mask] = 1
```

iii. CONVERSION_NOTES: "Value is 1 during the change stimulus presentation, 0 otherwise."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), derived directly from the `is_change` flag. No thresholding is needed. Output values are labeled `['no_change', 'change']`.

ii. ```python
['no_change', 'change'],  # image_change values
```

iii. The instructions specify it as a binary variable with value 1 right after a change in image identity, otherwise 0.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same alignment as image identity - mapped to ophys timestamps and extracted using the same frame indices for each trial.

ii. ```python
trial_change = is_change_at_ophys[frame_indices]
```

iii. Uses the same ophys timestamp-based alignment as all other signals.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii. ```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. CONVERSION_NOTES: "Running speed and pupil diameter are recorded at different rates than ophys."

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to ophys timestamps using `np.interp`. A two-pass approach is used: first pass collects all running speed values (interpolated to ophys timestamps) across all sessions; global percentile bin edges are computed; then in the second pass, each trial's running speed values are digitized into 5 bins.

ii. ```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'],
    nwb_data['running_timestamps'],
    nwb_data['ophys_timestamps']
)
# ...
running_edges = compute_percentile_bins(all_running_flat, n_bins=5)
running_binned = digitize_to_bins(running, running_edges)
```

iii. CONVERSION_NOTES: "Global percentile bins computed across ALL sessions' running speed data (interpolated to ophys timestamps). 5 equal percentile bins (0-20%, 20-40%, 40-60%, 60-80%, 80-100%)."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins (quintiles) computed globally across all sessions. The bin edges are computed using `np.percentile` at [0, 20, 40, 60, 80, 100] percentiles. Values are digitized using `np.digitize` with the inner edges, yielding bins 0-4.

ii. ```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    return edges

def digitize_to_bins(values, edges):
    bins = np.digitize(values, edges[1:-1])  # bins: 0 to n_bins-1
    return bins
```

iii. CONVERSION_NOTES: "Bin edges: [-23.98, 0.017, 1.83, 19.02, 35.06, 99.92] cm/s."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is linearly interpolated to ophys timestamps, then the per-trial values are extracted using the same frame indices as neural data.

ii. ```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
trial_running = running_at_ophys[frame_indices]
```

iii. Same ophys-timestamp alignment as neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the `width` field of `acquisition/EyeTracking/pupil_tracking` in the NWB file. The `likely_blink` data from `acquisition/EyeTracking/likely_blink/data` is used to mask blink periods. Eye timestamps come from `acquisition/EyeTracking/eye_tracking/timestamps`.

ii. ```python
data['pupil_width'] = pupil['width'][:]
data['pupil_area'] = pupil['area'][:]
eye_ts = f['acquisition/EyeTracking/eye_tracking/timestamps'][:]
data['eye_timestamps'] = eye_ts
data['likely_blink'] = f['acquisition/EyeTracking/likely_blink/data'][:]
```

iii. CONVERSION_NOTES: "Pupil width from fitted ellipse used as diameter measure."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink periods (from `likely_blink`) are set to NaN. The remaining valid (non-NaN) pupil width values are linearly interpolated to ophys timestamps. If fewer than 10 valid pupil samples exist, the entire session gets NaN pupil values. Global percentile bin edges are computed from all valid pupil data across sessions, and per-trial values are digitized into 5 bins. Sessions with no usable pupil data are assigned the median bin (bin 2).

ii. ```python
pupil_raw = nwb_data['pupil_width'].copy()
blink_mask = nwb_data['likely_blink']
if blink_mask is not None:
    pupil_raw[blink_mask] = np.nan
valid_mask = ~np.isnan(pupil_raw)
if np.sum(valid_mask) > 10:
    pupil_at_ophys = np.interp(ophys_ts, nwb_data['eye_timestamps'][valid_mask], pupil_raw[valid_mask])
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
# ...
if pupil_edges is not None and not np.all(np.isnan(pupil)):
    pupil_binned = digitize_to_bins(pupil, pupil_edges)
else:
    pupil_binned = np.full(len(pupil), 2, dtype=int)  # median bin
```

iii. CONVERSION_NOTES: "Blink periods excluded (set to NaN before interpolation). Sessions without pupil data: assigned median bin (bin 2)."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal percentile bins computed globally across all sessions with valid pupil data, using `np.percentile` at [0, 20, 40, 60, 80, 100].

ii. ```python
pupil_edges = compute_percentile_bins(all_pupil_flat, n_bins=5)
pupil_binned = digitize_to_bins(pupil, pupil_edges)
```

iii. CONVERSION_NOTES: "Bin edges: [4.81, 36.81, 41.44, 46.17, 52.78, 252.21] pixels."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated from eye tracking timestamps to ophys timestamps, then extracted per-trial using the same frame indices as neural data.

ii. ```python
pupil_at_ophys = np.interp(ophys_ts, nwb_data['eye_timestamps'][valid_mask], pupil_raw[valid_mask])
trial_pupil = pupil_at_ophys[frame_indices]
```

iii. Same ophys-timestamp alignment as neural and other behavioral data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean fields in the NWB `intervals/trials` table.

ii. ```python
data['trial_hit'] = trials['hit'][:]
data['trial_miss'] = trials['miss'][:]
data['trial_false_alarm'] = trials['false_alarm'][:]
data['trial_correct_reject'] = trials['correct_reject'][:]
```

iii. CONVERSION_NOTES: "Determined from trial flags: hit, miss, false_alarm, correct_reject."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The trial outcome is determined by checking the boolean flags in priority order: hit (0), miss (1), false_alarm (2), correct_reject (3). If none are set, the outcome is -1 (unknown) and the trial is excluded. The outcome is stored as a static value per trial but replicated across all timepoints in the output array (row 4 is filled with the outcome value for all timepoints).

ii. ```python
def get_trial_outcome(trial_idx, nwb_data):
    if nwb_data['trial_hit'][trial_idx]:
        return 0
    elif nwb_data['trial_miss'][trial_idx]:
        return 1
    elif nwb_data['trial_false_alarm'][trial_idx]:
        return 2
    elif nwb_data['trial_correct_reject'][trial_idx]:
        return 3
    else:
        return -1

# In output array:
np.full(n_t, outcome, dtype=np.int64),  # replicated across timepoints
```

iii. CONVERSION_NOTES: "Static per trial (replicated across all timepoints for time-varying format)."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is handled in several ways:
- **Missing pupil data**: Sessions without eye tracking are assigned median bin (bin 2). Blink periods are set to NaN before interpolation; valid samples are interpolated. Sessions with < 10 valid pupil samples get NaN then default bin.
- **Missing stimulus data**: Trials with no stimulus presentations are skipped entirely.
- **Omitted stimuli**: Treated as gray screen; last shown image identity is carried forward.
- **Unknown trial outcomes**: Trials where none of hit/miss/false_alarm/correct_reject is true are excluded.
- **Short trials**: Trials with < 2 ophys frames are excluded.
- **Empty sessions**: Sessions with 0 valid neurons or < 2 valid trials are skipped.

ii. ```python
# Pupil NaN handling
if nwb_data['pupil_width'] is None:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
# Missing pupil -> median bin
pupil_binned = np.full(len(pupil), 2, dtype=int)

# No stimulus in trial
if len(first_valid) == 0:
    continue  # skip trial
```

iii. CONVERSION_NOTES lists known limitations: "Pupil data quality varies across sessions; some sessions have no usable pupil data." "Some trials have all-zero neural data (sparse calcium events)."

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** - done twice (first pass for statistics, second for processing), each involving reading large HDF5 datasets.
2. **Image name collection** - a separate loop opens every NWB file just to collect image names.
3. **Building the image identity time series** - iterates over all stimulus presentations with per-stimulus masking of ophys timestamps.
4. **Forward-filling image identity** - per-element loop in Python for each trial.

ii. ```python
# First pass: loads all NWB files
for idx, (_, row) in enumerate(exp_table.iterrows()):
    nwb_data = load_nwb_data(nwb_path)  # loads EVERYTHING
    running = interpolate_to_ophys(...)

# Image name collection: opens every NWB file again
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        names = f[f'intervals/{stim_keys[0]}/image_name'][:]
```

iii. The code opens each NWB file 3 times total: once for image names, once for statistics, once for processing.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. **Image identity assignment** (line 237-248): The per-stimulus loop with boolean masking could use `np.searchsorted` on sorted timestamps.
2. **Image change assignment** (line 253-256): Similar per-stimulus loop with masking.
3. **Forward-fill of image identity** (line 294-303): The per-element Python loop could use `pd.Series.ffill()` or a vectorized approach.
4. **Stimulus key discovery** (line 91-94): Minor but repeated for each NWB file.

ii. ```python
# Per-stimulus loop (could be vectorized)
for i in range(len(stim_starts)):
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    image_at_ophys[mask] = img_idx

# Per-element forward-fill (could use pandas ffill)
for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
    elif last_img >= 0:
        trial_image[k] = last_img
```

iii. No explicit justification from the agent for these choices; the code follows a straightforward imperative style.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats several operations:
1. **Loading NWB files**: Each NWB file is opened 3 times - once for image name collection, once for first-pass statistics, once for second-pass processing.
2. **Running speed interpolation**: Computed in both the first pass (for global statistics) and second pass (in `process_experiment`).
3. **Pupil diameter processing**: Blink masking + interpolation done in both passes.
4. **DFF loading**: DFF traces are loaded (`f['processing/ophys/dff/traces/data'][:]`) but never used in processing.

ii. ```python
# Image name collection (1st open)
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f: ...

# First pass statistics (2nd open)
for idx, (_, row) in enumerate(exp_table.iterrows()):
    nwb_data = load_nwb_data(nwb_path)
    running = interpolate_to_ophys(...)

# Second pass processing (3rd open)
for idx, (_, row) in enumerate(exp_table.iterrows()):
    nwb_data = load_nwb_data(nwb_path)
    result = process_experiment(nwb_data, all_image_names)
    # process_experiment also calls interpolate_to_ophys for running
```

iii. No justification given. The two-pass approach for global percentile binning necessitates loading data twice, but the image name collection pass could have been merged with the first pass.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of data are loaded but not used in the final output:
1. **DFF traces**: Loaded in `load_nwb_data` (`dff = f['processing/ophys/dff/traces/data'][:]`) but never used in processing.
2. **Pupil area**: Loaded (`data['pupil_area'] = pupil['area'][:]`) but not used (only pupil width is used).
3. **Cell specimen IDs**: Loaded but not used in output.
4. **Events timestamps**: Loaded separately but never used (ophys timestamps from DFF are used instead).
5. **Full events array returned from process_experiment**: The `events_full` return value is captured but never used.

ii. ```python
dff = f['processing/ophys/dff/traces/data'][:]  # Never used
data['dff'] = dff

data['pupil_area'] = pupil['area'][:]  # Never used

data['cell_specimen_ids'] = cst['cell_specimen_id'][:]  # Never used

# events_full returned but discarded
neural_trials, raw_output_trials, trial_outcomes, running_at_ophys, pupil_at_ophys, events_full = result
```

iii. No justification. The comment "DFF traces (for fallback / comparison)" suggests loading was intentional for possible debugging but wasteful for production.
