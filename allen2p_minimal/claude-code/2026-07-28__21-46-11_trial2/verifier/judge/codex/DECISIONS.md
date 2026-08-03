# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the Allen SDK cache. It loaded the metadata CSV directly, enumerated NWB files present on disk, filtered the experiment table down to rows with available NWB files and `passive == False`, and then opened each NWB with `h5py`. It therefore loaded data experiment-by-experiment from raw files rather than session-by-session through the SDK.

ii. ```python
def load_experiment_table():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
    nwb_ids = [int(f.split('_')[-1].split('.')[0]) for f in nwb_files]
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(nwb_ids)]
    exp_table = exp_table[exp_table['passive'] == False]
    return exp_table

with h5py.File(nwb_path, 'r') as f:
    ...
```

iii. `CONVERSION_NOTES.md` says the dataset source is the local NWB release and that the AI included all active behavior experiments. In trajectory step 31, the AI explicitly said "Each NWB file = one ophys experiment" and that it would load data directly from NWB files.

## 1-b. How are the data split into subjects?

i. Subjects are defined as unique `mouse_id` values from the experiment metadata table. The final `subjects` list is a sorted string version of the unique mouse IDs, and each retained experiment/session gets a `subject_idx` from that mapping.

ii. ```python
all_subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
mouse_id = str(row['mouse_id'])
subject_idx_list.append(subject_to_idx[mouse_id])
```

iii. This choice is implicit in the code and consistent with `CONVERSION_NOTES.md`, which reports dataset statistics in terms of unique mice.

## 1-c. How are the data split into sessions?

i. The AI treated each `ophys_experiment_id` as a separate output session. It did not group experiments by shared `ophys_session_id`, so multi-plane sessions were split into multiple independent sessions.

ii. ```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
    nwb_data = load_nwb_data(nwb_path)
```

iii. `CONVERSION_NOTES.md` states "Each experiment = 1 session in the output." Trajectory step 31 also says "Each experiment is a 'session' for our purposes."

## 1-d. How are the data split into trials?

i. Trials are segmented from the NWB `intervals/trials` table. For each included trial, the AI used the full variable-length window from `start_time` to `stop_time` and took all ophys frames whose timestamps fall within `[start_time, stop_time)`.

ii. ```python
data['trial_start_times'] = trials['start_time'][:]
data['trial_stop_times'] = trials['stop_time'][:]
...
for trial_idx in range(n_trials_total):
    if not get_trial_mask(trial_idx, nwb_data):
        continue
    t_start = nwb_data['trial_start_times'][trial_idx]
    t_stop = nwb_data['trial_stop_times'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    frame_indices = np.where(frame_mask)[0]
```

iii. `CONVERSION_NOTES.md` says trials were segmented using `start_time` and `stop_time` from the NWB trials table. Trajectory step 31 describes trials as spanning from `start_time` to `stop_time` with variable duration.

## 1-e. How are trials filtered based on quality controls?

i. The AI excluded aborted trials, auto-rewarded trials, and any trial that was neither go nor catch. It also dropped trials with unknown outcome labels, trials with fewer than 2 ophys frames, and experiments with fewer than 2 valid trials.

ii. ```python
def get_trial_mask(trial_idx, nwb_data):
    if nwb_data['trial_aborted'][trial_idx]:
        return False
    if nwb_data['trial_auto_rewarded'][trial_idx]:
        return False
    if not (nwb_data['trial_go'][trial_idx] or nwb_data['trial_catch'][trial_idx]):
        return False
    return True

outcome = get_trial_outcome(trial_idx, nwb_data)
if outcome == -1:
    continue
if len(frame_indices) < 2:
    continue
...
if neural_trials is None or len(neural_trials) < 2:
    continue
```

iii. `CONVERSION_NOTES.md` explicitly says go and catch were included while aborted and auto-rewarded were excluded. The notes also mention excluding trials with fewer than 2 frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derived `neural` from the NWB event-detection output, `processing/ophys/event_detection/data`, not from dF/F traces.

ii. ```python
events = f['processing/ophys/event_detection/data'][:]
events_ts = f['processing/ophys/event_detection/timestamps'][:]
data['events'] = events
...
events = nwb_data['events']
```

iii. The top-level docstring, `README.md`, `CONVERSION_NOTES.md`, and trajectory step 31 all say the AI deliberately chose detected calcium events because it believed that matched the paper better than raw fluorescence or dF/F.

## 2-b. How is the `neural` data processed?

i. The AI filtered neurons to `valid_roi == True`, kept the event matrix at the native experiment timebase, and transposed each trial slice to shape `(n_neurons, n_timepoints)`. Because it treated each experiment as a session, it did not merge multiple planes within a session.

ii. ```python
events = nwb_data['events']  # (n_timepoints, n_cells)
valid = nwb_data['valid_roi']
events = events[:, valid]
...
trial_neural = events[frame_indices, :].T
neural_trials.append(trial_neural.astype(np.float32))
```

iii. `CONVERSION_NOTES.md` says the AI used detected calcium events and only valid ROIs. Trajectory steps 31 and 92 say the AI believed filtering by `valid_roi` mirrored SDK defaults.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applied one explicit neural quality-control filter: it removed ROIs whose `valid_roi` flag was false. Experiments with zero surviving neurons were skipped.

ii. ```python
valid = nwb_data['valid_roi']
events = events[:, valid]
...
if n_neurons == 0:
    return None, None, None, None, None, None
...
if n_neurons < 1:
    continue
```

iii. `CONVERSION_NOTES.md` says "Only valid ROIs included," and trajectory step 92 states that the AI considered this equivalent to the SDK's default invalid-ROI exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligned all streams to the ophys timestamp axis and then extracted per-trial neural data by selecting the ophys frames that fall between each trial's `start_time` and `stop_time`. In practice this means each trial starts at the first included ophys frame in the trial window.

ii. ```python
ophys_ts = nwb_data['ophys_timestamps']
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]
trial_neural = events[frame_indices, :].T
```

iii. `CONVERSION_NOTES.md` says all data are aligned to ophys timestamps and that the ophys timestamps are the common temporal reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI did not rebin neural data. It kept experiment-native ophys frames, which means single-plane and multiscope experiments remained at their own native frame rates. For metadata, it reported a single `time_bin_size` computed from the median imaging rate across retained sessions.

ii. ```python
data['imaging_rate'] = f['general/optophysiology/imaging_plane_1/imaging_rate'][()]
...
imaging_rates = [s['imaging_rate_hz'] for s in session_info]
median_rate = np.median(imaging_rates)
time_bin_ms = 1000.0 / median_rate
...
'time_bin_size': time_bin_ms,
```

iii. `CONVERSION_NOTES.md` says time bin size is the inter-frame interval and highlights the mixture of ~31 Hz single-plane and ~11 Hz multiscope data, while also saying the metadata reports the median value.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The AI derived image identity from the stimulus-presentation table rather than trial-level image-name fields. It used stimulus `start_time`, `stop_time`, `image_name`, and `omitted` to mark which image was on screen at each ophys frame.

ii. ```python
data['stim_start_times'] = stim['start_time'][:]
data['stim_stop_times'] = stim['stop_time'][:]
data['stim_image_names'] = ...
data['stim_omitted'] = stim['omitted'][:]
...
stim_starts = nwb_data['stim_start_times']
stim_stops = nwb_data['stim_stop_times']
stim_names = nwb_data['stim_image_names']
stim_omitted = nwb_data['stim_omitted']
```

iii. `CONVERSION_NOTES.md` says the AI determined the currently displayed image from stimulus presentation intervals and treated omitted stimuli as gray screen.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI built a full-session time series `image_at_ophys` by painting stimulus intervals onto the ophys timeline, skipped omitted stimuli, mapped image names to globally sorted integer codes, then within each trial forward-filled gray-screen gaps with the last seen image and back-filled the initial gray gap with the first valid image if needed.

ii. ```python
image_at_ophys = np.full(len(ophys_ts), -1, dtype=int)
image_name_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
for i in range(len(stim_starts)):
    if stim_omitted[i] == 1.0:
        continue
    ...
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    image_at_ophys[mask] = img_idx

trial_image = image_at_ophys[frame_indices]
last_img = -1
for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
    elif last_img >= 0:
        trial_image[k] = last_img
```

iii. `CONVERSION_NOTES.md` explicitly says the current image was determined from stimulus intervals and then carried forward across gray-screen periods because the desired output was the identity of the non-gray image.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is first constructed at ophys-frame resolution for the whole experiment and then sliced using the exact same `frame_indices` used for the neural matrix, so its time axis matches the neural data exactly within each trial.

ii. ```python
mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
image_at_ophys[mask] = img_idx
...
frame_indices = np.where(frame_mask)[0]
trial_neural = events[frame_indices, :].T
trial_image = image_at_ophys[frame_indices]
```

iii. `CONVERSION_NOTES.md` says all streams are aligned to ophys timestamps, and this code implements that alignment directly.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The AI derived image change from the stimulus-presentation `is_change` flag rather than from trial-level `change_time` or `go`.

ii. ```python
data['stim_is_change'] = stim['is_change'][:]
...
is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
stim_is_change = nwb_data['stim_is_change']
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        is_change_at_ophys[mask] = 1
```

iii. `CONVERSION_NOTES.md` says image change was derived from the stimulus presentations table and marks timepoints during which a change stimulus occurred.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI created a binary full-session time series on the ophys timebase and set it to 1 during stimulus intervals whose `is_change` flag was true. It did not extend the label into the post-stimulus gray period.

ii. ```python
is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        is_change_at_ophys[mask] = 1
...
trial_change = is_change_at_ophys[frame_indices]
```

iii. `CONVERSION_NOTES.md` says the change signal is 1 "during the change stimulus presentation" and 0 otherwise.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No additional thresholding was applied beyond making it binary. Frames are either `0` (no change) or `1` (change stimulus interval).

ii. ```python
is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
...
if stim_is_change[i] == 1.0:
    ...
    is_change_at_ophys[mask] = 1
```

iii. The justification in `CONVERSION_NOTES.md` treats image change as an already-binary variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is built on the ophys timeline first and then sliced per trial with the same `frame_indices` as the neural data.

ii. ```python
frame_indices = np.where(frame_mask)[0]
trial_neural = events[frame_indices, :].T
trial_change = is_change_at_ophys[frame_indices]
```

iii. `CONVERSION_NOTES.md` says all streams use ophys timestamps as the common temporal reference.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the NWB running-speed data stream: `processing/running/speed/data` and its timestamps.

ii. ```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. `CONVERSION_NOTES.md` describes running speed as wheel-encoder data sampled separately and then aligned to ophys.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolated running speed to the ophys timestamps with `np.interp`, collected all interpolated values across experiments to compute global percentile edges, then digitized each trial's values into 5 bins.

ii. ```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts):
    return np.interp(ophys_ts, signal_ts, signal)

running = interpolate_to_ophys(
    nwb_data['running_speed'],
    nwb_data['running_timestamps'],
    nwb_data['ophys_timestamps']
)
...
running_edges = compute_percentile_bins(all_running_flat, n_bins=5)
...
running_binned = digitize_to_bins(running, running_edges)
```

iii. `CONVERSION_NOTES.md` says running speed was linearly interpolated to ophys time and then discretized into five equal percentile bins computed globally across the dataset.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded by percentile binning into five global categories. The code uses `np.percentile` to define six bin edges and `np.digitize` to assign bins `0` through `4`.

ii. ```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    return edges

def digitize_to_bins(values, edges):
    bins = np.digitize(values, edges[1:-1])
    return bins
```

iii. `CONVERSION_NOTES.md` gives explicit percentile-bin edges and says the intent was five equal-percentile bins over the full dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto `ophys_timestamps` before trial segmentation, then trial values are taken with the same `frame_indices` used for neural data.

ii. ```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'],
    nwb_data['running_timestamps'],
    ophys_ts
)
...
trial_neural = events[frame_indices, :].T
trial_running = running_at_ophys[frame_indices]
```

iii. `CONVERSION_NOTES.md` explicitly says all streams were aligned to ophys timestamps.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking `pupil_tracking/width` values, along with eye-tracking timestamps and the `likely_blink` mask.

ii. ```python
data['pupil_width'] = pupil['width'][:]
data['eye_timestamps'] = eye_ts
data['likely_blink'] = f['acquisition/EyeTracking/likely_blink/data'][:]
```

iii. `CONVERSION_NOTES.md` says the AI used pupil width from the fitted ellipse as the diameter measure and excluded blink periods.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI copied the pupil-width signal, set likely-blink samples to `NaN`, interpolated across remaining valid samples onto the ophys timebase with `np.interp`, pooled those interpolated values across experiments to compute global percentile edges, and later digitized each trial into bins.

ii. ```python
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
...
pupil_edges = compute_percentile_bins(all_pupil_flat, n_bins=5)
```

iii. `CONVERSION_NOTES.md` says blinks were excluded before interpolation and that global percentile bins were used.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five global percentile bins using `np.percentile` plus `np.digitize`. If a trial/session has no usable pupil data at binning time, the AI assigns every frame to the median bin (`2`) instead of using a missing-value category.

ii. ```python
if pupil_edges is not None and not np.all(np.isnan(pupil)):
    pupil_binned = digitize_to_bins(pupil, pupil_edges)
else:
    pupil_binned = np.full(len(pupil), 2, dtype=int)
```

iii. `CONVERSION_NOTES.md` explicitly says sessions without pupil data were assigned the median bin.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is resampled to `ophys_timestamps` first, and each trial's pupil sequence is then selected with the same `frame_indices` used for the neural data.

ii. ```python
pupil_at_ophys = np.interp(
    ophys_ts,
    nwb_data['eye_timestamps'][valid_mask],
    pupil_raw[valid_mask]
)
...
trial_neural = events[frame_indices, :].T
trial_pupil = pupil_at_ophys[frame_indices]
```

iii. `CONVERSION_NOTES.md` says all data streams were aligned to the ophys frame timestamps.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial flags `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
data['trial_hit'] = trials['hit'][:]
data['trial_miss'] = trials['miss'][:]
data['trial_false_alarm'] = trials['false_alarm'][:]
data['trial_correct_reject'] = trials['correct_reject'][:]
...
if nwb_data['trial_hit'][trial_idx]:
    return 0
elif nwb_data['trial_miss'][trial_idx]:
    return 1
```

iii. `CONVERSION_NOTES.md` says trial outcome was determined from these trial flags.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI mapped the four mutually exclusive outcome flags to integer codes `0..3` and then repeated the chosen code across every time bin in the trial when constructing the final output tensor.

ii. ```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
output_combined = np.stack([
    img_id.astype(np.int64),
    img_change.astype(np.int64),
    running_binned.astype(np.int64),
    pupil_binned.astype(np.int64),
    np.full(n_t, outcome, dtype=np.int64),
], axis=0)
```

iii. `CONVERSION_NOTES.md` says trial outcome is static per trial and is replicated across all timepoints for the time-varying output format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handled several edge cases ad hoc: invalid ROIs were removed; experiments with zero neurons or fewer than 2 valid trials were skipped; trials with unknown outcome, no stimulus in the trial, or fewer than 2 frames were skipped; blink frames were set to `NaN` before pupil interpolation; and trials with unusable pupil data were assigned the median pupil bin.

ii. ```python
valid = nwb_data['valid_roi']
...
if outcome == -1:
    continue
if len(frame_indices) < 2:
    continue
...
if len(first_valid) > 0:
    trial_image[:first_valid[0]] = trial_image[first_valid[0]]
else:
    continue
...
if pupil_edges is not None and not np.all(np.isnan(pupil)):
    pupil_binned = digitize_to_bins(pupil, pupil_edges)
else:
    pupil_binned = np.full(len(pupil), 2, dtype=int)
```

iii. The justifications come mostly from `CONVERSION_NOTES.md`: blink masking, median-bin fallback for missing pupil, trial count minimum, and use of valid ROIs are all documented there.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming parts are repeated NWB I/O and repeated full-session processing. The code scans all experiments once to collect image names, again to collect running and pupil values for binning, and a third time to process experiments into trials.

ii. ```python
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...

for idx, (_, row) in enumerate(exp_table.iterrows()):
    nwb_data = load_nwb_data(nwb_path)
    ...

for idx, (_, row) in enumerate(exp_table.iterrows()):
    nwb_data = load_nwb_data(nwb_path)
    result = process_experiment(nwb_data, all_image_names)
```

iii. The AI did not explicitly write this in `CONVERSION_NOTES.md`, but it follows directly from the three-pass structure of `convert_data.py`.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops could have been vectorized: looping over all stimuli to paint image identity and image change onto the ophys timebase, looping over each frame to forward-fill gray-screen image labels, and looping over experiments multiple times for image discovery and percentile-statistics collection.

ii. ```python
for i in range(len(stim_starts)):
    ...
    image_at_ophys[mask] = img_idx

for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        ...

for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
    elif last_img >= 0:
        trial_image[k] = last_img
```

iii. The AI did not document these as optimization opportunities, but they are straightforwardly visible from the implementation.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats three classes of work: opening the same NWB files in multiple passes, interpolating running speed in both the first pass and the second pass, and performing the pupil blink-masking plus interpolation logic in both passes.

ii. ```python
# First pass
running = interpolate_to_ophys(...)
...
pupil = np.interp(...)

# Second pass
running_at_ophys = interpolate_to_ophys(...)
...
pupil_at_ophys = np.interp(...)
```

iii. This is not described in `CONVERSION_NOTES.md`; it is an implementation property of `convert_data.py`.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads or computes several values that are not used in the final output: `events_timestamps`, `dff`, `cell_specimen_ids`, `pupil_area`, `trial_change_times`, `trial_is_change`, `trial_change_image`, `trial_initial_image`, the returned `running_at_ophys`, `pupil_at_ophys`, `events_full`, and the unused `subjects_list` / `trial_outcomes` accumulators.

ii. ```python
events_ts = f['processing/ophys/event_detection/timestamps'][:]
dff = f['processing/ophys/dff/traces/data'][:]
data['cell_specimen_ids'] = cst['cell_specimen_id'][:]
data['pupil_area'] = pupil['area'][:]
data['trial_change_times'] = trials['change_time'][:]
...
return neural_trials, output_trials, trial_outcomes, running_at_ophys, pupil_at_ophys, events
...
subjects_list = []
```

iii. The AI did not justify this explicitly. It appears to be leftover exploratory or convenience loading rather than information retained for downstream decoding.
