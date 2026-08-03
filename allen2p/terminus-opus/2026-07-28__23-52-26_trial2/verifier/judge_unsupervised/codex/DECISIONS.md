# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script reads `ophys_experiment_table.csv`, keeps only experiment IDs that have downloaded NWB files in `data/behavior_ophys_experiments`, drops passive experiments, and then loads each remaining NWB directly with `h5py`. For each NWB it reads subject metadata, ophys events, ophys timestamps, ROI validity flags, running speed, pupil tracking, trials, and one stimulus-presentation table.

ii. ```python
def get_experiment_list(sample=False):
    expt_table = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    downloaded_ids = set()
    for f in os.listdir(EXPT_DIR):
        if f.endswith('.nwb'):
            eid = int(f.split('_')[-1].replace('.nwb', ''))
            downloaded_ids.add(eid)
    expt_table = expt_table[expt_table.ophys_experiment_id.isin(downloaded_ids)]
    expt_table = expt_table[expt_table.passive == False]
    return expt_table

def load_experiment_data(expt_id):
    fname = os.path.join(EXPT_DIR, f'behavior_ophys_experiment_{expt_id}.nwb')
    f = h5py.File(fname, 'r')
    data['events'] = f['processing/ophys/event_detection/data'][:].T
    data['ophys_timestamps'] = f['processing/ophys/dff/traces/timestamps'][:]
    data['running_speed'] = f['processing/running/speed/data'][:]
    data['trials'] = trial_data
    data['stimulus_presentations'] = stim_data
```

iii. In `CONVERSION_NOTES.md`, the agent says it would use “all downloaded data” but exclude passive sessions because they are not part of the active Visual Behavior task. In trajectory step 18-19 it explicitly decided to include all active experiments rather than only the paper’s familiar-session subset.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by the NWB field `general/subject/subject_id`. Each processed experiment contributes one session-level `subject_idx` entry pointing into the deduplicated `subjects` list.

ii. ```python
data['subject_id'] = f['general/subject/subject_id'][()]
if isinstance(data['subject_id'], bytes):
    data['subject_id'] = data['subject_id'].decode()

subj = str(result['subject_id'])
if subj not in unique_subjects:
    unique_subjects.append(subj)
subject_idx_list.append(unique_subjects.index(subj))
```

iii. There is no deeper justification in the script; the notes and trajectory treat mouse identity as the natural NWB subject identifier.

## 1-c. How are the data split into sessions?

i. The script treats each NWB file, i.e. each `ophys_experiment_id`, as one output session. It does not merge experiments that share the same `ophys_session_id`, so multi-plane MESO sessions become multiple decoder sessions.

ii. ```python
def load_experiment_data(expt_id):
    fname = os.path.join(EXPT_DIR, f'behavior_ophys_experiment_{expt_id}.nwb')
    ...

for i, (_, expt_info) in enumerate(expt_table.iterrows()):
    expt_id = expt_info['ophys_experiment_id']
    result = process_experiment(expt_id, expt_info, ...)
    all_neural.append(result['neural'])
```

iii. `CONVERSION_NOTES.md` says “Each NWB = one ophys experiment = one imaging plane.” In trajectory step 22 the agent explicitly decided that MESO experiments should be treated as separate sessions because each plane has different neurons even when behavior is shared.

## 1-d. How are the data split into trials?

i. Trials are taken from `intervals/trials`. For each included trial, the script uses `start_time` and `stop_time` as the trial boundaries and creates one neural trial matrix and one output trial matrix spanning that interval.

ii. ```python
trial_data = raw['trials']
include_mask = trial_data['go'] | trial_data['catch']
include_indices = np.where(include_mask)[0]

for trial_idx in include_indices:
    trial_start = trial_data['start_time'][trial_idx]
    trial_stop = trial_data['stop_time'][trial_idx]
    trial_neural, common_ts = resample_events_to_common_bins(
        events, ophys_ts, trial_start, trial_stop
    )
```

iii. `CONVERSION_NOTES.md` Step 5 says “Trial segmentation: trial start_time to stop_time from NWB,” and trajectory step 37 repeats that choice.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by task type rather than by an explicit quality metric: only `go` and `catch` trials are included. The script also drops experiments with fewer than two included trials and drops any trial whose outcome does not map to hit/miss/correct-reject/false-alarm.

ii. ```python
include_mask = trial_data['go'] | trial_data['catch']
include_indices = np.where(include_mask)[0]

if len(include_indices) < 2:
    return None

outcome = get_trial_outcome(trial_data, trial_idx)
if outcome == -1:
    continue
```

iii. `CONVERSION_NOTES.md` says “Trial curation: Include Go and Catch trials, exclude Aborted and Auto-rewarded,” and the trajectory uses the same rule. In this dataset those excluded trial types are disjoint from `go`/`catch`, so the simpler mask was sufficient for the agent’s intent.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from the NWB event-detection matrix `processing/ophys/event_detection/data`, filtered to ROIs where `valid_roi` is true. Ophys timestamps come from `processing/ophys/dff/traces/timestamps`.

ii. ```python
data['events'] = f['processing/ophys/event_detection/data'][:].T
data['ophys_timestamps'] = f['processing/ophys/dff/traces/timestamps'][:]
data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]

valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
ophys_ts = raw['ophys_timestamps']
```

iii. The notes say the agent chose “detected calcium events (not raw dff)” and matched SDK-style invalid-ROI filtering. Trajectory step 35 states that raw NWB events, not dF/F, are the intended neural signal.

## 2-b. How is the `neural` data processed?

i. The script uses raw event amplitudes, does not apply the AllenSDK half-Gaussian smoothing, and rebins every trial to a global `COMMON_DT = 0.09323` s grid. Within each common bin it sums all native ophys events whose timestamps fall into that bin.

ii. ```python
COMMON_DT = 0.09323  # seconds, ~10.73 Hz

def resample_events_to_common_bins(events, ophys_ts, trial_start, trial_stop):
    common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
    ...
    trial_mask = (ophys_ts >= trial_start - COMMON_DT) & (ophys_ts < trial_stop + COMMON_DT)
    trial_ophys_ts = ophys_ts[trial_mask]
    trial_events = events[:, trial_mask]
    ...
    bin_assignments = np.digitize(trial_ophys_ts, bin_edges) - 1
    ...
    for b in range(n_common):
        mask = bin_assignments == b
        if mask.any():
            resampled[:, b] = trial_events[:, mask].sum(axis=1)
```

iii. `CONVERSION_NOTES.md` lists two key decisions: use raw detected events and use a common 93.23 ms time bin to satisfy the decoder format. Trajectory steps 47-50 show the agent revising the first version after noticing mixed CAM2P/MESO frame rates and choosing MESO-rate downsampling as the “lowest common denominator.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural quality-control filter is ROI validity. Neurons with `valid_roi == False` are discarded. There is no additional event-amplitude thresholding or cell-level curation in the conversion script.

ii. ```python
data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
...
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
```

iii. The notes cite the AllenSDK default `exclude_invalid_rois=True` and say this matches the reference code’s curation behavior.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the ophys timebase first, then segmented by trial. For each trial the script constructs `common_ts = np.arange(trial_start, trial_stop, COMMON_DT)` and sums ophys events into those trial-local bins. In effect, the per-trial matrices start at trial onset, but the assignment of samples uses absolute ophys timestamps.

ii. ```python
common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
trial_mask = (ophys_ts >= trial_start - COMMON_DT) & (ophys_ts < trial_stop + COMMON_DT)
trial_ophys_ts = ophys_ts[trial_mask]
...
bin_assignments = np.digitize(trial_ophys_ts, bin_edges) - 1
```

iii. The task said to align streams “based on ophys timestamp.” The notes justify the implementation as trial segmentation on top of a common ophys-derived time grid, and the metadata later labels the zero point as trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. All converted data use 93.23 ms bins, corresponding to about 10.73 Hz. Yes: the script explicitly rebins native ophys events and resamples continuous variables onto that common grid, downsampling faster CAM2P data to the MESO rate.

ii. ```python
COMMON_DT = 0.09323  # seconds, ~10.73 Hz
...
print(f'Common time bin: {COMMON_DT*1000:.2f} ms ({1/COMMON_DT:.2f} Hz)')
...
'time_bin_size': COMMON_DT * 1000,
```

iii. `CONVERSION_NOTES.md` Step 4-5 and trajectory steps 47-50 document that the agent originally noticed mixed native frame rates and then chose a single MESO-rate bin size because the target format required consistent bins across sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation table, specifically `intervals/<Natural_Images...>/image_name` and its corresponding `start_time`.

ii. ```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    vals = stim[key][:]
    ...
    stim_data[key] = vals

start_times = stim_data['start_time']
image_names = stim_data['image_name']
```

iii. `CONVERSION_NOTES.md` Step 5 maps `stimulus_presentations.image_name` to `output[0] (image_identity)`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The script first collects all non-omitted image names across experiments, sorts them, and maps each image name to a categorical integer. Within each trial it finds the most recent stimulus presentation for each common time bin and uses that image. If the current presentation is `'omitted'`, or if a bin falls during gray time, it carries forward the last non-omitted image identity.

ii. ```python
all_images = sorted(all_images)
image_to_idx = {img: i for i, img in enumerate(all_image_names)}

indices = np.searchsorted(start_times, common_ts, side='right') - 1
last_valid = 0
for t in range(n_tp):
    idx = indices[t]
    ...
    if img == 'omitted':
        image_id[t] = last_valid
    elif img in image_to_idx:
        image_id[t] = image_to_idx[img]
        last_valid = image_to_idx[img]
```

iii. `CONVERSION_NOTES.md` explicitly justifies this with “Image identity during gray screen: Assign most recently shown non-omitted image,” reflecting the agent’s choice to force a time-varying label through gray and omission periods.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated directly on the same `common_ts` bins used for the rebinned neural data, so each neural time bin gets a same-length categorical image label.

ii. ```python
trial_neural, common_ts = resample_events_to_common_bins(...)
img_id = get_image_at_timepoints(common_ts, raw['stimulus_presentations'], image_to_idx)
...
trial_output = np.stack([
    img_id, img_change, run_binned, pupil_binned,
    np.full(n_common, outcome, dtype=np.int64)
], axis=0)
```

iii. The justification is implicit in the conversion design: all time-varying outputs are built on the same per-trial common timebase as `neural`.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change`, `start_time`, and `stop_time` fields.

ii. ```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    ...

is_change = stim_data.get('is_change', np.zeros(len(stim_data['start_time'])))
start_times = stim_data['start_time']
stop_times = stim_data['stop_time']
```

iii. `CONVERSION_NOTES.md` Step 5 maps `stimulus_presentations.is_change` to `output[1] (image_change)`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The script creates a binary time series. For each stimulus presentation flagged `is_change == 1`, it marks every common time bin that falls inside that presentation’s `[start_time, stop_time)` window as 1; all other bins stay 0.

ii. ```python
def get_image_change_at_timepoints(common_ts, stim_data):
    change = np.zeros(n_tp, dtype=np.int64)
    change_indices = np.where(np.array(is_change) == 1)[0]
    for ci in change_indices:
        mask = (common_ts >= start_times[ci]) & (common_ts < stop_times[ci])
        change[mask] = 1
    return change
```

iii. The notes describe this as “1 during change stimulus,” and trajectory step 77 says the agent checked that the expected bins inside the changed-image window were marked as 1.

## 4-c. How is `output` *Image change* thresholded into categories?

i. There is no numeric threshold. The variable is already binary and is encoded as category 0 for `no_change` and 1 for `change`.

ii. ```python
output_values = [
    all_image_names,
    ['no_change', 'change'],
    ...
]
```

iii. The binary coding follows directly from the decoder task specification, which asked for a binary image-change output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is sampled on `common_ts`, the same per-trial time grid used for `neural`.

ii. ```python
trial_neural, common_ts = resample_events_to_common_bins(...)
img_change = get_image_change_at_timepoints(common_ts, raw['stimulus_presentations'])
```

iii. The alignment choice is implicit in the common-timebase design and was sanity-checked in the trajectory after the agent worried about bins around change onset.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `processing/running/speed/data` and `processing/running/speed/timestamps`.

ii. ```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `running_speed` directly to `output[2]`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The script first estimates global percentile cutoffs by sampling up to 2000 non-NaN running-speed values per experiment across the dataset. Per trial, it linearly interpolates running speed onto `common_ts`, then bins the interpolated values using those percentile edges.

ii. ```python
run_data = f['processing/running/speed/data'][:]
valid_run = run_data[~np.isnan(run_data)]
if len(valid_run) > 2000:
    rng = np.random.RandomState(int(expt_id) % (2**31))
    valid_run = rng.choice(valid_run, 2000, replace=False)
...
running_percentiles = np.percentile(all_running, np.linspace(0, 100, 6))

run_interp = interpolate_to_common(
    raw['running_speed'], raw['running_timestamps'], common_ts
)
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
```

iii. `CONVERSION_NOTES.md` calls this a key decision: “Percentile binning: Computed across all sessions globally (sampled 2000 values per session).”

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is discretized into five percentile bins using four global cut points. The code clips indices to `[0, 4]`. Missing interpolated values are forced into the middle bin, index 2.

ii. ```python
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
run_binned[np.isnan(run_interp)] = 2
```

iii. The task required five equal percentile bins. The notes explicitly mention global percentiles; the “fill missing with middle bin” behavior is only stated in the notes’ general missing-data decision.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by linear interpolation from the native running timestamps to the same `common_ts` grid used for rebinned neural activity.

ii. ```python
def interpolate_to_common(data, source_ts, common_ts):
    f_interp = interpolate.interp1d(
        source_ts[valid], data[valid],
        kind='linear', bounds_error=False, fill_value=np.nan
    )
    return f_interp(common_ts)

run_interp = interpolate_to_common(
    raw['running_speed'], raw['running_timestamps'], common_ts
)
```

iii. The notes and trajectory both state that running speed must be aligned to the ophys-derived timebase.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Despite the output name `pupil_diameter`, the script actually derives it from the eye-tracking field `acquisition/EyeTracking/pupil_tracking/area` plus its timestamps.

ii. ```python
try:
    data['pupil_area'] = f['acquisition/EyeTracking/pupil_tracking/area'][:]
    data['pupil_timestamps'] = f['acquisition/EyeTracking/pupil_tracking/timestamps'][:]
    data['has_pupil'] = True
except KeyError:
    data['has_pupil'] = False
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly maps `pupil_tracking.area` to `output[3] (pupil_diameter)`, so this was a deliberate simplification rather than an accident in only one line of code.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The script does not compute a diameter from width, height, or area. It linearly interpolates pupil area onto `common_ts` and then bins that interpolated area as if it were the requested diameter.

ii. ```python
if raw['has_pupil']:
    pupil_interp = interpolate_to_common(
        raw['pupil_area'], raw['pupil_timestamps'], common_ts
    )
else:
    pupil_interp = np.full(n_common, np.nan)

if pupil_percentiles is not None:
    pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
```

iii. The notes justify this only as a variable mapping plus percentile binning. They also say missing pupil data are sent to the middle bin.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The script uses five global percentile bins computed from sampled non-NaN pupil-area values. It clips to `[0, 4]`. Missing pupil values, or experiments without pupil tracking, are assigned bin 2.

ii. ```python
if all_pupil:
    all_pupil = np.concatenate(all_pupil)
    pupil_percentiles = np.percentile(all_pupil, np.linspace(0, 100, 6))
...
pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
pupil_binned = np.clip(pupil_binned, 0, 4).astype(np.int64)
pupil_binned[np.isnan(pupil_interp)] = 2
```

iii. `CONVERSION_NOTES.md` lists global percentile binning and says missing pupil data are assigned the middle bin.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The pupil-derived signal is aligned by interpolation from eye-tracking timestamps onto the same `common_ts` grid used for `neural`.

ii. ```python
pupil_interp = interpolate_to_common(
    raw['pupil_area'], raw['pupil_timestamps'], common_ts
)
```

iii. The notes and trajectory describe this as part of the general plan to align behavioral streams to the ophys-derived timebase.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the per-trial boolean flags `hit`, `miss`, `correct_reject`, and `false_alarm` in `intervals/trials`.

ii. ```python
for key in ['go', 'catch', 'aborted', 'auto_rewarded', 'hit', 'miss',
            'correct_reject', 'false_alarm', 'start_time', 'stop_time',
            'change_time', 'change_image_name', 'initial_image_name', 'is_change']:
    vals = trials[key][:]
    trial_data[key] = vals

def get_trial_outcome(trial_data, trial_idx):
    if trial_data['hit'][trial_idx]: return 0
    elif trial_data['miss'][trial_idx]: return 1
    elif trial_data['correct_reject'][trial_idx]: return 2
    elif trial_data['false_alarm'][trial_idx]: return 3
```

iii. `CONVERSION_NOTES.md` Step 5 maps these trial flags directly to the four outcome classes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The script encodes each included trial as one of four categories: hit=0, miss=1, correct_reject=2, false_alarm=3. If none of those flags is set, the trial is skipped. The chosen category is then repeated across all time bins in that trial, even though the variable is logically static.

ii. ```python
outcome = get_trial_outcome(trial_data, trial_idx)
if outcome == -1:
    continue

trial_output = np.stack([
    img_id, img_change, run_binned, pupil_binned,
    np.full(n_common, outcome, dtype=np.int64)
], axis=0)
```

iii. The notes describe trial outcome as “Categorical static” with four categories.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing pupil tracking is handled by filling with `NaN` and then assigning the middle category bin. Missing running or pupil samples after interpolation also go to the middle bin. Trials shorter than two common bins are skipped. Trials with no recognized outcome are skipped. If no ophys frames fall in a trial window, the script creates an all-zero neural matrix. For image identity before any stimulus or during omitted stimuli, it carries forward the last valid image and defaults to image index 0 before the first valid image is seen.

ii. ```python
if n_common < 2:
    return None, None
...
if len(trial_ophys_ts) == 0:
    return np.zeros((n_neurons, n_common), dtype=np.float32), common_ts
...
if valid.sum() < 2:
    return np.full(len(common_ts), np.nan)
...
run_binned[np.isnan(run_interp)] = 2
...
if raw['has_pupil']:
    ...
else:
    pupil_interp = np.full(n_common, np.nan)
...
pupil_binned[np.isnan(pupil_interp)] = 2
...
last_valid = 0
```

iii. `CONVERSION_NOTES.md` explicitly justifies only one of these behaviors: “Missing pupil data: Fill with NaN, assign to middle bin (bin 2).” The other defaults are implementation choices visible in the code rather than separately argued in the notes.

## 9-a. What are the most time-consuming steps of the code?

i. The dominant costs are repeated full-NWB I/O and per-trial processing. Specifically: one full scan of every experiment in `collect_global_stats`, another full pass in `process_experiment`, and inside each trial the neural rebinning loop over common bins plus interpolation and output construction.

ii. ```python
def collect_global_stats(expt_table):
    for i, expt_id in enumerate(expt_ids):
        with h5py.File(fname, 'r') as f:
            ...

for i, (_, expt_info) in enumerate(expt_table.iterrows()):
    result = process_experiment(...)

for trial_idx in include_indices:
    trial_neural, common_ts = resample_events_to_common_bins(...)
```

iii. The trajectory repeatedly comments on runtime, first during the failed high-resolution version and then again after adding common-bin resampling. The notes’ runtime estimates also identify loading plus resampling as the main per-session costs.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization targets are the loop over common bins in `resample_events_to_common_bins`, the loop over timepoints in `get_image_at_timepoints`, the loop over changed stimuli in `get_image_change_at_timepoints`, and the per-trial repeated interpolation/lookup workflow that could have been done once per session and then sliced.

ii. ```python
for b in range(n_common):
    mask = bin_assignments == b
    if mask.any():
        resampled[:, b] = trial_events[:, mask].sum(axis=1)

for t in range(n_tp):
    idx = indices[t]
    ...

for ci in change_indices:
    mask = (common_ts >= start_times[ci]) & (common_ts < stop_times[ci])
    change[mask] = 1
```

iii. Trajectory step 50 explicitly says the agent noticed the original resampling approach would be too slow and tried to “vectorize” it, but the final code still leaves several Python loops in the inner path.

## 9-c. What processing does the code repeat multiple times?

i. The code reopens every NWB at least twice, recomputes per-trial `common_ts`, re-interpolates running and pupil separately for every trial, and repeatedly scans the full session stimulus table inside each trial to recover image identity and change labels.

ii. ```python
with h5py.File(fname, 'r') as f:
    ...  # collect_global_stats

raw = load_experiment_data(expt_id)
...
for trial_idx in include_indices:
    common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
    run_interp = interpolate_to_common(...)
    pupil_interp = interpolate_to_common(...)
    img_id = get_image_at_timepoints(common_ts, raw['stimulus_presentations'], image_to_idx)
    img_change = get_image_change_at_timepoints(common_ts, raw['stimulus_presentations'])
```

iii. The trajectory shows two distinct full-dataset passes by design: one to collect global statistics and one to build the converted output. The repeated per-trial interpolations and stimulus lookups are implicit in the implementation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads `cell_specimen_ids` but never uses them. It also expands the static `trial_outcome` label to a full time series for every trial, even though the variable is static. In `--show-processing` mode it carries raw data into `result` only for plotting, and the plotting path itself is separate from the final dataset. It also creates empty `input` arrays for every trial even though there are no decoder inputs.

ii. ```python
data['cell_specimen_ids'] = f['processing/ophys/image_segmentation/cell_specimen_table/cell_specimen_id'][:]
...
np.full(n_common, outcome, dtype=np.int64)
...
if show_processing:
    result['raw'] = raw
    result['ophys_ts'] = ophys_ts
...
all_input.append([np.zeros((0, t.shape[1]), dtype=np.float32) for t in result['neural']])
```

iii. These are not defended in the notes beyond satisfying the target format and optional plotting; they are simply side effects of the implementation.
