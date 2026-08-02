# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache. It reads `ophys_experiment_table.csv`, restricts to experiment IDs that have downloaded NWB files, filters out passive experiments, and then opens each NWB file directly with `h5py`. Trial information is loaded from each file's `intervals/trials` group.

ii.
```python
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
    ...
    trials = f['intervals/trials']
```

iii. In `CONVERSION_NOTES.md`, the AI says the local SDK cache layout was inconvenient, so it chose direct NWB access via `h5py` and processed the 202 downloaded active experiments. In the trajectory it states: "Each experiment (NWB file) = one session in output" and "Filter: active (non-passive) experiments only."

## 1-b. How are the data split into subjects?

i. Subjects are taken from each NWB file's `general/subject/subject_id` field. The script builds a unique ordered `subjects` list and records a `subject_idx` entry for each processed experiment/session.

ii.
```python
data['subject_id'] = f['general/subject/subject_id'][()]
if isinstance(data['subject_id'], bytes):
    data['subject_id'] = data['subject_id'].decode()
...
subj = str(result['subject_id'])
if subj not in unique_subjects:
    unique_subjects.append(subj)
subject_idx_list.append(unique_subjects.index(subj))
```

iii. The trajectory and notes consistently treat the mouse identifier stored in the NWB file as the subject definition. No separate justification beyond using the file-native subject field is given.

## 1-c. How are the data split into sessions?

i. The AI treats each ophys experiment NWB file as one output session. It does not group experiments that share the same `ophys_session_id`.

ii.
```python
for i, (_, expt_info) in enumerate(expt_table.iterrows()):
    expt_id = expt_info['ophys_experiment_id']
    ...
    result = process_experiment(expt_id, expt_info, ...)
    ...
    all_neural.append(result['neural'])
```

iii. The clearest justification is in trajectory step 38: "Each experiment (NWB file) = one session in output." The notes also summarize the data as 202 active experiments and report those experiments directly as the converted "sessions."

## 1-d. How are the data split into trials?

i. Trials are taken from the raw NWB `intervals/trials` table. For each included trial, the script uses `start_time` and `stop_time` as the trial window and builds one trial matrix per interval.

ii.
```python
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

iii. The trajectory states the intended trial definition explicitly: "Trial window: trial start_time to stop_time." The notes also list "Trial segmentation: trial start_time to stop_time from NWB."

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps trials where `go` or `catch` is true, skips experiments with fewer than 2 included trials, skips trials whose resampled window has fewer than 2 common time bins, and skips trials whose outcome does not map to one of the four expected labels.

ii.
```python
include_mask = trial_data['go'] | trial_data['catch']
include_indices = np.where(include_mask)[0]

if len(include_indices) < 2:
    print(f'  WARNING: Too few trials in experiment {expt_id}')
    return None
...
trial_neural, common_ts = resample_events_to_common_bins(...)
if trial_neural is None:
    continue
...
outcome = get_trial_outcome(trial_data, trial_idx)
if outcome == -1:
    continue
```

iii. The notes say "Trial curation: Include Go and Catch trials, exclude Aborted and Auto-rewarded." In the trajectory the AI frames this as `go + catch` filtering rather than explicit `aborted`/`auto_rewarded` checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB event-detection array, not from dF/F. The script loads `processing/ophys/event_detection/data`, transposes it to neuron-by-time, and later filters it by `valid_roi`.

ii.
```python
data['events'] = f['processing/ophys/event_detection/data'][:].T
data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
...
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
```

iii. The notes repeatedly justify this choice by citing the paper's use of "detected calcium events" rather than dF/F, and explicitly state: "Neural data: detected calcium events (not raw dff)."

## 2-b. How is the `neural` data processed?

i. The AI filters to valid ROIs, then rebins each trial's event train onto a fixed 93.23 ms grid and sums all event values from native ophys frames that fall in each common bin.

ii.
```python
COMMON_DT = 0.09323
...
def resample_events_to_common_bins(events, ophys_ts, trial_start, trial_stop):
    common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
    ...
    bin_assignments = np.digitize(trial_ophys_ts, bin_edges) - 1
    resampled = np.zeros((n_neurons, n_common), dtype=np.float32)
    for b in range(n_common):
        mask = bin_assignments == b
        if mask.any():
            resampled[:, b] = trial_events[:, mask].sum(axis=1)
```

iii. The notes justify this as a common-bin design choice: "Common time bin: 93.23ms (MESO rate) - required because 'time bins should be the same size for all trials and sessions'" and "Neural data: Raw detected events ... summed within each common bin."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural quality filter is `valid_roi`; neurons with `valid_roi == False` are removed.

ii.
```python
data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
...
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
```

iii. `CONVERSION_NOTES.md` states: "ROI filtering: valid_roi flag from segmentation pipeline" and "Valid ROI filtering: Exclude invalid ROIs matching SDK default."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to trial start. The per-trial time axis is `np.arange(trial_start, trial_stop, COMMON_DT)`, so the first bin starts at `start_time`.

ii.
```python
for trial_idx in include_indices:
    trial_start = trial_data['start_time'][trial_idx]
    trial_stop = trial_data['stop_time'][trial_idx]
    trial_neural, common_ts = resample_events_to_common_bins(
        events, ophys_ts, trial_start, trial_stop
    )
```

iii. The metadata and trajectory make this explicit. The output metadata sets `'temporal_alignment_event': 'Trial start time'`, and the trajectory says "Trial window: trial start_time to stop_time."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI forces a uniform 93.23 ms time bin (`~10.73 Hz`) across all experiments. Yes, it rebins both neural and behavioral streams onto this common grid.

ii.
```python
COMMON_DT = 0.09323  # seconds, ~10.73 Hz
...
common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
...
'time_bin_size': COMMON_DT * 1000,
```

iii. The notes justify the choice as needed to satisfy the task's fixed-bin requirement across CAM2P and MESO data. The trajectory shows the AI noticing mixed frame rates and deciding to downsample everything to the MESO rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation table, mainly `image_name` and `start_time`, not from the per-trial `initial_image_name`/`change_image_name` fields.

ii.
```python
stim = f[f'intervals/{stim_key}']
...
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    ...
    stim_data[key] = vals
...
start_times = stim_data['start_time']
image_names = stim_data['image_name']
indices = np.searchsorted(start_times, common_ts, side='right') - 1
```

iii. In trajectory step 38 the AI justifies this by saying that during gray periods it should "assign the image identity of the most recent image presentation." The notes say: "Image identity during gray screen: Assign most recently shown non-omitted image."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each resampled trial time bin, the script finds the most recent stimulus presentation, carries forward the last non-omitted image during gray or omitted periods, and then maps image names to global integer IDs.

ii.
```python
def get_image_at_timepoints(common_ts, stim_data, image_to_idx):
    indices = np.searchsorted(start_times, common_ts, side='right') - 1
    last_valid = 0
    for t in range(n_tp):
        idx = indices[t]
        if idx < 0:
            image_id[t] = last_valid
            continue
        img = image_names[idx]
        if img == 'omitted':
            image_id[t] = last_valid
        elif img in image_to_idx:
            image_id[t] = image_to_idx[img]
            last_valid = image_to_idx[img]
```

iii. The notes emphasize two justifications: using the most recent non-omitted image and using a global image-name mapping collected across all experiments.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same per-trial `common_ts` grid used for the rebinned neural data, so both share the same number of time bins and trial boundaries.

ii.
```python
trial_neural, common_ts = resample_events_to_common_bins(...)
img_id = get_image_at_timepoints(common_ts, raw['stimulus_presentations'], image_to_idx)
...
trial_output = np.stack([
    img_id, img_change, run_binned, pupil_binned,
    np.full(n_common, outcome, dtype=np.int64)
], axis=0)
```

iii. The AI's stated alignment principle throughout the notes is to use a common ophys-derived time base for all signals, then slice all outputs on that same trial grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table's `is_change`, `start_time`, and `stop_time` fields.

ii.
```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    ...
is_change = stim_data.get('is_change', np.zeros(len(stim_data['start_time'])))
start_times = stim_data['start_time']
stop_times = stim_data['stop_time']
```

iii. The notes describe the mapping as: "`stimulus_presentations.is_change` -> output[1] (image_change) | Binary time series | 1 during change stimulus."

## 4-b. What processing is involved in computing `output` *Image change*?

i. The script marks bins as 1 only when the resampled time falls inside a stimulus presentation whose `is_change` flag is true; otherwise the value is 0.

ii.
```python
def get_image_change_at_timepoints(common_ts, stim_data):
    change = np.zeros(n_tp, dtype=np.int64)
    change_indices = np.where(np.array(is_change) == 1)[0]
    for ci in change_indices:
        mask = (common_ts >= start_times[ci]) & (common_ts < stop_times[ci])
        change[mask] = 1
    return change
```

iii. The notes justify this as representing change only "during change stimulus." The trajectory later mentions that the 250 ms change window can be hard to see after resampling to 93 ms bins.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary categorical output: `0` for `no_change`, `1` for `change`.

ii.
```python
output_values = [
    all_image_names,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
    [f'bin_{i}' for i in range(5)],
    ['hit', 'miss', 'correct_reject', 'false_alarm'],
]
```

iii. No extra justification is given beyond the task requirement that image change be binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is computed directly on `common_ts`, the same trial grid used for `neural`.

ii.
```python
trial_neural, common_ts = resample_events_to_common_bins(...)
img_change = get_image_change_at_timepoints(common_ts, raw['stimulus_presentations'])
...
trial_output = np.stack([img_id, img_change, run_binned, pupil_binned, ...], axis=0)
```

iii. The notes consistently describe all outputs as being aligned after interpolation/resampling to the common ophys-based time grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is taken from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. The notes treat these as the canonical running-wheel signals and do not give a separate defense beyond matching the dataset structure.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The script linearly interpolates running speed to the per-trial `common_ts` grid and discretizes the interpolated values into 5 global percentile bins. The percentile edges are estimated from raw running data scanned across experiments, with at most 2000 non-NaN samples kept per experiment.

ii.
```python
run_interp = interpolate_to_common(
    raw['running_speed'], raw['running_timestamps'], common_ts
)
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
...
run_data = f['processing/running/speed/data'][:]
valid_run = run_data[~np.isnan(run_data)]
if len(valid_run) > 2000:
    rng = np.random.RandomState(int(expt_id) % (2**31))
    valid_run = rng.choice(valid_run, 2000, replace=False)
all_running.append(valid_run)
running_percentiles = np.percentile(all_running, np.linspace(0, 100, 6))
```

iii. The notes say "Running speed ... Interpolate + 5 percentile bins | Global percentiles" and justify the sampling as a speed/memory tradeoff during the global statistics pass.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is converted to 5 discrete bins labeled `bin_0` through `bin_4`. Values are thresholded using global percentile edges, and NaNs are assigned to the middle bin `2`.

ii.
```python
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
run_binned[np.isnan(run_interp)] = 2
...
[f'bin_{i}' for i in range(5)]
```

iii. There is no explicit running-speed note for NaNs, but the notes do justify the same "middle bin" strategy for missing pupil as a neutral category, and the code applies that same policy to running.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto `common_ts`, so it has one value per neural time bin within each trial.

ii.
```python
run_interp = interpolate_to_common(
    raw['running_speed'], raw['running_timestamps'], common_ts
)
...
trial_output = np.stack([
    img_id, img_change, run_binned, pupil_binned,
    np.full(n_common, outcome, dtype=np.int64)
], axis=0)
```

iii. The trajectory and notes repeatedly say the plan is to "Align running speed and pupil to ophys timestamps via interpolation," then use the shared common-bin trial grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI uses pupil tracking `area` and its timestamps from `acquisition/EyeTracking/pupil_tracking`, not the SDK's cleaned `pupil_width`.

ii.
```python
try:
    data['pupil_area'] = f['acquisition/EyeTracking/pupil_tracking/area'][:]
    data['pupil_timestamps'] = f['acquisition/EyeTracking/pupil_tracking/timestamps'][:]
    data['has_pupil'] = True
except KeyError:
    data['has_pupil'] = False
```

iii. The notes describe this simply as "pupil area" from the NWB structure. No explicit justification is given for choosing area over width.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. If pupil data exist, the script linearly interpolates pupil area to `common_ts` and bins it into 5 global percentile bins computed from a sampled global scan of pupil values. It does not remove blinks.

ii.
```python
if raw['has_pupil']:
    pupil_interp = interpolate_to_common(
        raw['pupil_area'], raw['pupil_timestamps'], common_ts
    )
...
pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
pupil_binned = np.clip(pupil_binned, 0, 4).astype(np.int64)
...
pupil_data = f['acquisition/EyeTracking/pupil_tracking/area'][:]
valid_pupil = pupil_data[~np.isnan(pupil_data)]
if len(valid_pupil) > 2000:
    rng = np.random.RandomState(int(expt_id) % (2**31))
    valid_pupil = rng.choice(valid_pupil, 2000, replace=False)
all_pupil.append(valid_pupil)
```

iii. The notes justify the global-percentile discretization and explicitly state the missing-data policy: "Missing pupil data: Fill with NaN, assign to middle bin (bin 2)."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil is discretized into 5 bins `bin_0` to `bin_4` using global percentiles. Missing values, or entirely missing pupil streams, are assigned to bin `2`.

ii.
```python
if pupil_percentiles is not None:
    pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
    pupil_binned = np.clip(pupil_binned, 0, 4).astype(np.int64)
    pupil_binned[np.isnan(pupil_interp)] = 2
else:
    pupil_binned = np.full(n_common, 2, dtype=np.int64)
```

iii. This is explicitly justified in `CONVERSION_NOTES.md` as a neutral fallback for missing pupil data.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are interpolated onto the same `common_ts` trial grid used for rebinned neural data, so each neural bin gets one pupil-bin label.

ii.
```python
pupil_interp = interpolate_to_common(
    raw['pupil_area'], raw['pupil_timestamps'], common_ts
)
...
trial_output = np.stack([
    img_id, img_change, run_binned, pupil_binned,
    np.full(n_common, outcome, dtype=np.int64)
], axis=0)
```

iii. The same shared-time-base rationale is used here as for running speed and image identity.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial fields `hit`, `miss`, `correct_reject`, and `false_alarm`.

ii.
```python
for key in ['go', 'catch', 'aborted', 'auto_rewarded', 'hit', 'miss',
            'correct_reject', 'false_alarm', 'start_time', 'stop_time',
            'change_time', 'change_image_name', 'initial_image_name', 'is_change']:
    ...

def get_trial_outcome(trial_data, trial_idx):
    if trial_data['hit'][trial_idx]: return 0
    elif trial_data['miss'][trial_idx]: return 1
    elif trial_data['correct_reject'][trial_idx]: return 2
    elif trial_data['false_alarm'][trial_idx]: return 3
```

iii. The notes summarize the intended categories as "hit/miss/CR/FA" and the trajectory describes them as the four static trial-outcome classes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The script maps the four mutually exclusive outcome booleans to integer codes and then broadcasts the selected code across all time bins of the trial.

ii.
```python
def get_trial_outcome(trial_data, trial_idx):
    if trial_data['hit'][trial_idx]: return 0
    elif trial_data['miss'][trial_idx]: return 1
    elif trial_data['correct_reject'][trial_idx]: return 2
    elif trial_data['false_alarm'][trial_idx]: return 3
    else: return -1
...
trial_output = np.stack([
    img_id, img_change, run_binned, pupil_binned,
    np.full(n_common, outcome, dtype=np.int64)
], axis=0)
```

iii. The trajectory justifies this as a static per-trial output. The notes state "Trial outcome: static per-trial."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses permissive fail-soft handling. Entire experiments are skipped on exceptions, trials are skipped if the rebinned window is too short or the outcome is unmapped, experiments with no valid neurons or no stimulus table are skipped, missing pupil streams are filled with bin `2`, and NaNs after interpolation for running or pupil are also set to bin `2`.

ii.
```python
if n_neurons == 0:
    return None
if raw['stimulus_presentations'] is None:
    return None
...
if n_common < 2:
    return None, None
...
if outcome == -1:
    continue
...
run_binned[np.isnan(run_interp)] = 2
...
except KeyError:
    data['has_pupil'] = False
...
else:
    pupil_binned = np.full(n_common, 2, dtype=np.int64)
...
except Exception as e:
    print(f'  ERROR: {e}')
    ...
    continue
```

iii. The notes explicitly justify the middle-bin treatment for missing pupil data and describe the rest as sanity/robustness checks to prevent bad files from crashing the whole conversion.

## 9-a. What are the most time-consuming steps of the code?

i. The AI's code spends most of its time in two places: a full-dataset pre-pass (`collect_global_stats`) that reopens every NWB file to compute image lists and percentile edges, and the per-trial common-bin resampling loop inside `process_experiment`.

ii.
```python
global_stats = collect_global_stats(expt_table)
...
for i, expt_id in enumerate(expt_ids):
    ...
    with h5py.File(fname, 'r') as f:
        ...
for trial_idx in include_indices:
    ...
    trial_neural, common_ts = resample_events_to_common_bins(...)
```

iii. The notes' runtime tables and trajectory both describe a scan phase followed by a heavier conversion phase. The trajectory also discusses file size and runtime as major concerns after the first full pass.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the inner loop over common bins in `resample_events_to_common_bins`, the per-timepoint loop in `get_image_at_timepoints`, and the repeated trial loop that performs interpolation and stacking one trial at a time.

ii.
```python
for b in range(n_common):
    mask = bin_assignments == b
    if mask.any():
        resampled[:, b] = trial_events[:, mask].sum(axis=1)
...
for t in range(n_tp):
    idx = indices[t]
    ...
for trial_idx in include_indices:
    ...
```

iii. The notes claim the event binning was "vectorized for speed," but the implementation still contains these explicit Python loops, so the code itself points to the likely remaining optimization opportunities.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats dataset loading. It first opens every NWB file in `collect_global_stats` to gather running/pupil samples and image names, then reopens every NWB file again in `process_experiment` to do the actual conversion.

ii.
```python
def collect_global_stats(expt_table):
    for i, expt_id in enumerate(expt_ids):
        with h5py.File(fname, 'r') as f:
            ...

def process_experiment(expt_id, expt_info, ...):
    raw = load_experiment_data(expt_id)
```

iii. This repeated full-file pass is visible directly in the code. The notes frame it as a practical way to compute global percentile bins and global image codes before final assembly.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads several raw fields that are never used downstream (`cell_specimen_ids`, `aborted`, `auto_rewarded`, `change_time`, `change_image_name`, `initial_image_name`, `is_change` at the trial level), and it optionally creates plotting artifacts that are not part of the converted dataset. It also scans raw running/pupil streams globally even though downstream outputs only use discretized per-trial values.

ii.
```python
data['cell_specimen_ids'] = f['processing/ophys/image_segmentation/cell_specimen_table/cell_specimen_id'][:]
...
for key in ['go', 'catch', 'aborted', 'auto_rewarded', 'hit', 'miss',
            'correct_reject', 'false_alarm', 'start_time', 'stop_time',
            'change_time', 'change_image_name', 'initial_image_name', 'is_change']:
    ...
if args.show_processing:
    create_processing_plots(...)
```

iii. There is no explicit defense for these extra loads besides development convenience. The notes present the plots and the global scan as validation/sanity-check machinery rather than required output construction.
