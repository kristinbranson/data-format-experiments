# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache. It reads the local metadata CSV `ophys_experiment_table.csv`, discovers downloaded `.nwb` files in `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, filters to `passive == False`, and then loads each experiment directly with `h5py`. Trials are then read from each experiment’s `intervals/trials` group.

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
```

```python
def load_experiment_data(expt_id):
    fname = os.path.join(EXPT_DIR, f'behavior_ophys_experiment_{expt_id}.nwb')
    f = h5py.File(fname, 'r')
    ...
    trials = f['intervals/trials']
```

iii. In `CONVERSION_NOTES.md`, the AI justifies this as “Efficient NWB loading with h5py (no SDK overhead)” and repeatedly states in the trajectory that direct NWB access uses “the same data source” as the SDK while avoiding cache/SDK overhead.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from experiment metadata and then stored from each NWB file’s `general/subject/subject_id`. The output `subjects` list contains unique stringified subject IDs, and each processed experiment/session appends an index into that list.

ii.
```python
data['subject_id'] = f['general/subject/subject_id'][()]
if isinstance(data['subject_id'], bytes):
    data['subject_id'] = data['subject_id'].decode()
```

```python
subj = str(result['subject_id'])
if subj not in unique_subjects:
    unique_subjects.append(subj)
subject_idx_list.append(unique_subjects.index(subj))
```

iii. The notes and trajectory treat NWB `subject_id` and metadata `mouse_id` as the mouse identifier, and use unique subjects as the natural split for mice.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` / NWB file as one output session. It does not group multiple imaging planes that share an `ophys_session_id`; MESO multi-plane sessions are therefore split into separate sessions.

ii.
```python
for i, (_, expt_info) in enumerate(expt_table.iterrows()):
    expt_id = expt_info['ophys_experiment_id']
    ...
    result = process_experiment(expt_id, expt_info, ...)
    ...
    all_neural.append(result['neural'])
```

iii. In the trajectory, the AI explicitly decided “Each experiment (NWB file) = one session in output” and argued that each experiment has a different neuron set even when MESO experiments share behavioral data.

## 1-d. How are the data split into trials?

i. Trials are read from the NWB `intervals/trials` table. For each included trial, the AI uses the full window from `start_time` to `stop_time`, creates a common timestamp grid for that interval, and extracts/resamples all outputs and neural data within that window.

ii.
```python
trial_data = raw['trials']
include_mask = trial_data['go'] | trial_data['catch']
include_indices = np.where(include_mask)[0]
...
for trial_idx in include_indices:
    trial_start = trial_data['start_time'][trial_idx]
    trial_stop = trial_data['stop_time'][trial_idx]
    trial_neural, common_ts = resample_events_to_common_bins(
        events, ophys_ts, trial_start, trial_stop
    )
```

iii. The notes say “Trial segmentation: trial start_time to stop_time from NWB,” and the trajectory says the trial should span the entire trial rather than a smaller event-centered window.

## 1-e. How are trials filtered based on quality controls?

i. The AI includes trials where `go` or `catch` is true, requires at least two included trials per experiment, skips trials whose rebinned window has fewer than two bins, and skips trials with unrecognized outcomes (`outcome == -1`). It also skips entire experiments with no valid neurons or no stimulus presentations.

ii.
```python
include_mask = trial_data['go'] | trial_data['catch']
include_indices = np.where(include_mask)[0]

if len(include_indices) < 2:
    print(f'  WARNING: Too few trials in experiment {expt_id}')
    return None
```

```python
if n_common < 2:
    return None, None
...
outcome = get_trial_outcome(trial_data, trial_idx)
if outcome == -1:
    continue
```

iii. In `CONVERSION_NOTES.md`, the AI describes this as “Include Go and Catch trials, exclude Aborted and Auto-rewarded,” but the code implements that indirectly through `go | catch` plus outcome filtering. The notes also justify a minimum-two-trial rule for decoder compatibility.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the raw event-detection matrix in the NWB file, specifically `processing/ophys/event_detection/data`, then transposes it to neuron-by-time.

ii.
```python
data['events'] = f['processing/ophys/event_detection/data'][:].T
```

iii. The notes justify this by citing the paper’s use of “detected calcium events” and arguing that raw events, not dF/F, match the paper better. The trajectory also states that the SDK’s half-Gaussian event filtering is “for visualization only.”

## 2-b. How is the `neural` data processed?

i. The AI filters to valid ROIs, then rebins each trial’s event trains onto a global 93.23 ms grid and sums all native ophys frames that fall into each common bin. No further normalization is applied.

ii.
```python
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
```

```python
def resample_events_to_common_bins(events, ophys_ts, trial_start, trial_stop):
    common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
    ...
    bin_assignments = np.digitize(trial_ophys_ts, bin_edges) - 1
    ...
    for b in range(n_common):
        mask = bin_assignments == b
        if mask.any():
            resampled[:, b] = trial_events[:, mask].sum(axis=1)
```

iii. `CONVERSION_NOTES.md` states the key decisions as “Neural data: Raw detected events (not filtered), summed within each common bin” and “Common time bin: 93.23ms (MESO rate).” The trajectory shows the AI changed its implementation specifically to downsample CAM2P experiments to the MESO rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC filter is `valid_roi`; neurons with `valid_roi == False` are excluded before any trial extraction.

ii.
```python
data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
...
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
```

iii. The notes justify this as matching the Allen SDK default `exclude_invalid_rois=True`, and describe it as reproducing the SDK’s ROI curation when loading directly from NWB.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data are aligned to trial start. Each trial uses a common timestamp vector beginning at `start_time` and ending before `stop_time`, and the metadata records the temporal alignment event as “Trial start time.”

ii.
```python
common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
```

```python
'metadata': {
    ...
    'temporal_alignment_event': 'Trial start time',
    'off_start': 0.0,
    'off_end': None,
```

iii. The notes and trajectory both frame the trial as the natural alignment unit, using the full start-to-stop trial window rather than alignment to `change_time`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 93.23 ms bin size (`COMMON_DT = 0.09323` seconds, about 10.73 Hz) for every experiment and trial. Native ophys events are rebinned by summing into these bins.

ii.
```python
COMMON_DT = 0.09323  # seconds, ~10.73 Hz
```

```python
'metadata': {
    ...
    'time_bin_size': COMMON_DT * 1000,
```

iii. The trajectory shows the AI first noticed mixed native frame rates (CAM2P about 31 Hz, MESO about 10.7 Hz), then decided that because “time bins should be the same size for all trials and sessions,” it should resample everything to the lower MESO rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation table in the NWB file, specifically `intervals/<Natural_Images...>/image_name` and `start_time`.

ii.
```python
stim_keys = [k for k in f['intervals'].keys() if 'Natural_Images' in k]
...
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    ...
data['stimulus_presentations'] = stim_data
```

```python
start_times = stim_data['start_time']
image_names = stim_data['image_name']
indices = np.searchsorted(start_times, common_ts, side='right') - 1
```

iii. The notes say “Image identity during gray screen: Assign most recently shown non-omitted image,” which explains why the AI preferred the stimulus-presentation stream over only using trial table fields.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI constructs a global sorted image vocabulary across experiments, maps names to integers, and for each time bin assigns the most recent non-omitted image. If the current stimulus is `omitted`, it carries forward the last valid image label.

ii.
```python
all_images = sorted(all_images)
image_to_idx = {img: i for i, img in enumerate(all_image_names)}
```

```python
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

iii. The notes explicitly justify the gray-screen handling as using “the most recently shown non-omitted image,” and the trajectory says this is meant to satisfy the instruction “image presented during the non-grey screen.”

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated on the same `common_ts` vector used for rebinned neural data within each trial, so it is aligned bin-for-bin with the neural matrix.

ii.
```python
trial_neural, common_ts = resample_events_to_common_bins(
    events, ophys_ts, trial_start, trial_stop
)
...
img_id = get_image_at_timepoints(common_ts, raw['stimulus_presentations'], image_to_idx)
```

iii. The notes repeatedly state that all streams are aligned to the common timestamps after rebinned neural data are created.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change`, `start_time`, and `stop_time` fields, not from the trial table’s `change_time`.

ii.
```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    ...
```

```python
is_change = stim_data.get('is_change', np.zeros(len(stim_data['start_time'])))
start_times = stim_data['start_time']
stop_times = stim_data['stop_time']
```

iii. The notes map `stimulus_presentations.is_change` directly to the image-change output and describe it as a time-varying binary series.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For every stimulus presentation marked as a change, the AI marks rebinned trial time bins as 1 from that stimulus’s `start_time` until its `stop_time`; all other bins are 0.

ii.
```python
change_indices = np.where(np.array(is_change) == 1)[0]
for ci in change_indices:
    mask = (common_ts >= start_times[ci]) & (common_ts < stop_times[ci])
    change[mask] = 1
```

iii. The notes describe this as “1 during change stimulus,” and the README repeats “1 during change stimulus presentation.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. No threshold beyond the existing binary indicator is applied. The AI directly stores 0/1 values and declares the categories `['no_change', 'change']`.

ii.
```python
change = np.zeros(n_tp, dtype=np.int64)
...
change[mask] = 1
```

```python
output_values = [
    all_image_names,
    ['no_change', 'change'],
    ...
]
```

iii. The AI treats image change as already categorical in the raw data and therefore does not add any extra thresholding step.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed on the same per-trial `common_ts` grid used for the rebinned neural data, so alignment is by shared time bins within each trial.

ii.
```python
trial_neural, common_ts = resample_events_to_common_bins(...)
...
img_change = get_image_change_at_timepoints(common_ts, raw['stimulus_presentations'])
```

iii. The notes and trajectory consistently justify a single common time base for neural and behavioral variables.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the NWB running-processing stream: `processing/running/speed/data` and its timestamps.

ii.
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. The notes describe this as the standard running signal from the Visual Behavior dataset.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global running-speed percentile edges from a subsample of up to 2000 non-NaN points per experiment, then linearly interpolates each trial’s running signal onto `common_ts` and discretizes it with those edges.

ii.
```python
run_data = f['processing/running/speed/data'][:]
valid_run = run_data[~np.isnan(run_data)]
if len(valid_run) > 2000:
    rng = np.random.RandomState(int(expt_id) % (2**31))
    valid_run = rng.choice(valid_run, 2000, replace=False)
all_running.append(valid_run)
...
running_percentiles = np.percentile(all_running, np.linspace(0, 100, 6))
```

```python
run_interp = interpolate_to_common(
    raw['running_speed'], raw['running_timestamps'], common_ts
)
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
```

iii. The notes justify global percentile binning and common-time interpolation; the trajectory also mentions sampling for global statistics as a performance optimization.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is binned into five percentile bins using global edges. Any NaN value after interpolation is assigned to the middle bin `2`.

ii.
```python
running_percentiles = np.percentile(all_running, np.linspace(0, 100, 6))
...
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
run_binned[np.isnan(run_interp)] = 2
```

iii. `CONVERSION_NOTES.md` explicitly says “Running speed bins: 5 equal percentile” and, for missing behavioral data, uses a fill strategy rather than dropping bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is linearly interpolated directly onto the same `common_ts` used for each trial’s rebinned neural data.

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

iii. The AI’s general alignment strategy is to make every output live on the common per-trial time base after neural resampling.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI uses pupil area, not pupil width/diameter: `acquisition/EyeTracking/pupil_tracking/area` plus timestamps.

ii.
```python
data['pupil_area'] = f['acquisition/EyeTracking/pupil_tracking/area'][:]
data['pupil_timestamps'] = f['acquisition/EyeTracking/pupil_tracking/timestamps'][:]
```

iii. The notes identify the source variable as `pupil_tracking.area` and map it to the target `pupil_diameter` output without any conversion from area to diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI computes global pupil percentile edges from sampled raw pupil-area values, linearly interpolates pupil area onto `common_ts`, and digitizes it into five bins. It does not remove blink frames.

ii.
```python
valid_pupil = pupil_data[~np.isnan(pupil_data)]
if len(valid_pupil) > 2000:
    rng = np.random.RandomState(int(expt_id) % (2**31))
    valid_pupil = rng.choice(valid_pupil, 2000, replace=False)
all_pupil.append(valid_pupil)
...
pupil_percentiles = np.percentile(all_pupil, np.linspace(0, 100, 6))
```

```python
if raw['has_pupil']:
    pupil_interp = interpolate_to_common(
        raw['pupil_area'], raw['pupil_timestamps'], common_ts
    )
...
pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
```

iii. The notes justify percentile binning globally and state that missing pupil data are filled rather than excluded. No separate blink-removal justification appears in the notes because the implementation does not attempt it.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil is digitized into five percentile bins using global edges. Missing or NaN pupil values are assigned to the middle bin `2`; if an experiment has no pupil stream, the entire pupil output is filled with `2`.

ii.
```python
if pupil_percentiles is not None:
    pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
    pupil_binned = np.clip(pupil_binned, 0, 4).astype(np.int64)
    pupil_binned[np.isnan(pupil_interp)] = 2
else:
    pupil_binned = np.full(n_common, 2, dtype=np.int64)
```

iii. The notes explicitly say “Missing pupil data: Fill with NaN, assign to middle bin (bin 2).”

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are interpolated onto the same per-trial `common_ts` used for the rebinned neural data.

ii.
```python
pupil_interp = interpolate_to_common(
    raw['pupil_area'], raw['pupil_timestamps'], common_ts
)
```

iii. As with running speed and image variables, the justification is a single common neural/behavioral time base after resampling.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `correct_reject`, and `false_alarm`.

ii.
```python
for key in ['go', 'catch', 'aborted', 'auto_rewarded', 'hit', 'miss',
            'correct_reject', 'false_alarm', 'start_time', 'stop_time',
            'change_time', 'change_image_name', 'initial_image_name', 'is_change']:
    ...
```

```python
def get_trial_outcome(trial_data, trial_idx):
    if trial_data['hit'][trial_idx]: return 0
    elif trial_data['miss'][trial_idx]: return 1
    elif trial_data['correct_reject'][trial_idx]: return 2
    elif trial_data['false_alarm'][trial_idx]: return 3
    else: return -1
```

iii. The notes map `trials.hit/miss/CR/FA` directly to the trial-outcome output and describe it as a four-category static target.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four outcome booleans to integer codes 0 to 3, skips trials with no recognized outcome, and then broadcasts the chosen code across all time bins in that trial.

ii.
```python
outcome = get_trial_outcome(trial_data, trial_idx)
if outcome == -1:
    continue
...
np.full(n_common, outcome, dtype=np.int64)
```

iii. The notes justify this as a static per-trial variable that still needs to fit the decoder’s time-varying output array shape.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases by skipping or filling: experiments with no valid neurons or no stimulus table are skipped; experiments with too few included trials are skipped; very short rebinned trials are skipped; unrecognized outcomes are skipped; missing pupil streams become all-`2`; and NaN running/pupil samples after interpolation are assigned to bin `2`.

ii.
```python
if n_neurons == 0:
    ...
if raw['stimulus_presentations'] is None:
    ...
if len(include_indices) < 2:
    ...
```

```python
if n_common < 2:
    return None, None
...
run_binned[np.isnan(run_interp)] = 2
...
pupil_binned[np.isnan(pupil_interp)] = 2
...
else:
    pupil_binned = np.full(n_common, 2, dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` explicitly justifies the middle-bin fill for missing pupil values. The trajectory also emphasizes robustness: a single bad experiment should not halt the full conversion.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive steps are repeatedly opening and scanning NWB files, computing global running/pupil/image statistics across all experiments, and the per-trial neural resampling loop that sums frames into common bins.

ii.
```python
def collect_global_stats(expt_table):
    ...
    for i, expt_id in enumerate(expt_ids):
        ...
        with h5py.File(fname, 'r') as f:
            ...
```

```python
for trial_idx in include_indices:
    ...
    trial_neural, common_ts = resample_events_to_common_bins(...)
```

```python
for b in range(n_common):
    mask = bin_assignments == b
    if mask.any():
        resampled[:, b] = trial_events[:, mask].sum(axis=1)
```

iii. The notes estimate “Load NWB” and “Resample + process” as the main per-session costs, and the trajectory explicitly flags the resampling loop as something that would otherwise be “extremely slow” without vectorization improvements.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves several loops that could be vectorized further: the loop over common bins in `resample_events_to_common_bins`, the per-timepoint loop in `get_image_at_timepoints`, the loop over change presentations in `get_image_change_at_timepoints`, and the outer per-trial processing loop.

ii.
```python
for b in range(n_common):
    mask = bin_assignments == b
    if mask.any():
        resampled[:, b] = trial_events[:, mask].sum(axis=1)
```

```python
for t in range(n_tp):
    idx = indices[t]
    ...
```

```python
for ci in change_indices:
    mask = (common_ts >= start_times[ci]) & (common_ts < stop_times[ci])
    change[mask] = 1
```

iii. The trajectory itself identifies resampling as a performance concern and says the initial loop-based version needed to be “optimized” because it would be too slow.

## 9-c. What processing does the code repeat multiple times?

i. The code makes a separate full-dataset scan in `collect_global_stats` and then opens each NWB file again for actual conversion. Within conversion, it also recomputes image identity and image change from the full session stimulus table separately for each trial.

ii.
```python
global_stats = collect_global_stats(expt_table)
...
for i, (_, expt_info) in enumerate(expt_table.iterrows()):
    ...
    result = process_experiment(...)
```

```python
img_id = get_image_at_timepoints(common_ts, raw['stimulus_presentations'], image_to_idx)
img_change = get_image_change_at_timepoints(common_ts, raw['stimulus_presentations'])
```

iii. The notes justify the extra global pass as necessary to get global percentile edges and image vocabularies before assembling the final output.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads fields that are never used downstream, such as `cell_specimen_ids`, and optionally generates diagnostic processing plots that are not part of `converted_data.pkl`. It also scans all stimulus image names globally even though the final decoder only needs the integer-coded outputs.

ii.
```python
data['cell_specimen_ids'] = f['processing/ophys/image_segmentation/cell_specimen_table/cell_specimen_id'][:]
```

```python
if args.show_processing:
    create_processing_plots(...)
```

```python
stim_keys = [k for k in f['intervals'].keys() if 'Natural_Images' in k]
...
all_images = sorted(all_images)
```

iii. The notes present the plots and several consistency scans as sanity checks rather than decoder-required processing, so these are best understood as validation overhead rather than essential conversion work.
