# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads data directly from the local NWB files and metadata CSVs, not through the Allen SDK cache used by the reference solution. It reads `ophys_experiment_table.csv`, keeps only experiment IDs that have downloaded `.nwb` files, excludes passive experiments, and then opens each NWB with `h5py`.

ii.
```python
expt_table = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
downloaded_ids = set()
for f in os.listdir(EXPT_DIR):
    if f.endswith('.nwb'):
        eid = int(f.split('_')[-1].replace('.nwb', ''))
        downloaded_ids.add(eid)
expt_table = expt_table[expt_table.ophys_experiment_id.isin(downloaded_ids)]
expt_table = expt_table[expt_table.passive == False]
...
fname = os.path.join(EXPT_DIR, f'behavior_ophys_experiment_{expt_id}.nwb')
f = h5py.File(fname, 'r')
```

iii. `CONVERSION_NOTES.md` says the local package contains downloaded NWB files plus project metadata, and says passive sessions were excluded because they are not part of the active behavioral task. The trajectory shows the agent switched to direct NWB loading after deciding the local data layout was easier to access that way than via the SDK cache.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the NWB `general/subject/subject_id` field. During assembly, the agent builds `unique_subjects` and stores one subject index per processed experiment.

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

iii. The agent did not give a separate long justification here. The notes and trajectory consistently treat `subject_id` / mouse identity as the subject split.

## 1-c. How are the data split into sessions?

i. The agent effectively treats each NWB experiment file as one output session. It iterates rows of the experiment table and appends one element to `all_neural`, `all_input`, and `all_output` for each experiment, without grouping experiments that share the same `ophys_session_id`.

ii.
```python
for i, (_, expt_info) in enumerate(expt_table.iterrows()):
    expt_id = expt_info['ophys_experiment_id']
    ...
    all_neural.append(result['neural'])
    all_input.append([np.zeros((0, t.shape[1]), dtype=np.float32) for t in result['neural']])
    all_output.append(result['output'])
```

iii. The trajectory explicitly lists “Each experiment (NWB file) = one session in output” as a design decision. `CONVERSION_NOTES.md` also emphasizes that each NWB is one experiment / imaging plane and that CAM2P and MESO have different experiment-to-session relationships, but the code never reconstructs multi-plane sessions.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table and segmented using each trial’s `start_time` and `stop_time`. The agent includes all `go` and `catch` trials for each experiment and creates a per-trial common timestamp grid with fixed-width bins between those two times.

ii.
```python
trial_data = raw['trials']
include_mask = trial_data['go'] | trial_data['catch']
include_indices = np.where(include_mask)[0]
...
trial_start = trial_data['start_time'][trial_idx]
trial_stop = trial_data['stop_time'][trial_idx]
trial_neural, common_ts = resample_events_to_common_bins(
    events, ophys_ts, trial_start, trial_stop
)
```

iii. In the trajectory, the agent states “Trial window: trial start_time to stop_time” and “Filter: Go + Catch trials only.” The notes describe trial curation as including Go and Catch while excluding Aborted and Auto-rewarded.

## 1-e. How are trials filtered based on quality controls?

i. The main trial filter is `go | catch`, plus a requirement that an experiment have at least two included trials. Trials that produce fewer than two common bins are skipped, and trials with unrecognized outcomes are skipped. There is no explicit extra filter on `aborted`, `auto_rewarded`, or `change_time` in code.

ii.
```python
include_mask = trial_data['go'] | trial_data['catch']
include_indices = np.where(include_mask)[0]
...
if len(include_indices) < 2:
    print(f'  WARNING: Too few trials in experiment {expt_id}')
    return None
...
if trial_neural is None:
    continue
...
outcome = get_trial_outcome(trial_data, trial_idx)
if outcome == -1:
    continue
...
if len(neural_trials) < 2:
    print(f'  WARNING: Too few valid trials in experiment {expt_id}')
    return None
```

iii. `CONVERSION_NOTES.md` says the intended curation was “Include Go and Catch trials, exclude Aborted and Auto-rewarded.” The trajectory frames `go|catch` as the operational implementation of that rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from event-detection output, not from dF/F. Specifically, the agent reads `processing/ophys/event_detection/data` and then filters rows by `valid_roi`.

ii.
```python
data['events'] = f['processing/ophys/event_detection/data'][:].T
data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
...
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
```

iii. The notes repeatedly justify this by saying the paper used “detected calcium events (not raw dff)” and that the output should therefore use raw detected events.

## 2-b. How is the `neural` data processed?

i. After filtering to valid ROIs, the agent resamples neural events into a fixed common bin width of 93.23 ms and sums all event values whose native ophys timestamps fall into each common bin. This is done separately for each trial.

ii.
```python
COMMON_DT = 0.09323  # seconds, ~10.73 Hz
...
common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
...
bin_assignments = np.digitize(trial_ophys_ts, bin_edges) - 1
...
for b in range(n_common):
    mask = bin_assignments == b
    if mask.any():
        resampled[:, b] = trial_events[:, mask].sum(axis=1)
```

iii. `CONVERSION_NOTES.md` lists as key decisions: “Common time bin: 93.23ms (MESO rate)” and “Neural data: Raw detected events (not filtered), summed within each common bin.” The trajectory explains that the agent chose a common frame rate because it interpreted the instruction “time bins should be the same size for all trials and sessions” literally.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC is ROI validity filtering using the `valid_roi` flag from the cell specimen table. Experiments with zero valid neurons are dropped entirely.

ii.
```python
data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
...
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
...
if n_neurons == 0:
    print(f'  WARNING: No valid neurons in experiment {expt_id}')
    return None
```

iii. The notes justify this as matching the Allen SDK default behavior: “Valid ROI filtering: Exclude invalid ROIs matching SDK default.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data are aligned to trial start. For each included trial, the agent builds `common_ts = np.arange(trial_start, trial_stop, COMMON_DT)` and bins neural events into that grid.

ii.
```python
trial_start = trial_data['start_time'][trial_idx]
trial_stop = trial_data['stop_time'][trial_idx]
trial_neural, common_ts = resample_events_to_common_bins(
    events, ophys_ts, trial_start, trial_stop
)
```

iii. The trajectory explicitly calls the alignment choice “trial start_time to stop_time.” The agent’s notes also say the fixed binning was imposed to keep a common time bin across all trials and sessions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 93.23 ms time bin for every trial. Yes, temporal rebinning is applied: native ophys samples are collapsed into this common timebase.

ii.
```python
COMMON_DT = 0.09323  # seconds, ~10.73 Hz
...
'time_bin_size': COMMON_DT * 1000,
```

iii. `CONVERSION_NOTES.md` says the common time bin was chosen to match the MESO rate and satisfy the requirement that all sessions share the same bin size. The trajectory shows the agent noticed mixed native frame rates and deliberately standardized them.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations`, specifically each stimulus presentation’s `start_time` and `image_name`. It is not taken from the trial table’s `initial_image_name` / `change_image_name` columns.

ii.
```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    ...
    stim_data[key] = vals
...
start_times = stim_data['start_time']
image_names = stim_data['image_name']
```

iii. The notes justify this by stating that image identity should match stimulus presentations directly, and by adopting the rule “Image identity during gray screen: Assign most recently shown non-omitted image.”

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent first builds a global mapping from all non-omitted image names to integer IDs. Within each trial, for each common-bin timestamp it finds the most recent stimulus presentation and uses that image. If the image is `omitted` or the timestamp precedes the first stimulus, it carries forward the last valid image.

ii.
```python
indices = np.searchsorted(start_times, common_ts, side='right') - 1
...
if idx < 0:
    image_id[t] = last_valid
    continue
img = image_names[idx]
if img == 'omitted':
    image_id[t] = last_valid
elif img in image_to_idx:
    image_id[t] = image_to_idx[img]
    last_valid = image_to_idx[img]
...
all_images = sorted(all_images)
image_to_idx = {img: i for i, img in enumerate(all_image_names)}
```

iii. `CONVERSION_NOTES.md` explicitly records two key decisions here: use a global categorical mapping, and during gray / omitted periods “assign most recently shown non-omitted image.”

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated on the same `common_ts` grid used for the rebinned neural data, so each time bin in `img_id` corresponds to one neural time bin in the same trial.

ii.
```python
trial_neural, common_ts = resample_events_to_common_bins(
    events, ophys_ts, trial_start, trial_stop
)
...
img_id = get_image_at_timepoints(common_ts, raw['stimulus_presentations'], image_to_idx)
...
trial_output = np.stack([
    img_id, img_change, run_binned, pupil_binned,
    np.full(n_common, outcome, dtype=np.int64)
], axis=0)
```

iii. The notes say the processing plots verified alignment of “neural events, image identity, running speed, and pupil data,” and the trajectory repeatedly frames all outputs as being aligned to the common ophys-derived time grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table, using `is_change` together with each presentation’s `start_time` and `stop_time`.

ii.
```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    ...
is_change = stim_data.get('is_change', np.zeros(len(stim_data['start_time'])))
start_times = stim_data['start_time']
stop_times = stim_data['stop_time']
```

iii. The notes map `stimulus_presentations.is_change` directly to the image-change decoder output and describe it as “1 during change stimulus.”

## 4-b. What processing is involved in computing `output` *Image change*?

i. For every stimulus presentation marked as a change, the agent marks all common-bin timestamps inside that presentation’s `[start_time, stop_time)` interval as 1. All other bins are 0.

ii.
```python
change = np.zeros(n_tp, dtype=np.int64)
...
change_indices = np.where(np.array(is_change) == 1)[0]
for ci in change_indices:
    mask = (common_ts >= start_times[ci]) & (common_ts < stop_times[ci])
    change[mask] = 1
```

iii. `CONVERSION_NOTES.md` says the planned transform for `is_change` was “Binary time series” and, more specifically, “1 during change stimulus.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding beyond binary categorization is applied. The output is encoded as 0 for no change and 1 for change.

ii.
```python
change = np.zeros(n_tp, dtype=np.int64)
...
change[mask] = 1
...
['no_change', 'change'],
```

iii. The task already asked for a binary variable, so the agent treated the raw indicator as the final two-category output. No additional justification was given beyond the task requirement and the notes’ “Binary time series.”

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed on the same `common_ts` grid as the rebinned neural data and stacked directly into the trial output matrix for that trial.

ii.
```python
trial_neural, common_ts = resample_events_to_common_bins(
    events, ophys_ts, trial_start, trial_stop
)
img_change = get_image_change_at_timepoints(common_ts, raw['stimulus_presentations'])
...
trial_output = np.stack([
    img_id, img_change, run_binned, pupil_binned,
    np.full(n_common, outcome, dtype=np.int64)
], axis=0)
```

iii. The notes say the agent generated processing plots to verify alignment across neural and behavioral variables; the trajectory likewise treats all outputs as sharing the common per-trial time grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is taken directly from the NWB running module: `processing/running/speed/data` and `processing/running/speed/timestamps`.

ii.
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. The notes map NWB running-speed data directly onto the running-speed decoder output and describe the next processing step as interpolation plus percentile binning.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The agent linearly interpolates running speed from its native timestamps to `common_ts`, then bins the interpolated values into five global percentile bins. The percentile edges are computed in a separate pre-pass over experiments using sampled raw running values rather than trial-aligned values.

ii.
```python
f_interp = interpolate.interp1d(
    source_ts[valid], data[valid],
    kind='linear', bounds_error=False, fill_value=np.nan
)
...
run_interp = interpolate_to_common(
    raw['running_speed'], raw['running_timestamps'], common_ts
)
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
...
valid_run = run_data[~np.isnan(run_data)]
if len(valid_run) > 2000:
    rng = np.random.RandomState(int(expt_id) % (2**31))
    valid_run = rng.choice(valid_run, 2000, replace=False)
all_running.append(valid_run)
running_percentiles = np.percentile(all_running, np.linspace(0, 100, 6))
```

iii. `CONVERSION_NOTES.md` says “Interpolate + 5 percentile bins” and “Global percentiles.” The notes also say sampling was used while computing global statistics for efficiency.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into five bins using global percentile edges. `np.digitize` assigns bins 0 through 4, values are clipped into that range, and NaNs are forced into the middle bin `2`.

ii.
```python
running_percentiles = np.percentile(all_running, np.linspace(0, 100, 6))
...
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
run_binned[np.isnan(run_interp)] = 2
```

iii. The notes explicitly record “Running speed bins: 5 equal percentile” and “Missing pupil data: Fill with NaN, assign to middle bin (bin 2).” The same middle-bin policy is used for running-speed NaNs in code.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation directly onto each trial’s `common_ts`, which is the same time grid used for the rebinned neural events.

ii.
```python
trial_neural, common_ts = resample_events_to_common_bins(
    events, ophys_ts, trial_start, trial_stop
)
...
run_interp = interpolate_to_common(
    raw['running_speed'], raw['running_timestamps'], common_ts
)
```

iii. The notes and trajectory both describe this as aligning continuous behavioral streams to the common ophys-derived time grid after deciding to impose a shared bin size across sessions.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The agent derives this output from pupil area, not pupil width: `acquisition/EyeTracking/pupil_tracking/area` and its timestamps.

ii.
```python
data['pupil_area'] = f['acquisition/EyeTracking/pupil_tracking/area'][:]
data['pupil_timestamps'] = f['acquisition/EyeTracking/pupil_tracking/timestamps'][:]
```

iii. The notes describe the mapping as `pupil_tracking.area -> output[3] (pupil_diameter)` and do not mention using pupil width or blink filtering.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. If pupil data exist, the agent linearly interpolates raw pupil area onto `common_ts` and bins the interpolated values into five global percentile bins. If no pupil data exist, it fills the trial with NaNs and later assigns the middle category.

ii.
```python
if raw['has_pupil']:
    pupil_interp = interpolate_to_common(
        raw['pupil_area'], raw['pupil_timestamps'], common_ts
    )
else:
    pupil_interp = np.full(n_common, np.nan)
...
pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
pupil_binned = np.clip(pupil_binned, 0, 4).astype(np.int64)
pupil_binned[np.isnan(pupil_interp)] = 2
```

iii. `CONVERSION_NOTES.md` says pupil should be handled by interpolation plus five percentile bins, and says missing pupil values should be assigned to the middle bin.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil area is discretized into five percentile bins using globally computed percentile edges. `np.digitize` and `np.clip` create categories 0 through 4, and NaNs are assigned to category 2. If no global pupil percentiles are available, the whole trial is filled with category 2.

ii.
```python
if pupil_percentiles is not None:
    pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
    pupil_binned = np.clip(pupil_binned, 0, 4).astype(np.int64)
    pupil_binned[np.isnan(pupil_interp)] = 2
else:
    pupil_binned = np.full(n_common, 2, dtype=np.int64)
```

iii. The notes explicitly justify the missing-data policy as “Fill with NaN, assign to middle bin (bin 2).”

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are interpolated directly onto each trial’s `common_ts`, so they are aligned to the same per-trial bins as the rebinned neural events.

ii.
```python
trial_neural, common_ts = resample_events_to_common_bins(
    events, ophys_ts, trial_start, trial_stop
)
...
pupil_interp = interpolate_to_common(
    raw['pupil_area'], raw['pupil_timestamps'], common_ts
)
```

iii. As with running speed, the notes justify alignment by saying all outputs were put onto the common time grid used for neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the four boolean trial columns `hit`, `miss`, `correct_reject`, and `false_alarm`.

ii.
```python
def get_trial_outcome(trial_data, trial_idx):
    if trial_data['hit'][trial_idx]: return 0
    elif trial_data['miss'][trial_idx]: return 1
    elif trial_data['correct_reject'][trial_idx]: return 2
    elif trial_data['false_alarm'][trial_idx]: return 3
    else: return -1
```

iii. The notes map those four trial labels directly to the decoder output for trial outcome.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The agent converts the mutually exclusive boolean outcome columns into one integer code per trial, then repeats that code across every time bin of the trial so that `trial_outcome` becomes a time series constant within trial.

ii.
```python
outcome = get_trial_outcome(trial_data, trial_idx)
if outcome == -1:
    continue
...
trial_output = np.stack([
    img_id, img_change, run_binned, pupil_binned,
    np.full(n_common, outcome, dtype=np.int64)
], axis=0)
...
['hit', 'miss', 'correct_reject', 'false_alarm'],
```

iii. The notes describe trial outcome as a “Categorical static” output; the code implements that by broadcasting one trial-level code across all bins in the trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several failure cases conservatively: experiments with missing stimuli or no valid neurons are skipped, processing exceptions are caught and skipped, experiments with too few usable trials are dropped, missing pupil data are converted to NaN then assigned to the middle bin, and unknown outcome trials are skipped. It does not add special repair logic beyond skipping or bin-filling.

ii.
```python
except KeyError:
    data['has_pupil'] = False
    data['pupil_area'] = None
    data['pupil_timestamps'] = None
...
if n_neurons == 0:
    return None
if raw['stimulus_presentations'] is None:
    return None
...
if trial_neural is None:
    continue
...
pupil_binned[np.isnan(pupil_interp)] = 2
...
if outcome == -1:
    continue
...
except Exception as e:
    print(f'  ERROR: {e}')
    import traceback; traceback.print_exc()
    continue
```

iii. `CONVERSION_NOTES.md` explicitly justifies the middle-bin strategy for missing pupil data. The trajectory also shows the agent prioritized robustness by skipping bad experiments rather than stopping the full conversion.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive parts are repeated NWB I/O and per-trial resampling. The code scans every experiment once in `collect_global_stats`, then opens each experiment again in `process_experiment`, and inside each trial it loops over common bins while summing events.

ii.
```python
for i, expt_id in enumerate(expt_ids):
    ...
    with h5py.File(fname, 'r') as f:
        ...
for i, (_, expt_info) in enumerate(expt_table.iterrows()):
    ...
    result = process_experiment(...)
...
for b in range(n_common):
    mask = bin_assignments == b
    if mask.any():
        resampled[:, b] = trial_events[:, mask].sum(axis=1)
```

iii. The notes estimate full processing time step-by-step and describe both NWB loading and resampling as the core runtime costs. The trajectory also shows the agent explicitly worrying about resampling speed and trying to optimize it.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Two obvious hot loops remain: the loop over common bins in `resample_events_to_common_bins`, and the loop over timepoints in `get_image_at_timepoints`. There is also a nested experiment/trial structure that repeats simple operations one trial at a time.

ii.
```python
for b in range(n_common):
    mask = bin_assignments == b
    if mask.any():
        resampled[:, b] = trial_events[:, mask].sum(axis=1)
...
for t in range(n_tp):
    idx = indices[t]
    if idx < 0:
        image_id[t] = last_valid
        continue
    ...
```

iii. The trajectory says the agent recognized the original resampling approach would be slow and attempted to make it “vectorized for speed,” but the final implementation still keeps explicit Python loops in these sections.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats a full read pass over all experiments to compute global running/pupil percentiles and image-name vocabularies, then opens the same NWB files again to do actual conversion. It also recomputes per-trial interpolation and image lookup independently for every trial after already scanning the same raw streams at experiment level.

ii.
```python
global_stats = collect_global_stats(expt_table)
...
for i, expt_id in enumerate(expt_ids):
    with h5py.File(fname, 'r') as f:
        ...
for i, (_, expt_info) in enumerate(expt_table.iterrows()):
    ...
    result = process_experiment(...)
```

iii. The notes justify the first pass as being needed for global percentile binning and image vocabularies. The trajectory also shows this was a deliberate tradeoff to preserve consistent global categories.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads some data it never uses downstream, notably `cell_specimen_ids`. It also builds empty `input` arrays for every trial even though the task specifies no decoder inputs. More broadly, the preprocessing pass scans full raw running/pupil streams and stimulus tables only to keep summary statistics.

ii.
```python
data['cell_specimen_ids'] = f['processing/ophys/image_segmentation/cell_specimen_table/cell_specimen_id'][:]
...
all_input.append([np.zeros((0, t.shape[1]), dtype=np.float32) for t in result['neural']])
...
valid_run = run_data[~np.isnan(run_data)]
...
stim_keys = [k for k in f['intervals'].keys() if 'Natural_Images' in k]
```

iii. The notes do not explicitly defend these extra steps beyond saying the script was built to satisfy the target format and to compute global discretization/image mappings.
