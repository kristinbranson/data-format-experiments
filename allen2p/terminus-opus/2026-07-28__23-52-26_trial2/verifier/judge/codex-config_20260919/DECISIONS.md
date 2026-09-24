# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the local project metadata CSV, intersects it with the NWB files actually present, excludes rows marked passive, and directly loads each remaining experiment NWB with `h5py`. It scans every selected file once for global statistics and again for conversion.

ii.
```python
expt_table = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
expt_table = expt_table[expt_table.ophys_experiment_id.isin(downloaded_ids)]
expt_table = expt_table[expt_table.passive == False]
...
f = h5py.File(fname, 'r')
```

iii. The notes justify direct NWB access as equivalent to SDK loading but faster, using all downloaded active Visual Behavior experiments (202 of 284 downloaded files). Passive recordings were excluded because they do not contain the behavioral task.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from each NWB and accumulated into a unique string list; each output experiment receives the corresponding index.

ii.
```python
data['subject_id'] = f['general/subject/subject_id'][()]
...
if subj not in unique_subjects:
    unique_subjects.append(subj)
subject_idx_list.append(unique_subjects.index(subj))
```

iii. The notes identify 38 mice in the downloaded subset and treat the NWB `subject_id` as the animal identifier.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id`/NWB imaging plane is emitted as one target-format session. Experiments sharing an `ophys_session_id` are not combined.

ii.
```python
for i, (_, expt_info) in enumerate(expt_table.iterrows()):
    expt_id = expt_info['ophys_experiment_id']
    result = process_experiment(expt_id, expt_info, ...)
    all_neural.append(result['neural'])
```

iii. The AI explicitly notes that each NWB is one experiment/imaging plane and follows the paper's per-plane decoding convention. It reports 202 converted “sessions,” despite metadata containing 174 active `ophys_session_id` values.

## 1-d. How are the data split into trials?

i. Within each experiment, trials are selected by `go | catch`, then sliced from the NWB trial `start_time` through `stop_time`. A common timestamp vector is constructed for that variable-duration interval.

ii.
```python
include_mask = trial_data['go'] | trial_data['catch']
include_indices = np.where(include_mask)[0]
...
trial_start = trial_data['start_time'][trial_idx]
trial_stop = trial_data['stop_time'][trial_idx]
common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
```

iii. The notes say this implements the requested Go/Catch inclusion and uses the SDK/NWB trial boundaries so the complete pre- and post-change trial is retained.

## 1-e. How are trials filtered based on quality controls?

i. `go | catch` implicitly excludes aborted and auto-rewarded trials. Trials producing fewer than two common bins or an unrecognized outcome are dropped, and an entire experiment is dropped if fewer than two included/processed trials remain.

ii.
```python
include_mask = trial_data['go'] | trial_data['catch']
if len(include_indices) < 2:
    return None
...
if n_common < 2:
    return None, None
...
if outcome == -1:
    continue
```

iii. The notes state that this exactly implements “include Go and Catch; exclude Aborted and Auto-rewarded,” and records that no short trials occurred in the full run.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB detected-calcium-event array and the `valid_roi` mask, with timestamps borrowed from the dF/F series.

ii.
```python
data['events'] = f['processing/ophys/event_detection/data'][:].T
data['ophys_timestamps'] = f['processing/ophys/dff/traces/timestamps'][:]
data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
```

iii. The AI cites the paper's “detected calcium events” language and says raw detected events, rather than dF/F or visualization-smoothed events, best match the paper.

## 2-b. How is the `neural` data processed?

i. Invalid ROIs are removed. For every trial, event samples near the trial are assigned to fixed 93.23-ms bins and summed per neuron; no event smoothing or normalization is applied.

ii.
```python
events = raw['events'][valid_mask]
...
bin_assignments = np.digitize(trial_ophys_ts, bin_edges) - 1
for b in range(n_common):
    mask = bin_assignments == b
    if mask.any():
        resampled[:, b] = trial_events[:, mask].sum(axis=1)
```

iii. The notes justify raw events as the paper's analysis signal and summation as appropriate for count-like events while harmonizing CAM2P and MESO frame rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only rows with `valid_roi == True` are retained. Experiments with zero valid neurons are skipped.

ii.
```python
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
if n_neurons == 0:
    return None
```

iii. The AI says this matches the AllenSDK default `exclude_invalid_rois=True` and the segmentation pipeline's curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural bins begin at each trial's `start_time`; the metadata declares “Trial start time,” `off_start=0`, and no fixed end offset. Trials end at their own `stop_time`.

ii.
```python
common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
...
'temporal_alignment_event': 'Trial start time',
'off_start': 0.0,
'off_end': None,
```

iii. The notes justify using the trial table boundaries and aligning every other stream to the resulting common timestamps.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. All data are rebinned to 93.23 ms (about 10.73 Hz), chosen as the MESO plane rate. CAM2P recordings are therefore downsampled and MESO data are approximately retained; event values are sums within bins.

ii.
```python
COMMON_DT = 0.09323
common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
...
'time_bin_size': COMMON_DT * 1000,
```

iii. The AI argues a common bin was required by the target format because downloaded recordings span about 31 Hz and 10.7 Hz.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from stimulus-presentation `start_time` and `image_name`, not the trials table's initial/change image columns.

ii.
```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    ...
indices = np.searchsorted(start_times, common_ts, side='right') - 1
```

iii. The notes map `stimulus_presentations.image_name` to image identity so identity follows every actual flash in the trial.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Globally discovered non-omitted image names are sorted and mapped to integer classes. At each bin the most recently started known, non-omitted image is used; omitted/gray/unknown periods carry the last valid identity, initially class 0.

ii.
```python
all_images = sorted(all_images)
image_to_idx = {img: i for i, img in enumerate(all_image_names)}
...
if img == 'omitted':
    image_id[t] = last_valid
elif img in image_to_idx:
    image_id[t] = image_to_idx[img]
```

iii. The notes explicitly decide that gray and omitted periods inherit the most recently shown non-omitted image, consistent with labeling the current repeated image block.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Stimulus onset times are searched at the identical `common_ts` used for neural bins, producing one image category per neural time bin.

ii.
```python
trial_neural, common_ts = resample_events_to_common_bins(...)
img_id = get_image_at_timepoints(common_ts, raw['stimulus_presentations'], image_to_idx)
```

iii. The AI says this common timestamp construction provides direct timepoint-by-timepoint alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses stimulus-presentation `is_change`, `start_time`, and `stop_time`.

ii.
```python
is_change = stim_data.get('is_change', np.zeros(len(stim_data['start_time'])))
start_times = stim_data['start_time']
stop_times = stim_data['stop_time']
```

iii. The notes identify `stimulus_presentations.is_change` as the direct event-level source.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code finds all stimulus rows marked as changes and marks common timepoints from that presentation's start (inclusive) to stop (exclusive).

ii.
```python
change_indices = np.where(np.array(is_change) == 1)[0]
for ci in change_indices:
    mask = (common_ts >= start_times[ci]) & (common_ts < stop_times[ci])
    change[mask] = 1
```

iii. The AI describes this as “1 during change stimulus,” yielding a reported change prevalence of about 2.8%.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is estimated: the NWB boolean `is_change` directly yields category 1 inside a marked presentation and 0 elsewhere.

ii.
```python
change = np.zeros(n_tp, dtype=np.int64)
change[mask] = 1
```

iii. The AI treats the source flag as already binary and labels the values `no_change` and `change`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Presentation intervals are evaluated on the same `common_ts` as the rebinned neural events.

ii.
```python
img_change = get_image_change_at_timepoints(common_ts, raw['stimulus_presentations'])
```

iii. The common timebase is the stated alignment mechanism.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from the NWB running-speed `data` and `timestamps` arrays.

ii.
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. The notes map the Allen running-speed series directly to this output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Up to 2,000 non-NaN raw samples per experiment are deterministically sampled to estimate global percentiles. Per trial, speed is linearly interpolated to common timestamps and digitized; interpolated NaNs are assigned the middle class.

ii.
```python
valid_run = rng.choice(valid_run, 2000, replace=False)
running_percentiles = np.percentile(all_running, np.linspace(0, 100, 6))
...
run_interp = interpolate_to_common(...)
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned[np.isnan(run_interp)] = 2
```

iii. The AI says global percentiles give consistent, approximately balanced classes while sampling controls the statistics pass's cost.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four inner boundaries of global 0/20/40/60/80/100 percentiles define five integer bins (0–4), with clipping and missing values mapped to bin 2.

ii.
```python
running_percentiles = np.percentile(all_running, np.linspace(0, 100, 6))
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
```

iii. The decision follows the requested five equal-percentile categories; the middle bin is described as a neutral missing-data fallback.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Linear interpolation evaluates running speed at each neural `common_ts`.

ii.
```python
run_interp = interpolate_to_common(
    raw['running_speed'], raw['running_timestamps'], common_ts)
```

iii. The notes say hardware timestamps plus interpolation establish direct alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Despite the output name, the code uses EyeTracking `pupil_tracking/area` and its timestamps; it does not use pupil width/diameter or a blink flag.

ii.
```python
data['pupil_area'] = f['acquisition/EyeTracking/pupil_tracking/area'][:]
data['pupil_timestamps'] = f['acquisition/EyeTracking/pupil_tracking/timestamps'][:]
```

iii. The notes consistently call this “pupil area” in the mapping while labeling the decoder output pupil diameter, without justifying the geometric substitution.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Up to 2,000 non-NaN area samples per experiment estimate global percentiles. Trial values are linearly interpolated and digitized. Missing pupil streams or values become bin 2; blink frames are not removed.

ii.
```python
valid_pupil = rng.choice(valid_pupil, 2000, replace=False)
pupil_percentiles = np.percentile(all_pupil, np.linspace(0, 100, 6))
...
pupil_interp = interpolate_to_common(...)
pupil_binned[np.isnan(pupil_interp)] = 2
```

iii. The AI applies the same global-percentile rationale as running and explicitly chooses the middle class for missing pupil data.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four inner global percentile boundaries define five classes 0–4; values are clipped and missing values receive category 2. If no experiment has pupil data, every value is category 2.

ii.
```python
pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
pupil_binned = np.clip(pupil_binned, 0, 4).astype(np.int64)
...
pupil_binned = np.full(n_common, 2, dtype=np.int64)
```

iii. The notes say the bins are approximately 20% each and the middle class is a neutral fallback.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil area is linearly interpolated from eye timestamps to the same neural `common_ts`.

ii.
```python
pupil_interp = interpolate_to_common(
    raw['pupil_area'], raw['pupil_timestamps'], common_ts)
```

iii. As for running, common timestamps are the stated alignment mechanism.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the trial-table booleans `hit`, `miss`, `correct_reject`, and `false_alarm`.

ii.
```python
if trial_data['hit'][trial_idx]: return 0
elif trial_data['miss'][trial_idx]: return 1
elif trial_data['correct_reject'][trial_idx]: return 2
elif trial_data['false_alarm'][trial_idx]: return 3
```

iii. The notes identify these as the canonical four mutually exclusive outcomes and report manual checks against NWB trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes are priority-mapped to integers in the order hit, miss, correct reject, false alarm and repeated across every bin of the trial. Trials with none of these flags are skipped.

ii.
```python
if outcome == -1:
    continue
...
np.full(n_common, outcome, dtype=np.int64)
```

iii. The AI treats outcome as a static categorical trial variable and says its spot checks all matched the source.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing pupil datasets and insufficient valid interpolation samples become all-NaN then middle-bin values. Interpolation outside a stream's range also becomes middle-bin. Files that fail during global scanning are logged and omitted from statistics; conversion exceptions skip the experiment. Experiments missing stimuli, neurons, or two trials are skipped; very short trials and unclassified outcomes are skipped.

ii.
```python
except KeyError:
    data['has_pupil'] = False
...
if valid.sum() < 2:
    return np.full(len(common_ts), np.nan)
...
except Exception as e:
    print(f'  ERROR: {e}')
    continue
```

iii. The notes call bin 2 a neutral missing-data category, document experiment/trial edge cases, and accept sparse all-zero neural trials as legitimate rather than errors.

## 9-a. What are the most time-consuming steps of the code?

i. The two full-dataset passes dominate: reading large NWB arrays for global statistics, then loading and per-trial processing every experiment. Neural resampling and creating the roughly 3-GB pickle are also substantial.

ii.
```python
global_stats = collect_global_stats(expt_table)
...
for i, (_, expt_info) in enumerate(expt_table.iterrows()):
    result = process_experiment(...)
```

iii. The notes estimate roughly 200 seconds of loading and 400 seconds of resampling/processing, about ten minutes total.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin neural aggregation loop, per-timepoint image-label loop, per-change-presentation mask loop, per-trial loop, and linear-list searches for subjects/regions could be vectorized or replaced with indexed reductions/dictionaries. The neural routine's “Vectorized for speed” description is only partly accurate.

ii.
```python
for b in range(n_common): ...
for t in range(n_tp): ...
for ci in change_indices: ...
for trial_idx in include_indices: ...
```

iii. The notes claim vectorized event binning via `np.digitize`, but do not discuss the remaining Python aggregation loop.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB is opened in `collect_global_stats` and reopened in `load_experiment_data`. Running, pupil, and stimulus arrays are consequently read/scanned twice. Trial-level interpolation objects are also rebuilt separately for every trial.

ii.
```python
with h5py.File(fname, 'r') as f:  # statistics pass
...
f = h5py.File(fname, 'r')         # conversion pass
```

iii. The notes frame the first pass as necessary for global bins and image vocabulary, but do not acknowledge the duplicated I/O.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads unused `cell_specimen_ids`, several unused trial columns, and stimulus `omitted`; passes unused function parameters; and, when plotting is enabled, retains raw data and creates diagnostic plots not used by the decoder. The full-session statistics pass processes values outside retained Go/Catch trial windows, and expanded neural margins can sum events outside trial boundaries.

ii.
```python
data['cell_specimen_ids'] = ...
for key in ['go', 'catch', 'aborted', 'auto_rewarded', ..., 'is_change']:
...
def process_experiment(expt_id, expt_info, all_image_names, image_to_idx, ...):
```

iii. The notes justify plots as verification and global raw sampling as a practical percentile estimate, but do not identify these discarded or unused values as overhead.
