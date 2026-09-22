# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the project experiment CSV, finds NWB files physically present, restricts them to four active session types, and directly opens each selected NWB with `h5py`. It therefore processes the available active subset, not every VisualBehavior experiment exposed through the SDK cache.

ii.
```python
exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
nwb_files = list(NWB_DIR.glob('*.nwb'))
mask = (exp_table['ophys_experiment_id'].isin(nwb_ids) &
        exp_table['session_type'].isin(ACTIVE_SESSION_TYPES))
active_exps = exp_table[mask].copy()
with h5py.File(nwb_path, 'r') as f:
    ...
```

iii. The notes say only 284 NWBs were on disk and characterize them as a subset. They justify excluding OPHYS_2/5 as passive sessions without meaningful behavioral outcomes and direct NWB loading as efficient access to the provided data.

## 1-b. How are the data split into subjects?

i. Subjects are unique metadata `mouse_id` values, converted to strings and sorted; each experiment receives the corresponding subject index.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. The notes identify `mouse_id` as the subject identifier and report 38 mice in the on-disk subset.

## 1-c. How are the data split into sessions?

i. Each selected `ophys_experiment_id`/NWB (one imaging plane) is emitted as a separate output session. Experiments sharing an `ophys_session_id` are not merged.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    all_neural.append(session_neural)
```

iii. The notes explicitly decide that “Each NWB experiment = one ‘session’ in output format (one imaging plane with its own neurons).”

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. For every retained trial, the AI takes regular 30-Hz samples from `start_time` inclusive to `stop_time` exclusive, producing variable-length trials.

ii.
```python
trials_grp = f['intervals']['trials']
...
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
```

iii. The notes say the full trial window preserves pre-change and response periods and follows the SDK/NWB trial definition.

## 1-e. How are trials filtered based on quality controls?

i. Only trials marked go or catch are retained; aborted and auto-rewarded trials are removed. Trials with fewer than three samples or without one recognized outcome are skipped, and experiments with fewer than two processed trials are dropped.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
...
if len(trial_time_indices) < 3:
    continue
...
else:
    continue  # Unknown outcome, skip
```

iii. The task explicitly requested go/catch inclusion and aborted/auto-rewarded exclusion. The notes also require at least two trials for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from the NWB event-detection `data` array, filtered using the cell table's `valid_roi` field.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
```

iii. The AI cites the paper's use of discrete calcium events and notes that FastLZeroSpikeInference events are already computed in the NWB.

## 2-b. How is the `neural` data processed?

i. Valid-ROI event traces are linearly interpolated neuron by neuron to a uniform 30-Hz session grid, clipped nonnegative, sliced by trial, transposed to neuron-by-time, and cast to float32.

ii.
```python
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes cite the paper's statement that signals were linearly interpolated to consistent 30-Hz timestamps and explain clipping because events should be nonnegative.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only cells with `valid_roi=True` are retained; an experiment with zero valid ROIs is skipped. No additional cell-level filtering is applied.

ii.
```python
n_valid = valid_roi.sum()
if n_valid == 0:
    return None
events_valid = events_data[:, valid_roi]
```

iii. The AI attributes `valid_roi` to the SDK's SVM-based ROI quality control and describes it as the standard automatic cell curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is segmented from trial start to trial stop on the regular time grid. Metadata calls trial start the alignment event and sets `off_start=0`, with variable end time.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
...
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
'off_start': 0.0,
'off_end': None,
```

iii. The notes choose trial start so the complete variable-length behavioral trial is retained.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output is 30 Hz (33.33 ms). All neural traces are linearly resampled from native ophys timestamps to a uniform grid; this is interpolation rather than aggregation into count bins.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ
regular_ts = np.arange(t_start, t_end, 1.0 / target_rate)
```

iii. The AI relies on the paper's 30-Hz interpolation and uses it to standardize recordings with different native rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the NWB stimulus-presentation table's `start_time`, `image_name`, and `omitted` fields, rather than the trial table's initial/change image columns.

ii.
```python
stim_data = {'start_time': stim['start_time'][()],
             'image_name': stim['image_name'][()],
             'omitted': stim['omitted'][()]}
```

iii. The notes say stimulus presentations provide frame-level identity, including repeated flashes and omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Omitted presentations are removed; each target time gets the most recent non-omitted image via `searchsorted`. Names are first locally coded, then remapped into a sorted global vocabulary collected by scanning every experiment.

ii.
```python
non_omitted = ~stim_data['omitted']
insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
img_name = stim_names[insert_idx[i]]
image_idx[i] = name_to_idx.get(img_name, 0)
```

iii. The AI decided that gray periods and omitted flashes inherit the last shown image and that a global mapping is needed for consistent decoder labels.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at the exact regular 30-Hz timestamps used to slice each neural trial, so its length equals the neural time dimension.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. The notes describe all streams as aligned to the common 30-Hz timebase.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change comes from stimulus-presentation `is_change`, `omitted`, and `start_time` (the loaded `stop_time` is assigned but not used).

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
```

iii. The AI says stimulus-level change flags identify actual non-omitted image changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created and every sample in the 750-ms interval beginning at each qualifying change onset is set to one.

ii.
```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The notes equate 750 ms to the changed 250-ms flash plus its following 500-ms gray interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: 1 inside a qualifying 750-ms change window and 0 otherwise; no statistical threshold is used.

ii.
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
change_signal[mask] = 1
```

iii. The task asks for a binary change output, and the AI names the categories `no_change` and `change`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The indicator is calculated directly on each trial's regular timestamps, identical to the neural sample timestamps.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. The common 30-Hz grid is the stated alignment mechanism.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses the NWB running-speed `data` and `timestamps` arrays.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. The notes identify this as the SDK/NWB running-wheel speed signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated to the 30-Hz grid and then digitized with five global percentile bins. However, edges are computed from all raw full-session running samples, including times outside retained trials.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The AI says interpolation synchronizes the stream and global percentile bins provide consistent, approximately balanced classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 0th, 20th, 40th, 60th, 80th, and 100th percentiles define five labels (0-4). Duplicate edges are nudged upward; NaNs map to bin 0.

ii.
```python
edges = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
binned = np.digitize(values, bin_edges[1:-1])
binned[np.isnan(values)] = 0
```

iii. This follows the requested five equal-percentile categories and adds defensive handling for constant or missing values.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to `regular_ts`, then indexed with the same `trial_time_indices` as resampled neural events.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
running_trial = running_resampled[trial_time_indices]
```

iii. The AI justifies the shared regular timebase as explicit synchronization.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The output actually uses eye-tracking pupil `area`, its timestamps, and the `likely_blink` mask—not pupil width/diameter.

ii.
```python
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The notes call area a proxy for pupil diameter and use the blink flag to reject artifacts.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN, missing samples are interpolated by array index (all-NaN becomes zeros), the result is interpolated to 30 Hz, and it is digitized with global pupil-area percentiles computed from raw non-blink samples.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. The AI says this prevents blink artifacts and provides consistent five-bin categories.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five global percentile categories are derived from pooled non-blink pupil-area samples. Missing pupil data in an experiment remains NaN at trial construction and is assigned category 0.

ii.
```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_trial = np.full(n_tp, np.nan)
...
binned[np.isnan(values)] = 0
```

iii. The notes invoke the requested equal-percentile discretization and describe bin 0 as the missing-data fallback.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Cleaned pupil area is interpolated from eye timestamps to the common 30-Hz session grid and sliced using the neural trial indices.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. The shared target grid is the AI's alignment rationale.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome comes from the trial table's `hit`, `miss`, `false_alarm`, and `correct_reject` booleans.

ii.
```python
if trials['hit'][trial_idx]: outcome = 0
elif trials['miss'][trial_idx]: outcome = 1
elif trials['false_alarm'][trial_idx]: outcome = 2
elif trials['correct_reject'][trial_idx]: outcome = 3
```

iii. The AI describes these as the canonical mutually exclusive outcomes for retained trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes are priority-mapped to integer codes 0-3; unrecognized trials are skipped. The selected code is broadcast across every timepoint of the trial.

ii.
```python
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. Broadcasting lets static and time-varying targets coexist in one rectangular output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. File/loading errors and absent stimulus tables skip experiments; zero valid ROIs, too few trials, too-short trials, and unknown outcomes are also skipped. Missing pupil data becomes NaN then bin 0; blink/NaN pupil samples are interpolated (all missing becomes zero). Degenerate percentile edges are separated by epsilon, and absent names default to code 0.

ii.
```python
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
...
if nans.all():
    return np.zeros_like(arr)
...
local_to_global[local_idx] = 0
```

iii. The notes frame these as robustness measures that keep one bad experiment from aborting conversion and ensure categorical arrays contain valid integers.

## 9-a. What are the most time-consuming steps of the code?

i. Repeatedly opening every NWB in three passes and, in the final pass, loading full event arrays and interpolating every neuron are the dominant operations. Serialization of the resulting 8.3-GB pickle is also substantial.

ii.
```python
for _, row in exp_table.iterrows():  # Pass 1
    with h5py.File(nwb_path, 'r') as f: ...
for _, row in exp_table.iterrows():  # Pass 2
    with h5py.File(nwb_path, 'r') as f: ...
for _, row in exp_table.iterrows():  # Pass 3
    raw_data = load_experiment_data(nwb_path, eid)
```

iii. The notes estimate loading and processing separately, report a six-minute full conversion, and identify multiple NWB passes as an inefficiency.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Neural interpolation loops over cells; image assignment loops over target timepoints; local-to-global image remapping loops over every trial sample; and change-window construction loops over change events. These could use vectorized indexing/mappings or batched interpolation.

ii.
```python
for i in range(data.shape[1]):
    result[:, i] = np.interp(...)
for i in range(n_tp):
    image_idx[i] = name_to_idx.get(stim_names[insert_idx[i]], 0)
img_id_global = np.array([local_to_global.get(v, 0) for v in ...])
```

iii. The notes mention vectorized `np.interp`/`searchsorted` as speedups but also acknowledge sequential experiment processing; the remaining Python loops are not specifically justified.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB is opened three times: once for global image names, once for full-session running/pupil percentile samples, and once for complete conversion. Stimulus/image mappings are also rebuilt per experiment and remapped per sample.

ii.
```python
print("--- Pass 1: Collecting global image names ---")
...
print("--- Pass 2: Collecting running speed and pupil data ...")
...
print("--- Pass 3: Processing experiments ---")
```

iii. The notes explicitly list “Multiple passes over NWB files” as an inefficiency, accepting it to obtain global vocabularies and bin edges before assembly.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads unused fields (`cell_specimen_ids`, several trial columns, and stimulus `stop_time`), computes an unused `trial_outcomes` list and `trial_outcome` array, and imports unused concurrency/system modules. It also resamples complete sessions even though only retained trial windows are stored; optional plotting performs additional work whose figures are not decoder inputs.

ii.
```python
cell_specimen_ids = cell_table['cell_specimen_id'][()]
trial_outcomes = []
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
```

iii. The AI's notes discuss memory/speed and optional plots but do not explicitly justify these unused values; full-session resampling simplifies common-grid alignment.
