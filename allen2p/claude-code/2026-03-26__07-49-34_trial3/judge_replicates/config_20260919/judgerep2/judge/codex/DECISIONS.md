# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the local ophys experiment metadata CSV, inventories local NWB files, retains on-disk experiments from four active session types, and loads every selected experiment directly with `h5py`. It makes three passes: image-name discovery, behavioral-value collection for percentile edges, and full conversion.

ii.
```python
exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
nwb_files = list(NWB_DIR.glob('*.nwb'))
mask = (exp_table['ophys_experiment_id'].isin(nwb_ids) &
        exp_table['session_type'].isin(ACTIVE_SESSION_TYPES))
active_exps = exp_table[mask].copy()
...
with h5py.File(nwb_path, 'r') as f:
    ...
```

iii. The AI justified direct NWB access as efficient and documented that the supplied directory was a partial local release. It excluded passive OPHYS_2/5 sessions because they lack meaningful active-task outcomes, and described the implementation as a three-pass conversion.

## 1-b. How are the data split into subjects?

i. Subjects are unique metadata `mouse_id` values, converted to strings and sorted. Each output session receives the corresponding integer subject index.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. The notes identify `mouse_id` as the subject identifier and report 38 mice in the on-disk active subset.

## 1-c. How are the data split into sessions?

i. Each NWB `ophys_experiment_id` is treated as one output session. Thus separate simultaneously recorded imaging planes from one `ophys_session_id` remain separate output sessions.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    all_neural.append(session_neural)
```

iii. The mapping plan explicitly states: “Each NWB experiment = one ‘session’ in output format (one imaging plane with its own neurons).” The full report distinguishes 202 experiments from 174 actual sessions.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. For each valid row, the AI extracts regular-grid samples in the half-open interval `[start_time, stop_time)`; trial lengths therefore vary.

ii.
```python
valid_indices = np.where(valid_mask)[0]
...
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes say the SDK/NWB trial table supplies the experimental trial definition and that the full start-to-stop interval preserves both pre-change and response periods.

## 1-e. How are trials filtered based on quality controls?

i. The AI retains rows marked go or catch and excludes aborted and auto-rewarded rows. It skips trials shorter than three resampled samples or without one of the four recognized outcomes, and excludes experiments left with fewer than two processed trials.

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

iii. The explicit go/catch and aborted/auto-rewarded rules follow the task. The notes also cite the two-trial decoder requirement; no additional engagement or session-QC filter is implemented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB event-detection matrix and the cell table’s `valid_roi` mask, not from dF/F traces.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
```

iii. The AI reasoned that the paper performed analyses on inferred discrete calcium events (FastLZeroSpikeInference), so it chose events rather than dF/F.

## 2-b. How is the `neural` data processed?

i. Valid-ROI event traces are linearly interpolated from native ophys timestamps to a session-wide regular 30 Hz grid, clipped nonnegative, sliced by trial, transposed to neuron-by-time, and cast to `float32`.

ii.
```python
regular_ts = np.arange(t_start, t_end, 1.0 / target_rate)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes cite the paper’s interpolation onto consistent 30 Hz timestamps and argue this harmonizes approximately 31 Hz Scientifica and 11 Hz Multiscope recordings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `valid_roi=True` are retained; experiments with no such ROIs are skipped. There is no further activity, engagement, or trace-quality filter.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
if valid_roi.sum() == 0:
    return None
events_valid = events_data[:, valid_roi]
```

iii. The AI attributes `valid_roi` to the Allen SVM ROI-quality classifier and lists common segmentation/artifact rejection reasons. It regarded this as the appropriate neuron curation rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are aligned to trial start by selecting the shared regular timestamps between each trial’s `start_time` and `stop_time`. Metadata names trial start as the alignment event, with offset start 0 and variable end.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
...
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
'off_start': 0.0,
'off_end': None,
```

iii. The AI justified this window as preserving time-varying stimulus information across the whole experimental trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 33.33 ms (30 Hz). All neural and behavioral streams are linearly resampled to the constructed 30 Hz grid; this is interpolation rather than aggregation into count bins.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ
regular_ts = np.arange(t_start, t_end, 1.0 / target_rate)
```

iii. The cited reason is the paper’s 30 Hz analysis grid and the need for uniform resolution across acquisition systems.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from the stimulus-presentation table’s `start_time`, `image_name`, and `omitted` fields, rather than directly from trial `initial_image_name` and `change_image_name`.

ii.
```python
stim_data = {
    'start_time': stim['start_time'][()],
    'image_name': stim['image_name'][()],
    'omitted': stim['omitted'][()],
}
```

iii. The AI chose presentation-level data to represent the actual image at each time point and explicitly planned special behavior for gray screens and omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Omitted presentations are removed, the most recent non-omitted onset is found with `searchsorted`, and that image is held through gray/omitted periods. Local sorted image codes are then remapped to a sorted global 16-image vocabulary.

ii.
```python
non_omitted = ~stim_data['omitted']
insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
...
image_idx[i] = name_to_idx.get(stim_names[insert_idx[i]], 0)
...
img_id_global = np.array([local_to_global.get(v, 0)
                           for v in trial_data_out['image_identity']])
```

iii. The notes say gray intervals inherit the just-shown image and omissions retain the previous image. A global map ensures consistent category meanings across image sets/sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at `trial_ts`, the exact same 30 Hz timestamps used to slice resampled neural events, producing one code per neural time point.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. The justification is that all streams are evaluated on the shared regular grid, guaranteeing equal lengths and timestamp correspondence.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses stimulus-presentation `is_change`, `omitted`, and `start_time` (with `stop_time` loaded but not used in the marking logic).

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. The AI characterized this as the first flash after an actual image change and used presentation timing to locate it.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is initialized, then each non-omitted change onset marks a fixed 750 ms interval as one. Although change stop times are iterated, the actual stop is `start + 0.75`.

ii.
```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The notes interpret one image interval as 250 ms stimulus plus 500 ms gray and therefore retain the change label for that full 750 ms interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No continuous threshold is estimated. Membership in any half-open 750 ms change window is category 1; every other sample is category 0, exposed as `no_change`/`change`.

ii.
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
change_signal[mask] = 1
...
['no_change', 'change']
```

iii. The AI treated image change as intrinsically binary, exactly as requested.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change windows are evaluated at the same `trial_ts` used for neural extraction.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. The common 30 Hz timebase is the stated alignment mechanism.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from NWB running-speed `data` and `timestamps`.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. The notes identify this as the SDK/NWB wheel-derived speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Raw speed is interpolated to the 30 Hz grid. Five global percentile edges are computed from all selected experiments’ complete raw speed streams, then trial values are digitized; NaNs would map to bin 0.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'],
                              raw_data['running_speed'])
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The AI states that global percentiles yield approximately balanced decoding classes and shared labels across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 0th, 20th, 40th, 60th, 80th, and 100th percentiles define five bins. Duplicate edges are nudged upward by `1e-10`, `np.digitize` assigns codes 0–4, and results are clipped to range.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
edges = np.percentile(valid, percentiles)
...
binned = np.digitize(values, bin_edges[1:-1])
binned = np.clip(binned, 0, n_bins - 1)
```

iii. This directly implements the required five equal-percentile categories, with defensive handling of constant-valued distributions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated to `regular_ts` before both speed and neural traces are indexed by the same trial indices.

ii.
```python
running_resampled = np.interp(regular_ts, ...)
running_trial = running_resampled[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T
```

iii. The AI relied on hardware-synchronized source clocks and a common target grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The output actually uses EyeTracking pupil `area`, its timestamps, and `likely_blink`; it does not derive diameter from `pupil_width`.

ii.
```python
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The notes explicitly call pupil area a proxy for the requested diameter and selected it because it was available in the NWB tracking data.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples are set to NaN; missing values are linearly interpolated by sample index (all missing becomes zero), then the repaired signal is interpolated to 30 Hz. Five global percentile edges are computed from complete raw non-blink pupil-area streams, and trial values are digitized. Missing pupil streams become all-NaN trials and therefore bin 0.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. The AI justified blink interpolation as preventing blink artifacts while preserving continuous time series, followed by global percentile binning for balanced categories.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The same 20-percentile procedure as running speed produces codes 0–4, with duplicate-edge repair, clipping, and NaN-to-zero behavior.

ii.
```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. This was intended to implement the requested five equal-percentile categories globally.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Repaired pupil area is interpolated to the shared 30 Hz grid, then sliced with the same `trial_time_indices` as neural activity.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. The common hardware-synchronized grid is the stated basis of alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the trial table’s mutually exclusive `hit`, `miss`, `false_alarm`, and `correct_reject` flags.

ii.
```python
if trials['hit'][trial_idx]: outcome = 0
elif trials['miss'][trial_idx]: outcome = 1
elif trials['false_alarm'][trial_idx]: outcome = 2
elif trials['correct_reject'][trial_idx]: outcome = 3
else: continue
```

iii. The AI identified these as the canonical four outcomes for valid go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Flags are mapped in fixed priority order to codes 0–3. Unknown outcomes are discarded. The selected scalar is broadcast across every time point of the trial in the final mixed output matrix.

ii.
```python
outcome_broadcast = np.full((1, n_tp),
                            trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. Broadcasting accommodates the target format’s single matrix per trial while keeping the outcome static; the names preserve code interpretation.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. File-load errors and absent stimulus tables skip experiments; zero valid ROIs, fewer than two valid/processed trials, very short trials, and unknown outcomes also cause skips. Missing pupil data becomes NaN and bin 0. Blink/NaN pupil samples are interpolated (or zero-filled if all missing), NaN categorical values map to zero, byte strings are decoded, and duplicate percentile edges are repaired. Errors in the preliminary scans are logged and processing continues.

ii.
```python
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
...
if nans.all():
    return np.zeros_like(arr)
...
binned[np.isnan(values)] = 0
```

iii. The notes describe these as defensive measures for partial/missing eye tracking, blinks, sparse/invalid experiments, and local subset files. The conversion and verification logs were used as sanity checks.

## 9-a. What are the most time-consuming steps of the code?

i. Repeated NWB I/O dominates: all selected files are opened during image discovery, behavioral percentile collection, and full loading. In the main pass, neural-event interpolation across every cell and construction/storage of the very large converted pickle are also costly.

ii.
```python
for _, row in exp_table.iterrows():  # Pass 1
    with h5py.File(nwb_path, 'r') as f: ...
for _, row in exp_table.iterrows():  # Pass 2
    with h5py.File(nwb_path, 'r') as f: ...
for _, row in exp_table.iterrows():  # Pass 3
    raw_data = load_experiment_data(nwb_path, eid)
```

iii. The AI’s notes explicitly identify sequential experiment processing and multiple passes over NWB files as inefficiencies; full conversion took about six minutes and produced an 8.3 GB pickle.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-timepoint loop in image lookup can be replaced by indexed vectorized mapping after `searchsorted`. Per-cell interpolation, local-to-global list mapping, per-trial assembly, and the loop over change windows could also be vectorized or batched, although variable trial lengths limit full vectorization.

ii.
```python
for i in range(n_tp):
    if insert_idx[i] >= 0:
        image_idx[i] = name_to_idx.get(stim_names[insert_idx[i]], 0)
...
for i in range(data.shape[1]):
    result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
```

iii. The notes claim `searchsorted` and interpolation are efficient but also acknowledge sequential processing. The explicit loops above remain opportunities for faster array operations or parallel file-level processing.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB is opened three times. Image names and complete running/pupil streams are read during preliminary passes and then read again in `load_experiment_data`; image vocabularies are sorted/mapped locally and then remapped globally. Behavioral data used to define edges are collected from full sessions, then interpolated and sliced again during conversion.

ii.
```python
# Pass 1: Collecting global image names
# Pass 2: Collecting running speed and pupil data
# Pass 3: Processing experiments
```

iii. The AI explicitly documented the three-pass design and identified “multiple passes over NWB files” as an inefficiency, accepting it to establish global category mappings before final assembly.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads cell specimen IDs but never exports them; loads trial `change_time`, `is_change`, and stimulus `stop_time` even though final calculations do not use some of them; constructs an unused scalar `trial_outcome` array; and, when plotting is disabled, still retains raw structures until each experiment iteration ends. Most significantly, it interpolates full-session neural traces to 30 Hz before retaining only trial windows, so all inter-trial resampled neural samples are discarded.

ii.
```python
cell_specimen_ids = cell_table['cell_specimen_id'][()]
...
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
...
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
```

iii. The AI did not specifically justify these discarded intermediates. They follow from convenient whole-session preprocessing and optional diagnostic plotting; the notes only generally acknowledge efficiency costs.
