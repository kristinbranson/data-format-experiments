# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads local metadata from `ophys_experiment_table.csv`, discovers available NWB files with `glob`, keeps only experiments whose NWB files exist locally, filters to `passive == False`, and then opens each remaining experiment directly with `h5py`. It does not use the Allen SDK cache and it processes data experiment-by-experiment rather than reconstructing full SDK sessions.

ii. 

```python
def get_experiment_metadata():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    ...
    our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
    active_exps = our_exps[our_exps['passive'] == False].copy()
    active_exps = active_exps.sort_values('ophys_experiment_id').reset_index(drop=True)
    return active_exps, nwb_map
```

```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. `CONVERSION_NOTES.md` says the local data consist of 284 NWB files and resolves the discrepancy with the larger released dataset by deciding to work from the files on disk. It also states 'Active sessions only' because passive sessions do not have meaningful trial outcomes, and later claims that direct `h5py` NWB reads are 'Equivalent' to SDK loading.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by unique `mouse_id` values in the filtered experiment metadata table.

ii. 

```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
```

```python
'mouse_id': str(exp_info['mouse_id'])
```

iii. The notes repeatedly summarize the dataset in terms of unique `mouse_id` counts and treat those IDs as subject identity.

## 1-c. How are the data split into sessions?

i. Each retained `ophys_experiment_id` is treated as its own session. The AI does not group multiple experiments that share an `ophys_session_id`.

ii. 

```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

```python
session_info.append({
    'experiment_id': eid,
    'mouse_id': result['mouse_id'],
    'brain_region': result['brain_region'],
    'n_neurons': result['n_neurons'],
    'n_trials': result['n_trials'],
})
```

iii. The notes describe each NWB file as 'one imaging plane from one session' and later report '202 active experiments' as if those were sessions. No grouping by `ophys_session_id` appears in the code or notes.

## 1-d. How are the data split into trials?

i. Trials are identified from the NWB `trials` table, but each trial is represented by the set of stimulus presentations whose `trials_id` matches that trial. The trial length is therefore the number of 750 ms image presentations assigned to that trial, not the continuous `start_time` to `stop_time` ophys window.

ii. 

```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
    n_bins = len(trial_stim_indices)
```

```python
output_trials.append({
    'image_identity': image_indices,
    'image_change': change_flags,
    ...
    'n_bins': n_bins,
})
```

iii. `CONVERSION_NOTES.md` explicitly says 'Time bin = 750ms (1 per stimulus flash)' and maps every output to stimulus-presentation bins. The justification given is that the 750 ms flash cycle is the 'natural task unit' and consistent across equipment types.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only go or catch trials and excludes aborted and auto-rewarded trials. It also skips trials that have no associated stimulus presentations and later drops experiments with fewer than two retained trials.

ii. 

```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
```

```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
if len(trial_stim_indices) == 0:
    continue
```

```python
if result is None or result['n_trials'] < 2:
    skipped += 1
    ...
    continue
```

iii. The notes say 'Include Go+Catch only per task instructions' and later note the minimum-trial threshold as an edge-case check. No additional justification is given for omitting other reference checks such as `change_time.notna()`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the NWB dF/F trace array at `processing/ophys/dff/traces/data`, using the matching trace timestamps from `processing/ophys/dff/traces/timestamps`.

ii. 

```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
```

iii. The notes say dF/F is precomputed in NWB and explicitly choose 'Use dF/F (not events)'.

## 2-b. How is the `neural` data processed?

i. The AI filters to valid ROIs, then averages dF/F within each 750 ms stimulus-presentation window using cumulative sums. It keeps each experiment separately rather than merging planes into a multi-plane session.

ii. 

```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

```python
bin_duration = TIME_BIN_MS / 1000.0
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
...
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The notes justify this by saying 750 ms is the natural task unit and that dF/F is the standard decoding signal. They also list cumulative-sum averaging as an optimization.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only ROIs where `valid_roi` is true and skips experiments with zero such neurons. It does not add any other neuron QC beyond that raw-file filter.

ii. 

```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
...
if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
```

iii. The notes call the `valid_roi` filter the SDK default behavior and treat it as the required ROI curation step when reading raw NWB directly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to stimulus-presentation onsets: each column in a trial matrix is the average dF/F over one 750 ms flash-plus-gray window anchored at `stim['start_time']`.

ii. 

```python
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
```

```python
'temporal_alignment_event': 'Stimulus presentation onset (each 750ms image flash)'
```

iii. The notes explicitly justify the choice as alignment to the stimulus cycle because it is the natural task unit and consistent across MESO and CAM2P recordings.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 750 ms bins, with explicit temporal rebinning by averaging all samples that fall in each image-plus-gray stimulus interval.

ii. 

```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
```

```python
'time_bin_size': TIME_BIN_MS
```

iii. The notes say 'Time bin = 750ms (1 per stimulus flash)' because it is the natural task unit and harmonizes recordings with different native frame rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stim['image_name']` in the stimulus-presentation table, indexed by the presentations attached to each trial via `stim['trials_id']`.

ii. 

```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stim['image_name'][:]])
stim_trials_id = stim['trials_id'][:]
```

```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
images = stim_image_name[trial_stim_indices].copy()
```

iii. The notes map `image_name` from `stimulus_presentations` directly to `output[0]` and frame this as the appropriate raw variable for a stimulus-binned representation.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI forward-fills omitted presentations from neighboring non-omitted stimuli, converts image names to integer IDs with a global mapping, and builds that mapping by sampling only the first up to 10 experiments instead of scanning all retained data.

ii. 

```python
for bi in range(len(images)):
    if images[bi] == 'omitted':
        if bi > 0:
            images[bi] = images[bi - 1]
        else:
            for bj in range(bi + 1, len(images)):
                if images[bj] != 'omitted':
                    images[bi] = images[bj]
                    break
```

```python
sample_exps = active_exps.head(min(10, len(active_exps)))
...
if name != 'omitted':
    all_images.add(name)
```

```python
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. The notes explicitly justify forward-filling omitted stimuli and say image names are sampled from only 10 experiments 'to be fast', based on the assumption that experiments use the same image set.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is one label per 750 ms stimulus bin and is aligned with neural data by using the same ordered `trial_stim_indices` list that was used for neural bin averaging.

ii. 

```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
...
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
...
output_trials.append({
    'image_identity': image_indices,
    ...
})
```

iii. The notes present all outputs as sharing the same 750 ms stimulus-presentation bins, so alignment is by common bin index rather than by framewise interpolation onto `ophys_ts`.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived directly from the stimulus-presentation `is_change` flag, again indexed by `trials_id` to pick the presentations belonging to each trial.

ii. 

```python
stim_is_change = stim['is_change'][:]
stim_trials_id = stim['trials_id'][:]
```

```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes map `is_change` from the stimulus-presentation table to `output[1]` and describe it as '1 at change flash only'.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI simply converts the selected `is_change` values to integers, replacing NaNs with 0. There is no extra change-time windowing beyond whatever is already encoded in the stimulus table.

ii. 

```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes describe image change as the stimulus-presentation `is_change` variable and do not document any further transformation besides bin-level coding.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is treated as a binary categorical variable with values 0=`no_change` and 1=`change`.

ii. 

```python
change_value_names = ['no_change', 'change']
```

```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. This follows directly from the task requirement that image change be binary and from the note that the output should be '1 at change flash only'.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned by stimulus-presentation bin: each change flag corresponds to the same 750 ms bin used to average neural activity.

ii. 

```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
...
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ot['image_change'].astype(np.int64),
    ...
], axis=0)
```

iii. The notes explicitly say all outputs are built on the same 750 ms per-stimulus time base.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the NWB running-speed dataset at `processing/running/speed/data` and its timestamps at `processing/running/speed/timestamps`.

ii. 

```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The notes map running speed from the NWB running stream and describe it as the filtered SDK running-speed source.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI averages running-speed samples within each 750 ms stimulus window, collects all such values in a first pass, computes global percentile edges for 5 bins, and then discretizes each trial’s binned running trace in the second pass.

ii. 

```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
run_cumsum = np.cumsum(running_speed)
...
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
    n_pts = e - s
    if n_pts > 0:
        running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
```

```python
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
...
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. The notes justify this as using the stimulus cycle as the common time base and using global percentile bins so the 5 categories are balanced across the dataset.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into five global percentile bins.

ii. 

```python
N_PERCENTILE_BINS = 5
```

```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(valid, percentiles)
```

iii. The notes explicitly list 'Discretization: Compute percentile bin edges across ALL valid time points in dataset (2-pass), then assign bins.'

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by using the same stimulus-presentation windows used for neural averaging; one running value is produced per neural bin.

ii. 

```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
```

```python
running_binned = np.zeros(n_bins, dtype=np.float32)
...
output_trials.append({
    ...
    'running_speed_raw': running_binned,
    ...
})
```

iii. The notes describe all streams as sharing the same 750 ms per-stimulus time base.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking `area` and `timestamps`, plus the `likely_blink` flag. The AI converts area to diameter with `2*sqrt(area/pi)` rather than reading a width field directly.

ii. 

```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

```python
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. The notes say 'Pupil tracking: area, width, height available' and explicitly choose 'Use area -> compute diameter, interpolate NaNs'.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI marks blink and nonpositive area samples as NaN, converts area to diameter, linearly interpolates NaNs, averages pupil diameter within each 750 ms stimulus bin, then discretizes the binned values into 5 global percentile bins; if an entire trial bin vector is NaN, it fills the middle category.

ii. 

```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
```

```python
for bi, si in enumerate(trial_stim_indices):
    s, e = pup_bin_starts[si], pup_bin_ends[si]
    n_valid = pup_count_cumsum[e] - pup_count_cumsum[s]
    if n_valid > 0:
        pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

```python
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The notes justify blink interpolation before averaging and the two-pass percentile discretization. They do not justify the all-NaN fallback beyond treating it as an edge-case handler.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into five global percentile bins.

ii. 

```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
if len(valid_pupil) > 0:
    _, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
```

```python
pupil_value_names = [f'pupil_q{i+1}' for i in range(N_PERCENTILE_BINS)]
```

iii. The notes make the same global-percentile-binning argument for pupil as for running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by the same stimulus-presentation windows used for neural and running-speed bins, yielding one pupil value per neural bin.

ii. 

```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
```

```python
'pupil_diameter_raw': pupil_binned
```

iii. The notes again describe a common 750 ms per-stimulus alignment scheme for all outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean `hit`, `miss`, `false_alarm`, and `correct_reject` fields in the NWB trials table.

ii. 

```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. The notes map 'hit/miss/fa/cr' from the trials table to `output[4]`.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four booleans to integer classes 0=hit, 1=miss, 2=false_alarm, 3=correct rejection with an `if/elif` cascade, defaults unmatched trials to miss, and then repeats that static label across all time bins in the trial output array.

ii. 

```python
if trial_hit[trial_idx]:
    outcome = 0  # Hit
elif trial_miss[trial_idx]:
    outcome = 1  # Miss
elif trial_fa[trial_idx]:
    outcome = 2  # False Alarm
elif trial_cr[trial_idx]:
    outcome = 3  # Correct Rejection
else:
    outcome = 1  # Default to Miss
```

```python
np.full(n_bins, ot['trial_outcome'], dtype=np.int64)
```

iii. The notes describe this as a 4-class categorical output and the decoder-format requirement makes it static per trial but repeated across time bins.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missingness and edge cases ad hoc: it skips experiments with no valid neurons or no stimulus table, skips trials with no linked stimulus presentations, interpolates NaNs in pupil traces, fills all-NaN pupil trials with the middle category, maps NaN `is_change` values to 0, and silently maps unknown image names to class 0.

ii. 

```python
if n_neurons == 0:
    ...
    return None
...
if stim_key is None:
    ...
    return None
```

```python
if len(trial_stim_indices) == 0:
    continue
```

```python
pupil_diameter = interpolate_nans(pupil_diameter)
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. Some of these choices are justified in the notes: pupil NaNs are interpolated because blinks should be removed, and failed/empty units are skipped to keep the pipeline running. The more ad hoc fallbacks (middle-bin pupil, unknown-image-to-0) are not explicitly justified in the notes.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive work is opening and reading every NWB experiment in two full passes, plus the repeated per-trial/per-bin averaging loops inside `process_experiment`.

ii. 

```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    ...
    stats = process_experiment(nwb_path, row, collect_stats_only=True)
```

```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    ...
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The notes explicitly describe a two-pass design and include timing estimates for Pass 1 and Pass 2. They also discuss cumulative-sum optimizations as attempts to reduce the cost of the inner averaging loops.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Despite some cumulative-sum optimization, the code still leaves several vectorizable loops: the per-trial `for bi, si in enumerate(trial_stim_indices)` loops for neural, running, and pupil averaging; the omitted-image forward-fill loop; and the repeated Python-level iteration over experiments and trials.

ii. 

```python
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    ...
    neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

```python
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
    ...
for bi, si in enumerate(trial_stim_indices):
    s, e = pup_bin_starts[si], pup_bin_ends[si]
    ...
```

```python
for bi in range(len(images)):
    if images[bi] == 'omitted':
        ...
```

iii. The notes justify the current structure as an optimization over boolean masking by using cumulative sums, but they do not claim the implementation is fully vectorized. This observation is inferred mostly from the code itself.

## 9-c. What processing does the code repeat multiple times?

i. The AI repeats substantial work: it reads and processes every experiment twice (once to collect running/pupil values and again for the final conversion), and it also opens a sampled subset of experiments earlier just to collect image names.

ii. 

```python
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
```

```python
stats = process_experiment(nwb_path, row, collect_stats_only=True)
```

```python
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The notes explicitly describe the conversion as a two-pass approach for percentile-bin computation, and separately mention sampling experiments for image-name collection.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads several raw variables it never uses downstream (`stim_stop`, `stim_omitted`, `valid_trial_ids`, `trial_start`, `trial_stop`, `trial_change_time`, `n_total_rois`), computes plotting-only data paths in the main script, and in the stats-only pass still does most of the same per-trial processing before discarding everything except running and pupil values.

ii. 

```python
stim_stop = stim['stop_time'][:]
stim_omitted = stim['omitted'][:]
...
valid_trial_ids = trial_ids[trial_mask]
...
trial_start = trials['start_time'][:]
trial_stop = trials['stop_time'][:]
trial_change_time = trials['change_time'][:]
```

```python
n_total_rois = dff_data.shape[1]
```

```python
if collect_stats_only:
    continue
...
if collect_stats_only:
    return {
        'running_values': running_values,
        'pupil_values': pupil_values,
    }
```

iii. There is no explicit justification for these discarded computations beyond general speed-oriented comments in the notes. The two-pass design explains why some work is intentionally duplicated, but not why unused variables are loaded and retained in the code path.
