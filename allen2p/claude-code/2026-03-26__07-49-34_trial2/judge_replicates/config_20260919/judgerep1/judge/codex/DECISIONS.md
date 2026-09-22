# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment metadata CSV and inventories local NWB files, retains experiments whose IDs have files, filters to active (`passive == False`) experiments, and opens each retained NWB directly with `h5py`. Full mode processes every retained active experiment; sample mode selects two.

ii.
```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
active_exps = our_exps[our_exps['passive'] == False].copy()
...
with h5py.File(nwb_path, 'r') as f:
```

iii. The notes say the 284 local NWBs are the available subset and that only 202 active experiments should be used because passive recordings lack meaningful trial outcomes. Direct HDF5 reads were considered equivalent to SDK loading.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique `mouse_id` values among retained active experiments; each stored experiment receives the corresponding index.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
subject_idx = subjects_list.index(result['mouse_id'])
all_subject_idx.append(subject_idx)
```

iii. The agent treats `mouse_id` as the animal identifier and reports 38 mice in the available active subset.

## 1-c. How are the data split into sessions?

i. Every NWB/`ophys_experiment_id` is emitted as a separate target “session.” Experiments sharing an `ophys_session_id` are not grouped, so simultaneous imaging planes remain separate.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    result = process_experiment(nwb_map[eid], row, collect_stats_only=False)
    ...
    all_sessions_neural.append(result['neural_trials'])
```

iii. The notes explicitly state “Each NWB file = one imaging plane from one session,” but then use active experiments as sessions and describe 202 processed experiments as 202 sessions. No justification is given for not reconstructing behavioral sessions from all their planes.

## 1-d. How are the data split into trials?

i. Trial rows are selected from the NWB trials table. For each selected trial ID, all stimulus presentations whose `trials_id` equals that ID become the trial's time bins. Thus trials have one bin per presentation and variable numbers of bins.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
    n_bins = len(trial_stim_indices)
```

iii. The agent says stimulus presentations are the natural task unit (250 ms image plus 500 ms gray), allowing all streams to share presentation-aligned 750 ms bins.

## 1-e. How are trials filtered based on quality controls?

i. Only Go or Catch trials are included; aborted and auto-rewarded trials are excluded. Trials with no associated stimulus presentation are skipped, and experiments with fewer than two resulting trials are skipped.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
if len(trial_stim_indices) == 0:
    continue
...
if result is None or result['n_trials'] < 2:
    continue
```

iii. This is justified directly by the task's instruction to include Go and Catch but exclude aborted and auto-rewarded trials, plus the target format's two-trial minimum.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB `processing/ophys/dff/traces/data` dataset, with timestamps from the adjacent `timestamps` dataset and ROI validity from `image_segmentation/cell_specimen_table/valid_roi`.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. The notes choose precomputed dF/F as a standard decoding signal even though the paper's neural analyses used inferred calcium events.

## 2-b. How is the `neural` data processed?

i. Invalid ROIs are removed. For every stimulus presentation, valid-neuron dF/F is averaged over `[stimulus_start, stimulus_start + 0.75 s)` using cumulative sums. The result is transposed conceptually into neuron-by-presentation-bin matrices. Planes are not merged.

ii.
```python
dff_valid = dff_data[:, valid_roi]
dff_cumsum = np.vstack([np.zeros((1, n_neurons)), np.cumsum(dff_valid, axis=0)])
...
neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The agent argues that 750 ms is the common task unit across 11 Hz and 31 Hz equipment and that cumulative sums make averaging efficient.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `valid_roi=True` are retained; experiments with zero valid neurons are skipped. No further cell or trace filtering is performed.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
if n_neurons == 0:
    return None
```

iii. The notes say this matches the AllenSDK default `exclude_invalid_rois=True` and cite the Allen ROI curation criteria.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to every stimulus-presentation onset, not directly to trial start: ophys-frame bounds are found with `searchsorted`, and one mean is produced per 750 ms presentation.

ii.
```python
ophys_bin_starts = np.searchsorted(ophys_ts, stim_start, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, stim_start + 0.75, side='left')
```

iii. The notes describe stimulus onset as the most natural alignment for time-varying image outputs and state that ophys timestamps remain the neural temporal reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is fixed at 750 ms. Native 11/31 Hz ophys frames are explicitly averaged into one bin per presentation.

ii.
```python
TIME_BIN_MS = 750.0
bin_duration = TIME_BIN_MS / 1000.0
...
'time_bin_size': TIME_BIN_MS,
```

iii. The agent justifies rebinning as matching the 250 ms image plus 500 ms gray stimulus cycle and standardizing equipment with different frame rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `image_name` and `omitted` in the stimulus-presentation interval table, restricted by `trials_id`.

ii.
```python
stim_image_name = np.array([... for x in stim['image_name'][:]])
stim_omitted = stim['omitted'][:]
images = stim_image_name[trial_stim_indices].copy()
```

iii. The notes say the presentation table represents the actual sequence and that each session has eight natural-scene images (16 globally across sets).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Omitted labels are filled from the preceding image, or from the next non-omitted image when omission is first. Names are mapped to globally consistent integer indices; the global vocabulary is sampled from at most ten experiments.

ii.
```python
if images[bi] == 'omitted':
    if bi > 0:
        images[bi] = images[bi - 1]
...
image_indices = np.array([img_to_idx.get(img, 0) for img in images])
```

iii. The agent says forward filling preserves the identity associated with an omission and that image sets are shared sufficiently for sampling ten experiments to be fast and complete.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The identity at each presentation index is paired with the dF/F mean from the same presentation's 750 ms window, so output and neural matrices have the same number and ordering of bins.

ii.
```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
images = stim_image_name[trial_stim_indices].copy()
for bi, si in enumerate(trial_stim_indices):
    neural_matrix[:, bi] = ...
```

iii. The agent considers shared stimulus-presentation indices sufficient temporal alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived directly from `is_change` in the stimulus-presentation table for the presentations belonging to each trial.

ii.
```python
stim_is_change = stim['is_change'][:]
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes identify the SDK's `is_change_event()` processing and use the stored flag as the canonical change marker.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The selected `is_change` values have NaNs replaced by zero and are cast to integer; there is no further derivation.

ii.
```python
change_flags = np.nan_to_num(
    stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The agent reports that all Go trials have one change and Catch trials have none, supporting use of the stored flags.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary; zero/NaN maps to category 0 (`no_change`) and one maps to category 1 (`change`). No numerical threshold is fitted.

ii.
```python
change_value_names = ['no_change', 'change']
```

iii. The source is a binary event flag, so the agent considered additional thresholding unnecessary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Each selected presentation's flag is paired with the neural mean for that same presentation bin.

ii.
```python
change_flags = ...[trial_stim_indices]
# neural_matrix[:, bi] is formed from the corresponding si in trial_stim_indices
```

iii. The notes validate an approximately 7.5% changed-presentation rate, consistent with one change presentation on most Go trials.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses NWB running `speed/data` and `speed/timestamps`.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The notes identify this as the Allen processed/filtered running-speed stream in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Native running samples are averaged independently within each 750 ms presentation window using cumulative sums, then the means are discretized using global percentile edges collected in a first pass.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, stim_start, side='left')
run_bin_ends = np.searchsorted(running_ts, stim_start + bin_duration, side='left')
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
running_disc, _ = discretize_values(ot['running_speed_raw'], 5, running_bin_edges)
```

iii. The agent says presentation averaging matches neural bins and global percentiles balance the decoder classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. All presentation means from all retained experiments define the 0, 20, 40, 60, 80, and 100 percentiles. Interior edges are applied with `np.digitize`, yielding five clipped categories.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(valid, percentiles)
binned = np.digitize(values, bin_edges[1:-1]).astype(np.int64)
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The explicit rationale is roughly equal class frequency and consistent categories across the whole dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running and ophys samples are independently bounded by the same stimulus start and start+0.75 s times and averaged, rather than running being interpolated to individual ophys timestamps.

ii.
```python
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts)
run_bin_starts = np.searchsorted(running_ts, all_stim_starts)
```

iii. The notes rely on hardware synchronization and common presentation windows to claim alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking pupil ellipse `area`, eye timestamps, and `likely_blink`; it does not use the stored `pupil_width` field.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The notes choose an equivalent-circle diameter computed from tracked pupil area and use the Allen blink flag for quality control.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink and nonpositive-area samples become NaN; diameter is `2*sqrt(area/pi)`; missing samples are linearly interpolated by sample index; valid values are averaged within presentation windows and discretized globally. Trial-level residual NaNs are interpolated, or assigned the middle bin when all are missing.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
...
pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. The agent says blink interpolation prevents missing/artifactual values from contaminating averages and diameter is more interpretable than area.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five global percentile categories are computed from non-NaN presentation means. Residual within-trial gaps are interpolated; an entirely missing trial is assigned category 2.

ii.
```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
_, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
...
pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
```

iii. Percentile bins are justified as balanced decoder classes; the middle category is used as a neutral fallback for wholly missing trials.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Eye and ophys samples are independently selected and averaged over the same stimulus-onset-to-750-ms windows; pupil is not first interpolated onto each ophys frame.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
```

iii. The agent relies on synchronized clocks and shared presentation boundaries.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It uses the trials-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. The notes call these the four task-defined outcomes and validate their relationship to Go and Catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Priority-ordered booleans map to hit=0, miss=1, false alarm=2, correct rejection=3; an unmatched trial defaults to miss. The static code is repeated across every trial time bin.

ii.
```python
if trial_hit[trial_idx]: outcome = 0
elif trial_miss[trial_idx]: outcome = 1
elif trial_fa[trial_idx]: outcome = 2
elif trial_cr[trial_idx]: outcome = 3
else: outcome = 1
...
np.full(n_bins, ot['trial_outcome'], dtype=np.int64)
```

iii. Replication makes the static output compatible with the decoder's common time-varying output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/invalid pupil samples are interpolated, all-missing pupil trials use the middle category, absent eye tracking leaves NaNs for later fallback, empty trials and zero-neuron experiments are skipped, unknown image names default to image code 0, NaN change flags become zero, and experiments with fewer than two trials are skipped. There is no per-experiment exception handler in the full loop.

ii.
```python
result[~valid] = np.interp(x[~valid], x[valid], values[valid])
image_indices = np.array([img_to_idx.get(img, 0) for img in images])
change_flags = np.nan_to_num(..., nan=0)
if result is None or result['n_trials'] < 2:
    continue
```

iii. The rationale is to preserve usable data and always emit categorical labels; middle-bin pupil filling is described as neutral.

## 9-a. What are the most time-consuming steps of the code?

i. Opening and fully processing every large NWB twice dominates runtime. The notes measured pass 1 and pass 2 separately and estimated roughly 24 minutes; pass 2 was slower. Full neural arrays, behavioral arrays, cumulative sums, and per-trial bins are rebuilt in both passes.

ii.
```python
for ... in active_exps.iterrows():
    stats = process_experiment(..., collect_stats_only=True)
...
for ... in active_exps.iterrows():
    result = process_experiment(..., collect_stats_only=False)
```

iii. The notes focus on cumulative sums as an optimization but their timing table shows repeated full experiment processing as the principal cost.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial search for presentation indices, the per-presentation neural/running/pupil averaging loops, omitted-image fill loops, categorical image mapping, and repeated list-index lookups could be vectorized or pre-grouped. Cumulative sums vectorize arithmetic but not these Python loops.

ii.
```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
for bi, si in enumerate(trial_stim_indices):
    neural_matrix[:, bi] = ...
for bi in range(len(images)):
    if images[bi] == 'omitted': ...
```

iii. The agent labels bin averaging “vectorized” because it uses cumulative sums; it does not discuss eliminating the remaining per-bin loops.

## 9-c. What processing does the code repeat multiple times?

i. `process_experiment` is called twice per experiment. Both passes reload neural, ROI, stimulus, trial, running, and pupil data; recompute diameter/interpolation, all bin boundaries and cumulative sums; and traverse every trial and presentation. The first pass retains only running/pupil means.

ii.
```python
stats = process_experiment(..., collect_stats_only=True)
...
result = process_experiment(..., collect_stats_only=False)
```

iii. The agent justifies two passes as necessary to obtain global percentile edges before final discretization, without caching first-pass results.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. During the statistics-only pass it still loads and filters dF/F, creates its cumulative sum, computes neural presentation means, image identities/change flags, and trial outcomes, then discards all of them. It also loads unused trial start/stop/change arrays and `stim_stop`, `stim_omitted`, `trial_go`, and `trial_catch` intermediates beyond filtering. Optional plots, if requested, are diagnostic only.

ii.
```python
dff_cumsum = np.cumsum(dff_valid, axis=0)
neural_matrix[:, bi] = ...
image_indices = ...
change_flags = ...
...
if collect_stats_only:
    continue
```

iii. The notes do not acknowledge these discarded computations; they present the first pass simply as collection of running/pupil statistics.
