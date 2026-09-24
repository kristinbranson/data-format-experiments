# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment metadata CSV, maps every locally available NWB filename to an experiment ID, intersects the table with those IDs, and then restricts the conversion to non-passive experiments. It opens each selected NWB directly with `h5py`; it does not use the SDK cache or combine experiments belonging to one ophys session.

ii.
```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
active_exps = our_exps[our_exps['passive'] == False].copy()
...
with h5py.File(nwb_path, 'r') as f:
```

iii. The notes say the local data contain 284 NWBs but only 202 active experiments; passive experiments were excluded because they lack meaningful hit/miss outcomes. Direct NWB access was considered equivalent to SDK loading.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique `mouse_id` strings in the selected experiment table, and each output session gets the corresponding index.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
subject_idx = subjects_list.index(result['mouse_id'])
```

iii. The agent treated `mouse_id` as the unique animal identifier and reported 38 mice in the available active subset.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id`/NWB imaging plane is emitted as a separate decoder session. Experiments sharing an `ophys_session_id` are not grouped.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    result = process_experiment(nwb_map[eid], row, collect_stats_only=False)
    all_sessions_neural.append(result['neural_trials'])
```

iii. The trajectory explicitly reasons that each NWB has its own neurons and therefore should be a separate decoder session. It later noticed that this inflated one mouse to 34 sessions because mesoscope planes were split, but did not revise the choice.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `trials` interval table, but their timepoints are the stimulus presentations whose `trials_id` equals the trial ID. Thus each trial becomes a variable-length sequence of 750 ms presentation bins, rather than all native ophys frames between trial start and stop.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
```

iii. The agent chose one bin per 250 ms image plus 500 ms gray interval as a common task-aligned grid across 11 Hz and 31 Hz recordings.

## 1-e. How are trials filtered based on quality controls?

i. Only rows marked go or catch are retained; aborted and auto-rewarded rows are excluded. Trials with no linked stimulus presentations are skipped, experiments with no valid neurons are skipped, and experiments with fewer than two resulting trials are omitted.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
if len(trial_stim_indices) == 0:
    continue
...
if result is None or result['n_trials'] < 2:
    continue
```

iii. This directly follows the requested go/catch inclusion and aborted/auto-reward exclusion. The notes report exact expected 87.5%/12.5% go/catch proportions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is read from the precomputed NWB dF/F trace data and timestamps, with the image-segmentation `valid_roi` mask supplying the retained columns.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. The agent selected precomputed dF/F as a standard decoder signal even though it noted that the paper's analyses used calcium events.

## 2-b. How is the `neural` data processed?

i. Valid-ROI dF/F is averaged within every `[stimulus onset, onset + 0.75 s)` interval. Cumulative sums accelerate the averages. Neurons from simultaneous planes are not stacked because each plane is processed separately.

ii.
```python
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
dff_cumsum = np.vstack([np.zeros((1, n_neurons)), np.cumsum(dff_valid, axis=0)])
neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. A 750 ms presentation unit was chosen to provide a uniform timebase across microscope frame rates and natural alignment to stimulus labels.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `valid_roi == True` columns are retained; an experiment with zero such cells is skipped. No further trace-level quality test or normalization is applied.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
if n_neurons == 0:
    return None
```

iii. The notes cite the Allen ROI classifier/QC criteria and state this matches the SDK's default invalid-ROI exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment uses ophys timestamps to locate frames relative to each stimulus-presentation onset. Each trial begins with its first linked presentation, not explicitly at `trial.start_time`; every output column represents the following 750 ms.

ii.
```python
all_stim_starts = stim_start
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
trial_stim_indices = np.where(stim_trials_id == tid)[0]
```

iii. The agent interpreted “temporally align based on ophys timestamp” as using ophys timestamps to average frames into stimulus-aligned bins.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 750 ms. Native 11 Hz or 31 Hz dF/F frames are rebinned by averaging all frames within each presentation interval.

ii.
```python
TIME_BIN_MS = 750.0
bin_duration = TIME_BIN_MS / 1000.0
```

iii. The notes call this the common natural task unit and use it to avoid differing native ophys frame rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from the stimulus-presentation table's `image_name`, linked to trials by `trials_id`.

ii.
```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in stim['image_name'][:]])
images = stim_image_name[trial_stim_indices].copy()
```

iii. The notes preferred presentation-level labels because each output bin corresponds to one presentation.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Omitted presentations are forward-filled from the preceding image (or backfilled from the next non-omitted image at the start). Names are mapped to global integer codes, but the global vocabulary is discovered from only the first ten selected experiments.

ii.
```python
sample_exps = active_exps.head(min(10, len(active_exps)))
...
if images[bi] == 'omitted':
    if bi > 0: images[bi] = images[bi - 1]
image_indices = np.array([img_to_idx.get(img, 0) for img in images])
```

iii. Forward filling preserves uniform timing while treating an omission as continuation of the prior identity. Sampling ten experiments was an efficiency shortcut; it happened to find both eight-image sets (16 names).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. One presentation label is assigned to the same 750 ms bin over which neural activity is averaged.

ii.
```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
images = stim_image_name[trial_stim_indices].copy()
n_bins = len(trial_stim_indices)
```

iii. The agent considered this exact stimulus-bin alignment natural for decoding, including the subsequent gray period in the labeled bin.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It comes directly from the stimulus presentation table's `is_change` field for presentations linked to the trial.

ii.
```python
stim_is_change = stim['is_change'][:]
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes describe this as one at the change flash and zero for catch/non-change presentations.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Missing flags are replaced by zero and the values are cast to integer; no duration or transition is recomputed from image identities.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The NWB flag was treated as the authoritative precomputed change annotation.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary in NWB, so values are simply encoded as integer 0 (`no_change`) or 1 (`change`); NaN becomes 0.

ii.
```python
change_value_names = ['no_change', 'change']
change_flags = np.nan_to_num(..., nan=0).astype(np.int64)
```

iii. No numerical threshold was needed because `is_change` is a binary source field.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The flag for a stimulus presentation is placed in the same 750 ms presentation bin as its averaged neural response.

ii.
```python
change_flags = stim_is_change[trial_stim_indices]
neural_matrix = np.zeros((n_neurons, len(trial_stim_indices)))
```

iii. This makes “right after a change” the entire change-presentation bin.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses the NWB running processing module's speed samples and timestamps.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The notes identify this as the SDK-filtered wheel-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed samples are averaged over each 750 ms presentation window using cumulative sums, then globally discretized using quintile edges collected in a first pass over all selected experiments.

ii.
```python
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
running_disc, _ = discretize_values(ot['running_speed_raw'], 5, running_bin_edges)
```

iii. Averaging matches the common presentation grid; global percentiles were chosen for approximately balanced, consistent decoder classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The pooled binned speeds are split at 0, 20, 40, 60, 80, and 100 percentiles (outer edges replaced by infinities), and `np.digitize` assigns codes 0–4.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(valid, percentiles)
bin_edges[0], bin_edges[-1] = -np.inf, np.inf
binned = np.digitize(values, bin_edges[1:-1])
```

iii. Equal-percentile bins satisfy the requested five categories and balance their global frequencies.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running timestamps are independently searched against the same stimulus-onset and onset-plus-750-ms boundaries used for neural bins, then samples in each interval are averaged.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
```

iii. The hardware clocks were considered synchronized, so matching absolute presentation boundaries aligns the streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It is derived from eye-tracking pupil `area`, its timestamps, and `likely_blink`, rather than the available `pupil_width` used by the reference.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The agent chose the equal-area circular diameter as a common definition, despite discussing the ellipse major-axis/width alternative.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink, nonpositive, and negative area samples become NaN; diameter is `2*sqrt(area/pi)`; NaNs are linearly interpolated by sample index; diameters are averaged in 750 ms bins and later discretized. Residual within-trial NaNs are interpolated, while an all-NaN trial is assigned the middle category.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
...
pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. The notes say blink interpolation prevents artifacts and area-derived diameter is reasonable; bin averaging puts pupil on the shared presentation grid.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Valid pooled 750 ms means are divided by global quintile boundaries and coded 0–4. A trial that remains entirely missing is forced to code 2.

ii.
```python
_, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    pupil_disc, _ = discretize_values(pupil_raw, 5, pupil_bin_edges)
```

iii. Global quintiles provide consistent balanced labels; the middle bin was selected as a neutral fallback for wholly missing trials.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Eye-tracking timestamps are searched against the same absolute 750 ms presentation windows, and valid pupil samples in each window are averaged.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
```

iii. The agent relied on hardware synchronization and common presentation boundaries.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It uses the trial table's `hit`, `miss`, `false_alarm`, and `correct_reject` booleans.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. These are the canonical mutually exclusive outcomes for retained go and catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes are assigned fixed codes hit=0, miss=1, false alarm=2, correct rejection=3, then repeated across all bins of the trial. An unmatched row defaults to miss.

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

iii. Repetition adapts the static label to the decoder's rectangular time-varying output. The fallback avoids an invalid category.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/blink pupil samples are interpolated; a wholly missing pupil trial gets the middle class. Missing change flags become zero, unknown image names map to image code zero, omitted images are filled from neighbors, empty trials are skipped, zero-neuron/fewer-than-two-trial experiments are skipped, and an unclassified outcome defaults to miss. File/session exceptions are not caught, so an unexpected malformed NWB can terminate the run.

ii.
```python
change_flags = np.nan_to_num(..., nan=0)
image_indices = np.array([img_to_idx.get(img, 0) for img in images])
if nan_mask.all(): pupil_disc = np.full(n_bins, 2)
if len(trial_stim_indices) == 0: continue
```

iii. The agent justified interpolation as artifact repair and categorical defaults as safe ways to keep decoder-compatible integers; its validation found no NaN/Inf in the produced neural arrays.

## 9-a. What are the most time-consuming steps of the code?

i. Opening and reading every large NWB twice, loading full dF/F arrays, computing trial/bin summaries, and finally serializing the roughly 386 MB pickle dominate runtime. The notes measured about 510 seconds for conversion.

ii.
```python
for ... in active_exps.iterrows():
    stats = process_experiment(..., collect_stats_only=True)
...
for ... in active_exps.iterrows():
    result = process_experiment(..., collect_stats_only=False)
```

iii. The agent initially projected about 24 minutes and optimized bin averaging with cumulative sums/searchsorted, reducing the full run to about 8.5 minutes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial and per-presentation loops remain. Neural, running, and pupil averages each loop over `trial_stim_indices`; omitted-image filling and image-code lookup also use Python loops. Grouped reductions over precomputed start/end arrays could reduce this overhead.

ii.
```python
for trial_idx in np.where(trial_mask)[0]:
    for bi, si in enumerate(trial_stim_indices):
        neural_matrix[:, bi] = ...
```

iii. The agent recognized bin loops as a bottleneck and introduced cumulative sums and vectorized boundary searches, but did not eliminate the loops themselves.

## 9-c. What processing does the code repeat multiple times?

i. `process_experiment` is run twice for every experiment. The statistics-only pass still loads/filter dF/F, trials, stimuli, running and pupil data and computes neural matrices and outcomes before discarding most results; the second pass repeats that work to build the dataset.

ii.
```python
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The two-pass design was chosen so global running/pupil quintiles are known before final categorical outputs are assembled.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In pass 1 it loads dF/F, applies ROI filtering, builds cumulative sums, calculates every trial's neural matrix, image labels/change flags, and outcome even though only running and pupil values are returned. It also reads unused trial start/stop/change-time arrays and stimulus stop/omitted arrays. Optional plotting receives `nwb_path` and `session_idx` without using them.

ii.
```python
neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
...
if collect_stats_only:
    continue
...
return {'running_values': running_values, 'pupil_values': pupil_values}
```

iii. No explicit justification was supplied for these discarded computations; they are a consequence of sharing one broad processing function between the statistics and conversion passes.
