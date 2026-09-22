# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the local experiment metadata CSV, discovers locally available NWB files, intersects the two by `ophys_experiment_id`, filters to non-passive experiments, and opens every selected NWB directly with `h5py`. It makes two full passes over the experiments, plus a small image-name scan.

ii.
```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
active_exps = our_exps[our_exps['passive'] == False].copy()
...
with h5py.File(nwb_path, 'r') as f:
```

iii. The notes say only 284 NWBs were locally available, so the AI intentionally processed that subset and active sessions only because passive sessions lack meaningful trial outcomes. Direct NWB access was chosen after inspecting the SDK’s NWB mappings.

## 1-b. How are the data split into subjects?

i. Subjects are unique metadata `mouse_id` values, converted to strings and sorted. Each retained experiment receives the corresponding index.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
subject_idx = subjects_list.index(result['mouse_id'])
```

iii. The notes identify `mouse_id` as the animal identifier and report 38 mice in the local active subset.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id`/NWB file is emitted as a separate target session. Simultaneously recorded planes sharing an `ophys_session_id` are not merged.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    result = process_experiment(nwb_map[eid], row, collect_stats_only=False)
    all_sessions_neural.append(result['neural_trials'])
```

iii. The notes explicitly describe each NWB as one imaging plane, but nevertheless call each processed experiment a session. The rationale emphasizes the locally available experiment files rather than reconstruction of multi-plane sessions.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. For each retained trial ID, the AI selects all stimulus-presentation rows whose `trials_id` equals that trial ID; those presentations become the trial’s time bins.

ii.
```python
trials = f['intervals']['trials']
...
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
```

iii. The AI viewed a stimulus presentation (250 ms image plus 500 ms gray) as the natural task unit and retained all such presentations associated with each SDK-defined trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI retains go or catch trials and excludes aborted and auto-rewarded trials. Trials with no linked stimulus presentations are skipped, and experiments with fewer than two resulting trials are omitted.

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

iii. This is justified directly by the task instruction to include go/catch and exclude aborted/auto-rewarded trials, plus the format’s two-trial minimum.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB precomputed dF/F trace matrix and the image-segmentation table’s `valid_roi` flags.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. The notes acknowledge that the paper analyzed inferred events, but choose dF/F as a standard, precomputed decoding signal.

## 2-b. How is the `neural` data processed?

i. Invalid ROIs are removed. For every 750 ms presentation interval, dF/F frames are averaged using cumulative sums, and the result is transposed into neuron-by-bin form. No further normalization is applied.

ii.
```python
dff_cumsum = np.vstack([np.zeros((1, n_neurons)), np.cumsum(dff_valid, axis=0)])
...
neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The AI chose 750 ms as a common natural task unit across 11 Hz and 31 Hz equipment and used cumulative sums for speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `valid_roi=True` are included; experiments with zero such neurons are skipped. No session-level imaging QC is independently applied.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
if n_neurons == 0:
    return None
```

iii. The notes state this matches the SDK’s default curation, which removes invalid/non-cell ROIs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are aligned to every stimulus-presentation onset, not one fixed trial event. Ophys indices cover `[stimulus_start, stimulus_start + 0.75)` for each presentation linked to the trial.

ii.
```python
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
```

iii. The notes call stimulus presentation onset the alignment event because presentations are the chosen time units.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted resolution is 750 ms. Native ophys frames are rebinned by averaging all frames in each presentation interval.

ii.
```python
TIME_BIN_MS = 750.0
bin_duration = TIME_BIN_MS / 1000.0
...
'time_bin_size': TIME_BIN_MS,
```

iii. The AI argues that one 250 ms image plus 500 ms gray interval is the natural task unit and gives consistent resolution across microscopes.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `image_name`, `omitted`, and `trials_id` in the detected stimulus-presentation interval table.

ii.
```python
stim_image_name = np.array([... for x in stim['image_name'][:]])
stim_omitted = stim['omitted'][:]
stim_trials_id = stim['trials_id'][:]
```

iii. The AI chose the presentation table because it directly describes the time-varying stimulus sequence.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Global image names are inferred from at most ten experiments, sorted, and integer encoded. An `omitted` label is forward-filled from the prior image, or backfilled from the next non-omitted image if it is first.

ii.
```python
sample_exps = active_exps.head(min(10, len(active_exps)))
...
if images[bi] == 'omitted':
    if bi > 0:
        images[bi] = images[bi - 1]
image_indices = np.array([img_to_idx.get(img, 0) for img in images])
```

iii. The notes say the dataset uses fixed eight-image sets and that forward-filling preserves image identity during omitted flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. One image code is taken for each `trial_stim_indices` element, exactly matching the corresponding 750 ms neural bin.

ii.
```python
images = stim_image_name[trial_stim_indices].copy()
n_bins = len(trial_stim_indices)
neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
```

iii. Both streams use the same ordered stimulus-presentation indices, so their columns match.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It comes directly from the stimulus table’s `is_change` values for presentations belonging to a trial.

ii.
```python
stim_is_change = stim['is_change'][:]
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes identify the SDK’s `is_change` as the reference change-event indicator.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Relevant flags are selected, NaNs become zero, and values are cast to integers.

ii.
```python
change_flags = np.nan_to_num(
    stim_is_change[trial_stim_indices], nan=0
).astype(np.int64)
```

iii. The AI intended one change flag per stimulus bin and no reconstructed transition logic.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numerical threshold is applied: the existing boolean/0–1 `is_change` flag is cast to integer, with missing flags treated as no change.

ii.
```python
change_flags = np.nan_to_num(..., nan=0).astype(np.int64)
change_value_names = ['no_change', 'change']
```

iii. The source is already categorical, so the AI preserves it.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Each selected presentation contributes both one change flag and one neural average over its 750 ms interval.

ii.
```python
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
...
change_flags = stim_is_change[trial_stim_indices]
```

iii. Shared presentation indices guarantee column-level alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed uses the NWB processing group’s filtered speed data and timestamps.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The notes identify this as the SDK-provided, low-pass-filtered wheel speed in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Samples in each 750 ms presentation interval are averaged by cumulative sums. Global quintile edges are learned during a first full pass, then applied in a second pass.

ii.
```python
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
...
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
running_disc, _ = discretize_values(ot['running_speed_raw'], 5, running_bin_edges)
```

iii. The AI says global percentiles balance decoder classes and keep categories consistent across experiments.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Dataset-wide 0/20/40/60/80/100 percentile edges form five bins via `np.digitize`; endpoints are replaced by infinities.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(valid, percentiles)
bin_edges[0], bin_edges[-1] = -np.inf, np.inf
binned = np.digitize(values, bin_edges[1:-1])
```

iii. Equal-percentile bins were required and chosen globally to approximate equal class frequency.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running timestamps are independently searched against the same presentation start/end times, and samples within each neural bin’s 750 ms interval are averaged.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
```

iii. The notes rely on hardware-synchronized clocks and identical temporal boundaries.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It is computed from eye-tracking pupil `area`, timestamps, and the `likely_blink` mask.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The notes choose area-derived diameter and use the SDK-provided blink QC.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink and nonpositive areas become NaN; diameter is `2*sqrt(area/pi)`; NaNs are linearly interpolated; values are averaged per 750 ms bin and globally quintile-discretized. Trial-level residual NaNs are interpolated, or assigned the middle bin if all are missing.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
...
pupil_disc, _ = discretize_values(pupil_raw, 5, pupil_bin_edges)
```

iii. The AI says this avoids blink artifacts, yields an actual diameter measure, and balances the five decoder classes.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five global percentile bins are computed over non-NaN binned diameters. Missing whole trials receive category 2; otherwise residual missing values are interpolated before digitization.

ii.
```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
_, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
...
pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
```

iii. The rationale is equal-frequency categories with a neutral middle category when pupil data are wholly unavailable.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Eye samples are selected with `searchsorted` using each stimulus bin’s start and end, then averaged into one value per neural column.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. Identical presentation-time boundaries on synchronized clocks provide alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome is derived from the trial table’s `hit`, `miss`, `false_alarm`, and `correct_reject` booleans.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. The notes identify these as the four canonical outcomes for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Priority-ordered conditionals encode hit/miss/false alarm/correct rejection as 0/1/2/3. Unclassified trials default to miss, and the code is repeated across all bins in that trial.

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

iii. Replication makes the static label compatible with the decoder’s common time-varying output matrix. No explicit rationale is given for defaulting unknown outcomes to miss.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI skips zero-neuron experiments, trials without presentations, and experiments with fewer than two trials. Blink/nonpositive pupil samples are interpolated; residual or entirely missing pupil trials use interpolation or the middle category. Missing change flags become zero. Missing image labels fall back to code 0. It does not catch arbitrary file/session exceptions.

ii.
```python
if n_neurons == 0: return None
if len(trial_stim_indices) == 0: continue
change_flags = np.nan_to_num(..., nan=0)
image_indices = np.array([img_to_idx.get(img, 0) for img in images])
if nan_mask.all(): pupil_disc = np.full(n_bins, 2)
```

iii. The notes justify interpolation as protection against blink artifacts and the middle pupil bin as a neutral fallback; other fallbacks are largely undocumented.

## 9-a. What are the most time-consuming steps of the code?

i. Every experiment is fully read and processed twice. Loading large dF/F arrays, computing cumulative sums, and looping through trials/presentation bins dominate; the notes estimated about 24 minutes for 202 experiments.

ii.
```python
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The notes report measured sample pass times and extrapolate them to the full run.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial search for stimulus IDs and the inner neural/running/pupil loops over presentation bins could be vectorized or grouped. Image omission filling and list/index lookups also remain Python loops.

ii.
```python
for trial_idx in np.where(trial_mask)[0]:
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    for bi, si in enumerate(trial_stim_indices):
        ...
```

iii. The notes describe cumulative sums and precomputed boundaries as optimizations, but do not discuss the remaining loops.

## 9-c. What processing does the code repeat multiple times?

i. `process_experiment` repeats complete NWB loading, ROI filtering, pupil conversion/interpolation, boundary construction, cumulative sums, trial selection, and bin averaging in both passes. Image names are also scanned separately before those passes.

ii.
```python
for ... in active_exps.iterrows():
    process_experiment(..., collect_stats_only=True)
...
for ... in active_exps.iterrows():
    process_experiment(..., collect_stats_only=False)
```

iii. The two-pass design is justified as necessary to obtain dataset-wide percentile edges before final discretization.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In the statistics-only pass it still loads and filters neural dF/F, creates neural cumulative sums and per-trial neural matrices, computes image mappings/change flags/outcomes, and constructs unused inputs; `collect_stats_only` is checked only after those operations. It also reads unused trial start/stop/change arrays and `stim_stop`, `stim_omitted`, and `valid_trial_ids`.

ii.
```python
neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    neural_matrix[:, bi] = ...
...
if collect_stats_only:
    continue
```

iii. No justification is given; this is an implementation inefficiency of sharing one routine between both passes.
