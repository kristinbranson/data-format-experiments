# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment metadata CSV, finds locally available NWB files, intersects the metadata with those file IDs, filters to active (`passive == False`) experiments, and opens each file directly with `h5py`. Thus “all” means all locally available active experiment files, not every experiment listed by the Allen SDK.

ii.
```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
active_exps = our_exps[our_exps['passive'] == False].copy()
...
with h5py.File(nwb_path, 'r') as f:
```

iii. The notes say only 284 NWBs were present locally and resolve the mismatch with the full release by processing that available subset. They exclude passive experiments because meaningful trial outcomes require active behavior.

## 1-b. How are the data split into subjects?

i. Subjects are unique metadata `mouse_id` values, converted to strings and sorted. Each retained experiment is mapped back to the corresponding index.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
subject_idx = subjects_list.index(result['mouse_id'])
```

iii. The agent treats the metadata mouse ID as the animal identifier, consistent with the dataset schema.

## 1-c. How are the data split into sessions?

i. Each NWB `ophys_experiment_id` (one imaging plane) becomes an independent output session. Simultaneously acquired planes sharing an `ophys_session_id` are not grouped.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    result = process_experiment(nwb_map[eid], row, collect_stats_only=False)
    ...
    all_sessions_neural.append(result['neural_trials'])
```

iii. The notes explicitly observe that each NWB is one imaging plane, but nevertheless call each processed experiment a session. The likely motivation was direct processing of the 284 local NWBs; no justification is given for failing to combine planes from the same behavioral session.

## 1-d. How are the data split into trials?

i. The agent uses the NWB trials table to select trials, then associates every stimulus presentation whose `trials_id` equals that trial’s ID. Each such presentation becomes one 750 ms time bin; trials therefore have one column per associated presentation.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
```

iii. The notes identify Go and Catch as required and use the stimulus-presentation interval as the “natural task unit.”

## 1-e. How are trials filtered based on quality controls?

i. Only Go or Catch trials are retained; Aborted and Auto-rewarded trials are removed. Trials with no linked stimulus presentations are skipped, experiments with no valid neurons are skipped, and output sessions with fewer than two trials are skipped.

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

iii. This directly follows the task instruction to include Go/Catch and exclude Aborted/Auto-rewarded trials. The minimum-trial rule satisfies decoder validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB `processing/ophys/dff/traces/data` array and the image-segmentation table’s `valid_roi` flag.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. The agent chose precomputed dF/F as a standard decoder signal even though its notes acknowledge that the paper’s neural analyses used inferred calcium events.

## 2-b. How is the `neural` data processed?

i. Invalid ROIs are removed. For each 750 ms stimulus interval, valid-neuron dF/F samples are averaged using cumulative sums, producing neurons-by-presentations trial matrices. No normalization or other filtering is applied.

ii.
```python
dff_cumsum = np.vstack([np.zeros((1, n_neurons)), np.cumsum(dff_valid, axis=0)])
...
neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The agent says averaging over one image-plus-gray cycle makes temporal resolution consistent across 11 Hz and 31 Hz equipment and uses cumulative sums for speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `valid_roi == True` are retained; files with zero such neurons are skipped. There is no additional neuron/session QC in the script.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
if n_neurons == 0:
    return None
```

iii. The notes state that this matches the Allen SDK default and summarizes the upstream ROI-curation rules represented by `valid_roi`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are aligned to each stimulus-presentation onset, not to one trial-level event. Frames in `[stimulus start, stimulus start + 0.75 s)` are averaged, and the linked presentation sequence defines the trial.

ii.
```python
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
```

iii. The agent calls stimulus onset the natural alignment event because the task repeats a 250 ms image and 500 ms gray cycle.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is fixed at 750 ms. Native ophys frames are explicitly rebinned by averaging all samples within each stimulus-presentation interval.

ii.
```python
TIME_BIN_MS = 750.0
bin_duration = TIME_BIN_MS / 1000.0
...
'time_bin_size': TIME_BIN_MS,
```

iii. The notes justify this as one complete stimulus cycle and a way to unify mesoscope and single-plane frame rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `image_name` in the selected stimulus-presentations interval and the presentation-to-trial `trials_id` association.

ii.
```python
stim_image_name = np.array([... for x in stim['image_name'][:]])
stim_trials_id = stim['trials_id'][:]
images = stim_image_name[trial_stim_indices].copy()
```

iii. The notes identify `stimulus_presentations.image_name` as the most direct time-varying source.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are encoded using a sorted global list sampled from up to ten experiments. Omitted presentations are forward-filled; a leading omission is filled from the next non-omitted image. Unrecognized names default to code 0.

ii.
```python
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
...
if images[bi] == 'omitted':
    if bi > 0:
        images[bi] = images[bi - 1]
...
image_indices = np.array([img_to_idx.get(img, 0) for img in images])
```

iii. The agent says omitted screens should retain the prior image identity while neural and behavioral samples remain present. Sampling ten experiments was an optimization based on the belief that sessions share the same image sets.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. There is one image code for each `trial_stim_indices` entry and one neural average for that same entry, so columns correspond by construction.

ii.
```python
n_bins = len(trial_stim_indices)
for bi, si in enumerate(trial_stim_indices):
    ...
images = stim_image_name[trial_stim_indices].copy()
```

iii. The shared stimulus-presentation indices were intended to guarantee alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It comes directly from the stimulus-presentations table’s `is_change` values for the selected trial presentations.

ii.
```python
stim_is_change = stim['is_change'][:]
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes identify `is_change` as the SDK/NWB change-event indicator and expect roughly one change presentation per Go trial.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Selected flags have NaNs replaced with zero and are cast to integers; no change is otherwise inferred from consecutive identities.

ii.
```python
change_flags = np.nan_to_num(
    stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The agent relies on the already processed NWB change flag instead of recomputing it.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is not numerically thresholded. Existing boolean/0–1 `is_change` values become category 0 (`no_change`) or 1 (`change`), with missing values assigned 0.

ii.
```python
change_value_names = ['no_change', 'change']
change_flags = np.nan_to_num(..., nan=0).astype(np.int64)
```

iii. Since the raw field is already binary, the agent considered further thresholding unnecessary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The flag and neural average are selected using the same stimulus index, giving one flag per 750 ms neural bin.

ii.
```python
for bi, si in enumerate(trial_stim_indices):
    neural_matrix[:, bi] = ...
change_flags = stim_is_change[trial_stim_indices]
```

iii. Alignment is based on the common presentation sequence.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses the NWB processed running `speed/data` and its timestamps.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The notes describe this as the dataset’s filtered running-speed stream in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Samples in each 750 ms presentation window are averaged via cumulative sums. Global quintile edges are learned in a first pass over all selected trials, and the averages are discretized in a second pass.

ii.
```python
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
...
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
running_disc, _ = discretize_values(ot['running_speed_raw'], 5, running_bin_edges)
```

iii. The agent chose global percentiles for consistent labels and approximately balanced decoder classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Global 0th, 20th, 40th, 60th, 80th, and 100th percentiles define five categories. The endpoints are changed to infinities and `np.digitize` assigns codes 0–4.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(valid, percentiles)
bin_edges[0] = -np.inf
bin_edges[-1] = np.inf
binned = np.digitize(values, bin_edges[1:-1]).astype(np.int64)
```

iii. This implements the requested five equal-percentile bins globally.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running timestamps are independently searched against the same stimulus start and end times used for neural bins; all running samples in each interval are averaged.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
```

iii. The agent relies on hardware-synchronized clocks and common interval boundaries.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses EyeTracking pupil ellipse `area`, EyeTracking timestamps, and the `likely_blink` mask.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The notes choose area converted to equivalent-circle diameter and use the dataset’s blink detector.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink and nonpositive areas become NaN; area is converted to equivalent diameter as `2*sqrt(area/pi)`; missing values are linearly interpolated; each 750 ms window is averaged over valid samples; global quintile bins are then applied. Missing eye tracking yields NaN bins initially.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
...
pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. The agent intended to prevent blink artifacts from biasing averages while retaining trials by interpolation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Valid globally collected binned diameters define five percentile bins. Per-trial residual NaNs are interpolated; an entirely NaN trial is assigned the middle category (2).

ii.
```python
_, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, 5, pupil_bin_edges)
```

iii. Global percentiles were chosen for balanced classes; middle-bin imputation is described as the fallback for absent pupil data.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil timestamps are searched against the same 750 ms stimulus boundaries used for neural data, and pupil values in each interval are averaged.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
```

iii. The common synchronized time boundaries are intended to align pupil and neural bins.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the trials table’s `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. The agent treats these as the canonical mutually exclusive outcomes for retained active trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. A priority-ordered conditional maps hit/miss/false alarm/correct rejection to 0/1/2/3. If none is true, it defaults to Miss (1). The code then repeats the static code across all time bins.

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

iii. Replication makes the static outcome compatible with the common `(outputs, time)` array. The Miss fallback is a defensive choice but is not substantively justified.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Trials/files lacking stimuli or valid neurons are skipped. Blink/nonpositive pupil samples are interpolated; partially missing trial pupil bins are interpolated and fully missing ones become the middle category. Missing change flags become zero, unknown image names become image code zero, and trials with no recognized outcome become Miss. Empty or very small sessions are skipped. The code does not wrap file processing in exception handling, so other malformed/missing fields abort conversion.

ii.
```python
if n_neurons == 0: return None
if len(trial_stim_indices) == 0: continue
change_flags = np.nan_to_num(..., nan=0)
image_indices = np.array([img_to_idx.get(img, 0) for img in images])
if nan_mask.all(): pupil_disc = np.full(n_bins, 2)
```

iii. The agent’s general policy is to preserve usable trials with interpolation/default categories while skipping unusable units. Notes emphasize blink interpolation and sanity checks but do not justify all default labels.

## 9-a. What are the most time-consuming steps of the code?

i. Every active NWB is fully opened and processed twice: once to collect running/pupil statistics and once to construct outputs. Loading large dF/F arrays and computing their cumulative sums during both passes dominate; plotting is optional.

ii.
```python
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The notes estimate about 24 minutes for 202 experiments and explicitly describe the two-pass design.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial/per-presentation loops for neural, running, and pupil interval means could be replaced by batched indexed cumulative-sum differences. Omitted-image filling, image encoding, and subject/region `.index()` lookups could also be vectorized or mapped.

ii.
```python
for trial_idx in np.where(trial_mask)[0]:
    ...
    for bi, si in enumerate(trial_stim_indices):
        neural_matrix[:, bi] = ...
    for bi, si in enumerate(trial_stim_indices):
        running_binned[bi] = ...
    for bi, si in enumerate(trial_stim_indices):
        pupil_binned[bi] = ...
```

iii. The agent claims cumulative sums avoid expensive boolean masking and calls averaging “vectorized,” but the outer assignment loops remain.

## 9-c. What processing does the code repeat multiple times?

i. `process_experiment` repeats NWB loading, dF/F filtering, stimulus/trial extraction, pupil cleanup, boundary searches, cumulative sums, and per-trial bin averaging in both passes. The first pass computes neural matrices and outcomes even though it returns only running and pupil values.

ii.
```python
for ... in active_exps.iterrows():
    stats = process_experiment(..., collect_stats_only=True)
...
for ... in active_exps.iterrows():
    result = process_experiment(..., collect_stats_only=False)
```

iii. The two passes are justified as necessary to learn global percentile edges before final discretization, but intermediate data are not cached.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In the statistics-only pass it loads and cumulative-sums all neural dF/F, creates every trial’s neural matrix, encodes images/change/outcome, and creates empty inputs, then discards all except running and pupil lists. Trial start/stop/change arrays are loaded but never used. Plotting, when requested, is diagnostic and not part of the saved dataset.

ii.
```python
neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
...
if collect_stats_only:
    continue
...
trial_start = trials['start_time'][:]
trial_stop = trials['stop_time'][:]
trial_change_time = trials['change_time'][:]
```

iii. The notes mention the two-pass scheme and cumulative-sum optimization, but do not acknowledge that the stats-only pass performs most full neural/output work before discarding it.
