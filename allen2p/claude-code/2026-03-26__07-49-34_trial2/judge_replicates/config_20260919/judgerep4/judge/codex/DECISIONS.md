# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment metadata CSV, maps every locally available NWB filename to its `ophys_experiment_id`, keeps metadata rows with a file, filters to active (`passive == False`) experiments, and opens each experiment directly with `h5py`. Thus “all” means all 202 active experiment files in the supplied 284-file subset, not every experiment listed by the Allen project cache.

ii.
```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
active_exps = our_exps[our_exps['passive'] == False].copy()
...
with h5py.File(nwb_path, 'r') as f:
```

iii. The notes say the local 284 NWBs are a subset of the release and choose active sessions because passive sessions do not provide meaningful trial outcomes. Direct NWB loading was chosen after examining the SDK’s NWB mappings.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique string-valued `mouse_id`s among active experiments. Each retained experiment receives the index of its mouse in that list.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
subject_idx = subjects_list.index(result['mouse_id'])
all_subject_idx.append(subject_idx)
```

iii. The agent treats the metadata `mouse_id` as the unique animal identifier. Its full run reports 38 subjects in the locally available active subset.

## 1-c. How are the data split into sessions?

i. Every NWB/`ophys_experiment_id` is emitted as a separate session. Experiments from different simultaneously recorded imaging planes are not grouped by `ophys_session_id`.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    result = process_experiment(nwb_map[eid], row, collect_stats_only=False)
    ...
    all_sessions_neural.append(result['neural_trials'])
```

iii. The notes explicitly state that each NWB file is one imaging plane, but the implementation nevertheless uses experiments as decoder sessions. The stated motivation is to process the 202 available active experiment files; no justification is given for not reconstructing behavioral sessions from their planes.

## 1-d. How are the data split into trials?

i. The trials table supplies valid trial IDs and outcomes. For each selected trial ID, all stimulus-presentation rows whose `trials_id` equals that ID become that trial’s time bins. Trial `start_time` and `stop_time` are loaded but not used for slicing.

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

iii. The agent describes one 750 ms bin per stimulus presentation as the natural task unit and uses the NWB trial linkage to retain all presentations assigned to each go/catch trial.

## 1-e. How are trials filtered based on quality controls?

i. It includes only rows flagged go or catch, excludes aborted and auto-rewarded rows, skips trials with no linked stimulus presentations, and later skips experiments with fewer than two retained trials. It does not separately require a nonmissing `change_time`.

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

iii. This directly follows the instruction to include go and catch trials while excluding aborted and auto-rewarded trials; the minimum-two rule is required by decoder validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from the NWB precomputed dF/F trace matrix and the image-segmentation table’s `valid_roi` mask.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. The notes acknowledge that the paper used inferred events, but select dF/F because it is precomputed and considered a standard, potentially denser decoding signal.

## 2-b. How is the `neural` data processed?

i. Invalid ROIs are removed. Ophys frames in each `[stimulus onset, onset + 0.75 s)` interval are averaged using cumulative sums, producing neurons-by-stimulus-bin matrices. No normalization beyond the stored dF/F is applied.

ii.
```python
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
dff_cumsum = np.vstack([np.zeros((1, n_neurons)), np.cumsum(dff_valid, axis=0)])
...
neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The 750 ms average was intended to normalize temporal resolution across mesoscope and single-plane equipment and to match the 250 ms image plus 500 ms gray task cycle. Cumulative sums were used for speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `valid_roi=True` are retained; experiments with zero such cells are skipped. No further cell-, trace-, or session-quality filter is applied.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
if n_neurons == 0:
    return None
```

iii. The agent says this matches the SDK default and cites the Allen ROI curation rules (duplicates, edge ROIs, artifacts, invalid traces, and related criteria).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to each stimulus-presentation onset, not to trial start: every column averages the following 750 ms, and the columns linked to a trial are concatenated in presentation order.

ii.
```python
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
...
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
```

iii. The notes call stimulus presentation the natural unit of the task and report hardware synchronization between streams. Metadata records `temporal_alignment_event` as stimulus presentation onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is 750 ms. Native ophys samples (about 11 or 31 Hz) are explicitly rebinned by averaging within each stimulus cycle.

ii.
```python
TIME_BIN_MS = 750.0
bin_duration = TIME_BIN_MS / 1000.0
...
'time_bin_size': TIME_BIN_MS,
```

iii. The agent chose one bin per 250 ms image plus 500 ms gray period to make equipment types consistent and align all variables to image presentations.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It comes from `image_name` in the stimulus-presentation interval table selected by `trials_id`.

ii.
```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in stim['image_name'][:]])
images = stim_image_name[trial_stim_indices].copy()
```

iii. The agent uses the presentation table because it provides the actual time-varying image sequence, including omissions, rather than only the trial’s initial/change labels.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are collected from at most the first ten experiments, excluding `omitted`, sorted globally, and mapped to integers. Omitted bins are forward-filled; a leading omission is filled from the next non-omitted presentation. Unknown names default to category 0.

ii.
```python
sample_exps = active_exps.head(min(10, len(active_exps)))
...
if name != 'omitted': all_images.add(name)
...
if images[bi] == 'omitted':
    if bi > 0: images[bi] = images[bi - 1]
    else: ...
image_indices = np.array([img_to_idx.get(img, 0) for img in images])
```

iii. The notes assume experiments share the same eight-image sets and say forward-filling represents the image identity associated with omitted presentations. Sampling ten files was an efficiency optimization.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. One image category is assigned to each stimulus index, while neural activity is averaged over the same presentation’s 750 ms interval; both arrays use the same ordered `trial_stim_indices`.

ii.
```python
n_bins = len(trial_stim_indices)
images = stim_image_name[trial_stim_indices].copy()
for bi, si in enumerate(trial_stim_indices):
    neural_matrix[:, bi] = ...
```

iii. The agent’s presentation-level representation guarantees equal column counts and direct presentation-by-presentation alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is read from the stimulus-presentation table’s `is_change` field for presentations linked to the trial.

ii.
```python
stim_is_change = stim['is_change'][:]
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes identify the SDK’s `is_change` as the canonical presentation-level change indicator.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The selected flags are converted to integer values, with NaN replaced by zero. There is no inferred comparison of adjacent encoded image identities.

ii.
```python
change_flags = np.nan_to_num(
    stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The source already identifies changes, so the agent considered further processing unnecessary.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is calculated: the existing boolean/0–1 flag is cast to integer, yielding `no_change` and `change` categories.

ii.
```python
change_value_names = ['no_change', 'change']
```

iii. The raw field is already binary; NaN is conservatively treated as no change.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Each `is_change` value labels the neural 750 ms bin beginning at that same presentation onset.

ii.
```python
change_flags = stim_is_change[trial_stim_indices]
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
```

iii. Shared presentation indices are the agent’s alignment mechanism.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the NWB running processing group’s `speed/data` and corresponding timestamps.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The notes describe this as the SDK-standard, low-pass-filtered running-speed stream in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Samples in every 750 ms presentation interval are averaged via cumulative sums. Global quintile edges are learned in a first pass over all retained trial bins, then applied in a second pass.

ii.
```python
run_cumsum = np.concatenate([[0], np.cumsum(running_speed)])
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
...
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
running_disc, _ = discretize_values(ot['running_speed_raw'], 5, running_bin_edges)
```

iii. Averaging matches the chosen stimulus-bin representation; global percentiles were chosen for balanced classes and consistent labels across experiments.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Dataset-wide 0th, 20th, 40th, 60th, 80th, and 100th percentiles define five categories. Outer edges are replaced by infinities, and `np.digitize` assigns labels 0–4.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(valid, percentiles)
bin_edges[0] = -np.inf
bin_edges[-1] = np.inf
binned = np.digitize(values, bin_edges[1:-1]).astype(np.int64)
```

iii. Equal-percentile bins satisfy the requested five equal percentile bins and improve class balance.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running and ophys timestamps are independently searched using identical stimulus onset and onset-plus-750-ms boundaries; their within-bin means occupy the same output column.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
```

iii. The agent relies on hardware-synchronized timestamps and common real-time bin boundaries rather than resampling running speed to every ophys frame.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking pupil ellipse `area`, eye timestamps, and `likely_blink`, then derives an equivalent circular diameter. It does not use the stored pupil-width field used by the reference.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. The notes describe area-to-equivalent-diameter conversion as their definition of pupil diameter and use the provided blink QC.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink and nonpositive area samples become NaN; equivalent diameter is computed, missing samples are linearly interpolated by sample index, valid samples are averaged in 750 ms bins, and global quintiles are applied. Within-trial residual NaNs are interpolated; an entirely missing trial receives the middle category.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = interpolate_nans(2.0 * np.sqrt(pupil_area / np.pi))
...
pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
```

iii. Blink removal prevents artifacts, interpolation preserves continuity, and bin averaging/global quintiles match the agent’s common temporal representation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five global percentile bins are computed from non-NaN binned diameters and applied with `np.digitize`; all-missing trials are assigned bin 2.

ii.
```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
_, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
...
pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The goal is equal-frequency categories and a consistent global mapping; the middle bin is used as a neutral missing-data fallback.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Eye timestamps are searched against the same stimulus onset and 750 ms endpoint as neural data, and the eye samples in that interval are averaged into the corresponding column.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
```

iii. Hardware-synchronized absolute timestamps and identical real-time presentation windows provide alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It uses the trials table’s mutually exclusive `hit`, `miss`, `false_alarm`, and `correct_reject` flags.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. These are the canonical task outcomes documented by the SDK and paper.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Flags are checked in hit/miss/false-alarm/correct-reject order and mapped to 0–3; an unclassified retained trial defaults to miss. The selected scalar is repeated across every time bin.

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

iii. Replication makes the static target compatible with the decoder’s common output matrix while retaining the four canonical classes.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing pupil samples from blinks/nonpositive areas are interpolated; remaining within-trial gaps are interpolated and wholly missing trials use the middle bin. Experiments with no eye tracking create all-NaN pupil bins and consequently middle-bin labels. Empty stimulus-linked trials, zero-cell experiments, and experiments with fewer than two trials are skipped. Unknown images map to 0 and NaN change flags map to 0. There is no per-experiment exception handler, so malformed/unexpected files can terminate the run.

ii.
```python
if has_eye_tracking: ...
else:
    pupil_ts = None
    pupil_diameter = None
...
image_indices = np.array([img_to_idx.get(img, 0) for img in images])
change_flags = np.nan_to_num(..., nan=0)
```

iii. The agent calls interpolation appropriate for blink gaps and a middle pupil category a neutral fallback. It also uses explicit skips to meet structural requirements.

## 9-a. What are the most time-consuming steps of the code?

i. Full NWB reading and `process_experiment` dominate, especially because every active file is processed twice. Neural matrices, stimulus tables, running, and eye data are loaded and binned in both passes. The notes estimated roughly 24 minutes from sample timings.

ii.
```python
for ... in active_exps.iterrows():
    stats = process_experiment(..., collect_stats_only=True)
...
for ... in active_exps.iterrows():
    result = process_experiment(..., collect_stats_only=False)
```

iii. The notes’ timing table attributes most runtime to the two processing passes (estimated 575 s and 878 s).

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial `np.where(stim_trials_id == tid)`, per-presentation neural/running/pupil loops, omitted-image fill loops, and per-trial discretization/assembly could be vectorized or grouped once. Cumulative sums already make each interval mean cheap but do not remove Python loops.

ii.
```python
for trial_idx in np.where(trial_mask)[0]:
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    for bi, si in enumerate(trial_stim_indices):
        ...
```

iii. The agent documents cumulative sums as its main optimization, implicitly retaining loops because trials are variable length; it does not discuss all remaining vectorization opportunities.

## 9-c. What processing does the code repeat multiple times?

i. `process_experiment` is run twice for every experiment. The first “stats-only” pass still loads/filter dF/F, constructs cumulative sums, finds trials, bins neural activity, encodes images and changes, and bins behaviors; most of that is repeated in the full pass. Metadata/image files are also opened separately while collecting image names.

ii.
```python
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The agent deliberately chose a two-pass design so global percentile thresholds are known before final categorical outputs are assembled.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. During the stats-only pass it loads and bins neural data and computes image identity, change labels, and outcomes even though only `running_values` and `pupil_values` are returned. It also loads unused trial start/stop/change arrays and, unless requested, defines plotting machinery that is never invoked.

ii.
```python
neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
...
if collect_stats_only:
    continue
...
return {'running_values': running_values, 'pupil_values': pupil_values}
```

iii. No explicit justification is offered for discarded first-pass neural/image work; it results from reusing one general processing function for both passes.
