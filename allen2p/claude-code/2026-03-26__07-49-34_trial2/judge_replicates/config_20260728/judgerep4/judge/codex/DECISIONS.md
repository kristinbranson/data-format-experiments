# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the Allen SDK cache. It loaded local NWB files directly with `h5py`, discovered available experiments by intersecting `ophys_experiment_table.csv` with filenames on disk, and then restricted processing to `passive == False` experiments. It then processed each remaining experiment one-by-one.

ii. ```python
def get_experiment_metadata():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    ...
    our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
    active_exps = our_exps[our_exps['passive'] == False].copy()
    ...

with h5py.File(nwb_path, 'r') as f:
    dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as working on “the 284 available NWB files (subset)” rather than the full SDK release, and treated direct NWB reading as “equivalent” to SDK loading. It also justified keeping only active sessions because passive sessions “have no meaningful trial outcomes.”

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values in the experiment metadata, after the active-session filter.

ii. ```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
'mouse_id': str(exp_info['mouse_id']),
...
subject_idx = subjects_list.index(result['mouse_id'])
```

iii. The justification in the notes is implicit: the Allen metadata `mouse_id` is treated as the subject identifier, matching the SDK metadata convention.

## 1-c. How are the data split into sessions?

i. The AI treated each NWB experiment file (`ophys_experiment_id`) as one session. It did not group experiments by `ophys_session_id`.

ii. ```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The notes state “Each NWB file = one imaging plane from one session,” and the implementation follows that literally. No separate justification was given for not reconstructing multi-experiment sessions.

## 1-d. How are the data split into trials?

i. Trials are split by taking each valid trial ID from the NWB `intervals/trials` table, then collecting all stimulus presentations whose `trials_id` matches that trial. Each trial becomes a sequence of 750 ms stimulus bins rather than a framewise slice from `start_time` to `stop_time`.

ii. ```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
    n_bins = len(trial_stim_indices)
```

iii. The main stated rationale was that one 750 ms bin per image presentation “matches the natural stimulus structure” and yields uniform bin sizes across equipment with different native frame rates.

## 1-e. How are trials filtered based on quality controls?

i. The AI kept only go or catch trials, excluded aborted and auto-rewarded trials, skipped trials with no linked stimulus presentations, and later skipped experiments with fewer than 2 retained trials.

ii. ```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
trial_stim_indices = np.where(stim_trials_id == tid)[0]
if len(trial_stim_indices) == 0:
    continue
...
if result is None or result['n_trials'] < 2:
    skipped += 1
    continue
```

iii. The notes explicitly cite the task rule “Include Go+Catch only per task instructions.” No explicit justification was given for omitting the reference check `change_time.notna()`; that difference follows from using the stimulus table instead of framewise trial windows.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from NWB dF/F traces in `processing/ophys/dff/traces/data`, filtered to ROIs where `valid_roi` is true.

ii. ```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. The notes justify this by saying dF/F is already precomputed in NWB, and that `valid_roi` matches the SDK’s default ROI curation.

## 2-b. How is the `neural` data processed?

i. The AI kept dF/F, filtered to valid ROIs, and averaged it within 750 ms stimulus-presentation bins using cumulative sums. It did not keep native ophys frames and did not merge multiple planes into one session.

ii. ```python
bin_duration = TIME_BIN_MS / 1000.0
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
dff_cumsum = np.cumsum(dff_valid, axis=0)
...
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The notes and trajectory justify this with the desire for a common bin size across 11 Hz and 31 Hz recordings, choosing 750 ms because it “corresponds to one image presentation.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural filtering consists only of dropping ROIs with `valid_roi == False`. Experiments with zero valid neurons are skipped entirely. No additional cell-wise filtering or normalization is applied.

ii. ```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
n_neurons = dff_valid.shape[1]

if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
```

iii. The notes explicitly say this matches the SDK default `exclude_invalid_rois=True`, and that no further QC beyond the Allen curation was applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to stimulus-presentation onset, not to trial start. Each trial is represented as consecutive 750 ms bins beginning at the onset of each stimulus presentation associated with that trial.

ii. ```python
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
```

iii. The AI’s stated rationale was that stimulus onset is the “natural task unit” and that this avoids mixed native frame rates across sessions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 750 ms time bin. Yes: the AI rebinned neural, running, and pupil data into one bin per stimulus presentation.

ii. ```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
...
'time_bin_size': TIME_BIN_MS,
```

iii. The notes make this an explicit key decision: “Time bin = 750ms (1 per stimulus flash)” for consistency across mesoscope and single-plane data.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation table’s `image_name` field, not from the trials table’s `initial_image_name` and `change_image_name`.

ii. ```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in stim['image_name'][:]])
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. The notes justify this as working directly at stimulus-presentation resolution, where each 750 ms bin already corresponds to one flashed image.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI forward-filled omitted stimulus flashes, then mapped image names to integer codes using a global image list gathered from only the first up to 10 experiments.

ii. ```python
def collect_all_image_names(active_exps, nwb_map):
    all_images = set()
    sample_exps = active_exps.head(min(10, len(active_exps)))
    ...
    return sorted(all_images)
...
for bi in range(len(images)):
    if images[bi] == 'omitted':
        if bi > 0:
            images[bi] = images[bi - 1]
...
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. The notes explicitly justify forward-filling omissions because the omitted flash is a continuation of gray while the task still has a latent image context. The 10-experiment sampling was justified as a speed optimization, with the assumption that sessions reuse the same image sets.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned per 750 ms stimulus bin and shares the same `trial_stim_indices` bin structure as the neural matrix.

ii. ```python
neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
...
images = stim_image_name[trial_stim_indices].copy()
...
output_trials.append({
    'image_identity': image_indices,
    ...
    'n_bins': n_bins,
})
```

iii. The AI’s justification is the same as for neural alignment: use one timepoint per stimulus presentation so neural and output arrays are naturally synchronized.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change` field.

ii. ```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes describe this as “1 at change flash only,” using the presentation table’s explicit change flag rather than recomputing from trial timing.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI simply selected `is_change` for the stimulus bins belonging to a trial and cast missing values to zero.

ii. ```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The justification was that, with one 750 ms bin per flash, the change variable naturally becomes a binary flag on the change presentation.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is directly represented as binary categories `0` and `1`; no extra thresholding is applied.

ii. ```python
change_value_names = ['no_change', 'change']
...
ot['image_change'].astype(np.int64)
```

iii. No separate justification was needed beyond the task definition of image change as a binary variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned per stimulus bin, using the same `trial_stim_indices` binning as neural activity.

ii. ```python
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ot['image_change'].astype(np.int64),
    ...
], axis=0)
```

iii. The stated rationale was to keep every decoded variable on the same 750 ms stimulus-presentation clock.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the NWB running stream `processing/running/speed/data` with its own timestamps.

ii. ```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The notes say this uses the NWB’s filtered running-speed signal, which corresponds to the Allen running-speed output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed samples are averaged within each 750 ms stimulus bin using timestamp-based cumulative-sum windows, then discretized globally into 5 percentile bins in a second pass.

ii. ```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
run_cumsum = np.cumsum(running_speed)
...
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
...
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. The notes justify this as a two-pass global percentile discretization, with 750 ms averaging chosen to match the stimulus-aligned trial representation.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The AI computed global percentile edges across all retained time bins and discretized running speed into 5 categories.

ii. ```python
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
...
binned = np.digitize(values, bin_edges[1:-1]).astype(np.int64)
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The notes explicitly justify global percentile binning as giving roughly balanced classes for the decoder.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by averaging running samples over the same stimulus-onset 750 ms windows used for neural binning.

ii. ```python
all_stim_starts = stim_start
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
...
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
```

iii. The AI’s rationale was to place behavior and neural data on the same one-bin-per-flash representation.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the eye-tracking `area` measurement plus `likely_blink`, not from the SDK’s `pupil_width`.

ii. ```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The notes explicitly state “Use area → compute diameter, interpolate NaNs,” treating the area-derived circle-equivalent diameter as the desired variable.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples and nonpositive areas are set to NaN, area is converted to diameter via `2*sqrt(area/pi)`, NaNs are interpolated, values are averaged in 750 ms stimulus bins, and then discretized globally into 5 percentile bins.

ii. ```python
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
...
pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
...
pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The notes justify this as handling blink artifacts before binning and producing the requested pupil-diameter output rather than retaining raw area.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The AI used 5 global percentile bins. If a trial’s pupil trace was entirely NaN after preprocessing, it filled that trial with the middle category; otherwise it interpolated remaining NaNs before discretization.

ii. ```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
_, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The notes justify the percentile discretization globally; the “middle bin” fallback is not separately justified in prose and appears to be an implementation choice for fully missing trials.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by averaging eye-tracking samples over the same 750 ms stimulus windows used for neural activity.

ii. ```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
...
pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. The same stimulus-bin alignment rationale was used across all time-varying outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean `hit`, `miss`, `false_alarm`, and `correct_reject` columns in the trials table.

ii. ```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. The notes describe these as the canonical task outcomes for active go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI mapped hit/miss/false alarm/correct rejection to integer codes 0/1/2/3, defaulted unmatched trials to miss, and repeated the code across all time bins in the trial output matrix.

ii. ```python
if trial_hit[trial_idx]:
    outcome = 0
elif trial_miss[trial_idx]:
    outcome = 1
elif trial_fa[trial_idx]:
    outcome = 2
elif trial_cr[trial_idx]:
    outcome = 3
else:
    outcome = 1
...
np.full(n_bins, ot['trial_outcome'], dtype=np.int64)
```

iii. The notes justify trial outcome as a static per-trial variable. No explicit rationale was given for the “default to miss” fallback.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handled several edge cases: skip experiments with no valid ROIs or no stimulus key; skip trials with no linked stimulus presentations; interpolate NaNs in pupil traces; if all pupil values are missing in a trial, assign the middle pupil bin; and skip experiments with fewer than 2 retained trials.

ii. ```python
if n_neurons == 0:
    return None
...
if stim_key is None:
    return None
...
if len(trial_stim_indices) == 0:
    continue
...
pupil_diameter = interpolate_nans(pupil_diameter)
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
...
if result is None or result['n_trials'] < 2:
    continue
```

iii. The notes explicitly justify interpolation for blink-related missingness and the 2-trial minimum. Other missing-data choices, especially the middle-bin fallback for all-missing pupil trials, were not explicitly defended in prose.

## 9-a. What are the most time-consuming steps of the code?

i. The AI’s own notes show the expensive part is repeated full-experiment processing: Pass 1 scans all experiments to collect running/pupil values, and Pass 2 rereads all experiments to build the final dataset. Within an experiment, the neural/running/pupil bin averaging is the main compute-heavy part.

ii. ```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
for idx, (_, row) in enumerate(active_exps.iterrows()):
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The notes report runtime separately for “Pass 1” and “Pass 2,” and the trajectory says the main bottleneck “appears to be the per-bin loop” before the AI added cumulative-sum optimizations.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the window-boundary lookup with `searchsorted` and cumulative sums, but it still loops over every trial and then every stimulus bin within the trial for neural, running, and pupil aggregation. The omitted-image forward-fill loop is also still scalar.

ii. ```python
for trial_idx in np.where(trial_mask)[0]:
    ...
    for bi, si in enumerate(trial_stim_indices):
        ...
    for bi in range(len(images)):
        if images[bi] == 'omitted':
            ...
    for bi, si in enumerate(trial_stim_indices):
        ...
    for bi, si in enumerate(trial_stim_indices):
        ...
```

iii. The notes and trajectory say the AI intentionally optimized some loops with cumulative sums and `searchsorted`, but they do not claim the implementation is fully vectorized.

## 9-c. What processing does the code repeat multiple times?

i. The code rereads and reprocesses every experiment twice: once in Pass 1 for global running/pupil statistics and again in Pass 2 for actual conversion. It also opens up to 10 experiments earlier just to discover image names.

ii. ```python
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
...
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The notes explicitly describe this as a “two-pass approach” chosen so percentile bins could be computed globally before final discretization.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several intermediate products are computed only to be discarded downstream: Pass 1’s per-bin raw running/pupil values exist only to compute percentile edges; Pass 2 stores `running_speed_raw` and `pupil_diameter_raw` in intermediate trial dicts only to discretize them immediately; and the optional plotting path computes visualization-only outputs not used by the saved dataset.

ii. ```python
if collect_stats_only:
    return {
        'running_values': running_values,
        'pupil_values': pupil_values,
    }
...
output_trials.append({
    'image_identity': image_indices,
    'image_change': change_flags,
    'running_speed_raw': running_binned,
    'pupil_diameter_raw': pupil_binned,
    'trial_outcome': outcome,
    'n_bins': n_bins,
})
...
if args.show_processing and idx < 2:
    plot_processing(result, idx, nwb_path, row)
```

iii. The two-pass design is explicitly justified in the notes by global percentile binning. The rest is not defended separately; it follows from the chosen implementation structure.
