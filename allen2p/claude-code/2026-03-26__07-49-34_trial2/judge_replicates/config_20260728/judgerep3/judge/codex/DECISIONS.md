# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache. It reads `ophys_experiment_table.csv`, scans the local NWB directory, keeps only experiments whose NWB files are present, then filters to `passive == False`. Each retained experiment is opened directly with `h5py`, and all per-session/per-trial extraction happens inside `process_experiment(...)`.

ii. 
```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
...
our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
active_exps = our_exps[our_exps['passive'] == False].copy()
...
with h5py.File(nwb_path, 'r') as f:
```

iii. In `CONVERSION_NOTES.md`, the AI says it is working from the 284 NWB files available on disk, treats direct NWB reads as equivalent to SDK loading, and excludes passive sessions because trial outcomes are needed.

## 1-b. How are the data split into subjects?

i. Subjects are defined as unique `mouse_id` values in the experiment metadata. The output `subjects` list is the sorted unique mouse IDs, converted to strings.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
subject_idx = subjects_list.index(result['mouse_id'])
all_subject_idx.append(subject_idx)
```

iii. The notes explicitly say subject identity comes from `mouse_id`, and the trajectory describes using the metadata table as the source of subject/session organization.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` / NWB file as a separate session. It does not group multiple experiments that share an `ophys_session_id`.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The trajectory states that each NWB file contains one imaging plane and that the AI decided to “treat each experiment as a separate session for the decoder.” The notes also summarize the dataset as one NWB file per experiment.

## 1-d. How are the data split into trials?

i. Trials are identified from the NWB `intervals/trials` table after trial-type filtering. For each kept trial, the AI finds all stimulus presentations whose `trials_id` matches that trial ID, and uses those stimulus presentations as the within-trial time bins.

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

iii. In the notes, the AI frames 750 ms stimulus presentations as the natural unit of the task and uses stimulus presentations within each trial as the temporal structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only go or catch trials, excludes aborted and auto-rewarded trials, skips trials with no associated stimulus presentations, and later skips experiments with fewer than 2 surviving trials. It also skips experiments with zero valid neurons.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
trial_stim_indices = np.where(stim_trials_id == tid)[0]
if len(trial_stim_indices) == 0:
    continue
...
if n_neurons == 0:
    return None
...
if result is None or result['n_trials'] < 2:
    skipped += 1
    continue
```

iii. `CONVERSION_NOTES.md` says “Include Go+Catch only per task instructions,” and the trajectory mentions excluding passive sessions and degenerate experiments to keep the decoder dataset usable.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from the NWB dF/F array and ophys timestamps, with ROI validity coming from the image segmentation cell table.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
```

iii. The notes say dF/F is already precomputed in NWB and that `valid_roi` matches the SDK’s default ROI curation behavior.

## 2-b. How is the `neural` data processed?

i. The AI filters to `valid_roi == True`, keeps the dF/F signal, and averages it within 750 ms stimulus-presentation bins. It uses cumulative sums plus per-bin frame counts to compute the mean dF/F for each neuron in each bin.

ii.
```python
dff_valid = dff_data[:, valid_roi]
...
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

iii. The notes justify this as “Time bin = 750ms (1 per stimulus flash)” and “Use dF/F (not events),” arguing that this gives a common time base across 11 Hz and 31 Hz recordings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC is a `valid_roi` filter, followed by dropping experiments that end up with zero valid neurons.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
n_neurons = dff_valid.shape[1]

if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
```

iii. The notes say the `valid_roi` field mirrors the Allen SDK’s ROI curation rules and was chosen to match `exclude_invalid_rois=True`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to stimulus-presentation onset bins. For every stimulus presentation, the AI uses `[start_time, start_time + 0.75 s)` and assigns all ophys frames in that interval to one bin; the trial is then represented as the sequence of those bins.

ii.
```python
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
...
trial_stim_indices = np.where(stim_trials_id == tid)[0]
```

iii. The trajectory shows the AI debating native ophys frames versus a common bin size, then explicitly choosing one 750 ms bin per stimulus presentation as the alignment/event unit.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 750 ms time bin. Yes: substantial temporal rebinning is applied by averaging neural and behavioral signals within each stimulus interval.

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
...
bin_duration = TIME_BIN_MS / 1000.0
...
'time_bin_size': TIME_BIN_MS,
```

iii. The notes repeatedly justify this as one stimulus flash plus grey period, chosen to harmonize sessions recorded at different native frame rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation `image_name` field rather than from trial-level `initial_image_name` / `change_image_name`.

ii.
```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in stim['image_name'][:]])
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. The notes say the AI wanted the time-varying image label at the stimulus-presentation level and viewed the stimulus table as the most direct source.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each trial, the AI takes the stimulus-presentation image names, forward-fills omitted presentations from nearby non-omitted images, then maps image names to integer category IDs using a global image list collected from up to the first 10 experiments.

ii.
```python
sample_exps = active_exps.head(min(10, len(active_exps)))
...
if name != 'omitted':
    all_images.add(name)
...
for bi in range(len(images)):
    if images[bi] == 'omitted':
        if bi > 0:
            images[bi] = images[bi - 1]
        else:
            for bj in range(bi + 1, len(images)):
                if images[bj] != 'omitted':
                    images[bi] = images[bj]
                    break

image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` says “Omitted stimuli: Forward-fill image identity from previous presentation,” and also says sampling a few experiments is fast because the image sets are assumed to repeat.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is aligned one-to-one with the same trial stimulus bins used for the neural matrix, so there is one image-identity label per 750 ms neural bin.

ii.
```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
...
neural_matrix[:, bi] = ...
...
images = stim_image_name[trial_stim_indices].copy()
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. The notes and trajectory both justify using stimulus presentations as the common time base for both neural and output variables.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation `is_change` field.

ii.
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The AI’s notes describe `is_change` as the direct stimulus-level indicator for whether that presentation is the change image.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI slices the `is_change` values for the stimulus presentations in the trial, converts NaNs to 0, and casts the result to integer labels.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The 750 ms binning decision makes a single stimulus presentation the natural “right after change” unit, so no extra event-window logic is added.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is applied beyond treating the boolean/indicator `is_change` values as binary categories `0` and `1`.

ii.
```python
change_value_names = ['no_change', 'change']
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes treat image change as inherently categorical and therefore not something that requires percentile binning or additional thresholding.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned to the same stimulus-presentation bins as the neural data, with one change flag per 750 ms neural bin.

ii.
```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
...
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ot['image_change'].astype(np.int64),
    ...
], axis=0)
```

iii. The notes describe stimulus presentation as the shared alignment unit across neural activity and outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the NWB running-speed data array and timestamps.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The notes say this matches the SDK’s running-speed source, but uses direct NWB reads instead of SDK accessors.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI averages running speed within each 750 ms stimulus bin using cumulative sums, gathers all binned values in a first pass to compute global percentile edges, then discretizes each trial’s running trace into 5 bins in the second pass.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
run_cumsum = np.cumsum(running_speed)
...
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
...
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
...
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. In the notes, this is justified as matching the 750 ms stimulus-centered time base and providing globally balanced decoder classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins computed globally across all collected running values.

ii.
```python
def discretize_values(values, n_bins, bin_edges=None):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(valid, percentiles)
...
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. The notes explicitly call for “5 percentile bins” computed in a two-pass pipeline so category definitions are shared across sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is not interpolated to every ophys frame. Instead, it is averaged over the exact same stimulus bins used for the neural data, so both streams have one value per stimulus presentation.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
...
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
    ...
    running_binned[bi] = ...
```

iii. The notes justify this as a common 750 ms presentation-level time base shared with neural activity.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking pupil area and blink flags in the NWB acquisition group, not from the SDK’s `pupil_width` field.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The notes say the AI intentionally used area and converted it to diameter, because the raw NWB stores area directly and blink frames can be nulled before conversion.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames and nonpositive areas are set to NaN, area is converted to diameter via `2*sqrt(area/pi)`, NaNs are linearly interpolated, the signal is averaged within 750 ms stimulus bins, and then discretized into 5 global percentile bins. If a trial’s pupil bins are all NaN, the AI assigns the middle category.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
...
pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. `CONVERSION_NOTES.md` says “area → compute diameter, interpolate NaNs” and frames this as a blink-handling decision plus a way to get a scalar diameter from the raw eye-tracking signal.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into 5 equal-percentile bins computed globally from all non-NaN pupil values gathered in the first pass.

ii.
```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
if len(valid_pupil) > 0:
    _, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
...
pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The notes explicitly state that pupil diameter should be put into 5 global percentile bins for balanced categorical decoding.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Like running speed, pupil diameter is averaged over the same stimulus-presentation bins as the neural signal, so neural and pupil outputs share the same per-bin indexing.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
...
for bi, si in enumerate(trial_stim_indices):
    s, e = pup_bin_starts[si], pup_bin_ends[si]
    ...
    pupil_binned[bi] = ...
```

iii. The AI’s notes describe stimulus bins as the universal alignment frame for neural, stimulus, and behavioral variables.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. The notes treat these four fields as the canonical outcome labels for non-aborted, non-auto-rewarded trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps outcomes to integer codes with a fixed if/elif order: hit=0, miss=1, false alarm=2, correct rejection=3. If none of those booleans is set, it defaults the label to miss. The chosen trial-outcome code is then repeated across all bins of the trial in the output array.

ii.
```python
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

iii. The notes say trial outcome is a static per-trial target, so the AI keeps one code per trial and broadcasts it across time for the decoder format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing pupil data are interpolated; trials with no stimulus presentations are skipped; experiments with zero valid neurons or fewer than 2 valid trials are skipped; omitted image presentations are forward-filled; and all-NaN pupil trials are forced to the middle pupil bin. The code also returns `None` for malformed experiments rather than stopping the entire conversion path that called it.

ii.
```python
if len(trial_stim_indices) == 0:
    continue
...
if n_neurons == 0:
    return None
...
if result is None or result['n_trials'] < 2:
    skipped += 1
    continue
...
pupil_diameter = interpolate_nans(pupil_diameter)
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
...
if images[bi] == 'omitted':
    images[bi] = images[bi - 1]
```

iii. The notes and trajectory frame these as pragmatic fixes to keep the dataset dense and decoder-ready, especially for pupil and image omissions.

## 9-a. What are the most time-consuming steps of the code?

i. The slowest parts are opening and scanning every NWB file, and doing it twice: once in pass 1 to collect running/pupil values and again in pass 2 for full conversion. Inside each pass, the per-trial/per-bin averaging loops over neural, running, and pupil data are the next main cost.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    ...
    stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
for idx, (_, row) in enumerate(active_exps.iterrows()):
    ...
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. `CONVERSION_NOTES.md` explicitly describes a two-pass implementation and gives runtime estimates for Pass 1 and Pass 2, indicating that repeated full-dataset traversal is the main expense.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still loops over every trial and then every stimulus bin within each trial for neural, running, and pupil averaging. The omitted-image forward-fill loop is also scalar. These loops could have been further vectorized or grouped by trial/bin boundaries.

ii.
```python
for trial_idx in np.where(trial_mask)[0]:
    ...
    for bi, si in enumerate(trial_stim_indices):
        ...
        neural_matrix[:, bi] = ...
...
for bi in range(len(images)):
    if images[bi] == 'omitted':
        ...
...
for bi, si in enumerate(trial_stim_indices):
    ...
    running_binned[bi] = ...
...
for bi, si in enumerate(trial_stim_indices):
    ...
    pupil_binned[bi] = ...
```

iii. The notes call some of this “vectorized” because cumulative sums are used, but the trial/bin loops are still explicit Python loops.

## 9-c. What processing does the code repeat multiple times?

i. The AI repeats session loading and most preprocessing in two full passes. It also runs a separate mini-pass over up to 10 experiments just to collect image names. In pass 1, `process_experiment(..., collect_stats_only=True)` still computes neural matrices, image identity, image change, and trial outcomes even though only running and pupil values are returned.

ii.
```python
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
...
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
result = process_experiment(nwb_path, row, collect_stats_only=False)
...
neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
...
images = stim_image_name[trial_stim_indices].copy()
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
...
if collect_stats_only:
    continue
```

iii. The notes explicitly describe a two-pass design for percentile bins, and the code shows that the “stats only” mode still performs much of the same per-trial work.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are thrown away. In pass 1, neural matrices, image labels, image-change flags, and trial outcomes are computed but never used. In pass 2, raw running and pupil traces are stored in intermediate `output_trials` dicts only to be immediately discretized and discarded. The code also loads several raw arrays that are never used (`stim_stop`, `stim_omitted`, `valid_trial_ids`, `trial_start`, `trial_stop`, `trial_change_time`, `n_total_rois`).

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
...
n_total_rois = dff_data.shape[1]
...
if collect_stats_only:
    continue
...
'running_speed_raw': running_binned,
'pupil_diameter_raw': pupil_binned,
...
running_disc, _ = discretize_values(ot['running_speed_raw'], ...)
pupil_disc, _ = discretize_values(pupil_raw, ...)
```

iii. The two-pass notes explain why some repetition exists, but the implementation still computes and retains intermediates that are not part of the final saved dataset.
