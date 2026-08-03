# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI bypasses the Allen SDK cache used in the reference solution and reads local NWB files directly with `h5py`. It builds the dataset from `ophys_experiment_table.csv` plus the set of NWB files present under `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`. It then iterates over each filtered experiment file and processes it one by one.

ii. 
```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
...
our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
active_exps = our_exps[our_exps['passive'] == False].copy()
...
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. In `CONVERSION_NOTES.md`, the AI says it worked from the local 284 NWB files available on disk rather than the full S3-backed Allen dataset, and that this local subset was the practical data source.

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values in the experiment metadata CSV, converted to strings and sorted.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
subject_idx = subjects_list.index(result['mouse_id'])
all_subject_idx.append(subject_idx)
```

iii. The AI’s notes describe `mouse_id` as the animal identifier and report subject counts from the metadata/NWB subset.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB `ophys_experiment_id` as a separate session. It does not group multiple experiments from the same `ophys_session_id` into a single session as the human reference does.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
    ...
    all_sessions_neural.append(result['neural_trials'])
```

iii. In the trajectory, the AI explicitly reasoned that “each NWB file contains its own set of neurons, I should treat each experiment as a separate session for the decoder rather than grouping by ophys_session”.

## 1-d. How are the data split into trials?

i. Trials are built by using the NWB `intervals/trials` table to select eligible trial IDs, then mapping each trial to the stimulus-presentation rows in the stimulus table via `trials_id`. Each trial becomes a sequence of 750 ms stimulus-presentation bins rather than a continuous `start_time` to `stop_time` ophys-frame window.

ii.
```python
trials = f['intervals']['trials']
trial_ids = trials['id'][:]
...
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
    n_bins = len(trial_stim_indices)
```

iii. The AI’s notes and trajectory say it chose 750 ms stimulus bins because they match the task structure and give a common time base across mesoscope and single-plane experiments.

## 1-e. How are trials filtered based on quality controls?

i. The AI includes go and catch trials, excludes aborted and auto-rewarded trials, skips trials with no associated stimulus presentations, skips experiments with zero valid neurons, and later drops sessions with fewer than 2 trials.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
...
if n_neurons == 0:
    return None
...
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
if len(trial_stim_indices) == 0:
    continue
...
if result is None or result['n_trials'] < 2:
    skipped += 1
    continue
```

iii. The notes say this follows the task’s go/catch versus aborted/auto-rewarded requirement, and that `valid_roi` matches SDK default ROI filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from NWB `processing/ophys/dff/traces/data`, filtered by `valid_roi` from the cell specimen table.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
...
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
...
dff_valid = dff_data[:, valid_roi]
```

iii. The notes say dF/F is precomputed in the NWB files and the AI chose it instead of events.

## 2-b. How is the `neural` data processed?

i. The AI filters to `valid_roi=True`, keeps dF/F rather than events, and then averages dF/F within each 750 ms stimulus bin using cumulative sums. It does not merge multiple planes into one session because each experiment is treated as its own session.

ii.
```python
dff_valid = dff_data[:, valid_roi]  # (timepoints, n_valid_neurons)
...
dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])
...
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The AI justified this in the notes as a common-bin strategy across 11 Hz and 31 Hz recordings, and in the trajectory as preferring dF/F because it was “pre-computed and standard for decoding”.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered by the NWB/SDK `valid_roi` flag, and experiments with zero valid neurons are discarded.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
n_neurons = dff_valid.shape[1]

if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
```

iii. The notes explicitly cite `exclude_invalid_rois=True` in the SDK as the rationale for mirroring `valid_roi` filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to stimulus presentation onsets, not directly to the native ophys frame timestamps spanning the full trial. For each stimulus flash, it averages neural data over `[stim_start, stim_start + 0.75 s)`.

ii.
```python
bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration

ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
```

iii. The AI’s notes say “Time bin = 750ms (1 per stimulus flash)” because it viewed that as the natural task unit and a way to unify different frame rates.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 750 ms bins, and substantial temporal rebinning is applied by averaging raw time series within each stimulus-presentation bin.

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
...
'metadata': {
    ...
    'time_bin_size': TIME_BIN_MS,
    'temporal_alignment_event': 'Stimulus presentation onset (each 750ms image flash)',
```

iii. The notes repeatedly justify 750 ms bins as matching the image-plus-grey stimulus cycle and standardizing across equipment types.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation table’s `image_name` field, taken for each stimulus presentation assigned to a trial.

ii.
```python
stim = f['intervals'][stim_key]
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stim['image_name'][:]])
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. The AI’s notes map `image_name` from stimulus presentations directly to the output variable.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI forward-fills omitted stimulus presentations, collects a global image-name vocabulary from a sample of experiments, and maps names to integer category IDs.

ii.
```python
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
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

iii. The notes say omitted flashes were forward-filled and that there were 16 global images across two image sets, even though each session used 8.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is aligned per 750 ms stimulus bin using the same `trial_stim_indices` that define the neural bins, so each neural column and image-identity value correspond to the same stimulus presentation.

ii.
```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
...
neural_trials.append(neural_matrix)
output_trials.append({
    'image_identity': image_indices,
    ...
    'n_bins': n_bins,
})
```

iii. The AI’s notes justify the 750 ms binning as a joint alignment strategy for neural and output variables.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change` field for each presentation within the trial.

ii.
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes map `is_change` from stimulus presentations directly to the image-change output.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI performs essentially no extra processing beyond converting NaNs to 0 and casting the per-presentation `is_change` values to integer flags.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes describe image change as “1 at change flash only”, reflecting this direct use of the stimulus table.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already treated as binary categories, with 0 for non-change and 1 for change; no further thresholding is applied.

ii.
```python
change_value_names = ['no_change', 'change']
...
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ot['image_change'].astype(np.int64),
```

iii. The notes present image change as an inherently binary output rather than a continuous value needing thresholding.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned one-for-one with the same 750 ms stimulus bins used for neural data and image identity.

ii.
```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
...
output_trials.append({
    'image_change': change_flags,
    ...
})
```

iii. The AI’s justification is the same global choice to represent trials as sequences of stimulus-presentation bins.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from NWB `processing/running/speed/data` and its timestamps.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The notes identify this as the standard running-speed source and say the NWB signal is already filtered.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI does not interpolate running speed to ophys timestamps. Instead it averages native running-speed samples within each 750 ms stimulus bin, then discretizes all values into 5 percentile bins computed in a first pass over all experiments.

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

iii. The notes justify this as a two-pass global percentile-binning strategy after stimulus-bin averaging.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is discretized into 5 global percentile bins using `np.percentile` and `np.digitize`.

ii.
```python
def discretize_values(values, n_bins, bin_edges=None):
    if bin_edges is None:
        valid = values[~np.isnan(values)]
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(valid, percentiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
    binned = np.digitize(values, bin_edges[1:-1]).astype(np.int64)
```

iii. The AI’s notes explicitly say running speed should be in “5 percentile bins” computed over all valid time points.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by assigning samples to the same per-stimulus 750 ms windows as neural data and averaging within each window.

ii.
```python
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
...
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
...
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
```

iii. The notes justify all alignment through shared stimulus bins rather than through an ophys-frame time base.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking `pupil_tracking/area`, plus `likely_blink` and timestamps, not from `pupil_width`.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The notes say the AI chose area because the NWB file provided it and then converted area to diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames and nonpositive values are set to NaN, pupil diameter is computed as `2*sqrt(area/pi)`, NaNs are linearly interpolated at the full time-series level, the result is averaged within 750 ms stimulus bins, and then discretized into 5 global percentile bins. Remaining all-NaN trial bins are assigned the middle category.

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
...
pupil_binned = np.full(n_bins, np.nan, dtype=np.float32)
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The notes justify blink removal and interpolation, and describe the conversion as “area → diameter, interpolate NaN, avg in bins, 5 percentile bins”.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is discretized into 5 global percentile bins. If an entire trial’s pupil bins are NaN, the AI assigns the middle bin.

ii.
```python
if len(valid_pupil) > 0:
    _, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
else:
    pupil_bin_edges = np.array([-np.inf, 0, 1, 2, 3, np.inf])
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
```

iii. The notes say the target required 5 percentile bins and that interpolation plus fallback handling was used for missing pupil data.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. It is aligned by averaging eye-tracking samples into the same 750 ms stimulus bins used for neural data.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
...
for bi, si in enumerate(trial_stim_indices):
    s, e = pup_bin_starts[si], pup_bin_ends[si]
```

iii. The AI’s alignment rationale is again the shared stimulus-presentation binning.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. The notes identify these four outcomes as the desired trial-level labels for active sessions.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four booleans to integer codes 0-3 with a default fallback to 1 (`miss`), then repeats the chosen code across all bins in the trial.

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
...
np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
```

iii. The notes say this output is static per trial and encoded as four classes.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: it skips experiments with no valid neurons or no stimulus table, skips trials with no stimulus presentations, sets blink and nonpositive pupil samples to NaN, interpolates pupil NaNs, converts `is_change` NaNs to 0, uses percentile edges from non-NaN values, assigns a middle pupil bin if a trial’s pupil values are all NaN, and drops sessions with fewer than 2 trials.

ii.
```python
if n_neurons == 0:
    return None
...
if stim_key is None:
    return None
...
if len(trial_stim_indices) == 0:
    continue
...
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = interpolate_nans(pupil_diameter)
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
...
valid_pupil = all_pupil[~np.isnan(all_pupil)]
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
```

iii. The notes say missing pupil data from blinks were interpolated and that validation/sanity checks showed no remaining format problems.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are the two full passes over all experiments, especially repeatedly opening NWB files and processing every trial/bin. The AI also identified per-bin averaging as a potential runtime bottleneck during development.

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

iii. In the trajectory and notes, the AI estimated runtime from a sample run, described the two-pass conversion, and discussed optimizing per-bin loops.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several per-trial and per-bin loops remain non-vectorized: iterating over trials, iterating over bins within each trial for neural/running/pupil averaging, and the omitted-image forward-fill loop.

ii.
```python
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

iii. The notes mention cumulative-sum optimization, but the code still keeps explicit Python loops around those cumulative-sum lookups.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats experiment loading and trial parsing twice, once to collect global running/pupil values for discretization and again for full conversion. It also samples experiments separately to collect image names.

ii.
```python
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
...
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The notes explicitly describe a “two-pass approach” and separately mention sampling experiments for image-name collection.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores raw intermediate running and pupil traces inside `output_trials` during pass 2 only to discretize them immediately afterward, and it builds plotting support (`--show-processing`) that is not used in the final converted dataset. It also computes `valid_trial_ids` but never uses it.

ii.
```python
valid_trial_ids = trial_ids[trial_mask]
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
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
...
if args.show_processing and idx < 2:
    plot_processing(result, idx, nwb_path, row)
```

iii. The notes frame this mainly as a practical implementation choice for two-pass discretization and optional visualization rather than as part of the final target representation.
