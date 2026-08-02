# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by first reading an experiment metadata CSV (`ophys_experiment_table.csv`) and scanning the NWB file directory to find available experiments. It matches experiment IDs between the CSV and the NWB filenames. Only "active" (non-passive) experiments are selected. Each NWB file is then opened with h5py and processed individually in two passes: Pass 1 collects running speed and pupil statistics for global percentile computation, Pass 2 performs full data extraction.

ii.
```python
def get_experiment_metadata():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    nwb_map = {}
    for f in nwb_files:
        eid = int(os.path.basename(f).replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
        nwb_map[eid] = f
    our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
    active_exps = our_exps[our_exps['passive'] == False].copy()
    ...
```

iii. The AI documented that it found 284 NWB files on disk (a subset of the full 1,936 experiments), identified 202 active experiments from 38 mice, and used h5py for direct NWB reading as equivalent to the SDK's `BehaviorOphysExperiment.from_nwb_path()`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by `mouse_id` from the experiment metadata CSV. A sorted list of unique mouse IDs is created. Each session is assigned a `subject_idx` indexing into this list.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
# ...
subject_idx = subjects_list.index(result['mouse_id'])
all_subject_idx.append(subject_idx)
```

iii. The AI noted 38 unique subjects in the NWB subset (vs. 82 in the full dataset). Mouse IDs are cast to strings and sorted for consistent ordering.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one experiment (one imaging plane from one session). Each experiment is treated as a separate session in the output. Sessions are processed in order of `ophys_experiment_id` (sorted).

ii.
```python
active_exps = active_exps.sort_values('ophys_experiment_id').reset_index(drop=True)
# Each NWB file is processed via:
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The AI treats each experiment (imaging plane) as a separate session. This is consistent with the NWB file organization where each file contains data from one imaging plane.

## 1-d. How are the data split into trials?

i. Trials are loaded from the NWB `intervals/trials` table. Stimulus presentations are linked to trials via the `trials_id` field in the stimulus presentations table. For each valid trial, the corresponding stimulus presentations are found and grouped.

ii.
```python
trials = f['intervals']['trials']
trial_ids = trials['id'][:]
# ...
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
valid_trial_ids = trial_ids[trial_mask]
# Per trial:
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
    n_bins = len(trial_stim_indices)
```

iii. Each trial consists of multiple stimulus presentations (typically ~11-12 flashes per trial). The number of time bins per trial equals the number of stimulus presentations in that trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only Go and Catch trials, excluding Aborted and Auto-rewarded trials, as specified in the task instructions.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
```

iii. The AI followed the task instructions exactly: "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials." No additional filtering beyond this was applied (e.g., no engagement-based filtering).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the pre-computed dF/F (delta fluorescence over fluorescence) traces stored in the NWB file at `processing/ophys/dff/traces/data`.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
```

iii. The AI chose dF/F over calcium events, justifying that "dF/F is the standard neural signal for decoding; pre-computed in NWB; events are sparser and may not decode as well for time-varying outputs." The CONVERSION_NOTES acknowledge that "The paper uses calcium events (not dF/F) for neural analysis, detected via FastLZeroSpikeInference."

## 2-b. How is the `neural` data processed?

i. dF/F values are averaged within each 750ms stimulus presentation bin. A cumulative sum approach is used for efficient bin averaging. The ophys timestamps are used to find frames falling within each stimulus presentation window using `searchsorted`.

ii.
```python
# Precompute cumulative sum for fast bin averaging
dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])

# Per bin:
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. Each time bin represents one stimulus presentation (750ms). The averaging window spans from stimulus onset to onset + 750ms. Bins with no ophys frames are left as zeros.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the `valid_roi` column from the NWB cell specimen table. Only neurons marked as valid ROIs are included.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]  # (timepoints, n_valid_neurons)
```

iii. The AI documented that this matches the SDK's default behavior (`exclude_invalid_rois=True` in `CellSpecimens`). The valid_roi filter excludes unions of cells, duplicates (>70% overlap), edge ROIs, apical dendrites, too small/narrow/dim cells, ghost cells, and negative/zero traces.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. For each stimulus presentation within a trial, ophys frames falling within the [start_time, start_time + 0.75s) window are averaged. This aligns neural data to stimulus presentation onsets.

ii.
```python
bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
```

iii. The instructions say to "Temporally align based on ophys timestamp." The AI uses ophys timestamps as the reference and aligns all other data streams to stimulus presentation times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 750ms, corresponding to one stimulus presentation interval (250ms image + 500ms grey screen). Each stimulus presentation becomes one time bin. This is a rebinning from the raw ophys frame rate (~11 Hz for mesoscope or ~31 Hz for single-plane).

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
```

iii. The AI chose 750ms as the natural task unit, ensuring consistency across different equipment types (MESO at ~11Hz and CAM2P at ~31Hz).

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` field in the stimulus presentations table in the NWB file.

ii.
```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stim['image_name'][:]])
```

iii. The AI noted there are 16 unique image names across all sessions (8 per image set A/B).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are extracted per stimulus presentation. For omitted presentations (where `image_name == 'omitted'`), the image name is forward-filled from the previous non-omitted presentation. Image names are then mapped to integer indices using a global sorted list of all image names.

ii.
```python
images = stim_image_name[trial_stim_indices].copy()
# Forward-fill omitted presentations
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

iii. The AI justified forward-filling omitted stimuli: "Neural and behavioral data still recorded during omissions" so the previous image identity is carried forward. Image names are collected from up to 10 experiments to build the global list.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned 1:1 with neural data time bins. Each stimulus presentation has both a neural average and an image name, so they are inherently aligned by the stimulus presentation index.

ii.
```python
# Both use the same trial_stim_indices:
images = stim_image_name[trial_stim_indices].copy()
# neural_matrix[:, bi] uses the same bi index
```

iii. The alignment is guaranteed by using the same stimulus presentation indices for both neural and image identity data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table in the NWB file.

ii.
```python
stim_is_change = stim['is_change'][:]
```

iii. The `is_change` field is a pre-computed boolean/flag in the NWB file indicating whether a change in image identity occurred at that presentation.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` values are extracted for each stimulus presentation in the trial. NaN values are replaced with 0 (no change), and the result is cast to int64.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The AI treats `is_change` as a binary variable: 1 when a change occurs, 0 otherwise.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1) from the raw data, so no thresholding is needed. The output values are labeled `['no_change', 'change']`.

ii.
```python
change_value_names = ['no_change', 'change']
```

iii. No additional thresholding or processing is applied; the raw binary values are used directly.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned identically to image identity - one value per stimulus presentation, matching 1:1 with neural time bins.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
# Same trial_stim_indices used for neural data
```

iii. Same alignment mechanism as image identity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` in the NWB file, which contains the pre-filtered (10 Hz lowpass Butterworth) running speed in cm/s.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The AI uses the filtered speed from the NWB, consistent with the SDK's `RunningSpeed.from_nwb()`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each 750ms stimulus bin using a cumulative sum approach. Then, all running speed values across the entire dataset are collected (Pass 1) to compute global percentile bin edges. In Pass 2, the binned running speed values are discretized into 5 equal percentile bins using these global edges.

ii.
```python
# Bin averaging:
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
run_cumsum = np.cumsum(running_speed)
run_cumsum = np.concatenate([[0], run_cumsum])
# Per bin:
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts

# Discretization:
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. The two-pass approach ensures that percentile bin edges are computed globally across all experiments, so the 5 bins represent equal fractions of the entire dataset.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins. The bin edges are computed from the full dataset. The first edge is set to -inf and the last to +inf to capture all values.

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
    binned = np.clip(binned, 0, n_bins - 1)
    return binned, bin_edges
```

iii. The AI documented running speed percentile edges as approximately [-inf, 0.004, 0.788, 15.82, 33.13, inf].

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned to the same stimulus presentation bins as neural data. For each stimulus presentation, running speed values within the [start_time, start_time + 0.75s) window are averaged.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
```

iii. Same temporal alignment as neural data, ensuring 1:1 correspondence between running speed bins and neural bins.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the `pupil_area` field in `acquisition/EyeTracking/pupil_tracking/area` and the `likely_blink` field in `acquisition/EyeTracking/likely_blink/data` in the NWB file.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The AI checks for the presence of eye tracking data (`has_eye_tracking = 'EyeTracking' in f['acquisition']`) before attempting to load it.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area values during likely blinks are set to NaN. Negative or zero pupil area values are also set to NaN. Diameter is computed from area as `d = 2*sqrt(area/pi)`. NaN values are linearly interpolated. The diameter is then averaged within 750ms bins, and discretized into 5 equal percentile bins using global edges. For sessions without eye tracking, pupil values are set to NaN and later assigned the middle bin.

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
```

iii. The AI performs blink removal (using the pre-computed `likely_blink` flag), removes non-positive values, converts area to diameter, interpolates gaps, then bins and discretizes.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into 5 equal percentile bins using global percentile edges computed across all valid pupil values in the dataset (excluding NaN). For trials where all pupil values are NaN (e.g., sessions without eye tracking), the middle bin index is assigned.

ii.
```python
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The AI documented pupil diameter percentile edges as approximately [-inf, 73.88, 83.84, 92.87, 105.38, inf].

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned to the same stimulus presentation bins as neural data. Pupil values within each [start_time, start_time + 0.75s) window are averaged, handling NaN values by computing the mean of only valid (non-NaN) values.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
# Average with NaN handling:
n_valid = pup_count_cumsum[e] - pup_count_cumsum[s]
if n_valid > 0:
    pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. Same temporal alignment as neural and running speed data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns in the NWB `intervals/trials` table.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. These are the four possible outcomes for Go and Catch trials in the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is assigned as a categorical integer based on priority: Hit=0, Miss=1, False Alarm=2, Correct Rejection=3. If none of the four flags are true, it defaults to Miss (1). The outcome is static per trial and replicated across all time bins.

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

# Replicated across time bins:
np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
```

iii. The AI noted that trial outcome is static per trial but is replicated across time bins in the output array, as the output format requires shape (5, n_bins).

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing data scenarios are handled:
- **Missing eye tracking**: Sessions without `EyeTracking` in acquisition set pupil to NaN, later assigned middle bin.
- **Blink frames**: Set to NaN in pupil data and linearly interpolated.
- **Negative/zero pupil area**: Set to NaN before diameter computation.
- **NaN in is_change**: Replaced with 0 (no change) via `np.nan_to_num`.
- **Omitted stimulus presentations**: Image name forward-filled from previous presentation.
- **Zero valid neurons**: Experiment skipped entirely.
- **Missing stimulus presentations**: Experiment skipped if no stimulus key found.
- **Trials with no stimulus presentations**: Trial skipped (continues to next trial).
- **Sessions with < 2 valid trials**: Session skipped.
- **Empty neural bins** (no ophys frames in window): Left as zero.

ii.
```python
# Blink handling:
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = interpolate_nans(pupil_diameter)

# NaN is_change:
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)

# Missing trials:
if len(trial_stim_indices) == 0:
    continue
```

iii. The AI documented these edge cases in the CONVERSION_NOTES under Check 5 (edge cases) and noted no NaN/Inf values in the final neural data.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are the two passes through all NWB files. Pass 1 (statistics collection) took ~262s and Pass 2 (full conversion) took ~247s, for a total of ~510s. The I/O of opening and reading large NWB files dominates the runtime.

ii.
```python
# Pass 1: ~262s
for idx, (_, row) in enumerate(active_exps.iterrows()):
    stats = process_experiment(nwb_path, row, collect_stats_only=True)

# Pass 2: ~247s
for idx, (_, row) in enumerate(active_exps.iterrows()):
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The AI noted timing information and estimated the full conversion at ~24 minutes, which was close to the actual ~8.5 minutes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. There are several inner loops that could potentially be vectorized:
1. The per-bin neural averaging loop (lines 228-232) iterates over stimulus presentations.
2. The running speed bin averaging loop (lines 254-258) similarly iterates per bin.
3. The pupil diameter bin averaging loop (lines 265-269) iterates per bin.
4. The image name forward-fill loop (lines 237-245) iterates over presentations.
5. The per-trial loop itself (line 216) iterates over all valid trials.

ii.
```python
# Example: neural bin averaging could be vectorized if all bins had same n_frames
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The AI partially vectorized by using cumulative sums and searchsorted, but the inner loops over bins within each trial remain. Full vectorization would require handling variable-length bins.

## 9-c. What processing does the code repeat multiple times?

i. The code processes each NWB file twice (Pass 1 and Pass 2). In Pass 1, it opens each NWB file, loads stimulus presentations, running speed, and pupil data, and computes bin averages - all of which are repeated in Pass 2. The `process_experiment` function is called once with `collect_stats_only=True` and once with `collect_stats_only=False`, duplicating most of the data loading and bin computation work.

ii.
```python
# Pass 1:
stats = process_experiment(nwb_path, row, collect_stats_only=True)
# Pass 2:
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The AI documented this two-pass approach but did not discuss the redundancy. An alternative would be to cache the bin-averaged running/pupil values from Pass 1 for reuse in Pass 2, or to compute percentile edges incrementally.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several processing steps produce data that may not be fully utilized:
1. **Image name collection samples 10 experiments** but opens and reads stimulus data from each - this could be cached.
2. **Loading `change_time` from trials** (line 141) which is read but never used in the output.
3. **Loading `trial_start` and `trial_stop`** (lines 139-140) which are read but never used.
4. **The `input` arrays** are created as empty arrays `np.zeros((0, n_bins))` for every trial - unnecessary allocation since the decoder has no inputs.
5. **The `stim_stop` times** are loaded but never used (bin duration is hardcoded to 750ms).

ii.
```python
# Unused variables:
trial_change_time = trials['change_time'][:]  # Never used
trial_start = trials['start_time'][:]  # Never used
trial_stop = trials['stop_time'][:]  # Never used
stim_stop = stim['stop_time'][:]  # Never used

# Empty input arrays created but serve no purpose:
input_trials.append(np.zeros((0, n_bins), dtype=np.float32))
```

iii. The AI loaded these variables likely for potential use during development/debugging but did not remove them in the final version.
