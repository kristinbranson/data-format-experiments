# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly using `h5py` rather than through the Allen SDK's `VisualBehaviorOphysProjectCache`. It loads experiment metadata from a CSV file (`ophys_experiment_table.csv`), discovers available NWB files on disk via `glob`, and then opens each NWB file with `h5py.File()` to extract dF/F, running speed, eye tracking, stimulus presentations, and trials data.

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

with h5py.File(nwb_path, 'r') as f:
    dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
    ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
    ...
```

iii. The AI chose to read NWB files directly with h5py rather than using the Allen SDK's cache/API, noting that the data was already present on disk. It filtered to active sessions only (passive == False) because passive sessions have no meaningful trial outcomes.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata CSV, sorted and converted to strings.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
```

iii. The `mouse_id` field is the standard identifier for each animal in the Allen dataset.

## 1-c. How are the data split into sessions?

i. Each NWB file (one imaging plane/experiment) is treated as a separate session. The AI does NOT group multiple experiments by `ophys_session_id`. For the VisualBehavior project (single-plane imaging), this is functionally equivalent since each session has only one imaging plane.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The AI noted that each NWB file corresponds to one imaging plane from one session. For the VisualBehavior project specifically (as opposed to VisualBehaviorMultiscope), there is only one imaging plane per session, so this is equivalent to grouping by session.

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB trials table, filtered to Go and Catch trials (excluding Aborted and Auto-rewarded). For each valid trial, the AI finds the corresponding stimulus presentations via `stim_trials_id == tid`, and each stimulus presentation becomes one time bin. This means trials have one time bin per stimulus flash (~11 bins per trial at 750ms each).

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
valid_trial_ids = trial_ids[trial_mask]
...
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
    n_bins = len(trial_stim_indices)
```

iii. The AI used stimulus presentations to define time bins within each trial, creating one bin per 750ms stimulus flash. This was chosen to provide consistent temporal bins across different equipment types (MESO at 11Hz vs CAM2P at 31Hz).

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by requiring `(go | catch) & ~aborted & ~auto_rewarded`. Trials with no stimulus presentations (`len(trial_stim_indices) == 0`) are skipped. Sessions with fewer than 2 valid trials are excluded.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
if len(trial_stim_indices) == 0:
    continue
...
if result is None or result['n_trials'] < 2:
    skipped += 1
```

iii. Per instructions, Go and Catch trials are included while Aborted and Auto-rewarded are excluded. The minimum 2-trial threshold prevents degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the dF/F traces stored in the NWB file at `processing/ophys/dff/traces/data`.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
```

iii. dF/F is the standard measure of neural activity for two-photon calcium imaging, pre-computed in the NWB files.

## 2-b. How is the `neural` data processed?

i. The dF/F data is filtered to valid ROIs, then averaged within 750ms stimulus presentation bins using a cumulative sum approach. Each time bin corresponds to one stimulus flash (250ms image + 500ms grey), and the neural activity is the mean dF/F within that window.

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

iii. The 750ms binning was chosen to match the natural stimulus presentation interval. Cumulative sum-based averaging was used for efficiency.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the `valid_roi` flag from the NWB cell specimen table. Only cells where `valid_roi == True` are included.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. The `valid_roi` flag is the Allen SDK's standard quality control filter, excluding non-cell ROIs, duplicates, edge artifacts, etc. This matches the SDK's default behavior (`exclude_invalid_rois=True`).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation onsets. For each stimulus flash within a trial, the ophys frames within the 750ms window starting at the stimulus onset are averaged to produce one time bin. The trial starts at the first stimulus presentation and ends at the last.

ii.
```python
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration  # bin_duration = 0.75s
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
```

iii. Each stimulus presentation onset serves as the alignment point for one time bin. This ensures each bin captures the neural response to exactly one image flash plus its following grey period.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 750ms (one bin per stimulus presentation: 250ms image + 500ms grey). This is a significant rebinning from the native ophys frame rate (~11 Hz for mesoscope, ~31 Hz for single-plane).

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
```

iii. The AI chose 750ms bins because it matches the natural stimulus presentation interval. The AI argued this provides consistent temporal resolution across different equipment types (MESO vs CAM2P).

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` column in the stimulus presentations table (`intervals/<stim_key>/image_name`), matched to trials via `trials_id`.

ii.
```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stim['image_name'][:]])
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. The stimulus presentations table contains per-flash image names, which directly identify which image was on screen at each time bin.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are extracted per stimulus presentation, omitted presentations are forward-filled from the previous non-omitted image, and then mapped to integer indices via a global sorted mapping of all image names.

ii.
```python
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

iii. Omitted stimuli (5% of non-change presentations) need forward-filling because no image was shown, but the neural activity still reflects the expected/previous image. The global mapping ensures consistent encoding across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is inherently aligned because both neural data and image identity are indexed by stimulus presentation — each time bin corresponds to one stimulus flash, and both share the same index.

ii.
```python
images = stim_image_name[trial_stim_indices].copy()
neural_matrix[:, bi] = ...  # indexed by same trial_stim_indices
```

iii. Same stimulus presentation indices are used for both neural and image identity data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the stimulus presentations table.

ii.
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The `is_change` flag directly marks which stimulus presentation is the change event. NaN values are treated as non-change (0).

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` flag is directly used as the binary indicator, with NaN values converted to 0.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. Minimal processing — the flag is already binary in the source data.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding needed — `is_change` is already binary (0 or 1). NaN values are mapped to 0.

ii. See 4-b.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — indexed by stimulus presentation within each trial.

ii. See 3-c.

iii. Same alignment mechanism as all other time-varying outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. This is the filtered running speed provided in the NWB file.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each 750ms stimulus presentation bin using cumulative sums, then discretized into 5 equal percentile bins computed globally across all sessions.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
run_cumsum = np.cumsum(running_speed)
...
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
...
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. Bin averaging captures the mean speed during each stimulus flash. Global percentile binning ensures consistent categories across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins using globally computed bin edges. The `discretize_values` function computes percentile boundaries and uses `np.digitize`.

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

iii. Equal percentile bins ensure roughly balanced class distribution. Bin edges are set to -inf/+inf at boundaries to capture all values.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is averaged within the same 750ms stimulus bins as the neural data, ensuring frame-by-frame alignment.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
```

iii. Both neural and running speed are binned using the same stimulus onset boundaries.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` in the NWB file's eye tracking data (`acquisition/EyeTracking/pupil_tracking/area`), with blink detection from `acquisition/EyeTracking/likely_blink/data`.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The AI chose to use `pupil_area` and convert to diameter, rather than using a direct diameter/width measurement.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Processing steps: (1) set blink frames to NaN, (2) set negative/zero values to NaN, (3) compute diameter from area as `2*sqrt(area/pi)`, (4) linearly interpolate NaN values, (5) average within 750ms stimulus bins, (6) discretize into 5 percentile bins.

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
```

iii. Blink frames are NaN'd out before diameter computation. The area-to-diameter conversion assumes circular pupil geometry.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into 5 equal percentile bins using globally computed bin edges. Remaining NaN values after interpolation are handled by interpolating again or assigning to the middle bin if all NaN.

ii.
```python
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The middle-bin fallback for all-NaN trials provides a neutral default.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — averaged within the same 750ms stimulus bins as neural data.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
```

iii. Same alignment as all other time-varying outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
...
if trial_hit[trial_idx]:
    outcome = 0
elif trial_miss[trial_idx]:
    outcome = 1
elif trial_fa[trial_idx]:
    outcome = 2
elif trial_cr[trial_idx]:
    outcome = 3
else:
    outcome = 1  # Default to Miss
```

iii. These four boolean columns are the standard trial outcome labels for the change detection task. The fallback to Miss (1) handles edge cases.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_rejection) using a priority-based if-elif chain. The code is replicated across all time bins within the trial.

ii.
```python
output_arr = np.stack([
    ...
    np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
], axis=0)
```

iii. The integer mapping is hardcoded and matches the order: hit, miss, false_alarm, correct_rejection.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions without eye tracking data get NaN pupil values, which are assigned to the middle bin.
- **Blink frames**: Set to NaN, then linearly interpolated.
- **Negative/zero pupil area**: Set to NaN before diameter computation.
- **Omitted stimuli**: Image names forward-filled from previous presentation.
- **NaN running speed**: Handled by cumulative sum averaging (implicitly treated as 0 in cumsum).
- **Trials with no stimulus presentations**: Skipped.
- **Sessions with < 2 trials**: Skipped.
- **NaN `is_change`**: Converted to 0 via `np.nan_to_num`.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = interpolate_nans(pupil_diameter)
...
if images[bi] == 'omitted':
    if bi > 0:
        images[bi] = images[bi - 1]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The AI documented these handling strategies in CONVERSION_NOTES.md. The approaches are generally conservative defaults.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the two-pass approach: each NWB file is loaded and processed twice — once for collecting running/pupil statistics (Pass 1) and once for full data extraction (Pass 2). NWB I/O is the bottleneck.

ii.
```python
# Pass 1
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
# Pass 2
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The AI estimated the full conversion at ~24 minutes for 202 sessions.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin loops within `process_experiment` for neural averaging, running speed averaging, and pupil averaging iterate over stimulus presentations within each trial. While cumulative sums are used, the loop over bins could be further vectorized.

ii.
```python
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The cumulative sum approach is efficient but the per-bin loop adds overhead. The same loop pattern is repeated for running speed and pupil.

## 9-c. What processing does the code repeat multiple times?

i. The code loads and processes each NWB file twice — once in Pass 1 (collect_stats_only=True) and once in Pass 2 (full processing). This duplicates all the I/O and most of the computation.

ii.
```python
# Pass 1
for idx, (_, row) in enumerate(active_exps.iterrows()):
    stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
# Pass 2
for idx, (_, row) in enumerate(active_exps.iterrows()):
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The two-pass design was chosen to compute global percentile bin edges before discretizing. The reference solution avoids this by keeping raw values in memory after the first pass.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The pupil area-to-diameter conversion (`2*sqrt(area/pi)`) is unnecessary since `pupil_width` is directly available in the NWB file. Also, the cumulative sum precomputation for all stimulus presentations is computed for the entire session even though only valid trial bins are used.

ii.
```python
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
...
dff_cumsum = np.cumsum(dff_valid, axis=0)  # computed for all timepoints
```

iii. Using pupil_area and converting to diameter adds unnecessary computation when a direct diameter measurement exists. The full-session cumulative sums include data from aborted/auto-rewarded trials that are never used.
