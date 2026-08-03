# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading NWB files directly via h5py and experiment metadata from CSV files. It filters the experiment table to active (non-passive) experiments only, then maps available NWB files on disk to experiment IDs. Each NWB file is opened with h5py and data arrays are read directly from the HDF5 groups.

ii.
```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
# ...
active_exps = our_exps[our_exps['passive'] == False].copy()
# ...
with h5py.File(nwb_path, 'r') as f:
    dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
    ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
```

iii. The AI chose direct h5py access rather than the Allen SDK to avoid SDK overhead and dependency issues. It filtered to active sessions only because passive sessions lack meaningful behavioral responses (no licking), making trial outcomes meaningless. The 284 available NWB files represent a subset of the full Allen dataset.

## 1-b. How are the data split into subjects?

i. Subjects are identified from unique `mouse_id` values in the experiment metadata CSV, sorted and converted to strings.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
# ...
subject_idx = subjects_list.index(result['mouse_id'])
```

iii. The `mouse_id` field uniquely identifies each animal. Sorting ensures deterministic ordering across runs.

## 1-c. How are the data split into sessions?

i. Each experiment (single imaging plane / NWB file) is treated as a separate session. There is no grouping of multiple imaging planes within the same ophys session.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
    # Each experiment becomes one session
    all_sessions_neural.append(result['neural_trials'])
```

iii. The AI treated each NWB file as an independent session. The CONVERSION_NOTES state "Each NWB file = one imaging plane from one session." The AI did not group experiments by `ophys_session_id` to merge multiple imaging planes recorded simultaneously.

## 1-d. How are the data split into trials?

i. Trials are identified from the NWB `intervals/trials` table. The AI filters to Go and Catch trials, excluding Aborted and Auto-rewarded trials. Within each trial, stimulus presentations are found by matching `trials_id` in the stimulus presentations table. Each trial consists of the stimulus presentations belonging to that trial.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
valid_trial_ids = trial_ids[trial_mask]
# ...
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
    n_bins = len(trial_stim_indices)
```

iii. The AI uses stimulus presentations grouped by trial ID to define trial boundaries, rather than using trial `start_time`/`stop_time`. This means each trial has a variable number of stimulus bins (typically ~11 bins of 750ms each).

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) Go or Catch type, (2) not Aborted, (3) not Auto-rewarded, (4) must have at least one stimulus presentation, (5) sessions with fewer than 2 trials are skipped, (6) experiments with 0 valid neurons are skipped.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
# ...
if len(trial_stim_indices) == 0:
    continue
# ...
if result is None or result['n_trials'] < 2:
    skipped += 1
```

iii. The filtering matches the instructions to include Go and Catch trials while excluding Aborted and Auto-rewarded. Additional quality checks ensure sessions have enough data for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the dF/F (delta F over F) calcium fluorescence traces stored in the NWB file at `processing/ophys/dff/traces/data`.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
```

iii. dF/F is the standard measure of neural activity for two-photon calcium imaging. It is pre-computed in the NWB files with neuropil correction and baseline normalization.

## 2-b. How is the `neural` data processed?

i. The AI filters neurons by `valid_roi`, then averages dF/F traces within 750ms stimulus bins using a cumulative-sum approach. Each time bin corresponds to one stimulus presentation (250ms image + 500ms grey period). No additional normalization or filtering is applied.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]  # (timepoints, n_valid_neurons)
# ...
dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])
# ...
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The AI chose 750ms bins to match the natural stimulus presentation cycle and to ensure consistent temporal resolution across equipment types (mesoscope at 11 Hz vs CAM2P at 31 Hz). The cumulative sum approach enables efficient bin averaging.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the `valid_roi` boolean column from the NWB cell specimen table. Only ROIs marked as valid (true cells) are included.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. The `valid_roi` filter matches the Allen SDK's default behavior (`exclude_invalid_rois=True`), removing non-cell ROIs, duplicates, edge artifacts, and ghost cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to individual stimulus presentation onsets. For each trial, the stimulus presentations are identified and neural activity is averaged within each 750ms presentation window. The alignment is to stimulus onset times, not to trial start or change time.

ii.
```python
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration  # bin_duration = 0.75s
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
```

iii. The AI aligned to stimulus presentations because each 750ms bin corresponds to one image flash cycle. The metadata records `temporal_alignment_event: 'Stimulus presentation onset (each 750ms image flash)'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins neural data to 750ms bins (one per stimulus presentation). This is a major departure from the native ophys frame rate (~90ms at 11 Hz for mesoscope, ~32ms at 31 Hz for single-plane).

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
```

iii. The AI justified 750ms bins as the "natural task unit" — one image flash + grey period. This ensures consistent temporal resolution regardless of equipment type (MESO vs CAM2P). However, it sacrifices significant temporal resolution.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` column in the stimulus presentations table (`intervals/<stim_key>/image_name` in the NWB file).

ii.
```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stim['image_name'][:]])
# ...
images = stim_image_name[trial_stim_indices].copy()
```

iii. The stimulus presentations table has per-flash image names, providing the most granular source of image identity information.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global sorted mapping. Omitted stimulus presentations (5% of flashes) are forward-filled with the previous image name.

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

iii. Forward-filling omitted presentations preserves continuity in the image identity signal. The global image name collection is done by sampling up to 10 experiments.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed per stimulus bin (750ms), using the same stimulus presentation indices as the neural data. Each time bin has a single image identity value.

ii.
```python
images = stim_image_name[trial_stim_indices].copy()
# Same trial_stim_indices used for neural data
```

iii. Both neural and image identity use the same stimulus presentation indices, guaranteeing temporal alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the stimulus presentations table.

ii.
```python
stim_is_change = stim['is_change'][:]
# ...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The `is_change` flag is pre-computed in the NWB file, indicating which stimulus flash was the change event.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` flag is used directly as a binary indicator. NaN values are converted to 0 (no change).

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. No additional processing is needed since `is_change` is already binary.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 = no change, 1 = change). No thresholding is applied.

ii. See 4-b.

iii. The `is_change` flag is inherently binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — per stimulus bin, using the same indices.

ii. Same indexing as 3-c.

iii. Same temporal alignment as all other per-bin variables.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. This is the SDK's standard filtered running speed data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within 750ms stimulus bins using a cumulative sum approach, then discretized into 5 equal percentile bins computed globally across all sessions in a two-pass approach.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
run_cumsum = np.cumsum(running_speed)
run_cumsum = np.concatenate([[0], run_cumsum])
# ...
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
    n_pts = e - s
    if n_pts > 0:
        running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
```

iii. Averaging within 750ms bins matches the neural data temporal resolution. Global percentile binning ensures balanced class distribution for decoding.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins. Bin edges are computed globally from all valid running speed values across all sessions (pass 1), then applied to each trial (pass 2).

ii.
```python
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
# ...
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. Percentile-based binning ensures roughly equal samples per bin.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is averaged into the same 750ms stimulus bins as the neural data, using the same stimulus presentation indices.

ii. See 5-b.

iii. Both use the same stimulus presentation timing, ensuring alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` in the eye tracking data (`acquisition/EyeTracking/pupil_tracking/area`), converted to diameter via `d = 2*sqrt(area/pi)`.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
# ...
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. The AI chose to compute diameter from area assuming a circular pupil. Blink frames (from `likely_blink`) and negative/zero values are set to NaN before conversion.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN, negative/zero areas are set to NaN, diameter is computed from area, NaN values are linearly interpolated, then averaged within 750ms bins, then discretized into 5 equal percentile bins globally.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
# ...
# bin averaging and discretization follow
```

iii. Interpolating NaN values before binning prevents gaps in the signal. The two-pass approach computes global percentile edges, then applies them.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed — 5 equal percentile bins computed globally.

ii.
```python
_, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
# ...
pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. Same rationale as running speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed — averaged into 750ms stimulus bins.

ii. See 6-b.

iii. Same temporal alignment as all other variables.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. These four columns are the standard trial outcome classifications for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (hit=0, miss=1, false_alarm=2, correct_rejection=3). If none of the four flags is set, the default is miss (code 1). The outcome is constant across all time bins within a trial.

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
# ...
np.full(n_bins, ot['trial_outcome'], dtype=np.int64)
```

iii. The if/elif chain checks outcomes in priority order. Defaulting to miss was chosen as a safe fallback, though in practice all valid Go/Catch trials should match one of the four outcomes.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions without eye tracking get NaN pupil values, which are filled with the median bin during discretization.
- **Blink frames**: Set to NaN and linearly interpolated before diameter computation.
- **Negative/zero pupil area**: Set to NaN and interpolated.
- **Omitted stimuli**: Image identity is forward-filled from the previous presentation.
- **Empty trials**: Trials with no stimulus presentations are skipped.
- **Zero valid neurons**: Experiments with 0 valid ROIs are skipped.
- **Few trials**: Sessions with <2 trials are skipped.
- **All-NaN pupil**: Assigned to middle bin.

ii.
```python
if not has_eye_tracking:
    pupil_diameter = None
# ...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
# ...
if n_neurons == 0:
    return None
```

iii. The AI documented multiple edge cases and handling strategies, prioritizing data preservation while avoiding propagation of invalid values.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via h5py, which involves reading large dF/F arrays and behavioral data. This is done twice (pass 1 for statistics, pass 2 for full processing).

ii. N/A

iii. Each NWB file contains full-session neural data (140K+ timepoints x neurons). The two-pass approach doubles the I/O cost.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin loops for neural averaging and running speed averaging iterate over stimulus indices. While the cumsum approach avoids per-element operations, the loop itself could be vectorized using advanced indexing.

ii.
```python
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The loops are already partially optimized via cumsum, but the outer trial loop and inner bin loop are sequential.

## 9-c. What processing does the code repeat multiple times?

i. The code uses a two-pass approach. Pass 1 loads each NWB file to collect running and pupil statistics for percentile computation. Pass 2 loads each NWB file again for full data extraction. This doubles the NWB file I/O.

ii.
```python
# Pass 1
stats = process_experiment(nwb_path, row, collect_stats_only=True)
# Pass 2
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The two-pass approach was chosen to compute global percentile bin edges before discretization. A single-pass approach could collect raw values and discretize afterward, avoiding the redundant file reads.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The two-pass processing is partially redundant since pass 1 data is discarded and recomputed in pass 2. Additionally, the `interpolate_nans` function is called on pupil data during both passes. The image name collection step (`collect_all_image_names`) reads up to 10 NWB files separately just to build the image mapping, which could be done during either pass.

ii.
```python
# Separate image name collection
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
```

iii. The separate image name collection adds extra file I/O that could be integrated into the main processing passes.
