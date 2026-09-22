# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads experiment metadata from a CSV file (`ophys_experiment_table.csv`) and discovers available NWB files on disk via glob. Each NWB file is opened directly with `h5py` rather than using the AllenSDK's cache API. The AI filters experiments to those that are "active" (non-passive) and have corresponding NWB files. Data loading happens in a two-pass approach: Pass 1 collects running/pupil statistics for percentile computation, Pass 2 performs the full conversion.

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

```python
with h5py.File(nwb_path, 'r') as f:
    dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
    ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
    ...
```

iii. The AI chose to read NWB files directly with h5py rather than using the AllenSDK's cache. This avoids SDK dependency overhead. The two-pass approach allows computing global percentile bins before discretization. The active-only filter was applied because passive sessions have no meaningful trial outcomes.

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values from the experiment metadata CSV, restricted to active experiments with available NWB files.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
```

iii. `mouse_id` is the standard unique animal identifier in the Allen dataset.

## 1-c. How are the data split into sessions?

i. Each experiment (one NWB file = one imaging plane) is treated as a separate session. The AI does NOT group experiments by `ophys_session_id`. This means multi-plane sessions from the mesoscope are split into separate sessions rather than combined.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
    ...
    all_sessions_neural.append(result['neural_trials'])
```

iii. The AI's CONVERSION_NOTES.md mentions "Each NWB file = one imaging plane from one session," and treats each as a separate session. This simplifies the code since each NWB file is self-contained.

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB trials table. The AI filters to Go and Catch trials, excluding Aborted and Auto-rewarded trials. For each valid trial, stimulus presentations are found by matching `trials_id`, and each stimulus presentation becomes one time bin (750ms). The number of time bins per trial equals the number of stimulus presentations in that trial.

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

iii. The AI uses stimulus presentations to define trial bins rather than the trial start/stop times. This ensures each time bin corresponds exactly to one stimulus flash (250ms image + 500ms grey = 750ms).

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) Must be Go or Catch (not Aborted, not Auto-rewarded), (2) Must have at least one stimulus presentation (via `stim_trials_id` match), (3) Sessions with fewer than 2 valid trials are skipped. Additionally, experiments with 0 valid neurons (after `valid_roi` filtering) are skipped.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
trial_stim_indices = np.where(stim_trials_id == tid)[0]
if len(trial_stim_indices) == 0:
    continue
...
if result is None or result['n_trials'] < 2:
    skipped += 1
```

iii. Per instructions: include Go and Catch trials, exclude Aborted and Auto-rewarded. The 2-trial minimum prevents degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from dF/F traces stored in `processing/ophys/dff/traces/data` in the NWB file, read via h5py.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
```

iii. dF/F is pre-computed in NWB and is the standard measure for two-photon calcium imaging.

## 2-b. How is the `neural` data processed?

i. The AI applies two processing steps: (1) Filter neurons to valid ROIs only using the `valid_roi` column from the cell specimen table, and (2) Average dF/F within each 750ms stimulus bin using cumulative sums for efficiency. This transforms the neural data from native frame rate to one value per stimulus presentation.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]  # (timepoints, n_valid_neurons)
...
# Cumulative sum for fast bin averaging
dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])
...
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The valid_roi filter removes non-cell ROIs, matching the SDK's default behavior. The 750ms binning averages dF/F within each stimulus presentation period. The cumulative sum approach provides efficient bin averaging.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the `valid_roi` boolean column from the NWB cell specimen table. Only neurons with `valid_roi=True` are included.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. The CONVERSION_NOTES state: "valid_roi filter: excludes unions of cells, duplicates (>70% overlap), edge ROIs, apical dendrites, too small/narrow/dim, ghost cells."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation onset times. For each stimulus presentation in a trial, the dF/F is averaged over the 750ms window starting at the stimulus start time. This means the neural data is aligned to each stimulus flash rather than to the ophys timestamp directly.

ii.
```python
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
...
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. Each time bin corresponds to one 750ms stimulus presentation (250ms image + 500ms grey), aligning the neural data to the task structure.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 750ms (one bin per stimulus presentation). This is a rebinning from the native ophys frame rate (~11 Hz for mesoscope, ~31 Hz for single-plane) to the stimulus presentation rate. The dF/F is averaged within each 750ms window.

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
...
all_stim_ends = all_stim_starts + bin_duration
```

iii. The AI chose 750ms bins to match the natural stimulus presentation interval (250ms image + 500ms grey), making bins consistent across equipment types (mesoscope vs. single-plane). This is a significant temporal downsampling from the native frame rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` field in the stimulus presentations table within the NWB file.

ii.
```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stim['image_name'][:]])
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. Each stimulus presentation has an associated image name. The AI uses this directly rather than deriving it from the trials table.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer indices via a global mapping built from all unique non-omitted image names across experiments. Omitted stimulus presentations (where image_name is 'omitted') are forward-filled from the previous presentation's image name.

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

iii. Omitted presentations (5% of non-change stimuli) don't show an image, but neural and behavioral data are still recorded. Forward-filling maintains a continuous image identity signal.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is inherently aligned because both neural data and image identity are indexed by the same stimulus presentation indices. Each time bin corresponds to one stimulus presentation, so the image name at that bin is the image shown during that presentation.

ii.
```python
images = stim_image_name[trial_stim_indices].copy()
...
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. Since both neural data and image identity are binned by stimulus presentation, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the stimulus presentations table.

ii.
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The `is_change` flag is pre-computed in the NWB file and marks stimulus presentations where the image identity changed.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` values are extracted per stimulus presentation for the trial, NaN values are replaced with 0, and the result is cast to integer.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. Minimal processing — the flag is directly available from the NWB stimulus table.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 or 1) from the `is_change` field. No additional thresholding is applied. The value is 1 at the single stimulus presentation where the change occurs, and 0 elsewhere.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The binary change indicator from `is_change` naturally represents the image change event.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — both are indexed by stimulus presentation, so they are inherently aligned.

ii. See 4-a.

iii. Same stimulus-bin alignment as all other variables.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. This is the SDK's standard filtered running speed signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each 750ms stimulus bin using cumulative sums, then discretized into 5 equal percentile bins. Percentile edges are computed globally across all experiments in a first pass.

ii.
```python
# Bin averaging via cumulative sum
run_cumsum = np.cumsum(running_speed)
run_cumsum = np.concatenate([[0], run_cumsum])
...
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
    n_pts = e - s
    if n_pts > 0:
        running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
...
# Global percentile discretization
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. Averaging within stimulus bins matches the temporal resolution of the neural data. Global percentile bins ensure consistent categories across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins. Bin edges are computed as 0th, 20th, 40th, 60th, 80th, 100th percentiles from all valid running speed values across all experiments. The first and last edges are set to -inf and +inf respectively. `np.digitize` assigns values to bins.

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

iii. Percentile-based binning ensures roughly equal class counts, which is important for balanced decoding.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is averaged within the same 750ms stimulus bins as the neural data, using `np.searchsorted` on the running speed timestamps. Both share the same bin boundaries derived from stimulus presentation times.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
```

iii. By averaging within the same temporal bins as the neural data, alignment is guaranteed.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` (from `acquisition/EyeTracking/pupil_tracking/area`) and `likely_blink` (from `acquisition/EyeTracking/likely_blink/data`) in the NWB file.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The AI uses `pupil_area` and computes diameter from it, rather than using `pupil_width` directly.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Processing involves: (1) Set blink frames (`likely_blink=True`) to NaN, (2) Set negative/zero values to NaN, (3) Compute diameter from area: `d = 2*sqrt(area/pi)`, (4) Linearly interpolate NaN values, (5) Average within 750ms stimulus bins, (6) Discretize into 5 percentile bins globally. Additional NaN in pupil after binning is interpolated again or mapped to the middle bin if all NaN.

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
...
# NaN handling during discretization
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The AI converts area to diameter assuming a circular pupil. Blink removal before interpolation prevents artifacts. NaN values are handled by interpolation and fallback to the middle bin.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: discretized into 5 equal percentile bins computed globally across all experiments. NaN values after binning are interpolated or mapped to the middle bin.

ii.
```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
_, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
...
pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. Global percentile bins ensure consistent discretization across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is averaged within the same 750ms stimulus bins as neural data and running speed.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
...
for bi, si in enumerate(trial_stim_indices):
    s, e = pup_bin_starts[si], pup_bin_ends[si]
    n_valid = pup_count_cumsum[e] - pup_count_cumsum[s]
    if n_valid > 0:
        pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. Same bin-based alignment as running speed and neural data.

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
    outcome = 0  # Hit
elif trial_miss[trial_idx]:
    outcome = 1  # Miss
elif trial_fa[trial_idx]:
    outcome = 2  # False Alarm
elif trial_cr[trial_idx]:
    outcome = 3  # Correct Rejection
else:
    outcome = 1  # Default to Miss
```

iii. The four trial outcomes are mutually exclusive for valid Go/Catch trials. The fallback to Miss handles edge cases where no outcome flag is set.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to integer codes 0-3 (hit=0, miss=1, false_alarm=2, correct_rejection=3). It is a static per-trial variable replicated across all time bins.

ii.
```python
np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
```

iii. Trial outcome is constant throughout a trial, so it is replicated for the time-varying output format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Experiments with 0 valid neurons**: Skipped with a warning.
- **Missing stimulus presentations key**: Experiment skipped.
- **No eye tracking**: Pupil set to NaN throughout; after binning, NaN-only trials get the middle bin.
- **Omitted stimulus presentations**: Image identity forward-filled from previous presentation.
- **Blink frames in pupil**: Set to NaN, then interpolated, then NaN values after binning are interpolated or defaulted.
- **Trials with no stimulus presentations**: Skipped.
- **NaN in is_change**: Replaced with 0 via `np.nan_to_num`.
- **Sessions with <2 trials**: Skipped.

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
if result is None or result['n_trials'] < 2:
    skipped += 1
```

iii. The code handles various edge cases to prevent crashes and ensure data integrity. Missing data is handled with interpolation or sensible defaults.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the two-pass processing of NWB files. Pass 1 (collecting running/pupil statistics) takes ~262s for 202 experiments, and Pass 2 (full conversion) takes ~247s. Total time is ~510s. Reading and processing each NWB file with h5py is I/O bound.

ii. N/A (from conversion_full_out.txt: "Pass 1 completed in 261.6s", "Pass 2 completed in 246.6s", "Total elapsed time: 510.4s")

iii. The NWB files contain large arrays (dF/F traces for all neurons across full sessions). Loading these from disk is the dominant cost.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop over stimulus presentations for neural bin averaging uses a Python for-loop with cumulative sum indexing. This could potentially be vectorized using advanced indexing. The forward-fill loop for omitted presentations is also sequential. The image name decoding loop over stimulus presentations is sequential.

ii.
```python
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The AI already partially optimized by using cumulative sums, but the inner loop over bins within each trial is still Python-level iteration.

## 9-c. What processing does the code repeat multiple times?

i. The code opens and reads each NWB file twice: once in Pass 1 (collecting running/pupil statistics) and once in Pass 2 (full conversion). This doubles the I/O cost. All processing within each NWB file is also repeated (loading dF/F, computing cumulative sums, finding trial stimulus indices, etc.).

ii.
```python
# Pass 1
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
# Pass 2
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The two-pass approach was chosen to compute global percentile bins before discretization, but it requires loading each NWB file twice.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores the full cumulative sum arrays for dF/F, running speed, and pupil diameter for the entire session, even though only a subset of stimulus bins (those belonging to valid Go/Catch trials) are ultimately used. Stimulus presentations for aborted/auto-rewarded trials and non-trial periods are processed but discarded.

ii.
```python
# All stimulus presentations are pre-computed
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
dff_cumsum = np.cumsum(dff_valid, axis=0)
```

iii. Pre-computing for all stimulus presentations is simpler than filtering first, and the overhead is minor compared to I/O. The main waste is the two-pass NWB reading.
