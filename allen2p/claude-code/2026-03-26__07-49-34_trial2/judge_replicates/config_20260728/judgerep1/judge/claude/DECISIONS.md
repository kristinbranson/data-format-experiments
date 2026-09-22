# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads data directly from NWB files using `h5py`, bypassing the AllenSDK. It loads experiment metadata from CSV files in the `project_metadata/` directory, identifies which NWB files are available on disk, and filters to active (non-passive) experiments. Each NWB file is opened individually via `h5py.File()` in the `process_experiment()` function.

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

iii. The AI chose to read NWB files directly via h5py rather than using the AllenSDK. This was documented in CONVERSION_NOTES.md Step 1 where the AI explored the SDK code to understand the NWB file structure, then implemented direct reading. The AI also filters to active sessions only (excluding passive sessions where there are no meaningful trial outcomes).

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `mouse_id` column in the experiment metadata CSV. Unique mouse IDs are sorted and used as the subjects list.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
subject_idx = subjects_list.index(result['mouse_id'])
```

iii. The `mouse_id` is the standard identifier for subjects in the Allen dataset. The AI found 38 unique mice in the available subset of NWB files.

## 1-c. How are the data split into sessions?

i. Each ophys experiment (single imaging plane / single NWB file) is treated as a separate session. The AI does NOT group experiments by `ophys_session_id`. Each of the 202 active NWB files becomes one session in the converted data.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
    ...
    all_sessions_neural.append(result['neural_trials'])
```

iii. The AI documented in CONVERSION_NOTES.md Step 5 that each NWB file corresponds to one imaging plane. Since the available NWB files map one-to-one with experiments, each experiment was treated as a session.

## 1-d. How are the data split into trials?

i. Trials are defined using the `trials` table in each NWB file. The AI filters to Go and Catch trials (excluding Aborted and Auto-rewarded). For each valid trial, stimulus presentations belonging to that trial (matched via `trials_id`) are identified, and each stimulus presentation becomes one time bin.

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

iii. The AI followed the task instructions to include Go and Catch trials while excluding Aborted and Auto-rewarded trials. Each trial's data is assembled from its constituent stimulus presentations.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be Go or Catch, (2) not Aborted, (3) not Auto-rewarded, (4) must have at least one stimulus presentation (via `stim_trials_id`), (5) sessions with fewer than 2 valid trials are skipped.

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

iii. Per task instructions, Aborted and Auto-rewarded trials are excluded. The minimum 2-trial threshold ensures sessions have enough data for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from dF/F traces stored at `processing/ophys/dff/traces/data` in the NWB files. The data is read as (timepoints, neurons) and filtered to valid ROIs.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
...
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]  # (timepoints, n_valid_neurons)
```

iii. The AI documented in CONVERSION_NOTES.md that dF/F is pre-computed in NWB files. The AI chose dF/F over calcium events because "dF/F is the standard neural signal for decoding; pre-computed in NWB; events are sparser and may not decode as well for time-varying outputs."

## 2-b. How is the `neural` data processed?

i. The dF/F values are temporally binned into 750ms bins aligned to each stimulus presentation. Within each 750ms bin, the dF/F values are averaged using a cumulative sum approach. The data is organized as (n_neurons, n_bins_per_trial).

ii.
```python
bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration

ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')

dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])
...
neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The AI chose 750ms bins (one per stimulus flash: 250ms image + 500ms grey) as the "natural task unit" that is consistent across different equipment types (MESO at 11 Hz and CAM2P at 31 Hz).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using the `valid_roi` boolean column from the NWB cell specimen table. Only neurons marked as valid ROIs are included.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. The AI documented in CONVERSION_NOTES.md Step 5 that the `valid_roi` filter matches the SDK's default behavior (`exclude_invalid_rois=True`), which excludes unions of cells, duplicates, edge ROIs, etc.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation onsets. For each stimulus presentation within a trial, the ophys frames within a 750ms window from `start_time` are averaged. This means alignment is to each stimulus flash onset, not to a single trial event.

ii.
```python
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
...
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The AI's metadata states `temporal_alignment_event: 'Stimulus presentation onset (each 750ms image flash)'`. Each time bin corresponds to one stimulus presentation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 750ms time bins (one per stimulus presentation). This is a rebinning from the native ophys frame rate (~11 Hz for MESO, ~31 Hz for CAM2P). The `time_bin_size` in metadata is set to 750.0 ms.

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
...
'time_bin_size': TIME_BIN_MS,
```

iii. The AI justified this in CONVERSION_NOTES.md Step 5: "Time bin = 750ms (1 per stimulus flash): Natural task unit, consistent across all equipment types (MESO/CAM2P), aligned to image presentations."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` column of the stimulus presentations table (`intervals/<stim_key>/image_name`) in the NWB file.

ii.
```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stim['image_name'][:]])
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. The stimulus presentations table contains the image shown during each flash. The AI uses this directly rather than deriving it from the trials table.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer indices using a global mapping built from all unique image names across sessions (excluding 'omitted'). Omitted stimulus presentations are forward-filled with the previous non-omitted image name. 16 unique images are found across the dataset.

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

iii. Omitted stimuli (5% of non-change presentations) have no image shown, so the AI forward-fills from the previous presentation. The global mapping ensures consistent encoding across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is naturally aligned because each time bin corresponds to one stimulus presentation. The image name for that presentation is the image identity for that bin.

ii.
```python
images = stim_image_name[trial_stim_indices].copy()
...
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. Since each bin = one stimulus presentation, the image identity is inherently aligned with the neural data averaged over the same bin.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the stimulus presentations table.

ii.
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The `is_change` column directly indicates whether a stimulus presentation is a change event, pre-computed in the NWB file.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` values are converted to integers and NaN values are set to 0. No additional processing is needed since the variable is already binary.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The NaN-to-0 conversion handles any edge cases in the data.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0/1) in the source data. No thresholding is applied.

ii. See 4-b.

iii. The `is_change` column is inherently binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity -- each bin corresponds to one stimulus presentation, so `is_change` for that presentation directly maps to the bin.

ii. See 4-a.

iii. Alignment is inherent due to the bin-per-presentation structure.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. This is the Allen SDK's pre-processed running speed (10 Hz lowpass Butterworth filtered).

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each 750ms stimulus bin using a cumulative sum approach. Then it is discretized into 5 percentile-based bins. Bin edges are computed globally across all sessions in a first pass.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
run_cumsum = np.cumsum(running_speed)
run_cumsum = np.concatenate([[0], run_cumsum])
...
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
...
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. Cumulative sum-based averaging is efficient. Global percentile edges ensure consistent discretization across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Discretized into 5 equal percentile bins. Bin edges are computed from all valid running speed values across all sessions (global). The first and last edges are set to -inf and +inf respectively.

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

iii. Percentile-based binning ensures roughly equal class counts for balanced decoding.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is averaged over the same 750ms stimulus windows as the neural data, using `searchsorted` to find running speed samples within each bin boundary. This ensures temporal alignment.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
...
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
    n_pts = e - s
    if n_pts > 0:
        running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
```

iii. Both running speed and neural data are averaged over the same 750ms windows, guaranteeing alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` in the eye tracking data (`acquisition/EyeTracking/pupil_tracking/area`), with blink detection from `acquisition/EyeTracking/likely_blink/data`.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The AI uses pupil area rather than pupil width, then converts to diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Processing steps: (1) Set blink frames to NaN, (2) Set non-positive values to NaN, (3) Convert area to diameter via `2*sqrt(area/pi)`, (4) Interpolate NaN values linearly, (5) Average within 750ms bins, (6) Handle remaining NaN with interpolation or fill with middle bin, (7) Discretize into 5 percentile bins.

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
...
# During bin averaging and discretization:
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. Blinks are removed before computing diameter. The area-to-diameter conversion assumes a circular pupil. NaN interpolation prevents data gaps from propagating.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: discretized into 5 equal percentile bins with global bin edges. All-NaN trials are assigned to the middle bin (bin 2).

ii.
```python
_, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. Global percentile bins ensure consistency. Middle-bin fallback for all-NaN is a conservative default.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed -- pupil diameter is averaged within the same 750ms stimulus bins as neural data.

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

iii. Same 750ms window alignment as neural and running speed data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table of the NWB file.

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

iii. These four outcomes are the standard categories for the change detection task. The fallback to Miss handles edge cases.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_rejection). The code is constant across all time bins within a trial.

ii.
```python
np.full(n_bins, ot['trial_outcome'], dtype=np.int64)
```

iii. Trial outcome is a static per-trial variable, replicated across time bins in the output array.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **No valid neurons**: Experiments with 0 valid neurons are skipped.
- **No stimulus presentations**: Experiments without a stimulus key are skipped.
- **Trials with no stimulus presentations**: Skipped (checked via `stim_trials_id`).
- **Sessions with < 2 trials**: Skipped.
- **No eye tracking**: Pupil values set to NaN, later filled with middle bin.
- **Blink frames**: Set to NaN, then interpolated.
- **Negative/zero pupil area**: Set to NaN.
- **Omitted stimuli**: Forward-filled with previous image name.
- **NaN in `is_change`**: Set to 0.
- **Unmatched trial outcome**: Defaults to Miss.

ii.
```python
if n_neurons == 0:
    return None
if stim_key is None:
    return None
if len(trial_stim_indices) == 0:
    continue
if result is None or result['n_trials'] < 2:
    skipped += 1; continue
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = interpolate_nans(pupil_diameter)
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0)
outcome = 1  # Default to Miss
```

iii. The AI handles multiple edge cases robustly, with graceful degradation (skip bad data rather than crash).

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are the two passes over all NWB files. Pass 1 (collecting statistics) took ~262s and Pass 2 (full conversion) took ~247s for 202 experiments. Within each experiment, opening and reading the NWB file via h5py is the bottleneck.

ii. N/A (timing from conversion_full_out.txt)

iii. The two-pass approach means each NWB file is opened and read twice, doubling the I/O cost.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin loops within each trial for neural, running, and pupil averaging use a cumulative sum approach but still iterate over each stimulus presentation index:

ii.
```python
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. These inner loops could potentially be vectorized using advanced indexing, though the cumulative-sum approach already avoids the costliest operations.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and read twice -- once in Pass 1 (collecting running/pupil statistics) and once in Pass 2 (full conversion). This includes re-reading neural data, stimulus presentations, and trials in both passes.

ii.
```python
# Pass 1
stats = process_experiment(nwb_path, row, collect_stats_only=True)
# Pass 2
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The two-pass design is for computing global percentile edges before discretization, but it means all data is read twice. The reference solution avoids this by storing extracted trial data from the first pass.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes the cumulative sums for the full session's neural data, running speed, and pupil diameter in Pass 1 even though only running/pupil values are collected. The neural cumulative sum is computed in both passes. Also, for sessions that end up with < 2 trials, all the processing up to that point is wasted.

ii.
```python
# In collect_stats_only=True mode, still computes:
dff_cumsum = np.cumsum(dff_valid, axis=0)
# But only returns running_values and pupil_values
```

iii. The `collect_stats_only` flag skips storing neural trials but still loads and processes all neural data unnecessarily in Pass 1.
