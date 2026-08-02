# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files on disk using `h5py`, rather than using the Allen SDK's `VisualBehaviorOphysProjectCache`. It first reads metadata from CSV files in `project_metadata/` to identify available experiments, then filters to experiments for which NWB files exist on disk. Each NWB file is opened individually with `h5py.File()` and the relevant data arrays are read directly from the HDF5 structure.

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

iii. The AI chose to read NWB files directly with h5py rather than using the Allen SDK's cache/API layer. This was documented in the CONVERSION_NOTES step 1, where the agent mapped out the NWB internal structure. The agent noted that the SDK's loading functions ultimately read from the same NWB HDF5 paths, so direct access was considered equivalent.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata CSV, filtered to only those experiments that have NWB files on disk and are non-passive.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
subject_idx = subjects_list.index(result['mouse_id'])
```

iii. The mouse_id field from the metadata CSV is the standard identifier for each animal. The AI sorts subjects alphabetically for deterministic ordering.

## 1-c. How are the data split into sessions?

i. The AI treats each individual NWB experiment file (one imaging plane) as a separate session. It does NOT group multiple imaging planes from the same ophys session together. Each experiment is processed independently.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
    ...
    all_sessions_neural.append(result['neural_trials'])
```

iii. The AI documented in CONVERSION_NOTES: "Each NWB file is one 'experiment' (one imaging plane) - treat each as a session for the decoder." The reasoning was that each NWB file contains its own distinct set of neurons, so combining planes would require handling dimension mismatches.

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table. For each valid trial, the AI finds all stimulus presentations belonging to that trial via the `trials_id` field in the stimulus presentations table. Each stimulus presentation becomes one time bin (750ms). This gives variable-length trials measured in number of stimulus presentations.

ii.
```python
trials = f['intervals']['trials']
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
    n_bins = len(trial_stim_indices)
```

iii. The AI segments trials by finding all stimulus presentations associated with each trial ID. This naturally gives one time bin per 750ms stimulus presentation (250ms image + 500ms grey). The AI noted this as a "natural task unit" that is consistent across equipment types.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only Go and Catch trials, excluding Aborted and Auto-rewarded trials. The filter is `(go | catch) & ~aborted & ~auto_rewarded`. Trials with no stimulus presentations (empty `trial_stim_indices`) are skipped. Sessions with fewer than 2 valid trials are excluded. Additionally, only active (non-passive) experiments are included.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
valid_trial_ids = trial_ids[trial_mask]
...
if len(trial_stim_indices) == 0:
    continue
...
if result is None or result['n_trials'] < 2:
    skipped += 1
```

iii. Per the task instructions, aborted and auto-rewarded trials are excluded. The AI also explicitly checks for go or catch trial types. Active-only filtering ensures meaningful trial outcomes exist (passive sessions have no licking behavior).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from dF/F traces stored at `processing/ophys/dff/traces/data` in each NWB file.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
```

iii. dF/F is the standard neural activity measure for two-photon calcium imaging, pre-computed in the NWB files by the Allen SDK pipeline.

## 2-b. How is the `neural` data processed?

i. The AI applies two processing steps: (1) filtering neurons by `valid_roi` from the cell specimen table, and (2) temporally rebinning from native ophys frame rate to 750ms bins by averaging dF/F within each stimulus presentation window. The averaging uses a cumulative sum approach for efficiency.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
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

iii. The valid_roi filter matches the SDK's default `exclude_invalid_rois=True` behavior. The 750ms binning was chosen to create a consistent temporal resolution across different equipment types (mesoscope ~11Hz vs single-plane ~31Hz) by aligning to the stimulus presentation interval.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the `valid_roi` boolean column from the NWB cell specimen table. Only neurons marked as valid ROIs are included.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. The AI documented in CONVERSION_NOTES: "valid_roi filter: excludes unions of cells, duplicates (>70% overlap), edge ROIs, apical dendrites, too small/narrow/dim, ghost cells (crosstalk), negative/zero traces." This matches the SDK's default cell curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation onsets. For each trial, the stimulus presentations belonging to that trial are identified, and dF/F is averaged within each 750ms stimulus window (from stimulus onset to onset + 750ms). This gives one neural data point per stimulus flash.

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

iii. The AI aligned to stimulus presentation onset rather than to the trial start_time or change_time, because the 750ms stimulus presentation is the fundamental temporal unit of the task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 750ms, corresponding to one stimulus presentation cycle (250ms image + 500ms grey screen). This is a significant rebinning from the native ophys frame rate (~93ms at 11Hz for mesoscope, ~32ms at 31Hz for single-plane). The AI explicitly sets `TIME_BIN_MS = 750.0`.

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
...
bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
all_stim_ends = all_stim_starts + bin_duration
```

iii. The AI chose 750ms bins as the "natural task unit" that provides consistency across equipment types. However, this represents a 8x (mesoscope) to 23x (single-plane) reduction in temporal resolution compared to native ophys frame rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` field in the stimulus presentations table (`intervals/<stim_key>/image_name` in NWB).

ii.
```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in stim['image_name'][:]])
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. The stimulus presentations table contains the image shown at each flash, which directly provides the time-varying image identity.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping built from all unique (non-omitted) image names across experiments. Omitted stimulus presentations are forward-filled with the previous non-omitted image name.

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

iii. Forward-filling for omitted stimuli was chosen because neural and behavioral data are still recorded during omissions, and the image that "should have been shown" is the one from the previous presentation.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is directly indexed by stimulus presentation, matching the neural data's 750ms binning. Each time bin in the output corresponds to the same stimulus presentation as the corresponding neural bin.

ii.
```python
images = stim_image_name[trial_stim_indices].copy()
...
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. Since both neural and image identity data are indexed by the same stimulus presentation indices, they are inherently aligned.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table.

ii.
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The `is_change` field is a pre-computed boolean in the NWB stimulus presentations table that marks the specific presentation where the image identity changed.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` values are taken directly from the stimulus presentations, with NaN values (from omitted stimuli) converted to 0. No additional processing beyond type conversion.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The pre-computed `is_change` field already encodes the binary change/no-change distinction.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed. Image change is already binary (0 or 1) from the `is_change` field.

ii. See 4-b above.

iii. The `is_change` field is inherently binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity -- indexed by stimulus presentation, matching the neural data's 750ms binning.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. Aligned via the same `trial_stim_indices` as neural and image identity data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. This is the Allen SDK's filtered running speed signal, recorded from the running wheel encoder.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each 750ms stimulus presentation bin using a cumulative sum approach. The binned values are then discretized into 5 equal percentile bins with edges computed globally across all sessions in a first pass.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
run_cumsum = np.cumsum(running_speed)
run_cumsum = np.concatenate([[0], run_cumsum])
...
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
    n_pts = e - s
    if n_pts > 0:
        running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
...
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. The two-pass approach ensures consistent percentile bins across all sessions. Cumulative sum averaging is used for efficiency.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins. Bin edges are computed from all valid running speed values across all sessions. The first and last edges are set to -inf and +inf respectively. Values are clipped to [0, n_bins-1].

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

i. Running speed is averaged within the same 750ms stimulus windows as the neural data, using `searchsorted` on running timestamps to find frames within each window.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
```

iii. Both neural and running speed are binned to the same stimulus presentation windows, ensuring temporal alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` in the eye tracking data (`acquisition/EyeTracking/pupil_tracking/area`), with blinks identified by `acquisition/EyeTracking/likely_blink/data`.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The AI chose pupil area as the source variable, noting it is "the standard measure from DeepLabCut ellipse fitting."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Processing involves: (1) setting blink frames to NaN, (2) setting negative/zero values to NaN, (3) converting area to diameter via `d = 2*sqrt(area/pi)`, (4) linearly interpolating NaN values, (5) averaging within 750ms stimulus bins, and (6) discretizing into 5 percentile bins. Remaining NaN pupil values after binning are either interpolated again or assigned the middle bin.

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
...
# During discretization:
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. Blink removal prevents corrupted data from affecting the diameter computation. The area-to-diameter conversion assumes a circular pupil approximation. NaN interpolation was preferred over assigning a default bin value.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal percentile bins computed globally across all sessions. All-NaN bins default to the middle bin (bin 2). Remaining NaN values are interpolated before discretization.

ii.
```python
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The middle-bin default for all-NaN cases is a conservative choice. NaN interpolation before discretization preserves the temporal continuity of the signal.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed -- averaged within the same 750ms stimulus presentation bins as the neural data.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
```

iii. Aligned via the same stimulus presentation binning scheme as all other variables.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` (read as `catch`) in the NWB trials table.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. These are the standard trial outcome labels from the Allen SDK's change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes: hit=0, miss=1, false_alarm=2, correct_rejection=3. A priority check is used (if hit, then miss, then false alarm, then correct rejection). Trials matching none default to miss (code 1). The outcome is replicated across all time bins within the trial.

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

iii. The mapping order follows the standard signal detection theory categories. Defaulting to miss for unclassified trials is a conservative choice. The outcome is replicated across time bins to create a time-varying representation as required by the output format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: If no `EyeTracking` in NWB, pupil data is set to NaN and assigned the middle bin.
- **Blink frames**: Set to NaN, then linearly interpolated before computing diameter.
- **Negative/zero pupil area**: Set to NaN.
- **Omitted stimuli**: Image name forward-filled from previous presentation; is_change NaN converted to 0.
- **No stimulus presentations for a trial**: Trial is skipped.
- **Sessions with <2 trials**: Skipped entirely.
- **Failed experiment processing**: Returns None, experiment is skipped.
- **NaN in running/pupil after binning**: Interpolated before discretization or assigned middle bin.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = interpolate_nans(pupil_diameter)
...
if result is None or result['n_trials'] < 2:
    skipped += 1
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The general strategy is to interpolate where possible and use conservative defaults otherwise. Missing data is documented in the conversion notes.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are (1) reading NWB files from disk with h5py (I/O bound), and (2) the two-pass approach means each NWB file is read twice -- once for statistics collection and once for full processing. The AI estimated ~24 minutes for full conversion (202 experiments).

ii.
```python
# Pass 1: stats collection
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
# Pass 2: full processing
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The agent noted that data loading is I/O bound and dominates runtime. The two-pass approach was chosen to compute global percentile bin edges before the final conversion.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin neural averaging loop iterates over stimulus presentation indices within each trial. While cumulative sums help, the inner loop could potentially be further vectorized. The forward-fill loop for omitted stimuli is also sequential.

ii.
```python
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The cumulative sum approach is already a significant optimization over naive per-frame averaging, but the per-bin loop remains.

## 9-c. What processing does the code repeat multiple times?

i. The code reads and processes each NWB file twice due to the two-pass approach. Pass 1 (`collect_stats_only=True`) loads neural data, running speed, pupil data, and computes stimulus bins -- essentially all the expensive operations -- just to collect running and pupil values for percentile computation. Pass 2 repeats all of this plus assembles the final output.

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

iii. The two-pass design is needed to compute global percentile edges before discretization, but it doubles the NWB file I/O.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In Pass 1 (stats collection), the code loads and processes all neural data (dF/F), computes cumulative sums, and builds stimulus bin assignments even though only running speed and pupil values are needed for statistics. The neural data processing in Pass 1 is entirely discarded.

ii.
```python
def process_experiment(nwb_path, exp_info, collect_stats_only=False):
    ...
    dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # Always loaded
    ...
    dff_cumsum = np.cumsum(dff_valid, axis=0)  # Always computed
    ...
    if collect_stats_only:
        return {
            'running_values': running_values,
            'pupil_values': pupil_values,
        }
```

iii. The neural data loading and cumsum computation in the stats-only pass is wasted work. A more efficient approach would use a separate lightweight function for statistics collection.
