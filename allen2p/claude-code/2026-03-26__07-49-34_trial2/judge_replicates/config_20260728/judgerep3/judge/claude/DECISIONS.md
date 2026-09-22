# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using `h5py`, bypassing the Allen SDK entirely. It reads experiment metadata from CSV files in the `project_metadata/` directory, then maps experiment IDs to NWB file paths on disk. Each NWB file is opened individually with `h5py.File()`.

ii.
```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
nwb_map = {}
for f in nwb_files:
    eid = int(os.path.basename(f).replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
    nwb_map[eid] = f
our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
# ...
with h5py.File(nwb_path, 'r') as f:
    dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
    ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
```

iii. The AI chose h5py for direct NWB access because the 284 NWB files were available on disk, avoiding SDK overhead and S3 download logic. The CONVERSION_NOTES document this as "Equivalent" to the SDK approach in the reference code comparison (Step 10, Check 3).

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata CSV. The list is sorted and used to create a subject index.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
# ...
subject_idx = subjects_list.index(result['mouse_id'])
```

iii. Mouse IDs come from the experiment table metadata, which is the standard identifier for each animal in the Allen dataset.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file (one imaging plane) is treated as a separate session. The AI does NOT group multiple imaging planes from the same `ophys_session_id` into a single session; each experiment is its own session.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The AI processes each experiment (imaging plane) independently. In contrast, the reference solution groups experiments by `ophys_session_id` and merges neurons from multiple planes. Since VisualBehavior sessions are single-plane, this distinction may not matter in practice for VisualBehavior data (each session has one plane), though it would matter for VisualBehaviorMultiscope sessions.

## 1-d. How are the data split into trials?

i. Trials are identified from the NWB `intervals/trials` table. For each valid trial, the AI finds stimulus presentations belonging to that trial via `stim_trials_id`, and each stimulus presentation within the trial becomes one time bin. The trial thus consists of the sequence of stimulus presentations associated with it.

ii.
```python
trials = f['intervals']['trials']
trial_ids = trials['id'][:]
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
# ...
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
    n_bins = len(trial_stim_indices)
```

iii. The AI segments trials by linking stimulus presentations to trial IDs, creating one time bin per stimulus flash (750ms). This differs from the reference, which extracts contiguous ophys frames between `start_time` and `stop_time` at native frame rate.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only Go and Catch trials, excluding Aborted and Auto-rewarded trials. Additionally, only active (non-passive) sessions are included. Trials with zero stimulus presentations are skipped. Sessions with fewer than 2 trials are skipped.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
# ...
active_exps = our_exps[our_exps['passive'] == False].copy()
# ...
if len(trial_stim_indices) == 0:
    continue
# ...
if result is None or result['n_trials'] < 2:
    skipped += 1
```

iii. The Go+Catch and exclude Aborted+Auto-rewarded filter matches the task instructions. The active-session filter excludes passive replay sessions where there is no behavioral response. The reference solution does not explicitly filter by active/passive but uses `project_code == 'VisualBehavior'` which may implicitly include only active sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the dF/F traces stored at `processing/ophys/dff/traces/data` in the NWB file. This is the pre-computed delta-F-over-F calcium fluorescence signal.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
```

iii. dF/F is the standard neural activity measure for two-photon calcium imaging, pre-computed in the NWB files by the Allen pipeline.

## 2-b. How is the `neural` data processed?

i. The dF/F traces are averaged within 750ms stimulus presentation bins. For each bin, the mean dF/F across ophys frames falling within the 750ms window is computed using a cumulative sum approach for efficiency. This is a significant temporal rebinning from the native frame rate.

ii.
```python
dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])
# ...
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The 750ms bin corresponds to one stimulus presentation cycle (250ms image + 500ms grey). The AI justified this as a "natural task unit" that is consistent across different equipment types (MESO at 11 Hz vs CAM2P at 31 Hz).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using the `valid_roi` column from the cell specimen table in the NWB file, keeping only cells marked as valid ROIs.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]  # (timepoints, n_valid_neurons)
```

iii. The `valid_roi` filter is the Allen SDK's default quality control, removing non-cell ROIs, duplicates, edge ROIs, etc. The reference solution does not apply this filter explicitly but relies on the SDK's default `exclude_invalid_rois=True` behavior, which achieves the same effect.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation onsets. Each time bin corresponds to one stimulus flash, starting from `stim_start` for that presentation and covering a 750ms window. The trial's neural data is the sequence of binned averages across its stimulus presentations.

ii.
```python
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
# ...
neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The instructions say "temporally align based on ophys timestamp." The AI aligned data to stimulus presentations rather than using the raw ophys timestamps directly. Each bin is anchored at a stimulus onset time, and ophys frames within that 750ms window are averaged.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 750ms, corresponding to one stimulus presentation cycle (250ms image + 500ms grey). This is a substantial rebinning from the native ophys frame rate (~11 Hz for MESO, ~31 Hz for single-plane). The `time_bin_size` in metadata is set to 750.0 ms.

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
```

iii. The AI chose 750ms bins to create a consistent temporal resolution across different equipment types. The reference solution preserves the native ophys frame rate (~93ms for 11 Hz MESO), resulting in approximately 8x finer temporal resolution.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` column in the stimulus presentations table (`intervals/<stim_key>/image_name`), with omitted presentations forward-filled from the previous non-omitted presentation.

ii.
```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stim['image_name'][:]])
# ...
images = stim_image_name[trial_stim_indices].copy()
for bi in range(len(images)):
    if images[bi] == 'omitted':
        if bi > 0:
            images[bi] = images[bi - 1]
```

iii. The stimulus presentations table records which image was shown at each flash. Omitted flashes (5% of non-change presentations) are marked as 'omitted' and forward-filled since the same image would have been presented had it not been omitted.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes using a global mapping built from all unique image names across all experiments (excluding 'omitted'). The mapping is sorted alphabetically for determinism.

ii.
```python
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
img_to_idx = {name: i for i, name in enumerate(IMAGE_NAMES_GLOBAL)}
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. A global mapping ensures consistent encoding across sessions. Sampling 10 experiments to collect image names is an efficiency optimization (all experiments use the same 8 images per set).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is taken per stimulus presentation bin, directly matching the neural data bins since both use the same `trial_stim_indices` array.

ii.
```python
images = stim_image_name[trial_stim_indices].copy()
```

iii. Since each time bin corresponds to one stimulus presentation, the image name for that presentation is the natural label for that bin.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the stimulus presentations table, which indicates whether a given stimulus presentation was a change event.

ii.
```python
stim_is_change = stim['is_change'][:]
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The `is_change` flag is pre-computed in the NWB stimulus presentations table by the Allen pipeline.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` values are converted to integers with NaN treated as 0 (no change). No additional processing is applied.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. Straightforward conversion of the pre-computed flag.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 or 1) from the `is_change` flag. No thresholding is needed.

ii. See 4-b.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity: per stimulus presentation bin, using the same `trial_stim_indices`.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. Direct alignment through the stimulus presentations indexing.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. This is the filtered running speed signal from the Allen pipeline.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within 750ms stimulus bins using a cumulative sum approach, then discretized into 5 equal percentile bins computed globally across all sessions (two-pass approach).

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
run_cumsum = np.cumsum(running_speed)
# ...
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
# ...
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. Bin averaging resamples running speed to the 750ms stimulus bin resolution. Percentile-based discretization ensures roughly equal class counts.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins. Bin edges are computed from all valid running speed values across all sessions, with the first edge set to -inf and last to +inf. Values are then digitized into bins 0-4.

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

iii. Equal percentile bins ensure balanced class distribution for decoding.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is averaged in the same 750ms bins as neural data, using the stimulus presentation start times to define bin boundaries.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
```

iii. Using the same stimulus presentation windows ensures temporal alignment between neural and running speed data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` in the eye tracking data (`acquisition/EyeTracking/pupil_tracking/area`), with blink detection from `acquisition/EyeTracking/likely_blink/data`.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The AI chose to use `pupil_area` rather than `pupil_width`. The reference solution uses `pupil_width` directly.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area undergoes several steps: (1) blink frames set to NaN, (2) negative/zero values set to NaN, (3) diameter computed as `2*sqrt(area/pi)`, (4) NaN values linearly interpolated, (5) averaged within 750ms stimulus bins, (6) any remaining NaNs interpolated again, then (7) discretized into 5 percentile bins.

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
# ...
if nan_mask.any():
    pupil_raw = interpolate_nans(pupil_raw)
pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The area-to-diameter conversion assumes a circular pupil. NaN interpolation before and after binning ensures continuous data. The reference uses `pupil_width` directly without the area-to-diameter conversion.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal percentile bins computed globally across all sessions.

ii.
```python
pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. Same rationale as running speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed: averaged in matching 750ms stimulus bins using `searchsorted` on the pupil timestamps.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
```

iii. Same bin-based alignment as neural and running speed data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
# ...
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

iii. These four outcome types are the canonical labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). If none of the four flags is True, the default is 1 (miss). The outcome is replicated across all time bins in the trial.

ii.
```python
np.full(n_bins, ot['trial_outcome'], dtype=np.int64)
```

iii. The default-to-miss for unclassified trials differs from the reference, which uses an 'other' label. However, in practice all valid Go+Catch, non-aborted, non-auto-rewarded trials should match one of the four outcomes.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Missing eye tracking**: Sessions without eye tracking data get NaN pupil values, which are mapped to the middle bin during discretization.
- **Blink frames**: Set to NaN, then linearly interpolated before diameter computation.
- **Negative/zero pupil area**: Set to NaN and interpolated.
- **Omitted stimuli**: Image name forward-filled from previous presentation.
- **Empty trials**: Trials with no stimulus presentations are skipped.
- **Failed experiments**: Return `None`, session is skipped.
- **All-NaN pupil**: Mapped to middle percentile bin.

ii.
```python
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
# ...
if images[bi] == 'omitted':
    if bi > 0:
        images[bi] = images[bi - 1]
```

iii. The handling is generally robust. The NaN-to-middle-bin for pupil differs from the reference which maps NaN to bin 0.

## 9-a. What are the most time-consuming steps of the code?

i. The code does a two-pass approach: Pass 1 collects running/pupil statistics, Pass 2 does full conversion. Each pass reads all NWB files, so the I/O is doubled. The most time-consuming step is reading NWB files with h5py.

ii.
```python
# Pass 1
stats = process_experiment(nwb_path, row, collect_stats_only=True)
# Pass 2
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The two-pass approach was chosen to compute global percentile bin edges before discretization.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop for neural bin averaging uses a Python loop over stimulus indices within each trial, even though cumulative sums are precomputed. The running speed and pupil bin averaging also use per-bin Python loops.

ii.
```python
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The cumulative sum precomputation mitigates much of the cost, but the inner loop over bins could potentially be vectorized further.

## 9-c. What processing does the code repeat multiple times?

i. The code reads each NWB file twice: once in Pass 1 (statistics collection) and once in Pass 2 (full processing). This means all data loading, stimulus parsing, trial filtering, and bin assignment are duplicated.

ii.
```python
# Pass 1:
stats = process_experiment(nwb_path, row, collect_stats_only=True)
# Pass 2:
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The reference solution avoids this by processing once, storing trial-level data in memory, then computing bin edges and assembling the output in a second in-memory pass.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The cumulative sum computation for neural, running, and pupil data is done for the entire session but only a subset of bins (those belonging to valid trials) are used. Also, the `collect_stats_only=True` pass computes neural bin averages and trial outcomes that are never used (only running/pupil values are returned).

ii.
```python
if collect_stats_only:
    continue  # skips appending neural/output but already computed them
```

iii. The early `continue` avoids storing the results but the computation has already been performed.
