# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading NWB files directly with `h5py`, rather than using the Allen SDK. It discovers available experiments from a metadata CSV file (`ophys_experiment_table.csv`) and maps experiment IDs to NWB file paths on disk. Each NWB file is opened individually and the relevant data arrays (dF/F traces, running speed, pupil tracking, stimulus presentations, trials) are extracted directly from the HDF5 structure.

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

iii. The AI chose to read NWB files directly via h5py rather than using the Allen SDK's `VisualBehaviorOphysProjectCache`. This avoids SDK overhead and gives direct control over which data arrays are loaded. The AI also explicitly filters to active (non-passive) sessions only, reasoning that passive sessions lack meaningful trial outcomes.

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values from the experiment metadata CSV. They are sorted and converted to strings.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
subject_idx = subjects_list.index(result['mouse_id'])
```

iii. The `mouse_id` field uniquely identifies each animal. Sorting ensures deterministic ordering.

## 1-c. How are the data split into sessions?

i. Each experiment (NWB file / imaging plane) is treated as a separate session. There is no grouping by `ophys_session_id`. Each NWB file is processed independently and produces one session in the output.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
    ...
    all_sessions_neural.append(result['neural_trials'])
```

iii. The AI treats each experiment as a session because each NWB file corresponds to one imaging plane. For the VisualBehavior project (single-plane recordings), this is largely equivalent to grouping by session, since most sessions have only one imaging plane. However, the reference groups by `ophys_session_id` and merges neurons across planes.

## 1-d. How are the data split into trials?

i. Trials are defined using the `trials` table from the NWB file. For each valid trial, the stimulus presentations belonging to that trial are identified via `trials_id`. Each stimulus presentation becomes one time bin (750ms). The trial is the collection of these stimulus bins.

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

iii. The AI uses stimulus presentations as the temporal unit, with each 750ms presentation (250ms image + 500ms grey) becoming one time bin. This differs from the reference which uses raw ophys frames from `start_time` to `stop_time`. The AI's approach creates a natural alignment between neural data and stimulus events.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only Go and Catch trials, excluding Aborted and Auto-rewarded trials. Additionally, passive sessions are excluded entirely. Trials with no matching stimulus presentations are skipped. Sessions with fewer than 2 trials are skipped.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
active_exps = our_exps[our_exps['passive'] == False].copy()
...
if result is None or result['n_trials'] < 2:
    skipped += 1
```

iii. The Go + Catch, exclude Aborted + Auto-rewarded filter matches the instructions. The passive session exclusion is an additional filter not in the reference — the AI reasoned that passive sessions lack meaningful trial outcomes for decoding.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `dff` (dF/F) traces stored in `processing/ophys/dff/traces/data` in the NWB file.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
```

iii. dF/F is the standard measure of calcium imaging neural activity. The AI reads it directly from the NWB file rather than through the SDK, but accesses the same underlying data.

## 2-b. How is the `neural` data processed?

i. The dF/F data is filtered to valid ROIs only (`valid_roi` flag), then averaged within 750ms stimulus presentation bins using a cumulative sum approach. Each time bin in the output is the mean dF/F across all ophys frames within that 750ms window.

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

iii. The AI uses the cumulative sum trick for efficient bin averaging. The 750ms binning was chosen to match the stimulus presentation interval, providing one neural activity value per stimulus flash. The valid_roi filter removes non-cell ROIs.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered using the `valid_roi` boolean flag from the NWB file's cell specimen table. Only ROIs marked as valid are retained. Experiments with 0 valid neurons are skipped entirely.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
n_neurons = dff_valid.shape[1]
if n_neurons == 0:
    return None
```

iii. The `valid_roi` filter matches the SDK's default behavior (`exclude_invalid_rois=True`), which removes unions, duplicates, edge ROIs, apical dendrites, ghost cells, etc.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation onsets. For each stimulus flash, ophys frames within a 750ms window starting at `start_time` of the presentation are averaged. This means each time bin corresponds to one stimulus presentation.

ii.
```python
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration  # bin_duration = 0.75s
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
```

iii. The AI aligns to stimulus presentation onsets rather than trial start. The reference aligns to trial start and keeps native frame rate. The AI's approach ensures each time bin corresponds to exactly one stimulus flash.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 750ms, corresponding to one stimulus presentation cycle (250ms image + 500ms grey). This is a significant rebinning from the native ophys frame rate (~11Hz/~93ms for MESO or ~31Hz/~32ms for single-plane).

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
...
bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
```

iii. The AI chose 750ms bins to match the natural task structure (one bin per stimulus flash). This ensures consistent temporal resolution across all equipment types (MESO vs CAM2P), since the stimulus interval is fixed at 750ms regardless of the imaging frame rate. However, this discards temporal information within each flash and significantly reduces temporal resolution compared to the native frame rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` column in the stimulus presentations table (`intervals/{stim_key}/image_name`).

ii.
```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stim['image_name'][:]])
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. The AI uses the per-presentation `image_name` from the stimulus table, which directly records which image was shown at each flash. This contrasts with the reference which uses `initial_image_name` and `change_image_name` from the trials table combined with `change_time`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are collected globally from up to 10 experiments, sorted, and mapped to integer indices. For omitted stimulus presentations (where `image_name` is 'omitted'), the AI forward-fills from the previous presentation.

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

iii. Forward-filling handles the ~5% of presentations that are omitted (no image shown). The AI fills with the previous image to avoid introducing a special category. The global image name collection ensures consistent encoding across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed per stimulus presentation bin (750ms), matching the neural data's temporal resolution. Each bin gets the image name for that presentation.

ii.
```python
images = stim_image_name[trial_stim_indices].copy()
```

iii. Since both neural data and image identity are indexed by stimulus presentation, they are inherently aligned — both use the same `trial_stim_indices` array.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the stimulus presentations table.

ii.
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The `is_change` flag from the stimulus table directly indicates whether a given presentation was a change event. NaN values are treated as 0 (no change).

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` values are extracted for each trial's stimulus presentations, NaN values are converted to 0, and the result is cast to int64. No additional processing is applied.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The `is_change` flag is already binary, so minimal processing is needed. The reference instead constructs a 750ms window after `change_time` and only marks go trials.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding needed — `is_change` is already binary (0 or 1). NaN values are mapped to 0.

ii. See 4-b.

iii. The data is inherently binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — one value per stimulus presentation bin, inherently aligned with the neural data bins.

ii. See 4-a.

iii. Both share the same `trial_stim_indices` indexing.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. This is the processed running speed data from the Allen SDK pipeline, stored in the NWB file.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each 750ms stimulus bin using a cumulative sum approach, then discretized into 5 percentile bins. Bin edges are computed globally across all sessions in a first pass.

ii.
```python
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

iii. The two-pass approach first collects all running speed values for percentile computation, then applies global bin edges in the second pass. The cumulative sum trick enables efficient bin averaging.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins. Bin edges are computed from all valid values across the dataset using `np.percentile` at [0, 20, 40, 60, 80, 100] percentiles. The first edge is set to -inf and last to +inf, then `np.digitize` assigns bins 0-4.

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

iii. Percentile-based binning ensures roughly equal class counts. The -inf/+inf edges handle extreme values. The reference uses a similar approach but without -inf/+inf and maps NaN to bin 0.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is averaged within the same 750ms stimulus bins as the neural data, using `np.searchsorted` on running timestamps. This ensures temporal alignment.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
```

iii. Both running speed and neural data are averaged over the same 750ms windows anchored to stimulus presentation onsets.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` in the eye tracking data (`acquisition/EyeTracking/pupil_tracking/area`), along with `likely_blink` for blink detection.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The AI uses `pupil_area` and computes diameter from it, rather than using `pupil_width` directly as the reference does.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN. Negative/zero values are set to NaN. Diameter is computed from area: `d = 2*sqrt(area/pi)`. NaN values are linearly interpolated. The result is averaged within 750ms bins and discretized into 5 percentile bins.

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
```

iii. The AI converts area to diameter assuming a circular pupil. The reference uses `pupil_width` directly, which is a linear measure. The sqrt transformation means the AI's diameter values will differ from the reference's. Additionally, the AI interpolates NaNs before binning (filling blink frames), while the reference maps NaN to bin 0 after interpolation to ophys timebase.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same percentile-based discretization as running speed: 5 bins with global edges. NaN values after binning are interpolated or assigned to the middle bin if all NaN.

ii.
```python
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The all-NaN case is handled by assigning the middle bin. Remaining NaNs are interpolated before discretization. The reference simply maps NaN to bin 0.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — averaged within 750ms stimulus bins, aligned to the same time windows as neural data.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
```

iii. Pupil diameter and neural data share the same 750ms temporal bins.

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

iii. The four outcome categories are the standard change detection outcomes. The same variables are used by the reference.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The boolean outcome flags are checked in priority order (hit → miss → false_alarm → correct_reject). The first True value determines the outcome code (0-3). If none match, the default is Miss (code 1). The outcome is replicated across all time bins of the trial.

ii.
```python
output_arr = np.stack([
    ...
    np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
], axis=0)
```

iii. The defaulting to Miss for unmatched trials differs from the reference which uses 'other' (mapped to code -1). The reference labels for the fourth outcome are `'correct_reject'`, while the AI uses `'correct_rejection'` in the output values.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions without eye tracking (`EyeTracking` not in `acquisition`) get NaN pupil values.
- **Pupil NaN/blinks**: Blink frames and negative/zero values are set to NaN, then linearly interpolated. All-NaN trials get the middle bin.
- **Missing stimulus presentations**: Trials with no matching stimulus presentations are skipped.
- **Omitted stimuli**: Image names are forward-filled from previous presentation.
- **Unknown trial outcome**: Defaults to Miss.
- **Session failures**: Sessions returning None or with <2 trials are skipped.

ii.
```python
has_eye_tracking = 'EyeTracking' in f['acquisition']
...
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = interpolate_nans(pupil_diameter)
...
if len(trial_stim_indices) == 0:
    continue
...
if images[bi] == 'omitted':
    images[bi] = images[bi - 1]
...
else:
    outcome = 1  # Default to Miss
```

iii. The AI takes a more aggressive interpolation approach for missing pupil data compared to the reference (which maps NaN to bin 0). The AI also handles omitted stimuli (forward-fill) which the reference doesn't need to address since it uses trial-level image names rather than per-presentation names.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading and processing NWB files. The code uses a two-pass approach, reading each NWB file twice — once for statistics collection (pass 1) and once for full processing (pass 2). Each NWB file read involves loading large neural data arrays.

ii.
```python
# Pass 1
stats = process_experiment(nwb_path, row, collect_stats_only=True)
# Pass 2
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The estimated total time was ~24 minutes for 202 experiments (from CONVERSION_NOTES).

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin neural averaging loop iterates over each stimulus presentation index with a Python for loop, despite using cumulative sums. The forward-fill loop for omitted stimuli could use numpy operations. The per-trial stimulus presentation matching (`np.where(stim_trials_id == tid)`) could be precomputed.

ii.
```python
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. While the cumulative sum approach is efficient, the Python loop over bins adds overhead. The reference avoids this entirely by not binning.

## 9-c. What processing does the code repeat multiple times?

i. The code reads each NWB file twice: once in pass 1 (collect_stats_only=True) for running/pupil statistics, and once in pass 2 for full processing. All the NWB loading, trial filtering, stimulus bin computation, and behavioral data extraction is duplicated.

ii.
```python
# Pass 1: Collecting running speed and pupil statistics
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
# Pass 2: Full data conversion
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The reference avoids this by doing a single pass through all sessions, storing intermediate results, and computing bin edges afterward.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The two-pass approach means all neural data processing (cumulative sums, bin averaging) is computed in pass 1 but discarded since only running/pupil values are kept. Also, the `collect_stats_only` flag still computes image identity and change flags even though only running/pupil values are returned.

ii.
```python
if collect_stats_only:
    return {
        'running_values': running_values,
        'pupil_values': pupil_values,
    }
```

iii. The neural cumulative sum computation and image processing in pass 1 are wasted work.
