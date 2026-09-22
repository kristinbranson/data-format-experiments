# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly using `h5py` rather than using the Allen SDK's `VisualBehaviorOphysProjectCache`. It loads experiment metadata from CSV files in `project_metadata/`, then finds corresponding NWB files on disk by matching experiment IDs to filenames. Each NWB file is opened individually to extract neural data, stimulus presentations, trials, running speed, and eye tracking.

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

iii. The AI chose to read NWB files directly via h5py rather than through the Allen SDK. From the trajectory (Step 31): "Each NWB file is one 'experiment' (one imaging plane) - treat each as a session for the decoder." The AI also filtered to only active sessions (`passive == False`) since passive sessions have no meaningful trial outcomes. The CONVERSION_NOTES.md documents this under Step 1 and Step 4.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata CSV. The list of subjects is built from `active_exps['mouse_id'].unique()`, sorted.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
subject_idx = subjects_list.index(result['mouse_id'])
```

iii. The `mouse_id` from the metadata CSV uniquely identifies each animal. This is the same identifier used by the Allen SDK.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file (one imaging plane) is treated as a separate session. The AI does NOT group multiple imaging planes from the same `ophys_session_id` into a single session. This means that for mesoscope sessions with multiple planes, each plane becomes its own "session" in the output.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
    ...
    all_sessions_neural.append(result['neural_trials'])
```

iii. From the trajectory (Step 31): "Each NWB file is one 'experiment' (one imaging plane) - treat each as a session for the decoder. Since each NWB file is one 'experiment'..." The AI chose to equate experiments with sessions for simplicity. This results in 202 "sessions" rather than the ~174 unique ophys sessions.

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table. For each valid trial (Go or Catch, not aborted, not auto-rewarded), the AI finds all stimulus presentations belonging to that trial via `stim_trials_id == tid`. Each trial then consists of the stimulus presentations within it, with one time bin per presentation (750ms bins).

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

iii. The AI uses stimulus presentations as the fundamental unit of time within a trial, giving ~11 bins per trial (matching the ~8s trial duration at 750ms/bin). From CONVERSION_NOTES.md Step 5: "Time bin = 750ms (1 per stimulus flash): Natural task unit, consistent across all equipment types."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by requiring `go | catch` to be True AND `aborted` to be False AND `auto_rewarded` to be False. Trials with no stimulus presentations (`len(trial_stim_indices) == 0`) are skipped. Sessions with fewer than 2 valid trials are excluded.

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

iii. Per the instructions: "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials." The AI also notes in CONVERSION_NOTES.md that the minimum 2-trial threshold prevents degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from dF/F traces stored in `processing/ophys/dff/traces/data` in the NWB file.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
```

iii. From CONVERSION_NOTES.md Step 5: "Use dF/F (not events): dF/F is the standard neural signal for decoding; pre-computed in NWB; events are sparser and may not decode as well for time-varying outputs."

## 2-b. How is the `neural` data processed?

i. The neural data is averaged within 750ms stimulus presentation bins. For each stimulus presentation, the AI finds ophys frames within [start_time, start_time + 0.75s) and computes the mean dF/F using a cumulative sum approach for efficiency. This rebins the data from native frame rate (~11 or ~31 Hz) to one value per 750ms stimulus presentation.

ii.
```python
bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])
...
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The AI chose to temporally bin the data to 750ms per stimulus presentation to create a common temporal resolution across all experiments (which have different native frame rates). From trajectory Step 41: "For a common time bin, I'll use the stimulus presentation interval approach (750ms bins). This is: 1. Consistent across all equipment types, 2. Aligned to the natural task structure."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using the `valid_roi` boolean column from the NWB cell specimen table, keeping only cells marked as valid.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]  # (timepoints, n_valid_neurons)
```

iii. From CONVERSION_NOTES.md Step 5: "valid_roi filter: Use only cells marked valid_roi=True in NWB, matching SDK default behavior." The SDK's `exclude_invalid_rois=True` default does the same filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation onsets. Each time bin corresponds to one stimulus flash (750ms window starting at each presentation's `start_time`). The trial's neural data is the sequence of bin-averaged dF/F values for all stimulus presentations in that trial.

ii.
```python
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
...
trial_stim_indices = np.where(stim_trials_id == tid)[0]
```

iii. The instructions say "Temporally align based on ophys timestamp." The AI aligns by averaging ophys frames within each stimulus presentation window, using `ophys_timestamps` as the temporal reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 750ms (one bin per stimulus presentation: 250ms image + 500ms grey screen). This represents significant temporal rebinning from the native ophys frame rate (~11 Hz or ~31 Hz).

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
```

iii. From trajectory Step 41: "For a common time bin, I'll use the stimulus presentation interval approach (750ms bins)." The AI reasoned this was consistent across equipment types and aligned to the task structure. CONVERSION_NOTES.md also notes this under Step 5.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `image_name` in the stimulus presentations table (`intervals/<stim_key>/image_name` in the NWB file).

ii.
```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stim['image_name'][:]])
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. The AI uses the image name associated with each stimulus presentation directly, which provides per-timebin image identity.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes using a global mapping built from all unique image names across all experiments (excluding 'omitted'). Omitted stimulus presentations are forward-filled from the previous presentation's image name.

ii.
```python
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
img_to_idx = {name: i for i, name in enumerate(IMAGE_NAMES_GLOBAL)}
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

iii. From CONVERSION_NOTES.md Step 5: "Omitted stimuli: Forward-fill image identity from previous presentation. Neural and behavioral data still recorded during omissions."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is inherently aligned because both use the same stimulus presentation bins. Each bin's image identity comes from the stimulus presentation's `image_name`, and neural data is averaged over the same 750ms window.

ii.
```python
images = stim_image_name[trial_stim_indices].copy()
# Same trial_stim_indices used for neural_matrix
```

iii. Both neural and image identity data are indexed by the same stimulus presentation indices.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the stimulus presentations table.

ii.
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The `is_change` flag is a pre-computed indicator in the NWB file that marks which stimulus presentation is the change event.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` values are converted to integers, with NaN values mapped to 0. No additional processing is applied.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The AI uses the pre-computed `is_change` flag directly rather than computing it from image identities.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary (0 or 1) directly from `is_change`. It is 1 only at the stimulus presentation where the change occurs. No thresholding is applied since it is already binary.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The `is_change` column is already a binary indicator in the NWB data.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity: aligned via the same stimulus presentation indices.

ii. Same indexing via `trial_stim_indices`.

iii. Both neural and change data are indexed by the same stimulus presentation indices.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. This is the SDK's standard running speed data, which is the 10 Hz lowpass Butterworth-filtered version.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each 750ms stimulus presentation bin using a cumulative sum approach. The averaged values are then discretized into 5 equal percentile bins. Bin edges are computed globally across all sessions in a first pass.

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

iii. From CONVERSION_NOTES.md Step 5: "Running speed: Average in 750ms bins, discretize to 5 percentile bins."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 percentile-based bins. Bin edges are computed from all valid running speed values across all sessions, with the first edge set to -inf and last to +inf.

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

iii. Percentile-based binning ensures roughly equal class counts.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is averaged within the same 750ms stimulus windows as the neural data, using the same stimulus presentation start times. This ensures temporal alignment.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
```

iii. Both running speed and neural data use the same stimulus presentation windows for averaging.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` in `acquisition/EyeTracking/pupil_tracking/area` in the NWB file, with blink detection from `acquisition/EyeTracking/likely_blink/data`.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The AI uses pupil area and computes diameter from it.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Processing steps: (1) Set blink frames to NaN, (2) Set negative/zero values to NaN, (3) Compute diameter from area as `2*sqrt(area/pi)`, (4) Interpolate NaN values using linear interpolation, (5) Average within 750ms bins, (6) Discretize into 5 percentile bins globally.

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
```

iii. From CONVERSION_NOTES.md Step 5: "Pupil NaN handling: Linear interpolation for blink frames before computing diameter and averaging."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 percentile-based bins with globally computed edges. For sessions without eye tracking, pupil values are NaN and filled with the middle bin. For remaining NaN values after initial interpolation, a second interpolation is applied during discretization.

ii.
```python
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. From CONVERSION_NOTES.md: global percentile bins ensure consistent categories across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: averaged within the same 750ms stimulus presentation windows.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
```

iii. Aligned via the same stimulus presentation time windows.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

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

iii. These are the SDK's canonical outcome labels. The fallback to Miss handles edge cases.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_rejection) and replicated across all time bins within the trial.

ii.
```python
np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
```

iii. The integer codes are in the same order as the output_values list.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Failed experiments**: If `process_experiment` returns None (e.g., 0 valid neurons or no stimulus presentations), the experiment is skipped.
- **Trials with no stimuli**: Trials with 0 stimulus presentations are skipped.
- **Missing eye tracking**: If no `EyeTracking` group exists, pupil values are NaN and later filled with the middle bin.
- **Pupil blinks/artifacts**: Blink frames and negative/zero values set to NaN, then linearly interpolated.
- **Omitted stimuli**: Image names forward-filled from previous presentation.
- **NaN running/pupil after binning**: Remaining NaN in pupil data interpolated again; running speed NaN left as-is (discretized via clip).
- **Sessions with <2 trials**: Skipped.

ii.
```python
if result is None or result['n_trials'] < 2:
    skipped += 1
    continue
...
if n_neurons == 0:
    return None
...
if len(trial_stim_indices) == 0:
    continue
```

iii. The AI documents error handling in CONVERSION_NOTES.md and the code handles multiple edge cases defensively.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading and processing each NWB file with h5py. The two-pass approach means every NWB file is read twice: once for statistics collection (Pass 1: ~262s) and once for full processing (Pass 2: ~247s). Total: ~510s.

ii. N/A (timing from conversion_full_out.txt: "Pass 1 completed in 261.6s", "Pass 2 completed in 246.6s")

iii. The AI acknowledged the I/O bottleneck but did not consolidate the two passes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop within `process_experiment` iterates over each valid trial sequentially, performing searchsorted and cumsum-based averaging. While the within-bin averaging uses cumulative sums (partially vectorized), the outer trial loop and the inner bin loop remain sequential.

ii.
```python
for trial_idx in np.where(trial_mask)[0]:
    ...
    for bi, si in enumerate(trial_stim_indices):
        s, e = ophys_bin_starts[si], ophys_bin_ends[si]
        ...
```

iii. The bin loops could potentially be vectorized further, but the I/O dominates runtime.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and fully processed twice: once in Pass 1 (to collect running/pupil statistics for percentile bin computation) and once in Pass 2 (for full conversion). This doubles the I/O and computation time.

ii.
```python
# Pass 1
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
# Pass 2
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. From CONVERSION_NOTES.md Step 6: "Two-pass approach: 1. Pass 1: Collect running speed and pupil statistics for percentile bin computation. 2. Pass 2: Full conversion with discretization using global percentile bins."

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores raw running speed and pupil diameter values in the intermediate `output_trials` dictionaries, which are then discretized in the assembly step and the raw values are discarded. Additionally, the cumulative sum precomputation for pupil count tracking is done even when not needed.

ii.
```python
output_trials.append({
    'running_speed_raw': running_binned,
    'pupil_diameter_raw': pupil_binned,
    ...
})
```

iii. These intermediate values are only used for discretization and not included in the final output.
