# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using `h5py`. It first reads the experiment metadata CSV (`ophys_experiment_table.csv`) to identify available experiments, then matches those experiment IDs to NWB files on disk. It filters to only active (non-passive) sessions. Each NWB file is opened individually and processed in the `process_experiment()` function. The code uses a two-pass approach: Pass 1 collects running speed and pupil statistics across all sessions for global percentile bin computation, then Pass 2 performs the full data extraction.

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

# In process_experiment():
with h5py.File(nwb_path, 'r') as f:
    dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
    ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
    ...
```

iii. The AI documented in CONVERSION_NOTES.md that 284 NWB files are available on disk (a subset of the full 1,936 experiments), and that it filters to 202 active experiments from 38 mice. The two-pass approach was chosen to compute global percentile bins for running speed and pupil diameter discretization.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `mouse_id` column in the experiment metadata CSV. Each experiment has a `mouse_id`, and all unique mouse IDs are collected into a sorted list. The `subject_idx` array maps each session to its subject.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
# ...
subject_idx = subjects_list.index(result['mouse_id'])
all_subject_idx.append(subject_idx)
```

iii. The AI noted 38 unique subjects in the available NWB subset (vs. 82 in the full dataset). Subject IDs are stored as strings.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one imaging experiment (one imaging plane from one session). The AI treats each NWB file/experiment as a separate session. Only active (non-passive) experiments are included.

ii.
```python
active_exps = our_exps[our_exps['passive'] == False].copy()
# Each row in active_exps is processed as one session
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The AI identified that passive sessions lack meaningful trial outcomes (no licking responses), so they are excluded. This results in 202 active sessions.

## 1-d. How are the data split into trials?

i. Trials are defined using the `intervals/trials` table in the NWB file. Each trial has an ID, and stimulus presentations are linked to trials via the `trials_id` column in the stimulus presentations table. The AI groups stimulus presentations by their `trials_id` to construct per-trial data.

ii.
```python
trial_ids = trials['id'][:]
# ...
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

iii. The AI noted that trials are defined by the experiment's behavioral trial structure, where each trial consists of multiple stimulus presentations (image flashes) until the next trial begins.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only Go and Catch trials, excluding Aborted and Auto-rewarded trials. Sessions with fewer than 2 valid trials are skipped. Additionally, trials with no associated stimulus presentations are skipped.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
# ...
if result is None or result['n_trials'] < 2:
    skipped += 1
    ...
    continue
# ...
trial_stim_indices = np.where(stim_trials_id == tid)[0]
if len(trial_stim_indices) == 0:
    continue
```

iii. The AI documented that the instruction says to include Go and Catch trials, excluding Aborted and Auto-rewarded. The 2-trial minimum comes from the target format requirement ("at least two trials within each session").

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the **dF/F traces** stored in `processing/ophys/dff/traces/data` in the NWB files. This is the pre-computed delta-F-over-F calcium fluorescence signal.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
```

iii. The AI documented in CONVERSION_NOTES.md Step 1 that "dF/F is pre-computed in NWB files" and chose dF/F over calcium events because "dF/F is the standard neural signal for decoding; pre-computed in NWB; events are sparser." The AI acknowledged that the paper (Piet et al.) uses calcium events for its neural analysis but decided dF/F was more appropriate for the decoding task.

## 2-b. How is the `neural` data processed?

i. The neural data is averaged within 750ms bins corresponding to each stimulus presentation. For each stimulus presentation, the code finds ophys frames that fall within [start_time, start_time + 0.75s) and averages the dF/F values across those frames. This is done efficiently using cumulative sums and searchsorted.

ii.
```python
bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])
# ...
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The AI decided on 750ms bins (one per stimulus presentation: 250ms image + 500ms grey screen) as the natural task unit that is consistent across both equipment types (MESO at ~11 Hz and CAM2P at ~31 Hz).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons/ROIs marked as `valid_roi=True` in the NWB cell specimen table are included. Experiments with 0 valid neurons are skipped.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
n_neurons = dff_valid.shape[1]
if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
```

iii. The AI noted that this matches the SDK's default behavior (`exclude_invalid_rois=True` in `CellSpecimens`), which filters unions, duplicates, edge ROIs, apical dendrites, too small/narrow/dim ROIs, ghost cells, and negative/zero traces.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation onsets. Each time bin corresponds to one stimulus presentation, starting at `start_time` and extending for 750ms. The ophys timestamps are used as the temporal reference, and frames falling within each 750ms window are averaged.

ii.
```python
all_stim_starts = stim_start  # from stimulus presentations table
all_stim_ends = all_stim_starts + bin_duration  # start + 0.75s
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
```

iii. The instructions say "Temporally align based on ophys timestamp." The AI uses ophys timestamps as the temporal reference for finding which neural frames fall within each stimulus bin. The metadata records `temporal_alignment_event` as "Stimulus presentation onset (each 750ms image flash)."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 750ms, corresponding to one stimulus presentation interval (250ms image + 500ms grey screen). This constitutes temporal rebinning from the native ophys frame rate (~11 Hz for mesoscope or ~31 Hz for single-plane) down to ~1.33 Hz (one bin per 750ms).

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
```

iii. The AI chose 750ms as it is the natural stimulus presentation interval. The AI documented this as a key decision, noting it provides consistency across different equipment types with different native frame rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` column in the stimulus presentations table (`intervals/<stim_key>/image_name`).

ii.
```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in stim['image_name'][:]])
```

iii. The AI noted that 8 unique image names are used per session (from sets A or B), totaling 16 unique images across all sessions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are collected across experiments, sorted, and assigned integer indices. For omitted stimulus presentations (where `image_name` is 'omitted'), the AI forward-fills the image identity from the previous non-omitted presentation. Images are then encoded as integer category indices using a global mapping.

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

iii. The AI documented that 5% of non-change presentations are omitted, and forward-filling ensures continuous image identity labels. It collects all unique image names across experiments (16 total: 8 per image set A/B).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is directly aligned with neural data because both use the same stimulus presentation bins. Each time bin in the neural data corresponds to one stimulus presentation, and the image identity is the image shown during that presentation.

ii.
```python
# Both neural and image identity use the same trial_stim_indices
for bi, si in enumerate(trial_stim_indices):
    # Neural: average dF/F in this bin
    neural_matrix[:, bi] = ...
# Image: same indices
images = stim_image_name[trial_stim_indices].copy()
```

iii. Alignment is inherent because both signals are indexed by the same stimulus presentation indices.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the stimulus presentations table.

ii.
```python
stim_is_change = stim['is_change'][:]
```

iii. The `is_change` flag is set by the experiment for the stimulus presentation when the image identity changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` values are converted to integer (0 or 1), with NaN values replaced by 0.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. NaN values in `is_change` are treated as no-change events (0).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 or 1), so no thresholding is needed. It has value 1 at the stimulus presentation where the image identity changes, and 0 otherwise.

ii.
```python
change_value_names = ['no_change', 'change']
```

iii. The instructions specify "Have value of 1 right after a change in image identity, otherwise 0."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned identically to image identity - both use the same stimulus presentation indices as the neural data bins.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. Same alignment mechanism as image identity (one value per stimulus presentation bin).

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the filtered running speed signal stored at `processing/running/speed/data` in the NWB files.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The AI uses the filtered (10 Hz lowpass Butterworth) speed rather than the unfiltered version (`speed_unfiltered`). The CONVERSION_NOTES reference indicates the SDK provides both filtered and unfiltered versions.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each 750ms stimulus bin using cumulative sums and searchsorted for efficiency. Then, values are discretized into 5 equal percentile bins using global percentile edges computed across all sessions in Pass 1.

ii.
```python
# Bin averaging:
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

# Discretization (global percentile bins):
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. The two-pass approach ensures percentile bins are computed globally across all data, not per-session.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins (quintiles). The bin edges are computed from all running speed values across all sessions, with the first edge set to -inf and the last to +inf.

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

iii. The instruction says "discretized into five equal percentile bins." The AI computes global percentile edges: [-inf, 0.004, 0.788, 15.82, 33.13, inf] cm/s.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is averaged over the same 750ms stimulus presentation windows as the neural data, then discretized. Alignment uses running speed timestamps and searchsorted to find samples within each bin window.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
```

iii. Same temporal alignment as neural data (stimulus presentation onsets, 750ms bins).

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` (pupil area) and `acquisition/EyeTracking/likely_blink/data` (blink detection flag) in the NWB files.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The AI uses pupil area rather than raw width/height, and converts to diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Processing steps: (1) Set blink frames to NaN, (2) Set negative/zero values to NaN, (3) Convert area to diameter via `d = 2*sqrt(area/pi)`, (4) Linearly interpolate NaN values, (5) Average within 750ms bins, (6) Handle remaining NaNs with interpolation, (7) Discretize into 5 equal percentile bins using global edges.

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
# ...
# Bin averaging with NaN-aware cumsum
# Discretization with global percentile bins
```

iii. The AI documented that pupil tracking uses DeepLabCut, blinks are detected by z-score > 3 with 2-frame dilation, and the `likely_blink` flag from the NWB is used directly.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: discretized into 5 equal percentile bins using global percentile edges computed across all valid (non-NaN) pupil diameter values. Sessions without eye tracking get the middle bin (bin 2).

ii.
```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
_, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
# ...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
```

iii. Global percentile edges: [-inf, 73.88, 83.84, 92.87, 105.38, inf] (in pixel units of diameter).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is averaged over the same 750ms stimulus presentation windows, using pupil tracking timestamps and searchsorted. NaN-aware averaging is used (cumulative sum of valid values divided by count of valid values).

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
# NaN-aware averaging
pup_valid = ~np.isnan(pupil_diameter)
pup_filled = np.where(pup_valid, pupil_diameter, 0.0)
pup_cumsum = np.cumsum(pup_filled)
```

iii. Same alignment as neural data and running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` columns in the trials table (`intervals/trials`).

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. These four boolean columns encode the behavioral outcome of each trial in the go/no-go change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is encoded as a categorical variable with 4 classes: Hit (0), Miss (1), False Alarm (2), Correct Rejection (3). The code checks each boolean column in priority order. If none match, it defaults to Miss. The outcome is static per trial but replicated across all time bins.

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

iii. The distribution in the full dataset: Hit 30.7%, Miss 56.8%, False Alarm 1.8%, Correct Rejection 10.7%.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing data handling strategies are used:
- **Missing eye tracking**: Sessions without `EyeTracking` in the NWB file get NaN pupil values, which are later assigned to the middle percentile bin.
- **Pupil blinks**: Frames marked as `likely_blink` are set to NaN, then linearly interpolated before bin averaging.
- **Negative/zero pupil area**: Set to NaN and interpolated.
- **Remaining NaN in pupil after binning**: Interpolated again; if all NaN, assigned middle bin.
- **NaN in is_change**: Replaced with 0 (no change) using `np.nan_to_num`.
- **Omitted stimuli**: Image identity forward-filled from previous presentation.
- **Empty trials**: Trials with no associated stimulus presentations are skipped.
- **Zero valid neurons**: Experiments are skipped.

ii.
```python
# Blink handling
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = interpolate_nans(pupil_diameter)

# Omitted stimuli forward-fill
if images[bi] == 'omitted':
    if bi > 0:
        images[bi] = images[bi - 1]

# NaN in is_change
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The AI documented these handling strategies in CONVERSION_NOTES.md and noted them as key decisions.

## 9-a. What are the most time-consuming steps of the code?

i. The two-pass approach through all NWB files is the most time-consuming. Pass 1 (statistics collection) took ~262s, and Pass 2 (full conversion) took ~247s, for a total of ~510s. Each pass involves opening each NWB file with h5py and reading large arrays.

ii.
```python
# Pass 1: ~262s
for idx, (_, row) in enumerate(active_exps.iterrows()):
    stats = process_experiment(nwb_path, row, collect_stats_only=True)

# Pass 2: ~247s
for idx, (_, row) in enumerate(active_exps.iterrows()):
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The AI reported total elapsed time of 510.4s in the conversion output.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin loops within each trial for neural averaging, running speed averaging, and pupil averaging could potentially be vectorized using advanced NumPy indexing or Numba. Despite using cumulative sums, there is still a Python-level loop over bins within each trial.

ii.
```python
# These inner loops iterate over bins within a trial
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The AI partially optimized by using cumulative sums and searchsorted, but the inner bin loops remain as Python loops.

## 9-c. What processing does the code repeat multiple times?

i. The code reads each NWB file twice (Pass 1 and Pass 2). In Pass 1, it reads and processes the same running speed, pupil data, stimulus presentations, and trial information that it processes again in Pass 2. The stimulus bin boundary computations (searchsorted) are repeated in both passes.

ii.
```python
# Pass 1: collect_stats_only=True
stats = process_experiment(nwb_path, row, collect_stats_only=True)

# Pass 2: collect_stats_only=False
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The AI noted the two-pass design as necessary for global percentile bin computation, but the NWB file I/O and much of the processing is duplicated.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several potentially unnecessary processing steps:
1. **Forward-filling omitted stimulus image names**: While the image identity is preserved, the neural data during omitted stimuli may not correspond to the filled image identity since no image was actually shown. This data is included in the output but may be misleading.
2. **Computing image name collection from 10 experiments**: The code opens up to 10 NWB files just to collect unique image names, which could be done during Pass 1.
3. **Input arrays**: Empty input arrays `np.zeros((0, n_bins))` are created for every trial, which serve no purpose since there are no decoder inputs.
4. **Plotting code**: The `plot_processing` function and its infrastructure are included even when not used.

ii.
```python
# Empty inputs created for every trial
input_trials.append(np.zeros((0, n_bins), dtype=np.float32))

# Separate image name collection pass
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
```

iii. The AI documented that there are no decoder inputs for this task, but still creates empty input arrays for format compliance.
