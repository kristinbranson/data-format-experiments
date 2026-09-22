# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads local metadata from `ophys_experiment_table.csv`, enumerates NWB files on disk with `glob`, keeps only experiment rows whose `ophys_experiment_id` has a corresponding NWB file, then filters further to `passive == False`. Each retained experiment is opened directly with `h5py`, not through the Allen SDK cache.

ii.
```python
def get_experiment_metadata():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    ...
    our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
    active_exps = our_exps[our_exps['passive'] == False].copy()
```

```python
with h5py.File(nwb_path, 'r') as f:
    dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
    ...
    stim = f['intervals'][stim_key]
    trials = f['intervals']['trials']
```

iii. In `CONVERSION_NOTES.md`, the agent says the available data are a 284-file local NWB subset and records the decision to use "active sessions only." It also states direct `h5py` reads are "Equivalent" to SDK loading.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the filtered experiment metadata, converted to strings and sorted.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
subject_idx = subjects_list.index(result['mouse_id'])
all_subject_idx.append(subject_idx)
```

iii. The notes explicitly treat `mouse_id` as the subject identifier, and the trajectory discusses counting active mice from metadata after filtering.

## 1-c. How are the data split into sessions?

i. The agent treats each retained NWB experiment as one session. It iterates row-by-row through `active_exps` and processes each `ophys_experiment_id` independently; it does not group multiple experiments by `ophys_session_id`.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The notes describe the on-disk dataset as "Each NWB file = one imaging plane from one session" and later report the full output as 202 sessions from 202 active experiments, implying the agent equated experiment files with sessions.

## 1-d. How are the data split into trials?

i. Trials are built by starting from the NWB `intervals/trials` table, selecting valid trial rows, then finding all stimulus presentations whose `trials_id` equals that trial id. Each trial becomes a sequence of 750 ms stimulus-presentation bins rather than a frame-level slice from `start_time` to `stop_time`.

ii.
```python
trials = f['intervals']['trials']
trial_ids = trials['id'][:]
...
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
    n_bins = len(trial_stim_indices)
```

iii. The trajectory shows the agent explicitly chose "750ms bins (one per image presentation)" because it thought that was the cleanest common time base across 11 Hz and 31 Hz sessions.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to Go or Catch trials and exclude `aborted` and `auto_rewarded` trials. Trials with no matching stimulus presentations are skipped. Entire experiments are skipped if they yield fewer than 2 trials; experiments with zero valid neurons or no stimulus table are also skipped.

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
    continue
```

iii. The notes justify Go/Catch-only and excluding Aborted/Auto-rewarded as matching task instructions. No separate justification is given for not checking `change_time`; the code simply never uses it.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from NWB dF/F traces in `processing/ophys/dff/traces/data`, together with the matching ophys timestamps and the ROI validity mask.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
```

iii. The notes say dF/F is precomputed in NWB and record the decision "Use dF/F (not events)." The trajectory says dF/F is standard for decoding and already available.

## 2-b. How is the `neural` data processed?

i. The agent filters to `valid_roi == True`, then averages each neuron's dF/F values within each 750 ms stimulus-presentation bin using cumulative sums. It does not keep native frame-level data.

ii.
```python
dff_valid = dff_data[:, valid_roi]
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

iii. The notes justify 750 ms bins as one stimulus cycle and justify dF/F over event traces. They also describe the cumulative-sum averaging as a speed optimization.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC is filtering ROIs with the NWB `valid_roi` flag and skipping experiments whose filtered neuron count is zero.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
n_neurons = dff_valid.shape[1]

if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
```

iii. `CONVERSION_NOTES.md` lists `valid_roi` as matching the SDK's ROI curation and explicitly calls this a key decision.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to stimulus-presentation onset. For each stimulus presentation in a trial, the code finds ophys frames in `[start_time, start_time + 0.75 s)` and averages them into one bin.

ii.
```python
bin_duration = TIME_BIN_MS / 1000.0
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
```

iii. The notes say the temporal alignment is "Stimulus presentation onset (each 750ms image flash)" and the trajectory shows the agent deliberately chose stimulus-level alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a 750 ms time bin. Native ophys frames are rebinned by averaging within each stimulus presentation.

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation
...
'time_bin_size': TIME_BIN_MS,
```

iii. The notes explicitly state "Time bin = 750ms (1 per stimulus flash)" as a key decision, justified as a common resolution across equipment types and aligned to the task structure.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation table's `image_name` field for the presentations attached to each trial via `trials_id`.

ii.
```python
stim = f['intervals'][stim_key]
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stim['image_name'][:]])
stim_trials_id = stim['trials_id'][:]
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. The notes say image identity comes from `stimulus_presentations`, not from the trials table, and note that omitted flashes are forward-filled.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent forward-fills omitted flashes and maps image names to integer codes using a global image-name list collected from only the first 10 experiments.

ii.
```python
for bi in range(len(images)):
    if images[bi] == 'omitted':
        if bi > 0:
            images[bi] = images[bi - 1]
```

```python
sample_exps = active_exps.head(min(10, len(active_exps)))
...
img_to_idx = {name: i for i, name in enumerate(IMAGE_NAMES_GLOBAL)}
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. The notes justify forward-filling omitted presentations to preserve regular 750 ms bins. The notes also say the first 10 experiments were sampled "to be fast" because the agent assumed they share the same image set structure.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is one categorical value per 750 ms stimulus bin, using exactly the same `trial_stim_indices` that determine the neural bins.

ii.
```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
...
images = stim_image_name[trial_stim_indices].copy()
...
neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The notes describe all time-varying outputs as aligned to stimulus presentations, and the code shares the same per-presentation indexing for both image labels and neural averages.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table's `is_change` flag for each presentation within a trial.

ii.
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes say `is_change` from stimulus presentations is mapped directly to the binary image-change output.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code applies `np.nan_to_num(..., nan=0)` to the selected `is_change` values and casts them to integers. There is no separate duration window or use of `change_time`.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
...
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ot['image_change'].astype(np.int64),
```

iii. The agent's notes justify 750 ms stimulus bins; within that design, it treated a presentation marked as a change presentation as the binary target.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: 0 for non-change presentations and 1 for change presentations after `nan_to_num` and integer casting.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
...
change_value_names = ['no_change', 'change']
```

iii. No extra thresholding rationale is given beyond using the binary change flag already present in the NWB stimulus table.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned per stimulus presentation and therefore shares the same 750 ms bins as the neural data.

ii.
```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
...
neural_trials.append(neural_matrix)
```

iii. The notes repeatedly describe the converted dataset as presentation-aligned, with each bin corresponding to one image flash.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived directly from NWB `processing/running/speed/data` and its timestamps.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The notes say this is the same source as the SDK's running-speed object and record that the NWB signal is already filtered.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. For each stimulus presentation, the code averages the running-speed samples that fall in that 750 ms interval, collects all such values in a first pass to compute global percentile edges, and then discretizes each trial's binned running speed in the second pass.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
run_cumsum = np.cumsum(running_speed)
...
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
```

```python
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
...
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. The notes justify 750 ms averaging as aligned to stimulus presentations and justify global percentile bins as a key discretization decision.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins using global bin edges computed from all binned running-speed values across the dataset.

ii.
```python
N_PERCENTILE_BINS = 5
...
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(valid, percentiles)
...
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. `CONVERSION_NOTES.md` explicitly says percentile bin edges are computed across all valid time points in a two-pass procedure.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by averaging running samples into the same per-presentation 750 ms bins used for neural activity.

ii.
```python
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
    n_pts = e - s
    if n_pts > 0:
        running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
```

iii. The notes say all outputs are aligned to stimulus-presentation bins rather than to native ophys frames.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from NWB eye-tracking `pupil_tracking/area`, eye-tracking timestamps, and the `likely_blink` mask.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The notes explicitly record the decision "Use area → compute diameter, interpolate NaNs." The trajectory shows the agent reasoning that area should be converted to a diameter via `2*sqrt(area/pi)`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames and nonpositive values are set to NaN, area is converted to an equivalent diameter, NaNs are linearly interpolated, then the signal is averaged within each 750 ms stimulus bin. In the second pass, any remaining per-trial NaNs are re-interpolated; if a whole trial is NaN, the output is forced to the middle bin.

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
```

```python
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The notes justify blink interpolation before averaging and present area-to-diameter conversion as a deliberate choice to match the requested output variable.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into 5 equal-percentile bins using global edges computed from all valid pupil values collected in the first pass.

ii.
```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
if len(valid_pupil) > 0:
    _, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
...
pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The notes explicitly describe a two-pass global percentile discretization for pupil diameter.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by averaging eye-tracking samples into the same per-stimulus 750 ms bins used for neural activity.

ii.
```python
for bi, si in enumerate(trial_stim_indices):
    s, e = pup_bin_starts[si], pup_bin_ends[si]
    n_valid = pup_count_cumsum[e] - pup_count_cumsum[s]
    if n_valid > 0:
        pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. The notes frame pupil processing the same way as running speed: average in stimulus bins, then discretize.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. The notes describe the output as the standard four behavioral outcomes from the trials table.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome booleans are mapped to integer labels 0, 1, 2, 3 for hit, miss, false alarm, and correct rejection. The selected label is then repeated across all time bins in the trial.

ii.
```python
if trial_hit[trial_idx]:
    outcome = 0
elif trial_miss[trial_idx]:
    outcome = 1
elif trial_fa[trial_idx]:
    outcome = 2
elif trial_cr[trial_idx]:
    outcome = 3
```

```python
np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
```

iii. The notes justify trial outcome as a static per-trial target and list the four-class mapping in the planned output specification.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several edge cases: experiments with zero valid neurons or no stimulus key are skipped; trials with no stimulus presentations are skipped; experiments with fewer than 2 trials are skipped; pupil blink and nonpositive values are interpolated; completely missing pupil within a trial is replaced by the middle category after discretization.

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
if result is None or result['n_trials'] < 2:
    continue
```

```python
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
```

iii. The notes justify interpolation for pupil blinks and say the trial-count filter is required for the decoder format. There is no stated justification for the specific "middle bin" fallback beyond keeping outputs valid.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive work is repeated full NWB loading and per-experiment processing in two passes. The notes' runtime table shows Pass 1 and Pass 2 dominate total runtime.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
for idx, (_, row) in enumerate(active_exps.iterrows()):
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. `CONVERSION_NOTES.md` estimates about 575 s for Pass 1 and about 878 s for Pass 2 on the full dataset, which is the clearest evidence of where the runtime goes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still loops over trials and then over stimulus bins within each trial for neural averaging, running-speed averaging, pupil averaging, and omitted-image forward-filling. These are the main remaining candidates for vectorization.

ii.
```python
for trial_idx in np.where(trial_mask)[0]:
    ...
    for bi, si in enumerate(trial_stim_indices):
        ...
```

```python
for bi in range(len(images)):
    if images[bi] == 'omitted':
        ...
```

iii. The notes claim cumulative sums were added as an optimization, but these nested loops remain in place.

## 9-c. What processing does the code repeat multiple times?

i. The same experiments are opened and processed twice: once in Pass 1 to collect running/pupil values for percentile edges, and again in Pass 2 to regenerate all per-trial neural and output arrays. Image-name collection also performs an additional read of up to 10 experiments before the two main passes.

ii.
```python
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
...
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The notes explicitly describe the implementation as a "two-pass approach," with a separate image-name collection step for speed.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads several arrays and scalars that are never used downstream, including `stim_stop`, `stim_omitted`, `trial_start`, `trial_stop`, `trial_change_time`, `valid_trial_ids`, and `n_total_rois`. It also performs end-of-run summary distributions after saving the pickle, which are diagnostic only.

ii.
```python
stim_stop = stim['stop_time'][:]
stim_omitted = stim['omitted'][:]
...
valid_trial_ids = trial_ids[trial_mask]
...
trial_start = trials['start_time'][:]
trial_stop = trials['stop_time'][:]
trial_change_time = trials['change_time'][:]
```

```python
all_outcomes = []
all_images = []
all_changes = []
for session_out in all_sessions_output:
    for trial_out in session_out:
        ...
```

iii. The notes emphasize diagnostics and validation throughout, but they do not justify these specific unused loads; they appear to be leftovers from exploration or reporting rather than required conversion logic.
