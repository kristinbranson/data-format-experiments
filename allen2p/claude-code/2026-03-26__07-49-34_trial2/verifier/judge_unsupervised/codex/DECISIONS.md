# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads local project metadata from `ophys_experiment_table.csv`, finds NWB files on disk with `glob`, intersects metadata with available files, and keeps only non-passive experiments. It then processes each NWB file directly with `h5py`, rather than using the Allen SDK cache.

ii. ```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
...
our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
active_exps = our_exps[our_exps['passive'] == False].copy()
...
with h5py.File(nwb_path, 'r') as f:
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as working on the 284 local NWB files available in `data/`, and later decided to use only active sessions because passive sessions do not have meaningful trial outcomes.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by unique `mouse_id` values from the experiment metadata, converted to strings for the output.

ii. ```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
'mouse_id': str(exp_info['mouse_id']),
```

iii. The notes state that `mouse_id` is the natural animal identifier in the Allen metadata.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB experiment file, identified by `ophys_experiment_id`, as one decoder session. It does not group multiple experiments from the same `ophys_session_id`.

ii. ```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. In the trajectory, the AI explicitly reasoned that each NWB file contains its own neurons and should therefore be treated as a separate session.

## 1-d. How are the data split into trials?

i. Trials are defined from the NWB `intervals/trials` table, but the per-trial time axis is built from stimulus presentations linked by `stim_trials_id == tid`. Each trial therefore becomes a sequence of 750 ms stimulus bins, not a continuous `start_time` to `stop_time` ophys-frame window.

ii. ```python
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

iii. The notes and trajectory show the AI debated frame-rate standardization, then chose one 750 ms bin per stimulus flash because it matched task structure and provided a uniform time bin across equipment types.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `(go | catch) & ~aborted & ~auto_rewarded`. Trials with no associated stimulus presentations are skipped, and experiments with fewer than two retained trials are dropped.

ii. ```python
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

iii. `CONVERSION_NOTES.md` says this follows the task instruction to include Go and Catch trials and exclude Aborted and Auto-rewarded trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from NWB dF/F traces in `processing/ophys/dff/traces/data`, filtered to ROIs marked `valid_roi` in the cell specimen table.

ii. ```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. The notes say dF/F is precomputed in NWB and that `valid_roi` matches SDK curation of non-cell or low-quality ROIs.

## 2-b. How is the `neural` data processed?

i. The AI averages dF/F within each 750 ms stimulus interval using cumulative sums, producing a `(n_neurons, n_bins)` matrix per trial.

ii. ```python
bin_duration = TIME_BIN_MS / 1000.0
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])
...
neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The justification in the notes is that a 750 ms stimulus bin is the common natural unit across mesoscope and single-plane recordings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC is filtering to `valid_roi == True`. Experiments with zero valid neurons are skipped.

ii. ```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
...
if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
```

iii. The notes cite SDK defaults and whitepaper ROI curation as the reason for keeping only valid ROIs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation onset. Each bin starts at a stimulus `start_time`, and trial neural activity is indexed by the ordered stimulus presentations belonging to that trial.

ii. ```python
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
...
trial_stim_indices = np.where(stim_trials_id == tid)[0]
...
'temporal_alignment_event': 'Stimulus presentation onset (each 750ms image flash)',
```

iii. The trajectory shows the AI chose stimulus onset as the alignment event because image identity and image change are naturally defined at the flash level.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses a fixed 750.0 ms bin. Yes: the native dF/F, running, and pupil streams are rebinned/averaged into those stimulus-length bins.

ii. ```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation
...
'time_bin_size': TIME_BIN_MS,
```

iii. The notes justify this as a single bin size compatible with both ~11 Hz mesoscope and ~31 Hz single-plane recordings.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table, specifically `intervals/<stim_key>/image_name` for the presentations assigned to each trial.

ii. ```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in stim['image_name'][:]])
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. The notes describe image identity as a stimulus-presentation-level variable rather than a trial-level pre/post-change label.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Omitted presentations are forward-filled from neighboring non-omitted images, then image names are mapped to integer codes using a global image list collected from a sample of experiments.

ii. ```python
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
...
if images[bi] == 'omitted':
    if bi > 0:
        images[bi] = images[bi - 1]
...
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` says omitted stimuli should inherit identity from surrounding context and that all sessions were assumed to use the same image set.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is aligned one-to-one with the stimulus bins used for neural data: one categorical image label per 750 ms neural bin.

ii. ```python
neural_trials.append(neural_matrix)
output_trials.append({
    'image_identity': image_indices,
    ...
    'n_bins': n_bins,
})
```

iii. The AI’s stated rationale was that using the same stimulus-presentation bins makes image identity and neural data share the same timeline.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentations table field `is_change` for the presentations in each trial.

ii. ```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes frame image change as a flash-level property, so the stimulus table was used directly.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI applies `np.nan_to_num(..., nan=0)` and casts to integer, with no additional temporal expansion beyond the 750 ms stimulus bin definition.

ii. ```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The justification was that each 750 ms stimulus presentation is already the unit of change/no-change.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary: `0` for no-change and `1` for change.

ii. ```python
change_value_names = ['no_change', 'change']
...
ot['image_change'].astype(np.int64),
```

iii. This follows the decoder task’s requirement for a binary change variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned at the same stimulus-bin resolution as neural data, with one change flag per neural 750 ms bin.

ii. ```python
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ot['image_change'].astype(np.int64),
    ...
], axis=0)
```

iii. The AI’s rationale was the same shared stimulus-presentation timeline used for all outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from NWB `processing/running/speed/data` and its timestamps.

ii. ```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The notes describe this as the standard running-wheel signal already available in the NWB file.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each 750 ms stimulus bin, pooled across the dataset to compute 5 percentile edges, then discretized with `np.digitize`.

ii. ```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
run_cumsum = np.cumsum(running_speed)
...
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
...
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. The notes justify percentile discretization for balanced classes and averaging within flash-aligned bins to match the chosen time axis.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into 5 equal-percentile bins across all collected running-speed values.

ii. ```python
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(valid, percentiles)
...
binned = np.digitize(values, bin_edges[1:-1]).astype(np.int64)
```

iii. The notes say global percentile bins were used to keep the categorical levels consistent across sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by averaging all running samples whose timestamps fall inside the same 750 ms stimulus bins used for neural activity.

ii. ```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
...
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
```

iii. The AI justified this as putting behavior and neural data on the same flash-level grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking `pupil_tracking/area`, plus `likely_blink`, not from `pupil_width`.

ii. ```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. In the trajectory and notes, the AI explicitly decided that area should be converted to diameter and that blink frames should be excluded first.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames and nonpositive values are set to NaN, area is converted to diameter with `2*sqrt(area/pi)`, NaNs are interpolated, values are averaged within 750 ms stimulus bins, then discretized into 5 percentile bins.

ii. ```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
...
pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. `CONVERSION_NOTES.md` says this was meant to turn area into the requested diameter while removing blink artifacts.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded into 5 global percentile bins. Remaining NaNs inside a trial are interpolated; if a whole trial’s pupil bins are NaN, the AI assigns the middle category.

ii. ```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
_, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The notes justify this as a pragmatic way to keep all outputs categorical even when pupil tracking is partially missing.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are aligned by averaging eye-tracking samples inside the same 750 ms stimulus bins used for neural activity.

ii. ```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
...
pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. The AI’s justification was the same flash-level shared time base used for running speed and neural activity.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. The notes identify these as the canonical task-outcome labels.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI converts those booleans into integer labels 0-3, defaults unmatched trials to `miss`, and replicates the chosen label across all bins in the trial output array.

ii. ```python
if trial_hit[trial_idx]:
    outcome = 0
elif trial_miss[trial_idx]:
    outcome = 1
elif trial_fa[trial_idx]:
    outcome = 2
elif trial_cr[trial_idx]:
    outcome = 3
else:
    outcome = 1
...
np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
```

iii. The notes describe trial outcome as a static per-trial output that is repeated across time for shape compatibility.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI skips experiments with zero valid neurons or missing stimulus tables, skips trials with no linked stimulus presentations, interpolates NaNs in pupil traces, fills all-NaN pupil trials with the middle bin, and drops experiments with fewer than two trials.

ii. ```python
if n_neurons == 0:
    return None
...
if stim_key is None:
    return None
...
if len(trial_stim_indices) == 0:
    continue
...
pupil_diameter = interpolate_nans(pupil_diameter)
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
...
if result is None or result['n_trials'] < 2:
    skipped += 1
    continue
```

iii. The trajectory presents these as pragmatic safeguards to keep the conversion running on the full dataset.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive steps are repeatedly opening and parsing NWB files, and iterating over many stimulus bins when computing per-bin neural and behavioral averages. The conversion also runs two full passes over the dataset.

ii. ```python
with h5py.File(nwb_path, 'r') as f:
...
for bi, si in enumerate(trial_stim_indices):
    ...
for idx, (_, row) in enumerate(active_exps.iterrows()):
    ...
for idx, (_, row) in enumerate(active_exps.iterrows()):
```

iii. In the trajectory, the AI explicitly estimated full-run cost, identified NWB I/O and per-bin loops as bottlenecks, and then optimized averaging with cumulative sums.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining per-trial/per-bin loops for neural, running, and pupil averaging could be further vectorized. Image-name forward-filling is also a Python loop.

ii. ```python
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    ...
for bi in range(len(images)):
    if images[bi] == 'omitted':
        ...
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
```

iii. The trajectory says the AI considered replacing these loops with frame-to-bin assignment logic but stopped after cumulative-sum optimization.

## 9-c. What processing does the code repeat multiple times?

i. The code processes experiments twice: once in Pass 1 to collect running/pupil statistics and again in Pass 2 for full conversion. It also separately scans a sample of experiments just to collect image names.

ii. ```python
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
result = process_experiment(nwb_path, row, collect_stats_only=False)
...
sample_exps = active_exps.head(min(10, len(active_exps)))
for _, row in sample_exps.iterrows():
```

iii. `CONVERSION_NOTES.md` explicitly describes the implementation as a two-pass approach for global percentile discretization.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores raw running and pupil traces in temporary `output_trials` structures only to discretize them later, computes extra plotting support (`--show-processing`), and performs a separate image-name collection pass that is not part of the final dataset.

ii. ```python
output_trials.append({
    ...
    'running_speed_raw': running_binned,
    'pupil_diameter_raw': pupil_binned,
    ...
})
...
if args.show_processing and idx < 2:
    plot_processing(result, idx, nwb_path, row)
...
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
```

iii. The notes justify the two-pass raw-value storage for discretization and the plotting path for manual validation, even though neither raw arrays nor plots are part of the final saved decoder dataset.
