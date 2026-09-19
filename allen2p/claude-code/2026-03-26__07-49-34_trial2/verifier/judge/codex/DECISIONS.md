# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads the local metadata CSV and enumerates local NWB files on disk, keeps only experiments whose `ophys_experiment_id` has a matching NWB file, filters to active (`passive == False`) experiments, and then reads each NWB file directly with `h5py`. It makes a first full pass to collect running/pupil values for global percentile bins and a second full pass to build the converted dataset.

ii.
```python
def get_experiment_metadata():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    ...
    our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
    active_exps = our_exps[our_exps['passive'] == False].copy()
    active_exps = active_exps.sort_values('ophys_experiment_id').reset_index(drop=True)
    return active_exps, nwb_map

with h5py.File(nwb_path, 'r') as f:
    ...

for idx, (_, row) in enumerate(active_exps.iterrows()):
    stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
for idx, (_, row) in enumerate(active_exps.iterrows()):
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. In `CONVERSION_NOTES.md`, the agent argues that the available data are a local NWB subset, that passive sessions should be excluded because they do not support the required trial-outcome output, and that direct NWB reads are equivalent to SDK loading for this subset. The notes also explicitly justify the two-pass design as necessary for global percentile binning of running speed and pupil diameter.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values in the experiment metadata.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
subject_idx = subjects_list.index(result['mouse_id'])
all_subject_idx.append(subject_idx)
```

iii. The notes repeatedly summarize the dataset in terms of unique mice from metadata, e.g. “202 active experiments from 38 mice,” so the agent treated `mouse_id` as the canonical subject identifier.

## 1-c. How are the data split into sessions?

i. The script treats each NWB experiment file, i.e. each `ophys_experiment_id`, as a separate session. It does not group multiple experiments from the same `ophys_session_id`.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
    ...
    session_info.append({
        'experiment_id': eid,
        'mouse_id': result['mouse_id'],
        'brain_region': result['brain_region'],
        'n_neurons': result['n_neurons'],
        'n_trials': result['n_trials'],
    })
```

iii. The trajectory explicitly says, “Each NWB file is one experiment (one imaging plane) - treat each as a session for the decoder,” because each file has its own neurons. The notes describe this as using 202 active experiments as 202 sessions.

## 1-d. How are the data split into trials?

i. Trials are built by taking each included trial ID from the NWB `trials` table and then collecting all stimulus presentations in the stimulus table whose `trials_id` equals that trial ID. Trial length is therefore the number of stimulus presentations in that trial, not the number of ophys frames between `start_time` and `stop_time`.

ii.
```python
trials = f['intervals']['trials']
trial_ids = trials['id'][:]
...
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto

for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
    n_bins = len(trial_stim_indices)
```

iii. The notes justify 750 ms “one bin per stimulus presentation” as the “natural task unit,” so the agent chose to define the trial as a sequence of stimulus bins rather than as a variable-length ophys-frame slice from `start_time` to `stop_time`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to Go or Catch trials and exclude aborted and auto-rewarded trials. Trials with no associated stimulus presentations are skipped implicitly, and experiments with fewer than two surviving trials are dropped from the final dataset.

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

iii. The notes say, “Include Go+Catch only per task instructions,” and also list a minimum-trials check as an edge-case safeguard. I did not find a separate justification for omitting an explicit `change_time` validity filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the NWB dF/F trace matrix `processing/ophys/dff/traces/data`, filtered by `valid_roi` from the cell specimen table.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. The notes say dF/F is precomputed in NWB and should be used rather than recomputed, and they explicitly cite `valid_roi` as the SDK-style ROI quality filter.

## 2-b. How is the `neural` data processed?

i. After filtering to valid ROIs, the script averages each neuron’s dF/F over every 750 ms stimulus window using cumulative sums. The per-trial neural matrix is `(n_neurons, n_stimulus_bins)`.

ii.
```python
bin_duration = TIME_BIN_MS / 1000.0
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])
...
neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The notes justify this by saying 750 ms is the common task unit across microscopes and that dF/F should be averaged within each image-plus-grey flash interval for a uniform bin size across sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC is keeping ROIs with `valid_roi == True`. Experiments with zero valid neurons are skipped entirely.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
n_neurons = dff_valid.shape[1]

if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
```

iii. `CONVERSION_NOTES.md` says this matches the Allen SDK default behavior (`exclude_invalid_rois=True`) and lists the whitepaper rationale for `valid_roi` as the main cell-level curation rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script aligns neural data to stimulus-presentation onsets: each timepoint is the average neural activity in the 750 ms window starting at a stimulus flash onset. It does not preserve the native ophys-frame timeline within trials.

ii.
```python
TIME_BIN_MS = 750.0
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
```

iii. The notes explicitly describe the alignment event as “Stimulus presentation onset (each 750ms image flash)” and defend it as the natural unit of the task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 750 ms time bin, one bin per stimulus presentation. Neural, running, and pupil data are all rebinned into these windows.

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
...
'time_bin_size': TIME_BIN_MS,
```

iii. The notes make this one of the core design decisions: 750 ms is shared across equipment types and aligns to image presentations, so the agent preferred this over native 11 Hz or 31 Hz frame bins.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation table’s `image_name` field for the presentations associated with each trial.

ii.
```python
stim = f['intervals'][stim_key]
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stim['image_name'][:]])
stim_trials_id = stim['trials_id'][:]
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. The notes say image identity should come from stimulus presentations and mention that each session contains 8 natural scene images, with omissions handled separately.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The script forward-fills omitted presentations within a trial so every stimulus bin has an image label, then maps image names to integer codes using a global vocabulary collected from only the first 10 experiments.

ii.
```python
def collect_all_image_names(active_exps, nwb_map):
    all_images = set()
    sample_exps = active_exps.head(min(10, len(active_exps)))
    ...
    if name != 'omitted':
        all_images.add(name)

images = stim_image_name[trial_stim_indices].copy()
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

iii. The notes justify forward-filling by saying each session uses the same image set and omissions should inherit the surrounding identity. The notes also say the image-name sampling is a speed optimization because “they use the same 8 images,” although later notes acknowledge 16 images globally across two image sets.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is aligned bin-by-bin to the same 750 ms stimulus windows used for the neural data, with one image category per stimulus presentation.

ii.
```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
n_bins = len(trial_stim_indices)
...
neural_trials.append(neural_matrix)
output_trials.append({
    'image_identity': image_indices,
    ...
    'n_bins': n_bins,
})
```

iii. The justification is the same as for neural alignment: the agent wanted all outputs to live on the shared stimulus-presentation timeline rather than on native ophys frames.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change` field.

ii.
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes say the goal is to identify the change flash itself, and on the stimulus-bin representation `is_change` is the direct indicator for that event.

## 4-b. What processing is involved in computing `output` *Image change*?

i. There is almost no additional processing beyond subsetting the trial’s stimulus presentations and converting missing values to `0`.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes describe image change as “1 at change flash only,” so the agent treated the stored `is_change` flag as already being the desired output signal.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary variable with categories 0 = no change and 1 = change.

ii.
```python
change_value_names = ['no_change', 'change']
...
'output_values': [
    image_value_names,
    change_value_names,
    running_value_names,
    pupil_value_names,
    outcome_value_names,
],
```

iii. The notes and README both describe image change as a binary output, and no further thresholding beyond that binary coding is applied.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned directly to the same per-stimulus bins as the neural matrix, one change flag per 750 ms stimulus window.

ii.
```python
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ot['image_change'].astype(np.int64),
    running_disc.astype(np.int64),
    pupil_disc.astype(np.int64),
    np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
], axis=0)
```

iii. The notes justify this shared binning as making the change signal naturally coincide with the corresponding neural bin for the change flash.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the NWB running-speed stream `processing/running/speed`.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The notes identify NWB running speed as the standard locomotion measure and state that the saved NWB speed is already the Allen-processed filtered signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed samples are averaged within each 750 ms stimulus window using cumulative sums, collected across all experiments to compute global quintile edges, and then discretized to 5 bins.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
run_cumsum = np.cumsum(running_speed)
run_cumsum = np.concatenate([[0], run_cumsum])
...
running_binned = np.zeros(n_bins, dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
    n_pts = e - s
    if n_pts > 0:
        running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
...
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. The notes justify this as using one running-speed value per stimulus presentation and using global percentile binning so the five decoder classes are roughly balanced across the dataset.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into 5 equal-percentile bins computed globally across all experiments.

ii.
```python
N_PERCENTILE_BINS = 5
...
percentiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(valid, percentiles)
...
running_value_names = [f'speed_q{i+1}' for i in range(N_PERCENTILE_BINS)]
```

iii. The notes state that the decoder task explicitly asked for five equal-percentile bins, so the agent used global quintiles rather than session-wise thresholds.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by averaging all wheel-speed samples whose timestamps fall inside the same 750 ms stimulus bins used for neural activity.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
...
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
```

iii. The notes defend this as using the stimulus presentation as the common temporal reference for all streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the eye-tracking `area` signal and `likely_blink` mask in the NWB file, not from an already stored width/diameter variable.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The notes say the raw NWB has area, width, and height, and justify using area so diameter can be computed explicitly as `2*sqrt(area/pi)` after blink masking.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to `NaN`, nonpositive areas are set to `NaN`, area is converted to diameter via `2*sqrt(area/pi)`, missing values are linearly interpolated, then values are averaged in 750 ms stimulus bins and discretized globally to quintiles.

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
...
pupil_binned = np.full(n_bins, np.nan, dtype=np.float32)
if has_eye_tracking and pupil_diameter is not None:
    for bi, si in enumerate(trial_stim_indices):
        s, e = pup_bin_starts[si], pup_bin_ends[si]
        n_valid = pup_count_cumsum[e] - pup_count_cumsum[s]
        if n_valid > 0:
            pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The notes justify the blink masking from Allen’s eye-tracking QC and argue that interpolation is needed to avoid losing bins; they also describe the global five-bin discretization as matching the decoder spec.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into 5 equal-percentile bins computed globally across the dataset.

ii.
```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
if len(valid_pupil) > 0:
    _, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
...
pupil_value_names = [f'pupil_q{i+1}' for i in range(N_PERCENTILE_BINS)]
```

iii. The notes explicitly describe the target output as “5 percentile bins” and use the same global-threshold logic as for running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by averaging eye-tracking samples within the same 750 ms stimulus windows as the neural data.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
...
pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. The notes present stimulus-presentation bins as the common alignment grid for all outputs and neural activity.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the NWB trials-table boolean fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. The notes describe these as the behavioral outcome labels required by the decoder task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes are mapped to integers 0-3 in priority order hit, miss, false alarm, correct rejection, with an unmatched case defaulting to miss. The chosen outcome is then repeated across every time bin of the trial.

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
np.full(n_bins, ot['trial_outcome'], dtype=np.int64)
```

iii. The notes describe trial outcome as a static per-trial variable, so the agent replicated it across time bins. I did not find a stronger justification for the “default to miss” fallback beyond it being a convenient default.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The script skips experiments with no valid neurons or no stimulus table, skips individual trials with no stimulus presentations, interpolates missing pupil values, fills all-NaN pupil trials with the middle discrete bin, skips experiments with fewer than two trials, and skips eye tracking entirely if it is absent. It also leaves empty running bins at zero because the binned running array is initialized to zeros.

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
pupil_diameter = interpolate_nans(pupil_diameter)
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
...
if result is None or result['n_trials'] < 2:
    skipped += 1
    continue
```

iii. The notes frame these choices as pragmatic robustness measures: do not crash on bad files, keep enough data for the decoder, and replace missing continuous behavior with interpolated or neutral categorical values rather than dropping sessions.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive steps are the two full passes over every NWB experiment and the repeated per-trial/per-bin extraction inside `process_experiment`. There is also an extra partial pass to collect global image names.

ii.
```python
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
...
for idx, (_, row) in enumerate(active_exps.iterrows()):
    stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
for idx, (_, row) in enumerate(active_exps.iterrows()):
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The notes explicitly report Pass 1 and Pass 2 runtimes and present the whole converter as a two-pass pipeline. The trajectory also identifies the per-bin processing inside each experiment as a major runtime concern.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining expensive Python loops are the loop over trials, the loop over stimulus bins inside each trial for neural/running/pupil averaging, and the loop that forward-fills omitted image presentations.

ii.
```python
for trial_idx in np.where(trial_mask)[0]:
    ...
    for bi, si in enumerate(trial_stim_indices):
        ...
    for bi in range(len(images)):
        if images[bi] == 'omitted':
            ...
    for bi, si in enumerate(trial_stim_indices):
        ...
    for bi, si in enumerate(trial_stim_indices):
        ...
```

iii. The notes say the script already uses cumulative sums and `searchsorted` as optimizations, but the trajectory still singles out the per-bin loop as the main bottleneck that motivated those optimizations.

## 9-c. What processing does the code repeat multiple times?

i. The dataset is reread and reprocessed multiple times: once to collect image names, once for running/pupil percentile statistics, and once again for the final conversion. Percentile discretization is intentionally structured as a two-pass pipeline.

ii.
```python
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
...
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. `CONVERSION_NOTES.md` explicitly documents this as an implementation choice: “two-pass approach” with Pass 1 for percentile statistics and Pass 2 for full conversion.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script reads several variables it never uses downstream (`stim_stop`, `stim_omitted`, `valid_trial_ids`, `trial_start`, `trial_stop`, `trial_change_time`, `n_total_rois`), stores raw running and pupil arrays in intermediate trial dictionaries only to discard them after discretization, and includes plotting support that is unrelated to the final pickle output.

ii.
```python
stim_stop = stim['stop_time'][:]
stim_omitted = stim['omitted'][:]
valid_trial_ids = trial_ids[trial_mask]
trial_start = trials['start_time'][:]
trial_stop = trials['stop_time'][:]
trial_change_time = trials['change_time'][:]
n_total_rois = dff_data.shape[1]
...
output_trials.append({
    'image_identity': image_indices,
    'image_change': change_flags,
    'running_speed_raw': running_binned,
    'pupil_diameter_raw': pupil_binned,
    'trial_outcome': outcome,
    'n_bins': n_bins,
})
```

iii. The notes focus on speed and validation rather than minimizing intermediate work, so these discarded computations appear to be side effects of the two-pass design and the optional plotting/debug workflow rather than explicitly justified scientific choices.
