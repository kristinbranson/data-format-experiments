# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the Allen SDK cache or `VisualBehaviorOphysProjectCache`. It loaded a local metadata CSV, globbed local NWB files, intersected the two, kept only rows with `passive == False`, and then opened each NWB file directly with `h5py`.

ii.
```python
def get_experiment_metadata():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    ...
    our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
    active_exps = our_exps[our_exps['passive'] == False].copy()
    ...

with h5py.File(nwb_path, 'r') as f:
    experiment_id = exp_info['ophys_experiment_id']
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as working with the 284 NWB files available on disk rather than the full release, and said active sessions were required because passive sessions lack meaningful trial outcomes. The notes also claimed direct NWB reads were "equivalent" to SDK loading.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the filtered experiment metadata, converted to strings and sorted.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
subject_idx = subjects_list.index(result['mouse_id'])
all_subject_idx.append(subject_idx)
```

iii. The notes say the subset contains 38 mice and treat `mouse_id` as the subject identifier.

## 1-c. How are the data split into sessions?

i. The AI treated each NWB experiment file (`ophys_experiment_id`) as one session. It did not group experiments by `ophys_session_id`, so multi-plane sessions were not reconstructed.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. In trajectory step 31, the AI explicitly reasoned that each NWB file should be treated as a decoder session because each file contains its own neurons, even though it recognized that one biological session can include multiple experiments/planes.

## 1-d. How are the data split into trials?

i. Trials are defined by the NWB `intervals/trials` table, but the actual segmentation is done through stimulus presentations: for each valid trial id, the AI gathers all stimulus presentations with matching `trials_id`, and each trial becomes a sequence of 750 ms stimulus bins rather than an ophys-frame slice from `start_time` to `stop_time`.

ii.
```python
trials = f['intervals']['trials']
trial_ids = trials['id'][:]
...
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
    n_bins = len(trial_stim_indices)
```

iii. The notes and trajectory say the AI chose one 750 ms bin per image presentation because it viewed that as the "natural task unit" and a way to enforce a single time-bin size across mixed 11 Hz and 31 Hz sessions.

## 1-e. How are trials filtered based on quality controls?

i. The AI kept only `go` or `catch` trials, excluded `aborted` and `auto_rewarded` trials, skipped trials with no associated stimulus presentations, and later dropped experiments with fewer than two remaining trials.

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

iii. The notes say this follows the task instruction to include Go and Catch trials but exclude Aborted and Auto-rewarded trials, and to enforce at least two trials per session for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from NWB dF/F traces in `processing/ophys/dff/traces/data`, with timestamps from `processing/ophys/dff/traces/timestamps`.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
```

iii. The notes say dF/F is precomputed in the NWB files and that the AI chose dF/F instead of the paper's events because it viewed dF/F as the standard signal for decoding.

## 2-b. How is the `neural` data processed?

i. The AI filtered ROIs to `valid_roi == True`, then averaged each neuron's dF/F over each 750 ms stimulus presentation using cumulative sums. It did not keep native ophys frames or merge multiple planes within one session.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
...
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

iii. The notes say the AI wanted a common time bin across equipment types, and trajectory steps 39 and 41 show it chose 750 ms stimulus bins as a compromise between the 11 Hz and 31 Hz frame rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC is filtering to ROIs marked `valid_roi`. Experiments with zero valid neurons are skipped.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
n_neurons = dff_valid.shape[1]

if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
```

iii. The notes justify this by matching the SDK's default invalid-ROI exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus presentation onsets. For each stimulus presentation, the AI finds ophys frames in `[start_time, start_time + 0.75 s)` and averages them. Trial structure is therefore represented as a series of per-presentation bins, not as continuous ophys frames aligned to trial start.

ii.
```python
bin_duration = TIME_BIN_MS / 1000.0
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
```

iii. The notes say this was chosen because one bin per flash is consistent across equipment and "aligned to the natural task structure."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses a fixed 750 ms time bin. Yes: substantial temporal rebinning is applied by averaging neural and behavioral signals within each 750 ms stimulus interval.

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
...
'time_bin_size': TIME_BIN_MS,
```

iii. The notes say this was a deliberate choice to enforce one common bin size for mesoscope and single-plane recordings.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table: specifically `image_name` values for the presentations assigned to each trial via `trials_id`.

ii.
```python
stim = f['intervals'][stim_key]
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stim['image_name'][:]])
stim_trials_id = stim['trials_id'][:]
...
trial_stim_indices = np.where(stim_trials_id == tid)[0]
images = stim_image_name[trial_stim_indices].copy()
```

iii. The notes say image identity should come from the actual sequence of stimulus presentations, and that omitted flashes should be handled explicitly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Within each trial, the AI takes the sequence of stimulus `image_name` values, forward-fills any `omitted` entries using the previous non-omitted image when possible, then maps names to integer codes using a global image list collected from up to 10 sampled experiments.

ii.
```python
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
...
sample_exps = active_exps.head(min(10, len(active_exps)))
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as preserving identity during omitted flashes and assumed the same image set recurs across experiments, so sampling 10 experiments was enough to build the codebook.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is one categorical value per 750 ms stimulus bin, using the same `trial_stim_indices` bins as the neural averages.

ii.
```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
...
neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
...
images = stim_image_name[trial_stim_indices].copy()
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. The notes say image identity "naturally aligns" to the one-bin-per-flash representation.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentations table's `is_change` field for the presentations assigned to each trial.

ii.
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes say image change should be read at the stimulus-presentation level, where one presentation is the natural unit of a change event.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI directly converts `is_change` to integers, replacing NaNs with 0. There is no extra construction from `change_time`; the changed stimulus presentation alone carries the value 1.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes say a 750 ms stimulus bin should correspond to one image flash, so marking the changed presentation itself is sufficient.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary. The AI uses `0` for no-change presentations and `1` for change presentations.

ii.
```python
change_value_names = ['no_change', 'change']
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The justification is implicit in the notes: the AI treated `is_change` as the categorical variable directly, without any further thresholding.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned one-to-one with the 750 ms stimulus bins used for neural averaging.

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

iii. The notes say all time-varying outputs were intentionally aligned to the same stimulus-presentation bins as the neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from NWB running-speed samples and timestamps in `processing/running/speed`.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The notes say this uses the NWB's filtered running speed source.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI averages running speed within each 750 ms stimulus bin using cumulative sums, pools all binned values across the dataset in a first pass, computes global percentile edges, and then discretizes each trial's binned running signal in a second pass.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
run_cumsum = np.cumsum(running_speed)
run_cumsum = np.concatenate([[0], run_cumsum])
...
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
...
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
...
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. The notes justify 750 ms averaging as matching the task structure and say global percentile binning is needed to keep categories consistent across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five global percentile bins computed from all collected running values in pass 1, then assigned with `np.digitize`.

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

iii. `CONVERSION_NOTES.md` explicitly says discretization uses dataset-wide percentile bin edges computed in a first pass and applied in a second pass.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by averaging all running samples whose timestamps fall inside the same 750 ms stimulus bins used for neural averaging.

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

iii. The notes say behavioral variables should be averaged inside the same per-flash bins as neural activity.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking `area` values and timestamps in `acquisition/EyeTracking/pupil_tracking`, together with `likely_blink`.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The notes explicitly say "Use area -> compute diameter, interpolate NaNs" and cite the paper/whitepaper's blink handling as motivation.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI marks blink frames and nonpositive areas as NaN, converts area to diameter via `2*sqrt(area/pi)`, linearly interpolates missing values, averages pupil diameter within each 750 ms stimulus bin, computes global percentile edges in pass 1, and discretizes in pass 2. If a trial's binned pupil is all NaN, it assigns the middle bin.

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
...
pupil_binned = np.full(n_bins, np.nan, dtype=np.float32)
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The notes justify this as producing a literal pupil diameter signal while removing blink artifacts and keeping discretization global across the dataset.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five global percentile bins using `np.percentile` and `np.digitize`. Residual all-NaN trials are assigned the middle bin.

ii.
```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
if len(valid_pupil) > 0:
    _, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    ...
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The notes say discretization should use global percentile edges; the middle-bin fallback is an implementation decision to avoid invalid categorical outputs.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by averaging the eye-tracking samples that fall inside the same 750 ms stimulus bins used for neural activity.

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

iii. The notes say all time-varying outputs should share the same stimulus-aligned time bins.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the NWB trial booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. The notes treat these as the canonical trial-outcome fields for active behavior sessions.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps those four booleans to integers `0..3` with an if/elif chain, defaults unmatched trials to `miss`, and then repeats the resulting code across all time bins in the trial.

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
else:
    outcome = 1
...
np.full(n_bins, ot['trial_outcome'], dtype=np.int64)
```

iii. The notes say trial outcome is a static per-trial variable, so it should be repeated across time bins to fit the decoder format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI skips experiments with zero valid neurons or no stimulus table, skips trials with no associated stimulus presentations, removes blink and nonpositive pupil samples, linearly interpolates NaNs in pupil traces, fills all-NaN pupil trials with the middle category, forward-fills omitted images, and drops experiments with fewer than two remaining trials.

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
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
...
if images[bi] == 'omitted':
    images[bi] = images[bi - 1]
```

iii. The notes justify these choices as pragmatic cleanup to keep all outputs categorical and decoder-compatible even when eye tracking or stimulus logs are imperfect.

## 9-a. What are the most time-consuming steps of the code?

i. The code's main expensive work is reading every NWB file and processing it twice: once in pass 1 to collect running/pupil statistics and again in pass 2 for full conversion. Inside each pass, per-trial/per-bin averaging of neural, running, and pupil data also contributes.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    ...
    stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
for idx, (_, row) in enumerate(active_exps.iterrows()):
    ...
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The notes' runtime table shows pass 1 and pass 2 dominate wall time, and the script comments describe cumulative-sum averaging as a speed optimization inside those passes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious candidates are the repeated per-bin loops inside each trial for neural, running, and pupil averaging, plus the per-trial omitted-image forward-fill loop.

ii.
```python
for bi, si in enumerate(trial_stim_indices):
    ...
    neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
...
for bi, si in enumerate(trial_stim_indices):
    ...
    running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
...
for bi, si in enumerate(trial_stim_indices):
    ...
    pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
...
for bi in range(len(images)):
    if images[bi] == 'omitted':
        ...
```

iii. The notes say the AI already optimized these computations with cumulative sums; the remaining loops were left in place for implementation simplicity.

## 9-c. What processing does the code repeat multiple times?

i. The largest repeated processing is the full two-pass traversal of every experiment: first for global running/pupil statistics and again for full conversion. The script also separately samples experiments just to collect image names before the two main passes.

ii.
```python
IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
...
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The notes explicitly describe a "two-pass approach" so global discretization edges can be computed before the final conversion.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script carries some work that is not used in the final saved dataset: it loads several unused arrays (`stim_stop`, `stim_omitted`, `trial_start`, `trial_stop`, `trial_change_time`, `valid_trial_ids`, `n_total_rois`), stores intermediate raw running/pupil traces only to immediately discretize them later, and includes optional plotting logic unrelated to the final pickle.

ii.
```python
stim_stop = stim['stop_time'][:]
stim_omitted = stim['omitted'][:]
...
trial_start = trials['start_time'][:]
trial_stop = trials['stop_time'][:]
trial_change_time = trials['change_time'][:]
valid_trial_ids = trial_ids[trial_mask]
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
...
if args.show_processing and idx < 2:
    plot_processing(result, idx, nwb_path, row)
```

iii. The notes justify some of this as development-time validation and visualization, but these intermediate products are not preserved in `converted_data.pkl`.
