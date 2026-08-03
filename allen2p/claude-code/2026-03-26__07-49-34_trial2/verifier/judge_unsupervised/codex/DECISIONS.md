# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script hard-codes metadata and NWB directories, reads `ophys_experiment_table.csv`, glob-matches all local NWB files, intersects the two by `ophys_experiment_id`, filters to `passive == False`, and then processes each selected NWB twice: once to collect global running/pupil statistics and once for the full conversion.

ii. 
```python
NWB_DIR = 'data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/'
METADATA_DIR = 'data/visual-behavior-ophys-1.1.0/project_metadata/'

def get_experiment_metadata():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    ...
    our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
    active_exps = our_exps[our_exps['passive'] == False].copy()
    ...

for idx, (_, row) in enumerate(active_exps.iterrows()):
    stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
for idx, (_, row) in enumerate(active_exps.iterrows()):
    result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. In `CONVERSION_NOTES.md`, the agent says it chose "active sessions only" because passive sessions do not provide meaningful trial outcomes for the decoder, and that direct `h5py` NWB loading is "equivalent" to the AllenSDK loader.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values from the filtered experiment metadata. The subject list is sorted, and each kept session stores an integer index into that list.

ii. 
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
subject_idx = subjects_list.index(result['mouse_id'])
all_subject_idx.append(subject_idx)
...
'subjects': subjects_list,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. The notes describe mice as the subject unit and report subject counts from metadata, so the subject split follows the metadata field directly.

## 1-c. How are the data split into sessions?

i. Each kept `ophys_experiment_id` is treated as one output session. The script does not merge experiments that share the same `ophys_session_id`, so multi-plane sessions remain split across multiple output sessions.

ii. 
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
    ...
    all_sessions_neural.append(result['neural_trials'])
    all_sessions_input.append(result['input_trials'])
    all_sessions_output.append(session_output)
```

iii. In the notes the agent states that "Each NWB file = one imaging plane from one session" and proceeds with the NWB file as the session unit for the decoder.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table after masking. For each retained trial ID, the script finds all stimulus presentations whose `trials_id` equals that trial ID, and each such set becomes one decoder trial with `n_bins = len(trial_stim_indices)`.

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

iii. The notes say the task should be segmented by experiment-defined trials and that Go/Catch trials should be represented by their constituent image presentations.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by behavioral type rather than a rich QC pipeline: keep trials where `go` or `catch` is true, exclude `aborted` and `auto_rewarded`, skip trials with no associated stimulus presentations, and later drop sessions with fewer than two retained trials.

ii. 
```python
trial_go = trials['go'][:].astype(bool)
trial_catch = trials['catch'][:].astype(bool)
trial_aborted = trials['aborted'][:].astype(bool)
trial_auto = trials['auto_rewarded'][:].astype(bool)

trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
if len(trial_stim_indices) == 0:
    continue
...
if result is None or result['n_trials'] < 2:
    skipped += 1
```

iii. The agent repeatedly cites the task instruction "Include both the Go and Catch trials, but exclude the Aborted and Auto-rewarded trials" as the governing filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB dF/F traces and their timestamps, then filtered by the ROI validity mask from the cell specimen table.

ii. 
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. In `CONVERSION_NOTES.md` the agent explicitly says dF/F is pre-computed in NWB and chose it despite also noting that the paper's neural analyses used calcium events.

## 2-b. How is the `neural` data processed?

i. The script computes cumulative sums over valid dF/F traces and averages dF/F within each 750 ms stimulus-presentation bin to produce a `(n_neurons, n_bins)` matrix per trial.

ii. 
```python
bin_duration = TIME_BIN_MS / 1000.0
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')

dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])

neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The notes justify this as "one bin per stimulus presentation" and describe it as the natural task unit that is shared across 11 Hz and 31 Hz recordings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC is limited to keeping `valid_roi == True` ROIs and skipping experiments with zero surviving neurons.

ii. 
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
n_neurons = dff_valid.shape[1]

if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
```

iii. The notes cite the AllenSDK default `exclude_invalid_rois=True` behavior and describe `valid_roi` as the key curation rule taken from the SDK/whitepaper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are not aligned to a single per-trial event like trial start or change time. Instead, the agent aligns to each stimulus presentation onset, using stimulus timestamps to define 750 ms bins and ophys timestamps only to decide which dF/F frames fall in each bin.

ii. 
```python
all_stim_starts = stim_start
all_stim_ends = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
...
'temporal_alignment_event': 'Stimulus presentation onset (each 750ms image flash)',
```

iii. The trajectory shows the agent debated 30 Hz, 11 Hz, 100 ms, and 750 ms options, then chose flash onset because it thought it was the cleanest common alignment across rigs.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is fixed at 750 ms. Yes: all modalities are rebinned by averaging within 750 ms windows spanning one 250 ms image plus the following 500 ms gray period.

ii. 
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
...
bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
...
'time_bin_size': TIME_BIN_MS,
```

iii. The notes state this was chosen to create one common bin size across mesoscope and single-plane recordings and to align naturally to the flash cycle.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation table's `image_name` field, with the categorical vocabulary collected from a sample of active experiments.

ii. 
```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in stim['image_name'][:]])
...
names = f['intervals'][stim_key]['image_name'][:]
for n in names:
    name = n.decode() if isinstance(n, bytes) else str(n)
    if name != 'omitted':
        all_images.add(name)
```

iii. The notes map `image_name` directly to `output[0]` and say image names are collected globally for consistent categorical encoding.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The script decodes bytes to strings, forward-fills omitted flashes with the previous image (or the next non-omitted image if the omission is the first flash in the trial), and then converts names to integer category IDs using `IMAGE_NAMES_GLOBAL`.

ii. 
```python
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

iii. In the notes the agent calls this "Forward-fill for omitted" and argues it preserves a usable categorical identity for omissions instead of introducing a separate class.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is one integer per stimulus-presentation bin, using the same `trial_stim_indices` and `n_bins` as the neural matrix for that trial.

ii. 
```python
n_bins = len(trial_stim_indices)
...
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
...
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ...
], axis=0)
```

iii. The agent's stated rationale was that every output should live on the same 750 ms flash grid as the rebinned neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived directly from the stimulus presentation table's `is_change` field for the flashes belonging to a trial.

ii. 
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes explicitly map `is_change` to `output[1]` and describe it as "1 at change flash only."

## 4-b. What processing is involved in computing `output` *Image change*?

i. Processing is minimal: the selected `is_change` values are taken per trial, NaNs are replaced with zero, and the result is cast to integer.

ii. 
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The agent justified this as a direct binary readout already present in the stimulus table, so no extra derivation from image-name transitions was needed.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is not thresholded from a continuous variable. The code keeps it as a binary categorical output with labels `no_change` and `change`.

ii. 
```python
change_value_names = ['no_change', 'change']
...
ot['image_change'].astype(np.int64),
```

iii. The notes describe image change as already binary in the source data and therefore just encoded into two categories.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change shares the same one-value-per-flash, 750 ms binning as the neural data, with one flag for each column of the trial's neural matrix.

ii. 
```python
n_bins = len(trial_stim_indices)
...
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ot['image_change'].astype(np.int64),
    ...
], axis=0)
```

iii. The trajectory shows the agent wanted every decoded variable to sit on the same flash-level grid as the rebinned neural matrix.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and its timestamps.

ii. 
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The notes say this matches the AllenSDK `RunningSpeed.from_nwb()` source and uses the filtered running-speed stream already stored in NWB.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code precomputes stimulus-bin start and end indices on the running-speed timeline, then averages running speed within each 750 ms flash bin. It also runs a first pass over all experiments to collect all per-bin running values for later percentile binning.

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
```

iii. The notes justify this as putting running speed on the same common flash grid as neural activity and making global percentile discretization possible.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. After the first pass, the script computes five global percentile bins across all collected running-speed values and digitizes each per-trial binned time series into those five categories.

ii. 
```python
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
...
running_disc, _ = discretize_values(ot['running_speed_raw'],
                                    N_PERCENTILE_BINS,
                                    running_bin_edges)
```

iii. The notes explicitly say "Compute percentile bin edges across ALL valid time points in dataset (2-pass)."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by averaging onto the same 750 ms stimulus bins used for neural data, then placed in the third row of the per-trial output matrix.

ii. 
```python
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ot['image_change'].astype(np.int64),
    running_disc.astype(np.int64),
    ...
], axis=0)
```

iii. The agent's rationale was that all decoded variables should share the same common binning as the neural matrix.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking `pupil_tracking/area`, `pupil_tracking/timestamps`, and the `likely_blink` mask.

ii. 
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The notes map pupil area to diameter and reference the eye-tracking loader and blink QC described in the whitepaper/SDK.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN, nonpositive areas are also set to NaN, area is converted to diameter via `2*sqrt(area/pi)`, missing values are linearly interpolated, and per-bin means are computed across the same 750 ms flash bins.

ii. 
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
...
if n_valid > 0:
    pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. The notes justify interpolation as a practical way to handle blink-related NaNs before averaging and discretization.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The script computes five global percentile bins from all valid pupil values in pass 1. During pass 2 it digitizes each trial's binned pupil trace using those edges; if an entire trial is NaN it assigns the middle bin, and if only some bins are NaN it interpolates them first.

ii. 
```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
if len(valid_pupil) > 0:
    _, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
...
nan_mask = np.isnan(pupil_raw)
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The notes say pupil uses global percentile discretization, while the code adds an ad hoc fallback for fully missing trials.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is averaged into the same 750 ms flash bins as neural data and stored in the fourth row of the per-trial output matrix.

ii. 
```python
output_arr = np.stack([
    ...
    running_disc.astype(np.int64),
    pupil_disc.astype(np.int64),
    ...
], axis=0)
```

iii. The stated intent was a common per-flash time base across neural, stimulus, and behavioral outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trials-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. 
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. The notes map these four trial-outcome columns directly to the decoder's static per-trial label.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code uses an `if/elif` cascade to map each trial to one of four integer classes: hit, miss, false alarm, or correct rejection. Later, that trial-level label is repeated across all time bins in the trial when constructing `output_arr`.

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
np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
```

iii. The notes say trial outcome is static per trial, so repeating it across the trial's time axis keeps all outputs in a single `(n_output, n_timepoints)` array.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses pragmatic imputations rather than strict exclusion. It linearly interpolates NaNs in pupil traces, replaces full-NaN pupil trials with the middle category, forward-fills omitted image presentations, defaults unclassified trial outcomes to miss, skips trials with no linked stimulus presentations, and skips sessions with zero valid neurons or fewer than two retained trials.

ii. 
```python
def interpolate_nans(values):
    ...
    result[~valid] = np.interp(x[~valid], x[valid], values[valid])

if images[bi] == 'omitted':
    images[bi] = images[bi - 1]
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
...
else:
    outcome = 1  # Default to Miss
...
if len(trial_stim_indices) == 0:
    continue
if result is None or result['n_trials'] < 2:
    skipped += 1
```

iii. The notes frame these choices as keeping the dataset decoder-compatible and avoiding dropped trials/sessions unless data are unusable.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive parts are the two full passes over all experiments and the per-trial/per-bin averaging loops for neural, running, and pupil data inside `process_experiment`.

ii. 
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
for idx, (_, row) in enumerate(active_exps.iterrows()):
    result = process_experiment(nwb_path, row, collect_stats_only=False)
...
for trial_idx in np.where(trial_mask)[0]:
    ...
    for bi, si in enumerate(trial_stim_indices):
        ...
```

iii. The notes' runtime estimates explicitly identify pass 1 and pass 2 as the dominant wall-clock costs.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner loops over `trial_stim_indices` for neural averaging, running-speed averaging, pupil averaging, and omitted-image forward filling are still scalar Python loops and could be vectorized further.

ii. 
```python
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

iii. The notes mention cumulative sums and `searchsorted` as optimizations, which implies these remaining loops were left unvectorized despite being obvious hotspots.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats NWB loading and most preprocessing twice because `process_experiment` is run once for global percentile statistics and again for full conversion. It also recomputes searchsorted bin boundaries and cumulative sums on both passes.

ii. 
```python
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The notes explicitly describe a "two-pass approach" and justify it as necessary for global percentile-bin computation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several quantities are loaded or computed and then not used in the final dataset: `stim_stop`, `stim_omitted`, `trial_start`, `trial_stop`, `trial_change_time`, `valid_trial_ids`, `n_total_rois`, and the raw continuous running/pupil traces stored temporarily in `output_trials` only to be immediately discretized later. The `interp1d` import is also unused.

ii. 
```python
from scipy.interpolate import interp1d
...
stim_stop = stim['stop_time'][:]
stim_omitted = stim['omitted'][:]
...
n_total_rois = dff_data.shape[1]
...
trial_start = trials['start_time'][:]
trial_stop = trials['stop_time'][:]
trial_change_time = trials['change_time'][:]
...
valid_trial_ids = trial_ids[trial_mask]
...
'running_speed_raw': running_binned,
'pupil_diameter_raw': pupil_binned,
```

iii. The agent's notes focus on decoder outputs after discretization, so these extra loads/intermediate arrays are bookkeeping or implementation leftovers rather than needed downstream products.
