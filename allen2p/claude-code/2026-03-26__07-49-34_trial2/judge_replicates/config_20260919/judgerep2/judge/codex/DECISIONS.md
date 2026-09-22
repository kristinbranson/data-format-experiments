# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment metadata CSV, maps every available NWB filename to an experiment ID, retains only available non-passive experiments, and opens each file directly with `h5py`. It processes the 202 active files twice.

ii.
```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
active_exps = our_exps[our_exps['passive'] == False].copy()
with h5py.File(nwb_path, 'r') as f:
```

iii. The notes say the local 284 NWBs are the available subset and passive experiments lack meaningful trial outcomes; direct HDF5 access was considered equivalent to the SDK loader.

## 1-b. How are the data split into subjects?

i. Subjects are unique metadata `mouse_id` strings; each retained experiment is mapped to its mouse's index in a sorted global list.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
subject_idx = subjects_list.index(result['mouse_id'])
```

iii. The notes identify 38 mice in the active local subset and use `mouse_id` as the animal identifier.

## 1-c. How are the data split into sessions?

i. Each NWB/`ophys_experiment_id` (one imaging plane) is emitted as a separate session. Simultaneous planes sharing an `ophys_session_id` are not grouped.

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    result = process_experiment(nwb_map[eid], row, collect_stats_only=False)
    all_sessions_neural.append(result['neural_trials'])
```

iii. Although the notes recognize that each NWB is one imaging plane, they describe the 202 active experiments as sessions and choose experiment-level records.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. For each accepted trial ID, all linked stimulus presentations (`stim_trials_id == tid`) become that trial's time bins.

ii.
```python
trial_ids = trials['id'][:]
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
```

iii. The agent says this follows the experimental trial definitions while making each stimulus presentation a natural task unit.

## 1-e. How are trials filtered based on quality controls?

i. It includes go or catch trials and excludes aborted and auto-rewarded trials. Trials with no linked stimulus presentation are skipped, and experiments with fewer than two resulting trials are dropped.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
if len(trial_stim_indices) == 0:
    continue
if result is None or result['n_trials'] < 2:
    continue
```

iii. This is justified directly by the task's inclusion/exclusion rule and minimum-session-size requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB precomputed dF/F trace dataset and the image-segmentation `valid_roi` mask.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. The notes choose dF/F as a standard, already processed decoding signal despite noting that the paper used inferred calcium events.

## 2-b. How is the `neural` data processed?

i. Invalid ROIs are removed, then dF/F is averaged over each 750 ms interval beginning at a stimulus onset. Cumulative sums accelerate the means, and the result is transposed to neuron-by-bin form.

ii.
```python
dff_cumsum = np.vstack([np.zeros((1, n_neurons)), np.cumsum(dff_valid, axis=0)])
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / (e - s)
```

iii. The agent wanted one comparable bin per 250 ms image plus 500 ms gray interval across 11 Hz and 31 Hz acquisitions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `valid_roi=True` are retained; experiments with zero such neurons are skipped. No further cell/session QC is applied.

ii.
```python
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
if n_neurons == 0:
    return None
```

iii. The notes state this matches the AllenSDK default and its ROI curation rules.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Ophys timestamps are searched against every stimulus-presentation onset; each trial is therefore aligned to a sequence of stimulus onsets rather than retained from trial `start_time` through `stop_time` at native frames.

ii.
```python
ophys_bin_starts = np.searchsorted(ophys_ts, stim_start, side='left')
ophys_bin_ends = np.searchsorted(ophys_ts, stim_start + 0.75, side='left')
```

iii. The agent calls stimulus onset the temporal alignment event because it is the natural unit of this flashed-image task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is fixed at 750 ms, with temporal averaging of native ophys frames in every bin.

ii.
```python
TIME_BIN_MS = 750.0
'time_bin_size': TIME_BIN_MS
```

iii. The notes justify this as one complete image/gray presentation and a way to standardize different microscope frame rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `image_name` in the stimulus-presentation interval table associated with each trial.

ii.
```python
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in stim['image_name'][:]])
images = stim_image_name[trial_stim_indices].copy()
```

iii. The notes prefer presentation-level labels because identity can vary through a trial.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Omitted labels are forward-filled (or backfilled for a leading omission), then names are mapped through a sorted global vocabulary collected from at most ten experiments.

ii.
```python
if images[bi] == 'omitted':
    images[bi] = images[bi - 1] if bi > 0 else images[bj]
image_indices = np.array([img_to_idx.get(img, 0) for img in images])
```

iii. The agent says the preceding displayed identity remains relevant during omissions and sampling ten files is faster because sessions use the same image sets.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. One image code is assigned to each stimulus-presentation bin, exactly matching the corresponding 750 ms neural column.

ii.
```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
n_bins = len(trial_stim_indices)
image_indices = np.array([img_to_idx.get(img, 0) for img in images])
```

iii. Both labels and neural averages use the same ordered stimulus indices.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It comes directly from presentation-level `is_change` for the presentations belonging to a trial.

ii.
```python
stim_is_change = stim['is_change'][:]
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The notes identify the SDK/NWB flag as the canonical change-event indicator.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Missing flags are converted to zero and values are cast to integer; otherwise the stored flag is unchanged.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. No additional derivation was considered necessary.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: false/zero becomes category 0 and true/one category 1.

ii.
```python
change_value_names = ['no_change', 'change']
```

iii. The task explicitly requests a binary change variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Each `is_change` flag labels the same 750 ms stimulus bin used for its neural average, so a go change is one bin long.

ii.
```python
change_flags = stim_is_change[trial_stim_indices]
output_arr = np.stack([ot['image_identity'], ot['image_change'], ...], axis=0)
```

iii. The agent reports the expected roughly one change bin per go trial.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses the NWB running `speed/data` and its timestamps.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. The notes say this is the SDK's filtered running-speed source.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Samples within each 750 ms stimulus interval are averaged with cumulative sums, then the means are globally percentile-discretized in a two-pass conversion.

ii.
```python
running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
running_disc, _ = discretize_values(ot['running_speed_raw'], 5, running_bin_edges)
```

iii. Averaging was chosen to summarize locomotion per task bin; global quintiles balance decoder classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Global 0, 20, 40, 60, 80, and 100 percentile edges of all trial-bin means define five categories; outer edges are replaced by infinities.

ii.
```python
bin_edges = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
bin_edges[0], bin_edges[-1] = -np.inf, np.inf
binned = np.digitize(values, bin_edges[1:-1])
```

iii. The agent cites balanced, consistent classes across experiments.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running timestamps are independently searched using the same stimulus start and start+0.75 s boundaries used for neural data; each average labels one neural bin.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
```

iii. The streams are hardware synchronized, and shared absolute-time boundaries provide alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking pupil ellipse `area`, eye timestamps, and `likely_blink`, rather than the SDK `pupil_width` used by the reference.

ii.
```python
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. The notes interpret area as a source from which an equivalent circular diameter can be computed.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink and nonpositive areas become NaN; diameter is `2*sqrt(area/pi)`; missing samples are linearly interpolated; valid samples are averaged per 750 ms bin. Remaining within-trial NaNs are interpolated, or an all-NaN trial is assigned the middle category.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = interpolate_nans(2.0 * np.sqrt(pupil_area / np.pi))
pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. The agent aimed to suppress blink artifacts while retaining trials and producing an interpretable diameter.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five global percentile bins are computed from non-NaN per-bin diameter means. All-missing trials receive category 2; otherwise residual NaNs are interpolated before digitization.

ii.
```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
_, pupil_bin_edges = discretize_values(valid_pupil, 5)
pupil_disc = np.full(n_bins, 5 // 2, dtype=np.int64) if nan_mask.all() else ...
```

iii. Global quintiles balance categories; the middle bin is presented as a neutral missing-data fallback.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Eye timestamps are searched against the same 750 ms stimulus boundaries, and each pupil average labels the corresponding neural bin.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
```

iii. The notes rely on hardware synchronization and common absolute-time bin boundaries.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It derives outcome from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. These are documented as the canonical mutually exclusive outcomes for valid go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. A priority chain maps hit, miss, false alarm, and correct rejection to 0–3; unmatched trials default to miss. The code repeats the static value across all time bins.

ii.
```python
if trial_hit[trial_idx]: outcome = 0
elif trial_miss[trial_idx]: outcome = 1
elif trial_fa[trial_idx]: outcome = 2
elif trial_cr[trial_idx]: outcome = 3
else: outcome = 1
np.full(n_bins, ot['trial_outcome'], dtype=np.int64)
```

iii. Replication satisfies the decoder's common output matrix shape; the notes expect every retained trial to have one canonical outcome.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/blink pupil samples are interpolated; all-missing pupil trials use the middle bin; absent eye tracking remains all-missing until that fallback; empty running bins remain zero; missing change flags become zero; unknown images map to code 0; zero-neuron, zero-stimulus, and fewer-than-two-trial experiments are skipped. There is no per-file exception handler.

ii.
```python
result[~valid] = np.interp(x[~valid], x[valid], values[valid])
change_flags = np.nan_to_num(..., nan=0)
image_indices = np.array([img_to_idx.get(img, 0) for img in images])
if result is None or result['n_trials'] < 2: continue
```

iii. The agent describes interpolation as preserving data through blinks and neutral/default categories as decoder-safe fallbacks.

## 9-a. What are the most time-consuming steps of the code?

i. Reading full NWBs and processing every experiment twice dominate runtime; neural cumulative sums and trial/bin assembly also scan large arrays. The notes estimate about 24 minutes from the sample run.

ii.
```python
stats = process_experiment(nwb_path, row, collect_stats_only=True)
result = process_experiment(nwb_path, row, collect_stats_only=False)
```

iii. The agent reports 5.7 s for pass 1 and 8.7 s for pass 2 on two experiments and accepts the two-pass cost for global percentile edges.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial stimulus selection, per-bin neural/running/pupil averaging, omission filling, image encoding, and repeated list-index lookups remain Python loops. Start/end arrays plus cumulative sums could vectorize most bin means.

ii.
```python
for trial_idx in np.where(trial_mask)[0]:
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    for bi, si in enumerate(trial_stim_indices):
```

iii. The notes call the cumulative-sum implementation “vectorized averaging,” but only the summation is vectorized; loops still dispatch each bin.

## 9-c. What processing does the code repeat multiple times?

i. `process_experiment` rereads and reprocesses every active NWB in both passes, including dF/F loading/filtering, stimulus/trial parsing, pupil conversion, cumulative sums, bin averaging, image handling, and outcome derivation.

ii.
```python
for ... in active_exps.iterrows():
    process_experiment(..., collect_stats_only=True)
for ... in active_exps.iterrows():
    process_experiment(..., collect_stats_only=False)
```

iii. The two passes are justified as necessary to learn global running and pupil quantiles before final categorical output.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In statistics-only pass 1 it still loads and filters all neural traces, builds their full cumulative sum, computes neural bin averages, image labels/change flags, outcomes, and input/output containers that are then discarded. Optional plotting also rereads NWBs only to pass unused path/file arguments.

ii.
```python
neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
image_indices = np.array([img_to_idx.get(img, 0) for img in images])
if collect_stats_only:
    continue
```

iii. The notes emphasize faster cumulative sums and the two-pass design but do not acknowledge that most first-pass neural/task work is unnecessary for collecting behavioral quantiles.
