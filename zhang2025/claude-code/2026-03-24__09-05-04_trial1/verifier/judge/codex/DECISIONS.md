# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI code does not use the ONE API. It loads a release CSV (`bwm_release.csv`) to enumerate sessions, then opens files directly from the ONE cache on disk under `/app/data/one_cache/<lab>/Subjects/<subject>/<date>/001/alf`. Trials are read with `pandas.read_parquet`, and spikes, wheel, and camera data are read with `numpy.load`.

ii.
```python
DATA_DIR = Path('/app/data/one_cache')
BWM_CSV = Path('/app/code/code_zhang2025/data/bwm_release.csv')

def find_session_path(lab, subject, date, number=1):
    sess_path = DATA_DIR / lab / 'Subjects' / subject / date / f'{number:03d}' / 'alf'
```

```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
sessions = get_session_list(bwm_df)
trials = pd.read_parquet(trials_file)
spikes = {
    'times': np.load(rev_dir / 'spikes.times.npy').flatten(),
    'clusters': np.load(rev_dir / 'spikes.clusters.npy').flatten(),
}
```

iii. The justification in `CONVERSION_NOTES.md` is that this is “Direct file loading (no ONE API needed, works offline from cache)” and that the BWM CSV should be the ground-truth session list.

## 1-b. How are the data split into subjects?

i. Subject identity comes from the BWM release CSV row for each session. After processing, the code collects the sorted unique subject names from the successfully converted sessions and builds `subject_idx` from that list.

ii.
```python
subject = session_info['subject']
```

```python
all_subjects = sorted(set(r['subject'] for r in all_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
subject_idx.append(subject_to_idx[r['subject']])
```

iii. The notes say the session list should come from the BWM CSV, so subject IDs are taken from those session records rather than derived from paths in the final assembly step.

## 1-c. How are the data split into sessions?

i. Sessions are defined as unique `eid` rows in the BWM release CSV. The code groups the CSV by `eid`, keeps one row per `eid`, and processes those sessions one by one.

ii.
```python
def get_session_list(bwm_df):
    sessions = bwm_df.groupby('eid').first().reset_index()
    return sessions
```

```python
for idx in range(len(sessions)):
    sess = sessions.iloc[idx]
    result = process_session(sess, br, show_processing=args.show_processing)
```

iii. The notes say “Session list: Use BWM release CSV (459 sessions) as ground truth.” There is no explicit justification for ignoring the session `number` field when resolving the cache path.

## 1-d. How are the data split into trials?

i. Trials are taken from rows of `_ibl_trials.table.pqt`. First the code filters the table with `create_trial_mask`, then it applies a second `combined_mask` after wheel and whisker interpolation; only rows surviving both masks become final trials.

ii.
```python
trials = load_trials(alf_path)
mask = create_trial_mask(trials)
valid_trials = trials[mask].reset_index(drop=True)
```

```python
final_trials = valid_trials[combined_mask].reset_index(drop=True)
```

iii. The notes claim this “matches reference code exactly” and also emphasize the decoder requirement that each kept session must retain at least two trials.

## 1-e. How are trials filtered based on quality controls?

i. The code drops trials with NaNs in several task-event columns, drops reaction times outside 0.08 to 2.0 s, drops no-choice trials, optionally drops trials longer than 10 s from `goCue_times` to `feedback_times`, and then drops any trial whose interpolated wheel or whisker trace contains NaNs.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']

rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
mask &= (trials['choice'] != 0)
```

```python
if 'goCue_times' in trials.columns:
    trial_len = trials['feedback_times'] - trials['goCue_times']
    mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
```

```python
combined_mask &= ~np.any(np.isnan(wheel_speed_binned), axis=1)
combined_mask &= ~np.any(np.isnan(me_binned), axis=1)
```

iii. The notes justify this as the reference trial mask plus valid-data-period checks, and explicitly say “Trial mask matching reference code exactly.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are built from `spikes.times.npy` and `spikes.clusters.npy` from each probe. The code also loads cluster/channel metadata to count units and assign brain regions, but the trial neural matrices themselves come from spike times and cluster IDs.

ii.
```python
spikes = {
    'times': np.load(rev_dir / 'spikes.times.npy').flatten(),
    'clusters': np.load(rev_dir / 'spikes.clusters.npy').flatten(),
}
```

```python
clusters = {
    'channels': clusters_channels,
    'depths': clusters_depths,
    'metrics': metrics,
    'chan_brain_ids': chan_brain_ids,
}
```

iii. The notes map “spikes.times + spikes.clusters” directly to the target `neural` field.

## 2-b. How is the `neural` data processed?

i. The code merges probes, sorts all spikes by time, bins spikes into 20 ms bins inside a stimulus-aligned 2 s window, and stores the resulting per-trial matrices as raw spike counts cast to `uint8`. It does not divide by bin width to convert to firing rate.

ii.
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
sort_idx = np.argsort(all_times, kind='stable')
```

```python
bin_idx = np.minimum(((t_sel - t_start) / binsize).astype(np.int32), n_bins - 1)
np.add.at(binned[trial_idx].ravel(), flat_idx, 1)
```

```python
for t in range(len(final_spikes)):
    neural_trials.append(final_spikes[t].astype(np.uint8))
```

iii. The notes justify spike binning as “vectorized numpy operations” and later justify `uint8` storage as a memory optimization because spike counts fit within 8 bits. There is no explicit justification for leaving the data as counts rather than Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered by neural quality at all. The script loads `clusters.metrics.pqt`, but never uses the `label` column or any other QC metric to exclude units.

ii.
```python
metrics_file = rev_dir / 'clusters.metrics.pqt'
if metrics_file.exists():
    metrics = pd.read_parquet(metrics_file)
else:
    metrics = None
```

```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
n_clusters = len(clusters['channels'])
```

iii. The trajectory and notes repeatedly justify this as “qc=None” / “all clusters (not filtered by quality label),” based on the AI’s reading of the reference code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times`. For each kept trial the code uses `[stimOn - 0.5, stimOn + 1.5]` as the interval and bins spikes relative to that interval.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
align_times = valid_trials[ALIGN_TIME].values
interval_starts = align_times + TIME_WINDOW[0]
interval_ends = align_times + TIME_WINDOW[1]
```

iii. The notes say the task specification overrode any other alignment and therefore everything should be aligned to stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code uses 20 ms bins and 100 bins per trial. There is no later temporal rebinning.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The notes say the code follows the 20 ms binning used by the reference code.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is only indirectly derived from raw data: `stimOn_times` defines the alignment event, and the actual input values are a synthetic time axis relative to that event.

ii.
```python
align_times = valid_trials[ALIGN_TIME].values
```

```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. The notes describe this as “Manual construction” and say it should be the same for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code creates a fixed 100-point vector from `-0.48` to `1.5` seconds with `np.linspace`, then reuses that vector for every trial.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

```python
inp = np.array([
    time_since_stim,
    np.full(N_BINS, trial_in_block_final[t], dtype=np.float32),
], dtype=np.float32)
```

iii. There is no detailed justification beyond the notes saying the time input is manually constructed from the chosen window and bin size.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The intended alignment is to use the same trial window and number of bins as the neural data. In practice, the time axis is `t_start + 0.02` through `t_end`, while spikes are counted in bins beginning at `t_start`, so the labels are shifted relative to spike-bin centers.

ii.
```python
interval_starts = align_times + TIME_WINDOW[0]
interval_ends = align_times + TIME_WINDOW[1]
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. The notes assert that everything is on a common stimulus-aligned grid, but they do not discuss the half-bin/endpoint mismatch in the actual implementation.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft` in the trials table.

ii.
```python
def compute_trial_in_block(prob_left):
    ...
    if i > 0 and prob_left[i] != prob_left[i-1]:
        count = 1
```

```python
prob_left_all = trials['probabilityLeft'].values
trial_in_block_all = compute_trial_in_block(prob_left_all)
```

iii. The notes justify this by saying block boundaries are detected from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans all trials in order, resets the counter whenever `probabilityLeft` changes, counts trials starting from `1` rather than `0`, computes this before the trial masks are applied, then subsets and broadcasts the scalar value across all time bins of each kept trial.

ii.
```python
trial_in_block = np.zeros(len(prob_left), dtype=np.float32)
count = 1
for i in range(len(prob_left)):
    if i > 0 and prob_left[i] != prob_left[i-1]:
        count = 1
    trial_in_block[i] = count
    count += 1
```

```python
trial_in_block_valid = trial_in_block_all[mask.values]
trial_in_block_final = trial_in_block_valid[combined_mask]
np.full(N_BINS, trial_in_block_final[t], dtype=np.float32)
```

iii. The notes justify computing it from block changes and carrying it through the masks, but they do not explicitly justify the one-based indexing.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the `choice` column of the trials table.

ii.
```python
mask &= (trials['choice'] != 0)
choice = final_trials['choice'].values.copy()
```

iii. The notes explicitly map `trials.choice` to the target choice output.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code assumes the raw coding is `-1=left, 1=right`, converts it with `((choice + 1) // 2)`, and broadcasts the resulting binary value across all 100 time bins of the trial.

ii.
```python
# IBL: -1=left, 1=right -> convert to 0=left, 1=right
choice_encoded = ((choice + 1) // 2).astype(int)
```

```python
np.full(N_BINS, choice_encoded[t], dtype=int)
```

iii. The notes justify this with the explicit statement “left(-1)->0, right(1)->1.”

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column of the trials table.

ii.
```python
prob_left = final_trials['probabilityLeft'].values
```

iii. The notes say the task specification required categorizing `probabilityLeft` into three classes.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then broadcasts the category across all time bins of each trial.

ii.
```python
prior_encoded = np.zeros(len(prob_left), dtype=int)
prior_encoded[prob_left == 0.2] = 0
prior_encoded[prob_left == 0.5] = 1
prior_encoded[prob_left == 0.8] = 2
```

```python
np.full(N_BINS, prior_encoded[t], dtype=int)
```

iii. The notes say this follows the task specification exactly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from raw wheel position and timestamp files: `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
wheel_pos_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.position.npy')).flatten()
wheel_ts_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')).flatten()
wheel_times, wheel_speed = interpolate_wheel(wheel_ts_raw, wheel_pos_raw)
```

iii. The notes justify this as matching the ibllib wheel-processing path.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code linearly interpolates wheel position onto a 1 kHz grid, applies an 8th-order 20 Hz Butterworth low-pass filter, differentiates to velocity, takes the absolute value, and then linearly interpolates that speed trace into each trial’s 100-bin stimulus-aligned window.

ii.
```python
t = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
pos_interp = interp1d(timestamps, position, kind='linear')(t)
sos = signal.butter(N=order, Wn=corner_freq / fs * 2, btype='lowpass', output='sos')
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
return t, np.abs(vel).astype(np.float32)
```

```python
y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. The notes say this was chosen to “match ibllib” and specifically mention the 1 kHz interpolation and Butterworth filter.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. All wheel-speed samples from the surviving trials of a session are flattened together, percentile boundaries are computed over that flattened array, and values are digitized into three equal-frequency bins labeled low/medium/high.

ii.
```python
def discretize_to_bins(values, n_bins=N_DISCRETE_BINS):
    valid = values[~np.isnan(values)]
    quantiles = np.linspace(0, 100, n_bins + 1)
    boundaries = np.percentile(valid, quantiles)
    result = np.digitize(values, boundaries[1:-1])
```

```python
all_wheel_flat = final_wheel.flatten()
wheel_disc, wheel_boundaries = discretize_to_bins(all_wheel_flat, N_DISCRETE_BINS)
wheel_disc_2d = wheel_disc.reshape(final_wheel.shape).astype(int)
```

iii. The notes explicitly justify this as “equal-frequency (quantile) bins across all trials within a session.”

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is aligned to the same stimulus-centered trial windows as the neural data and resampled to 100 samples per trial, but those samples are evaluated on `np.linspace(t_start + binsize, t_end, n_bins)` rather than on the neural bin centers.

ii.
```python
wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. The notes justify the alignment by saying all decoder variables should be stimulus-onset aligned.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy`, falling back to the right camera if left is unavailable.

ii.
```python
def load_motion_energy(alf_path, side='left'):
    if side == 'left':
        me_file = find_latest_revision(alf_path, 'leftCamera.ROIMotionEnergy.npy')
        times_file = find_latest_revision(alf_path, '_ibl_leftCamera.times.npy')
    else:
        me_file = find_latest_revision(alf_path, 'rightCamera.ROIMotionEnergy.npy')
        times_file = find_latest_revision(alf_path, '_ibl_rightCamera.times.npy')
```

```python
me_times, me_values = load_motion_energy(alf_path, side='left')
if me_times is None:
    me_times, me_values = load_motion_energy(alf_path, side='right')
```

iii. The notes say motion energy should come from the left camera with right-camera fallback when needed.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is loaded as-is, trimmed to match the shorter of the time and value arrays when their lengths differ, cast to `float32`, and then linearly interpolated into each trial’s 100-bin window.

ii.
```python
me = np.load(me_file).flatten()
times = np.load(times_file).flatten()
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len].astype(np.float32)
```

```python
me_binned, me_good = interpolate_behavior_to_bins(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. The notes justify this as using the released whisker motion energy directly, with discretization happening only afterward.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As with wheel speed, the code flattens all whisker-motion-energy samples from the surviving trials of a session and applies three quantile bins.

ii.
```python
all_me_flat = final_me.flatten()
me_disc, me_boundaries = discretize_to_bins(all_me_flat, N_DISCRETE_BINS)
me_disc_2d = me_disc.reshape(final_me.shape).astype(int)
```

iii. The notes explicitly say whisker motion energy uses the same quantile-based low/medium/high discretization as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It uses the same kept trials, stimulus-aligned windows, and 100-sample interpolation grid as wheel speed. As implemented, the grid is `t_start + binsize` through `t_end`, not the neural bin centers.

ii.
```python
me_binned, me_good = interpolate_behavior_to_bins(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. The notes justify this the same way as wheel alignment: all outputs should be stimulus-onset aligned.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles revisioned files by picking the latest revision directory, trims motion-energy length mismatches to the shorter array, catches file-loading errors and skips the affected session, and also skips sessions with no probes, missing session paths, or fewer than two usable trials.

ii.
```python
def find_latest_revision(base_path, filename_pattern):
    ...
    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]
```

```python
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len].astype(np.float32)
```

```python
if alf_path is None:
    print(f"  Session {eid} ({subject}/{date}): path not found, skipping")
    return None
...
except Exception as e:
    print(f"  Session {eid}: error loading wheel: {e}")
```

iii. The notes justify this as sensible missing-data handling for offline cache processing and explicitly report that 67 sessions were skipped for missing paths/data or too few valid trials.

## 10-a. What are the most time-consuming steps of the code?

i. The code suggests that the expensive steps are per-session file I/O for trials/spikes/behavior, wheel interpolation and filtering at 1 kHz, and the per-trial loops used for spike binning and behavior interpolation.

ii.
```python
spk, clu = load_spike_sorting(alf_path, probe_name)
wheel_times, wheel_speed = interpolate_wheel(wheel_ts_raw, wheel_pos_raw)
```

```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx].ravel(), flat_idx, 1)
```

```python
for trial_idx in range(n_trials):
    ...
    y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. The notes only provide coarse runtime estimates and say the implementation uses direct file loading and “vectorized numpy operations”; they do not include a more detailed bottleneck analysis.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain: the per-trial loop inside `bin_spikes_vectorized`, the per-trial loop inside `interpolate_behavior_to_bins`, the loop computing `trial_in_block`, the loops that build per-trial input/output arrays, and the top-level session loop in `main`.

ii.
```python
for trial_idx in range(n_trials):
    ...
```

```python
for i in range(len(prob_left)):
    ...
```

```python
for t in range(len(final_trials)):
    input_trials.append(inp)
...
for idx in range(len(sessions)):
    ...
```

iii. The notes describe the implementation as vectorized, but the code still leaves these loops in place.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly rescans revision directories with `find_latest_revision`, separately interpolates wheel and whisker traces trial-by-trial with nearly identical logic, and rebuilds per-trial arrays in Python loops after already having session-level arrays.

ii.
```python
trials_file = find_latest_revision(alf_path, '_ibl_trials.table.pqt')
wheel_pos_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.position.npy')).flatten()
wheel_ts_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')).flatten()
```

```python
wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(...)
me_binned, me_good = interpolate_behavior_to_bins(...)
```

iii. The notes do not explicitly justify or acknowledge this repetition.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `clusters.depths` and `clusters.metrics` but never uses them downstream, computes `wheel_boundaries` and `me_boundaries` but never stores them, and imports multiprocessing helpers even though the final `main` processes sessions serially.

ii.
```python
clusters_depths = np.load(rev_dir / 'clusters.depths.npy').flatten()
...
metrics = pd.read_parquet(metrics_file)
```

```python
wheel_disc, wheel_boundaries = discretize_to_bins(all_wheel_flat, N_DISCRETE_BINS)
me_disc, me_boundaries = discretize_to_bins(all_me_flat, N_DISCRETE_BINS)
```

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
...
for idx in range(len(sessions)):
    ...
```

iii. There is no explicit justification for these discarded computations in the notes or trajectory.
